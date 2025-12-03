# sde_denoiser_multiplicative.py

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# --- SDE definition (log-domain VP) ---

@dataclass
class LogDomainVPSDE:
    beta: float = 10.0

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.beta * t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(-self.beta * t) + 1e-12)


# --- Time embedding + network (same as training) ---

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half_dim
            )
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)  # (B, 1)
        args = t * self.frequencies  # (B, half_dim)
        emb_sin = torch.sin(args)
        emb_cos = torch.cos(args)
        return torch.cat([emb_sin, emb_cos], dim=-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        time = self.time_proj(t_emb).unsqueeze(-1)
        h = h + time
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h + x)


class ScoreNet1D(nn.Module):
    def __init__(self, channels: int = 64, num_blocks: int = 4, time_dim: int = 128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.input_conv = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1D(channels, time_dim) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_conv(h)


# --- Global cache so we only load once ---

_model_cache_mult = None


def load_sde_model_multiplicative(
    ckpt_path="checkpoints/nmr_mult_sde_model.pt",
    device=None,
):
    global _model_cache_mult
    if _model_cache_mult is not None:
        return _model_cache_mult

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    beta = ckpt.get("beta", 10.0)
    mean = float(ckpt["mean"])
    std = float(ckpt["std"])
    log_eps = float(ckpt.get("log_eps", math.log(1e-6)))

    model = ScoreNet1D(channels=64, num_blocks=4, time_dim=128).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sde = LogDomainVPSDE(beta=beta)
    _model_cache_mult = (model, sde, mean, std, log_eps, device)
    return _model_cache_mult


# --- Single-channel multiplicative SDE denoiser ---

def denoise_sde_multiplicative_single_channel(
    noisy_signal: np.ndarray,
    snr_db: float,
    ckpt_path: str = "checkpoints/nmr_mult_sde_model.pt",
) -> np.ndarray:
    """
    noisy_signal: 1D numpy array (L,) in linear domain
    snr_db: SNR from the npz file (linear-domain definition)
    returns: denoised 1D numpy array (L,) in linear domain
    """
    model, sde, mean, std, log_eps, device = load_sde_model_multiplicative(
        ckpt_path=ckpt_path
    )
    eps = math.exp(log_eps)

    # Separate sign and log-magnitude
    sig = noisy_signal.astype(np.float32)
    sign = np.sign(sig).astype(np.float32)
    sign[sign == 0] = 1.0  # avoid zero sign
    mag = np.abs(sig)
    log_mag = np.log(mag + eps).astype(np.float32)

    # Normalize to training scale (log-domain)
    x = (log_mag - mean) / (std + 1e-8)
    x = torch.from_numpy(x)[None, None, :].to(device)  # (1,1,L)

    # Estimate noise variance in normalized log-domain from SNR
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / max(snr_linear, 1e-6)
    noise_var = min(noise_var, 0.99)  # cap for stability

    # Solve sigma_t^2 = noise_var = 1 - exp(-beta t) => t_start
    t_start = -1.0 / sde.beta * math.log(1.0 - noise_var)

    num_steps = 50
    t_vals = torch.linspace(t_start, 0.0, num_steps + 1, device=device)

    with torch.no_grad():
        for i in range(num_steps):
            t = t_vals[i]
            t_next = t_vals[i + 1]
            dt = t_next - t
            t_batch = t.expand(x.shape[0])

            sigma_t = sde.sigma(t_batch)
            while sigma_t.dim() < x.dim():
                sigma_t = sigma_t.unsqueeze(-1)

            eps_pred = model(x, t_batch)
            # score = ∇_x log p_t(x) = -eps / sigma
            s_theta = -eps_pred / (sigma_t + 1e-12)

            # Probability-flow ODE: dX/dt = -0.5 β X - β score
            drift = -0.5 * sde.beta * x - sde.beta * s_theta
            x = x + drift * dt

    # Back to numpy: normalized log-magnitude
    x_denoised = x[0, 0].cpu().numpy()
    log_mag_denoised = x_denoised * std + mean

    # Invert log transform
    mag_denoised = np.exp(log_mag_denoised) - eps
    mag_denoised = np.maximum(mag_denoised, 0.0)

    # Reapply sign
    x_final = sign * mag_denoised
    return x_final.astype(np.float32)
