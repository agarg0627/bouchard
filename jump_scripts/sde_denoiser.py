# sde_denoiser.py
# Jump reverse sampler that directly matches the paper's Algorithm 1 (Gaussian step + jump thinning).

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional


# ---------------------- SDE (paper jump section) ----------------------

@dataclass
class JumpDiffusionSDE:
    beta: float = 1.0

    def g(self) -> float:
        return math.sqrt(self.beta)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta * torch.clamp(t, 0.0, 1.0) + 1e-12)


# ---------------------- Network definition (must match training) ----------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        time = self.time_proj(t_emb).unsqueeze(-1)
        h = self.act(h + time)
        h = self.conv2(h)
        return self.act(h + x)


class JumpScoreNet1D(nn.Module):
    def __init__(self, channels: int = 64, num_blocks: int = 4, time_dim: int = 128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.input_conv = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock1D(channels, time_dim) for _ in range(num_blocks)])
        self.output_conv = nn.Conv1d(channels, 2, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_emb(t)
        h = self.input_conv(x_t)
        for b in self.blocks:
            h = b(h, t_emb)
        out = self.output_conv(h)
        eps_pred = out[:, 0:1, :]
        sJ_pred  = out[:, 1:2, :]
        return eps_pred, sJ_pred


# ---------------------- Model loader cache ----------------------

_model_cache = None


def load_model(ckpt_path="checkpoints/ecg_jump_sde_model.pt", device: Optional[str] = None):
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    beta = float(ckpt.get("beta", 1.0))
    mean = float(ckpt["mean"])
    std = float(ckpt["std"])

    arch = ckpt.get("arch", {})
    channels = int(arch.get("channels", 64))
    num_blocks = int(arch.get("num_blocks", 4))
    time_dim = int(arch.get("time_dim", 128))

    model = JumpScoreNet1D(channels=channels, num_blocks=num_blocks, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Use training λ and jump params for the reverse sampler unless overridden
    lambda_rate = float(ckpt.get("lambda_rate", 2.0))
    jump_scale = float(ckpt.get("jump_scale", 2.0))
    impulse_len = int(ckpt.get("impulse_len", 1))

    sde = JumpDiffusionSDE(beta=beta)
    _model_cache = (model, sde, mean, std, device, lambda_rate, jump_scale, impulse_len)
    return _model_cache


# ---------------------- Sampling utilities ----------------------

def _laplace_marks(rng: torch.Generator, n: int, scale: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    e1 = torch.empty((n,), device=device).exponential_(1.0, generator=rng)
    e2 = torch.empty((n,), device=device).exponential_(1.0, generator=rng)
    return (scale * (e1 - e2)).to(dtype)


def denoise_sde_single_channel(
    noisy_signal: np.ndarray,
    snr_db: float,
    ckpt_path="checkpoints/ecg_jump_sde_model.pt",
    num_steps: int = 80,
    impulse_len_override: Optional[int] = None,
    lambda_rate_override: Optional[float] = None,
    mark_scale_override: Optional[float] = None,
    cap_logit: float = 2.0,   # paper mentions capping to avoid exploding jumps
    seed: int = 0,
) -> np.ndarray:
    """
    Implements the paper's Algorithm 1:

    (a) Gaussian predictor:
        X^G = X - g^2 sθ(X,t) Δt + g sqrt(Δt) ξ
    (b) Jump proposals:
        m ~ Pois(λ Δt), marks Z ~ μ
    (c) Accept–reject:
        accept each mark with prob min(1, exp( sθ(X^G, t_next)^T Z ))
        and set X_next = X^G + sum accepted Z.

    Total score used for jump tilt:
        sθ = s^G + s^J
    with s^G computed from ε-head: s^G = -ε/σ(t).
    """
    model, sde, mean, std, device, lambda_rate, jump_scale, impulse_len = load_model(ckpt_path=ckpt_path)
    device_t = torch.device(device)

    if impulse_len_override is not None:
        impulse_len = int(impulse_len_override)
    if lambda_rate_override is not None:
        lambda_rate = float(lambda_rate_override)
    if mark_scale_override is not None:
        jump_scale = float(mark_scale_override)

    # Normalize to training scale
    x = (noisy_signal.astype(np.float32) - mean) / (std + 1e-8)
    x = torch.from_numpy(x)[None, None, :].to(device_t)  # (1,1,L)
    B, _, L = x.shape
    max_start = max(L - impulse_len, 1)

    # Compute start time from background SNR:
    # noise_var = 1/snr_linear in normalized units, and sigma(t)^2 = beta * t
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / max(snr_linear, 1e-8)
    t_start = float(min(1.0, max(1e-6, noise_var / max(sde.beta, 1e-8))))

    # Time grid: t_start -> 0
    t_grid = torch.linspace(t_start, 0.0, num_steps + 1, device=device_t)

    rng = torch.Generator(device=device_t)
    rng.manual_seed(seed)

    with torch.no_grad():
        for k in range(num_steps):
            t_cur = float(t_grid[k].item())
            t_next = float(t_grid[k + 1].item())
            dt = t_cur - t_next  # positive

            t_cur_t = torch.tensor([t_cur], device=device_t)
            t_next_t = torch.tensor([t_next], device=device_t)

            # ---------- (a) Gaussian predictor step ----------
            sig = sde.sigma(t_cur_t)  # (1,)
            sig_e = sig.view(1,1,1)

            eps_pred, sJ_pred = model(x, t_cur_t)

            # s^G = -eps/sigma
            sG = -eps_pred / (sig_e + 1e-12)
            s_total = sG + sJ_pred  # sθ

            g = sde.g()
            # X^G = X - g^2 sθ dt + g sqrt(dt) ξ
            xi = torch.randn(x.shape, device=x.device, dtype=x.dtype)
            xG = x - (g * g) * s_total * dt + g * math.sqrt(dt) * xi

            # ---------- (b) Jump proposals ----------
            # m ~ Pois(λ dt)
            m = int(torch.poisson(torch.tensor([lambda_rate * dt], device=device_t)).item())

            if m > 0:
                # marks Z ~ μ (Laplace), scale relative to RMS(x0) ~ 1 in normalized space
                # Use scale = jump_scale (since x is normalized); this mirrors training.
                mags = _laplace_marks(rng, m, scale=jump_scale, device=device_t, dtype=x.dtype)
                locs = torch.randint(0, max_start, (m,), device=device_t, generator=rng)

                # Compute score at (xG, t_next) for accept-reject tilt (paper Algorithm 1 uses tk)
                eps_pred2, sJ_pred2 = model(xG, t_next_t)
                sig2 = sde.sigma(t_next_t).view(1,1,1)
                sG2 = -eps_pred2 / (sig2 + 1e-12)
                s_total2 = sG2 + sJ_pred2

                # ---------- (c) Accept–reject ----------
                # accept prob α = min(1, exp( <sθ, Z> )), cap exponent to cap_logit
                accepted_jump = torch.zeros_like(xG)

                for j in range(m):
                    loc = int(locs[j].item())
                    mag = mags[j]

                    # inner product <sθ, Z> for 1D delta/burst is mag * sum sθ over the window
                    s_sum = s_total2[0,0, loc:loc+impulse_len].sum()
                    logit = torch.clamp(mag * s_sum, max=cap_logit)  # cap to avoid exploding jumps

                    a = torch.exp(logit)  # could exceed 1; acceptance uses min(1, a)
                    u = torch.rand((), device=device_t, generator=rng)
                    if u < torch.clamp(a, max=1.0):
                        accepted_jump[0,0, loc:loc+impulse_len] += mag

                x = xG + accepted_jump
            else:
                x = xG

    x_hat = x[0,0].cpu().numpy()
    x_hat = x_hat * std + mean
    return x_hat
