# sde_train_multiplicative.py

import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ---------------------- Log-domain VP SDE (same as additive) ----------------------

@dataclass
class LogDomainVPSDE:
    """
    Same functional form as your AdditiveVPSDE, but conceptually
    applied to log-magnitude signals.

    x_t = alpha(t) * x_0 + sigma(t) * eps,  eps ~ N(0, I)
    """
    beta: float = 10.0  # diffusion strength

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.beta * t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(-self.beta * t) + 1e-12)

    def marginal_forward(self, x0: torch.Tensor, t: torch.Tensor):
        """
        x0: (B, 1, L)
        t:  (B,)
        """
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        eps = torch.randn_like(x0)
        x_t = alpha_t * x0 + sigma_t * eps
        return x_t, eps, alpha_t, sigma_t


# ---------------------- Time embedding ----------------------

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
        # t: (B,)
        t = t.unsqueeze(-1)  # (B, 1)
        args = t * self.frequencies  # (B, half_dim)
        emb_sin = torch.sin(args)
        emb_cos = torch.cos(args)
        return torch.cat([emb_sin, emb_cos], dim=-1)  # (B, dim)


# ---------------------- Score network ----------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), t_emb: (B, time_dim)
        h = self.conv1(x)
        time = self.time_proj(t_emb).unsqueeze(-1)  # (B, C, 1)
        h = h + time
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h + x)


class ScoreNet1D(nn.Module):
    """
    Predicts epsilon given (x_t, t).
    Input:  x_t: (B, 1, L), t: (B,)
    Output: eps_pred: (B, 1, L)
    """
    def __init__(self, channels: int = 64, num_blocks: int = 4, time_dim: int = 128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.input_conv = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1D(channels, time_dim) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)  # (B, time_dim)
        h = self.input_conv(x_t)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_conv(h)


# ---------------------- Multiplicative NMR dataset (log-domain) ----------------------

class NMRMultiplicativeDataset(Dataset):
    """
    Wraps clean channels from all NPZ files in ../synthetic_data/NMR_multiplicative.

    For multiplicative noise we operate in log-magnitude space:
        y = log(|x| + eps)

    Each sample is a 1D log-domain signal of shape (1, L) (float32),
    normalized with global mean/std.
    """

    def __init__(self, data_dir: str, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

        data_path = Path(data_dir)
        npz_files = sorted(list(data_path.glob("*.npz")))
        if not npz_files:
            raise RuntimeError(f"No .npz files found in {data_dir}")

        signals: List[np.ndarray] = []

        for filepath in npz_files:
            data = np.load(filepath)
            clean = data["clean"]  # (n_channels, n_samples) or (n_samples,)
            if clean.ndim == 1:
                clean = clean[None, :]

            # convert each channel to log-magnitude
            for ch in range(clean.shape[0]):
                sig = clean[ch].astype(np.float32)  # (L,)
                mag = np.abs(sig)
                log_mag = np.log(mag + self.eps).astype(np.float32)
                signals.append(log_mag)

        # Stack into (N, 1, L)
        self.signals = np.stack(signals, axis=0)  # (N, L)
        self.signals = self.signals[:, None, :]   # (N, 1, L)

        # Compute normalization stats in log-domain
        self.mean = float(self.signals.mean())
        self.std = float(self.signals.std() + 1e-8)

        # Normalize
        self.signals = (self.signals - self.mean) / self.std

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.signals[idx])  # (1, L)


# ---------------------- Training loop ----------------------

def train_multiplicative_sde(
    model: nn.Module,
    sde: LogDomainVPSDE,
    dataloader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x0 in dataloader:
            x0 = x0.to(device)  # (B, 1, L)
            B = x0.shape[0]

            # Sample times uniformly in [0,1]
            t = torch.rand(B, device=device)
            x_t, eps, alpha_t, sigma_t = sde.marginal_forward(x0, t)

            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[Epoch {epoch+1}/{num_epochs}] loss = {avg_loss:.6f}")

    return model


# ---------------------- Main entry ----------------------

def main():
    # Adjust paths if needed
    data_dir = "../synthetic_data/NMR_multiplicative"
    dataset = NMRMultiplicativeDataset(data_dir=data_dir, eps=1e-6)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sde = LogDomainVPSDE(beta=10.0)
    model = ScoreNet1D(channels=64, num_blocks=4, time_dim=128)

    model = train_multiplicative_sde(
        model=model,
        sde=sde,
        dataloader=dataloader,
        num_epochs=20,  # tweak as needed
        lr=1e-3,
        device=device,
    )

    # Save checkpoint
    ckpt = {
        "state_dict": model.state_dict(),
        "beta": sde.beta,
        "mean": dataset.mean,
        "std": dataset.std,
        "signal_length": dataset.signals.shape[-1],
        "log_eps": math.log(dataset.eps),
    }
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/nmr_mult_sde_model.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved multiplicative SDE model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
