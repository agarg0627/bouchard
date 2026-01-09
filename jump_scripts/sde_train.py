# sde_train.py
# ECG jump-noise SDE training that matches the paper's JUMP section (Eq. 44, 49, 50).

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ---------------------- Jump-diffusion SDE (paper Eq. 44) ----------------------

@dataclass
class JumpDiffusionSDE:
    """
    Forward (paper Eq. 44):
        dX_t = g dW_t + ∫ z \tilde N(dt,dz)
    with finite activity ν(dz)=λ μ(dz).
    We use constant g for simplicity, consistent with the paper's jump section.
    """
    beta: float = 1.0  # with constant g = sqrt(beta), sigma(t)^2 = beta * t

    def g(self) -> float:
        return math.sqrt(self.beta)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        # sigma(t)^2 = ∫_0^t g^2 ds = beta * t
        return torch.sqrt(self.beta * torch.clamp(t, 0.0, 1.0) + 1e-12)

    def marginal_forward_gaussian(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x0: (B,1,L), t: (B,)
        Returns x_g = x0 + sigma(t)*eps, eps, sigma(t)
        """
        sig = self.sigma(t)  # (B,)
        while sig.dim() < x0.dim():
            sig = sig.unsqueeze(-1)
        eps = torch.randn_like(x0)
        xg = x0 + sig * eps
        return xg, eps, sig


# ---------------------- Time embedding ----------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)  # (B,1)
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------- Network: epsilon head + jump-score head ----------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        time = self.time_proj(t_emb).unsqueeze(-1)  # (B,C,1)
        h = self.act(h + time)
        h = self.conv2(h)
        return self.act(h + x)


class JumpScoreNet1D(nn.Module):
    """
    Outputs:
      eps_pred: (B,1,L)   (for Gaussian score s^G = -eps/sigma)
      sJ_pred : (B,1,L)   (jump score field used in mark inner products)
    """
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


# ---------------------- Dataset (clean ECG only) ----------------------

class ECGCleanDataset(Dataset):
    """
    Loads clean signals from ../synthetic_data/ECG_jump/*.npz
    Each channel is a training sample.
    """
    def __init__(self, data_dir: str):
        super().__init__()
        files = sorted(list(Path(data_dir).glob("*.npz")))
        if not files:
            raise RuntimeError(f"No .npz files found in {data_dir}")

        sigs: List[np.ndarray] = []
        for fp in files:
            d = np.load(fp)
            clean = d["clean"]
            if clean.ndim == 1:
                clean = clean[None, :]
            for ch in range(clean.shape[0]):
                sigs.append(clean[ch].astype(np.float32))

        self.signals = np.stack(sigs, axis=0)[:, None, :]  # (N,1,L)
        self.mean = float(self.signals.mean())
        self.std = float(self.signals.std() + 1e-8)
        self.signals = (self.signals - self.mean) / self.std

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.signals[idx])  # (1,L)


# ---------------------- Jump sampling utilities (finite activity) ----------------------

def _laplace_marks(rng: torch.Generator, shape: Tuple[int, ...], scale: torch.Tensor) -> torch.Tensor:
    """
    Sample Laplace(0, scale) via Exp difference: scale*(E1 - E2).
    scale can be scalar tensor.
    """
    e1 = torch.empty(shape, device=scale.device).exponential_(1.0, generator=rng)
    e2 = torch.empty(shape, device=scale.device).exponential_(1.0, generator=rng)
    return scale * (e1 - e2)


def sample_forward_jumps(
    x0: torch.Tensor,
    t: torch.Tensor,
    lambda_rate: float,
    jump_scale: float,
    impulse_len: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, List[List[Tuple[int, float]]], float]:
    """
    Forward finite-activity compound Poisson:
      Nt ~ Pois(lambda_rate * t)   (per-signal expected jumps at time 1 is lambda_rate)

    Each jump is a delta/burst located uniformly on the grid, mark magnitude from Laplace,
    scaled relative to RMS(x0).

    Returns:
      jump_signal: (B,1,L)
      events: per batch element list of (loc, mag)
      b_mean: mean Laplace scale used (for negative proposal scaling)
    """
    device = x0.device
    B, _, L = x0.shape
    t = torch.clamp(t, 0.0, 1.0)

    rms = torch.sqrt(torch.mean(x0**2, dim=(1,2)) + 1e-12)  # (B,)
    b = jump_scale * rms  # (B,)
    b_mean = float(b.mean().item())

    lam_t = lambda_rate * t  # (B,)
    m = torch.poisson(lam_t)  # (B,)
    m_int = m.to(torch.int64)

    jump = torch.zeros((B,1,L), device=device, dtype=x0.dtype)
    events: List[List[Tuple[int, float]]] = [[] for _ in range(B)]
    max_start = max(L - impulse_len, 1)

    for bi in range(B):
        mi = int(m_int[bi].item())
        if mi <= 0:
            continue
        locs = torch.randint(0, max_start, (mi,), device=device, generator=rng)
        mags = _laplace_marks(rng, (mi,), b[bi].to(x0.dtype))

        for j in range(mi):
            loc = int(locs[j].item())
            mag = float(mags[j].item())
            jump[bi,0, loc:loc+impulse_len] += mags[j]
            events[bi].append((loc, mag))

    return jump, events, b_mean


def contrastive_jump_loss(
    sJ_pred: torch.Tensor,
    events: List[List[Tuple[int, float]]],
    impulse_len: int,
    rng: torch.Generator,
    neg_per_pos: int = 8,
    neg_scale: float = 3.0,
) -> torch.Tensor:
    """
    Paper Eq. (50):
      -log σ( sJ(x,t)^T Z_pos )  - log(1-σ( sJ(x,t)^T Z_neg ))

    For 1-D impulses, Z is sparse at a location:
      sJ^T Z = mag * sum_{window} sJ[loc:loc+impulse_len]
    """
    device = sJ_pred.device
    B, _, L = sJ_pred.shape
    max_start = max(L - impulse_len, 1)
    losses = []

    for bi in range(B):
        pos = events[bi]
        n_pos = len(pos)
        n_neg = max(neg_per_pos * max(n_pos, 1), neg_per_pos)

        # Negatives: random locations, Laplace marks from broad proposal μ0
        neg_locs = torch.randint(0, max_start, (n_neg,), device=device, generator=rng)
        neg_mags = _laplace_marks(rng, (n_neg,), torch.tensor(neg_scale, device=device, dtype=sJ_pred.dtype))

        neg_logits = []
        for j in range(n_neg):
            loc = int(neg_locs[j].item())
            s_sum = sJ_pred[bi,0, loc:loc+impulse_len].sum()
            neg_logits.append(neg_mags[j] * s_sum)
        neg_logits = torch.stack(neg_logits)
        neg_loss = F.softplus(neg_logits).mean()  # -log(1-sigmoid) = softplus

        if n_pos > 0:
            pos_logits = []
            for (loc, mag) in pos:
                loc = int(loc)
                mag_t = torch.tensor(mag, device=device, dtype=sJ_pred.dtype)
                s_sum = sJ_pred[bi,0, loc:loc+impulse_len].sum()
                pos_logits.append(mag_t * s_sum)
            pos_logits = torch.stack(pos_logits)
            pos_loss = F.softplus(-pos_logits).mean()  # -log(sigmoid) = softplus(-)
            losses.append(pos_loss + neg_loss)
        else:
            losses.append(neg_loss)

    return torch.stack(losses).mean()


# ---------------------- Training loop (paper Eq. 49 + 50) ----------------------

def train_jump_model(
    model: nn.Module,
    sde: JumpDiffusionSDE,
    dataloader: DataLoader,
    device: str,
    num_epochs: int,
    lr: float,
    lambda_rate: float,
    jump_scale: float,
    impulse_len: int,
    gamma: float,
    neg_per_pos: int,
    grad_clip: float,
    seed: int,
):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for epoch in range(num_epochs):
        model.train()
        tot, tot_eps, tot_con, nb = 0.0, 0.0, 0.0, 0

        for x0 in dataloader:
            x0 = x0.to(device)  # (B,1,L)
            B = x0.shape[0]

            t = torch.rand(B, device=device, generator=rng)  # U(0,1)

            # Gaussian forward (paper uses drift f=0): xg = x0 + sigma(t)*eps
            xg, eps, sig = sde.marginal_forward_gaussian(x0, t)

            # Jump forward: compound Poisson with ν=λ μ
            jump, events, b_mean = sample_forward_jumps(
                x0=x0, t=t,
                lambda_rate=lambda_rate,
                jump_scale=jump_scale,
                impulse_len=impulse_len,
                rng=rng,
            )

            x_t = xg + jump

            eps_pred, sJ_pred = model(x_t, t)

            # Gaussian DSM term (Eq. 49 uses s^G target; eps loss is equivalent)
            loss_eps = F.mse_loss(eps_pred, eps)

            # Jump contrastive term (Eq. 50)
            loss_con = contrastive_jump_loss(
                sJ_pred=sJ_pred,
                events=events,
                impulse_len=impulse_len,
                rng=rng,
                neg_per_pos=neg_per_pos,
                neg_scale=max(1e-3, 3.0 * b_mean),
            )

            loss = loss_eps + gamma * loss_con

            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tot += float(loss.item())
            tot_eps += float(loss_eps.item())
            tot_con += float(loss_con.item())
            nb += 1

        print(f"[Epoch {epoch+1}/{num_epochs}] loss={tot/nb:.6f} eps={tot_eps/nb:.6f} contrast={tot_con/nb:.6f}")

    return model


# ---------------------- Main ----------------------

def main():
    data_dir = "../synthetic_data/ECG_jump/train"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mirrors your additive script simplicity
    batch_size = 8
    num_epochs = 20
    lr = 1e-3
    seed = 0

    # Paper jump section: state-independent diffusion coefficient g(t)
    beta = 1.0  # g = sqrt(beta), sigma(t)^2 = beta * t

    # Jump parameters (these should match your generator regime)
    lambda_rate = 2.0   # expected jumps per signal at t=1
    jump_scale  = 2.0   # Laplace scale multiplier relative to RMS(x0)
    impulse_len = 1

    # Loss weights (paper Eq. 49)
    gamma = 0.2
    neg_per_pos = 8
    grad_clip = 1.0

    print(f"Loading clean ECG from {data_dir} ...")
    ds = ECGCleanDataset(data_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Dataset: {len(ds)} signals, mean={ds.mean:.4f}, std={ds.std:.4f}")

    sde = JumpDiffusionSDE(beta=beta)
    model = JumpScoreNet1D(channels=64, num_blocks=4, time_dim=128)

    model = train_jump_model(
        model=model,
        sde=sde,
        dataloader=dl,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        lambda_rate=lambda_rate,
        jump_scale=jump_scale,
        impulse_len=impulse_len,
        gamma=gamma,
        neg_per_pos=neg_per_pos,
        grad_clip=grad_clip,
        seed=seed,
    )

    ckpt = {
        "state_dict": model.state_dict(),
        "beta": float(beta),
        "mean": float(ds.mean),
        "std": float(ds.std),
        "signal_length": int(ds.signals.shape[-1]),
        "arch": {"channels": 64, "num_blocks": 4, "time_dim": 128},
        "lambda_rate": float(lambda_rate),
        "jump_scale": float(jump_scale),
        "impulse_len": int(impulse_len),
        "gamma": float(gamma),
        "neg_per_pos": int(neg_per_pos),
    }

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/ecg_jump_sde_model.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
