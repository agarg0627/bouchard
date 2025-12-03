import os
from pathlib import Path

import numpy as np


def generate_clean_nmr(
    duration_s: float = 10.0,
    fs: int = 256,
    n_channels: int = 4,
    random_state: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Generate a synthetic multi-channel NMR-like FID signal.

    Each channel is a sum of 3–8 damped sinusoids with different
    amplitudes, frequencies, T2 (decay constants), and phases,
    plus a small baseline (offset + slow drift).

    Returns
    -------
    clean : np.ndarray
        Array of shape (n_channels, n_samples), dtype float32.
    fs : int
        Sampling frequency (Hz).
    """
    rng = np.random.default_rng(random_state)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples, dtype=np.float32) / fs

    clean = np.zeros((n_channels, n_samples), dtype=np.float32)

    for ch in range(n_channels):
        # Number of damped components
        n_peaks = rng.integers(3, 9)  # 3–8 peaks

        signal = np.zeros_like(t)

        # Draw frequencies in a band typical for NMR-like oscillations
        freqs_hz = rng.uniform(5.0, 80.0, size=n_peaks)  # adjust if desired
        # Amplitudes
        amps = rng.uniform(0.2, 1.0, size=n_peaks)
        # Decay time constants T2 (seconds)
        t2s = rng.uniform(0.1 * duration_s, 1.0 * duration_s, size=n_peaks)
        # Random phases
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_peaks)

        for a, f, t2, ph in zip(amps, freqs_hz, t2s, phases):
            # Damped cosine: exp(-t/T2) * cos(2π f t + phase)
            signal += (
                a
                * np.exp(-t / t2, dtype=np.float32)
                * np.cos(2.0 * np.pi * f * t + ph, dtype=np.float32)
            )

        # Add a small baseline (offset + slow drift)
        offset = rng.normal(0.0, 0.02)
        slope = rng.normal(0.0, 0.01)  # slow linear drift
        baseline = offset + slope * (t - t.mean())
        signal += baseline.astype(np.float32)

        clean[ch] = signal.astype(np.float32)

    # Normalize each channel roughly to [-1, 1] for stable noise scaling
    max_abs = np.max(np.abs(clean), axis=1, keepdims=True) + 1e-8
    clean = (clean / max_abs).astype(np.float32)

    return clean, fs


def add_multiplicative_lognormal_noise(
    clean: np.ndarray,
    sigma_log: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply multiplicative log-normal noise:

        y = x * n

    where n is log-normal with E[n] = 1 and Var controlled by sigma_log^2.

    epsilon ~ N(0, sigma_log^2)
    n = exp(epsilon - 0.5 * sigma_log^2)  => E[n] = 1.

    Parameters
    ----------
    clean : np.ndarray
        Clean signal, shape (n_channels, n_samples).
    sigma_log : float
        Std of the Gaussian in log-domain.
    rng : np.random.Generator

    Returns
    -------
    noisy : np.ndarray
        Noisy signal, same shape as clean.
    snr_db : np.ndarray
        Per-channel SNR in dB, shape (n_channels,).
    """
    # Draw epsilon per sample and channel
    epsilon = rng.normal(
        loc=0.0,
        scale=sigma_log,
        size=clean.shape,
    ).astype(np.float32)

    # Log-normal multiplier with mean 1
    n = np.exp(epsilon - 0.5 * sigma_log**2).astype(np.float32)

    noisy = (clean * n).astype(np.float32)

    # Compute realized SNR per channel
    # SNR = 10 * log10( E[x^2] / E[(y - x)^2] )
    noise = noisy - clean
    sig_power = np.mean(clean**2, axis=1) + 1e-12
    noise_power = np.mean(noise**2, axis=1) + 1e-12
    snr_linear = sig_power / noise_power
    snr_db = 10.0 * np.log10(snr_linear).astype(np.float32)

    return noisy, snr_db


def main():
    # Output directory (parallel to your additive NMR_additive)
    output_root = "../synthetic_data/NMR_multiplicative"
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # Sampling parameters
    duration_s = 10.0
    fs = 256
    n_channels = 4

    # Multiplicative noise strengths in log-domain
    # (you can tune these to roughly match -5, 0, 5, 10, 20 dB regimes)
    sigma_log_list = [0.05, 0.10, 0.20, 0.30, 0.40]

    # Number of signals per sigma_log level
    num_files_per_level = 100

    base_seed = 12345
    total_files = 0

    for level_idx, sigma_log in enumerate(sigma_log_list):
        out_dir = Path(output_root) 
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== sigma_log = {sigma_log:.3f} -> {out_dir} ===")

        for i in range(num_files_per_level):
            # Use different seeds for variety
            seed = base_seed + level_idx * 10000 + i
            rng = np.random.default_rng(seed)

            clean, fs_used = generate_clean_nmr(
                duration_s=duration_s,
                fs=fs,
                n_channels=n_channels,
                random_state=seed,
            )

            noisy, snr_db = add_multiplicative_lognormal_noise(
                clean=clean,
                sigma_log=sigma_log,
                rng=rng,
            )

            fname = f"nmr_mult_sigma{sigma_log:.2f}_idx{i:04d}.npz".replace(".", "p")
            out_path = out_dir / fname

            np.savez_compressed(
                out_path,
                clean=clean.astype(np.float32),
                noisy=noisy.astype(np.float32),
                fs=np.array(fs_used, dtype=np.int32),
                sigma_log=np.array(sigma_log, dtype=np.float32),
                snr_db=snr_db,  # per-channel realized SNR
            )

            total_files += 1
            print(f"Generated: {out_path}")

    print(f"\nGenerated {total_files} files under {output_root}/")


if __name__ == "__main__":
    main()
