import os
from pathlib import Path

import numpy as np


def generate_clean_nmr(
    duration_s: float = 10.0,
    fs: int = 256,
    n_channels: int = 4,
    random_state: int = None,
) -> tuple[np.ndarray, int]:
    """
    Generate a synthetic multi-channel NMR FID (Free Induction Decay) signal.

    Each channel is a sum of 3–8 damped sinusoids with different:
      - amplitudes
      - frequencies (chemical shifts)
      - decay constants (T2-like)
      - phases

    We also add a small baseline (offset + slow drift) to better mimic real data.

    Returns:
        clean : np.ndarray, shape (n_channels, n_samples)
        fs    : int, sampling rate (Hz)
    """
    rng = np.random.default_rng(random_state)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples, dtype=np.float32) / fs  # (n_samples,)

    clean = np.zeros((n_channels, n_samples), dtype=np.float32)

    for ch in range(n_channels):
        # Number of peaks (chemical shifts) for this channel
        n_peaks = rng.integers(3, 9)  # 3–8 peaks

        # Amplitudes: log-normal so we have a few dominant peaks + smaller ones
        amps = rng.lognormal(mean=0.0, sigma=0.5, size=n_peaks)  # >= 0
        # Frequencies in Hz: choose a plausible range of FID frequencies
        # (tune these as needed for your real NMR settings)
        freqs_hz = rng.uniform(5.0, 100.0, size=n_peaks)
        # Decay constants (seconds): shorter → faster decay
        t2s = rng.uniform(0.2, 2.5, size=n_peaks)
        # Random phases
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_peaks)

        signal = np.zeros_like(t, dtype=np.float32)

        for a, f, t2, ph in zip(amps, freqs_hz, t2s, phases):
            # Damped cosine component
            # exp(-t/T2) * cos(2π f t + phase)
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

        # Channel-specific gain + tiny additional offset
        gain = rng.uniform(0.8, 1.2)
        ch_offset = rng.normal(0.0, 0.01)
        clean[ch] = gain * (signal + ch_offset).astype(np.float32)

        # Optionally, you can normalize per-channel to a fixed RMS if desired
        # rms = np.sqrt(np.mean(clean[ch] ** 2) + 1e-12)
        # clean[ch] /= (rms + 1e-8)

    return clean, fs


def generate_white_noise(shape, random_state: int | None = None) -> np.ndarray:
    """
    Generate white Gaussian noise with unit variance.
    """
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)


def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale 'noise' to achieve the desired SNR (in dB) relative to 'clean', then add.

    SNR_dB = 10 * log10(P_signal / P_noise).

    Args:
        clean : np.ndarray, (n_channels, n_samples)
        noise : np.ndarray, same shape as clean
        snr_db: desired SNR level in dB

    Returns:
        noisy : np.ndarray, same shape as clean
    """
    # Compute average power over all channels and time
    sig_power = np.mean(clean.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2) + 1e-12

    # Desired noise power given SNR
    snr_linear = 10.0 ** (snr_db / 10.0)
    desired_noise_power = sig_power / max(snr_linear, 1e-12)

    # Scale noise to match desired noise power
    scale = np.sqrt(desired_noise_power / noise_power)
    noisy = clean + scale.astype(np.float32) * noise
    return noisy.astype(np.float32)


def main():
    # Output directory (same as before so your other scripts still work)
    output_dir = "../synthetic_data/NMR_additive"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sampling parameters
    duration_s = 10.0
    fs = 256
    n_channels = 4

    # Noise/SNR parameters
    snr_list_db = [-5, 0, 5, 10, 20]        # SNR levels to generate
    num_base_signals = 50                   # how many distinct clean FIDs to generate
    num_noise_realizations_per_snr = 5      # noise reps per clean+SNR

    print(f"Saving data to {output_dir}")
    print(
        f"Generating {num_base_signals} base clean signals, "
        f"{len(snr_list_db)} SNRs, "
        f"{num_noise_realizations_per_snr} noise realizations per SNR."
    )

    total_files = 0

    base_seed = 12345
    rng_global = np.random.default_rng(base_seed)

    for base_idx in range(num_base_signals):
        # Different clean NMR FID each time
        clean, fs_used = generate_clean_nmr(
            duration_s=duration_s,
            fs=fs,
            n_channels=n_channels,
            random_state=int(rng_global.integers(0, 1_000_000)),
        )

        for snr_db in snr_list_db:
            for k in range(num_noise_realizations_per_snr):
                # Independent noise realization
                white = generate_white_noise(clean.shape, random_state=int(rng_global.integers(0, 1_000_000)))
                noisy = add_noise_at_snr(clean, white, snr_db)

                fname = (
                    f"nmr_base{base_idx:03d}_white_snr_{snr_db:+d}dB_"
                    f"noise{k:02d}.npz"
                )

                np.savez(
                    os.path.join(output_dir, fname),
                    clean=clean,
                    noisy=noisy,
                    fs=np.array(fs_used, dtype=np.int32),
                    snr_db=np.array(snr_db, dtype=np.float32),
                )
                total_files += 1
                print(f"Generated: {fname}")

    print(f"\nGenerated {total_files} files in {output_dir}/")


if __name__ == "__main__":
    main()
