# generate_ecg_jump.py
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class ECGMorphology:
    """
    Simple PQRST morphology as sum of Gaussians in time within each beat.
    This is a common synthetic ECG approximation and is aligned with the
    morphology controls used in ECGSYN-style generators. :contentReference[oaicite:2]{index=2}
    """
    # Peak locations (seconds) relative to R-peak at t=0
    t_p: float = -0.20
    t_q: float = -0.05
    t_r: float = 0.00
    t_s: float = 0.05
    t_t: float = 0.30

    # Amplitudes (mV-ish scale; relative)
    a_p: float = 0.10
    a_q: float = -0.15
    a_r: float = 1.00
    a_s: float = -0.25
    a_t: float = 0.30

    # Widths (seconds)
    w_p: float = 0.040
    w_q: float = 0.012
    w_r: float = 0.010
    w_s: float = 0.014
    w_t: float = 0.060


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def generate_clean_ecg(
    duration_s: float = 10.0,
    fs: int = 360,
    n_channels: int = 1,
    mean_hr_bpm: float = 70.0,
    std_hr_bpm: float = 2.5,
    baseline_wander: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Generate synthetic multi-channel ECG by stitching beats whose morphology
    is a sum of Gaussians (P,Q,R,S,T). Heart rate varies mildly beat-to-beat.

    This is a pragmatic, literature-consistent ECG synth approach suitable
    for denoising benchmarks. ECGSYN uses a dynamical model; we use a
    morphology-controlled beat template approach that preserves PQRST structure. :contentReference[oaicite:3]{index=3}

    Returns:
        clean : (n_channels, n_samples) float32
        fs    : sampling rate
    """
    rng = np.random.default_rng(random_state)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples, dtype=np.float32) / fs

    clean = np.zeros((n_channels, n_samples), dtype=np.float32)

    # Per-channel slight morphology variation (like different leads)
    for ch in range(n_channels):
        morph = ECGMorphology()

        # Randomize morphology a bit per channel/base sample
        # (kept modest to stay "normal ECG"-like)
        morph.a_r *= rng.uniform(0.85, 1.15)
        morph.a_t *= rng.uniform(0.80, 1.20)
        morph.a_p *= rng.uniform(0.80, 1.20)
        morph.w_r *= rng.uniform(0.85, 1.15)
        morph.w_t *= rng.uniform(0.85, 1.15)

        # Generate RR intervals (seconds) from HR distribution
        # HR ~ N(mean, std), clipped to physiological range
        rr_times = []
        time_cursor = 0.0
        while time_cursor < duration_s + 1.0:  # overshoot to fill tail safely
            hr = float(rng.normal(mean_hr_bpm, std_hr_bpm))
            hr = float(np.clip(hr, 45.0, 120.0))
            rr = 60.0 / hr
            rr_times.append(rr)
            time_cursor += rr

        # Beat start times (R-peaks) in seconds
        r_peaks = np.cumsum(np.array(rr_times, dtype=np.float32))
        r_peaks = r_peaks[r_peaks < duration_s]

        # Build the signal by adding PQRST Gaussians around each R-peak
        sig = np.zeros_like(t, dtype=np.float32)

        # Beat-local time axis around each R-peak (we evaluate globally for simplicity)
        for r_t in r_peaks:
            # local time (seconds) relative to R peak
            tau = t - r_t

            # Sum of Gaussians
            beat = (
                morph.a_p * gaussian(tau, morph.t_p, morph.w_p)
                + morph.a_q * gaussian(tau, morph.t_q, morph.w_q)
                + morph.a_r * gaussian(tau, morph.t_r, morph.w_r)
                + morph.a_s * gaussian(tau, morph.t_s, morph.w_s)
                + morph.a_t * gaussian(tau, morph.t_t, morph.w_t)
            ).astype(np.float32)

            sig += beat

        # Optional baseline wander (respiration + drift)
        if baseline_wander:
            # Typical baseline wander components ~0.1–0.5 Hz
            bw1_f = rng.uniform(0.10, 0.33)
            bw2_f = rng.uniform(0.33, 0.50)
            bw_amp = rng.uniform(0.02, 0.06)  # mV-ish amplitude
            baseline = (
                bw_amp * np.sin(2 * np.pi * bw1_f * t)
                + 0.5 * bw_amp * np.sin(2 * np.pi * bw2_f * t + rng.uniform(0, 2 * np.pi))
            ).astype(np.float32)
            sig += baseline

        # Small DC offset + gain per channel
        sig += rng.normal(0.0, 0.01)
        sig *= rng.uniform(0.9, 1.1)

        # Normalize to a stable RMS so SNR is consistent across samples
        rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2) + 1e-12))
        sig = (sig / (rms + 1e-8)).astype(np.float32)

        clean[ch] = sig

    return clean, fs


def generate_white_noise(shape, random_state: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)


def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Same SNR definition as your NMR script:
    SNR_dB = 10 * log10(P_signal / P_noise).
    """
    sig_power = np.mean(clean.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2) + 1e-12

    snr_linear = 10.0 ** (snr_db / 10.0)
    desired_noise_power = sig_power / max(snr_linear, 1e-12)

    scale = np.sqrt(desired_noise_power / noise_power)
    noisy = clean + scale.astype(np.float32) * noise
    return noisy.astype(np.float32)


def sample_jump_noise(
    clean: np.ndarray,
    fs: int,
    duration_s: float,
    lambda_rate: float,
    jump_scale: float,
    impulse_len: int = 1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float, int]]]:
    """
    Compound Poisson impulses (finite activity):
      m ~ Poisson(lambda_rate * duration_s) per channel,
      locations uniform,
      magnitudes ~ Laplace(0, b) where b = jump_scale * RMS(clean).

    impulse_len:
      1  -> single-sample spikes
      >1 -> short burst (rectangular), common for sensor pops/clicks.

    Returns:
      jump_signal: (n_channels, n_samples)
      jump_mask:   (n_channels, n_samples) bool
      events:      list of (ch, idx_start, magnitude, impulse_len)
    """
    rng = np.random.default_rng(random_state)
    n_channels, n_samples = clean.shape

    jump_signal = np.zeros_like(clean, dtype=np.float32)
    jump_mask = np.zeros_like(clean, dtype=bool)
    events: list[tuple[int, int, float, int]] = []

    # Scale impulses relative to clean RMS for consistency
    clean_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2) + 1e-12))
    b = float(jump_scale * clean_rms)

    for ch in range(n_channels):
        m = int(rng.poisson(lam=max(lambda_rate * duration_s, 0.0)))
        if m <= 0:
            continue

        # Choose start indices; ensure we fit impulse_len
        max_start = max(n_samples - impulse_len, 1)
        idxs = rng.integers(low=0, high=max_start, size=m)

        # Laplace marks (common in impulsive-noise modeling; also mentioned as an example in your draft)
        mags = rng.laplace(loc=0.0, scale=b, size=m).astype(np.float32)

        for idx, mag in zip(idxs, mags):
            jump_signal[ch, idx : idx + impulse_len] += mag
            jump_mask[ch, idx : idx + impulse_len] = True
            events.append((ch, int(idx), float(mag), int(impulse_len)))

    return jump_signal, jump_mask, events


def main():
    # Mirror your folder style, but for ECG jump noise
    output_dir = "../synthetic_data/ECG_jump"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sampling parameters (ECG commonly evaluated at 360 Hz) :contentReference[oaicite:4]{index=4}
    duration_s = 10.0
    fs = 360
    n_channels = 1  # typical single-lead synthetic benchmark; keep configurable

    # Background SNR parameters (same meaning as your additive generator)
    snr_list_db = [-5, 0, 5, 10, 20]
    num_base_signals = 50
    num_noise_realizations_per_snr = 5

    # Jump parameters (finite-activity, sparse impulses)
    # lambda_rate: expected impulses per second
    lambda_rate_list = [0.2, 0.5, 1.0]  # sparse -> moderate (tune as needed)
    # jump_scale: impulse magnitude scale relative to clean RMS
    jump_scale_list = [1.0, 2.0, 4.0]   # stronger impulses as scale increases

    # Impulse shape
    impulse_len = 1  # set to 3 or 5 for short bursts

    print(f"Saving data to {output_dir}")
    print(
        f"Generating {num_base_signals} base clean ECG signals, "
        f"{len(snr_list_db)} SNRs, "
        f"{len(lambda_rate_list)} lambdas, "
        f"{len(jump_scale_list)} jump_scales, "
        f"{num_noise_realizations_per_snr} noise realizations per config."
    )

    total_files = 0
    base_seed = 12345
    rng_global = np.random.default_rng(base_seed)

    for base_idx in range(num_base_signals):
        # Vary HR stats slightly per base signal (normal rhythm range)
        mean_hr = float(rng_global.uniform(55.0, 95.0))
        std_hr = float(rng_global.uniform(1.0, 4.0))

        clean, fs_used = generate_clean_ecg(
            duration_s=duration_s,
            fs=fs,
            n_channels=n_channels,
            mean_hr_bpm=mean_hr,
            std_hr_bpm=std_hr,
            baseline_wander=True,
            random_state=int(rng_global.integers(0, 1_000_000)),
        )

        for snr_db in snr_list_db:
            for lambda_rate in lambda_rate_list:
                for jump_scale in jump_scale_list:
                    for k in range(num_noise_realizations_per_snr):
                        # Background Gaussian noise at target SNR
                        white = generate_white_noise(
                            clean.shape,
                            random_state=int(rng_global.integers(0, 1_000_000))
                        )
                        noisy_bg = add_noise_at_snr(clean, white, snr_db)

                        # Jump (impulsive) component
                        jump_signal, jump_mask, events = sample_jump_noise(
                            clean=clean,
                            fs=fs_used,
                            duration_s=duration_s,
                            lambda_rate=float(lambda_rate),
                            jump_scale=float(jump_scale),
                            impulse_len=int(impulse_len),
                            random_state=int(rng_global.integers(0, 1_000_000)),
                        )

                        noisy = (noisy_bg + jump_signal).astype(np.float32)

                        fname = (
                            f"ecg_base{base_idx:03d}_snr_{snr_db:+d}dB_"
                            f"lam_{lambda_rate:.2f}Hz_js_{jump_scale:.2f}_"
                            f"noise{k:02d}.npz"
                        )

                        fpath = os.path.join(output_dir, fname)
                        np.savez(
                            fpath,
                            clean=clean,
                            noisy=noisy,
                            fs=np.array(fs_used, dtype=np.int32),
                            snr_db=np.array(snr_db, dtype=np.float32),
                            lambda_rate=np.array(lambda_rate, dtype=np.float32),
                            jump_scale=np.array(jump_scale, dtype=np.float32),
                            jump_signal=jump_signal.astype(np.float32),
                            jump_mask=jump_mask.astype(np.bool_),
                        )

                        # Separate per-file txt with impulse events (as requested)
                        txt_path = fpath.replace(".npz", "_jumps.txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(f"file: {fname}\n")
                            f.write(f"fs: {fs_used}\n")
                            f.write(f"duration_s: {duration_s}\n")
                            f.write(f"snr_db (background): {snr_db}\n")
                            f.write(f"lambda_rate (Hz): {lambda_rate}\n")
                            f.write(f"jump_scale (x clean RMS): {jump_scale}\n")
                            f.write(f"impulse_len (samples): {impulse_len}\n")
                            f.write(f"mean_hr_bpm: {mean_hr:.2f}\n")
                            f.write(f"std_hr_bpm: {std_hr:.2f}\n")
                            f.write("\n# events: (channel, start_index, magnitude, impulse_len)\n")
                            for (ch, idx, mag, ilen) in events:
                                f.write(f"{ch}\t{idx}\t{mag:+.6f}\t{ilen}\n")

                        total_files += 1
                        print(f"Generated: {fname}")

    print(f"\nGenerated {total_files} files in {output_dir}/")


if __name__ == "__main__":
    main()
