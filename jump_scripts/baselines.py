# baselines_jump.py
"""
Baselines + our model for JUMP / IMPULSIVE noise denoising on 1-D ECG signals.

Only includes the most relevant impulsive-noise baselines:
  - Median filter
  - Hampel filter

Plus our method:
  - SDE-Jump (trained model from sde_train_jump.py, applied via sde_denoiser_jump.py)

Matches the high-level structure of your additive baselines.py:
  - scan a folder of .npz files
  - dynamically detect SNR levels from the dataset
  - compute metrics per file/channel
  - aggregate by SNR and print a dynamic table

Expected dataset folder:
  ../synthetic_data/ECG_jump

Expected NPZ keys:
  clean, noisy, fs, snr_db
Optional but strongly recommended:
  jump_mask (for jump-aware metrics)

Run:
  python baselines_jump.py
"""

import numpy as np
from pathlib import Path

from sde_denoiser import denoise_sde_single_channel


# ---------------------- Robust impulsive baselines ----------------------

def sliding_window_view_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x[:, None]
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(xp, window_shape=window)


def median_filter_1d(noisy: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return noisy.copy()
    sw = sliding_window_view_1d(noisy, window)
    return np.median(sw, axis=-1)


def hampel_filter_1d(noisy: np.ndarray, window: int = 9, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Hampel filter: robust outlier suppression via local median + MAD.
    """
    if window <= 1:
        return noisy.copy()

    sw = sliding_window_view_1d(noisy, window)
    med = np.median(sw, axis=-1)

    abs_dev = np.abs(sw - med[:, None])
    mad = np.median(abs_dev, axis=-1) + 1e-12
    sigma = 1.4826 * mad  # MAD->std for Gaussian

    out = noisy.copy()
    outlier_mask = np.abs(noisy - med) > (n_sigmas * sigma)
    out[outlier_mask] = med[outlier_mask]
    return out


# ---------------------- Metrics ----------------------

def _snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    sig_pow = np.mean(clean.astype(np.float64) ** 2)
    noise_pow = np.mean((clean.astype(np.float64) - est.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(sig_pow / (noise_pow + 1e-12)))


def calculate_metrics(
    clean: np.ndarray,
    den: np.ndarray,
    noisy: np.ndarray,
    jump_mask: np.ndarray | None = None,
) -> dict:
    clean = clean.astype(np.float64)
    den = den.astype(np.float64)
    noisy = noisy.astype(np.float64)

    mse = float(np.mean((clean - den) ** 2))
    snr_out = _snr_db(clean, den)
    snr_in = _snr_db(clean, noisy)
    snr_impr = snr_out - snr_in

    max_val = float(np.max(np.abs(clean)) + 1e-12)
    psnr = float(10.0 * np.log10((max_val ** 2) / (mse + 1e-12)))

    corr = float(np.corrcoef(clean, den)[0, 1]) if np.std(clean) > 0 and np.std(den) > 0 else 0.0
    rel_err = float(np.linalg.norm(clean - den) / (np.linalg.norm(clean) + 1e-12))

    out = {
        "input_snr": float(snr_in),
        "snr": float(snr_out),
        "snr_improvement": float(snr_impr),
        "mse": mse,
        "psnr": psnr,
        "correlation": corr,
        "relative_error": rel_err,
    }

    # Jump-aware metrics (if available)
    if jump_mask is not None:
        jm = jump_mask.astype(bool)
        if jm.shape != clean.shape:
            jm = jm.reshape(clean.shape)

        inlier = ~jm
        outlier = jm

        if np.any(inlier):
            in_mse = float(np.mean((clean[inlier] - den[inlier]) ** 2))
            in_snr = _snr_db(clean[inlier], den[inlier])
        else:
            in_mse, in_snr = float("nan"), float("nan")

        if np.any(outlier):
            out_mae = float(np.mean(np.abs(clean[outlier] - den[outlier])))
            out_mse = float(np.mean((clean[outlier] - den[outlier]) ** 2))
        else:
            out_mae, out_mse = float("nan"), float("nan")

        out.update({
            "inlier_snr": float(in_snr),
            "inlier_mse": float(in_mse),
            "outlier_mae": float(out_mae),
            "outlier_mse": float(out_mse),
        })

    return out


# ---------------------- Dynamic table printing ----------------------

def print_dynamic_table(grouped_results: dict, methods: list, snr_levels: list, rows: list):
    for method_name in methods:
        print(f"Method: {method_name}")

        snr_headers = [f"{snr:.1f} dB" for snr in snr_levels]
        all_rows = []
        for label, key, fmt in rows:
            row_vals = []
            for snr in snr_levels:
                vals = grouped_results[method_name][snr][key]
                if vals:
                    row_vals.append(fmt.format(float(np.nanmean(vals))))
                else:
                    row_vals.append("-")
            all_rows.append((label, row_vals))

        metric_width = max(len("Metric"), max(len(label) for label, _ in all_rows))
        snr_widths = []
        for i, hdr in enumerate(snr_headers):
            w = len(hdr)
            for _, rv in all_rows:
                w = max(w, len(rv[i]))
            snr_widths.append(w)

        header_parts = [f"{'Metric':<{metric_width}}"]
        for hdr, w in zip(snr_headers, snr_widths):
            header_parts.append(f"{hdr:>{w}}")
        header_line = " | ".join(header_parts)
        print(header_line)
        print("-" * len(header_line))

        for label, rv in all_rows:
            row_parts = [f"{label:<{metric_width}}"]
            for val, w in zip(rv, snr_widths):
                row_parts.append(f"{val:>{w}}")
            print(" | ".join(row_parts))

        print()


# ---------------------- Main ----------------------

def main():
    data_dir = Path("../synthetic_data/ECG_jump/test")

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        print("Please generate data first (generate_ecg_jump.py).")
        return

    npz_files = sorted(list(data_dir.glob("*.npz")))
    if not npz_files:
        print(f"Error: No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_files)} files in {data_dir}")

    # Detect SNR levels dynamically
    snr_levels = []
    has_jump_mask = False
    for fp in npz_files:
        d = np.load(fp)
        if "snr_db" in d:
            snr_levels.append(float(d["snr_db"]))
        if "jump_mask" in d:
            has_jump_mask = True

    if not snr_levels:
        print("Error: No 'snr_db' values found in NPZ files.")
        return
    snr_levels = sorted(set(snr_levels))

    methods = ["Median", "Hampel", "SDE-Jump"]

    # Keep your standard metrics + jump-aware ones if available
    metric_keys = [
        "input_snr",
        "snr",
        "snr_improvement",
        "mse",
        "psnr",
        "correlation",
        "relative_error",
    ]
    if has_jump_mask:
        metric_keys += ["inlier_snr", "inlier_mse", "outlier_mae", "outlier_mse"]

    grouped_results = {
        method: {snr: {k: [] for k in metric_keys} for snr in snr_levels}
        for method in methods
    }

    baseline_specs = [
        ("Median", lambda x: median_filter_1d(x, window=5)),
        ("Hampel", lambda x: hampel_filter_1d(x, window=9, n_sigmas=3.0)),
    ]

    print("Processing files...\n")

    for fp in npz_files:
        d = np.load(fp)
        clean = d["clean"]
        noisy = d["noisy"]
        snr_db = float(d["snr_db"])

        jump_mask = d["jump_mask"] if "jump_mask" in d else None

        # Ensure shape (n_channels, n_samples)
        if clean.ndim == 1:
            clean = clean[None, :]
            noisy = noisy[None, :]
            if jump_mask is not None and jump_mask.ndim == 1:
                jump_mask = jump_mask[None, :]

        n_channels = clean.shape[0]

        # Median + Hampel
        for method_name, method_func in baseline_specs:
            for ch in range(n_channels):
                c = clean[ch]
                n = noisy[ch]
                jm = jump_mask[ch] if jump_mask is not None else None

                den = method_func(n)
                metrics = calculate_metrics(c, den, n, jm)

                bucket = grouped_results[method_name][snr_db]
                for k in metric_keys:
                    bucket[k].append(metrics.get(k, np.nan))

        # Our model
        method_name = "SDE-Jump"
        for ch in range(n_channels):
            c = clean[ch]
            n = noisy[ch]
            jm = jump_mask[ch] if jump_mask is not None else None

            den = denoise_sde_single_channel(n, snr_db)
            metrics = calculate_metrics(c, den, n, jm)

            bucket = grouped_results[method_name][snr_db]
            for k in metric_keys:
                bucket[k].append(metrics.get(k, np.nan))

    print("\n" + "=" * 90)
    print("IMPULSIVE / JUMP DENOISING RESULTS (ECG_jump), grouped by BACKGROUND SNR")
    print("Averaged across all files and channels for each SNR level")
    print("=" * 90 + "\n")

    rows = [
        ("Input SNR (dB)", "input_snr", "{:.2f}"),
        ("Output SNR (dB)", "snr", "{:.2f}"),
        ("SNR Improvement (dB)", "snr_improvement", "{:.2f}"),
        ("MSE", "mse", "{:.4e}"),
        ("PSNR (dB)", "psnr", "{:.2f}"),
        ("Correlation", "correlation", "{:.4f}"),
        ("Relative Error", "relative_error", "{:.4f}"),
    ]
    if has_jump_mask:
        rows += [
            ("Inlier SNR (dB)", "inlier_snr", "{:.2f}"),
            ("Inlier MSE", "inlier_mse", "{:.4e}"),
            ("Outlier MAE", "outlier_mae", "{:.4e}"),
            ("Outlier MSE", "outlier_mse", "{:.4e}"),
        ]

    print_dynamic_table(grouped_results, methods, snr_levels, rows)

    print("=" * 90)
    print("Processing complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
