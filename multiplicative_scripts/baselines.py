import os
from pathlib import Path

import numpy as np
import pywt
from scipy.fft import fft, ifft
from sde_denoiser import denoise_sde_multiplicative_single_channel


def estimate_noise_mad(signal: np.ndarray) -> float:
    """
    Estimate noise standard deviation using Median Absolute Deviation (MAD)
    on high-frequency wavelet coefficients.
    """
    wavelet = "db4"
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    detail_coeffs = coeffs[-1]
    mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
    sigma = mad / 0.6745
    return float(sigma)


def wiener_filter(noisy_signal: np.ndarray, noise_variance: float | None = None) -> np.ndarray:
    """
    Frequency-domain Wiener filter for 1D signal denoising.
    """
    if noise_variance is None:
        noise_variance = estimate_noise_mad(noisy_signal) ** 2

    signal_fft = fft(noisy_signal)
    power_spectrum = np.abs(signal_fft) ** 2
    wiener_gain = power_spectrum / (power_spectrum + noise_variance + 1e-12)
    filtered_fft = signal_fft * wiener_gain
    filtered_signal = np.real(ifft(filtered_fft))
    return filtered_signal.astype(np.float32)


def wavelet_soft_threshold(noisy_signal: np.ndarray, noise_sigma: float | None = None) -> np.ndarray:
    """
    Wavelet soft-thresholding denoising using VisuShrink.
    """
    if noise_sigma is None:
        noise_sigma = estimate_noise_mad(noisy_signal)

    wavelet = "db4"
    max_level = pywt.dwt_max_level(len(noisy_signal), wavelet)
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=max_level)

    threshold = noise_sigma * np.sqrt(2 * np.log(len(noisy_signal) + 1e-12))

    coeffs_thresh = [coeffs[0]]
    for detail_coeffs in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(detail_coeffs, threshold, mode="soft"))

    denoised_signal = pywt.waverec(coeffs_thresh, wavelet)
    if len(denoised_signal) > len(noisy_signal):
        denoised_signal = denoised_signal[: len(noisy_signal)]
    return denoised_signal.astype(np.float32)


def total_variation_denoise(
    noisy_signal: np.ndarray,
    lambda_tv: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Total Variation (ROF) denoising for 1D signals.
    """
    if lambda_tv is None:
        noise_sigma = estimate_noise_mad(noisy_signal)
        lambda_tv = 0.1 * noise_sigma

    n = len(noisy_signal)
    u = noisy_signal.astype(np.float32).copy()

    dt = 0.25
    eps = 1e-8

    for _ in range(max_iter):
        u_old = u.copy()

        grad_forward = np.zeros(n, dtype=np.float32)
        grad_forward[:-1] = u[1:] - u[:-1]

        grad_backward = np.zeros(n, dtype=np.float32)
        grad_backward[1:] = u[1:] - u[:-1]

        grad_mag_forward = np.abs(grad_forward) + eps

        div = np.zeros(n, dtype=np.float32)
        div[1:-1] = (
            grad_forward[1:-1] / grad_mag_forward[1:-1]
            - grad_backward[1:-1] / grad_mag_forward[:-2]
        )
        div[0] = grad_forward[0] / grad_mag_forward[0]
        div[-1] = -grad_backward[-1] / grad_mag_forward[-2]

        u = u + dt * (noisy_signal - u + lambda_tv * div)

        rel_change = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + eps)
        if rel_change < tol:
            break

    return u.astype(np.float32)


# ----------------- Multiplicative-specific helpers -----------------


def _split_sign_and_log_magnitude(noisy_signal: np.ndarray, eps: float = 1e-6):
    noisy_signal = noisy_signal.astype(np.float32)
    sign = np.sign(noisy_signal)
    sign[sign == 0] = 1.0
    log_mag = np.log(np.abs(noisy_signal) + eps).astype(np.float32)
    return log_mag, sign.astype(np.float32)


def log_wiener_denoise(noisy_signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Log + Wiener (homomorphic) baseline.
    """
    y_log, sign = _split_sign_and_log_magnitude(noisy_signal, eps=eps)
    y_log_denoised = wiener_filter(y_log)
    mag_denoised = np.exp(y_log_denoised)
    return (sign * mag_denoised).astype(np.float32)


def vst_wavelet_denoise(noisy_signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    VST (log) + wavelet baseline.
    """
    y_log, sign = _split_sign_and_log_magnitude(noisy_signal, eps=eps)
    y_log_denoised = wavelet_soft_threshold(y_log)
    mag_denoised = np.exp(y_log_denoised)
    return (sign * mag_denoised).astype(np.float32)


def log_tv_denoise(noisy_signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Log + 1D TV baseline.
    """
    y_log, sign = _split_sign_and_log_magnitude(noisy_signal, eps=eps)
    y_log_denoised = total_variation_denoise(y_log)
    mag_denoised = np.exp(y_log_denoised)
    return (sign * mag_denoised).astype(np.float32)


def lee_filter_1d(noisy_signal: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    1D Lee-like filter for multiplicative noise.
    """
    noisy_signal = noisy_signal.astype(np.float32)
    n = len(noisy_signal)
    if n == 0:
        return noisy_signal

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size should be odd and >= 3")

    pad = window_size // 2
    padded = np.pad(noisy_signal, pad_width=pad, mode="reflect")
    kernel = np.ones(window_size, dtype=np.float32) / window_size

    local_mean = np.convolve(padded, kernel, mode="same")[pad:-pad]
    local_mean_sq = np.convolve(padded**2, kernel, mode="same")[pad:-pad]
    local_var = local_mean_sq - local_mean**2
    local_var = np.maximum(local_var, 0.0)

    noise_var = np.median(local_var)
    eps = 1e-8

    w = (local_var - noise_var) / (local_var + noise_var + eps)
    w = np.clip(w, 0.0, 1.0)

    filtered = local_mean + w * (noisy_signal - local_mean)
    return filtered.astype(np.float32)


# ----------------- Metrics & evaluation -----------------


def calculate_metrics(clean: np.ndarray, denoised: np.ndarray, noisy: np.ndarray) -> dict:
    clean = clean.astype(np.float32)
    denoised = denoised.astype(np.float32)
    noisy = noisy.astype(np.float32)

    mse = np.mean((clean - denoised) ** 2)
    signal_power = np.mean(clean**2)
    noise_power = np.mean((clean - denoised) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    max_val = np.max(np.abs(clean))
    psnr = 10 * np.log10((max_val**2) / (mse + 1e-10))

    clean_flat = clean - np.mean(clean)
    denoised_flat = denoised - np.mean(denoised)
    numerator = np.sum(clean_flat * denoised_flat)
    denominator = np.sqrt(
        np.sum(clean_flat**2) * np.sum(denoised_flat**2) + 1e-10
    )
    corr = numerator / denominator

    rel_error = np.linalg.norm(clean - denoised) / (np.linalg.norm(clean) + 1e-10)

    input_noise_power = np.mean((clean - noisy) ** 2)
    input_snr = 10 * np.log10(signal_power / (input_noise_power + 1e-10))

    return {
        "mse": float(mse),
        "snr": float(snr),
        "psnr": float(psnr),
        "correlation": float(corr),
        "relative_error": float(rel_error),
        "input_snr": float(input_snr),
    }


def bin_snr(input_snr: float, step: float = 5.0) -> float:
    """
    Bin an SNR value (in dB) to the nearest multiple of `step` (default 5 dB).
    e.g., 3.2 -> 5.0, -6.7 -> -5.0
    """
    return float(step * round(input_snr / step))


def main():
    # Point this at your flat multiplicative NMR folder
    data_dir = Path("../synthetic_data/NMR_multiplicative")

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        return

    # Allow for either flat or nested structure
    npz_files = sorted(list(data_dir.rglob("*.npz")))
    if not npz_files:
        print(f"Error: No .npz files found under {data_dir}")
        return

    print(f"Found {len(npz_files)} files under {data_dir}")

    methods = ["LogWiener", "VST", "LogTV", "Lee", "SDE"]
    metric_keys = [
        "input_snr",
        "snr",
        "snr_improvement",
        "mse",
        "psnr",
        "correlation",
        "relative_error",
    ]

    # grouped_results[method][snr_bin][metric] -> list of values
    grouped_results: dict[str, dict[float, dict[str, list[float]]]] = {
        m: {} for m in methods
    }

    print("Processing files...\n")

    for filepath in npz_files:
        data = np.load(filepath)
        clean = data["clean"]
        noisy = data["noisy"]
        data.close()

        if clean.ndim == 1:
            clean = clean[None, :]
            noisy = noisy[None, :]

        n_channels = clean.shape[0]

        for ch in range(n_channels):
            clean_ch = clean[ch]
            noisy_ch = noisy[ch]

            # Compute true input SNR for this channel (linear domain)
            signal_power = np.mean(clean_ch**2)
            noise_power = np.mean((clean_ch - noisy_ch) ** 2)
            true_snr = 10.0 * np.log10(signal_power / (noise_power + 1e-10))

            # Define methods for THIS channel, SDE gets the channel's SNR via default arg
            methods_for_channel = [
                ("LogWiener", log_wiener_denoise),
                ("VST", vst_wavelet_denoise),
                ("LogTV", log_tv_denoise),
                ("Lee", lee_filter_1d),
                (
                    "SDE",
                    lambda x, s=true_snr: denoise_sde_multiplicative_single_channel(
                        x, snr_db=s
                    ),
                ),
            ]

            for method_name, method_func in methods_for_channel:
                denoised_ch = method_func(noisy_ch)
                metrics = calculate_metrics(clean_ch, denoised_ch, noisy_ch)
                snr_improvement = metrics["snr"] - metrics["input_snr"]

                # Compute SNR bin from *this* channel's input SNR
                snr_bin = bin_snr(metrics["input_snr"], step=5.0)

                # Initialize bin dict if needed
                if snr_bin not in grouped_results[method_name]:
                    grouped_results[method_name][snr_bin] = {
                        k: [] for k in metric_keys
                    }

                bucket = grouped_results[method_name][snr_bin]
                bucket["input_snr"].append(metrics["input_snr"])
                bucket["snr"].append(metrics["snr"])
                bucket["snr_improvement"].append(float(snr_improvement))
                bucket["mse"].append(metrics["mse"])
                bucket["psnr"].append(metrics["psnr"])
                bucket["correlation"].append(metrics["correlation"])
                bucket["relative_error"].append(metrics["relative_error"])

    print("\n" + "=" * 80)
    print("MULTIPLICATIVE NOISE BASELINE DENOISING RESULTS (grouped by binned input SNR)")
    print("SNR bins are in 5 dB steps (rounded to nearest 5 dB).")
    print("=" * 80 + "\n")

    rows = [
        ("Input SNR (dB)", "input_snr", "{:.2f}"),
        ("Output SNR (dB)", "snr", "{:.2f}"),
        ("SNR Improvement (dB)", "snr_improvement", "{:.2f}"),
        ("MSE", "mse", "{:.4e}"),
        ("PSNR (dB)", "psnr", "{:.2f}"),
        ("Correlation", "correlation", "{:.4f}"),
        ("Relative Error", "relative_error", "{:.4f}"),
    ]

    for method_name in methods:
        method_bins = grouped_results[method_name]
        if not method_bins:
            print(f"Method: {method_name} (no data)")
            continue

        snr_bins = sorted(method_bins.keys())
        print(f"Method: {method_name}")

        snr_headers = [f"{snr:.1f} dB" for snr in snr_bins]
        all_rows: list[tuple[str, list[str]]] = []

        for label, key, fmt in rows:
            row_vals: list[str] = []
            for snr in snr_bins:
                values = method_bins[snr][key]
                if values:
                    row_vals.append(fmt.format(float(np.mean(values))))
                else:
                    row_vals.append("-")
            all_rows.append((label, row_vals))

        metric_width = max(len("Metric"), max(len(label) for label, _ in all_rows))
        snr_widths: list[int] = []
        for i, snr_header in enumerate(snr_headers):
            col_width = len(snr_header)
            for _, row_vals in all_rows:
                col_width = max(col_width, len(row_vals[i]))
            snr_widths.append(col_width)

        header_parts = [f"{'Metric':<{metric_width}}"]
        for snr_header, width in zip(snr_headers, snr_widths):
            header_parts.append(f"{snr_header:>{width}}")
        header_line = " | ".join(header_parts)

        print(header_line)
        print("-" * len(header_line))

        for label, row_vals in all_rows:
            row_parts = [f"{label:<{metric_width}}"]
            for val, width in zip(row_vals, snr_widths):
                row_parts.append(f"{val:>{width}}")
            print(" | ".join(row_parts))

        print()

    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
