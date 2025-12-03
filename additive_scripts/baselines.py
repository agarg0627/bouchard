import os
import numpy as np
import pywt
from scipy.fft import fft, ifft
from pathlib import Path
from sde_denoiser import denoise_sde_single_channel


def estimate_noise_mad(signal):
    """
    Estimate noise standard deviation using Median Absolute Deviation (MAD)
    on high-frequency wavelet coefficients.
    
    Reference: Donoho & Johnstone (1994)
    """
    # Use db4 wavelet for noise estimation
    coeffs = pywt.wavedec(signal, 'db4', level=1)
    # Use highest frequency detail coefficients
    high_freq_coeffs = coeffs[-1]
    # MAD-based sigma estimate
    sigma = np.median(np.abs(high_freq_coeffs)) / 0.6745
    return sigma


def wiener_filter(noisy_signal, noise_variance=None):
    """
    Frequency-domain Wiener filter for 1D signal denoising.
    
    If noise_variance is None, it will be estimated using MAD.
    """
    if noise_variance is None:
        noise_variance = estimate_noise_mad(noisy_signal) ** 2
    
    # Transform to frequency domain
    signal_fft = fft(noisy_signal)
    
    # Estimate signal power spectrum
    power_spectrum = np.abs(signal_fft) ** 2
    
    # Wiener filter in frequency domain
    # H(f) = S(f) / (S(f) + N(f))
    # where S(f) is signal power, N(f) is noise power
    wiener_gain = power_spectrum / (power_spectrum + noise_variance)
    
    # Apply filter
    filtered_fft = signal_fft * wiener_gain
    
    # Transform back to time domain
    filtered_signal = np.real(ifft(filtered_fft))
    
    return filtered_signal


def wavelet_soft_threshold(noisy_signal, noise_sigma=None):
    """
    Wavelet soft-thresholding denoising using VisuShrink.
    
    Reference: Donoho & Johnstone (1994)
    Wavelet: db4
    Threshold: Universal threshold (VisuShrink)
    """
    if noise_sigma is None:
        noise_sigma = estimate_noise_mad(noisy_signal)
    
    # Decompose signal
    wavelet = 'db4'
    max_level = pywt.dwt_max_level(len(noisy_signal), wavelet)
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=max_level)
    
    # Universal threshold (VisuShrink)
    threshold = noise_sigma * np.sqrt(2 * np.log(len(noisy_signal)))
    
    # Apply soft thresholding to all detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    for detail_coeffs in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(detail_coeffs, threshold, mode='soft'))
    
    # Reconstruct signal
    denoised_signal = pywt.waverec(coeffs_thresh, wavelet)
    
    # Handle length mismatch due to decomposition
    if len(denoised_signal) > len(noisy_signal):
        denoised_signal = denoised_signal[:len(noisy_signal)]
    
    return denoised_signal


def total_variation_denoise(noisy_signal, lambda_tv=None, max_iter=100, tol=1e-4):
    """
    Total Variation (ROF) denoising for 1D signals.
    
    Minimizes: ||u - f||^2 + lambda * TV(u)
    where TV(u) = sum(|u[i+1] - u[i]|)
    
    Reference: Rudin, Osher & Fatemi (1992)
    """
    if lambda_tv is None:
        noise_sigma = estimate_noise_mad(noisy_signal)
        lambda_tv = 0.1 * noise_sigma
    
    n = len(noisy_signal)
    u = noisy_signal.copy()
    
    # Iterative gradient descent
    dt = 0.25  # Time step
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Compute gradients (forward differences)
        grad_forward = np.zeros(n)
        grad_forward[:-1] = u[1:] - u[:-1]
        
        # Compute gradients (backward differences)
        grad_backward = np.zeros(n)
        grad_backward[1:] = u[1:] - u[:-1]
        
        # TV term (divergence of normalized gradient)
        eps = 1e-8  # Small constant for numerical stability
        
        # Forward difference magnitude
        grad_mag_forward = np.abs(grad_forward) + eps
        
        # Compute divergence
        div = np.zeros(n)
        div[1:-1] = (grad_forward[1:-1] / grad_mag_forward[1:-1] - 
                     grad_backward[1:-1] / grad_mag_forward[:-2])
        div[0] = grad_forward[0] / grad_mag_forward[0]
        div[-1] = -grad_backward[-1] / grad_mag_forward[-2]
        
        # Update
        u = u + dt * (noisy_signal - u + lambda_tv * div)
        
        # Check convergence
        rel_change = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + eps)
        if rel_change < tol:
            break
    
    return u


def calculate_metrics(clean, denoised, noisy):
    """Calculate denoising performance metrics"""
    
    # Mean Squared Error
    mse = np.mean((clean - denoised) ** 2)
    
    # Signal-to-Noise Ratio (dB)
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - denoised) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Peak Signal-to-Noise Ratio (dB)
    max_val = np.max(np.abs(clean))
    psnr = 10 * np.log10((max_val ** 2) / (mse + 1e-10))
    
    # Correlation Coefficient
    corr = np.corrcoef(clean, denoised)[0, 1]
    
    # Relative Error
    rel_error = np.linalg.norm(clean - denoised) / np.linalg.norm(clean)
    
    # Input SNR (for reference)
    input_noise_power = np.mean((clean - noisy) ** 2)
    input_snr = 10 * np.log10(signal_power / (input_noise_power + 1e-10))
    
    return {
        'mse': mse,
        'snr': snr,
        'psnr': psnr,
        'correlation': corr,
        'relative_error': rel_error,
        'input_snr': input_snr
    }


def main():
    data_dir = Path("../synthetic_data/NMR_additive")

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        print("Please run nmr_noise.py first to generate the data.")
        return

    # Find all NPZ files
    npz_files = sorted(list(data_dir.glob("*.npz")))
    if not npz_files:
        print(f"Error: No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_files)} files in {data_dir}")

    # 1) Collect unique SNR levels from the data (snr_db field in NPZs)
    snr_levels = []
    for filepath in npz_files:
        data = np.load(filepath)
        if "snr_db" in data:
            snr_levels.append(float(data["snr_db"]))

    if not snr_levels:
        print("Error: No 'snr_db' field found in NPZ files; cannot group by SNR.")
        return

    # Deduplicate + sort
    snr_levels = sorted(set(snr_levels))

    methods = ["Wiener", "Wavelet", "TV", "SDE"]
    # We'll store these per-method, per-SNR, per-metric
    metric_keys = [
        "input_snr",
        "snr",
        "snr_improvement",
        "mse",
        "psnr",
        "correlation",
        "relative_error",
    ]

    # grouped_results[method][snr][metric] -> list of values (over files & channels)
    grouped_results = {
        method: {
            snr: {k: [] for k in metric_keys}
            for snr in snr_levels
        }
        for method in methods
    }

    # 2) Process all files and channels, accumulate into grouped_results
    print("Processing files...\n")
    for filepath in npz_files:
        data = np.load(filepath)
        clean = data["clean"]
        noisy = data["noisy"]
        true_snr = float(data["snr_db"])

        # Ensure shape (n_channels, n_samples)
        if clean.ndim == 1:
            clean = clean[None, :]
            noisy = noisy[None, :]

        n_channels = clean.shape[0]

        for method_name, method_func in [
            ("Wiener", wiener_filter),
            ("Wavelet", wavelet_soft_threshold),
            ("TV", total_variation_denoise),
        ]:
            for ch in range(n_channels):
                clean_ch = clean[ch]
                noisy_ch = noisy[ch]

                # Denoise with chosen baseline
                denoised_ch = method_func(noisy_ch)

                # Compute metrics for this (file, channel, method)
                metrics = calculate_metrics(clean_ch, denoised_ch, noisy_ch)
                snr_improvement = metrics["snr"] - metrics["input_snr"]

                bucket = grouped_results[method_name][true_snr]
                bucket["input_snr"].append(metrics["input_snr"])
                bucket["snr"].append(metrics["snr"])
                bucket["snr_improvement"].append(snr_improvement)
                bucket["mse"].append(metrics["mse"])
                bucket["psnr"].append(metrics["psnr"])
                bucket["correlation"].append(metrics["correlation"])
                bucket["relative_error"].append(metrics["relative_error"])

        # SDE denoiser (score-based SDE model)
        method_name = "SDE"
        for ch in range(n_channels):
            clean_ch = clean[ch]
            noisy_ch = noisy[ch]

            denoised_ch = denoise_sde_single_channel(noisy_ch, true_snr)
            metrics = calculate_metrics(clean_ch, denoised_ch, noisy_ch)
            snr_improvement = metrics["snr"] - metrics["input_snr"]

            bucket = grouped_results[method_name][true_snr]
            bucket["input_snr"].append(metrics["input_snr"])
            bucket["snr"].append(metrics["snr"])
            bucket["snr_improvement"].append(snr_improvement)
            bucket["mse"].append(metrics["mse"])
            bucket["psnr"].append(metrics["psnr"])
            bucket["correlation"].append(metrics["correlation"])
            bucket["relative_error"].append(metrics["relative_error"])



    # 3) Print dynamic tables: rows = metrics, columns = SNR levels
    print("\n" + "=" * 80)
    print("BASELINE DENOISING RESULTS (grouped by SNR)")
    print("Averaged across all files and channels for each SNR level")
    print("=" * 80 + "\n")

    # (row_label, metric_key, format_string)
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
        print(f"Method: {method_name}")
        
        # Pre-calculate all cell values for width determination
        snr_headers = [f"{snr:.1f} dB" for snr in snr_levels]
        all_rows = []
        for label, key, fmt in rows:
            row_vals = []
            for snr in snr_levels:
                values = grouped_results[method_name][snr][key]
                if values:
                    row_vals.append(fmt.format(float(np.mean(values))))
                else:
                    row_vals.append("-")
            all_rows.append((label, row_vals))
        
        # Calculate column widths
        metric_width = max(len("Metric"), max(len(label) for label, _ in all_rows))
        snr_widths = []
        for i, snr_header in enumerate(snr_headers):
            col_width = len(snr_header)
            for _, row_vals in all_rows:
                col_width = max(col_width, len(row_vals[i]))
            snr_widths.append(col_width)
        
        # Print header
        header_parts = [f"{'Metric':<{metric_width}}"]
        for snr_header, width in zip(snr_headers, snr_widths):
            header_parts.append(f"{snr_header:>{width}}")
        header_line = " | ".join(header_parts)
        print(header_line)
        print("-" * len(header_line))
        
        # Print data rows
        for label, row_vals in all_rows:
            row_parts = [f"{label:<{metric_width}}"]
            for val, width in zip(row_vals, snr_widths):
                row_parts.append(f"{val:>{width}}")
            print(" | ".join(row_parts))
        
        print()  # blank line between methods

    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()