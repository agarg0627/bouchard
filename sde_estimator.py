"""
Estimate forward SDE parameters from a 1D noisy signal.

Uses two-level jump detection: mild (5σ) for g_add, strict (Lee-Mykland) for jumps/g_mult.

Returns: dict with g_add, g_mult, lambda_jump, sigma_jump, n_jumps, total_time
"""

import numpy as np
import math

def estimate_sde_parameters(
    y: np.ndarray,
    dt: float = 1.0,
    max_iter: int = 6,
    kernel_window: int = 16,
    window_size: int | None = None,
    min_windows_for_regression: int = 6,
    jump_alpha: float = 0.01,
):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    increments = np.diff(y)
    n_inc = len(increments)
    T = dt * n_inc

    # Step 1: Mild jump detection (Mancini 5σ threshold)
    jump_mask = np.zeros(n_inc, dtype=bool)
    if n_inc > 2:
        mad = np.median(np.abs(increments - np.median(increments)))
        sigma_est = max(1e-12, mad / 0.6745)
        thresh_mild = 5.0 * sigma_est  
        jump_mask = np.abs(increments) > thresh_mild

    # Step 2: g_add via TSRV autocovariance
    g_add = 0.0
    
    if n_inc >= 3:
        inc_clean = increments.copy().astype(float)
        inc_clean[jump_mask] = np.nan
        
        # Find consecutive non-jump pairs
        valid_pairs = []
        for i in range(len(inc_clean) - 1):
            if not np.isnan(inc_clean[i]) and not np.isnan(inc_clean[i+1]):
                valid_pairs.append((inc_clean[i], inc_clean[i+1]))
        
        if len(valid_pairs) >= 5:
            pairs = np.array(valid_pairs)
            mean_inc = np.mean(pairs)
            cov_lag1 = np.mean((pairs[:, 0] - mean_inc) * (pairs[:, 1] - mean_inc))
            
            g_add_tsrv = float(-cov_lag1 * dt)
            g_add_tsrv = max(g_add_tsrv, 0.0)
            
            # Variance-based estimate for blending
            non_jump_inc = increments[~jump_mask]
            if len(non_jump_inc) > 2:
                T_eff = dt * len(non_jump_inc)
                g_add_var = float(np.sum(non_jump_inc ** 2) / (2.0 * T_eff))
            else:
                g_add_var = g_add_tsrv
            
            # Blend TSRV and variance estimates
            if len(valid_pairs) >= 20:
                g_add = 0.8 * g_add_tsrv + 0.2 * g_add_var
            elif len(valid_pairs) >= 10:
                g_add = 0.6 * g_add_tsrv + 0.4 * g_add_var
            else:
                g_add = 0.4 * g_add_tsrv + 0.6 * g_add_var
        else:
            non_jump_inc = increments[~jump_mask]
            if len(non_jump_inc) > 2:
                T_eff = dt * len(non_jump_inc)
                g_add = float(np.sum(non_jump_inc ** 2) / (2.0 * T_eff))
            else:
                g_add = 0.0
    else:
        g_add = 0.0

    # Step 3: Strict jump detection (Lee & Mykland)
    jump_mask = np.zeros(n_inc, dtype=bool)
    
    if n_inc >= 5:
        K = max(5, min(int(np.sqrt(n_inc)), n_inc // 4))
        
        # Critical value
        n = n_inc
        c_n = (np.sqrt(2 * np.log(n)) - 
               (np.log(np.pi) + np.log(np.log(n))) / (2 * np.sqrt(2 * np.log(n))))
        beta_star = -np.log(-np.log(1 - jump_alpha))
        critical_value = beta_star + c_n
        
        for i in range(n_inc):
            window_start = max(0, i - K)
            window_end = i
            
            if window_end - window_start < 2:
                continue
                
            window_inc = increments[window_start:window_end]
            
            if len(window_inc) < 2:
                continue
            
            # Local volatility
            abs_inc = np.abs(window_inc)
            if len(abs_inc) >= 2:
                bipower_sum = np.sum(abs_inc[:-1] * abs_inc[1:])
                bipower_var = (np.pi / 2.0) * bipower_sum / max(len(abs_inc) - 1, 1)
                sigma_local = np.sqrt(bipower_var / dt)
            else:
                med = np.median(window_inc)
                mad = np.median(np.abs(window_inc - med))
                sigma_local = max(mad / 0.6745, 1e-12)
            
            # Test statistic
            if sigma_local > 1e-12:
                L_i = np.abs(increments[i]) / (sigma_local * np.sqrt(dt))
                
                if L_i > critical_value:
                    jump_mask[i] = True
    else:
        mad = np.median(np.abs(increments - np.median(increments)))
        sigma_est = max(1e-12, mad / 0.6745)
        thresh = 4.0 * sigma_est
        jump_mask = np.abs(increments) > thresh

    # Jump statistics
    jump_indices = np.where(jump_mask)[0]
    jump_sizes = increments[jump_mask]
    n_jumps = int(jump_indices.size)
    lambda_jump = float(n_jumps / T) if T > 0 else 0.0
    
    if n_jumps >= 2:
        jump_magnitudes = np.abs(jump_sizes.astype(float))
        sigma_jump_std = np.std(jump_magnitudes, ddof=1)
        
        median_mag = np.median(jump_magnitudes)
        mad = np.median(np.abs(jump_magnitudes - median_mag))
        sigma_jump_mad = mad / 0.6745
        
        sigma_jump = 0.3 * sigma_jump_mad + 0.7 * sigma_jump_std
    else:
        sigma_jump = 0.0

    # Step 4: g_mult via second-order difference regression
    if window_size is None:
        window = max(16, len(y) // 128)
    else:
        window = max(8, int(window_size))

    seg_mean_sqs = []
    seg_vars = []

    if len(y) >= 3:
        second_diff = y[2:] - 2.0 * y[1:-1] + y[:-2]
    else:
        second_diff = np.array([], dtype=np.float64)

    if len(jump_mask) >= 2:
        jump_mask_2nd = jump_mask[:-1] | jump_mask[1:]
    else:
        jump_mask_2nd = np.array([], dtype=bool)

    for i in range(0, len(y), window):
        seg = y[i : i + window]
        if seg.size < 5:
            continue
        
        seg_2nd = second_diff[max(0, i) : i + window - 2] if i + window - 2 <= len(second_diff) else np.array([])
        if seg_2nd.size < 3:
            continue
            
        jm_slice_2nd = jump_mask_2nd[max(0, i) : i + len(seg_2nd)] if i + len(seg_2nd) <= len(jump_mask_2nd) else np.array([], dtype=bool)
        
        min_len = min(len(seg_2nd), len(jm_slice_2nd))
        if min_len < 3:
            continue
        seg_2nd = seg_2nd[:min_len]
        jm_slice_2nd = jm_slice_2nd[:min_len]
        
        if jm_slice_2nd.sum() > max(0, 0.25 * len(jm_slice_2nd)):
            continue
        seg_2nd_clean = seg_2nd[~jm_slice_2nd]
        if seg_2nd_clean.size < 3:
            continue
            
        seg_mean_sq = float(np.mean(seg ** 2))
        seg_var = float(np.var(seg_2nd_clean))
        seg_mean_sqs.append(seg_mean_sq)
        seg_vars.append(seg_var)

    seg_mean_sqs = np.asarray(seg_mean_sqs, dtype=np.float64)
    seg_vars = np.asarray(seg_vars, dtype=np.float64)

    g_mult = 0.0
    if seg_mean_sqs.size >= min_windows_for_regression:
        X = np.vstack([np.ones_like(seg_mean_sqs), seg_mean_sqs]).T
        coef, _, _, _ = np.linalg.lstsq(X, seg_vars, rcond=None)
        a_fit, b_fit = float(coef[0]), float(coef[1])
        
        # Refit without outliers
        preds = X @ coef
        residuals = seg_vars - preds
        abs_res = np.abs(residuals)
        thr = np.percentile(abs_res, 90.0)
        keep_mask = abs_res <= thr
        
        if keep_mask.sum() >= min_windows_for_regression:
            coef, _, _, _ = np.linalg.lstsq(X[keep_mask], seg_vars[keep_mask], rcond=None)
            a_fit, b_fit = float(coef[0]), float(coef[1])
        
        b_fit = max(b_fit, 0.0)
        g_mult_est = float(np.sqrt(b_fit / 6.0))
        g_add_est = max(a_fit / 6.0, 0.0)
        
        # EIV correction iterations
        for _ in range(3):
            denom = 1.0 + g_mult_est ** 2
            seg_mean_sqs_corrected = (seg_mean_sqs - g_add_est) / max(denom, 1e-12)
            seg_mean_sqs_corrected = np.maximum(seg_mean_sqs_corrected, 1e-12)
            
            X_corr = np.vstack([np.ones_like(seg_mean_sqs_corrected), seg_mean_sqs_corrected]).T
            if keep_mask.sum() >= min_windows_for_regression:
                coef, _, _, _ = np.linalg.lstsq(X_corr[keep_mask], seg_vars[keep_mask], rcond=None)
            else:
                coef, _, _, _ = np.linalg.lstsq(X_corr, seg_vars, rcond=None)
            a_fit, b_fit = float(coef[0]), float(coef[1])
            
            b_fit = max(b_fit, 0.0)
            g_mult_new = float(np.sqrt(b_fit / 6.0))
            g_add_est = max(a_fit / 6.0, 0.0)
            
            if abs(g_mult_new - g_mult_est) < 1e-9:
                break
            g_mult_est = g_mult_new
        
        g_mult = g_mult_est
    else:
        g_mult = 0.0

    # Step 5: g_add correction for multiplicative noise
    if g_mult > 0.0:
        mean_y_sq = float(np.mean(y ** 2))
        mult_contribution = (g_mult ** 2) * mean_y_sq
        
        correction_factor = min(mult_contribution / max(g_add, 1e-12), 0.5)
        g_add_corrected = g_add * (1 - correction_factor)
        g_add = max(g_add_corrected, 0.0)

    return {
        "g_add": float(g_add),
        "g_mult": float(g_mult),
        "lambda_jump": float(lambda_jump),
        "sigma_jump": float(sigma_jump),
        "n_jumps": int(n_jumps),
        "total_time": float(T),
    }