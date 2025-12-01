"""
sde_estimator.py

Estimate forward SDE parameters from a 1D noisy signal:
- g_add      : additive noise variance (σ_add²)
- g_mult     : multiplicative noise std (std of multiplicative factor m, where noise = clean * m)
- lambda_jump: jump rate (jumps per unit time)
- sigma_jump : jump magnitude std (std of detected jump increments)
- n_jumps    : number of detected jumps

Estimation approach:
1. Mancini (2009) / Figueroa-López & Mancini (2019) optimal threshold jump detection.
2. Quadratic variation on jump-free increments -> raw total variance estimate.
3. Second-order difference regression with iterative errors-in-variables correction:
   - Uses Δ²y to reduce signal variation confound (filters linear trends)
   - Iteratively corrects E[y²] -> E[clean²] to handle EIV bias
   - Slope of Var(Δ²y) vs E[clean²] gives 6·σ_mult²
4. Correct g_add by removing multiplicative contribution.
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
):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    increments = np.diff(y)
    n_inc = len(increments)
    T = dt * n_inc

    # ----------------------------
    # Step 1: Mancini (2009) optimal threshold jump detection
    # Threshold: r(dt) = c * σ * dt^ω, where ω ∈ (0, 0.5)
    # This scales correctly: diffusive increments ~ √dt, jumps ~ O(1)
    # ----------------------------
    if n_inc <= 0:
        return {
            "g_add": 0.0,
            "g_mult": 0.0,
            "lambda_jump": 0.0,
            "sigma_jump": 0.0,
            "n_jumps": 0,
            "total_time": 0.0,
        }

    # initial volatility estimate using MAD (handles jumps better than std)
    mad = np.median(np.abs(increments - np.median(increments)))
    sigma_est = max(1e-12, mad / 0.6745)  # scale MAD to std for Gaussian

    # Mancini threshold parameters
    # Figueroa-López & Mancini (2019) show optimal threshold is proportional to
    # the Lévy modulus of continuity: σ * √(2 * dt * log(1/dt))
    # For dt >= 1, log(1/dt) <= 0, so fall back to Mancini (2009) form
    if dt < 1.0:
        modulus = np.sqrt(2.0 * dt * np.log(1.0 / dt))
    else:
        # fall back to dt^omega form for large dt
        omega = 0.47
        modulus = dt ** omega
    
    c = 3.0  # threshold constant

    jump_mask = np.zeros(n_inc, dtype=bool)

    for _ in range(max_iter):
        # optimal threshold per Figueroa-López & Mancini (2019)
        thresh = c * sigma_est * modulus
        jump_mask_new = np.abs(increments) > thresh

        # update volatility from non-jump increments via quadratic variation
        non_jump_inc = increments[~jump_mask_new]
        if len(non_jump_inc) > 2:
            T_eff = dt * len(non_jump_inc)
            sigma_est_new = np.sqrt(np.sum(non_jump_inc ** 2) / T_eff)
        else:
            sigma_est_new = sigma_est

        # check convergence
        if np.array_equal(jump_mask_new, jump_mask) and abs(sigma_est_new - sigma_est) < 1e-9:
            break

        jump_mask = jump_mask_new
        sigma_est = max(1e-12, sigma_est_new)

    # finalize jump stats
    jump_indices = np.where(jump_mask)[0]
    jump_sizes = increments[jump_mask]
    n_jumps = int(jump_indices.size)
    lambda_jump = float(n_jumps / T) if T > 0 else 0.0
    
    # Truncation-corrected jump size std (Gaussian truncated MLE) 
    # Source: Mancini (2009); Figueroa-López & Mancini (2019)
    if n_jumps > 1:
        # use the final threshold used in detection
        r = float(thresh)
        z = jump_sizes.astype(float)
        abs_r = abs(r)

        # Negative log-likelihood for N(0, σ²) truncated to |z| > r
        def nll(sigma):
            sigma = float(sigma)
            if sigma <= 0:
                return 1e12
            # truncation probability: P(|J| > r)
            denom = 1.0 - 2.0 * 0.5 * (1.0 + math.erf(-abs_r / (sigma * np.sqrt(2))))
            if denom <= 0:
                return 1e12
            quad = np.sum(z * z) / (2.0 * sigma * sigma)
            n = len(z)
            return n * np.log(sigma) + n * (-np.log(denom)) + quad

        # 1D golden-section search 
        sigma0 = np.std(z)
        a, b = 1e-9, max(5 * sigma0, 1e-6)
        phi = (1 + np.sqrt(5)) / 2
        for _ in range(80):
            c1 = b - (b - a) / phi
            c2 = a + (b - a) / phi
            if nll(c1) < nll(c2):
                b = c2
            else:
                a = c1

        sigma_jump = float((a + b) / 2)
    else:
        sigma_jump = 0.0

    # ----------------------------
    # Step 2: g_add (raw) via quadratic variation on jump-free increments
    # Note: This captures total non-jump variance; corrected in Step 4 after g_mult estimation
    # ----------------------------
    non_jump_inc = increments[~jump_mask]
    if non_jump_inc.size > 0:
        # For SDE dX = sqrt(2D) dW + jumps, E[dX^2] = 2 D dt, so D ≈ sum(dX^2) / (2 T_eff)
        T_eff = dt * len(non_jump_inc)  # effective time without jumps
        g_add = float(np.sum(non_jump_inc ** 2) / (2.0 * T_eff))
    else:
        g_add = 0.0

    # ----------------------------
    # Step 3: g_mult via local mean-squared signal vs variance regression (robust)
    # 
    # Use SECOND-ORDER differences to reduce signal variation confound:
    #   Δ²y[i] = y[i+2] - 2*y[i+1] + y[i]
    # For smooth signals, Δ²clean ≈ clean''·dt² (much smaller than Δclean ≈ clean'·dt)
    # For iid noise: Var(Δ²n) = 6·σ² (coefficients: 1² + 2² + 1² = 6)
    #
    # Model: Var(Δ²y) ≈ 6·σ_add² + 6·σ_mult²·E[x²] + Var(Δ²clean)
    # Regress: seg_var_2nd ≈ a + b * seg_mean_sq, where b ≈ 6*σ_mult²
    # ----------------------------
    # choose window size for local mean/var if not provided
    if window_size is None:
        # use smaller windows than before to reduce signal variation inside windows
        window = max(16, len(y) // 128)
    else:
        window = max(8, int(window_size))

    seg_mean_sqs = []  # E[y²] in each segment (proxy for E[clean²])
    seg_vars = []

    # Compute second-order differences: Δ²y[i] = y[i+2] - 2*y[i+1] + y[i]
    if len(y) >= 3:
        second_diff = y[2:] - 2.0 * y[1:-1] + y[:-2]
    else:
        second_diff = np.array([], dtype=np.float64)

    # Build jump mask for second differences: a second diff is contaminated if
    # any of its constituent first differences contain a jump
    # second_diff[i] uses increments[i] and increments[i+1]
    if len(jump_mask) >= 2:
        jump_mask_2nd = jump_mask[:-1] | jump_mask[1:]
    else:
        jump_mask_2nd = np.array([], dtype=bool)

    for i in range(0, len(y), window):
        seg = y[i : i + window]
        if seg.size < 5:  # need at least 5 points for meaningful second diff stats
            continue
        
        # second differences for this segment
        seg_2nd = second_diff[max(0, i) : i + window - 2] if i + window - 2 <= len(second_diff) else np.array([])
        if seg_2nd.size < 3:
            continue
            
        # corresponding jump mask slice
        jm_slice_2nd = jump_mask_2nd[max(0, i) : i + len(seg_2nd)] if i + len(seg_2nd) <= len(jump_mask_2nd) else np.array([], dtype=bool)
        
        # handle size mismatch
        min_len = min(len(seg_2nd), len(jm_slice_2nd))
        if min_len < 3:
            continue
        seg_2nd = seg_2nd[:min_len]
        jm_slice_2nd = jm_slice_2nd[:min_len]
        
        # ignore windows with too many jumps
        if jm_slice_2nd.sum() > max(0, 0.25 * len(jm_slice_2nd)):
            continue
        seg_2nd_clean = seg_2nd[~jm_slice_2nd]
        if seg_2nd_clean.size < 3:
            continue
            
        seg_mean_sq = float(np.mean(seg ** 2))  # E[y²] as proxy for E[clean²]
        seg_var = float(np.var(seg_2nd_clean))  # variance of second differences
        seg_mean_sqs.append(seg_mean_sq)
        seg_vars.append(seg_var)

    seg_mean_sqs = np.asarray(seg_mean_sqs, dtype=np.float64)
    seg_vars = np.asarray(seg_vars, dtype=np.float64)

    g_mult = 0.0
    if seg_mean_sqs.size >= min_windows_for_regression:
        # ----------------------------
        # Iterative regression with errors-in-variables correction
        # Problem: We regress against E[y²], but model uses E[clean²]
        # E[y²] = E[clean²]·(1 + σ_mult²) + σ_add²
        # Solution: Iterate to refine E[clean²] estimate
        # ----------------------------
        
        # Initial regression using E[y²] directly
        X = np.vstack([np.ones_like(seg_mean_sqs), seg_mean_sqs]).T
        coef, _, _, _ = np.linalg.lstsq(X, seg_vars, rcond=None)
        a_fit, b_fit = float(coef[0]), float(coef[1])
        
        # Robust refit: remove outliers
        preds = X @ coef
        residuals = seg_vars - preds
        abs_res = np.abs(residuals)
        thr = np.percentile(abs_res, 90.0)
        keep_mask = abs_res <= thr
        
        if keep_mask.sum() >= min_windows_for_regression:
            coef, _, _, _ = np.linalg.lstsq(X[keep_mask], seg_vars[keep_mask], rcond=None)
            a_fit, b_fit = float(coef[0]), float(coef[1])
        
        # Initial g_mult estimate (slope is 6·σ_mult² for second differences)
        b_fit = max(b_fit, 0.0)
        g_mult_est = float(np.sqrt(b_fit / 6.0))
        
        # Iterative EIV correction (2-3 iterations usually sufficient)
        g_add_est = max(a_fit / 6.0, 0.0)  # rough g_add from intercept
        
        for _ in range(3):
            # Correct seg_mean_sqs: E[clean²] ≈ (E[y²] - σ_add²) / (1 + σ_mult²)
            denom = 1.0 + g_mult_est ** 2
            seg_mean_sqs_corrected = (seg_mean_sqs - g_add_est) / max(denom, 1e-12)
            seg_mean_sqs_corrected = np.maximum(seg_mean_sqs_corrected, 1e-12)  # keep positive
            
            # Re-regress with corrected regressors
            X_corr = np.vstack([np.ones_like(seg_mean_sqs_corrected), seg_mean_sqs_corrected]).T
            if keep_mask.sum() >= min_windows_for_regression:
                coef, _, _, _ = np.linalg.lstsq(X_corr[keep_mask], seg_vars[keep_mask], rcond=None)
            else:
                coef, _, _, _ = np.linalg.lstsq(X_corr, seg_vars, rcond=None)
            a_fit, b_fit = float(coef[0]), float(coef[1])
            
            b_fit = max(b_fit, 0.0)
            g_mult_new = float(np.sqrt(b_fit / 6.0))
            g_add_est = max(a_fit / 6.0, 0.0)
            
            # Check convergence
            if abs(g_mult_new - g_mult_est) < 1e-9:
                break
            g_mult_est = g_mult_new
        
        g_mult = g_mult_est
    else:
        # not enough windows to regress, fall back to zero
        g_mult = 0.0

    # ----------------------------
    # Step 4: Correct g_add by removing multiplicative noise contribution
    # The raw quadratic variation estimate includes: σ_add² + σ_mult²·E[y²]
    # So: g_add_corrected = g_add_raw - g_mult² · E[y²]
    # ----------------------------
    if g_mult > 0.0:
        mean_y_sq = float(np.mean(y ** 2))
        g_add_corrected = g_add - (g_mult ** 2) * mean_y_sq
        g_add = max(g_add_corrected, 0.0)  # clamp to non-negative

    return {
        "g_add": float(g_add),
        "g_mult": float(g_mult),
        "lambda_jump": float(lambda_jump),
        "sigma_jump": float(sigma_jump),
        "n_jumps": int(n_jumps),
        "total_time": float(T),
    }