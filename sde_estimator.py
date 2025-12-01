"""
sde_estimator.py

Estimate forward SDE parameters from a 1D noisy signal:
- g_add      : additive diffusion coefficient (variance per sample, SDE diffusion D)
- g_mult     : multiplicative noise std (std of multiplicative factor m_scaled)
- lambda_jump: jump rate (jumps per unit time)
- sigma_jump : jump magnitude std (std of detected jump increments)
- n_jumps    : number of detected jumps

Literature estimator:
1. Mancini (2009) optimal threshold jump detection.
2. Quadratic variation on jump-free increments -> g_add (diffusion D).
3. Local mean vs variance regression on jump-free windows -> estimate multiplicative variance b,
   then g_mult = sqrt(max(b, 0)).
"""

import numpy as np


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
    
    # jump size std: sample std of detected jump increments
    # Mancini (2009) shows jump size estimation is consistent as dt -> 0
    sigma_jump = float(np.std(jump_sizes)) if n_jumps > 1 else 0.0

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
    # For multiplicative noise n = x * m with m ~ N(0, σ_m),
    # Var(Δn) = 2 * σ_m² * E[x²] in each window (factor of 2 from differencing iid noise)
    # So we regress: seg_var ≈ a + b * seg_mean_sq, where b ≈ 2*σ_m²
    # ----------------------------
    # choose window size for local mean/var if not provided
    if window_size is None:
        # use smaller windows than before to reduce signal variation inside windows
        window = max(16, len(y) // 128)
    else:
        window = max(8, int(window_size))

    seg_mean_sqs = []  # E[x²] in each segment
    seg_vars = []

    # we need a jump mask aligned with segments of y: increments have length len(y)-1
    # for a segment y[i:i+w], increments are length w-1 and correspond to jump_mask[i:i+w-1]
    for i in range(0, len(y), window):
        seg = y[i : i + window]
        if seg.size < 4:
            continue
        seg_inc = np.diff(seg)  # length seg.size-1
        # slice jump mask to match seg_inc
        jm_slice = jump_mask[i : i + len(seg_inc)]
        # ignore windows with too many jumps (they contaminate variance)
        if jm_slice.sum() > max(0, 0.25 * len(jm_slice)):
            continue
        seg_inc_clean = seg_inc[~jm_slice]
        if seg_inc_clean.size < 2:
            continue
        seg_mean_sq = float(np.mean(seg ** 2))  # E[x²], not E[x]²
        seg_var = float(np.var(seg_inc_clean))  # use increment variance in the window
        seg_mean_sqs.append(seg_mean_sq)
        seg_vars.append(seg_var)

    seg_mean_sqs = np.asarray(seg_mean_sqs, dtype=np.float64)
    seg_vars = np.asarray(seg_vars, dtype=np.float64)

    g_mult = 0.0
    if seg_mean_sqs.size >= min_windows_for_regression:
        # Fit the model: seg_var ≈ a + b * seg_mean_sq
        X = np.vstack([np.ones_like(seg_mean_sqs), seg_mean_sqs]).T

        # initial ordinary least squares
        coef, _, _, _ = np.linalg.lstsq(X, seg_vars, rcond=None)
        a_init, b_init = float(coef[0]), float(coef[1])

        # remove windows with large residuals and refit
        preds = X @ coef
        residuals = seg_vars - preds
        abs_res = np.abs(residuals)
        # compute 90th percentile threshold
        thr = np.percentile(abs_res, 90.0)
        keep_mask = abs_res <= thr
        if keep_mask.sum() >= min_windows_for_regression:
            coef2, _, _, _ = np.linalg.lstsq(X[keep_mask], seg_vars[keep_mask], rcond=None)
            a_fit, b_fit = float(coef2[0]), float(coef2[1])
        else:
            a_fit, b_fit = a_init, b_init

        # b_fit corresponds to 2*σ_m² (factor of 2 from increment variance of iid noise)
        # Var(Δn) = Var(n[i+1] - n[i]) = 2*Var(n) for iid n
        # So slope b ≈ 2*σ_mult², hence σ_mult = sqrt(b/2)
        b_fit = max(b_fit, 0.0)
        g_mult = float(np.sqrt(b_fit / 2.0))
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