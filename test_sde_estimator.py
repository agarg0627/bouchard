from sde_estimator import estimate_sde_parameters
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path("synthetic_mixed_noise")
files = sorted(data_dir.glob("*.npz"))

mse_g_add, mse_g_mult, mse_lambda, mse_sigma = [], [], [], []
mse_by_snr = {}
nmse_by_snr = {}  # estimator-based normalized reconstruction by noise power

# NEW: oracle reconstruction metrics
mse_by_snr_oracle = {}
nmse_by_snr_oracle = {}

rng = np.random.default_rng(42)

for f in files:
    d = np.load(f)
    y = d["noisy"].astype(np.float64)
    clean = d["clean"].astype(np.float64)
    snr_db = float(d["snr_total_db"])

    # -------------------------
    # Compute true quantities
    # -------------------------
    clean_power = float(np.mean(clean ** 2))
    noise_true = y - clean
    noise_power = float(np.mean(noise_true ** 2)) + 1e-12  # avoid divide-by-zero

    # Ground truth additive
    g_add_true = float(d["noise_power_add"])

    # Ground truth multiplicative (std of multiplicative factor m)
    if clean_power > 1e-12 and d["noise_power_mult"] > 1e-12:
        g_mult_true = float(np.sqrt(d["noise_power_mult"] / clean_power))
    else:
        g_mult_true = 0.0

    # Ground truth jump stats
    noise_jump = d["noise_jump"].astype(np.float64)
    n_actual_jumps = np.sum(np.abs(noise_jump) > 1e-10)
    lambda_true = float(n_actual_jumps / len(noise_jump))

    if n_actual_jumps > 1:
        sigma_jump_true = float(np.std(noise_jump[np.abs(noise_jump) > 1e-10]))
    else:
        sigma_jump_true = 0.0

    # -------------------------
    # Estimate SDE parameters
    # -------------------------
    est = estimate_sde_parameters(y, dt=1.0)

    mse_g_add.append((est["g_add"] - g_add_true) ** 2)
    mse_g_mult.append((est["g_mult"] - g_mult_true) ** 2)
    mse_lambda.append((est["lambda_jump"] - lambda_true) ** 2)
    mse_sigma.append((est["sigma_jump"] - sigma_jump_true) ** 2)

    # -------------------------
    # Forward simulation using estimated params
    # -------------------------
    n = len(clean)

    # Additive (est)
    g_add_est = max(float(est["g_add"]), 0.0)
    noise_add_sim = rng.normal(0, np.sqrt(g_add_est), n)

    # Multiplicative (est)
    g_mult_est = max(float(est["g_mult"]), 0.0)
    noise_mult_sim = clean * rng.normal(0, g_mult_est, n)

    # Jump (est)
    noise_jump_sim = np.zeros(n)
    lambda_est = max(float(est["lambda_jump"]), 0.0)
    sigma_est = max(float(est["sigma_jump"]), 0.0)

    if lambda_est > 0 and sigma_est > 0:
        n_jumps = rng.poisson(lambda_est * n)
        if n_jumps > 0:
            locs = rng.choice(n, min(n_jumps, n), replace=False)
            noise_jump_sim[locs] = rng.normal(0, sigma_est, len(locs))

    noisy_sim = clean + noise_add_sim + noise_mult_sim + noise_jump_sim

    # -------------------------
    # Forward simulation using oracle (ground-truth) params
    # -------------------------
    # Use the same RNG object (continues the stream) so draws are independent
    noise_add_oracle = rng.normal(0, np.sqrt(max(g_add_true, 0.0)), n)
    noise_mult_oracle = clean * rng.normal(0, max(g_mult_true, 0.0), n)

    noise_jump_oracle = np.zeros(n)
    lambda_o = max(float(lambda_true), 0.0)
    sigma_o = max(float(sigma_jump_true), 0.0)
    if lambda_o > 0 and sigma_o > 0:
        n_jumps_o = rng.poisson(lambda_o * n)
        if n_jumps_o > 0:
            locs_o = rng.choice(n, min(n_jumps_o, n), replace=False)
            noise_jump_oracle[locs_o] = rng.normal(0, sigma_o, len(locs_o))

    noisy_sim_oracle = clean + noise_add_oracle + noise_mult_oracle + noise_jump_oracle

    # -------------------------
    # Reconstruction error metrics
    # -------------------------
    recon_mse = np.mean((noisy_sim - y) ** 2)
    recon_nmse_noise = recon_mse / noise_power  # key metric

    recon_mse_oracle = np.mean((noisy_sim_oracle - y) ** 2)
    recon_nmse_noise_oracle = recon_mse_oracle / noise_power

    if snr_db not in mse_by_snr:
        mse_by_snr[snr_db] = []
        nmse_by_snr[snr_db] = []
        mse_by_snr_oracle[snr_db] = []
        nmse_by_snr_oracle[snr_db] = []

    mse_by_snr[snr_db].append(recon_mse)
    nmse_by_snr[snr_db].append(recon_nmse_noise)

    mse_by_snr_oracle[snr_db].append(recon_mse_oracle)
    nmse_by_snr_oracle[snr_db].append(recon_nmse_noise_oracle)

# -------------------------------------------------------------------
# Summary printout
# -------------------------------------------------------------------
print("Forward SDE parameter estimation summary")
print("==============================================")
print(f"g_add      : MSE = {np.mean(mse_g_add):.6f}")
print(f"g_mult     : MSE = {np.mean(mse_g_mult):.6f}")
print(f"lambda_jump: MSE = {np.mean(mse_lambda):.6f}")
print(f"sigma_jump : MSE = {np.mean(mse_sigma):.6f}")
print("==============================================")

print("Forward Reconstruction Error (Estimator -> noisy_sim):")
for snr in sorted(mse_by_snr.keys()):
    avg_mse = np.mean(mse_by_snr[snr])
    avg_nmse = np.mean(nmse_by_snr[snr])
    avg_mse_or = np.mean(mse_by_snr_oracle[snr])
    avg_nmse_or = np.mean(nmse_by_snr_oracle[snr])
    print(f"  {snr:+5.1f} dB:   EST MSE = {avg_mse:.6f}   EST NMSE_noise = {avg_nmse:.4f} | "
          f"ORACLE MSE = {avg_mse_or:.6f}   ORACLE NMSE_noise = {avg_nmse_or:.4f}")

overall_mse = np.mean(sum(mse_by_snr.values(), []))
overall_nmse = np.mean(sum(nmse_by_snr.values(), []))
overall_mse_or = np.mean(sum(mse_by_snr_oracle.values(), []))
overall_nmse_or = np.mean(sum(nmse_by_snr_oracle.values(), []))

print("==============================================")
print(f"Overall EST MSE         : {overall_mse:.6f}")
print(f"Overall EST NMSE_noise  : {overall_nmse:.4f}")
print(f"Overall ORACLE MSE      : {overall_mse_or:.6f}")
print(f"Overall ORACLE NMSE_noise: {overall_nmse_or:.4f}")
print("==============================================")

# -------------------------------------------------------------------
# Plot example (last simulated signals)
# -------------------------------------------------------------------

def plot_results(ax, title, noisy_x, noisy_sim_est, noisy_sim_oracle, color_est='green', color_oracle='blue'):
    ax.plot(noisy_x, color='gray', linewidth=1.5, label='Observed Noisy', alpha=0.7)
    ax.plot(noisy_sim_est, linewidth=2.0, linestyle='--', label='Simulated (EST)', alpha=0.9)
    ax.plot(noisy_sim_oracle, linewidth=2.0, linestyle=':', label='Simulated (ORACLE)', alpha=0.9)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True)

# Last loop's y, noisy_sim, noisy_sim_oracle are available
fig, ax = plt.subplots(1, 1, figsize=(24, 7))
plot_results(ax, "Example Forward Reconstruction — Observed vs Simulated (EST & ORACLE)", y, noisy_sim, noisy_sim_oracle)
plt.tight_layout()
plt.show()
