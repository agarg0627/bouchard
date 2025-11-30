from sde_estimator import estimate_sde_parameters
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path("synthetic_mixed_noise")
files = sorted(data_dir.glob("*.npz"))

mse_g_add, mse_g_mult, mse_lambda, mse_sigma = [], [], [], []
mse_by_snr = {}
nmse_by_snr = {}  # NEW: normalized reconstruction error by noise power

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

    # Ground truth multiplicative
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

    # Additive
    g_add_est = max(float(est["g_add"]), 0.0)
    noise_add_sim = rng.normal(0, np.sqrt(g_add_est), n)

    # Multiplicative
    g_mult_est = max(float(est["g_mult"]), 0.0)
    noise_mult_sim = clean * rng.normal(0, g_mult_est, n)

    # Jump
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
    # Reconstruction error metrics
    # -------------------------
    recon_mse = np.mean((noisy_sim - y) ** 2)
    recon_nmse_noise = recon_mse / noise_power  # key metric

    if snr_db not in mse_by_snr:
        mse_by_snr[snr_db] = []
        nmse_by_snr[snr_db] = []

    mse_by_snr[snr_db].append(recon_mse)
    nmse_by_snr[snr_db].append(recon_nmse_noise)

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

print("Forward Reconstruction Error (MSE and NMSE_noise):")
for snr in sorted(mse_by_snr.keys()):
    avg_mse = np.mean(mse_by_snr[snr])
    avg_nmse = np.mean(nmse_by_snr[snr])
    print(f"  {snr:+5.1f} dB:   MSE = {avg_mse:.6f}   NMSE_noise = {avg_nmse:.4f}")

overall_mse = np.mean(sum(mse_by_snr.values(), []))
overall_nmse = np.mean(sum(nmse_by_snr.values(), []))
print("==============================================")
print(f"Overall MSE         : {overall_mse:.6f}")
print(f"Overall NMSE_noise  : {overall_nmse:.4f}")
print("==============================================")

# -------------------------------------------------------------------
# Plot example
# -------------------------------------------------------------------

def plot_results(ax, title, noisy_x, noisy_sim, color):
    ax.plot(noisy_x, color='gray', linewidth=1.5, label='Noisy Signal', alpha=0.7)
    ax.plot(noisy_sim, color=color, linewidth=2.5, linestyle='--', label='Simulated Noisy')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True)

# Last simulated plot still in `noisy_sim`
fig, ax = plt.subplots(1, 1, figsize=(24, 7))
plot_results(ax, "Example Forward Reconstruction", y, noisy_sim, 'green')
plt.show()
