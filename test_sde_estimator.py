from sde_estimator import estimate_sde_parameters
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.style as style

data_dir = Path("synthetic_mixed_noise")
files = sorted(data_dir.glob("*.npz"))

mse_g_add, mse_g_mult, mse_lambda, mse_sigma = [], [], [], []
mse_by_snr = {}
rng = np.random.default_rng(42)

for f in files:
    d = np.load(f)
    y = d["noisy"]
    clean = d["clean"]
    snr_db = float(d["snr_total_db"])
    est = estimate_sde_parameters(y, dt=1.0)

    # ground truth
    g_add_true = float(d["noise_power_add"])
    
    clean_power = float(np.mean(clean.astype(np.float64) ** 2))
    if clean_power > 1e-12 and d["noise_power_mult"] > 1e-12:
        g_mult_true = float(np.sqrt(d["noise_power_mult"] / clean_power))
    else:
        g_mult_true = 0.0

    noise_jump = d["noise_jump"]
    n_actual_jumps = np.sum(np.abs(noise_jump) > 1e-10)
    lambda_true = float(n_actual_jumps / len(noise_jump))

    if n_actual_jumps > 1:
        sigma_jump_true = float(np.std(noise_jump[np.abs(noise_jump) > 1e-10]))
    else:
        sigma_jump_true = 0.0

    mse_g_add.append((est["g_add"] - g_add_true) ** 2)
    mse_g_mult.append((est["g_mult"] - g_mult_true) ** 2)
    mse_lambda.append((est["lambda_jump"] - lambda_true) ** 2)
    mse_sigma.append((est["sigma_jump"] - sigma_jump_true) ** 2)

    # Forward simulation
    n = len(clean)
    noise_add_sim = rng.normal(0, np.sqrt(max(est["g_add"], 0)), n) if est["g_add"] > 0 else np.zeros(n)
    noise_mult_sim = clean * rng.normal(0, est["g_mult"], n) if est["g_mult"] > 0 else np.zeros(n)
    noise_jump_sim = np.zeros(n)
    if est["lambda_jump"] > 0 and est["sigma_jump"] > 0:
        n_jumps = rng.poisson(est["lambda_jump"] * n)
        if n_jumps > 0:
            locs = rng.choice(n, min(n_jumps, n), replace=False)
            noise_jump_sim[locs] = rng.normal(0, est["sigma_jump"], len(locs))
    noisy_sim = clean + noise_add_sim + noise_mult_sim + noise_jump_sim
    recon_mse = np.mean((noisy_sim - y) ** 2)
    
    if snr_db not in mse_by_snr:
        mse_by_snr[snr_db] = []
    mse_by_snr[snr_db].append(recon_mse)

print("Forward SDE parameter estimation summary")
print("==============================================")
print(f"g_add      : MSE = {np.mean(mse_g_add):.6f}")
print(f"g_mult     : MSE = {np.mean(mse_g_mult):.6f}")
print(f"lambda_jump: MSE = {np.mean(mse_lambda):.6f}")
print(f"sigma_jump : MSE = {np.mean(mse_sigma):.6f}")
print("==============================================")
print("Reconstruction MSE by SNR:")
for snr in sorted(mse_by_snr.keys()):
    print(f"  {snr:+5.1f} dB: {np.mean(mse_by_snr[snr]):.6f}")
print(f"  Overall:  {np.mean(sum(mse_by_snr.values(), [])):.6f}")
print("==============================================")

def plot_results(ax, title, noisy_x, noisy_sim, color):
    """Helper function to plot the results on a given axis."""
    # The noisy signal is a thin gray solid line.
    ax.plot(noisy_x, color='gray', linestyle='-', linewidth=1.5, label='Noisy Signal', alpha=0.7, zorder=1)
    # The simulated noisy signal is a thicker, dashed line plotted on top.
    ax.plot(noisy_sim, color=color, linewidth=3, linestyle='--', label='Simulated Noisy Signal', zorder=3)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel("Signal Dimension")
    ax.set_ylabel("Amplitude")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

fig, ax = plt.subplots(1, 1, figsize=(24, 7))
fig.suptitle('SDE Parameter Estimation in Python', fontsize=22, fontweight='bold')

print("Running SDE Parameter Estimation...")
plot_results(ax, 'Reconstruction of Noisy Signal', y, noisy_sim, 'green')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()