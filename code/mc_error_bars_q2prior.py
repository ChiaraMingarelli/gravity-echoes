#!/usr/bin/env python3
"""
Monte Carlo error propagation for N_PTA as a function of distance.

Target: equal-mass (q > 0.9) SMBHBs with M_tot = 10^9 Msun
        in the PTA band (f >= 1 nHz to merger), circular orbits.

Mass-ratio prior: p(q) ~ q^2 (LM24 preferred).

Uncertainties propagated:
  1. BHMF normalization: log10(phi_*) = -2.00 +/- 0.07  (LM24)
  2. BHMF low-mass slope: alpha = -1.27 +/- 0.02        (LM24)
  3. BHMF cutoff shape: beta = 0.45 +/- 0.02            (LM24)
  4. BHMF cutoff mass: log10(M_*) = 8.09 +/- 0.09       (LM24)
  5. Galaxy pairing rate: f_dot_0 = 0.04 +/- 0.01       (CC25 Eq. A1)

Author: Zheng, Bécsy, Mingarelli (2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.dpi': 300,
})

np.random.seed(42)
N_MC = 100_000

# ---- Physical constants ----
G_cgs  = 6.67430e-8
c_cgs  = 2.99792458e10
Msun   = 1.98892e33
yr_s   = 3.15576e7

# ---- Target mass ----
M_BH = 5e8
M_tot = 1e9
q_binary = 1.0
Mc = M_tot * (q_binary / (1 + q_binary)**2)**(3.0/5.0)

# ---- T_c (circular) ----
f_0 = 1e-9
Mc_cgs = Mc * Msun
T_c_Gyr = (5.0/256.0) * c_cgs**5 / ((G_cgs * Mc_cgs)**(5.0/3.0) * (np.pi * f_0)**(8.0/3.0)) / (yr_s * 1e9)

# ---- q^2 prior: f_q for q in [0.9, 1] ----
f_q = (1.0**3 - 0.9**3) / (1.0**3 - 0.01**3)

# ---- Monte Carlo: draw parameters once ----
log10_phi   = np.random.normal(-2.00, 0.07, N_MC)
alpha       = np.random.normal(-1.27, 0.02, N_MC)
beta        = np.random.normal(0.45, 0.02, N_MC)
log10_Mstar = np.random.normal(8.09, 0.09, N_MC)
f_dot_0     = np.clip(np.random.normal(0.04, 0.01, N_MC), 0.005, None)

phi_star = 10**log10_phi
M_star   = 10**log10_Mstar
x = M_BH / M_star
bhmf_dlnM = phi_star * x**(alpha + 1) * np.exp(-x**beta)
delta_lnM = 0.2 * np.log(10)

P_active = f_dot_0 * T_c_Gyr

# N_PTA per unit volume for each MC realization
n_PTA = bhmf_dlnM * delta_lnM * P_active * f_q  # Mpc^-3

# ---- Distance array ----
D_arr = np.linspace(20, 200, 300)
V_arr = (4.0/3.0) * np.pi * D_arr**3

# Compute percentiles at each distance
median_arr = np.zeros_like(D_arr)
p16_arr = np.zeros_like(D_arr)
p84_arr = np.zeros_like(D_arr)
p5_arr  = np.zeros_like(D_arr)
p95_arr = np.zeros_like(D_arr)

for i, Vi in enumerate(V_arr):
    N_samples = n_PTA * Vi
    median_arr[i] = np.median(N_samples)
    p16_arr[i] = np.percentile(N_samples, 16)
    p84_arr[i] = np.percentile(N_samples, 84)
    p5_arr[i]  = np.percentile(N_samples, 5)
    p95_arr[i] = np.percentile(N_samples, 95)

# ---- Find where median crosses N=1 ----
idx_cross = np.searchsorted(median_arr, 1.0)
D_cross = D_arr[idx_cross] if idx_cross < len(D_arr) else None

# ---- Also find where 90% lower bound crosses 1 ----
idx_cross_90 = np.searchsorted(p5_arr, 1.0)
D_cross_90 = D_arr[idx_cross_90] if idx_cross_90 < len(D_arr) else None

print(f"T_c = {T_c_Gyr*1e3:.1f} Myr, f_q = {f_q:.3f}")
print(f"Median crosses N=1 at D = {D_cross:.0f} Mpc")
print(f"90% lower bound crosses N=1 at D = {D_cross_90:.0f} Mpc")

# Specific distances
for D_check in [76, 85, 108]:
    V_check = (4.0/3.0) * np.pi * D_check**3
    N_check = n_PTA * V_check
    print(f"D = {D_check} Mpc: median {np.median(N_check):.2f}, "
          f"68% CI [{np.percentile(N_check,16):.2f}, {np.percentile(N_check,84):.2f}], "
          f"90% CI [{np.percentile(N_check,5):.2f}, {np.percentile(N_check,95):.2f}]")

# ==================================================================
# FIGURE
# ==================================================================
fig, ax = plt.subplots(figsize=(3.375, 2.8))  # PRX single-column width

ax.fill_between(D_arr, p5_arr, p95_arr, color='steelblue', alpha=0.2,
                label=r'90%')
ax.fill_between(D_arr, p16_arr, p84_arr, color='steelblue', alpha=0.4,
                label=r'68%')
ax.plot(D_arr, median_arr, color='steelblue', lw=1.5, label='Median')

# N = 1 line
ax.axhline(1, color='k', ls='--', lw=0.7, alpha=0.5)

# Mark MASSIVE volume
V_108 = (4.0/3.0) * np.pi * 108**3
N_108 = np.median(n_PTA * V_108)
ax.axvline(108, color='firebrick', ls=':', lw=0.8, alpha=0.7)
ax.plot(108, N_108, 'o', color='firebrick', ms=5, zorder=5)
ax.annotate(f'MASSIVE\n$N \\sim {N_108:.0f}$',
            xy=(108, N_108), xytext=(148, N_108 - 0.8),
            fontsize=7, color='firebrick', ha='center',
            arrowprops=dict(arrowstyle='->', color='firebrick', lw=0.6))

ax.set_xlabel(r'Distance $D$ [Mpc]')
ax.set_ylabel(r'$N_{\rm SMBHB}(<D)$  [$M_{\rm tot}=10^9\,M_\odot$]')
ax.legend(loc='upper left', frameon=False)
ax.set_xlim(20, 200)
ax.set_ylim(0, 9)
ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.minorticks_on()

plt.tight_layout()
plt.savefig('/sessions/busy-vibrant-lamport/mnt/gravity-echo/N_PTA_vs_distance.png',
            bbox_inches='tight', dpi=150)
plt.savefig('/sessions/busy-vibrant-lamport/mnt/gravity-echo/N_PTA_vs_distance.pdf',
            bbox_inches='tight')
print("\nFigure saved: N_PTA_vs_distance.png / .pdf")
plt.close()
