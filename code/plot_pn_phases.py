"""
plot_pn_phases.py

Cumulative GW cycles by post-Newtonian order for the optimistic binary,
plotted versus GW frequency.  Uses the TaylorF2 perturbative decomposition
(Mingarelli et al. 2012 convention) via smbhb_evolution.py.

Phase accumulates from f_P (pulsar epoch) to f_E (Earth epoch).
"""

import numpy as np
import matplotlib.pyplot as plt
from smbhb_evolution import SMBHBEvolution
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

outdir = Path(__file__).resolve().parent

# ============================================================
# Binary parameters — optimistic source
# ============================================================
m1 = 5e8   # M_sun (M_tot = 10^9)
m2 = 5e8
chi = 0.7
f_E = 1e-6  # 1 μHz
D_L = 100   # Mpc

# J1713 baseline: maximum look-back time tau = L/c, L = 1176 pc
tau_yr = 3836.0

# ============================================================
# Build binary and run TaylorF2 decomposition
# ============================================================
binary = SMBHBEvolution(
    m1=m1, m2=m2,
    chi1=chi, chi2=chi,
    kappa1=0.0, kappa2=0.0,
    f_gw_earth=f_E,
    D_L=D_L,
)

result = binary.pn_decomposition(t_span_yr=tau_yr)
cyc = result["cycles"]

# Print analytic TaylorF2 cycle decomposition
print(f"f_P = {cyc['f_P_nHz']:.2f} nHz,  f_E = {cyc['f_E_nHz']:.0f} nHz")
print(f"Baseline: tau = {tau_yr:.0f} yr")
print()
print("TaylorF2 cycle decomposition (Mingarelli 2012 convention):")
print(f"  Newtonian:  {cyc['Newtonian']:.1f}")
print(f"  1pN:        {cyc['1pN']:+.1f}")
print(f"  1.5pN:      {cyc['1.5pN']:+.1f}")
print(f"  SO:         {cyc['SO']:+.1f}")
print(f"  2pN:        {cyc['2pN']:+.1f}")
print(f"  Thomas:     {cyc['Thomas']:+.2f}")
print(f"  Total:      {cyc['Total']:.1f}")

# ============================================================
# Analytic TaylorF2 curves in frequency domain
# ============================================================
# Compute the remaining cycles from frequency f to f_E at each pN order
# using the TaylorF2 phase formula directly.  This gives clean curves
# from f_P all the way to f_E with no numerical artefacts.

eta = binary.eta
M_s = binary.M_s
v_P = cyc["v_P"]
v_E = cyc["v_E"]

# TaylorF2 coefficients (same as in smbhb_evolution.py pn_decomposition)
phi_2 = 3715.0 / 1008 + 55.0 * eta / 12
phi_3_mass = -10.0 * np.pi
phi_3_SO = (10.0 / 3) * binary.beta_so
phi_3 = phi_3_mass + phi_3_SO
phi_4 = (15293365.0 / 508032 + 27145.0 * eta / 504
         + 3085.0 * eta ** 2 / 72)
phi_4_SS = -(10.0 / eta) * binary.sigma_ss if eta > 0 else 0.0
phi_4_full = phi_4 + phi_4_SS

psi_2 = (8 * phi_2 - 5 * (743.0/252 + 11.0*eta/3)) / 3.0
psi_3_mass = (8 * phi_3_mass - 5 * (-(32.0/5)*np.pi)) / 3.0
psi_4 = phi_4  # standard TaylorF2 2pN coeff

phase_prefac = 1.0 / (32.0 * np.pi * eta)  # GW cycles
fd_pf = 3.0 / (256.0 * np.pi * eta)

N_pts = 2000
f_grid = np.logspace(np.log10(f_E * 0.01), np.log10(f_E), N_pts)
v_grid = (np.pi * M_s * f_grid) ** (1.0 / 3)

# Remaining cycles from f to f_E (decreasing as f -> f_E)
# Newtonian: (1/(32 pi eta)) * (v^{-5} - v_E^{-5})
N_Newt_arr = phase_prefac * (v_grid**(-5) - v_E**(-5))

# Individual pN corrections (remaining from f to f_E)
N_1pN_arr = fd_pf * psi_2 * (v_grid**(-3) - v_E**(-3))
N_15pN_arr = fd_pf * psi_3_mass * (v_grid**(-2) - v_E**(-2))
# SO contribution: use the full phase difference with/without SO
# For plotting, approximate as the phi_3_SO term scaled like the tail
N_SO_arr = fd_pf * ((8*phi_3_SO - 5*(binary.beta_so*48.0/5.0))/3.0) * (v_grid**(-2) - v_E**(-2))
N_15pN_total_arr = N_15pN_arr + N_SO_arr
N_2pN_arr = fd_pf * psi_4 * (v_grid**(-1) - v_E**(-1))

f_nHz = f_grid * 1e9

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(4.5, 3.5))

curves = [
    ("Newtonian", N_Newt_arr,        "C0", "-",  2.0),
    ("1pN",       N_1pN_arr,         "C1", "-",  1.6),
    ("1.5pN",     N_15pN_total_arr,  "C2", "-",  1.6),
    ("2pN",       N_2pN_arr,         "C3", "-",  1.6),
]

# Mask below f_P — curves only shown from f_P to f_E
mask = f_nHz >= cyc["f_P_nHz"]
for name, data, color, ls, lw in curves:
    ax.plot(f_nHz[mask], np.abs(data[mask]), color=color, ls=ls, lw=lw, label=name)

ax.set_yscale("log")
ax.set_ylim(5e-3, 2e4)
ax.set_xlabel(r"$f_{\rm GW}$ [nHz]")
ax.set_ylabel("Accumulated GW cycles")

# f_P and f_E markers
f_P_nHz = cyc["f_P_nHz"]  # ~43 nHz for tau = L/c
f_E_nHz = cyc["f_E_nHz"]
ax.axvline(f_P_nHz, color="0.6", ls=":", lw=0.7, zorder=0)
ax.axvline(f_E_nHz, color="0.6", ls=":", lw=0.7, zorder=0)
ax.text(f_P_nHz * 1.15, 1e4, r'$f_P$', fontsize=9, color='0.4')
ax.text(f_E_nHz * 0.85, 1e4, r'$f_E$', fontsize=9, color='0.4', ha='right')

ax.legend(fontsize=8, loc="lower left", framealpha=0.95, handlelength=1.5,
          borderpad=0.4, labelspacing=0.3,
          bbox_to_anchor=(0.15, 0.0))

fig.subplots_adjust(left=0.16, right=0.95, bottom=0.15, top=0.95)
fig.savefig(outdir / "fig_pn_phases.pdf", bbox_inches="tight")
fig.savefig(outdir / "fig_pn_phases.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {outdir}/fig_pn_phases.pdf")
