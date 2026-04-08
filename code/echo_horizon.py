#!/usr/bin/env python3
"""
echo_horizon.py — Contour plot of echo detectability in (M, D_L) space.

For each (M_tot, D_L), compute ρ_comb for a 200-pulsar SKA array
and plot contours of ρ_comb = 5, 10, 30.

Includes the full antenna pattern functions F+, Fx (Eq. 4 of the paper)
for face-on orientation (iota=0). The timing residual amplitude is
r_P = h0/(2πf) * sqrt( (F+ (1+ci²)/2)² + (Fc ci)² )
where h0 = 4 (G Mc)^{5/3} (π f)^{2/3} / (c^4 D_L).

Also shows the μAres ρ_E = 8 threshold for comparison.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase_matching import (
    Source, Pulsar, generate_ska_array, geometric_delay,
    f_pulsar, timing_residual, t_merge, h0,
    G, c, Msun, Mpc, yr, pc
)

# ============================================================
# Grid
# ============================================================
log_M = np.linspace(7.5, 10.5, 80)   # log10(M_tot/Msun)
log_DL = np.linspace(0.5, 3.5, 80)   # log10(D_L/Mpc)
M_grid, DL_grid = np.meshgrid(log_M, log_DL)

# ============================================================
# PTA setup — generate once
# ============================================================
rng = np.random.default_rng(42)
pulsars = generate_ska_array(200, 10, rng=rng)
N_obs = 520  # biweekly × 20 yr
sigma_TOA = 100e-9  # 100 ns

# Source direction (fixed)
theta_s, phi_s = np.pi / 3, 1.0

# Inclination: face-on (iota=0, ci=1)
# For face-on (ci=1): r_P = h0/(2πf) * sqrt(F+^2 + Fx^2)
# where h0 = 4 (G Mc)^{5/3} (π f)^{2/3} / (c^4 D_L).
iota = 0.0
ci = np.cos(iota)  # = 1 for face-on
psi = 0.0  # arbitrary for face-on since F+^2+Fx^2 is psi-independent

# Source direction unit vector
st, ct = np.sin(theta_s), np.cos(theta_s)
sp, cp = np.sin(phi_s), np.cos(phi_s)
Omega_hat = np.array([st * cp, st * sp, ct])

# GW frame principal axes
m_hat = np.array([np.sin(phi_s), -np.cos(phi_s), 0.0])
n_hat = np.array([
    -ct * cp,
    -ct * sp,
    st,
])

# Polarization tensors (psi=0)
c2p = np.cos(2 * psi)
s2p = np.sin(2 * psi)
mm = np.outer(m_hat, m_hat)
nn = np.outer(n_hat, n_hat)
mn = np.outer(m_hat, n_hat) + np.outer(n_hat, m_hat)
e_plus = c2p * (mm - nn) + s2p * mn
e_cross = -s2p * (mm - nn) + c2p * mn

# Precompute per-pulsar quantities (independent of source mass/distance)
psr_phats = np.array([p.p_hat for p in pulsars])
psr_Lps = np.array([p.L_p for p in pulsars])

# Geometric delays
taus = np.array([
    (Lp / c) * (1.0 + np.dot(Omega_hat, phat))
    for Lp, phat in zip(psr_Lps, psr_phats)
])

# Antenna pattern functions F+, Fx for each pulsar
Fp_arr = np.zeros(len(pulsars))
Fc_arr = np.zeros(len(pulsars))
for k, phat in enumerate(psr_phats):
    denom = 2 * (1 + np.dot(Omega_hat, phat))
    if abs(denom) < 1e-15:
        Fp_arr[k] = 0.0
        Fc_arr[k] = 0.0
    else:
        Fp_arr[k] = np.einsum("i,ij,j", phat, e_plus, phat) / denom
        Fc_arr[k] = np.einsum("i,ij,j", phat, e_cross, phat) / denom

# ============================================================
# Compute ρ_comb over the grid
# ============================================================
rho_comb_grid = np.zeros_like(M_grid)
rho_E_grid = np.zeros_like(M_grid)

for i in range(len(log_DL)):
    for j in range(len(log_M)):
        M_tot = 10**log_M[j] * Msun
        DL = 10**log_DL[i] * Mpc
        eta = 0.25
        Mc = M_tot * eta**(3./5)
        GMc = G * Mc / c**3

        # f_E: fix v/c = 0.2 → f_E = 0.008 c^3/(π G M)
        f_E = 0.008 * c**3 / (np.pi * G * M_tot)

        # μAres SNR (rough: from flat noise model)
        Mc_ref = (1e9 * Msun) * 0.25**(3./5)
        rho_E = 4e5 * (Mc / Mc_ref)**(5./3) * (100 * Mpc / DL)
        rho_E_grid[i, j] = rho_E

        # Echo ρ_comb with antenna pattern
        rho_sq_sum = 0.0
        for k in range(len(pulsars)):
            tau = taus[k]
            tau_yr_val = tau / yr
            if tau_yr_val < 1:
                continue

            fP = f_pulsar(tau, f_E, GMc)
            if np.isnan(fP) or fP <= 0 or fP > 1e-3:
                continue

            # Correct strain amplitude h_0 = 4 (G Mc)^{5/3} (π f)^{2/3} / (c^4 D_L)
            h0_val = h0(fP, Mc, DL)

            # Timing residual amplitude:
            # r_P = h_0/(2πf) * sqrt( (F+ (1+ci²)/2)² + (Fc ci)² )
            Fp = Fp_arr[k]
            Fc = Fc_arr[k]
            rP = h0_val / (2 * np.pi * fP) * np.sqrt(
                (Fp * (1 + ci**2) / 2)**2 + (Fc * ci)**2
            )

            # SNR (Eq. 5): ρ_i = r_P sqrt(N_obs/2) / σ_TOA
            rho_i = rP * np.sqrt(N_obs / 2) / sigma_TOA

            rho_sq_sum += rho_i**2

        rho_comb_grid[i, j] = np.sqrt(rho_sq_sum)

# ============================================================
# Print diagnostics
# ============================================================
print(f"Inclination: iota = {np.degrees(iota):.0f} deg (face-on)")
print(f"Source direction: theta_s = {np.degrees(theta_s):.0f} deg, phi_s = {np.degrees(phi_s):.0f} deg")
print(f"Array: 200 pulsars, sigma_TOA = 100 ns, T = 20 yr, biweekly")

# ρ_comb at the optimistic source position
j_opt = np.argmin(abs(log_M - 9.0))
i_opt = np.argmin(abs(log_DL - np.log10(100)))
print(f"\nOptimistic source (10^9 Msun, 100 Mpc):")
print(f"  ρ_comb = {rho_comb_grid[i_opt, j_opt]:.1f}")

print('\nEcho horizon (ρ_comb = 5):')
for lm in [8, 8.5, 9, 9.5, 10]:
    j = np.argmin(abs(log_M - lm))
    col = rho_comb_grid[:, j]
    above = np.where(col >= 5)[0]
    if len(above) > 0:
        max_idx = above[-1]
        horizon = 10**log_DL[max_idx]
        rho_at_100 = rho_comb_grid[i_opt, j]
        print(f'  M = 10^{lm:.1f} Msun: D_L,max ~ {horizon:.0f} Mpc (ρ at 100 Mpc: {rho_at_100:.1f})')
    else:
        print(f'  M = 10^{lm:.1f} Msun: below threshold everywhere')

# ============================================================
# Plot — PRX style matching Figure 3
# ============================================================
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

# Convert grid coordinates to physical units for plotting
M_phys = 10**log_M       # M_tot in Msun
DL_phys = 10**log_DL     # D_L in Mpc

fig, ax = plt.subplots(1, 1, figsize=(3.375, 2.8))
ax.set_xscale('log')
ax.set_yscale('log')

# Fill: detectable region (ρ_comb > 5)
ax.contourf(M_phys, DL_phys, rho_comb_grid,
            levels=[5, 1e6], colors=['steelblue'], alpha=0.12)

# Echo ρ_comb contours
levels_echo = [5, 10, 20]
cs = ax.contour(M_phys, DL_phys, rho_comb_grid,
                levels=levels_echo, colors='steelblue',
                linewidths=[1.2, 0.9, 0.7])
ax.clabel(cs, fmt=r'$\rho_{\rm comb}=%g$', fontsize=5.5, inline=True,
          inline_spacing=4, manual=[(5e8, 20), (2.5e8, 20), (1.2e8, 20)])

# μAres ρ_E = 8 contour (for comparison)
cs2 = ax.contour(M_phys, DL_phys, rho_E_grid,
                 levels=[8], colors='C3', linewidths=[1.0], linestyles=['--'])
ax.clabel(cs2, fmt=r'$\rho_E^{\mu{\rm Ares}}=8$', fontsize=6, inline=True)

# Fiducial sources
ax.plot(5e8, 200, 'k*', markersize=8, zorder=5)
ax.plot(1e9, 100, 'k*', markersize=8, zorder=5)

# Annotations
ax.annotate('Typical', (5e8, 200),
            textcoords="offset points", xytext=(-40, 3), fontsize=7)
ax.annotate('Optimistic', (1e9, 100),
            textcoords="offset points", xytext=(8, -10), fontsize=7)

# MASSIVE survey line
ax.axhline(108, color='gray', ls=':', lw=0.6, alpha=0.6)
ax.text(4e7, 125, 'MASSIVE (108 Mpc)', fontsize=6,
        color='gray', ha='left')

ax.set_xlabel(r'$M_{\rm tot}\;[M_\odot]$')
ax.set_ylabel(r'$D_L\;[{\rm Mpc}]$')
ax.set_xlim(3e7, 3e10)
ax.set_ylim(3, 3000)

plt.tight_layout()
plt.savefig('fig_echo_horizon.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_echo_horizon.png', dpi=300, bbox_inches='tight')
print('\nSaved fig_echo_horizon.pdf and .png')
