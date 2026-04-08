#!/usr/bin/env python3
"""
compute_table3.py — Canonical computation of Table III (tab:horizon).

Echo detectability for 10^9 Msun equal-mass, face-on binary at f_E = 1 uHz.
Reports rho_comb, N_det at 100 Mpc; horizon distances for three tiers.

Conventions
-----------
  h_0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L)    [Eq. (2), prefactor 4]
  r_P = h_0/(2 pi f) * sqrt[(F+ (1+ci^2)/2)^2
                            + (Fx ci)^2]               [Eq. (3)]
  rho_i = r_P sqrt(N_obs/2) / sigma_TOA               [Eq. (4)]
  Tier 1: rho_comb > 5
  Tier 2: N(rho_i >= 3) >= 5                           [Sec. III]
  Tier 3: N_anchor(rho_i >= 3) >= 3                    [Sec. III]

Array configurations
--------------------
  SKA:      200 pulsars, 20 yr, biweekly, sigma_TOA = 100 ns
  IPTA 2050: 131 pulsars, 50 yr, biweekly, sigma_TOA = 100 ns
  Combined: 331 pulsars (both populations)
"""

import numpy as np
from phase_matching import (
    Pulsar, generate_ska_array, f_pulsar, h0,
    G, c, Msun, Mpc, yr, pc
)

# ============================================================
# Source parameters: 10^9 Msun, equal mass, face-on
# ============================================================
M_tot = 1e9 * Msun
eta = 0.25
Mc = M_tot * eta ** (3.0 / 5)
GMc_c3 = G * Mc / c ** 3

# Earth-term frequency (from v/c = 0.2, consistent with echo_horizon.py)
f_E = 0.008 * c ** 3 / (np.pi * G * M_tot)

iota = 0.0
ci = np.cos(iota)  # = 1 for face-on
psi = 0.0

# Source direction (fixed, same as echo_horizon.py)
theta_s, phi_s = np.pi / 3, 1.0
st, ct = np.sin(theta_s), np.cos(theta_s)
sp, cp = np.sin(phi_s), np.cos(phi_s)
Omega_hat = np.array([st * cp, st * sp, ct])

# GW polarization frame
m_hat = np.array([np.sin(phi_s), -np.cos(phi_s), 0.0])
n_hat = np.array([-ct * cp, -ct * sp, st])
c2p, s2p = np.cos(2 * psi), np.sin(2 * psi)
mm = np.outer(m_hat, m_hat)
nn = np.outer(n_hat, n_hat)
mn = np.outer(m_hat, n_hat) + np.outer(n_hat, m_hat)
e_plus = c2p * (mm - nn) + s2p * mn
e_cross = -s2p * (mm - nn) + c2p * mn

# ============================================================
# PTA arrays (deterministic seeds for reproducibility)
# ============================================================
ska_pulsars = generate_ska_array(200, 10, rng=np.random.default_rng(42))
ska_N_obs = 520    # biweekly x 20 yr
ska_sigma = 100e-9

ipta_pulsars = generate_ska_array(131, 5, rng=np.random.default_rng(123))
ipta_N_obs = 1300  # biweekly x 50 yr
ipta_sigma = 100e-9

combined_pulsars = ska_pulsars + ipta_pulsars
combined_N_obs = np.array(
    [ska_N_obs] * len(ska_pulsars) + [ipta_N_obs] * len(ipta_pulsars),
    dtype=float,
)
combined_sigma = np.array(
    [ska_sigma] * len(ska_pulsars) + [ipta_sigma] * len(ipta_pulsars),
    dtype=float,
)

# ============================================================
# Antenna patterns
# ============================================================
def precompute_antenna(pulsars):
    """Compute F+, Fx, geometric delay for each pulsar."""
    n = len(pulsars)
    Fp = np.zeros(n)
    Fc = np.zeros(n)
    tau = np.zeros(n)
    for k, p in enumerate(pulsars):
        phat = p.p_hat
        dot = np.dot(Omega_hat, phat)
        tau[k] = (p.L_p / c) * (1 + dot)
        denom = 2 * (1 + dot)
        if abs(denom) < 1e-15:
            continue
        Fp[k] = np.einsum("i,ij,j", phat, e_plus, phat) / denom
        Fc[k] = np.einsum("i,ij,j", phat, e_cross, phat) / denom
    return Fp, Fc, tau


ska_Fp, ska_Fc, ska_tau = precompute_antenna(ska_pulsars)
ipta_Fp, ipta_Fc, ipta_tau = precompute_antenna(ipta_pulsars)

comb_Fp = np.concatenate([ska_Fp, ipta_Fp])
comb_Fc = np.concatenate([ska_Fc, ipta_Fc])
comb_tau = np.concatenate([ska_tau, ipta_tau])

ska_is_anchor = np.array([p.sigma_d_frac <= 0.015 for p in ska_pulsars])
ipta_is_anchor = np.array([p.sigma_d_frac <= 0.015 for p in ipta_pulsars])
comb_is_anchor = np.concatenate([ska_is_anchor, ipta_is_anchor])

# ============================================================
# SNR computation — Eqs. (2)-(4)
# ============================================================
RHO_TIER2 = 3.0  # paper threshold for individual echo recovery
RHO_TIER3 = 3.0  # paper threshold for anchor pulsars

def compute_rho_array(DL_m, Fp_arr, Fc_arr, tau_arr, N_obs_arr, sigma_arr):
    """Per-pulsar SNR at luminosity distance DL_m [metres]."""
    n = len(Fp_arr)
    if np.isscalar(N_obs_arr):
        N_obs_arr = np.full(n, N_obs_arr)
    if np.isscalar(sigma_arr):
        sigma_arr = np.full(n, sigma_arr)

    rho_i = np.zeros(n)
    for k in range(n):
        tau_k = tau_arr[k]
        if tau_k / yr < 1:
            continue
        fP = f_pulsar(tau_k, f_E, GMc_c3)
        if np.isnan(fP) or fP <= 0 or fP > 1e-3:
            continue

        # Eq. (2): h_0 with prefactor 4
        h0_val = h0(fP, Mc, DL_m)

        # Eq. (3): timing residual including antenna pattern
        rP = h0_val / (2 * np.pi * fP) * np.sqrt(
            (Fp_arr[k] * (1 + ci ** 2) / 2) ** 2
            + (Fc_arr[k] * ci) ** 2
        )

        # Eq. (4): matched-filter SNR
        rho_i[k] = rP * np.sqrt(N_obs_arr[k] / 2) / sigma_arr[k]

    return rho_i


# ============================================================
# Distance scan (1 Mpc steps)
# ============================================================
D_scan = np.arange(10, 1501, 1, dtype=float)


def find_horizons(Fp_arr, Fc_arr, tau_arr, N_obs, sigma, is_anchor, label):
    """Scan distances to find Tier 1/2/3 horizons."""
    rho_comb_arr = np.zeros(len(D_scan))
    Ndet_arr = np.zeros(len(D_scan), dtype=int)
    Nanc_arr = np.zeros(len(D_scan), dtype=int)

    for i, DL_Mpc in enumerate(D_scan):
        rho_i = compute_rho_array(DL_Mpc * Mpc, Fp_arr, Fc_arr, tau_arr, N_obs, sigma)
        rho_comb_arr[i] = np.sqrt(np.sum(rho_i ** 2))
        Ndet_arr[i] = np.sum(rho_i >= RHO_TIER2)
        Nanc_arr[i] = np.sum((rho_i >= RHO_TIER3) & is_anchor)

    # Horizons: farthest distance where threshold is met
    mask1 = rho_comb_arr > 5
    tier1 = int(D_scan[mask1][-1]) if mask1.any() else 0
    mask2 = Ndet_arr >= 5
    tier2 = int(D_scan[mask2][-1]) if mask2.any() else 0
    mask3 = Nanc_arr >= 3
    tier3 = int(D_scan[mask3][-1]) if mask3.any() else 0

    # Values at 100 Mpc
    idx100 = np.argmin(np.abs(D_scan - 100))
    rho100 = rho_comb_arr[idx100]
    Ndet100 = Ndet_arr[idx100]

    print(f"  {label}")
    print(f"    At 100 Mpc: rho_comb = {rho100:.1f},  N(rho_i >= {RHO_TIER2:.0f}) = {Ndet100}")
    print(f"    Tier 1 (rho_comb > 5):                     {tier1} Mpc")
    print(f"    Tier 2 (N(rho_i >= {RHO_TIER2:.0f}) >= 5):              {tier2} Mpc")
    print(f"    Tier 3 (N_anchor(rho_i >= {RHO_TIER3:.0f}) >= 3):       {tier3} Mpc")
    print()

    return rho100, Ndet100, tier1, tier2, tier3


def main():
    print("=" * 70)
    print("Table III — Echo detectability")
    print(f"  h_0 convention: prefactor 4  [Eq. (2)]")
    print(f"  Tier 2 threshold: rho_i >= {RHO_TIER2:.0f}")
    print(f"  Tier 3 threshold: rho_i >= {RHO_TIER3:.0f} (anchors)")
    print(f"  f_E = {f_E * 1e6:.3f} uHz")
    print(f"  M_tot = 10^9 Msun, eta = 0.25, face-on")
    print("=" * 70)
    print()

    results = {}
    results["SKA"] = find_horizons(
        ska_Fp, ska_Fc, ska_tau, ska_N_obs, ska_sigma, ska_is_anchor,
        "SKA (200 psr, 20 yr)",
    )
    results["IPTA"] = find_horizons(
        ipta_Fp, ipta_Fc, ipta_tau, ipta_N_obs, ipta_sigma, ipta_is_anchor,
        "IPTA 2050 (131 psr, 50 yr)",
    )
    results["Combined"] = find_horizons(
        comb_Fp, comb_Fc, comb_tau, combined_N_obs, combined_sigma, comb_is_anchor,
        "Combined (331 psr)",
    )

    # Summary table
    print("=" * 70)
    print("SUMMARY (for paper Table III)")
    print("=" * 70)
    fmt = "{:<30s} | {:>6s} | {:>5s} | {:>6s} | {:>6s} | {:>6s}"
    print(fmt.format("Array", "rho", "N_det", "T1", "T2", "T3"))
    print("-" * 70)
    for name, (rho, ndet, t1, t2, t3) in results.items():
        print(fmt.format(name, f"{rho:.1f}", f"{ndet}", f"{t1}", f"{t2}", f"{t3}"))

    # Cross-check against paper values
    print()
    print("Paper Table III values:")
    print("  SKA:      rho=16.9, N_det=0,  T1=337,  T2=60,  T3=51")
    print("  IPTA:     rho=20.4, N_det=2,  T1=408,  T2=93,  T3=76")
    print("  Combined: rho=26.5, N_det=2,  T1=529,  T2=93,  T3=76")


if __name__ == "__main__":
    main()
