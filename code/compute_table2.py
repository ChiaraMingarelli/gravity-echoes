#!/usr/bin/env python3
"""
compute_table2.py — Canonical computation of Table III (tab:scenarios).

Echo parameters and TaylorF2 cycle counts for three fiducial mu-Ares-band
SMBHB scenarios.  All numbers in the paper's Table III must come from this
script.  (Filename retained for backwards compatibility; paper numbering
changed when the tables were reordered.)

Conventions
-----------
  h_0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L)        [Eq. (2)]
  r_P = h_0 / (2 pi f_P)                                  [intrinsic, face-on]
  Cycles: TaylorT2 total + TaylorF2 individual corrections [Sec. IV]
  chi = 0.98 aligned spins for all scenarios
"""

import numpy as np
from smbhb_evolution import SMBHBEvolution
from compute_delta_f import compute_fP_at_each_order
from anchor_pulsars import J0437_DISTANCE_PC

# ---------- constants ----------
YR = 365.25 * 86400
PC = 3.08567758e16
MPC = 1e6 * PC
G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN = 1.98892e30

# ---------- scenarios ----------
SCENARIOS = [
    {"name": "Conservative", "m1": 5e7,  "m2": 5e7,   "f_E": 10e-6, "D_L": 2000},
    {"name": "Typical",      "m1": 3e8,  "m2": 3e8,    "f_E": 1e-6,  "D_L": 200},
    {"name": "Optimistic",   "m1": 5e8,  "m2": 5e8,    "f_E": 1e-6,  "D_L": 100},
]

CHI = 0.98  # aligned spins (near-extremal Kerr; migrated 2026-04-20 from 0.7)

# ---------- pulsars ----------
# tau = L_p / c corresponds to (1 + Omega.p) = 1, i.e. theta ~ 90 deg.
TAU_1KPC_YR = 1000 * PC / C_SI / YR                    # ~3262 yr
TAU_J0437_YR = J0437_DISTANCE_PC * PC / C_SI / YR      # ~511.94 yr (Reardon 2024)
PULSARS = [
    ("1 kpc",  TAU_1KPC_YR),
    ("J0437",  TAU_J0437_YR),
]


def h0_strain(f, Mc_kg, D_L_m):
    """GW strain amplitude h_0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L)."""
    return 4.0 * (G_SI * Mc_kg) ** (5.0 / 3) * (np.pi * f) ** (2.0 / 3) / (C_SI ** 4 * D_L_m)


def main():
    hdr = (
        f"{'Scenario':<14s} {'Pulsar':<8s} {'M_tot':>8s} {'f_E':>10s} "
        f"{'tau[yr]':>8s} {'f_P[nHz]':>9s} {'r_P[ns]':>8s} "
        f"{'Newt':>8s} {'1pN':>8s} {'1.5pN_t':>8s} {'SO/beta':>8s} "
        f"{'2pN':>6s} {'Total':>8s} | "
        f"{'df_1pN':>8s} {'df_tail':>8s} {'df_SO':>8s}"
    )
    sep = "-" * len(hdr)

    print("=" * len(hdr))
    print("Table III — Echo parameters (chi = 0.98 aligned)")
    print("h_0 convention: prefactor 4  [Eq. (2)]")
    print("1.5pN cycle column is tail-only (4 pi); SO reported separately as SO/beta_SO.")
    print("delta-f columns split 1.5pN into tail and SO. delta-f_2pN < 0.03 nHz everywhere; omitted.")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for sc in SCENARIOS:
        m1, m2 = sc["m1"], sc["m2"]
        f_E = sc["f_E"]
        D_L_Mpc = sc["D_L"]
        D_L_m = D_L_Mpc * MPC
        M_tot = m1 + m2
        eta = m1 * m2 / M_tot ** 2
        Mc_kg = M_tot * M_SUN * eta ** (3.0 / 5)

        for psr_name, tau_yr in PULSARS:
            binary = SMBHBEvolution(
                m1=m1, m2=m2,
                chi1=CHI, chi2=CHI,
                kappa1=0.0, kappa2=0.0,
                f_gw_earth=f_E,
                D_L=D_L_Mpc,
            )
            result = binary.pn_decomposition(t_span_yr=tau_yr)
            cyc = result["cycles"]

            f_P = cyc["f_P_nHz"] * 1e-9  # Hz
            r_P = h0_strain(f_P, Mc_kg, D_L_m) / (2 * np.pi * f_P)  # intrinsic, face-on

            # 1.5pN tail-only cycle count and SO/beta_SO normalization
            N_tail = cyc["1.5pN"]
            N_SO_over_beta = cyc["SO"] / binary.beta_so

            # delta-f decomposition (matches Eq. fdot in restructured-echo.tex)
            df = compute_fP_at_each_order(m1, m2, CHI, f_E, D_L_Mpc, tau_yr)

            M_str = f"{M_tot / 1e6:.0e}"
            f_str = f"{f_E * 1e6:.0f} uHz"

            print(
                f"{sc['name']:<14s} {psr_name:<8s} {M_str:>8s} {f_str:>10s} "
                f"{tau_yr:>8.0f} {cyc['f_P_nHz']:>9.1f} "
                f"{r_P * 1e9:>8.3f} "
                f"{cyc['Newtonian']:>8.0f} {cyc['1pN']:>+8.0f} "
                f"{N_tail:>+8.0f} {N_SO_over_beta:>+8.2f} "
                f"{cyc['2pN']:>+6.0f} {cyc['Total']:>8.0f} | "
                f"{df['df_1pN_nHz']:>+8.2f} "
                f"{df['df_15pN_tail_nHz']:>+8.2f} "
                f"{df['df_15pN_SO_nHz']:>+8.2f}"
            )

    # ---------- validation ----------
    print()
    print(sep)
    print("VALIDATION: Mingarelli et al. 2012 Table I")
    print("m1 = m2 = 10^9 Msun, f_E = 100 nHz, 1 kpc, non-spinning")
    print(sep)
    binary_val = SMBHBEvolution(
        m1=1e9, m2=1e9, chi1=0.0, chi2=0.0,
        f_gw_earth=100e-9, D_L=100.0,
    )
    vc = binary_val.pn_decomposition(t_span_yr=TAU_1KPC_YR)["cycles"]
    print(f"  Newtonian: {vc['Newtonian']:>8.1f}  (paper: 4267.8)")
    print(f"  1pN:       {vc['1pN']:>8.1f}  (paper:   77.3)")
    print(f"  1.5pN:     {vc['1.5pN']:>8.1f}  (paper:  -45.8)")
    print(f"  2pN:       {vc['2pN']:>8.1f}  (paper:    2.2)")
    print(f"  Total:     {vc['Total']:>8.1f}  (paper: 4305.1)")


if __name__ == "__main__":
    main()
