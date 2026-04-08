#!/usr/bin/env python3
"""
compute_table2.py — Canonical computation of Table II (tab:scenarios).

Echo parameters and TaylorF2 cycle counts for three fiducial mu-Ares-band
SMBHB scenarios.  All numbers in the paper's Table II must come from this
script.

Conventions
-----------
  h_0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L)        [Eq. (2)]
  r_P = h_0 / (2 pi f_P)                                  [intrinsic, face-on]
  Cycles: TaylorT2 total + TaylorF2 individual corrections [Sec. IV]
  chi = 0.7 aligned spins for all scenarios
"""

import numpy as np
from smbhb_evolution import SMBHBEvolution

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
    {"name": "Typical",      "m1": 2.5e8,"m2": 2.5e8,  "f_E": 1e-6,  "D_L": 200},
    {"name": "Optimistic",   "m1": 5e8,  "m2": 5e8,    "f_E": 1e-6,  "D_L": 100},
]

CHI = 0.7  # aligned spins

# ---------- pulsars ----------
# tau = L_p / c corresponds to (1 + Omega.p) = 1, i.e. theta ~ 90 deg.
TAU_1KPC_YR = 1000 * PC / C_SI / YR  # ~3262 yr
PULSARS = [
    ("1 kpc",  TAU_1KPC_YR),
    ("J0437",  533.0),
]


def h0_strain(f, Mc_kg, D_L_m):
    """GW strain amplitude h_0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L)."""
    return 4.0 * (G_SI * Mc_kg) ** (5.0 / 3) * (np.pi * f) ** (2.0 / 3) / (C_SI ** 4 * D_L_m)


def main():
    hdr = (
        f"{'Scenario':<14s} {'Pulsar':<8s} {'M_tot':>8s} {'f_E':>10s} "
        f"{'h_E':>10s} {'tau[yr]':>8s} {'f_P[nHz]':>9s} {'r_P[ns]':>8s} "
        f"{'N_Newt':>8s} {'N_1pN':>8s} {'N_1.5pN':>8s} {'N_2pN':>6s} {'N_total':>8s}"
    )
    sep = "-" * len(hdr)

    print("=" * len(hdr))
    print("Table II — Echo parameters (chi = 0.7 aligned)")
    print("h_0 convention: prefactor 4  [Eq. (2)]")
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

        # Earth-term strain
        h_E = h0_strain(f_E, Mc_kg, D_L_m)

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

            # 1.5pN column in the table combines tail + SO
            N_15pN_total = cyc["1.5pN"] + cyc["SO"]

            M_str = f"{M_tot / 1e6:.0e}"
            f_str = f"{f_E * 1e6:.0f} uHz" if f_E >= 1e-6 else f"{f_E * 1e6:.0f} uHz"

            print(
                f"{sc['name']:<14s} {psr_name:<8s} {M_str:>8s} {f_str:>10s} "
                f"{h_E:>10.1e} {tau_yr:>8.0f} {cyc['f_P_nHz']:>9.1f} "
                f"{r_P * 1e9:>8.1f} "
                f"{cyc['Newtonian']:>8.0f} {cyc['1pN']:>+8.0f} "
                f"{N_15pN_total:>+8.0f} {cyc['2pN']:>+6.0f} {cyc['Total']:>8.0f}"
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
