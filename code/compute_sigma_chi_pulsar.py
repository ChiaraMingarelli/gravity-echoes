#!/usr/bin/env python3
"""
compute_sigma_chi_pulsar.py -- single-pulsar bound on sigma_chi.

Canonical generator for the quoted sigma_chi values in the
spin-distance degeneracy appendix (Sec. app:spin_degeneracy).
Implements Eq.~(eq:sigma_chi_prior) of the paper:

    sigma_chi^(i) ~ chi f_P,i sigma_tau,i / N_SO,i ,
    sigma_tau,i  = (delta L_p / L_p) tau_i .

f_P,i and N_SO,i are computed by smbhb_evolution.SMBHBEvolution at
each scenario and pulsar baseline. No hardcoding. beta_SO = (47/6) chi
for equal-mass aligned spins (see compute_table4.py).

J0437 distance is imported from anchor_pulsars.J0437_DISTANCE_PC.
J1713 distance and sigma from Deller 2019 (VLBI): 1176 +/- 11 pc.
"""

import numpy as np
from smbhb_evolution import SMBHBEvolution
from anchor_pulsars import J0437_DISTANCE_PC

# ---- constants ----
YR = 365.25 * 86400.0
PC = 3.08567758e16
C_SI = 2.99792458e8

# ---- scenarios (match compute_table2.py and compute_table4.py) ----
SCENARIOS = [
    {"name": "Conservative", "m1": 5e7, "m2": 5e7, "f_E": 10e-6, "D_L": 2000},
    {"name": "Typical",      "m1": 3e8, "m2": 3e8, "f_E": 1e-6,  "D_L": 200},
    {"name": "Optimistic",   "m1": 5e8, "m2": 5e8, "f_E": 1e-6,  "D_L": 100},
]
CHI = 0.98

# ---- pulsars: (name, L_p [pc], sigma_L_p [pc], source) ----
PULSARS = [
    ("J0437-4715", J0437_DISTANCE_PC, 0.11, "Reardon 2024 timing parallax"),
    ("J1713+0747", 1176.0,            11.0, "Deller 2019 VLBI"),
]


def sigma_chi_pulsar(m1, m2, chi, f_E, D_L_Mpc, L_p_pc, sigma_L_p_pc):
    """Return dict with f_P, N_SO, sigma_chi for one (source, pulsar)."""
    tau_yr = L_p_pc * PC / C_SI / YR
    b = SMBHBEvolution(
        m1=m1, m2=m2, chi1=chi, chi2=chi,
        kappa1=0.0, kappa2=0.0,
        f_gw_earth=f_E, D_L=D_L_Mpc,
    )
    cyc = b.pn_decomposition(t_span_yr=tau_yr)["cycles"]
    f_P_Hz = cyc["f_P_nHz"] * 1e-9
    N_SO = cyc["SO"]                            # full spin-orbit cycles
    # sigma_tau in seconds
    sigma_tau_s = (sigma_L_p_pc / L_p_pc) * tau_yr * YR
    sigma_chi = chi * f_P_Hz * sigma_tau_s / N_SO
    return {
        "tau_yr":       tau_yr,
        "f_P_nHz":      cyc["f_P_nHz"],
        "N_SO":         N_SO,
        "sigma_tau_yr": sigma_tau_s / YR,
        "sigma_chi":    sigma_chi,
        "frac_dL":      sigma_L_p_pc / L_p_pc,
    }


def main():
    print("=" * 100)
    print("sigma_chi from single-pulsar distance-prior bound "
          "(Eq. eq:sigma_chi_prior)")
    print(f"chi = {CHI} aligned, equal-mass q=1")
    print("=" * 100)

    for psr, L_p, sL_p, src in PULSARS:
        print(f"\n{psr}: L_p = {L_p:.2f} pc +/- {sL_p:.2f} pc "
              f"({100 * sL_p / L_p:.3f}%) -- {src}")
        print(f"  tau = L_p/c = {L_p * PC / C_SI / YR:.1f} yr")
        hdr = (f"  {'scenario':<14s}{'f_P[nHz]':>10s}{'N_SO':>9s}"
               f"{'sigma_tau[yr]':>16s}{'sigma_chi':>12s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for sc in SCENARIOS:
            r = sigma_chi_pulsar(
                sc["m1"], sc["m2"], CHI, sc["f_E"], sc["D_L"],
                L_p, sL_p,
            )
            print(f"  {sc['name']:<14s}"
                  f"{r['f_P_nHz']:>10.2f}{r['N_SO']:>+9.2f}"
                  f"{r['sigma_tau_yr']:>16.4f}"
                  f"{r['sigma_chi']:>12.4f}")

    print()
    print("=" * 100)
    print("Range across fiducial scenarios (scenario-dependent, "
          "pick whichever matches paper quote)")
    print("=" * 100)


if __name__ == "__main__":
    main()
