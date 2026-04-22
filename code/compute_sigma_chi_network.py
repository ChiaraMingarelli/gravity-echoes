#!/usr/bin/env python3
"""
compute_sigma_chi_network.py -- Fisher-network bound on sigma_chi.

Verifies the paper's Appendix C claim (L639 of restructured-echo.tex):

  "A Fisher sum over the ~20 VLBI-calibrated MSPs with delta L_p/L_p
   <~ 1% distributed over 0.5--3 kpc gives sigma_chi ~ 0.2 ...
   Even a factor-of-ten improvement in VLBI precision would only bring
   this down to sigma_chi ~ 0.02."

Method.  For a network of pulsars the per-pulsar distance-prior bound
(Eq. sigma_chi_prior) combines in quadrature:

    sigma_chi^{-2}  =  sum_i  1 / sigma_chi_i^2 ,
    sigma_chi_i     =  chi * f_{P,i} * sigma_tau_i / N_{SO,i} ,
    sigma_tau_i     =  (delta L_p / L_p) * tau_i .

Under a uniform fractional distance cut, sigma_chi_net scales as
    sigma_chi_net  =  (delta L_p / L_p) * [ sum_i (N_SO,i / (chi f_P,i tau_i))^2 ]^{-1/2} ,
so the 10x VLBI improvement gives exactly 10x better sigma_chi_net.

We construct a hypothetical ~20-pulsar network as stated in the paper:
distances uniformly spaced over [0.5, 3] kpc, every pulsar at the
fractional distance cut delta L_p/L_p = 1% (this is an optimistic
ceiling; the actual VLBI-calibrated sample reaches that precision
only for J1713+0747).  f_P,i and N_SO,i are computed by
smbhb_evolution.SMBHBEvolution once per (scenario, pulsar) and then
rescaled for the 0.1% case.
"""

import numpy as np
from smbhb_evolution import SMBHBEvolution

# ---- constants ----
YR = 365.25 * 86400.0
PC = 3.08567758e16
C_SI = 2.99792458e8

# ---- scenarios (match compute_sigma_chi_pulsar.py) ----
SCENARIOS = [
    {"name": "Conservative", "m1": 5e7, "m2": 5e7, "f_E": 10e-6, "D_L": 2000},
    {"name": "Typical",      "m1": 3e8, "m2": 3e8, "f_E": 1e-6,  "D_L": 200},
    {"name": "Optimistic",   "m1": 5e8, "m2": 5e8, "f_E": 1e-6,  "D_L": 100},
]
CHI = 0.98

# ---- hypothetical VLBI-calibrated anchor network ----
N_PULSARS = 20
D_MIN_KPC = 0.5
D_MAX_KPC = 3.0
PULSAR_DISTANCES_PC = np.linspace(
    D_MIN_KPC * 1e3, D_MAX_KPC * 1e3, N_PULSARS
)

FRAC_DL_FIDUCIAL = 0.01     # 1% VLBI precision cut
FRAC_DL_IMPROVED = 0.001    # 10x improvement


def per_pulsar(scenario, L_p_pc):
    """Return (tau_yr, f_P_nHz, N_SO) for one (scenario, pulsar)."""
    tau_yr = L_p_pc * PC / C_SI / YR
    b = SMBHBEvolution(
        m1=scenario["m1"], m2=scenario["m2"],
        chi1=CHI, chi2=CHI,
        kappa1=0.0, kappa2=0.0,
        f_gw_earth=scenario["f_E"], D_L=scenario["D_L"],
    )
    # lower resolution; we only need analytic cycle counts
    cyc = b.pn_decomposition(t_span_yr=tau_yr, n_points=1000)["cycles"]
    return tau_yr, cyc["f_P_nHz"], cyc["SO"]


def network_sigma_chi(rows, frac_dL):
    """rows: list of (tau_yr, f_P_nHz, N_SO). Return (sigma_chi_net, per_pulsar_sigmas)."""
    inv_sq = 0.0
    sigmas = []
    for tau_yr, f_P_nHz, N_SO in rows:
        f_P_Hz = f_P_nHz * 1e-9
        sigma_tau_s = frac_dL * tau_yr * YR
        sig_chi = CHI * f_P_Hz * sigma_tau_s / N_SO
        sigmas.append(sig_chi)
        inv_sq += 1.0 / sig_chi**2
    return 1.0 / np.sqrt(inv_sq), sigmas


def main():
    print("=" * 92)
    print("Network Fisher-sum bound on sigma_chi "
          "(hypothetical 20-MSP VLBI anchor set)")
    print(f"chi = {CHI}, equal-mass q=1, N_p = {N_PULSARS}, "
          f"L_p uniform in [{D_MIN_KPC}, {D_MAX_KPC}] kpc")
    print("=" * 92)

    # Precompute per-(scenario, pulsar) tau/f_P/N_SO once
    table = {}
    for sc in SCENARIOS:
        rows = []
        print(f"\n[computing {sc['name']}...]")
        for L_p_pc in PULSAR_DISTANCES_PC:
            tau_yr, f_P_nHz, N_SO = per_pulsar(sc, L_p_pc)
            rows.append((tau_yr, f_P_nHz, N_SO))
        table[sc["name"]] = rows

    for frac_dL, label in [(FRAC_DL_FIDUCIAL, "Fiducial VLBI (1%)"),
                           (FRAC_DL_IMPROVED, "10x improved VLBI (0.1%)")]:
        print(f"\n--- {label}: delta L_p / L_p = {frac_dL*100:g}% ---")
        print(f"  {'scenario':<14s}{'sigma_chi (network)':>24s}")
        print("  " + "-" * 38)
        for sc in SCENARIOS:
            s_net, _ = network_sigma_chi(table[sc["name"]], frac_dL)
            print(f"  {sc['name']:<14s}{s_net:>24.4f}")

    print()
    print("=" * 92)
    print("Per-pulsar breakdown at the Optimistic scenario "
          "(matches the single-pulsar quotes):")
    print("=" * 92)
    print("\n--- Optimistic, 1% VLBI ---")
    print(f"  {'L_p[pc]':>9s}{'tau[yr]':>11s}{'f_P[nHz]':>10s}{'N_SO':>9s}{'sigma_chi_i':>13s}")
    _, sigmas = network_sigma_chi(table["Optimistic"], FRAC_DL_FIDUCIAL)
    for (tau_yr, f_P_nHz, N_SO), sig_chi in zip(table["Optimistic"], sigmas):
        L_p_pc = tau_yr * YR * C_SI / PC
        print(f"  {L_p_pc:>9.0f}{tau_yr:>11.1f}{f_P_nHz:>10.2f}{N_SO:>+9.2f}{sig_chi:>13.4f}")


if __name__ == "__main__":
    main()
