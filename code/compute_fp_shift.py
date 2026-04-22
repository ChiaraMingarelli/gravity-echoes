#!/usr/bin/env python3
"""
compute_fp_shift.py — Pulsar-term frequency shift from omitting spin,
computed with the SAME TaylorT2 machinery used for Table III.

Uses smbhb_evolution.py so that f_P values are identical to Table III when
spin is on (chi = 0.98 aligned) and to a chi = 0 "no-spin template" when
spin is off.  This avoids the TaylorT2/TaylorF2 reorganization mismatch
between a standalone integration and the library.

For each scenario:
    f_P^spin    = SMBHBEvolution(chi1=chi2=0.98).pn_decomposition(...).f_P
    f_P^nospin  = SMBHBEvolution(chi1=chi2=0.0).pn_decomposition(...).f_P
    Delta f_P   = f_P^spin - f_P^nospin

This matches the definition "Delta f = f_N * delta_SO is the error incurred
by omitting spin from the template", where dropping spin means chi = 0
throughout, killing both 1.5pN spin-orbit and 2pN spin-spin terms.
"""

import numpy as np
from smbhb_evolution import SMBHBEvolution

YR = 365.25 * 86400.0
PC = 3.08567758e16
KPC = 1e3 * PC
C = 2.99792458e8

L_KPC = 1.0
TAU_S = L_KPC * KPC / C
TAU_YR = TAU_S / YR

SCENARIOS = [
    {"name": "Conservative", "m1": 5e7, "m2": 5e7, "f_E": 10e-6},
    {"name": "Typical",      "m1": 3e8, "m2": 3e8, "f_E": 1e-6},
    {"name": "Optimistic",   "m1": 5e8, "m2": 5e8, "f_E": 1e-6},
]
CHI = 0.98  # aligned, near-extremal Kerr (migrated 2026-04-20 from 0.7)


def fP_for_chi(m1, m2, chi, f_E, tau_yr):
    b = SMBHBEvolution(
        m1=m1, m2=m2,
        chi1=chi, chi2=chi,
        kappa1=0.0, kappa2=0.0,   # aligned
        f_gw_earth=f_E, D_L=100.0,
    )
    out = b.pn_decomposition(t_span_yr=tau_yr)
    return out["cycles"]["f_P_nHz"]  # in nHz


def main():
    print("=" * 100)
    print(f"Pulsar-term frequency shift using smbhb_evolution (TaylorT2)  "
          f"tau = {TAU_YR:.0f} yr, chi = {CHI}")
    print("=" * 100)
    hdr = (f"{'Scenario':<14}{'f_E [nHz]':>12}{'f_P^spin':>14}"
           f"{'f_P^nospin':>14}{'Delta f_P':>14}"
           f"{'|D|/bin(20 yr)':>18}{'|D|/bin(50 yr)':>18}")
    print(hdr)
    print(f"{'':<14}{'':>12}{'[nHz]':>14}{'[nHz]':>14}{'[nHz]':>14}")
    print("-" * len(hdr))

    bin_20 = 1.0 / (20 * YR) * 1e9  # nHz
    bin_50 = 1.0 / (50 * YR) * 1e9  # nHz

    for s in SCENARIOS:
        fP_spin = fP_for_chi(s["m1"], s["m2"], CHI, s["f_E"], TAU_YR)
        fP_nosp = fP_for_chi(s["m1"], s["m2"], 0.0, s["f_E"], TAU_YR)
        dfP = fP_spin - fP_nosp
        print(f"{s['name']:<14}{s['f_E']*1e9:>12.0f}"
              f"{fP_spin:>14.3f}{fP_nosp:>14.3f}{dfP:>14.3f}"
              f"{abs(dfP)/bin_20:>18.2f}{abs(dfP)/bin_50:>18.2f}")
    print()
    print(f"PTA bin widths:  1/(20 yr) = {bin_20:.3f} nHz,  "
          f"1/(50 yr) = {bin_50:.3f} nHz")


if __name__ == "__main__":
    main()
