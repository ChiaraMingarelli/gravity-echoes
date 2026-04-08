#!/usr/bin/env python3
"""
compute_delta_f.py — Frequency shift at each pN order for Table II.

For each scenario and pulsar, compute f_P by solving the TaylorT2 time
equation at successive pN orders:
  - Newtonian only (tau_n = phi_n = 0 for n >= 2)
  - + 1pN
  - + 1.5pN (tail + SO)
  - + 2pN (mass + SS)

Then report Delta f at each order = f_P(cumulative) - f_P(previous order).

Uses the same TaylorF2/T2 framework as smbhb_evolution.py pn_decomposition().
"""

import numpy as np
from smbhb_evolution import SMBHBEvolution

YR = 365.25 * 86400
PC = 3.08567758e16
MPC = 1e6 * PC
G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN = 1.98892e30

SCENARIOS = [
    {"name": "Conservative", "m1": 5e7,   "m2": 5e7,   "f_E": 10e-6, "D_L": 2000},
    {"name": "Typical",      "m1": 2.5e8, "m2": 2.5e8, "f_E": 1e-6,  "D_L": 200},
    {"name": "Optimistic",   "m1": 5e8,   "m2": 5e8,   "f_E": 1e-6,  "D_L": 100},
]

CHI = 0.7
TAU_1KPC_YR = 1000 * PC / C_SI / YR  # ~3262 yr

PULSARS = [
    ("1 kpc",  TAU_1KPC_YR),
    ("J0437",  533.0),
]


def compute_fP_at_each_order(m1, m2, chi, f_E, D_L_Mpc, tau_yr):
    """
    Solve the TaylorT2 time equation for v_P at each cumulative pN order,
    convert to f_P.

    Returns dict with f_P at each order and the incremental shifts.
    """
    from scipy.optimize import brentq

    binary = SMBHBEvolution(
        m1=m1, m2=m2,
        chi1=chi, chi2=chi,
        kappa1=0.0, kappa2=0.0,
        f_gw_earth=f_E,
        D_L=D_L_Mpc,
    )

    eta = binary.eta
    M_s = binary.M_s
    v_E = (np.pi * M_s * f_E) ** (1.0 / 3)
    T_baseline = tau_yr * YR
    time_prefac = 5.0 * M_s / (256.0 * eta)

    # TaylorT2 time coefficients (from smbhb_evolution.py)
    tau_2 = 743.0 / 252 + 11.0 * eta / 3
    tau_3_mass = -(32.0 / 5) * np.pi
    tau_3_SO = (binary.beta_so * 48.0) / 5.0
    tau_4 = (3058673.0 / 508032 + 5429.0 * eta / 504
             + 617.0 * eta ** 2 / 72)
    tau_4_SS = -binary.sigma_ss * 40.0 / eta if eta > 0 else 0.0

    def time_from_v(v, t2, t3, t4):
        return time_prefac * v ** (-8) * (
            1.0 + t2 * v ** 2 + t3 * v ** 3 + t4 * v ** 4
        )

    def solve_vP(t2, t3, t4):
        target = T_baseline + time_from_v(v_E, t2, t3, t4)
        def residual(v):
            return time_from_v(v, t2, t3, t4) - target
        v_lo, v_hi = v_E * 0.001, v_E * 0.9999
        if residual(v_lo) * residual(v_hi) > 0:
            v_est = (v_E ** (-8) + 256.0 * eta * T_baseline
                     / (5.0 * M_s)) ** (-1.0 / 8)
            v_lo = v_est * 0.5
        return brentq(residual, v_lo, v_hi, xtol=1e-15, rtol=1e-14)

    # Solve at each cumulative order
    # Newtonian: all correction coefficients = 0
    v_P_Newt = solve_vP(0.0, 0.0, 0.0)

    # + 1pN
    v_P_1pN = solve_vP(tau_2, 0.0, 0.0)

    # + 1.5pN (tail + SO)
    v_P_15pN = solve_vP(tau_2, tau_3_mass + tau_3_SO, 0.0)

    # + 2pN (full)
    v_P_2pN = solve_vP(tau_2, tau_3_mass + tau_3_SO, tau_4 + tau_4_SS)

    # Also: 1.5pN without SO (tail only) for decomposition
    v_P_15pN_noSO = solve_vP(tau_2, tau_3_mass, 0.0)

    # Convert to frequencies
    def v_to_f(v):
        return v ** 3 / (np.pi * M_s)

    f_Newt = v_to_f(v_P_Newt)
    f_1pN = v_to_f(v_P_1pN)
    f_15pN = v_to_f(v_P_15pN)
    f_2pN = v_to_f(v_P_2pN)
    f_15pN_noSO = v_to_f(v_P_15pN_noSO)

    return {
        "f_Newt_nHz": f_Newt * 1e9,
        "f_1pN_nHz": f_1pN * 1e9,
        "f_15pN_nHz": f_15pN * 1e9,
        "f_2pN_nHz": f_2pN * 1e9,
        "f_15pN_noSO_nHz": f_15pN_noSO * 1e9,
        # Incremental shifts
        "df_1pN_nHz": (f_1pN - f_Newt) * 1e9,
        "df_15pN_nHz": (f_15pN - f_1pN) * 1e9,
        "df_15pN_tail_nHz": (f_15pN_noSO - f_1pN) * 1e9,
        "df_15pN_SO_nHz": (f_15pN - f_15pN_noSO) * 1e9,
        "df_2pN_nHz": (f_2pN - f_15pN) * 1e9,
    }


def main():
    print("=" * 110)
    print("Frequency decomposition by pN order (chi = 0.7 aligned)")
    print("f_P computed from TaylorT2 time equation at each cumulative order")
    print("=" * 110)

    for sc in SCENARIOS:
        m1, m2 = sc["m1"], sc["m2"]
        f_E = sc["f_E"]
        M_tot = m1 + m2

        print(f"\n{'='*110}")
        print(f"{sc['name']} — M_tot = {M_tot:.0e} Msun, f_E = {f_E*1e6:.0f} uHz, "
              f"D_L = {sc['D_L']} Mpc")
        print(f"{'='*110}")

        hdr = (f"{'Pulsar':<8s} {'tau[yr]':>8s} "
               f"{'f_Newt':>10s} {'f_+1pN':>10s} {'f_+1.5pN':>10s} {'f_+2pN':>10s} | "
               f"{'df_1pN':>10s} {'df_1.5pN':>10s} {'(tail)':>10s} {'(SO)':>10s} {'df_2pN':>10s}")
        print(hdr)
        print("-" * len(hdr))

        for psr_name, tau_yr in PULSARS:
            result = compute_fP_at_each_order(
                m1, m2, CHI, f_E, sc["D_L"], tau_yr
            )

            print(
                f"{psr_name:<8s} {tau_yr:>8.0f} "
                f"{result['f_Newt_nHz']:>10.3f} "
                f"{result['f_1pN_nHz']:>10.3f} "
                f"{result['f_15pN_nHz']:>10.3f} "
                f"{result['f_2pN_nHz']:>10.3f} | "
                f"{result['df_1pN_nHz']:>+10.3f} "
                f"{result['df_15pN_nHz']:>+10.3f} "
                f"{result['df_15pN_tail_nHz']:>+10.3f} "
                f"{result['df_15pN_SO_nHz']:>+10.3f} "
                f"{result['df_2pN_nHz']:>+10.3f}"
            )
            print(f"{'':>18s} (all values in nHz)")

    # Also print fractional shifts
    print(f"\n\n{'='*110}")
    print("Fractional frequency shifts df/f_Newt at each pN order")
    print(f"{'='*110}")

    for sc in SCENARIOS:
        m1, m2 = sc["m1"], sc["m2"]
        f_E = sc["f_E"]
        M_tot = m1 + m2
        print(f"\n{sc['name']} — M_tot = {M_tot:.0e} Msun")

        for psr_name, tau_yr in PULSARS:
            result = compute_fP_at_each_order(
                m1, m2, CHI, f_E, sc["D_L"], tau_yr
            )
            f_N = result["f_Newt_nHz"]
            print(
                f"  {psr_name:<8s}  "
                f"df_1pN/f = {result['df_1pN_nHz']/f_N:>+8.4f}  "
                f"df_1.5pN/f = {result['df_15pN_nHz']/f_N:>+8.4f}  "
                f"  (tail: {result['df_15pN_tail_nHz']/f_N:>+8.4f}, "
                f"SO: {result['df_15pN_SO_nHz']/f_N:>+8.4f})  "
                f"df_2pN/f = {result['df_2pN_nHz']/f_N:>+8.4f}"
            )


if __name__ == "__main__":
    main()
