#!/usr/bin/env python3
"""
crosscheck_cycle_counts.py — Independent verification of cycle counts.

Compares smbhb_evolution.py pn_decomposition() against two physically
independent methods:

  Method B : Hand-coded TaylorF2 SPA with Blanchet 2006 coefficients
             written fresh from the literature. The TaylorT2 time
             equation is solved for v_P by root-finding, and
             N_total = [v^{-5}(1 + phi_2 v^2 + phi_3 v^3 + phi_4 v^4)]
                       / (32 pi eta)  (evaluated between v_E and v_P)
             Catches algebraic typos in the library's coefficient table
             for the TaylorF2/T2 closed-form path (pn_decomposition).

  Method A : TaylorT1 numerical ODE. Backward integration of df/dt with
             cumulative GW cycles tracked as an auxiliary ODE state.
             Compared to the library's evolve() call (same approximant),
             this isolates coefficient/integration correctness from the
             known TaylorT1 vs TaylorF2 convention gap.

  External : Mingarelli et al. 2012 Table I. Non-spinning 1e9+1e9 Msun
             binary at f_E = 100 nHz, 1 kpc baseline. Independently
             published numbers.

All Blanchet coefficients in this file are typed fresh from Blanchet
2006 Living Review Rel. Eqs. 227 (flux), 193 (energy), 232 (TaylorT2
time), 234 (TaylorT2 phase). Nothing is imported from smbhb_evolution
beyond the SMBHBEvolution class used for comparison.

Pass criteria:
  library.pn_decomposition() vs Method B   : |dN| < 0.05 cyc, |dv/v| < 1e-6
  library.evolve(pn_order=4) vs Method A   : |dN| < 0.05 cyc, |dv/v| < 1e-6
  library N_total vs Mingarelli12 Table I  : |dN| < 0.5 cyc
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

from smbhb_evolution import SMBHBEvolution

# ======================================================================
# Independent physical constants (do NOT import from smbhb_evolution)
# ======================================================================
G_SI = 6.67430e-11        # m^3 kg^-1 s^-2  (CODATA 2018)
C_SI = 2.99792458e8       # m/s             (exact)
M_SUN = 1.98892e30        # kg              (IAU 2015 nominal solar mass)
PC = 3.08567758e16        # m
YR = 365.25 * 86400       # s               (Julian year)

# ======================================================================
# Method B: Blanchet 2006 coefficients, typed fresh from the literature
# ----------------------------------------------------------------------
# TaylorT2: t(v) = t_c - (5 M_s / (256 eta)) v^{-8} (1 + sum_k tau_k v^k)
#   tau_2 = 743/252 + 11 eta/3                    (Eq. 232, 1pN)
#   tau_3^tail = -(32/5) pi                       (Eq. 232, 1.5pN tail)
#   tau_3^SO   = (8/5) beta_SO                    (1.5pN SO; derived from CF+T2,
#                                                  see memory/reference_SO_coefficients_derivation.md)
#   tau_4 = 3058673/508032 + 5429 eta/504 + 617 eta^2/72 (Eq. 232, 2pN)
#   tau_4^SS = -(40/eta) sigma_SS                 (2pN spin-spin; Poisson 1998)
#
# TaylorT2 phase: Phi(v) = Phi_c - (1/(32 eta)) v^{-5} (1 + sum_k phi_k v^k)
#   phi_2 = 3715/1008 + 55 eta/12                 (Eq. 234, 1pN)
#   phi_3^tail = -10 pi                           (Eq. 234, 1.5pN tail)
#   phi_3^SO   = (5/2) beta_SO                    (1.5pN SO; derived from CF+T2)
#   phi_4 = 15293365/508032 + 27145 eta/504 + 3085 eta^2/72 (2pN mass)
#   phi_4^SS = -(10/eta) sigma_SS                 (2pN spin-spin)
#
# Spin-orbit coefficient (Blanchet Eq. 230, Kidder 1995 Eq. 4.21):
#   beta_SO = (1/12) sum_i [113 (m_i/M)^2 + 75 eta] chi_i cos kappa_i
#
# Spin-spin coefficient (Poisson 1998; aligned case):
#   sigma_SS = (eta/48) [-247 chi1 chi2 cos(kappa1 - kappa2)
#                        + 721 chi1 cos kappa1 chi2 cos kappa2]
# ======================================================================


def freshtau(eta, beta_SO, sigma_SS):
    """TaylorT2 time-equation coefficients (Blanchet Eq. 232)."""
    tau_2 = 743.0 / 252 + 11.0 * eta / 3
    tau_3 = -(32.0 / 5) * np.pi + (8.0 / 5) * beta_SO
    tau_4 = (3058673.0 / 508032 + 5429.0 * eta / 504
             + 617.0 * eta ** 2 / 72)
    if eta > 0:
        tau_4 += -(40.0 / eta) * sigma_SS
    return tau_2, tau_3, tau_4


def freshphi(eta, beta_SO, sigma_SS):
    """TaylorT2 phase-equation coefficients (Blanchet Eq. 234, GW cycles)."""
    phi_2 = 3715.0 / 1008 + 55.0 * eta / 12
    phi_3 = -10.0 * np.pi + (5.0 / 2) * beta_SO
    phi_4 = (15293365.0 / 508032 + 27145.0 * eta / 504
             + 3085.0 * eta ** 2 / 72)
    if eta > 0:
        phi_4 += -(10.0 / eta) * sigma_SS
    return phi_2, phi_3, phi_4


def fresh_beta_sigma(m1, m2, chi1, chi2, kappa1, kappa2):
    """Spin-orbit beta_SO and spin-spin sigma_SS; aligned cases collapse
    to scalar products."""
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    ck1, ck2 = np.cos(kappa1), np.cos(kappa2)
    sk1, sk2 = np.sin(kappa1), np.sin(kappa2)

    beta_SO = (1.0 / 12) * (
        (113 * (m1 / M) ** 2 + 75 * eta) * chi1 * ck1
        + (113 * (m2 / M) ** 2 + 75 * eta) * chi2 * ck2
    )

    cos12 = ck1 * ck2 + sk1 * sk2  # cos(kappa1 - kappa2)
    sigma_SS = (eta / 48) * (
        -247 * chi1 * chi2 * cos12
        + 721 * chi1 * ck1 * chi2 * ck2
    )
    return beta_SO, sigma_SS


# ======================================================================
# Method B: TaylorF2 closed-form total cycles
# ======================================================================
def method_B_total_cycles(m1_msun, m2_msun, chi1, chi2, kappa1, kappa2,
                          f_E, tau_yr):
    """
    Independent TaylorT2 computation of N_total, v_P, f_P.

    Returns
    -------
    dict with keys v_E, v_P, f_P, N_total.
    """
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    M_s = G_SI * M / C_SI ** 3              # geometrized total mass [s]

    beta_SO, sigma_SS = fresh_beta_sigma(m1, m2, chi1, chi2, kappa1, kappa2)
    tau_2, tau_3, tau_4 = freshtau(eta, beta_SO, sigma_SS)
    phi_2, phi_3, phi_4 = freshphi(eta, beta_SO, sigma_SS)

    v_E = (np.pi * M_s * f_E) ** (1.0 / 3)
    T_baseline = tau_yr * YR
    time_prefac = 5.0 * M_s / (256.0 * eta)

    def t_of_v(v):
        return time_prefac * v ** (-8) * (
            1.0 + tau_2 * v ** 2 + tau_3 * v ** 3 + tau_4 * v ** 4
        )

    target = T_baseline + t_of_v(v_E)
    v_P = brentq(lambda v: t_of_v(v) - target,
                 v_E * 1e-4, v_E * 0.99999,
                 xtol=1e-16, rtol=1e-14)

    phase_prefac = 1.0 / (32.0 * np.pi * eta)

    def N_of_v(v):
        return phase_prefac * v ** (-5) * (
            1.0 + phi_2 * v ** 2 + phi_3 * v ** 3 + phi_4 * v ** 4
        )

    N_total = N_of_v(v_P) - N_of_v(v_E)
    f_P = v_P ** 3 / (np.pi * M_s)

    return {"v_E": v_E, "v_P": v_P, "f_P": f_P, "N_total": N_total,
            "beta_SO": beta_SO, "sigma_SS": sigma_SS}


# ======================================================================
# Method A: TaylorT1 ODE, numerical backward integration
# ----------------------------------------------------------------------
# State: (f, N) where N is cumulative GW cycle count.
#   df/dt  = (96/5) pi^{8/3} Mc_s^{5/3} f^{11/3} * F_hat(v)/E'_hat(v)
#   dN/dt  = f
# Integrated backward from t=0 (Earth epoch) to t = -tau * YR.
# Flux F_hat(v) and energy derivative E'_hat(v) are written fresh from
# Blanchet 2006 Eq. 227 (flux) and Eq. 193 (energy), NOT imported.
# ======================================================================
def method_A_total_cycles(m1_msun, m2_msun, chi1, chi2, kappa1, kappa2,
                          f_E, tau_yr):
    """
    Independent TaylorT1 numerical computation of f_P and N_total.
    """
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    Mc = M * eta ** 0.6
    M_s = G_SI * M / C_SI ** 3
    Mc_s = G_SI * Mc / C_SI ** 3

    ck1, ck2 = np.cos(kappa1), np.cos(kappa2)
    sk1, sk2 = np.sin(kappa1), np.sin(kappa2)
    delta = (m1 - m2) / M
    chi_s = (chi1 * ck1 + chi2 * ck2) / 2.0
    chi_a = (chi1 * ck1 - chi2 * ck2) / 2.0

    beta_SO, sigma_SS = fresh_beta_sigma(m1, m2, chi1, chi2, kappa1, kappa2)

    # ---- Flux coefficients (Blanchet 2006 Eq. 227, Blanchet 2014 Eq. 314)
    F2 = -(1247.0 / 336 + 35.0 * eta / 12)
    F3 = 4.0 * np.pi - beta_SO
    F4 = (-(44711.0 / 9072 + 9271.0 * eta / 504 + 65.0 * eta ** 2 / 18)
          + sigma_SS)

    # ---- Energy derivative coefficients (Blanchet Eq. 193; Kidder 1995)
    # E(v) = -(mu/2) v^2 (1 + A2 v^2 + A3 v^3 + A4 v^4)
    # E'_hat(v) = 1 + 2 A2 v^2 + (5/2) A3 v^3 + 3 A4 v^4
    A2 = -(3.0 / 4 + eta / 12)
    A3 = (14.0 / 3 * chi_s + 2.0 * delta * chi_a) / 3.0  # SO part
    A4_mass = -(27.0 / 8 - 19.0 * eta / 8 + eta ** 2 / 24)
    A4_ss = -(1.0 / 48) * (
        -247 * chi1 * chi2 * (ck1 * ck2 + sk1 * sk2)
        + 721 * chi1 * ck1 * chi2 * ck2
    )
    A4 = A4_mass + A4_ss

    E2 = 2.0 * A2
    E3 = 2.5 * A3
    E4 = 3.0 * A4

    def rhs(neg_t, state):
        """State (f, cum_cycles). neg_t is positive-valued time before Earth.
        d/d(neg_t) = -d/dt_forward, so f decreases in neg_t (into past).
        Cycles accumulated along the past-pointing trajectory equal
        cycles between pulsar and Earth epochs (positive)."""
        f, _N = state
        if f <= 0:
            return [0.0, 0.0]
        v = (np.pi * M_s * f) ** (1.0 / 3)
        x = v * v
        F_hat = 1.0 + F2 * x + F3 * v ** 3 + F4 * x * x
        E_hat = 1.0 + E2 * x + E3 * v ** 3 + E4 * x * x
        dfdt = (96.0 / 5) * np.pi ** (8.0 / 3) * Mc_s ** (5.0 / 3) * \
            f ** (11.0 / 3) * F_hat / E_hat
        # df/d(neg_t) = -df/dt (frequency decreases in the past)
        # dN/d(neg_t) = +f  (cycle count accumulated as we walk the baseline)
        return [-dfdt, f]

    T = tau_yr * YR
    sol = solve_ivp(rhs, [0.0, T], [f_E, 0.0],
                    method="DOP853", rtol=1e-12, atol=1e-15,
                    max_step=T / 5000)

    f_P = sol.y[0, -1]
    N_cum = sol.y[1, -1]   # int_{-T}^{0} f dt, positive, GW cycles
    v_P = (np.pi * M_s * f_P) ** (1.0 / 3)
    return {"v_P": v_P, "f_P": f_P, "N_total": N_cum}


# ======================================================================
# Scenarios (mirror compute_table2.py)
# ======================================================================
SCENARIOS = [
    {"name": "Conservative", "m1": 5e7,  "m2": 5e7,  "f_E": 10e-6, "chi": 0.98},
    {"name": "Typical",      "m1": 3e8,  "m2": 3e8,  "f_E": 1e-6,  "chi": 0.98},
    {"name": "Optimistic",   "m1": 5e8,  "m2": 5e8,  "f_E": 1e-6,  "chi": 0.98},
]
TAU_1KPC_YR = 1000 * PC / C_SI / YR          # ~3262 yr
TAU_J0437_YR = 533.0

# Mingarelli 2012 Table I: m1 = m2 = 1e9, f_E = 100 nHz, 1 kpc, non-spinning
MINGARELLI12_TABLE1 = {
    "m1": 1e9, "m2": 1e9, "chi": 0.0, "f_E": 100e-9, "tau_yr": TAU_1KPC_YR,
    "N_total_paper": 4305.1,
}


# ======================================================================
# Runner
# ======================================================================
def compare(label, m1, m2, chi, f_E, tau_yr, expected_N=None):
    """Run all methods for one binary configuration.

    Comparison matrix:
      library.pn_decomposition()  vs  Method B      (TaylorF2 path)
      library.evolve(pn_order=4)  vs  Method A      (TaylorT1 path)
      library.pn_decomposition()  vs  Mingarelli12  (external check)
    """
    binary = SMBHBEvolution(
        m1=m1, m2=m2, chi1=chi, chi2=chi,
        kappa1=0.0, kappa2=0.0, f_gw_earth=f_E, D_L=100.0,
    )

    # ---- Library TaylorF2 path (pn_decomposition) --------------------
    lib_F2 = binary.pn_decomposition(t_span_yr=tau_yr)
    lib_F2_cyc = lib_F2["cycles"]
    libN_F2 = lib_F2_cyc["Total"] - lib_F2_cyc["Thomas"]  # 2pN without Thomas
    lib_fP_F2 = lib_F2_cyc["f_P_nHz"] * 1e-9
    lib_vP_F2 = lib_F2_cyc["v_P"]

    # ---- Library TaylorT1 path (evolve) ------------------------------
    # evolve() integrates backward in time. Its Phi[-1] is the phase at
    # the pulsar epoch measured relative to Earth epoch (Phi=0 at Earth),
    # so |Phi[-1]/(2*pi)| = cycles between pulsar and Earth epochs.
    lib_T1 = binary.evolve(t_span_yr=tau_yr, n_points=10000, pn_order=4)
    libN_T1 = abs(lib_T1["Phi"][-1] / (2 * np.pi))
    lib_fP_T1 = lib_T1["f_gw"][-1]
    M_s = binary.M_s
    lib_vP_T1 = (np.pi * M_s * lib_fP_T1) ** (1.0 / 3)

    # ---- Independent methods -----------------------------------------
    B = method_B_total_cycles(m1, m2, chi, chi, 0.0, 0.0, f_E, tau_yr)
    A = method_A_total_cycles(m1, m2, chi, chi, 0.0, 0.0, f_E, tau_yr)

    print(f"\n--- {label}")
    print(f"  lib TaylorF2  : N = {libN_F2:>12.3f}  v_P = {lib_vP_F2:.8e}  "
          f"f_P = {lib_fP_F2 * 1e9:.6f} nHz")
    print(f"  Method B      : N = {B['N_total']:>12.3f}  v_P = {B['v_P']:.8e}  "
          f"f_P = {B['f_P'] * 1e9:.6f} nHz")
    print(f"  lib TaylorT1  : N = {libN_T1:>12.3f}  v_P = {lib_vP_T1:.8e}  "
          f"f_P = {lib_fP_T1 * 1e9:.6f} nHz")
    print(f"  Method A      : N = {A['N_total']:>12.3f}  v_P = {A['v_P']:.8e}  "
          f"f_P = {A['f_P'] * 1e9:.6f} nHz")

    dN_F2 = libN_F2 - B["N_total"]
    dv_F2 = lib_vP_F2 / B["v_P"] - 1.0
    dN_T1 = libN_T1 - A["N_total"]
    dv_T1 = lib_vP_T1 / A["v_P"] - 1.0

    print(f"  dN  libF2 - B = {dN_F2:+.3e} cyc    "
          f"dv  libF2 - B = {dv_F2:+.2e}")
    print(f"  dN  libT1 - A = {dN_T1:+.3e} cyc    "
          f"dv  libT1 - A = {dv_T1:+.2e}")

    fail = []
    if abs(dN_F2) > 0.05:
        fail.append(f"libF2 vs B: |dN| = {abs(dN_F2):.3e} > 0.05")
    if abs(dv_F2) > 1e-6:
        fail.append(f"libF2 vs B: |dv/v| = {abs(dv_F2):.2e} > 1e-6")
    if abs(dN_T1) > 0.05:
        fail.append(f"libT1 vs A: |dN| = {abs(dN_T1):.3e} > 0.05")
    if abs(dv_T1) > 1e-6:
        fail.append(f"libT1 vs A: |dv/v| = {abs(dv_T1):.2e} > 1e-6")
    if expected_N is not None:
        dN_ext = libN_F2 - expected_N
        print(f"  dN  libF2 - Mingarelli12 paper = {dN_ext:+.3f} cyc")
        if abs(dN_ext) > 0.5:
            fail.append(f"lib vs Mingarelli12: |dN| = {abs(dN_ext):.3f} > 0.5")
    return fail


def main():
    print("=" * 70)
    print("crosscheck_cycle_counts.py")
    print("  lib TaylorF2 : smbhb_evolution.pn_decomposition()")
    print("  lib TaylorT1 : smbhb_evolution.evolve(pn_order=4)")
    print("  Method B     : hand-coded TaylorF2 SPA, fresh Blanchet coeffs")
    print("  Method A     : TaylorT1 numerical ODE, fresh Blanchet coeffs")
    print("  External     : Mingarelli et al. 2012 Table I")
    print("=" * 70)

    all_fail = []

    # External check: Mingarelli 2012 Table I
    p = MINGARELLI12_TABLE1
    all_fail += compare(
        "Mingarelli12 Table I (1e9+1e9, 100 nHz, chi=0, 1 kpc)",
        p["m1"], p["m2"], p["chi"], p["f_E"], p["tau_yr"],
        expected_N=p["N_total_paper"],
    )

    # Fiducial scenarios from compute_table2.py
    for sc in SCENARIOS:
        for psr_name, tau in [("1 kpc", TAU_1KPC_YR), ("J0437", TAU_J0437_YR)]:
            label = f"{sc['name']:<13s} + {psr_name}"
            all_fail += compare(
                label, sc["m1"], sc["m2"], sc["chi"], sc["f_E"], tau,
            )

    print("\n" + "=" * 70)
    if all_fail:
        print(f"FAIL: {len(all_fail)} discrepancy(ies)")
        for f in all_fail:
            print(f"  - {f}")
    else:
        print("PASS: all methods agree within tolerance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
