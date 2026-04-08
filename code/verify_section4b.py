"""
verify_section4b.py

Reproduces every number in Section 4b (The spin-distance degeneracy)
of restructured-echo.tex using the TaylorF2 analytic decomposition
from smbhb_evolution.py.

All numbers are computed from first principles using the library's
beta_so and sigma_ss coefficients. No guessing.
"""

import numpy as np
from scipy.optimize import brentq
from smbhb_evolution import SMBHBEvolution, G_SI, C_SI, M_SUN, PC, YR


def quick_taylorf2(m1, m2, chi1, chi2, kappa1, kappa2, f_E, D_L, t_span_yr):
    """
    Analytic TaylorF2 cycle decomposition using SMBHBEvolution for
    correct coefficients, without running the slow numerical TaylorT1
    evolution.
    """
    b = SMBHBEvolution(m1=m1, m2=m2, chi1=chi1, chi2=chi2,
                       kappa1=kappa1, kappa2=kappa2,
                       f_gw_earth=f_E, D_L=D_L)
    eta, M_s = b.eta, b.M_s

    # TaylorT2 time coefficients (Blanchet 2006, Eq. 232)
    tau_2 = 743.0 / 252 + 11.0 * eta / 3
    tau_3_mass = -(32.0 / 5) * np.pi
    tau_3_SO = b.beta_so * 48.0 / 5.0
    tau_3 = tau_3_mass + tau_3_SO
    tau_4 = 3058673.0 / 508032 + 5429.0 * eta / 504 + 617.0 * eta**2 / 72
    tau_4_SS = -b.sigma_ss * 40.0 / eta if eta > 0 else 0.0
    tau_4_full = tau_4 + tau_4_SS

    # TaylorT2 phase coefficients (Blanchet 2006, Eq. 234)
    phi_2 = 3715.0 / 1008 + 55.0 * eta / 12
    phi_3_mass = -10.0 * np.pi
    phi_3_SO = (10.0 / 3) * b.beta_so
    phi_3 = phi_3_mass + phi_3_SO
    phi_4 = 15293365.0 / 508032 + 27145.0 * eta / 504 + 3085.0 * eta**2 / 72
    phi_4_SS = -(10.0 / eta) * b.sigma_ss if eta > 0 else 0.0
    phi_4_full = phi_4 + phi_4_SS

    # TaylorF2 cycle-counting coefficients
    psi_2 = (8 * phi_2 - 5 * tau_2) / 3.0
    psi_3_mass = (8 * phi_3_mass - 5 * tau_3_mass) / 3.0
    psi_4 = phi_4  # standard TaylorF2 2pN

    v_E = (np.pi * M_s * f_E) ** (1.0 / 3)
    T = t_span_yr * YR
    time_pf = 5.0 * M_s / (256.0 * eta)
    phase_pf = 1.0 / (32.0 * np.pi * eta)
    fd_pf = 3.0 / (256.0 * np.pi * eta)

    def t_from_v(v, t2, t3, t4):
        return time_pf * v**(-8) * (1.0 + t2 * v**2 + t3 * v**3 + t4 * v**4)

    def solve_vP(t2, t3, t4):
        target = T + t_from_v(v_E, t2, t3, t4)
        def res(v):
            return t_from_v(v, t2, t3, t4) - target
        return brentq(res, v_E * 0.001, v_E * 0.9999, xtol=1e-15)

    def phi_T2(v, p2, p3, p4):
        return phase_pf * v**(-5) * (1.0 + p2 * v**2 + p3 * v**3 + p4 * v**4)

    v_P = solve_vP(tau_2, tau_3, tau_4_full)
    v_P_noSO = solve_vP(tau_2, tau_3_mass, tau_4_full)

    N_total = phi_T2(v_P, phi_2, phi_3, phi_4_full) - phi_T2(v_E, phi_2, phi_3, phi_4_full)
    N_total_noSO = (phi_T2(v_P_noSO, phi_2, phi_3_mass, phi_4_full)
                    - phi_T2(v_E, phi_2, phi_3_mass, phi_4_full))

    N_1pN = fd_pf * psi_2 * (v_P**(-3) - v_E**(-3))
    N_15pN = fd_pf * psi_3_mass * (v_P**(-2) - v_E**(-2))
    N_2pN = fd_pf * psi_4 * (v_P**(-1) - v_E**(-1))
    N_SO = N_total - N_total_noSO
    N_Newt = N_total - N_1pN - N_15pN - N_SO - N_2pN

    f_P = v_P**3 / (np.pi * M_s)
    return {
        "Newt": N_Newt, "1pN": N_1pN, "1.5pN": N_15pN,
        "SO": N_SO, "2pN": N_2pN, "Total": N_total,
        "v_E": v_E, "v_P": v_P, "f_P": f_P,
        "beta_so": b.beta_so, "sigma_ss": b.sigma_ss,
    }


def sigma_chi_pulsar(m1, m2, chi, f_E, D_L, tau_yr, sig_d_over_d):
    """Per-pulsar sigma_chi from Eq. 16 of the paper."""
    r = quick_taylorf2(m1, m2, chi, chi, 0, 0, f_E, D_L, tau_yr)
    sigma_tau = sig_d_over_d * tau_yr * YR
    sc = chi * r["f_P"] * sigma_tau / abs(r["SO"])
    return sc, r


# ======================================================================
# Line 343: SO cycles over kpc baseline
# ======================================================================
print("=" * 72)
print("Line 343: SO cycles over 1 kpc baseline (tau = L/c)")
print("=" * 72)
tau_kpc = 1000 * PC / C_SI / YR
print(f"tau(1 kpc) = {tau_kpc:.0f} yr\n")
so_vals = []
for M_tot in [1e8, 5e8, 1e9]:
    for chi in [0.3, 0.7, 0.9]:
        r = quick_taylorf2(M_tot / 2, M_tot / 2, chi, chi, 0, 0, 1e-6, 100, tau_kpc)
        so_vals.append(abs(r["SO"]))
        print(f"  M={M_tot:.0e}, chi={chi}: N_SO = {r['SO']:.1f}")
print(f"\nRange: ~{min(so_vals):.0f}--{max(so_vals):.0f}")
print("Paper says: ~36--215  [CHECK]")

# ======================================================================
# Line 343: Spin-spin cycles
# ======================================================================
print("\n" + "=" * 72)
print("Line 343: Spin-spin (2pN) cycles")
print("=" * 72)
r = quick_taylorf2(5e8, 5e8, 0.7, 0.7, 0, 0, 1e-6, 100, 7671)
print(f"Optimistic J1713 max baseline: N_2pN = {r['2pN']:.1f}")
print("Paper says: ~6  [CHECK]")

# ======================================================================
# Line 345: N_SO ~ 102 and ~640 rad
# ======================================================================
print("\n" + "=" * 72)
print("Line 345: N_SO for optimistic J1713 max baseline")
print("=" * 72)
print(f"N_SO = {r['SO']:.1f}")
print(f"Phase = 2pi * |N_SO| = {2 * np.pi * abs(r['SO']):.0f} rad")
print("Paper says: ~102 cycles, ~640 rad  [CHECK]")

# ======================================================================
# Line 354: Earth-term SO cycles
# ======================================================================
print("\n" + "=" * 72)
print("Line 354: Earth-term SO cycles (f_E to ~ISCO)")
print("=" * 72)
for M_tot, label in [(1e9, "Optimistic"), (5e8, "Typical")]:
    M_s = G_SI * M_tot * M_SUN / C_SI**3
    v_E = (np.pi * M_s * 1e-6) ** (1.0 / 3)
    t_merge = (5.0 / (256.0 * 0.25)) * M_s * v_E ** (-8) / YR
    r = quick_taylorf2(M_tot / 2, M_tot / 2, 0.7, 0.7, 0, 0, 1e-6, 100, t_merge * 0.99)
    print(f"{label}: t_merge = {t_merge:.2f} yr, N_SO^(E) = {r['SO']:.1f}")
print("Paper says: ~8 (optimistic), ~11 (typical)  [CHECK]")

# ======================================================================
# Line 357: sigma_chi^(E)
# ======================================================================
print("\n" + "=" * 72)
print("Line 357: sigma_chi^(E)")
print("=" * 72)
rho_E = 4e5
chi = 0.7
for M_tot, label in [(1e9, "Optimistic"), (5e8, "Typical")]:
    M_s = G_SI * M_tot * M_SUN / C_SI**3
    v_E = (np.pi * M_s * 1e-6) ** (1.0 / 3)
    t_merge = (5.0 / (256.0 * 0.25)) * M_s * v_E ** (-8) / YR
    r = quick_taylorf2(M_tot / 2, M_tot / 2, 0.7, 0.7, 0, 0, 1e-6, 100, t_merge * 0.99)
    sc = chi / (2 * np.pi * abs(r["SO"]) * rho_E)
    print(f"{label}: sigma_chi^(E) = {sc:.2e}")
print("Paper says: ~3e-8  [CHECK]")

# ======================================================================
# Line 361: N_SO^(P) range across array
# ======================================================================
print("\n" + "=" * 72)
print("Line 361: Pulsar-term SO range")
print("=" * 72)
for M_tot, label in [(1e9, "Optimistic"), (5e8, "Typical")]:
    for tau, name in [(533, "J0437"), (7671, "J1713")]:
        r = quick_taylorf2(M_tot / 2, M_tot / 2, 0.7, 0.7, 0, 0, 1e-6, 100, tau)
        print(f"  {label} {name}: N_SO = {r['SO']:.1f}")
print("Paper says: ~45--125  [CHECK]")

# ======================================================================
# Line 382: Per-pulsar sigma_chi
# ======================================================================
print("\n" + "=" * 72)
print("Line 382: Per-pulsar sigma_chi")
print("=" * 72)
sc0437, r0437 = sigma_chi_pulsar(5e8, 5e8, 0.7, 1e-6, 100, 533, 0.11 / 156.96)
sc1713, r1713 = sigma_chi_pulsar(5e8, 5e8, 0.7, 1e-6, 100, 7671, 11.0 / 1176)
print(f"J0437: sigma_d/d = {0.11/156.96*100:.3f}%, N_SO = {r0437['SO']:.1f}, sigma_chi = {sc0437:.4f}")
print(f"J1713: sigma_d/d = {11./1176*100:.3f}%, N_SO = {r1713['SO']:.1f}, sigma_chi = {sc1713:.4f}")
print("Paper says: J0437 ~0.01, J1713 ~0.5  [CHECK]")

# ======================================================================
# Line 382: Required sigma_d/d for single kpc pulsar
# ======================================================================
print("\n" + "=" * 72)
print("Line 382: Required sigma_d/d for sigma_chi = 1e-3 (single kpc pulsar)")
print("=" * 72)
r_kpc = quick_taylorf2(5e8, 5e8, 0.7, 0.7, 0, 0, 1e-6, 100, tau_kpc)
req = 1e-3 * abs(r_kpc["SO"]) / (0.7 * r_kpc["f_P"] * tau_kpc * YR)
print(f"Required sigma_d/d = {req * 100:.4f}%")
print("Paper says: ~0.003%  [CHECK]")

# ======================================================================
# Line 384: Array scaling
# ======================================================================
print("\n" + "=" * 72)
print("Line 384: Fisher sum over VLBI array")
print("=" * 72)
for N_psr, sig_dd, d_range, label in [
    (20, 0.01, (500, 3000), "20 VLBI, 1%"),
]:
    inv_var = 0.0
    for d_pc in np.linspace(d_range[0], d_range[1], N_psr):
        tau_yr = d_pc * PC / C_SI / YR
        sc_i, _ = sigma_chi_pulsar(5e8, 5e8, 0.7, 1e-6, 100, tau_yr, sig_dd)
        inv_var += 1.0 / sc_i**2
    sigma_arr = 1.0 / np.sqrt(inv_var)
    print(f"{label}: sigma_chi = {sigma_arr:.4f}")

# Factor-of-10 improvement in VLBI precision
inv_var = 0.0
for d_pc in np.linspace(500, 3000, 20):
    tau_yr = d_pc * PC / C_SI / YR
    sc_i, _ = sigma_chi_pulsar(5e8, 5e8, 0.7, 1e-6, 100, tau_yr, 0.001)
    inv_var += 1.0 / sc_i**2
sigma_10x = 1.0 / np.sqrt(inv_var)
print(f"20 VLBI, 0.1% (10x improvement): sigma_chi = {sigma_10x:.4f}")
print("Paper says: ~0.1 (1%), ~0.01 (0.1%)  [CHECK]")

# ======================================================================
# Spin phase error from mu Ares
# ======================================================================
print("\n" + "=" * 72)
print("Step 4 check: spin phase error from mu Ares")
print("=" * 72)
sigma_chi_muares = 3e-8
N_SO_J1713 = abs(r1713["SO"])
phase_err = 2 * np.pi * N_SO_J1713 * sigma_chi_muares / 0.7
print(f"delta_Phi = 2pi * N_SO * sigma_chi / chi = {phase_err:.2e} rad")
print("Paper says: ~1e-5 rad  [CHECK]")
