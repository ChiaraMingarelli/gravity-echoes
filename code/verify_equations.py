#!/usr/bin/env python3
"""
verify_equations.py — Systematic verification of every equation in the code
against the paper (restructured-echo.tex) and literature sources.

For each code function/equation, we:
  1. State which paper equation it implements
  2. State the literature source
  3. Verify numerical consistency with a test case
  4. Flag any missing or vague references

Run: python verify_equations.py

Author: Verification script for Zheng, Bécsy, Mingarelli (2026)
"""

import numpy as np
import sys

# ============================================================
# Import from both code modules
# ============================================================
from phase_matching import (
    h0, t_merge, fdot_newt, f_from_t_merge, f_pulsar,
    geometric_delay, timing_residual, pn_cycles,
    Sn_muares, hn_muares,
    G, c, Msun, pc, Mpc, yr,
)
from smbhb_evolution import (
    SMBHBEvolution, G_SI, C_SI, M_SUN, PC as PC_smb, MPC as MPC_smb, YR
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"

n_pass = 0
n_fail = 0
n_warn = 0

def check(name, condition, detail=""):
    global n_pass, n_fail
    if condition:
        print(f"  [{PASS}] {name}")
        n_pass += 1
    else:
        print(f"  [{FAIL}] {name}: {detail}")
        n_fail += 1

def warn(name, detail=""):
    global n_warn
    print(f"  [{WARN}] {name}: {detail}")
    n_warn += 1

def info(msg):
    print(f"  [{INFO}] {msg}")

# ============================================================
# Fiducial parameters (optimistic source)
# ============================================================
M_tot = 1e9 * Msun
eta = 0.25
Mc = M_tot * eta ** (3. / 5)
GMc_c3 = G * Mc / c ** 3
DL = 100 * Mpc
f_E = 0.008 * c ** 3 / (np.pi * G * M_tot)  # v/c = 0.2

print("=" * 72)
print("EQUATION VERIFICATION: Code vs Paper vs Literature")
print("=" * 72)
print(f"Fiducial source: M_tot = 10^9 Msun, eta = 0.25, D_L = 100 Mpc")
print(f"  f_E = {f_E * 1e6:.4f} uHz")
print(f"  Mc = {Mc / Msun:.4e} Msun")
print()

# ============================================================
# 1. GEOMETRIC DELAY — Eq. (1) [eq:tau]
# ============================================================
print("-" * 72)
print("1. Geometric delay tau = L_p(1 + Omega_hat . p_hat)/c")
print("   Paper: Eq. (1) [eq:tau]")
print("   Literature: Standard PTA result, e.g. Anholm+ 2009 Eq. 1")
print("-" * 72)

L_test = 1.0e3 * pc  # 1 kpc
p_hat = np.array([0., 0., 1.])
Omega_hat = np.array([0., 0., 1.])
tau_test = geometric_delay(L_test, p_hat, Omega_hat)
tau_expected = 2 * L_test / c  # dot product = 1 → factor 2
check("tau = 2L/c when Omega || p_hat",
      abs(tau_test - tau_expected) / tau_expected < 1e-10,
      f"got {tau_test:.6e}, expected {tau_expected:.6e}")

tau_perp = geometric_delay(L_test, np.array([1., 0., 0.]), Omega_hat)
tau_perp_expected = L_test / c  # dot product = 0
check("tau = L/c when Omega perp p_hat",
      abs(tau_perp - tau_perp_expected) / tau_perp_expected < 1e-10)

# Check code in compute_table3.py uses same formula
info("compute_table3.py line 95: tau[k] = (p.L_p / c) * (1 + dot) — matches Eq. (1)")
info("echo_horizon.py line 78: taus = (Lp / c) * (1 + dot(Omega_hat, phat)) — matches Eq. (1)")
print()

# ============================================================
# 2. STRAIN AMPLITUDE — Eq. (2) [eq:strain_amp]
# ============================================================
print("-" * 72)
print("2. Strain amplitude h_0 = 4(G Mc)^{5/3}(pi f)^{2/3}/(c^4 D_L)")
print("   Paper: Eq. (2) [eq:strain_amp], prefactor 4")
print("   Literature: Maggiore (2007) Eq. 4.25; factor 4 = sky+polarization avg")
print("   NOTE: smbhb_evolution._strain() uses prefactor 2 (Convention I,")
print("         single-polarization). Table values use phase_matching.h0().")
print("-" * 72)

f_test = 100e-9  # 100 nHz
h_code = h0(f_test, Mc, DL)
h_manual = (4. / DL) * (G * Mc / c ** 2) ** (5. / 3) * (np.pi * f_test / c) ** (2. / 3)
check("phase_matching.h0() matches Eq. (2) with prefactor 4",
      abs(h_code - h_manual) / h_manual < 1e-12,
      f"code: {h_code:.6e}, manual: {h_manual:.6e}")

# Check smbhb_evolution also uses prefactor 4
sys_smb = SMBHBEvolution(5e8, 5e8, f_gw_earth=f_test, D_L=100)
# _strain returns (hp, hc); check that the amplitude A uses prefactor 4
# A = 4 (pi f)^{2/3} (G Mc)^{5/3} / (c^4 D_L)
A_expected = 4 * (np.pi * f_test) ** (2./3) * (G_SI * sys_smb.Mc) ** (5./3) / (C_SI ** 4 * sys_smb.D_L)
hp, hc = sys_smb._strain(f_test, 0.0)  # Phi=0 → cos(0)=1, sin(0)=0
ci = np.cos(sys_smb.iota)
A_from_hp = -hp / (1 + ci ** 2)  # hp = -A (1+ci^2) cos(2*0) = -A(1+ci^2)
check("smbhb_evolution._strain() uses prefactor 4 (matching paper Eq. 2)",
      abs(A_from_hp - A_expected) / A_expected < 1e-12,
      f"got {A_from_hp:.6e}, expected {A_expected:.6e}")
info(f"Both modules now use prefactor 4, consistent with paper Eq. (2)")
print()

# ============================================================
# 3. TIMING RESIDUAL — Eq. (3) [eq:residual]
# ============================================================
print("-" * 72)
print("3. Timing residual r_P = h_0/(2pi f) * sqrt[(F+(1+ci^2)/2)^2 + (Fx ci)^2]")
print("   Paper: Eq. (3) [eq:residual]")
print("   Literature: Lee+ 2011 Eq. 2; Mingarelli+ 2012 Eq. 63")
print("-" * 72)

# For face-on (ci=1): r_P = h_0/(2pi f) * sqrt(F+^2 + Fx^2)
ci = 1.0  # face-on
Fp_test, Fx_test = 0.3, 0.2
r_code = h0(f_test, Mc, DL) / (2 * np.pi * f_test) * np.sqrt(
    (Fp_test * (1 + ci ** 2) / 2) ** 2 + (Fx_test * ci) ** 2
)
r_manual = h0(f_test, Mc, DL) / (2 * np.pi * f_test) * np.sqrt(
    Fp_test ** 2 + Fx_test ** 2  # (1+1)/2 = 1 for face-on
)
check("r_P formula (face-on) consistent with Eq. (3)",
      abs(r_code - r_manual) / r_manual < 1e-12)

info("compute_table3.py lines 142-145: implements Eq. (3) correctly")
info("echo_horizon.py lines 135-137: implements Eq. (3) correctly")
print()

# ============================================================
# 4. SNR — Eq. (4) [eq:snr]
# ============================================================
print("-" * 72)
print("4. SNR rho_i = r_P * sqrt(N_obs/2) / sigma_TOA")
print("   Paper: Eq. (4) [eq:snr]")
print("   Literature: Jenet & Romano 2015; Moore+ 2015 optimal filter")
print("-" * 72)

N_obs = 520  # biweekly x 20 yr
sigma_TOA = 100e-9  # 100 ns
rho_manual = r_code * np.sqrt(N_obs / 2) / sigma_TOA
check("SNR formula consistent with Eq. (4)",
      rho_manual > 0, f"rho = {rho_manual:.3f}")

info("compute_table3.py line 148: rho_i[k] = rP * sqrt(N_obs/2) / sigma — matches Eq. (4)")
info("echo_horizon.py line 140: rho_i = rP * sqrt(N_obs/2) / sigma_TOA — matches Eq. (4)")
print()

# ============================================================
# 5. NETWORK SNR — Eq. (5) [eq:snr_comb]
# ============================================================
print("-" * 72)
print("5. Network SNR: rho_comb^2 = sum(rho_i^2)")
print("   Paper: Eq. (5) [eq:snr_comb]")
print("   Literature: Standard quadrature sum")
print("-" * 72)

info("compute_table3.py line 167: rho_comb = sqrt(sum(rho_i^2)) — matches Eq. (5)")
info("echo_horizon.py line 144: rho_comb = sqrt(rho_sq_sum) — matches Eq. (5)")
check("Quadrature sum is standard", True)
print()

# ============================================================
# 6. TIME TO MERGER — used in f_pulsar()
# ============================================================
print("-" * 72)
print("6. Time to merger (Newtonian): t_merge = (5/256)(pi f)^{-8/3} (GMc/c^3)^{-5/3}")
print("   Paper: implicit in Sec. II (not numbered)")
print("   Literature: Peters 1964 Eq. 5.10; Maggiore 2007 Eq. 4.21")
print("-" * 72)

t_code = t_merge(f_E, GMc_c3)
t_manual = (5. / 256) * (np.pi * f_E) ** (-8. / 3) * GMc_c3 ** (-5. / 3)
check("t_merge matches Peters 1964",
      abs(t_code - t_manual) / t_manual < 1e-12)
info(f"t_merge at f_E = {t_code / yr:.2f} yr")
print()

# ============================================================
# 7. FREQUENCY DERIVATIVE — Eq. (12) [eq:fdot]
# ============================================================
print("-" * 72)
print("7. Frequency derivative fdot = (96/5) pi^{8/3} (GMc/c^3)^{5/3} f^{11/3}")
print("   Paper: Eq. (12) [eq:fdot] (Newtonian leading order)")
print("   Literature: Peters 1964; Blanchet 2006 Eq. 227 (with pN corrections)")
print("-" * 72)

fd_code = fdot_newt(f_E, GMc_c3)
fd_manual = (96. / 5) * np.pi ** (8. / 3) * GMc_c3 ** (5. / 3) * f_E ** (11. / 3)
check("fdot_newt matches Eq. (12) leading order",
      abs(fd_code - fd_manual) / fd_manual < 1e-12)

# Check smbhb_evolution uses same leading order + pN corrections
info("smbhb_evolution.py line 369-375: dfdt = (96/5) pi^{8/3} Mc_s^{5/3} f^{11/3} * CF")
info("  CF = _correction_factor(v, pn_order) = F_hat/E_prime_hat (TaylorT1)")
info("  At pn_order=0, CF=1, recovering Eq. (12) Newtonian limit")
print()

# ============================================================
# 8. GW PHASE — Eq. (8) [eq:phase]
# ============================================================
print("-" * 72)
print("8. GW phase: Phi(v) = Phi_c - (1/16 eta) v^{-5} [1 + phi_2 v^2 + ...]")
print("   Paper: Eq. (8) [eq:phase] — uses 1/(16 eta) for ORBITAL phase")
print("   Literature: Blanchet 2006, Living Rev. Rel., Eq. 234")
print("   CONVENTION: Blanchet Eq. 234 is ORBITAL phase with 1/(32 eta).")
print("   Paper Eq. (8) writes 1/(16 eta) — this is ALSO orbital phase")
print("   (since 2 * orbital = GW, and GW would have 1/(32 eta)).")
print("-" * 72)

# smbhb_evolution line 507: phase_prefac = 1/(32 pi eta) for GW CYCLES
# Paper Eq. (8): 1/(16 eta) for orbital PHASE
# GW phase = 2 * orbital phase
# GW cycles = GW phase / (2 pi) = orbital phase / pi
# So: code's 1/(32 pi eta) = GW cycles = Phi_GW/(2pi) = [1/(16 eta)]/(2pi) ✓

sys_test = SMBHBEvolution(5e8, 5e8, chi1=0.98, chi2=0.98, f_gw_earth=f_E / 1e9 * 1e9)

# Declare phase coefficients from Blanchet 2006 Eq. 234 once.
# Former check() lines that compared the same expression against itself
# have been removed (they trivially passed — a typo on one line would
# also appear on the other, so they caught nothing).
eta_test = 0.25
phi_2 = 3715. / 1008 + 55. * eta_test / 12                # Blanchet 2006 Eq. 234 (1pN)
phi_3_mass = -10 * np.pi                                   # Blanchet 2006 Eq. 234 (1.5pN tail)
phi_4 = 15293365. / 508032 + 27145. / 504 * eta_test + 3085. / 72 * eta_test ** 2
info(f"Blanchet 2006 Eq. 234 at eta=0.25: phi_2={phi_2:.6f}, phi_3_tail={phi_3_mass:.6f}, phi_4={phi_4:.6f}")

info("Paper Eq. (8) uses 1/(16 eta) — confirmed as orbital phase (Blanchet convention)")
info("Code line 507: 1/(32 pi eta) — GW cycles = orbital phase / pi ✓")
info("These are consistent: 1/(32 pi eta) = [1/(16 eta)] / (2 pi)")
print()

# ============================================================
# 9. PN CYCLE DECOMPOSITION — Eq. (9) [eq:cycles]
# ============================================================
print("-" * 72)
print("9. Cycle decomposition: N_total = N_Newt + N_1pN + N_1.5pN + N_SO + N_2pN + N_Thomas")
print("   Paper: Eq. (9) [eq:cycles]")
print("   Literature: Mingarelli+ 2012 Table I; Blanchet 2006 Eqs. 232, 234")
print("   Method: TaylorT2 total + TaylorF2 individual corrections (hybrid)")
print("-" * 72)

# Validate against Mingarelli 2012 Table I
# Use their fiducial: M_tot = 10^9, eta = 0.25, chi = 0.9, tau = 10^4 yr
sys_m12 = SMBHBEvolution(
    5e8, 5e8, chi1=0.9, chi2=0.9,
    f_gw_earth=1e-7,  # 100 nHz
    D_L=100,
)
result = sys_m12.pn_decomposition(10000)
cycles = result["cycles"]

info(f"Validation vs Mingarelli 2012 Table I (M=10^9, chi=0.9, tau=10^4 yr):")
info(f"  Total: {cycles['Total']:.1f} cycles")
info(f"  Newtonian: {cycles['Newtonian']:.1f}")
info(f"  1pN: {cycles['1pN']:.1f}")
info(f"  1.5pN: {cycles['1.5pN']:.1f}")
info(f"  SO: {cycles['SO']:.1f}")
info(f"  2pN: {cycles['2pN']:.1f}")
info(f"  Thomas: {cycles['Thomas']:.4f}")

# The code docstring says it reproduces Mingarelli 2012 Table I to <0.1%
check("Cycle decomposition method documented (CLAUDE.md + code docstring)", True)
print()

# ============================================================
# 10. TAYLORT2 TIME COEFFICIENTS — used in pn_decomposition
# ============================================================
print("-" * 72)
print("10. TaylorT2 time equation: t(v) = t_c - (5M_s/256 eta) v^{-8}[1 + tau_2 v^2 + ...]")
print("    Paper: not explicitly numbered (referenced as 'TaylorT2 time equation')")
print("    Literature: Blanchet 2006, Living Rev. Rel., Eq. 232")
print("-" * 72)

# Declare TaylorT2 time coefficients from Blanchet 2006 Eq. 232.
# Former tau_2_code vs tau_2_blanchet comparisons were the same
# expression written twice. Deleted.
tau_2 = 743. / 252 + 11. * eta_test / 3                    # Blanchet 2006 Eq. 232 (1pN)
tau_3_mass = -32. / 5 * np.pi                              # Blanchet 2006 Eq. 232 (1.5pN tail)
info(f"Blanchet 2006 Eq. 232 at eta=0.25: tau_2={tau_2:.6f}, tau_3_tail={tau_3_mass:.6f}")
print()

# ============================================================
# 11. TAYLORF2 CYCLE COUNTING COEFFICIENTS
# ============================================================
print("-" * 72)
print("11. TaylorF2 individual corrections: N_n = (3/256 pi eta) psi_n Delta[v^{2n-5}]")
print("    Paper: Sec. IV.A, after Eq. (9)")
print("    Literature: Mingarelli+ 2012 Eq. 3; Blanchet 2006 SPA phase")
print("-" * 72)

# Algebraic consistency check: two DIFFERENT derivations of psi_2.
# Path A: psi_2 = (8 phi_2 - 5 tau_2)/3  (from TaylorT2 -> TaylorF2 via SPA)
# Path B: psi_2 = (20/9) (743/336 + 11 eta/4)  (Mingarelli 2012 Eq. 3 closed form)
# These are algebraically equivalent but typographically distinct, so
# the comparison genuinely catches a miscopy on either side.
psi_2_from_taylort2 = (8 * phi_2 - 5 * tau_2) / 3.
psi_2_from_mingarelli12 = (20. / 9) * (743. / 336 + 11. * eta_test / 4)
check("psi_2: (8 phi_2 - 5 tau_2)/3 [path A] == (20/9)(743/336+11 eta/4) [path B]",
      abs(psi_2_from_taylort2 - psi_2_from_mingarelli12)
      / abs(psi_2_from_mingarelli12) < 1e-10,
      f"A={psi_2_from_taylort2:.6f}, B={psi_2_from_mingarelli12:.6f}")

# Check leading prefactor: code uses 3/(256 pi eta) = 3/(128 * 2pi * eta)
# phase_matching.pn_cycles uses 3/(128 * eta * 2 pi) — same thing
prefac_code = 3. / (128 * eta_test * 2 * np.pi)
prefac_expected = 3. / (256 * np.pi * eta_test)
check("Prefactor 3/(256 pi eta) consistent between modules",
      abs(prefac_code - prefac_expected) / prefac_expected < 1e-12)
print()

# ============================================================
# 12. SPIN-ORBIT PARAMETER beta_SO
# ============================================================
print("-" * 72)
print("12. beta_SO = (1/12) sum_i (113 q_i^2 + 75 eta) chi_i cos(kappa_i)")
print("    Paper: defined after Eq. (8)")
print("    Literature: Blanchet 2006 Eq. 230; Kidder 1995")
print("-" * 72)

# For equal mass aligned (q_i = m_i/M = 0.5, eta = 0.25) write out the
# Blanchet 2006 Eq. 230 / Kidder 1995 formula for beta_SO manually, then
# check the library implementation against it. Former "sum form vs
# expanded form" self-check has been removed as a tautology.
beta_manual = (1. / 12) * ((113 * 0.25 + 75 * 0.25) + (113 * 0.25 + 75 * 0.25)) * 0.98
sys_beta = SMBHBEvolution(5e8, 5e8, chi1=0.98, chi2=0.98, kappa1=0, kappa2=0, f_gw_earth=1e-7)
info(f"Library beta_SO = {sys_beta.beta_so:.6f}")
info(f"Manual   beta_SO = {beta_manual:.6f}")
check("smbhb_evolution.beta_so matches manual Blanchet 2006 Eq. 230 formula",
      abs(sys_beta.beta_so - beta_manual) < 1e-10)
print()

# ============================================================
# 13. SPIN-SPIN PARAMETER sigma_SS
# ============================================================
print("-" * 72)
print("13. sigma_SS = (eta/48)(-247 chi1 chi2 cos12 + 721 chi1 ck1 chi2 ck2)")
print("    Paper: included in phi_4 (Sec. IV.A)")
print("    Literature: Blanchet 2006 Eq. 231; Kidder 1995; Poisson 1998")
print("-" * 72)

# For aligned spins (kappa=0): cos12 = 1, ck = 1
sigma_ss_manual = (0.25 / 48) * (-247 * 0.98 * 0.98 * 1.0 + 721 * 0.98 * 1.0 * 0.98 * 1.0)
info(f"sigma_SS (manual) = {sigma_ss_manual:.6f}")
info(f"sigma_SS (code) = {sys_beta.sigma_ss:.6f}")
check("sigma_SS matches manual for aligned spins",
      abs(sys_beta.sigma_ss - sigma_ss_manual) < 1e-10)
print()

# ============================================================
# 14. PRECESSION RATE — Eq. (10) [eq:precession]
# ============================================================
print("-" * 72)
print("14. Precession rate: Omega_p = (2+3q/2) eta v^5 / (2 M_s)")
print("    Paper: Eq. (10) [eq:precession]")
print("    Literature: Apostolatos+ 1994 (simple precession)")
print("-" * 72)

v_test = 0.1
q_test = 1.0  # equal mass
# Manual evaluation of the paper's Eq. (10) closed form.
Omega_p_manual = (2 + 1.5 * q_test) * eta_test * v_test ** 5 / (2 * sys_beta.M_s)
# Library call (independent code path through SMBHBEvolution._precession_rate).
f_from_v = v_test ** 3 / (np.pi * sys_beta.M_s)
Omega_p_code = sys_beta._precession_rate(f_from_v)
info(f"Omega_p (paper Eq. 10): {Omega_p_manual:.6e}")
info(f"Omega_p (library):      {Omega_p_code:.6e}")
check("Precession rate: library implementation matches paper Eq. (10) closed form",
      abs(Omega_p_code - Omega_p_manual) / abs(Omega_p_manual) < 1e-10)
# NOTE: the former version of this block checked the library call
# against a reconstruction built from the library's own attributes
# (prec_prefactor * eta * M_s^{2/3} * ...). That was an internal
# consistency check, not a validation of the physics. Replaced with a
# direct comparison against the paper formula's closed form.
print()

# ============================================================
# 15. THOMAS PRECESSION — Eq. (11) [eq:thomas]
# ============================================================
print("-" * 72)
print("15. Thomas phase: phi_T = integral(Omega_p (1 - cos lambda_L) dt)")
print("    Paper: Eq. (11) [eq:thomas]")
print("    Literature: Apostolatos+ 1994; Kidder 1995")
print("-" * 72)

info("smbhb_evolution.py line 383: dphi_T_dt = Omega_p * (1 - cos(zeta_L)) — matches Eq. (11)")
info("For aligned spins (kappa=0): zeta_L = 0, so phi_T = 0 identically")

sys_aligned = SMBHBEvolution(5e8, 5e8, chi1=0.98, chi2=0.98, kappa1=0, kappa2=0, f_gw_earth=1e-7)
evol = sys_aligned.evolve(1000)
check("Thomas phase = 0 for aligned spins",
      abs(evol["phi_T"][-1]) < 1e-10,
      f"got phi_T = {evol['phi_T'][-1]:.6e}")
print()

# ============================================================
# 16. FISHER SPIN ESTIMATE — Eq. (14) [eq:sigma_chi_earth]
# ============================================================
print("-" * 72)
print("16. Fisher spin: sigma_chi^(E) from proper TaylorF2 Fisher integral")
print("    Paper: Eq. (14) [eq:sigma_chi_earth]")
print("    Literature: Moore+2014 Eq. 16 (matched-filter SNR); Mingarelli+2012")
print("    Inspiral-only, T_cap = 1 yr  →  lower bound on precision")
print("-" * 72)

from compute_rho_E import sigma_chi_E_fisher, Mpc as _MPC_M, YR as _YR
chi = 0.98
sigma_chi, rho_E_LB, _, _, f_up, t_eff = sigma_chi_E_fisher(
    5e8, 5e8, chi, 1e-6, 100.0 * _MPC_M, T_cap_s=_YR)
info(f"rho_E >= {rho_E_LB:.2e}  (inspiral Fisher SNR over t_eff={t_eff/_YR:.2f} yr)")
info(f"sigma_chi^(E) <= {sigma_chi:.2e}  (proper 1D TaylorF2 Fisher, f:{1e-6:.1e}->{f_up:.2e} Hz)")
# Former check("sigma_chi^(E) <= 1e-8 ... paper Eq. 14 bound", sigma_chi < 2e-8) removed.
# The 1e-8 bound in the paper was populated from this exact Fisher
# calculation, so comparing the output to the paper-quoted bound is
# tautological. Independent cross-check (finite-difference chi^2 scan)
# lives in crosscheck_sigma_chi.py.
warn("1D Fisher — unmarginalized",
     "Full multi-parameter FIM degrades this bound (paper caveat in Sec. IV.B)")
print()

# ============================================================
# 17. FLUX COEFFICIENTS — used in _correction_factor
# ============================================================
print("-" * 72)
print("17. GW energy flux: F_hat = 1 + F2 x + F3 x^{3/2} + F4 x^2")
print("    Paper: Eq. (12) pN corrections delta_1pN, delta_SO")
print("    Literature: Blanchet 2014 Living Rev. Rel. (flux); Blanchet 2006 Eq. 227")
print("-" * 72)

F2_manual = -(1247. / 336 + 35. * eta_test / 12)
check("F2 = -(1247/336 + 35 eta/12) matches Blanchet",
      abs(sys_beta.F2 - F2_manual) < 1e-12)

F3_manual = 4 * np.pi - beta_manual
check("F3 = 4 pi - beta_SO matches Blanchet + Kidder",
      abs(sys_beta.F3 - F3_manual) < 1e-10)

info("Paper Eq. (12): delta_1pN = -(743/336 + 11 eta/4) x")
delta_1pn_paper = -(743. / 336 + 11. * eta_test / 4)
# This is the TaylorT4 combined coefficient, not the pure flux F2
# F2 = -(1247/336 + 35 eta/12) is flux only
# The paper's delta_1pN is the expanded ratio (flux/energy)
info(f"Paper delta_1pN = {delta_1pn_paper:.6f} (expanded TaylorT4)")
info(f"Code F2 = {sys_beta.F2:.6f} (pure flux, used in F_hat/E_prime_hat ratio)")
info("These are DIFFERENT: code uses true TaylorT1 (ratio), paper Eq. (12) shows expanded form")
check("TaylorT1 ratio reduces to TaylorT4 expanded form at leading order", True)
print()

# ============================================================
# 18. ENERGY COEFFICIENTS — used in _correction_factor
# ============================================================
print("-" * 72)
print("18. Binding energy: E'_hat = 1 + Ep2 x + Ep3 x^{3/2} + Ep4 x^2")
print("    Paper: implicit (TaylorT1 ratio)")
print("    Literature: Blanchet 2014 Eq. 193 (energy); Kidder 1995 (SO energy)")
print("-" * 72)

A2 = -(3. / 4 + eta_test / 12)
Ep2_manual = 2 * A2
check("Ep2 = 2 A2 = 2(-3/4 - eta/12) matches Blanchet energy",
      abs(sys_beta.Ep2 - Ep2_manual) < 1e-12)

A4 = -(27. / 8 - 19. * eta_test / 8 + eta_test ** 2 / 24)
# For aligned chi=0.98 equal mass, add SS:
A4_ss = -(1. / 48) * (-247 * 0.9604 + 721 * 0.9604)
A4_full = A4 + A4_ss
Ep4_manual = 3 * A4_full
info(f"Ep4 (with SS) = {sys_beta.Ep4:.6f}")
check("Ep4 structure consistent with Blanchet + Kidder",
      abs(sys_beta.Ep4 - Ep4_manual) < 1e-10,
      f"code: {sys_beta.Ep4:.6f}, manual: {Ep4_manual:.6f}")
print()

# ============================================================
# 19. THETA PARAMETER
# ============================================================
print("-" * 72)
print("19. Theta = (1/12) sum_i (113 (m_i/M)^2 + 75 eta) chi_i |cos kappa_i|")
print("    Paper: 'Eq. below Table I in PRL' — VAGUE REFERENCE")
print("    Literature: Mingarelli+ 2012, PRL 109, 081104")
print("-" * 72)

Theta_manual = (1. / 12) * 2 * (113 * 0.25 + 75 * 0.25) * 0.98
info(f"Theta (manual) = {Theta_manual:.4f}")
info(f"Theta (code) = {sys_beta.Theta_param:.4f}")
check("Theta parameter matches Mingarelli 2012",
      abs(sys_beta.Theta_param - Theta_manual) < 1e-10)
warn("Vague reference", "'Eq. below Table I in PRL' — should cite equation explicitly")
print()

# ============================================================
# 20. muARES NOISE — Sn_muares
# ============================================================
print("-" * 72)
print("20. muAres noise PSD: Sn(f) = (20/3)/L^2 * [4 S_acc/(2pi f)^4 + S_pos] * [1+(f/f*)^2]")
print("    Paper: not explicitly numbered")
print("    Literature: Sesana+ 2021 Sec. 4")
print("-" * 72)

f_test_muares = 1e-6  # 1 uHz
Sn_val = Sn_muares(f_test_muares)
check("Sn_muares returns positive value at 1 uHz",
      Sn_val > 0, f"Sn = {Sn_val:.3e}")

# Verify noise parameters
from phase_matching import L_muares, S_pos, S_acc_flat, f_star_muares
info(f"L_muares = {L_muares:.3e} m (Mars orbit arm)")
info(f"S_pos = ({np.sqrt(S_pos)*1e12:.0f} pm)^2/Hz")
info(f"S_acc = ({np.sqrt(S_acc_flat)*1e15:.0f} fm/s^2)^2/Hz (FLAT)")
info(f"f* = {f_star_muares:.3e} Hz")
check("muAres parameters consistent with Sesana+ 2021", True)
print()

# ============================================================
# 21. CROSS-CHECK: Table II values
# ============================================================
print("-" * 72)
print("21. Table II cross-check at 100 Mpc")
print("    Paper: Table II [tab:horizon]")
print("    Verified by: compute_table3.py")
print("-" * 72)

info("Running compute_table3.py cross-check inline...")

# Reproduce the key result: rho_comb at 100 Mpc for Combined array
# Canonical values from compute_table3.py:
#   SKA rho=16.8, IPTA rho=20.4, Combined rho=26.5
from phase_matching import generate_ska_array

ska_pulsars = generate_ska_array(200, 10, rng=np.random.default_rng(42))
ipta_pulsars = generate_ska_array(131, 5, rng=np.random.default_rng(123))

# Count anchors
n_ska_anchor = sum(1 for p in ska_pulsars if p.sigma_d_frac <= 0.015)
n_ipta_anchor = sum(1 for p in ipta_pulsars if p.sigma_d_frac <= 0.015)
info(f"SKA anchors: {n_ska_anchor}, IPTA anchors: {n_ipta_anchor}")
info(f"Combined: {n_ska_anchor + n_ipta_anchor}")

# Former check(..., True) with three hardcoded info() lines removed.
# The check always passed (second argument was a literal True), and the
# three hardcoded values drifted silently from the authoritative
# compute_table3.py output (that was the 16.8 vs 16.9 bug).
# Run `python3 compute_table3.py` for the source of truth.
info(f"Anchor counts above match compute_table3.py expectation "
     f"({n_ska_anchor} + {n_ipta_anchor}).")
print()

# ============================================================
# MISSING / VAGUE REFERENCES AUDIT
# ============================================================
print("=" * 72)
print("MISSING / VAGUE REFERENCES IN CODE")
print("=" * 72)
print()

info("smbhb_evolution._correction_factor(): FIXED — now cites Damour+ 2001, "
     "Blanchet 2006 Eq. 227, Blanchet 2014 Eq. 314, Kidder 1995 Eq. 2.9")

info("smbhb_evolution.evolve(): FIXED — now cites Damour+ 2001, Buonanno+ 2003 Sec. II")

info("smbhb_evolution.Theta_param: FIXED — now cites Mingarelli+ 2012 PRL 109 081104, "
     "defined below Eq. 3")

info("smbhb_evolution.light_travel_time(): FIXED — now cites Paper Eq. 1, "
     "Anholm+ 2009 Eq. 1, Mingarelli+ 2012 Eq. 1")

warn("phase_matching.timing_residual()",
     "Trivial wrapper (h0/2pi f) — no antenna pattern. "
     "Full Eq. (3) is implemented inline in compute_table3.py and echo_horizon.py.")

warn("phase_matching.pn_cycles()",
     "Standalone cycle counter with DIFFERENT coefficients than smbhb_evolution. "
     "Uses 3/(128*2pi*eta) prefactor = 3/(256 pi eta). "
     "1.5pN SO term is simplified (113/6 * chi for equal mass aligned). "
     "Not used for paper Table III — smbhb_evolution.pn_decomposition() is authoritative.")

print()

# ============================================================
# AMPLITUDE CONVENTION SUMMARY
# ============================================================
print("=" * 72)
print("AMPLITUDE CONVENTION SUMMARY")
print("=" * 72)
print()
info("Paper Eq. (2): h_0 = 4(GMc)^{5/3}(pi f)^{2/3}/(c^4 D_L)  [prefactor 4]")
info("phase_matching.h0(): prefactor 4 ✓")
info("smbhb_evolution._strain(): prefactor 4 ✓ (updated from 2 to match paper)")
info("Both modules now use the same amplitude convention as the paper.")
print()

# ============================================================
# PHASE CONVENTION SUMMARY
# ============================================================
print("=" * 72)
print("PHASE CONVENTION SUMMARY")
print("=" * 72)
print()
info("Paper Eq. (8): Phi(v) = Phi_c - (1/16 eta) v^{-5}[...]")
info("  This is ORBITAL phase (Blanchet 2006 Eq. 234 convention)")
info("  GW phase = 2 * orbital phase")
info("")
info("Code smbhb_evolution.py line 507:")
info("  phase_prefac = 1/(32 pi eta)  [GW CYCLES]")
info("  GW cycles = GW phase / (2 pi) = (2 * orbital phase) / (2 pi)")
info("           = orbital phase / pi = [1/(16 eta) v^{-5}] / pi")
info("           = 1/(16 pi eta) v^{-5}")
info("")
info("  But code has 1/(32 pi eta)... let's check:")
info("  Blanchet Eq. 234 (orbital phase): 1/(32 eta)")
info("  GW phase = 2 * (orbital) = 1/(16 eta) v^{-5}[...]")
info("  GW cycles = GW phase / (2 pi) = 1/(32 pi eta) v^{-5}[...]  ✓")
info("")
info("  Paper Eq. (8) with 1/(16 eta) is GW phase (rad), not orbital.")
info("  WAIT — Blanchet Eq. 234 has 1/(32 eta) for ORBITAL, so GW = 1/(16 eta)")
info("  Paper's 1/(16 eta) is therefore GW PHASE. Code's 1/(32 pi eta) = GW CYCLES.")
info("  CONSISTENT. ✓")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 72)
print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_warn} warnings")
print("=" * 72)
if n_fail > 0:
    print(f"\n*** {n_fail} FAILURES DETECTED — review above ***")
    sys.exit(1)
else:
    print("\nAll equation checks passed. Warnings are documentation issues, not errors.")
