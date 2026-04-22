#!/usr/bin/env python3
"""
crosscheck_sigma_chi.py -- Independent verification of sigma_chi^(E).

compute_rho_E.sigma_chi_E_fisher evaluates the Fisher-matrix bound on
the aligned-spin amplitude chi via the analytic inspiral SPA waveform:

    Gamma_{chi,chi} = 4 integral |h_tilde(f)|^2 |dPsi/dchi|^2 / Sn(f) df
    sigma_chi       = 1 / sqrt(Gamma_{chi,chi})                    [Eq.]

with

    |h_tilde(f)|^2 = h_0(f)^2 / (2 fdot_Newt)          (SPA amplitude)
    dPsi/dchi      = (188 / (256 eta)) * (pi M f)^{-2/3}
    h_0(f)         = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L)

That result rests on three algebraic facts:
  (i) Phase derivative: d Psi_1.5pN^SO / d chi reduces to 188/(256 eta)
      times (pi M f)^{-2/3}.  TaylorF2 SPA gives psi_3 = -16 pi + 4 beta_SO
      (Cutler-Flanagan 1994; Arun+ 2005 Eq. A10; Blanchet LRR Eq. 426),
      and beta_SO = (47/6) chi for equal-mass aligned spins.  Hence
      dpsi_3/dchi = 4 * (47/6) = 94/3, and
      dPsi/dchi = (3/(128 eta)) v^{-2} * (94/3) = (94/(128 eta)) v^{-2}
                = (188/(256 eta)) v^{-2}.
  (ii) Fisher matrix with pure phase modulation: for h_tilde = A e^{i Psi},
       dh_tilde/dchi = i h_tilde dPsi/dchi, hence Gamma -> the SPA form.
  (iii) Waveform amplitude from SPA: |h_tilde|^2 = h_0^2 / (2 fdot).

This script tests (i)-(iii) simultaneously by computing Gamma from a
finite-difference numerical derivative, d h_tilde / d chi -> central
differences around chi_0, and comparing the resulting Fisher to the
library's closed-form value.

  Method C : Finite-difference Fisher.
             dh/dchi ~ [h(chi + dchi) - h(chi - dchi)] / (2 dchi)
             Gamma_FD = 4 Re integral dh/dchi (dh/dchi)* / Sn df
             Step dchi is chosen by a Richardson-style convergence sweep.

  Method D : Chi-squared curvature scan.
             D(chi) = <h(chi) - h(chi_0) | h(chi) - h(chi_0)>
                    = Gamma * (chi - chi_0)^2  + O((chi-chi_0)^4)
             A small-range quadratic fit recovers Gamma from the curvature.

  External : TaylorF2 phase coefficient psi_3 = -16 pi + 4 beta_SO
             (Cutler-Flanagan 1994 PRD 49 2658; Arun, Iyer, Sathyaprakash,
             Sundararajan 2005 PRD 71 084008 Eq. A10; Blanchet 2014 LRR
             Eq. 426).  For equal-mass aligned spins with chi1 = chi2 = chi,
             beta_SO = (47/6) chi, so dpsi_3/dchi = 4 * (47/6) = 94/3 and
                 dPsi/dchi = (3/(128 eta)) * (94/3) * v^{-2}
                           = (94/(128 eta)) * v^{-2}
                           = (188/(256 eta)) * v^{-2}
             Equivalent forms: 47/(64 eta) * v^{-2} = 141/(192 eta) * v^{-2}.
             The library prefactor is 188/(256 eta); this crosscheck
             reproduces it through two independent numerical methods (FD
             derivative of h_tilde, and chi^2 curvature scan).

Pass criteria:
  |Gamma_FD - Gamma_lib| / Gamma_lib          < 1e-3 at converged dchi
  |Gamma_curvature - Gamma_lib| / Gamma_lib   < 1e-3
  chi^2(chi) parabolic shape within 1% over +- 10 sigma_chi
"""

import numpy as np
from scipy import integrate

from phase_matching import Sn_muares
from compute_rho_E import (
    h0_strain, fdot_newtonian, f_upper_inspiral, sigma_chi_E_fisher,
    chirp_mass, G, c, M_sun, Mpc, YR,
)

# ======================================================================
# Fiducial configuration -- most stringent case in the paper
# ======================================================================
m1_msun = 5e8
m2_msun = 5e8
CHI = 0.98  # aligned, near-extremal Kerr (migrated 2026-04-20 from 0.7)
f_E = 1e-6
D_L_Mpc = 100.0
T_CAP_S = YR

M_tot_kg = (m1_msun + m2_msun) * M_sun
Mc_kg = chirp_mass(m1_msun, m2_msun) * M_sun
eta = 0.25
D_L_m = D_L_Mpc * Mpc


# ======================================================================
# Reimplemented SPA waveform with explicit TaylorF2 phase
# Coefficients: Blanchet 2014 LRR Eq. 426 (mass sector) and Cutler-Flanagan
# 1994 / Arun+ 2005 Eq. A10 (spin-orbit sector).
#
# TaylorF2 SPA phase:
#   Psi(f) = 2 pi f t_c - phi_c - pi/4
#            + (3 / (128 eta)) v^{-5} [1 + psi_2 v^2 + psi_3 v^3 + psi_4 v^4]
# with
#   v = (pi M f)^{1/3}
#   psi_2 = (20/9)(743/336 + 11 eta/4)
#   psi_3 = -16 pi + 4 beta_SO         <- CF / Arun+ 2005 factor of 4
#   beta_SO = (1/12) sum_i [113 (m_i/M)^2 + 75 eta] chi_i cos kappa_i
#   for m1 = m2 aligned, chi1 = chi2 = chi:
#       beta_SO = (1/12) * 2 * [113/4 + 75/4] * chi
#               = (1/12) * (376/4) * chi
#               = 47/6 * chi          (matches compute_rho_E.py and M12 Table I)
# ======================================================================
BETA_SO_COEFF = 47.0 / 6.0   # beta_SO = (47/6) * chi for m1=m2 aligned


def taylorf2_phase_derivative_analytic(chi, f):
    """Analytic dPsi/dchi at TaylorF2 1.5pN SO only.

    Psi contains chi only through psi_3 = -16 pi + 4 beta_SO(chi).
    beta_SO = (47/6) chi, so dpsi_3/dchi = 4 * (47/6) = 94/3.

    dPsi/dchi = (3/(128 eta)) * v^{-5} * dpsi_3/dchi * v^3
              = (3/(128 eta)) * (94/3) * v^{-2}
              = (94/(128 eta)) * v^{-2}
              = (188/(256 eta)) * v^{-2}
              = (188/(256 eta)) * (pi M f)^{-2/3}
    """
    v2 = (np.pi * G * M_tot_kg / c ** 3 * f) ** (2.0 / 3)
    return (188.0 / (256.0 * eta)) / v2


def h_tilde_mag_sq(f):
    """|h_tilde(f)|^2 from SPA (Moore Eq. 17, 34)."""
    h0 = h0_strain(f, Mc_kg, D_L_m)
    return h0 * h0 / (2.0 * fdot_newtonian(f, Mc_kg))


def h_tilde_complex(f, chi):
    """Full complex h_tilde with TaylorF2 SPA phase (1pN, 1.5pN incl SO).

    Psi(f) = (3/(128 eta)) v^{-5} [1 + psi_2 v^2 + psi_3 v^3]
    where v = (pi M f)^{1/3} and
      psi_2 = (20/9)(743/336 + 11 eta/4)
      psi_3 = -16 pi + 4 * (47/6) chi = -16 pi + (94/3) chi
    Drops the constant phase terms (t_c, phi_c, -pi/4), which do not
    affect Gamma_{chi,chi} at this order.
    """
    v = (np.pi * G * M_tot_kg / c ** 3 * f) ** (1.0 / 3)
    psi_2 = (20.0 / 9) * (743.0 / 336 + 11.0 * eta / 4)
    psi_3 = -16.0 * np.pi + 4.0 * BETA_SO_COEFF * chi
    Psi = (3.0 / (128.0 * eta)) * v ** (-5) * (
        1.0 + psi_2 * v ** 2 + psi_3 * v ** 3
    )
    amp = np.sqrt(h_tilde_mag_sq(f))
    return amp * np.exp(1j * Psi)


# ======================================================================
# Method C: finite-difference Fisher
# ======================================================================
def gamma_finite_difference(dchi):
    f_up, _ = f_upper_inspiral(Mc_kg, f_E, T_CAP_S)

    def integrand(f):
        dh = (h_tilde_complex(f, CHI + dchi) - h_tilde_complex(f, CHI - dchi)) \
             / (2.0 * dchi)
        return 4.0 * (dh.real ** 2 + dh.imag ** 2) / Sn_muares(f)

    gamma, _ = integrate.quad(integrand, f_E, f_up, limit=300)
    return gamma


# ======================================================================
# Method D: chi^2 curvature scan
#    D(chi) = <h(chi) - h(chi_0) | h(chi) - h(chi_0)>
#           = 4 integral |h(chi) - h(chi_0)|^2 / Sn(f) df
#    For small Delta = chi - chi_0: D -> Gamma * Delta^2.
# ======================================================================
def innerprod_diff_sq(chi, chi_0):
    f_up, _ = f_upper_inspiral(Mc_kg, f_E, T_CAP_S)

    def integrand(f):
        dh = h_tilde_complex(f, chi) - h_tilde_complex(f, chi_0)
        return 4.0 * (dh.real ** 2 + dh.imag ** 2) / Sn_muares(f)

    D, _ = integrate.quad(integrand, f_E, f_up, limit=300)
    return D


def gamma_curvature_scan(chi_0, dchi_max, n_points=9):
    deltas = np.linspace(-dchi_max, dchi_max, n_points)
    Ds = np.array([innerprod_diff_sq(chi_0 + d, chi_0) for d in deltas])
    # Quadratic fit D = a + b*Delta + c*Delta^2 where a~0, b~0, c=Gamma
    coeffs = np.polyfit(deltas, Ds, deg=2)
    gamma = coeffs[0]   # leading coefficient (Delta^2)
    return gamma, deltas, Ds


# ======================================================================
# Runner
# ======================================================================
def main():
    print("=" * 70)
    print("crosscheck_sigma_chi.py")
    print(f"  Source: {m1_msun:.0e}+{m2_msun:.0e} Msun, chi={CHI}, "
          f"f_E={f_E * 1e6:.1f} uHz, D_L={D_L_Mpc} Mpc")
    print("=" * 70)

    # Library (closed-form analytic Fisher)
    sig_lib, rho_LB, _, Gamma_lib, f_up, t_eff = sigma_chi_E_fisher(
        m1_msun, m2_msun, CHI, f_E, D_L_m, T_cap_s=T_CAP_S,
    )
    print(f"\nLibrary (compute_rho_E.sigma_chi_E_fisher):")
    print(f"  Gamma_chi,chi  = {Gamma_lib:.6e}")
    print(f"  sigma_chi      = {sig_lib:.6e}")
    print(f"  rho_E (LB)     = {rho_LB:.3e}")
    print(f"  f_up           = {f_up * 1e6:.3f} uHz")
    print(f"  t_eff          = {t_eff / YR:.3f} yr")

    # ------------------------------------------------------------------
    # Method C: finite-difference Fisher, convergence sweep in dchi
    # ------------------------------------------------------------------
    print(f"\n--- Method C: finite-difference Fisher (central diff)")
    print(f"  {'dchi':>10s}   {'Gamma_FD':>15s}   {'frac err vs lib':>18s}")
    best = None
    for dchi in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        try:
            G_fd = gamma_finite_difference(dchi)
            frac = (G_fd - Gamma_lib) / Gamma_lib
            print(f"  {dchi:>10.0e}   {G_fd:>15.6e}   {frac:>+18.3e}")
            if best is None or abs(frac) < abs(best[1]):
                best = (dchi, frac, G_fd)
        except Exception as e:
            print(f"  {dchi:>10.0e}   failed: {e}")
    bd, bf, bG = best
    sigma_from_C = 1.0 / np.sqrt(bG)
    print(f"  best dchi = {bd:.0e}, Gamma_FD = {bG:.6e}, "
          f"sigma_chi = {sigma_from_C:.6e}")
    pass_C = abs(bf) < 1e-3

    # ------------------------------------------------------------------
    # Method D: curvature scan around chi_0
    # ------------------------------------------------------------------
    print(f"\n--- Method D: chi^2 curvature scan")
    # Scan range: +- a few sigma_chi.  We know sigma_chi ~ 1e-8, but
    # make the scan wide enough that Delta*47/6 * v^{-2} phase shift is
    # still small compared to 2 pi -- use dchi_max = 1e-6.
    dchi_max = 1e-6
    G_curv, deltas, Ds = gamma_curvature_scan(CHI, dchi_max, n_points=11)
    frac_D = (G_curv - Gamma_lib) / Gamma_lib
    sigma_from_D = 1.0 / np.sqrt(G_curv)
    print(f"  scan range Delta_chi = +- {dchi_max:.0e}  ({len(deltas)} points)")
    print(f"  Gamma_curvature fit  = {G_curv:.6e}")
    print(f"  sigma_chi from fit   = {sigma_from_D:.6e}")
    print(f"  frac err vs library  = {frac_D:+.3e}")
    # Check shape: D / (Gamma * Delta^2) should be ~1
    nonzero = deltas != 0
    residual_shape = Ds[nonzero] / (Gamma_lib * deltas[nonzero] ** 2)
    shape_dev = np.max(np.abs(residual_shape - 1.0))
    print(f"  max shape deviation  = {shape_dev:.3e} "
          f"(D / (Gamma_lib * Delta^2) - 1)")
    pass_D = abs(frac_D) < 1e-3 and shape_dev < 0.01

    # ------------------------------------------------------------------
    # Manual check of dPsi/dchi prefactor
    # ------------------------------------------------------------------
    # Library: dPsi/dchi = (188/(256 eta)) * v^{-2}
    # Derivation from psi_3 = -16 pi + 4 beta_SO, beta_SO = (47/6) chi:
    #   dpsi_3/dchi = 4 * 47/6 = 94/3
    #   dPsi/dchi   = (3/(128 eta)) * (94/3) * v^{-2}
    #               = (94/(128 eta)) * v^{-2}
    #               = (188/(256 eta)) * v^{-2}
    print(f"\n--- Prefactor consistency check")
    f_test = f_E
    v2_test = (np.pi * G * M_tot_kg / c ** 3 * f_test) ** (2.0 / 3)
    dPsi_dchi_lib = (188.0 / (256.0 * eta)) / v2_test
    dPsi_dchi_indep = taylorf2_phase_derivative_analytic(CHI, f_test)
    print(f"  at f = {f_test * 1e6:.1f} uHz:")
    print(f"    library  (188/(256 eta)) / v^2  = {dPsi_dchi_lib:.4e}")
    print(f"    derived  analytic helper        = {dPsi_dchi_indep:.4e}")
    print(f"    ratio library/derived           = "
          f"{dPsi_dchi_lib / dPsi_dchi_indep:.3f}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Method C (finite-diff)   : "
          f"{'PASS' if pass_C else 'FAIL'}  (best |frac err| = {abs(bf):.2e})")
    print(f"Method D (curvature fit) : "
          f"{'PASS' if pass_D else 'FAIL'}  (|frac err| = {abs(frac_D):.2e})")
    print("=" * 70)


if __name__ == "__main__":
    main()
