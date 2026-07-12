#!/usr/bin/env python3
"""
audit_ss_convention.py — Adjudicate the 2pN spin-spin (SS) phasing convention
against LALSuite, LIGO's reference post-Newtonian implementation.

Context (2026-07-11 audit): three SS treatments coexist in this repo and
disagree:

  smbhb_evolution.py (main module)   : tau_4_SS = -2 sigma,  phi_4_SS = -10 sigma
  crosscheck_cycle_counts.py Method B: tau_4_SS = -(40/eta) sigma,
                                       phi_4_SS = -(10/eta) sigma
  verify_equations.py check 18       : expects SS in the binding energy with an
                                       eta-free (-sigma/eta style) normalization

where all three use the SAME eta-inclusive Poisson & Will (1995) definition
  sigma = (eta/48) [-247 chi1 chi2 cos(k1-k2) + 721 chi1 cos k1 chi2 cos k2].

PN phasing coefficients are dimensionless functions of (eta, chi_i) only, so
LAL evaluated at LIGO-scale masses arbitrates for PTA-scale binaries exactly
(the mass/frequency scale drops out).

Method: lalsimulation.SimInspiralTaylorF2AlignedPhasing returns the TaylorF2
phasing series psi(f) = Sum_k [pn.v[k] + pn.vlogv[k] log v] v^(k-5), with
pn.v[0] = 3/(128 eta). We test:

  1. sanity     : pn.v[0] = 3/(128 eta); 1pN mass ratio v[2]/v[0] = alpha_2
  2. 2pN mass   : v[4]/v[0] at chi=0  = alpha_4_mass  (checks psi_4 = phi_4)
  3. 1.5pN SO   : [v3(chi,chi) - v3(0,0)]/v[0] = 4 beta_SO
  4. 2pN SS     : the chi1*chi2 bilinear of v[4]/v[0],
                    D = F(c,c) - F(c,0) - F(0,c) + F(0,0),  F = v4/v0,
                  isolates the spin1-spin2 (Kidder/PW sigma) term, cancelling
                  self-spin/quadrupole-monopole chi_i^2 terms.
                    module prediction    : D = -10 sigma      = -(4740/48) eta chi^2
                    crosscheck prediction: D = -(10/eta) sigma = -(4740/48) chi^2
                  Two mass ratios (eta = 0.25, 0.1875) separate the two by
                  their eta-scaling.

Exit 0 if LAL confirms the main module; exit 1 otherwise.
"""

import sys
import numpy as np
import lal
import lalsimulation as lalsim


def phasing_v(m1_msun, m2_msun, chi1, chi2):
    """TaylorF2 aligned-spin phasing coefficients pn.v[] from lalsimulation."""
    pn = lalsim.SimInspiralTaylorF2AlignedPhasing(
        m1_msun, m2_msun, chi1, chi2, lal.CreateDict())
    return np.array(pn.v)


def alpha2(eta):
    return (20.0 / 9) * (743.0 / 336 + 11.0 * eta / 4)


def alpha4_mass(eta):
    return 15293365.0 / 508032 + 27145.0 * eta / 504 + 3085.0 * eta ** 2 / 72


def beta_so_aligned(m1, m2, chi1, chi2):
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    return (1.0 / 12) * ((113 * (m1 / M) ** 2 + 75 * eta) * chi1
                         + (113 * (m2 / M) ** 2 + 75 * eta) * chi2)


def sigma_pw_aligned(eta, chi1, chi2):
    """Poisson & Will 1995 sigma, aligned spins (kappa1 = kappa2 = 0)."""
    return (eta / 48.0) * (-247.0 + 721.0) * chi1 * chi2


def main():
    ok = True
    print("=" * 72)
    print("SS-convention audit vs lalsimulation", lalsim.__version__)
    print("=" * 72)

    cases = [("q = 1   (eta = 0.2500)", 10.0, 10.0),
             ("q = 1/3 (eta = 0.1875)", 15.0, 5.0),
             ("q = 1, SMBHB masses",    1e9, 1e9)]
    chi = 0.98

    for label, m1, m2 in cases:
        M = m1 + m2
        eta = m1 * m2 / M ** 2
        v00 = phasing_v(m1, m2, 0.0, 0.0)
        vcc = phasing_v(m1, m2, chi, chi)
        vc0 = phasing_v(m1, m2, chi, 0.0)
        v0c = phasing_v(m1, m2, 0.0, chi)

        print(f"\n--- {label}:  m1 = {m1:g}, m2 = {m2:g}, chi = {chi}")

        # 1. Normalization sanity
        v0_expect = 3.0 / (128.0 * eta)
        d = abs(v00[0] / v0_expect - 1)
        print(f"  v[0] = {v00[0]:.10f}  vs 3/(128 eta) = {v0_expect:.10f}"
              f"   [{'OK' if d < 1e-12 else 'MISMATCH'}]")
        ok &= d < 1e-12

        # 1b. 1pN mass sector
        r2 = v00[2] / v00[0]
        d = abs(r2 / alpha2(eta) - 1)
        print(f"  1pN   v[2]/v[0] = {r2:.10f}  vs alpha_2 = {alpha2(eta):.10f}"
              f"   [{'OK' if d < 1e-12 else 'MISMATCH'}]")
        ok &= d < 1e-12

        # 2. 2pN mass sector (chi = 0): checks psi_4 = phi_4 identity
        r4 = v00[4] / v00[0]
        d = abs(r4 / alpha4_mass(eta) - 1)
        print(f"  2pN   v[4]/v[0] = {r4:.10f}  vs alpha_4 = {alpha4_mass(eta):.10f}"
              f"   [{'OK' if d < 1e-12 else 'MISMATCH'}]")
        ok &= d < 1e-12

        # 3. 1.5pN spin-orbit
        so_lal = (vcc[3] - v00[3]) / v00[0]
        so_mod = 4.0 * beta_so_aligned(m1, m2, chi, chi)
        d = abs(so_lal / so_mod - 1)
        print(f"  1.5pN SO  LAL = {so_lal:.10f}  vs 4 beta_SO = {so_mod:.10f}"
              f"   [{'OK' if d < 1e-9 else 'MISMATCH'}]")
        ok &= d < 1e-9

        # 4. 2pN spin1-spin2 bilinear — THE DISPUTED TERM
        D_lal = (vcc[4] - vc0[4] - v0c[4] + v00[4]) / v00[0]
        sigma = sigma_pw_aligned(eta, chi, chi)
        D_module = -10.0 * sigma            # smbhb_evolution convention
        D_crosschk = -(10.0 / eta) * sigma  # crosscheck_cycle_counts convention
        dm = abs(D_lal / D_module - 1) if D_module else np.inf
        dc = abs(D_lal / D_crosschk - 1) if D_crosschk else np.inf
        print(f"  2pN SS bilinear  LAL       = {D_lal:+.10f}")
        print(f"                   -10 sigma = {D_module:+.10f}"
              f"  (module)     rel.dev = {dm:.2e}")
        print(f"              -(10/eta)sigma = {D_crosschk:+.10f}"
              f"  (crosscheck) rel.dev = {dc:.2e}")
        if dm < 1e-9:
            print("                   => LAL CONFIRMS the smbhb_evolution "
                  "convention (phi_4_SS = -10 sigma)")
        elif dc < 1e-9:
            print("                   => LAL CONFIRMS the crosscheck "
                  "convention (phi_4_SS = -(10/eta) sigma)")
            ok = False
        else:
            print("                   => LAL matches NEITHER — investigate")
            ok = False

    print("\n" + "=" * 72)
    print("VERDICT:", "main module (smbhb_evolution) convention CONFIRMED by LAL"
          if ok else "module convention NOT confirmed — see mismatches above")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
