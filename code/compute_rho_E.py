"""
Derive ρ_E and σ_χ^(E) from first principles — inspiral only, no hardcoded values.

References:
- Moore, Cole & Berry 2014, Class. Quant. Grav. 32, 015014 (arXiv:1408.0740)
    Eq. 16:  ρ² = 4 ∫ |h̃(f)|² / Sn(f) df           (matched-filter SNR)
    Eq. 19:  ρ² = ∫ [h_c(f)/h_n(f)]² d(ln f)      (characteristic-strain form)
    Eq. 35:  h_c(f) = h_0(f) · sqrt(2 f²/ḟ)        (inspiral, SPA)
- Sesana+ 2021 (arXiv:1908.11391), Fig. 1 caption: μAres Sn is sky-averaged.
- Mingarelli+ 2012 (arXiv:1207.5645): TaylorF2 pN phasing, 1.5pN SO term.
- Blanchet 2014 Living Rev. Rel. 17, 2: pN waveform formulae.

Convention:
- h_0(f) = 4 (G Mc)^(5/3) (π f)^(2/3) / (c^4 D_L)     [PTA strain factor]
- Sn_muares(f) from phase_matching.py (already sky-averaged, 20/3 factor)
- TaylorF2 phase: Ψ(f) = (3/(128 η)) v^(-5) Σ_k ψ_k v^k,  v = (π M f)^(1/3)
  ψ_3 = -16π + 4 β_SO (Cutler-Flanagan 1994; Arun+ 2005 Eq. A10; Blanchet LRR Eq. 426)
  β_SO = (47/6) χ  for equal-mass aligned spins (Mingarelli+ 2012 Table I)
  so ∂Ψ/∂χ = (3/(128 η)) · 4 · (47/6) · v^(-2) = (188/(256 η)) · (π M f)^(-2/3)
       = (47/(64 η)) · (π M f)^(-2/3)

Integration domain: inspiral only, from f_E up to f(min(T_cap, t_merge_Newt)).
Cap at T_cap = 1 yr so we stay in the adiabatic inspiral regime where
TaylorF2 + Newtonian ḟ are reliable. The result is a lower bound on ρ_E and
an upper bound on σ_χ^(E).
"""
from __future__ import annotations

import numpy as np
from scipy import integrate

from phase_matching import Sn_muares

# Physical constants (SI)
G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.98892e30
Mpc = 3.0857e22
YR = 3.15576e7


# ======================================================================
# Amplitude and frequency evolution (Newtonian inspiral)
# ======================================================================
def h0_strain(f, Mc_kg, D_L_m):
    """PTA-convention strain amplitude h_0(f) = 4 (G Mc)^(5/3) (π f)^(2/3) / (c^4 D_L)."""
    return 4.0 * (G * Mc_kg) ** (5.0 / 3) * (np.pi * f) ** (2.0 / 3) / (c ** 4 * D_L_m)


def fdot_newtonian(f, Mc_kg):
    """Newtonian chirp rate ḟ = (96/5) π^(8/3) (G Mc / c³)^(5/3) f^(11/3)."""
    return (96.0 / 5.0) * np.pi ** (8.0 / 3) * (G * Mc_kg / c ** 3) ** (5.0 / 3) * f ** (11.0 / 3)


def t_to_merger(f, Mc_kg):
    """Newtonian time to merger from frequency f (inspiral only, not physical at merger).
       t(f) = (5/256) (G Mc)^(-5/3) c^5 (π f)^(-8/3).
    """
    return (5.0 / 256.0) * (G * Mc_kg) ** (-5.0 / 3) * c ** 5 / (np.pi * f) ** (8.0 / 3)


def f_of_t(f_start, t_elapsed, Mc_kg):
    """Newtonian frequency after t_elapsed of chirping from f_start."""
    t_start = t_to_merger(f_start, Mc_kg)
    t_rem = t_start - t_elapsed
    if t_rem <= 0:
        raise ValueError("f_of_t: t_elapsed exceeds t_to_merger from f_start")
    return (1.0 / np.pi) * ((5.0 / 256.0) * (G * Mc_kg) ** (-5.0 / 3) * c ** 5 / t_rem) ** (3.0 / 8)


def f_upper_inspiral(Mc_kg, f_E, T_cap_s):
    """Upper integration bound: f reached after min(T_cap, t_merge) of chirping.
       Returns (f_upper, t_eff) where t_eff is the effective integration time."""
    t_merge = t_to_merger(f_E, Mc_kg)
    t_eff = min(T_cap_s, t_merge)
    if t_eff >= t_merge:
        # Would run to Newtonian "merger" — instead back off slightly to stay in inspiral.
        # Use t_eff = 0.99 * t_merge so the SPA integrand remains well-defined.
        t_eff = 0.99 * t_merge
    f_up = f_of_t(f_E, t_eff, Mc_kg)
    return f_up, t_eff


# ======================================================================
# Inspiral-only matched-filter SNR (lower bound)
# ======================================================================
def rho_E_lower_bound(Mc_kg, f_E, D_L_m, T_cap_s=YR):
    """
    Lower bound on μAres Earth-term SNR, inspiral only, Newtonian chirp.

    ρ² = ∫_{f_E}^{f_up} [h_c(f) / h_n(f)]² d(ln f),    f_up = f(min(T_cap, t_merge)).
    """
    f_up, t_eff = f_upper_inspiral(Mc_kg, f_E, T_cap_s)

    def integrand(ln_f):
        f = np.exp(ln_f)
        h0 = h0_strain(f, Mc_kg, D_L_m)
        fdot = fdot_newtonian(f, Mc_kg)
        h_c = h0 * np.sqrt(2.0 * f * f / fdot)
        h_n = np.sqrt(f * Sn_muares(f))
        return (h_c / h_n) ** 2

    rho_sq, _ = integrate.quad(integrand, np.log(f_E), np.log(f_up), limit=200)
    return np.sqrt(rho_sq), f_up, t_eff


# ======================================================================
# Proper Fisher for σ_χ^(E), equal-mass aligned spins (1.5pN SO term)
# ======================================================================
def sigma_chi_E_fisher(m1_msun, m2_msun, chi, f_E, D_L_m, T_cap_s=YR):
    """
    Γ_χχ = 4 ∫ |h̃(f)|² |∂Ψ/∂χ|² / Sn(f) df,    σ_χ = 1/√Γ_χχ.

    For TaylorF2 with aligned equal-mass spins:
        ψ_3 = -16 π + 4 β_SO        [Cutler-Flanagan 1994; Arun+ 2005 Eq. A10]
        β_SO = (47/6) χ             [Mingarelli+ 2012 Table I]
        ∂Ψ/∂χ = (3/(128 η)) · 4 · (47/6) · (π M f)^(-2/3)
              = (188/(256 η)) · (π M f)^(-2/3)
              = (47/(64 η))  · (π M f)^(-2/3)

    |h̃(f)|² from SPA: |h̃|² = h_0(f)² / (2 ḟ) · (factor of 1/2 for standard
    Moore Eq. 34 convention, with h_0 the RMS strain).  For our PTA h_0 which
    is the peak amplitude, <h(t)²> = h_0²/2 per pol; with two pols summed in
    quadrature <h_+² + h_x²> = h_0², so the Fourier amplitude squared is
    |h̃|² = h_0² / (2 ḟ)     (matches Moore's derivation with his h_0_RMS = our h_0).

    Also equivalently: |h̃|² = h_c² / (4 f²) per Moore Eq. 17.

    Returns (sigma_chi, rho_E_LB, rho_sq, Gamma_xx).
    """
    M_tot_msun = m1_msun + m2_msun
    Mc_msun = (m1_msun * m2_msun) ** 0.6 / M_tot_msun ** 0.2
    eta = (m1_msun * m2_msun) / M_tot_msun ** 2
    M_tot_kg = M_tot_msun * M_sun
    Mc_kg = Mc_msun * M_sun

    f_up, t_eff = f_upper_inspiral(Mc_kg, f_E, T_cap_s)

    def h_tilde_sq(f):
        h0 = h0_strain(f, Mc_kg, D_L_m)
        fdot = fdot_newtonian(f, Mc_kg)
        return h0 * h0 / (2.0 * fdot)

    def dPsi_dchi(f):
        # ∂Ψ_1.5pN_SO / ∂χ = (188/(256 η)) · (π M f)^(-2/3)
        # from ψ_3 = -16π + 4 β_SO, β_SO = (47/6) χ for equal-mass aligned
        # = (3/(128 η)) · 4 · (47/6) · v^(-2)
        return (188.0 / (256.0 * eta)) * (np.pi * G * M_tot_kg / c ** 3 * f) ** (-2.0 / 3)

    def fisher_integrand(f):
        return 4.0 * h_tilde_sq(f) * dPsi_dchi(f) ** 2 / Sn_muares(f)

    def snr_integrand(f):
        return 4.0 * h_tilde_sq(f) / Sn_muares(f)

    Gamma, _ = integrate.quad(fisher_integrand, f_E, f_up, limit=200)
    rho_sq, _ = integrate.quad(snr_integrand, f_E, f_up, limit=200)

    sigma_chi = 1.0 / np.sqrt(Gamma)
    rho_E_LB = np.sqrt(rho_sq)
    return sigma_chi, rho_E_LB, rho_sq, Gamma, f_up, t_eff


# ======================================================================
# Schematic σ_χ for comparison (the paper's current formula)
# ======================================================================
def sigma_chi_E_schematic(Mc_kg, chi, N_SO, rho_E):
    """σ_χ ≈ χ / (2π N_SO ρ_E). Rigid-phase approximation."""
    return chi / (2.0 * np.pi * abs(N_SO) * rho_E)


def chirp_mass(m1_msun, m2_msun):
    return (m1_msun * m2_msun) ** (3.0 / 5) / (m1_msun + m2_msun) ** (1.0 / 5)


# ======================================================================
# Run on the three fiducial scenarios
# ======================================================================
if __name__ == "__main__":
    # Import N_SO from the canonical evolution library for the schematic comparison.
    from smbhb_evolution import SMBHBEvolution

    # Fiducial scenarios: (label, m1, m2, f_E, D_L_Mpc, chi)
    scenarios = [
        ("Optimistic (100 Mpc)", 5e8, 5e8, 1e-6, 100.0, 0.98),
        ("Optimistic (80 Mpc)",  5e8, 5e8, 1e-6, 80.0,  0.98),
        ("Typical (200 Mpc)",    3e8, 3e8, 1e-6, 200.0, 0.98),
        ("Conservative (2 Gpc)", 5e7, 5e7, 1e-5, 2000.0, 0.98),
    ]

    print("=" * 110)
    print("μAres Earth-term: inspiral-only lower bounds (T_cap = 1 yr)")
    print("Waveform: TaylorF2 amplitude (SPA) + Newtonian chirp; 1.5pN SO phase for Fisher")
    print("=" * 110)

    hdr = (f"{'Scenario':<22s} {'t_merge':>9s} {'t_eff':>7s} {'f_up':>10s} "
           f"{'N_SO':>7s} {'ρ_E^LB':>11s} {'σ_χ (full)':>13s} {'σ_χ (schem)':>13s}")
    print(hdr)
    print(f"{'':<22s} {'[yr]':>9s} {'[yr]':>7s} {'[μHz]':>10s} "
          f"{'':>7s} {'(integ)':>11s} {'Fisher':>13s} {'χ/(2π N_SO ρ)':>13s}")
    print("-" * len(hdr))

    for label, m1, m2, f_E, D_L_Mpc, chi in scenarios:
        Mc_kg = chirp_mass(m1, m2) * M_sun
        D_L_m = D_L_Mpc * Mpc

        sig_full, rho_LB, _, _, f_up, t_eff = sigma_chi_E_fisher(
            m1, m2, chi, f_E, D_L_m, T_cap_s=YR)
        t_merge = t_to_merger(f_E, Mc_kg)

        # N_SO from canonical evolution library (Earth-term only, over t_eff)
        binary = SMBHBEvolution(
            m1=m1, m2=m2, chi1=chi, chi2=chi,
            kappa1=0.0, kappa2=0.0,
            f_gw_earth=f_E, D_L=D_L_Mpc,
        )
        pn = binary.pn_decomposition(t_eff / YR)
        N_SO = pn["cycles"]["SO"]

        sig_schem = sigma_chi_E_schematic(Mc_kg, chi, N_SO, rho_LB)

        print(f"{label:<22s} {t_merge/YR:>9.3f} {t_eff/YR:>7.3f} {f_up*1e6:>10.3f} "
              f"{N_SO:>7.1f} {rho_LB:>11.3e} {sig_full:>13.3e} {sig_schem:>13.3e}")

    print()
    print("Notes:")
    print("  ρ_E^LB  = lower-bound matched-filter SNR (Moore Eq.19, inspiral only, T_cap=1yr)")
    print("  σ_χ full Fisher = 1/√(4∫|h̃|² (∂Ψ/∂χ)² / Sn df) — the proper inspiral Fisher")
    print("  σ_χ schematic   = χ/(2π N_SO ρ_E) — rigid-phase approx (for reference only)")
