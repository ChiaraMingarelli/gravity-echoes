#!/usr/bin/env python3
"""
phase_matching.py — Phase matching analysis for gravity echoes

Given a μAres-detected SMBHB, how well can we connect the echo phases
across a PTA of 200 pulsars, with N_anchor VLBI-distance pulsars?

Source: 10^9 Msun equal-mass binary at 100 Mpc, f_E = 1 μHz, χ = 0.9
PTA: 200 SKA-era pulsars, isotropically distributed, d ~ 0.5-3 kpc

Key questions:
1. What is the cycle ambiguity per pulsar as a function of distance precision?
2. With N_anchor anchors, can we uniquely identify the correct waveform?
3. What is the false alarm probability of a coincident match?
4. How does sky localization improve with echo detections?

Author: Chiara Mingarelli, with computational assistance
"""

import numpy as np
from dataclasses import dataclass

# ============================================================
# Constants
# ============================================================
G = 6.67430e-11       # m^3 kg^-1 s^-2
c = 2.99792458e8      # m/s
Msun = 1.98892e30     # kg
pc = 3.08567758e16    # m
Mpc = 1e6 * pc
yr = 365.25 * 86400   # s


# ============================================================
# Source parameters
# ============================================================
@dataclass
class Source:
    M_tot: float    # total mass [kg]
    eta: float      # symmetric mass ratio
    chi: float      # aligned spin magnitude
    DL: float       # luminosity distance [m]
    f_E: float      # earth-term frequency [Hz]
    theta: float    # source colatitude [rad]
    phi: float      # source longitude [rad]

    @property
    def Mc(self):
        return self.M_tot * self.eta ** (3. / 5)

    @property
    def GMc_c3(self):
        return G * self.Mc / c ** 3

    @property
    def f_isco(self):
        return c ** 3 / (6 ** 1.5 * np.pi * G * self.M_tot)

    @property
    def v_E(self):
        """Velocity parameter at f_E"""
        return (np.pi * G * self.M_tot * self.f_E / c ** 3) ** (1. / 3)

    def Omega_hat(self):
        """Source direction unit vector"""
        st, ct = np.sin(self.theta), np.cos(self.theta)
        sp, cp = np.sin(self.phi), np.cos(self.phi)
        return np.array([st * cp, st * sp, ct])


# ============================================================
# GW physics functions
# ============================================================
def t_merge(f, GMc_c3):
    """Time to merger from frequency f (Newtonian)"""
    return (5. / 256) * (np.pi * f) ** (-8. / 3) * GMc_c3 ** (-5. / 3)


def fdot_newt(f, GMc_c3):
    """Newtonian frequency derivative"""
    return (96. / 5) * np.pi ** (8. / 3) * GMc_c3 ** (5. / 3) * f ** (11. / 3)


def h0(f, Mc, DL):
    """GW strain amplitude (face-on, leading order)"""
    return (4. / DL) * (G * Mc / c ** 2) ** (5. / 3) * (np.pi * f / c) ** (2. / 3)


def f_from_t_merge(t, GMc_c3):
    """Frequency given time to merger (Newtonian, inverse of t_merge)"""
    return ((5. / 256) * GMc_c3 ** (-5. / 3) / t) ** (3. / 8) / np.pi


def geometric_delay(L_p, p_hat, Omega_hat):
    """
    Geometric delay τ = (L_p/c)(1 + Ω̂·p̂)

    Parameters
    ----------
    L_p : float, pulsar distance [m]
    p_hat : array, pulsar direction unit vector
    Omega_hat : array, source direction unit vector

    Returns
    -------
    tau : float, geometric delay [s]
    """
    return (L_p / c) * (1.0 + np.dot(Omega_hat, p_hat))


def f_pulsar(tau, f_E, GMc_c3):
    """
    Pulsar-term frequency given geometric delay tau [s].
    The pulsar sees the binary at an earlier epoch, so f_P < f_E.
    """
    t_E = t_merge(f_E, GMc_c3)
    t_P = t_E + tau  # more time to merger at pulsar epoch
    if t_P <= 0:
        return np.nan
    return f_from_t_merge(t_P, GMc_c3)


def timing_residual(f_P, Mc, DL):
    """Timing residual r_P = h_0 / (2π f_P)"""
    return h0(f_P, Mc, DL) / (2 * np.pi * f_P)


def pn_cycles(f_low, f_high, M_tot, eta, chi):
    """
    GW cycle count from f_low to f_high at each pN order.

    Returns (N_newt, N_1pN, N_15pN, N_2pN)
    """
    v_low = (np.pi * G * M_tot * f_low / c ** 3) ** (1. / 3)
    v_high = (np.pi * G * M_tot * f_high / c ** 3) ** (1. / 3)

    prefactor = 3. / (128 * eta * 2 * np.pi)

    # Newtonian
    N_0 = prefactor * (v_low ** (-5) - v_high ** (-5))

    # 1pN
    c2 = 3715. / 756 + 55. * eta / 9
    N_1 = prefactor * c2 * (v_low ** (-3) - v_high ** (-3))

    # 1.5pN (spin-orbit, equal mass aligned)
    c3 = -16 * np.pi + (113. / 6) * chi
    N_15 = prefactor * c3 * (v_low ** (-2) - v_high ** (-2))

    # 2pN (including spin-spin for equal mass aligned)
    sigma_SS = eta * (721. / 48 - 247. / 48) * chi ** 2
    c4 = (15293365. / 508032 + 27145. / 504 * eta
           + 3085. / 72 * eta ** 2 - 10 * sigma_SS / eta)
    N_2 = prefactor * c4 * (v_low ** (-1) - v_high ** (-1))

    return N_0, N_1, N_15, N_2


# ============================================================
# μAres noise model — Sesana+2021 Sec 4, flat acceleration noise
# ============================================================
L_muares = 3.95e11     # m (Mars orbit arm length)
S_pos = (50e-12) ** 2  # m^2/Hz
S_acc_flat = (1e-15) ** 2  # m^2/s^4/Hz — FLAT per Sesana+2021
f_star_muares = c / (2 * np.pi * L_muares)


def Sn_muares(f):
    """Sky-averaged μAres strain noise PSD (instrument only, flat accel)"""
    return ((20. / 3) / L_muares ** 2
            * (4 * S_acc_flat / (2 * np.pi * f) ** 4 + S_pos)
            * (1 + (f / f_star_muares) ** 2))


def hn_muares(f):
    """Characteristic noise strain"""
    return np.sqrt(f * Sn_muares(f))


# ============================================================
# Pulsar array generation
# ============================================================
@dataclass
class Pulsar:
    name: str
    d_kpc: float       # distance [kpc]
    sigma_d_frac: float # fractional distance uncertainty
    theta_p: float     # colatitude [rad]
    phi_p: float       # longitude [rad]
    sigma_TOA_ns: float  # timing precision [ns]

    @property
    def L_p(self):
        return self.d_kpc * 1e3 * pc

    @property
    def p_hat(self):
        st, ct = np.sin(self.theta_p), np.cos(self.theta_p)
        sp, cp = np.sin(self.phi_p), np.cos(self.phi_p)
        return np.array([st * cp, st * sp, ct])


def generate_ska_array(N_psr, N_anchor, rng=None):
    """
    Generate a mock SKA pulsar array.

    Parameters
    ----------
    N_psr : int
        Total number of pulsars
    N_anchor : int
        Number with VLBI-quality distances (1% precision)
    rng : numpy RNG

    Returns
    -------
    list of Pulsar objects
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pulsars = []
    for i in range(N_psr):
        # Distance: log-uniform between 0.3 and 5 kpc
        # (MSPs are concentrated in the disk, ~1-2 kpc typical)
        d_kpc = 10 ** rng.uniform(np.log10(0.3), np.log10(5.0))

        # Sky position: isotropic
        cos_theta = rng.uniform(-1, 1)
        theta_p = np.arccos(cos_theta)
        phi_p = rng.uniform(0, 2 * np.pi)

        # Distance precision
        if i < N_anchor:
            sigma_d = 0.01  # 1% VLBI
        else:
            sigma_d = 0.20  # 20% typical (DM-based or parallax)

        # Timing precision: 100 ns typical for SKA
        sigma_TOA = 100.0  # ns

        pulsars.append(Pulsar(
            name=f"PSR_{i:04d}" if i >= N_anchor else f"ANCHOR_{i:03d}",
            d_kpc=d_kpc,
            sigma_d_frac=sigma_d,
            theta_p=theta_p,
            phi_p=phi_p,
            sigma_TOA_ns=sigma_TOA,
        ))

    return pulsars


# ============================================================
# Phase matching analysis
# ============================================================
def analyze_echo(source, pulsar):
    """
    Compute echo observables for one pulsar.

    Returns dict with tau, f_P, r_P, SNR, cycle counts, ambiguity, etc.
    """
    Omega = source.Omega_hat()
    tau = geometric_delay(pulsar.L_p, pulsar.p_hat, Omega)
    tau_yr = tau / yr

    if tau_yr < 1.0:
        return None  # negligible delay

    fP = f_pulsar(tau, source.f_E, source.GMc_c3)
    if np.isnan(fP) or fP <= 0:
        return None

    rP = timing_residual(fP, source.Mc, source.DL)
    rP_ns = rP * 1e9

    # Per-pulsar SNR: ρ = r_P sqrt(N_obs/2) / σ_TOA
    # With biweekly cadence over 20 yr: N_obs = 26 * 20 = 520
    N_obs = 520
    sigma_TOA = pulsar.sigma_TOA_ns * 1e-9
    rho_i = rP * np.sqrt(N_obs / 2) / sigma_TOA

    # Cycle counts
    N0, N1, N15, N2 = pn_cycles(
        fP, source.f_E, source.M_tot, source.eta, source.chi
    )
    N_total = N0 + N1 + N15 + N2

    # Cycle ambiguity from distance uncertainty
    delta_d = pulsar.sigma_d_frac * pulsar.L_p
    delta_tau = delta_d / c  # seconds
    cycle_ambiguity = fP * delta_tau  # number of cycles

    # Phase error from μAres spin uncertainty
    # σ_χ ~ 1.7e-8 → δΦ_spin = 2π |N_SO| σ_χ/χ
    sigma_chi = 1.7e-8
    N_SO = abs(N15)
    phase_err_spin = 2 * np.pi * N_SO * sigma_chi / source.chi

    return {
        'name': pulsar.name,
        'd_kpc': pulsar.d_kpc,
        'sigma_d_frac': pulsar.sigma_d_frac,
        'tau_yr': tau_yr,
        'f_P_nHz': fP * 1e9,
        'r_P_ns': rP_ns,
        'rho_i': rho_i,
        'N_newt': N0,
        'N_1pN': N1,
        'N_15pN': N15,
        'N_2pN': N2,
        'N_total': N_total,
        'cycle_ambiguity': cycle_ambiguity,
        'phase_err_spin': phase_err_spin,
    }


def run_analysis(source, pulsars, detection_threshold=1.0):
    """
    Run the full phase matching analysis.

    Parameters
    ----------
    source : Source
    pulsars : list of Pulsar
    detection_threshold : float
        Minimum per-pulsar SNR for echo detection

    Returns
    -------
    results : list of dicts (one per detected pulsar)
    summary : dict with aggregate statistics
    """
    results = []
    for psr in pulsars:
        echo = analyze_echo(source, psr)
        if echo is not None and echo['rho_i'] >= detection_threshold:
            results.append(echo)

    if not results:
        return results, {'N_detected': 0}

    # Sort by SNR
    results.sort(key=lambda x: -x['rho_i'])

    # Aggregate
    rho_comb = np.sqrt(sum(r['rho_i'] ** 2 for r in results))

    anchors = [r for r in results if r['sigma_d_frac'] <= 0.02]
    non_anchors = [r for r in results if r['sigma_d_frac'] > 0.02]

    # Anchor statistics
    if anchors:
        anchor_ambiguities = [r['cycle_ambiguity'] for r in anchors]
        anchor_taus = [r['tau_yr'] for r in anchors]
        anchor_fPs = [r['f_P_nHz'] for r in anchors]
    else:
        anchor_ambiguities = []
        anchor_taus = []
        anchor_fPs = []

    # False alarm analysis for anchors:
    # The correct waveform predicts a specific integer cycle offset
    # for each anchor. A wrong waveform (e.g., wrong Mc by δMc)
    # shifts the predicted cycle count at each anchor by a different
    # amount (because each is at different f_P).
    #
    # For N_anch anchors, each with ambiguity ΔN_i, the total
    # search space is Π ΔN_i. But a wrong waveform must simultaneously
    # satisfy all N_anch constraints. The probability of a false
    # coincidence (random alignment) is ~ 1/Π ΔN_i × (degeneracy factor).
    #
    # More precisely: the waveform has ~5 free parameters
    # (Mc, M, η, χ, f_E). Each anchor provides 1 constraint
    # (the integer cycle offset). So with N_anch > 5 anchors,
    # the system is overdetermined.
    #
    # The false alarm probability: for each anchor beyond the 5th,
    # the predicted cycle must match to within ±0.5. The probability
    # of this by chance is 1/(2 ΔN_i) per anchor. So:
    # P_FA ~ Π_{i=6}^{N_anch} 1/(2 ΔN_i)

    if len(anchors) > 5:
        # Sort anchors by ambiguity (largest first = most constraining last)
        sorted_amb = sorted(anchor_ambiguities, reverse=True)
        # First 5 determine the waveform; remaining are cross-checks
        log_pfa = sum(-np.log10(2 * a) for a in sorted_amb[5:])
        pfa = 10 ** log_pfa
    else:
        pfa = None  # underdetermined

    summary = {
        'N_total': len(pulsars),
        'N_detected': len(results),
        'N_anchors_detected': len(anchors),
        'N_non_anchors': len(non_anchors),
        'rho_comb': rho_comb,
        'rho_max': results[0]['rho_i'] if results else 0,
        'anchor_ambiguities': anchor_ambiguities,
        'anchor_taus': anchor_taus,
        'anchor_fPs': anchor_fPs,
        'mean_anchor_ambiguity': np.mean(anchor_ambiguities) if anchors else 0,
        'median_anchor_ambiguity': np.median(anchor_ambiguities) if anchors else 0,
        'P_false_alarm': pfa,
        'N_overdetermined': max(0, len(anchors) - 5),
    }

    return results, summary


def print_summary(results, summary, label=""):
    """Pretty-print the analysis results."""
    print(f"\n{'='*70}")
    print(f"PHASE MATCHING ANALYSIS{': ' + label if label else ''}")
    print(f"{'='*70}")
    print(f"Total pulsars: {summary['N_total']}")
    print(f"Detected (ρ > 1): {summary['N_detected']}")
    print(f"  Anchors (1% VLBI): {summary['N_anchors_detected']}")
    print(f"  Non-anchors: {summary['N_non_anchors']}")
    print(f"Combined SNR: ρ_comb = {summary['rho_comb']:.1f}")
    print(f"Max single-pulsar SNR: {summary['rho_max']:.1f}")

    if summary['N_anchors_detected'] > 0:
        print(f"\n--- ANCHOR PULSARS ---")
        print(f"{'Name':<14} {'d[kpc]':>6} {'τ[yr]':>7} {'f_P[nHz]':>8} "
              f"{'ρ_i':>6} {'ΔN_cyc':>7} {'N_newt':>7} {'N_SO':>6}")
        for r in results:
            if r['sigma_d_frac'] <= 0.02:
                print(f"{r['name']:<14} {r['d_kpc']:>6.2f} {r['tau_yr']:>7.0f} "
                      f"{r['f_P_nHz']:>8.1f} {r['rho_i']:>6.1f} "
                      f"{r['cycle_ambiguity']:>7.0f} {r['N_newt']:>7.0f} "
                      f"{r['N_15pN']:>+6.0f}")

        amb = summary['anchor_ambiguities']
        print(f"\nCycle ambiguity (1% VLBI): "
              f"mean = {summary['mean_anchor_ambiguity']:.0f}, "
              f"median = {summary['median_anchor_ambiguity']:.0f}")
        print(f"Range of τ: {min(summary['anchor_taus']):.0f} "
              f"to {max(summary['anchor_taus']):.0f} yr")
        print(f"Range of f_P: {min(summary['anchor_fPs']):.1f} "
              f"to {max(summary['anchor_fPs']):.1f} nHz")

        print(f"\n--- CYCLE AMBIGUITY RESOLUTION ---")
        print(f"Free waveform parameters: ~5 (Mc, M, η, χ, f_E)")
        print(f"Anchor constraints: {summary['N_anchors_detected']}")
        print(f"Overdetermination: {summary['N_overdetermined']} extra constraints")

        if summary['P_false_alarm'] is not None:
            print(f"False alarm probability: {summary['P_false_alarm']:.1e}")
            print(f"  (prob. of random cycle alignment for "
                  f"{summary['N_overdetermined']} cross-check anchors)")
        else:
            print(f"System is underdetermined — need > 5 anchors")

    # Show a few non-anchor detections
    non_anch = [r for r in results if r['sigma_d_frac'] > 0.02]
    if non_anch:
        print(f"\n--- NON-ANCHOR DETECTIONS (top 10 by SNR) ---")
        print(f"{'Name':<14} {'d[kpc]':>6} {'τ[yr]':>7} {'f_P[nHz]':>8} "
              f"{'ρ_i':>6} {'r_P[ns]':>8}")
        for r in non_anch[:10]:
            print(f"{r['name']:<14} {r['d_kpc']:>6.2f} {r['tau_yr']:>7.0f} "
                  f"{r['f_P_nHz']:>8.1f} {r['rho_i']:>6.1f} "
                  f"{r['r_P_ns']:>8.1f}")

    # Phase error budget
    if results:
        print(f"\n--- PHASE ERROR BUDGET (best anchor) ---")
        best = [r for r in results if r['sigma_d_frac'] <= 0.02]
        if best:
            b = best[0]
            print(f"Phase error from μAres spin (σ_χ=1.7e-8): "
                  f"{b['phase_err_spin']:.1e} rad")
            print(f"Phase error from distance (1% VLBI): "
                  f"{2*np.pi*b['cycle_ambiguity']:.0f} rad "
                  f"({b['cycle_ambiguity']:.0f} cycle ambiguity)")
            print(f"→ Resolved by discrete grid search over "
                  f"{b['cycle_ambiguity']:.0f} trials")


# ============================================================
# Main analysis
# ============================================================
if __name__ == "__main__":

    # Define the source
    source = Source(
        M_tot=1e9 * Msun,
        eta=0.25,
        chi=0.9,
        DL=100 * Mpc,
        f_E=1e-6,
        theta=np.pi / 3,   # 60 deg colatitude
        phi=1.0,            # arbitrary longitude
    )

    print("SOURCE PARAMETERS")
    print(f"  M_tot = 10^9 Msun, m1 = m2 = 5×10^8 Msun")
    print(f"  Mc = {source.Mc/Msun:.2e} Msun")
    print(f"  D_L = {source.DL/Mpc:.0f} Mpc")
    print(f"  f_E = {source.f_E*1e6:.0f} μHz")
    print(f"  χ = {source.chi}")
    print(f"  v/c at f_E = {source.v_E:.4f}")
    print(f"  f_ISCO = {source.f_isco*1e6:.1f} μHz")
    print(f"  t_merge from f_E = {t_merge(source.f_E, source.GMc_c3)/yr:.1f} yr")

    # μAres SNR
    rho_E = 4e5
    sigma_chi = 1.7e-8
    print(f"\n  μAres ρ_E ≈ {rho_E:.0e}")
    print(f"  σ_χ = {sigma_chi:.1e}")

    # Run for different anchor configurations
    for N_anchor in [10, 20]:
        rng = np.random.default_rng(42)  # reproducible
        pulsars = generate_ska_array(200, N_anchor, rng=rng)

        results, summary = run_analysis(source, pulsars)
        print_summary(results, summary,
                      label=f"200 pulsars, {N_anchor} anchors")

    # Also run with 0 anchors to show the problem
    print("\n" + "="*70)
    print("COMPARISON: NO ANCHORS (all 20% distance precision)")
    print("="*70)
    rng = np.random.default_rng(42)
    pulsars_no_anchor = generate_ska_array(200, 0, rng=rng)
    results_na, summary_na = run_analysis(source, pulsars_no_anchor)
    if summary_na['N_detected'] > 0:
        # Show ambiguity for a few pulsars
        detected = [r for r in results_na if r['rho_i'] > 1]
        print(f"Detected: {len(detected)}")
        print(f"Typical cycle ambiguity (20% distance): "
              f"{np.median([r['cycle_ambiguity'] for r in detected]):.0f}")
        print(f"→ Phase connection impossible without VLBI anchors")
