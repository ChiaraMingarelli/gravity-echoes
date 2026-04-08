#!/usr/bin/env python3
"""
Compute the accumulated GW dephasing from a circumbinary disk torque
over an echo baseline.

Physics:
--------
Above the decoupling frequency f_dec, the disk torque is subdominant
to GW emission and contributes a fractional perturbation

    epsilon(f) = fdot_disk / fdot_GW = (f_dec / f)^alpha

to the inspiral rate, where alpha depends on the disk model:

  - alpha = 2    :  constant da/dt (hardening rate)
  - alpha = 7/3  :  constant torque dJ/dt
  - alpha = 10/3 :  constant energy dissipation rate dE/dt

Derivation (all for circular orbits):
  fdot_GW propto f^{11/3}  (leading Newtonian order)
  f propto a^{-3/2}, so df/da propto f^{5/3}

  constant da/dt:
    fdot_disk = (df/da)(da/dt) propto f^{5/3}
    epsilon = f^{5/3}/f^{11/3} = f^{-2}  -->  alpha = 2

  constant dJ/dt (torque):
    dJ/da propto a^{-1/2} propto f^{1/3}
    da/dt = T/(dJ/da) propto f^{-1/3}
    fdot_disk propto f^{5/3} * f^{-1/3} = f^{4/3}
    epsilon = f^{4/3}/f^{11/3} = f^{-7/3}  -->  alpha = 7/3

  constant dE/dt (energy dissipation):
    dE/da propto a^{-2} propto f^{4/3}
    da/dt propto f^{-4/3}
    fdot_disk propto f^{5/3} * f^{-4/3} = f^{1/3}
    epsilon = f^{1/3}/f^{11/3} = f^{-10/3}  -->  alpha = 10/3

The previous version of this script used alpha = 5/3, which does not
correspond to any standard physical assumption. The correct exponents
are all >= 2 (steeper), meaning epsilon falls off faster with frequency
and the dephasing is smaller than previously estimated.

Equations used:
--------------
  fdot_GW:  Peters 1964 Eq. 5.6, rewritten for f and Mc
            (96/5) pi^{8/3} (G Mc/c^3)^{5/3} f^{11/3}

  epsilon:  epsilon = fdot_disk / fdot_GW = (f_dec/f)^alpha
            This parameterization follows Haiman, Kocsis & Menou 2009
            (ApJ 700, 1952), Sec. 3.2, where f_dec is the frequency at
            which disk and GW torques are equal.  The exponent alpha is
            set by the assumed disk model (see derivation above).

  Phase integration:  direct Euler integration of
            dphi = 2 pi f dt,  df = -fdot (1 + eps) dt
            backward from f_E for look-back time tau.

References:
-----------
  Peters 1964, Phys. Rev. 136, B1224 (GW inspiral)
  Haiman, Kocsis & Menou 2009, ApJ 700, 1952 (disk torque framework, Sec. 3.2)
  Siwek, Weinberger & Hernquist 2023, MNRAS 522, 2707 (orbital evolution from CBD torques, arXiv:2302.01785)
  Siwek, Kelley & Hernquist 2024, MNRAS 534, 2609 (CBD population study, arXiv:2403.08871)
  Tiede, Zrake, MacFadyen & Haiman 2025, ApJ 984, 144 (suppressed accretion in thin disks, arXiv:2410.03830)

  NOTE: The scaling exponents alpha = 2, 7/3, 10/3 are derived from first
  principles (Kepler + assumed conserved quantity), not taken from any single
  paper. No published simulation has explicitly measured the frequency
  dependence of epsilon above decoupling.

Author: Chiara Mingarelli / Claude (verification script)
"""
import numpy as np

# --- Constants ---
G = 6.67430e-11       # m^3 kg^-1 s^-2
c = 2.99792458e8      # m/s
Msun = 1.98892e30     # kg
pc = 3.08567758e16    # m
yr = 365.25 * 86400   # s

# --- Binary parameters (optimistic source) ---
M_tot = 1e9 * Msun    # total mass
eta = 0.25             # symmetric mass ratio (equal mass)
Mc = eta**0.6 * M_tot  # chirp mass
f_E = 1e-6             # Earth-term frequency [Hz]

# --- GW frequency derivative (leading Newtonian order) ---
def fdot_gw(f):
    """Newtonian GW frequency derivative.

    df/dt = (96/5) pi^{8/3} (G Mc / c^3)^{5/3} f^{11/3}

    This is the leading-order (0pN) GW-driven frequency evolution
    for circular orbits.  Derived from Peters 1964 Eq. 5.6 rewritten
    in terms of GW frequency f = 2 f_orb and chirp mass Mc.
    See also Maggiore (2007) Eq. 4.21.
    """
    return (96.0 / 5.0) * np.pi**(8.0/3.0) * (G * Mc / c**3)**(5.0/3.0) * f**(11.0/3.0)

# --- Disk perturbation ---
def epsilon(f, f_dec, alpha):
    """Fractional disk perturbation above decoupling.

    epsilon(f) = (f_dec / f)^alpha

    Following Haiman+ 2009 Sec. 3.2: at f = f_dec the disk and GW
    torques balance, so epsilon(f_dec) = 1.  Above f_dec the disk
    contribution falls as a power law whose index alpha depends on
    the disk physics (see module docstring for derivation of each case).

    Parameters
    ----------
    f : float
        GW frequency [Hz]
    f_dec : float
        Decoupling frequency [Hz]
    alpha : float
        Scaling exponent: 2 (const da/dt), 7/3 (const torque),
        or 10/3 (const dE/dt)
    """
    return (f_dec / f)**alpha

# --- Evolve backward from f_E and compute phase ---
def compute_dephasing(f_dec_nHz, tau_yr, alpha):
    """
    Evolve the binary backward from f_E for time tau,
    with and without disk torque.  Return dephasing in radians.

    Parameters
    ----------
    f_dec_nHz : float
        Decoupling frequency in nHz
    tau_yr : float
        Look-back time in years
    alpha : float
        Disk torque scaling exponent

    Returns
    -------
    delta_phi : float
        Accumulated dephasing [rad] (positive = disk adds cycles)
    f_P_vac : float
        Pulsar-term frequency in vacuum [Hz]
    f_P_disk : float
        Pulsar-term frequency with disk [Hz]
    N_vac : float
        Total vacuum cycles
    eps_at_fP : float
        epsilon evaluated at f_P_vac
    """
    f_dec = f_dec_nHz * 1e-9  # Hz
    tau = tau_yr * yr          # seconds

    # Evolve backward with small timesteps
    N_steps = max(10000, int(tau_yr * 10))
    dt = tau / N_steps

    # Vacuum evolution
    f_vac = f_E
    phi_vac = 0.0

    # Disk evolution
    f_disk = f_E
    phi_disk = 0.0

    for i in range(N_steps):
        # Phase accumulation
        phi_vac += 2.0 * np.pi * f_vac * dt
        phi_disk += 2.0 * np.pi * f_disk * dt

        # Frequency step backward
        f_vac -= fdot_gw(f_vac) * dt

        eps = epsilon(f_disk, f_dec, alpha) if f_disk > f_dec else 1.0
        f_disk -= fdot_gw(f_disk) * (1.0 + eps) * dt

    delta_phi = phi_disk - phi_vac
    N_vac = phi_vac / (2.0 * np.pi)
    eps_fP = epsilon(f_vac, f_dec, alpha) if f_vac > f_dec else 1.0

    return delta_phi, f_vac, f_disk, N_vac, eps_fP


# =====================================================================
# Main calculation
# =====================================================================
print("=" * 80)
print("Circumbinary disk dephasing over echo baselines")
print(f"Binary: M_tot = {M_tot/Msun:.0e} Msun, equal mass, f_E = {f_E*1e6:.0f} uHz")
print("=" * 80)

# --- Three disk models ---
models = [
    (2.0,    "constant da/dt"),
    (7.0/3,  "constant torque (dJ/dt)"),
    (10.0/3, "constant dE/dt"),
]

f_dec_values = [1, 3, 5, 10]   # nHz
tau_values = [500, 1000, 2500, 3800, 7670]  # yr

for alpha, label in models:
    print(f"\n{'='*80}")
    print(f"Model: alpha = {alpha:.4g}  ({label})")
    print(f"  epsilon(f) = (f_dec/f)^{alpha:.4g}")
    print(f"{'='*80}")

    print(f"\n{'f_dec [nHz]':>12s} {'tau [yr]':>10s} {'f_P [nHz]':>10s} "
          f"{'eps(f_P)':>10s} {'N_vac':>10s} {'dPhi [rad]':>12s} {'dPhi [cyc]':>12s}")
    print("-" * 82)

    for f_dec in f_dec_values:
        for tau in tau_values:
            dphi, f_P, f_P_d, N, eps_fP = compute_dephasing(f_dec, tau, alpha)
            print(f"{f_dec:>12d} {tau:>10d} {f_P*1e9:>10.1f} "
                  f"{eps_fP:>10.2e} {N:>10.0f} {dphi:>12.1f} {dphi/(2*np.pi):>12.1f}")
        print()

# --- Key numbers for the paper (f_dec = 3 nHz) ---
print("\n" + "=" * 80)
print("KEY NUMBERS FOR PAPER: f_dec = 3 nHz, comparison across models")
print("=" * 80)

for alpha, label in models:
    dphi_1k, fP_1k, _, N_1k, eps_1k = compute_dephasing(3, 1000, alpha)
    dphi_4k, fP_4k, _, N_4k, eps_4k = compute_dephasing(3, 3800, alpha)
    dphi_8k, fP_8k, _, N_8k, eps_8k = compute_dephasing(3, 7670, alpha)

    print(f"\n  Model: alpha = {alpha:.4g} ({label})")
    print(f"    tau = 1000 yr:  eps(f_P) = {eps_1k:.1e}, "
          f"dPhi = {dphi_1k:.1f} rad ({dphi_1k/(2*np.pi):.1f} cycles)")
    print(f"    tau = 3800 yr:  eps(f_P) = {eps_4k:.1e}, "
          f"dPhi = {dphi_4k:.1f} rad ({dphi_4k/(2*np.pi):.1f} cycles)")
    print(f"    tau = 7670 yr:  eps(f_P) = {eps_8k:.1e}, "
          f"dPhi = {dphi_8k:.1f} rad ({dphi_8k/(2*np.pi):.1f} cycles)")

# --- Comparison: old (wrong) 5/3 vs corrected ---
print("\n" + "=" * 80)
print("COMPARISON: old alpha=5/3 (WRONG) vs corrected models")
print("f_dec = 3 nHz, tau = 3800 yr")
print("=" * 80)

for alpha, label in [(5.0/3, "WRONG (old paper)"), (2.0, "const da/dt"),
                      (7.0/3, "const torque"), (10.0/3, "const dE/dt")]:
    dphi, _, _, _, eps = compute_dephasing(3, 3800, alpha)
    print(f"  alpha = {alpha:.4g} ({label:25s}): dPhi = {dphi:8.1f} rad = {dphi/(2*np.pi):6.1f} cycles")

# --- Minimum detectable epsilon ---
print(f"\n{'='*80}")
print("Minimum detectable eps for 1-rad dephasing (model-independent)")
print("=" * 80)
for tau, label in [(1000, "1000 yr"), (3800, "3800 yr")]:
    dphi, fP, _, N, _ = compute_dephasing(3, tau, 7.0/3)  # N_vac is model-independent
    eps_min = 1.0 / (2 * np.pi * N)
    print(f"  tau = {label}: N = {N:.0f} cycles, eps_min = {eps_min:.1e}")
