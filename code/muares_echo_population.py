r"""
muAres Echo Source Population: LM24 BHMF + Phinney Merger Rates
===============================================================

Computes the expected SMBHB population detectable by muAres and identifies
the most likely echo sources.  Draws realistic golden and typical binaries
from the mass function and plots their pulsar-term / earth-term positions
on the GW sensitivity landscape.


Derivation
----------

1. LM24 Black Hole Mass Function (Schechter form, LM24 Eq. 3)

    dn/d(ln M) = phi * (M/M_s)^{alpha+1} * exp[-(M/M_s)^beta]

    with alpha = -1.27, beta = 0.45, phi = 10^{-2.00} Mpc^{-3},
    M_s = 10^{8.09} M_sun.  This gives the comoving number density of
    SMBHs per unit ln(M).


2. Phinney (2001) GWB formula and merger rate inversion

    The characteristic strain from circular, GW-driven SMBHBs is:

        h_c^2(f) = C * f^{-4/3} * R0 * F_z * I_M

    where:
        C    = 4 G^{5/3} / (3 pi^{1/3} c^2)
        R0   = merger event rate per primary SMBH  [s^{-1}]
        F_z  = int_0^inf dz / [(1+z)^{4/3} H(z)]  [s]
             = 3.44e17 s = 10.9 Gyr  (cosmological time integral)
        I_M  = int d(ln M) * [dn/d(ln M)]_{SI} * M^{5/3} * <q(1+q)^{-1/3}>
             (mass integral with BHMF converted to m^{-3} and M in kg)

    The mass-ratio distribution is p(q) = 3 q^2 on [0,1], giving
        <q(1+q)^{-1/3}> = 0.6178

    Solving for R0:
        R0 = h_c^2 * f^{4/3} / (C * F_z * I_M)

    Calibrating to NANOGrav 15yr h_c = 2.4e-15 at f = 1/yr:
        R0 = 0.0485 Gyr^{-1} per primary SMBH

    Verification: setting R0 = H0/2 (one merger per Hubble time per BH)
    recovers h_c = 2.02e-15, consistent with LM24's reported ~2.0e-15.


3. ISCO frequency and residence time

    The Schwarzschild ISCO GW frequency for total mass M is:
        f_ISCO = c^3 / (6^{3/2} pi G M)

    The residence time in the muAres band [f_lo, f_hi] for a circular,
    GW-driven binary with chirp mass Mc is:
        T_band = (5/256) * (pi Mc_s)^{-5/3} * (f_lo^{-8/3} - f_hi^{-8/3})

    where Mc_s = G Mc / c^3 is the chirp mass in seconds, and f_hi is
    capped at f_ISCO.


4. Number of sources in band at any instant

    The differential detection rate per unit log10(M) is:
        dN/dlog10(M) = R0 * [dn/dlog10(M)] * T_band(M) * V(M)

    where V(M) = (4pi/3) D_hor^3 is the comoving volume within which
    muAres can detect a source of mass M.


5. muAres horizon (monochromatic SNR approximation)

    For a quasi-monochromatic source at frequency f with strain amplitude
    h_0, observed for T_obs:
        SNR = h_0 * sqrt(f * T_obs) / h_n(f)

    where h_n(f) = sqrt(f * S_n(f)) is the detector characteristic strain
    noise.  The horizon distance for SNR >= 8:
        D_hor = (SNR at 1 Mpc) / 8

    This is capped at the comoving distance to z = 10 (~9634 Mpc).
    For all masses M > 10^7 M_sun, muAres reaches z = 10, so the
    detection rate is set entirely by dn/dlnM * T_band, not by the
    horizon.  (The chirp-integrated SNR would be higher, but the volume
    is already saturated.)


6. Strain amplitude h_0(f) for a circular binary

    The dimensionless strain amplitude at GW frequency f is:
        h_0 = (4/D_L) * (G Mc / c^2)^{5/3} * (pi f / c)^{2/3}

    On a characteristic strain plot, the effective h_c for a source
    observed for time T_obs is:
        h_c = h_0 * sqrt(f * T_obs)


7. Pulsar-term frequency (chirp look-back)

    A pulsar at distance d_p and angle theta from the GW source sees
    the binary at a geometric delay:
        tau = d_p * (1 - cos theta) / c

    The GW frequency at look-back time tau before the earth-term
    measurement at f_E is:
        f_P = f_E * (1 + (256/5) pi^{8/3} (G Mc/c^3)^{5/3} f_E^{8/3} tau)^{-3/8}


References
----------
- Liepold & Ma (2024), ApJ 974, 209  [LM24]
- Phinney (2001), astro-ph/0108028
- Sesana et al. (2021), Exp. Astron. 51, 1333  [muAres]
- Agazie et al. (2023), ApJL 951, L8  [NANOGrav 15yr]
- Mingarelli et al. (2012), PRL 109, 081104

Author: Mingarelli group
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# =====================================================================
# Physical constants (SI)
# =====================================================================
G     = 6.67430e-11       # m^3 kg^-1 s^-2
c     = 2.99792458e8      # m/s
Msun  = 1.98892e30        # kg
pc    = 3.08567758e16     # m
Mpc   = 1e6 * pc
yr    = 365.25 * 86400    # s
H0    = 67.4e3 / Mpc      # s^-1  (Planck 2018)
Om    = 0.315
OL    = 0.685

# =====================================================================
# 1.  LM24 Black Hole Mass Function
# =====================================================================
# Schechter form, LM24 Eq. (3): dn/d(ln M) in Mpc^-3
ALPHA_LM = -1.27
BETA_LM  =  0.45
PHI_LM   = 10**(-2.00)   # Mpc^-3
MS_LM    = 10**(8.09)    # Msun


def bhmf_lm24(M_msun):
    """dn/d(ln M_BH) [Mpc^-3].  Schechter form from LM24 Eq. (3)."""
    x = np.asarray(M_msun, dtype=float) / MS_LM
    return PHI_LM * x**(ALPHA_LM + 1) * np.exp(-x**BETA_LM)


def n_above(M_min, log10M_hi=11.5):
    """Cumulative number density n(>M_min) [Mpc^-3]."""
    integrand = lambda lm: bhmf_lm24(10**lm) / np.log(10)
    result, _ = quad(integrand, np.log10(M_min), log10M_hi, limit=200)
    return result


# =====================================================================
# 2.  Cosmological helpers
# =====================================================================
# E(z) = H(z)/H0 for flat Lambda-CDM
def Ez(z):
    return np.sqrt(Om * (1 + z)**3 + OL)


def comoving_distance(z_max):
    """Comoving distance D_c(z) [Mpc].  D_c = int_0^z c dz' / H(z')."""
    result, _ = quad(lambda z: c / (H0 * Ez(z) * Mpc), 0, z_max, limit=200)
    return result


# Build z <-> D_c lookup (used for horizon conversions)
_z_tab  = np.linspace(0, 15, 600)
_Dc_tab = np.array([comoving_distance(z) for z in _z_tab])


def z_at_Dc(Dc_Mpc):
    return float(np.interp(Dc_Mpc, _Dc_tab, _z_tab))


# Cosmological time integral for the Phinney formula (Sec. 2 of docstring):
#   F_z = int_0^inf dz / [(1+z)^{4/3} H(z)]   [seconds]
# This arises from converting the comoving merger rate density to the
# observed GWB, accounting for redshift of both frequency and energy.
F_z, _ = quad(lambda z: 1.0 / ((1 + z)**(4./3) * Ez(z) * H0), 0, 20,
              limit=200)


# =====================================================================
# 3.  Phinney inversion → merger rate R0
# =====================================================================
# We parameterize the coalescence rate as (Sec. 2 of docstring):
#   d(n-dot) / (dM1 dq) = R0 * (dn_BH/dM1) * p(q)
#
# Mass-ratio distribution: p(q) = 3 q^2 on [0,1]  (NANOGrav population)
# The q-averaged chirp mass factor is:
#   <q (1+q)^{-1/3}> = int_0^1 3 q^2 * q * (1+q)^{-1/3} dq
Q_MC_AVG, _ = quad(lambda q: 3 * q**3 * (1 + q)**(-1./3), 0, 1)

# Phinney constant:  C = 4 G^{5/3} / (3 pi^{1/3} c^2)
C_phinney = 4 * G**(5./3) / (3 * np.pi**(1./3) * c**2)


def mass_integral(log10M_lo=7.0, log10M_hi=11.5):
    r"""Mass integral I_M (Sec. 2 of docstring).

    I_M = \int d(\ln M) \cdot [dn/d(\ln M)]_{\rm SI} \cdot M_{\rm kg}^{5/3}
          \cdot \langle q(1+q)^{-1/3} \rangle

    The BHMF is in Mpc^{-3}; we convert to m^{-3} by dividing by Mpc^3.
    The integral variable is log10(M), so we include a factor of ln(10)
    for the d(ln M) = ln(10) d(log10 M) Jacobian.
    """
    def integrand(log10M):
        M_msun = 10**log10M
        M_kg   = M_msun * Msun
        dn_SI  = bhmf_lm24(M_msun) / Mpc**3   # Mpc^{-3} -> m^{-3}
        return dn_SI * M_kg**(5./3) * Q_MC_AVG * np.log(10)
    result, _ = quad(integrand, log10M_lo, log10M_hi, limit=200)
    return result


I_M = mass_integral()

# Solve for R0 from the Phinney formula (Sec. 2 of docstring):
#   h_c^2(f) = C * f^{-4/3} * R0 * F_z * I_M
#   =>  R0 = h_c^2 * f^{4/3} / (C * F_z * I_M)
hc_NG15 = 2.4e-15          # NANOGrav 15yr measured amplitude
f_ref   = 1.0 / yr         # reference frequency = 1/yr
R0      = hc_NG15**2 * f_ref**(4./3) / (C_phinney * F_z * I_M)  # s^-1
R0_yr   = R0 * yr           # yr^-1
R0_Gyr  = R0 * 1e9 * yr     # Gyr^-1


# =====================================================================
# 4.  Source physics: chirp mass, ISCO, residence time
# =====================================================================
def chirp_mass_msun(M_total, q=1.0):
    """Chirp mass [M_sun].  Mc = eta^{3/5} M_total, eta = q/(1+q)^2."""
    eta = q / (1 + q)**2
    return eta**(3./5) * M_total


def f_isco(M_total):
    """Schwarzschild ISCO GW frequency [Hz] (Sec. 3 of docstring).

    f_ISCO = c^3 / (6^{3/2} pi G M)
    """
    return c**3 / (6**1.5 * np.pi * G * M_total * Msun)


def T_band_yr(M_total, q=1.0, f_lo=0.1e-6, f_hi=100e-6):
    """Residence time in [f_lo, f_hi] for a circular GW-driven binary [yr].

    (Sec. 3 of docstring)
    T_band = (5/256) (pi Mc_s)^{-5/3} (f_lo^{-8/3} - f_hi^{-8/3})

    where Mc_s = G Mc / c^3 and f_hi is capped at f_ISCO.
    """
    Mc_s = G * chirp_mass_msun(M_total, q) * Msun / c**3
    f_hi_eff = min(f_hi, f_isco(M_total))
    if f_lo >= f_hi_eff:
        return 0.0
    T_s = (5 / (256 * np.pi**(8./3))) * Mc_s**(-5./3) * (
        f_lo**(-8./3) - f_hi_eff**(-8./3))
    return T_s / yr


# =====================================================================
# 5.  muAres sensitivity (Sesana et al. 2021)
# =====================================================================
def muares_sensitivity(T_obs_yr=10, acc=1.0, f_min=1e-7, f_max=1e-1,
                       nfreqs=2000):
    """Return (f, h_n) for muAres characteristic strain noise."""
    L = 3.95e11;  S_pos = 2.5e-21;  f_knee = 1e-4
    S_acc_amp = (acc * 1e-15)**2
    f = np.logspace(np.log10(f_min), np.log10(f_max), nfreqs)
    f_star = c / (2 * np.pi * L)
    S_acc = S_acc_amp * (1 + (f_knee / f)**2)
    Sn = (20./3) / L**2 * (4 * S_acc / (2*np.pi*f)**4 + S_pos) * (
        1 + (f / f_star)**2)
    # Galactic confusion noise
    A_c = 1.4e-44;  f_k = 0.0016 * T_obs_yr**(-2./9)
    gamma = 1100 * T_obs_yr**(3./10)
    Sc = A_c * f**(-7./3) * (1 + np.tanh(gamma * (f_k - f)))
    Sn += Sc
    return f, np.sqrt(f * Sn)


f_mu, h_mu = muares_sensitivity()


def muares_horizon_Mpc(M_total, q=1.0, snr_thr=8, z_cap=10):
    """Max luminosity distance [Mpc] for muAres detection (Sec. 5 of docstring).

    Uses the monochromatic SNR approximation:
        SNR = h_0(f) * sqrt(f * T_obs) / h_n(f)

    evaluated at 1 Mpc, then scaled by 1/D_L to find the horizon.
    The source is restricted to f <= f_ISCO, and D_hor is capped at
    the comoving distance to z_cap.
    """
    Mc_kg = chirp_mass_msun(M_total, q) * Msun
    D_1Mpc = Mpc
    # h_0 at D_L = 1 Mpc across the muAres frequency grid (Sec. 6)
    h0_1 = (4.0 / D_1Mpc) * (G * Mc_kg / c**2)**(5./3) * (
        np.pi * f_mu / c)**(2./3)
    # Monochromatic SNR at 1 Mpc with T_obs = 10 yr
    snr_1 = h0_1 * np.sqrt(f_mu * 10 * yr) / h_mu
    fi = f_isco(M_total)
    mask = f_mu <= fi
    if not np.any(mask):
        return 0.0, 0.0
    best = np.argmax(snr_1[mask])
    # Horizon = (SNR at 1 Mpc) / threshold  [since SNR ~ 1/D_L]
    D_hor = float(snr_1[mask][best]) / snr_thr   # Mpc
    f_best = float(f_mu[mask][best])
    D_cap = comoving_distance(z_cap) if z_cap else 1e30
    return min(D_hor, D_cap), f_best


# =====================================================================
# 6.  Differential detection rate dN/dlog10(M)
# =====================================================================
# Sec. 4 of docstring:
#   dN/dlog10(M) = R0 * [dn/dlog10(M)] * T_band(M) * V_hor(M)
#
# where  dn/dlog10(M) = [dn/d(ln M)] / ln(10)
# and    V_hor = (4pi/3) D_hor^3   is the comoving detection volume.

log10M_grid = np.arange(7.0, 11.01, 0.1)
M_grid      = 10**log10M_grid

dn_dlnM_grid  = np.array([bhmf_lm24(M) for M in M_grid])      # Mpc^{-3}
T_band_grid   = np.array([T_band_yr(M) for M in M_grid])       # yr
D_hor_grid    = np.array([muares_horizon_Mpc(M)[0] for M in M_grid])  # Mpc
V_hor_grid    = (4 * np.pi / 3) * D_hor_grid**3                # Mpc^3
dN_dlogM_grid = R0_yr * (dn_dlnM_grid / np.log(10)) * T_band_grid * V_hor_grid


# =====================================================================
# 7.  Echo source computation (Mingarelli+2012)
# =====================================================================
def compute_echo_sources(M_total, q, D_L_Mpc, f_earth,
                         n_pulsars=20, T_pta_yr=25.0, T_muares_yr=10.0,
                         seed=42):
    """
    Compute earth-term and pulsar-term (f, h_c) for an SMBHB.

    Returns
    -------
    earth   : dict  {f, hc, h0, f_isco}
    pulsars : list of dicts  {f, hc, h0, d_kpc, tau_yr, cos_theta}
    """
    rng = np.random.default_rng(seed)
    eta   = q / (1 + q)**2
    Mc_kg = M_total * Msun * eta**0.6
    D_L_m = D_L_Mpc * 1e6 * pc
    fi    = f_isco(M_total)

    # Clamp to ISCO
    f_E = min(f_earth, 0.9 * fi)

    def h0(f):
        return (4.0 / D_L_m) * (G * Mc_kg / c**2)**(5./3) * (
            np.pi * f / c)**(2./3)

    coeff = (256./5) * np.pi**(8./3) * (G * Mc_kg / c**3)**(5./3)

    def f_pulsar(tau_s):
        return f_E * (1 + coeff * f_E**(8./3) * tau_s)**(-3./8)

    h0_E  = h0(f_E)
    hc_E  = h0_E * np.sqrt(f_E * T_muares_yr * yr)
    earth = dict(f=f_E, hc=hc_E, h0=h0_E, f_isco=fi)

    # Random pulsars: d ~ 0.2-5 kpc (log-uniform), isotropic angles
    d_kpc     = 10**rng.uniform(np.log10(0.2), np.log10(5.0), n_pulsars)
    cos_theta = rng.uniform(-1, 1, n_pulsars)

    pulsars = []
    for i in range(n_pulsars):
        d_m   = d_kpc[i] * 1e3 * pc
        tau_s = d_m * (1 - cos_theta[i]) / c
        if tau_s < 1.0:
            continue
        fp = f_pulsar(tau_s)
        if fp < 1e-10 or fp > 1e0:
            continue
        h0_p  = h0(fp)
        hc_p  = h0_p * np.sqrt(fp * T_pta_yr * yr)
        pulsars.append(dict(f=fp, hc=hc_p, h0=h0_p,
                            d_kpc=d_kpc[i], tau_yr=tau_s / yr,
                            cos_theta=cos_theta[i]))
    return earth, pulsars


# =====================================================================
# 8.  Compute the two fiducial echo sources
# =====================================================================
golden_earth, golden_pulsars = compute_echo_sources(
    M_total=1e9, q=1.0, D_L_Mpc=100, f_earth=1e-6,
    seed=42)

typical_earth, typical_pulsars = compute_echo_sources(
    M_total=1e8, q=1.0, D_L_Mpc=500, f_earth=1e-5,
    seed=137)


# =====================================================================
# 9.  Print summary tables
# =====================================================================
print("=" * 72)
print("  LM24 BHMF + Phinney Merger Rate Inversion")
print("  Calibrated to NANOGrav 15yr h_c = {:.1e}".format(hc_NG15))
print("=" * 72)
print()
print(f"  R0 = {R0_Gyr:.4f} Gyr^-1 per primary SMBH")
print(f"  BH participation rate ~ {2*R0_Gyr:.3f} Gyr^-1")
print(f"  F_z = {F_z:.4e} s = {F_z/(1e9*yr):.2f} Gyr")
print(f"  Q_MC_AVG = <q(1+q)^(-1/3)> = {Q_MC_AVG:.4f}")
print(f"  I_M = {I_M:.4e} m^-3 kg^(5/3)")
print()

# Verification
hc_check = np.sqrt(C_phinney * f_ref**(-4./3) * (H0/2) * F_z * I_M)
print(f"  Verification: R0 = H0/2 => h_c = {hc_check:.2e}  (LM24: ~2.0e-15)")
print()

print("-" * 72)
print(f"{'log10(M)':>9s} {'dn/dlnM':>10s} {'n(>M)':>10s} {'T_band':>10s}"
      f" {'f_ISCO':>8s} {'D_hor':>8s} {'z_hor':>6s} {'dN/dlogM':>10s}")
print(f"{'':>9s} {'Mpc-3':>10s} {'Mpc-3':>10s} {'yr':>10s}"
      f" {'uHz':>8s} {'Mpc':>8s} {'':>6s} {'':>10s}")
print("-" * 72)

for i in range(0, len(log10M_grid), 5):
    lm = log10M_grid[i]
    M  = M_grid[i]
    print(f" {lm:8.1f} {dn_dlnM_grid[i]:10.2e} {n_above(M):10.2e}"
          f" {T_band_grid[i]:10.1e} {f_isco(M)*1e6:8.1f}"
          f" {D_hor_grid[i]:8.0f} {z_at_Dc(D_hor_grid[i]):6.1f}"
          f" {dN_dlogM_grid[i]:10.2e}")

print()
print("Integrated N in muAres band at any instant (z < 10):")
for Mlo, Mhi, lab in [(1e7, 1e8, "10^7 - 10^8"),
                       (1e8, 3e8, "10^8 - 3x10^8"),
                       (3e8, 1e9, "3x10^8 - 10^9"),
                       (1e9, 1e10, "10^9 - 10^10")]:
    mask = (M_grid >= Mlo) & (M_grid < Mhi)
    sub = np.trapezoid(dN_dlogM_grid[mask], log10M_grid[mask])
    print(f"  M = {lab:15s}:  {sub:10.0f}")

print()
print("=" * 72)
print("  FIDUCIAL ECHO SOURCES")
print("=" * 72)

for label, earth, pulsars, M, q, DL in [
    ("Golden binary", golden_earth, golden_pulsars, 1e9, 1.0, 100),
    ("Typical binary", typical_earth, typical_pulsars, 1e8, 1.0, 500),
]:
    fp_all  = [p['f'] for p in pulsars]
    hc_all  = [p['hc'] for p in pulsars]
    h0_all  = [p['h0'] for p in pulsars]
    tau_all = [p['tau_yr'] for p in pulsars]
    d_all   = [p['d_kpc'] for p in pulsars]

    print(f"\n  {label}: M = {M:.0e} Msun, q = {q}, D_L = {DL} Mpc")
    print(f"    f_ISCO  = {earth['f_isco']*1e6:.1f} uHz")
    print(f"    Earth term:  f_E = {earth['f']*1e6:.1f} uHz,"
          f"  h_0 = {earth['h0']:.2e},  h_c = {earth['hc']:.2e}")
    print(f"    {len(pulsars)} pulsar terms:")
    print(f"      f_P    = {min(fp_all)*1e9:.0f} - {max(fp_all)*1e9:.0f} nHz")
    print(f"      h_0    = {min(h0_all):.2e} - {max(h0_all):.2e}")
    print(f"      h_c    = {min(hc_all):.2e} - {max(hc_all):.2e}")
    print(f"      tau    = {min(tau_all):.0f} - {max(tau_all):.0f} yr")
    print(f"      d_psr  = {min(d_all):.1f} - {max(d_all):.1f} kpc")
    print()
    print(f"    {'i':>3s} {'d_kpc':>6s} {'cos_th':>7s} {'tau_yr':>8s}"
          f" {'f_P [nHz]':>10s} {'h_0':>10s} {'h_c':>10s}")
    print(f"    {'-'*60}")
    for j, p in enumerate(pulsars):
        print(f"    {j:3d} {p['d_kpc']:6.1f} {p['cos_theta']:7.3f}"
              f" {p['tau_yr']:8.0f} {p['f']*1e9:10.1f}"
              f" {p['h0']:10.2e} {p['hc']:10.2e}")


# =====================================================================
# 10.  Figures
# =====================================================================
plt.rcParams.update({
    'font.size': 12, 'axes.linewidth': 1.2,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
})

# ----- Figure 1: LM24 BHMF and dN/dlog10M -----
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): BHMF
log10M_fine = np.linspace(6.5, 11.5, 500)
M_fine = 10**log10M_fine
dn_fine = np.array([bhmf_lm24(M) for M in M_fine])

ax1.semilogy(log10M_fine, dn_fine, 'k-', lw=2.5)
ax1.axvspan(np.log10(3e8), np.log10(1e10), alpha=0.12, color='red',
            label='Echo-source range')
ax1.axvline(np.log10(1e8), color='#E66100', ls='--', lw=1.5,
            label=r'Typical ($10^8\,M_\odot$)')
ax1.axvline(np.log10(1e9), color='#377EB8', ls='--', lw=1.5,
            label=r'Golden ($10^9\,M_\odot$)')
ax1.set_xlabel(r'$\log_{10}(M_{\rm BH}\,/\,M_\odot)$')
ax1.set_ylabel(r'$dn/d\ln M$ [Mpc$^{-3}$]')
ax1.set_title('(a) LM24 Black Hole Mass Function')
ax1.set_xlim(6.5, 11.5)
ax1.set_ylim(1e-12, 1e-1)
ax1.legend(fontsize=10, loc='lower left')
ax1.grid(True, alpha=0.3)

# Panel (b): Differential detection rate
ax2.semilogy(log10M_grid, dN_dlogM_grid, 'k-', lw=2.5)
ax2.axvspan(np.log10(3e8), np.log10(1e10), alpha=0.12, color='red')
ax2.axvline(np.log10(1e8), color='#E66100', ls='--', lw=1.5)
ax2.axvline(np.log10(1e9), color='#377EB8', ls='--', lw=1.5)
ax2.axhline(1, color='gray', ls=':', lw=1)
ax2.set_xlabel(r'$\log_{10}(M_{\rm BH}\,/\,M_\odot)$')
ax2.set_ylabel(r'$dN/d\log_{10}M$ [in $\mu$Ares band at any instant]')
ax2.set_title(r'(b) $\mu$Ares detection rate ($z < 10$, SNR $> 8$)')
ax2.set_xlim(7, 11)
ax2.set_ylim(1e-4, 1e7)
ax2.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig('fig_bhmf_detection_rate.png', dpi=150, bbox_inches='tight')
fig1.savefig('fig_bhmf_detection_rate.pdf', bbox_inches='tight')
print("\nSaved fig_bhmf_detection_rate.pdf/png")


# ----- Figure 2: Residence time and f_ISCO vs mass -----
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5))

T_fine = np.array([T_band_yr(M) for M in M_fine])
fi_fine = np.array([f_isco(M) * 1e6 for M in M_fine])

ax3.loglog(M_fine, T_fine, 'k-', lw=2.5)
ax3.axvspan(3e8, 1e10, alpha=0.12, color='red')
ax3.axvline(1e8, color='#E66100', ls='--', lw=1.5)
ax3.axvline(1e9, color='#377EB8', ls='--', lw=1.5)
ax3.set_xlabel(r'$M_{\rm total}$ [$M_\odot$]')
ax3.set_ylabel(r'$T_{\rm band}$ [yr]')
ax3.set_title(r'(a) Residence time in $\mu$Ares band (0.1–100 $\mu$Hz)')
ax3.set_xlim(1e7, 1e11)
ax3.grid(True, alpha=0.3, which='both')

ax4.loglog(M_fine, fi_fine, 'k-', lw=2.5)
ax4.axhspan(0.1, 100, alpha=0.08, color='purple', label=r'$\mu$Ares band')
ax4.axvspan(3e8, 1e10, alpha=0.12, color='red')
ax4.axvline(1e8, color='#E66100', ls='--', lw=1.5)
ax4.axvline(1e9, color='#377EB8', ls='--', lw=1.5)
ax4.axhline(10, color='#E66100', ls=':', lw=1, alpha=0.6)
ax4.axhline(1, color='#377EB8', ls=':', lw=1, alpha=0.6)
ax4.set_xlabel(r'$M_{\rm total}$ [$M_\odot$]')
ax4.set_ylabel(r'$f_{\rm ISCO}$ [$\mu$Hz]')
ax4.set_title('(b) ISCO frequency vs total mass')
ax4.set_xlim(1e7, 1e11)
ax4.set_ylim(0.01, 1e4)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

fig2.tight_layout()
fig2.savefig('fig_Tband_fisco.png', dpi=150, bbox_inches='tight')
fig2.savefig('fig_Tband_fisco.pdf', bbox_inches='tight')
print("Saved fig_Tband_fisco.pdf/png")


# ----- Figure 3: Pulsar distance & angle distributions for both sources -----
fig3, axes = plt.subplots(2, 3, figsize=(15, 9))

for row, (label, pulsars, color) in enumerate([
    ("Golden binary (10⁹ M☉, 100 Mpc)", golden_pulsars, '#377EB8'),
    ("Typical binary (10⁸ M☉, 500 Mpc)", typical_pulsars, '#E66100'),
]):
    d_arr   = [p['d_kpc'] for p in pulsars]
    ct_arr  = [p['cos_theta'] for p in pulsars]
    tau_arr = [p['tau_yr'] for p in pulsars]
    fp_arr  = [p['f'] * 1e9 for p in pulsars]
    hc_arr  = [p['hc'] for p in pulsars]

    # (a) Pulsar distance distribution
    ax = axes[row, 0]
    ax.hist(d_arr, bins=8, color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pulsar distance [kpc]')
    ax.set_ylabel('Count')
    ax.set_title(f'{label}\n(a) Distance distribution')

    # (b) Look-back time vs frequency
    ax = axes[row, 1]
    ax.scatter(tau_arr, fp_arr, c=color, edgecolors='black', s=60, zorder=3)
    ax.set_xlabel(r'Look-back time $\tau$ [yr]')
    ax.set_ylabel(r'$f_P$ [nHz]')
    ax.set_title('(b) Pulsar-term frequency vs look-back')
    ax.grid(True, alpha=0.3)

    # (c) h_c vs frequency
    ax = axes[row, 2]
    ax.scatter(fp_arr, hc_arr, c=color, edgecolors='black', s=60, zorder=3)
    ax.set_xlabel(r'$f_P$ [nHz]')
    ax.set_ylabel(r'$h_c$')
    ax.set_title('(c) Characteristic strain vs frequency')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig('fig_echo_pulsar_distributions.png', dpi=150, bbox_inches='tight')
fig3.savefig('fig_echo_pulsar_distributions.pdf', bbox_inches='tight')
print("Saved fig_echo_pulsar_distributions.pdf/png")


# ----- Figure 4: Both sources on the GW landscape -----
fig4, ax5 = plt.subplots(figsize=(14, 7))
ax5.set_xscale('log')
ax5.set_yscale('log')

# muAres sensitivity
ax5.plot(f_mu, h_mu, color='#984EA3', lw=2.5, label=r'$\mu$Ares')

# PTA floor (approximate Legacy IPTA)
f_pta = np.logspace(np.log10(1e-9), np.log10(1e-7), 100)
h_pta = 8e-16 * np.sqrt(1 + (1.5/(25*yr*f_pta))**4 +
                         (f_pta * yr / 3)**2)
h_pta /= np.min(h_pta) / 8e-16
ax5.plot(f_pta, h_pta, color='#A65628', lw=2, label='Legacy IPTA')

# SKA PTA
h_ska = 7e-17 * np.sqrt(1 + (1.5/(20*yr*f_pta))**4 +
                         (f_pta * yr / 3)**2)
h_ska /= np.min(h_ska) / 7e-17
ax5.plot(f_pta, h_ska, color='#F781BF', lw=2, ls='--', label='SKA PTA')

# Golden binary
ax5.scatter(golden_earth['f'], golden_earth['hc'],
            marker='*', s=350, color='#377EB8', edgecolors='black',
            linewidths=0.5, zorder=10, label='Earth term (golden)')
gf = [p['f'] for p in golden_pulsars]
gh = [p['hc'] for p in golden_pulsars]
ax5.scatter(gf, gh, marker='o', s=60, color='#377EB8', edgecolors='black',
            linewidths=0.5, zorder=10, label='Pulsar terms (golden)')
for p in golden_pulsars:
    ax5.plot([p['f'], golden_earth['f']], [p['hc'], golden_earth['hc']],
             color='#377EB8', ls=':', lw=0.5, alpha=0.3)

# Typical binary
ax5.scatter(typical_earth['f'], typical_earth['hc'],
            marker='*', s=350, color='#E66100', edgecolors='black',
            linewidths=0.5, zorder=10, label='Earth term (typical)')
tf = [p['f'] for p in typical_pulsars]
th = [p['hc'] for p in typical_pulsars]
ax5.scatter(tf, th, marker='D', s=50, color='#E66100', edgecolors='black',
            linewidths=0.5, zorder=10, label='Pulsar terms (typical)')
for p in typical_pulsars:
    ax5.plot([p['f'], typical_earth['f']], [p['hc'], typical_earth['hc']],
             color='#E66100', ls=':', lw=0.5, alpha=0.3)

ax5.set_xlim(1e-10, 1e-3)
ax5.set_ylim(1e-18, 1e-10)
ax5.set_xlabel('Frequency [Hz]', fontsize=14)
ax5.set_ylabel(r'Characteristic Strain $h_c$', fontsize=14)
ax5.set_title('Gravity Echoes on the GW Sensitivity Landscape', fontsize=14)
ax5.legend(fontsize=10, loc='upper right', ncol=2)
ax5.grid(True, which='major', alpha=0.25)
ax5.grid(True, which='minor', alpha=0.10, ls=':')

fig4.tight_layout()
fig4.savefig('fig_echo_landscape.png', dpi=150, bbox_inches='tight')
fig4.savefig('fig_echo_landscape.pdf', bbox_inches='tight')
print("Saved fig_echo_landscape.pdf/png")

plt.close('all')
print("\nDone.")
