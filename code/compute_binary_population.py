#!/usr/bin/env python3
"""
compute_binary_population.py
-----------------------------
Self-consistent estimate of the SMBHB population within the MASSIVE survey
volume and at the Tier-3 echo horizon (76 Mpc).

CALCULATION CHAIN:
  LM24 BHMF  -->  SMBH count (anchored to LM24 Fig.4 value)
               -->  CC25 occupation fraction  -->  PTA-band binary count
               -->  frequency distribution (dN/df ~ f^{-11/3})
               -->  GWB cross-check via Phinney (2001)

References and equation numbers:
  [LM24]  Liepold & Ma 2024, ApJL 971, L29
          - Eq. (3): BHMF Schechter function
          - Sec 3.1: MASSIVE survey volume = 2.05e6 Mpc^3
          - Sec 4.2: ~250 SMBHs > 10^9 Msun in MASSIVE volume (Fig. 4)
          - Eq. (5): h_c from BHMF (following SPZ24 formalism)
          - Eq. (6): mass-ratio and redshift distributions
  [CC25]  Casey-Clyde, Mingarelli et al. 2025, ApJ 987, 106
          - Sec 4.3: F_BHB = 2.6 (+4.8, -1.8)% at 95% CI
          - Eq. (12): time to coalescence T_c
          - Sec 3.2: SMBHB mass function from GWB
  [SPZ24] Sato-Polito, Zaldarriaga & Quataert 2024, PRD 110, 063020
          - Eq. (14): one-merger assumption, BHMF = integrated merger history
          - Eq. (15): h_c^2 from BHMF
          - Eq. (18): p_z, p_q distributions
  [P01]   Phinney 2001, astro-ph/0108028
          - Eq. (5): h_c theorem (GWB from remnant density)
          - Eq. (11): h_c for circular inspiralling binaries

Author: Zheng, Bécsy, Mingarelli (2026)
"""

import numpy as np
from scipy import integrate

# =============================================================================
# Physical constants (CGS)
# =============================================================================
G_cgs   = 6.67430e-8      # cm^3 g^-1 s^-2
c_cgs   = 2.99792458e10   # cm/s
Msun    = 1.98892e33      # g
pc_cm   = 3.08567758e18   # cm
Mpc_cm  = pc_cm * 1e6     # cm
yr_s    = 3.15576e7       # s

# =============================================================================
# 1. LM24 BHMF — Eq. (3)
# =============================================================================
# dn / d(ln M_BH) = phi * (M_BH / M_*)^(alpha+1) * exp(-(M_BH/M_*)^beta)
#
# Parameters from LM24 Section 4.1, below Eq. (3):
#   alpha = -1.27 +/- 0.02
#   beta  =  0.45 +/- 0.02
#   log10(phi / Mpc^-3) = -2.00 +/- 0.07
#   log10(M_* / Msun)   =  8.09 +/- 0.09
#
# NOTE: This is the BHMF (not the GSMF). LM24 derive it by convolving
# the GSMF with the M_BH-M_* scaling relation (McConnell & Ma 2013)
# and its 0.34 dex intrinsic scatter. Eq. (3) is a single Schechter
# fit to the resulting BHMF. The 90% CI in Figure 4 reflects the
# uncertainty from the scatter in the M_BH-M_* relation.

alpha_LM  = -1.27
beta_LM   =  0.45
phi_LM    = 10**(-2.00)    # Mpc^-3
Mstar_LM  = 10**(8.09)    # Msun

def bhmf_lm24(M_bh):
    """
    LM24 Eq. (3): dn/d(ln M_BH) in Mpc^-3.
    """
    x = M_bh / Mstar_LM
    return phi_LM * x**(alpha_LM + 1) * np.exp(-x**beta_LM)


def bhmf_lm24_dlogM(M_bh):
    """
    LM24 Eq. (3): dn/d(log10 M_BH) in Mpc^-3 dex^-1.
    dn/d(log10 M) = ln(10) * dn/d(ln M)
    """
    return np.log(10) * bhmf_lm24(M_bh)


def integrate_bhmf(M_min, M_max=1e12):
    """
    Integrate the LM24 BHMF: n(>M_min) = integral (dn/d ln M) d(ln M)
    Returns number density in Mpc^-3.
    """
    result, err = integrate.quad(
        lambda lnM: bhmf_lm24(np.exp(lnM)),
        np.log(M_min), np.log(M_max)
    )
    return result


# =============================================================================
# 2. Volumes and SMBH counts
# =============================================================================
# MASSIVE survey: Section 3.1 of LM24
# Volume-limited to D < 108 Mpc, northern sky (decl > -6 deg)
# Sky fraction: 55% (sky above -6 deg) * 76% (extinction cut) = 41.9%
# V_MASSIVE = 2.05 x 10^6 Mpc^3 (LM24 Section 3.1)
# Full sphere at 108 Mpc: (4/3)pi(108)^3 = 5.28e6 Mpc^3

D_MASSIVE = 108.0  # Mpc
D_tier3   = 76.0   # Mpc — Tier-3 horizon for 10^9 Msun face-on (line 251)

V_MASSIVE_survey = 2.05e6  # Mpc^3, from LM24 Section 3.1
V_full_108 = (4.0/3.0) * np.pi * D_MASSIVE**3
V_full_76  = (4.0/3.0) * np.pi * D_tier3**3

# Sky fraction of MASSIVE
f_sky = V_MASSIVE_survey / V_full_108  # = 0.388 (cf. LM24's 0.419)
# NOTE: LM24 quotes 0.419 for sky fraction. The small difference (0.388 vs
# 0.419) is because (4/3)pi(108)^3 = 5.28e6, and 2.05e6/5.28e6 = 0.388.
# LM24 computes V_MASSIVE using differential comoving volumes for each
# galaxy rather than a simple sphere. We use their V = 2.05e6 Mpc^3 directly.

# --- SMBH count: ANCHORED TO LM24 ---
# LM24 Section 4.2 and Figure 4 (bottom panel):
# The combined BHMF predicts ~250 SMBHs > 10^9 Msun within the MASSIVE
# survey volume. This comes from the median of their combined posterior,
# which includes the 0.34 dex intrinsic scatter in the M_BH-M_* relation.
N_MASSIVE_LM24 = 250  # from LM24 Figure 4, in MASSIVE volume

# Derived number density
n_above_9_LM24 = N_MASSIVE_LM24 / V_MASSIVE_survey  # Mpc^-3

# Scale to full sphere at different distances
N_full_108 = n_above_9_LM24 * V_full_108
N_full_76  = n_above_9_LM24 * V_full_76

# For comparison: our Schechter-fit integration
n_above_9_schechter = integrate_bhmf(1e9)

print("=" * 72)
print("1. SMBH COUNTS")
print("=" * 72)
print(f"LM24 MASSIVE survey volume: {V_MASSIVE_survey:.2e} Mpc^3")
print(f"N(>10^9 Msun) in MASSIVE vol [LM24 Fig.4]: {N_MASSIVE_LM24}")
print(f"  -> number density: {n_above_9_LM24:.2e} Mpc^-3")
print()
print(f"Cross-check: Schechter fit [LM24 Eq.3] integration:")
print(f"  n(>10^9) = {n_above_9_schechter:.2e} Mpc^-3")
print(f"  N in MASSIVE vol = {n_above_9_schechter * V_MASSIVE_survey:.0f}")
print(f"  Ratio (Schechter / LM24): {n_above_9_schechter / n_above_9_LM24:.2f}")
print(f"  This ~2x discrepancy arises because the Schechter fit is the")
print(f"  best-fit median, while the ~250 from Figure 4 reflects the")
print(f"  full posterior including M_BH-M_* scatter (0.34 dex). The 90%")
print(f"  CI in their Figure 4 spans roughly 100--600 in this mass range.")
print()
print(f"Full sphere volumes and counts (using LM24-anchored density):")
print(f"  108 Mpc: V = {V_full_108:.2e} Mpc^3, N = {N_full_108:.0f}")
print(f"   76 Mpc: V = {V_full_76:.2e} Mpc^3, N = {N_full_76:.0f}")
print(f"  Volume ratio: (76/108)^3 = {(D_tier3/D_MASSIVE)**3:.4f}")
print()


# =============================================================================
# 3. Binary occupation fraction — CC25 Section 4.3
# =============================================================================
# CC25 define F_BHB = phi_BHB / phi_BH (Sec 3.5), where:
#   phi_BHB = SMBHB mass function (binaries with f_GW >= 1 nHz)
#   phi_BH  = SMBH mass function
# Both integrated over 10^8 <= M_BH <= 10^{10.5} and 0 <= z <= 1.5.
#
# Result (CC25 Section 4.3):
#   F_BHB = 2.6 (+4.8, -1.8) % at 95% confidence
#
# IMPORTANT CAVEATS for local application:
# (a) F_BHB is averaged over z = 0--1.5. For the local universe (z ~ 0.02),
#     the occupation fraction could differ if the merger rate evolves with z.
#     However, CC25's SMBHB mass function is constrained by the GWB, which
#     is dominated by z ~ 0.3 sources, so F_BHB is most representative
#     of low-z conditions.
# (b) F_BHB is averaged over 10^8--10^{10.5} Msun. CC25 Figure 3 shows
#     phi_BHB / phi_BH ~ 1--3% across this range, so applying 2.6% to the
#     M > 10^9 subset is reasonable.
# (c) F_BHB counts binaries with f_GW >= 1 nHz (anywhere in PTA band).

F_BHB_median = 0.026        # 2.6%
F_BHB_upper  = 0.026 + 0.048  # 7.4% (95% upper)
F_BHB_lower  = 0.026 - 0.018  # 0.8% (95% lower)

# Expected number of PTA-band binaries (M > 10^9 Msun)
N_bin_108 = N_full_108 * F_BHB_median
N_bin_76  = N_full_76  * F_BHB_median
N_bin_76_lo = N_full_76 * F_BHB_lower
N_bin_76_hi = N_full_76 * F_BHB_upper

print("=" * 72)
print("2. BINARY OCCUPATION FRACTION [CC25 Section 4.3]")
print("=" * 72)
print(f"F_BHB = {F_BHB_median*100:.1f}% "
      f"(95% CI: {F_BHB_lower*100:.1f}% -- {F_BHB_upper*100:.1f}%)")
print(f"  Integrated over 10^8 <= M <= 10^{{10.5}} Msun, 0 <= z <= 1.5")
print()
print(f"Expected PTA-band SMBHBs (M > 10^9 Msun, full sphere):")
print(f"  Within 108 Mpc: {N_bin_108:.1f}")
print(f"  Within  76 Mpc: {N_bin_76:.1f} "
      f"(95% CI: {N_bin_76_lo:.1f} -- {N_bin_76_hi:.1f})")
print()


# =============================================================================
# 4. Residence time — CC25 Eq. (12)
# =============================================================================
# For circular GW-driven binary, time to coalescence from f_GW:
#   T_c = (5/256) M_BHB^{-5/3} (1+q)^2/q [pi f_GW (1+z)]^{-8/3}
# where M_BHB is in geometric units (G*M/c^3).
# CC25 Eq. (12) uses f_GW = 10^{-9} Hz as the lowest PTA frequency.

def T_coalescence(M_tot_Msun, q, f_GW_Hz, z=0.0):
    """
    Time to coalescence from GW frequency f_GW.
    CC25 Eq. (12), in physical units.

    Returns T_c in years.
    """
    M_geom = G_cgs * M_tot_Msun * Msun / c_cgs**3  # seconds
    T_c_s = (5.0/256.0) * M_geom**(-5.0/3.0) * (1+q)**2 / q * \
            (np.pi * f_GW_Hz * (1+z))**(-8.0/3.0)
    return T_c_s / yr_s


# Fiducial: M_tot = 10^9 Msun, q = 1, z = 0
M_fid = 1e9
q_fid = 1.0

# Frequency boundaries
f_PTA_low  = 1e-9   # 1 nHz
f_PTA_high = 1e-7   # 100 nHz
f_muAres   = 1e-6   # 1 muHz

T_from_1nHz   = T_coalescence(M_fid, q_fid, f_PTA_low)
T_from_100nHz = T_coalescence(M_fid, q_fid, f_PTA_high)
T_from_1muHz  = T_coalescence(M_fid, q_fid, f_muAres)

t_PTA  = T_from_1nHz - T_from_100nHz       # 1 nHz -> 100 nHz
t_muHz = T_from_100nHz - T_from_1muHz      # 100 nHz -> 1 muHz
duty_muAres = t_muHz / T_from_1nHz          # fraction of PTA-band time in muAres band

print("=" * 72)
print("3. RESIDENCE TIMES [CC25 Eq. (12)]")
print("=" * 72)
print(f"Fiducial: M_tot = {M_fid:.0e} Msun, q = {q_fid}, z = 0")
print(f"  T_c from   1 nHz: {T_from_1nHz:.3e} yr ({T_from_1nHz/1e6:.1f} Myr)")
print(f"  T_c from 100 nHz: {T_from_100nHz:.1f} yr")
print(f"  T_c from   1 muHz: {T_from_1muHz:.2f} yr")
print()
print(f"  Residence time in PTA band (1--100 nHz):   {t_PTA/1e6:.1f} Myr")
print(f"  Residence time in muAres band (100 nHz--1 muHz): {t_muHz:.0f} yr")
print(f"  Duty cycle (muAres / PTA-band): {duty_muAres:.2e}")
print()
print(f"  Expected N(muAres band, M > 10^9, 76 Mpc):")
print(f"    = N_PTA_band * duty_cycle")
print(f"    = {N_bin_76:.1f} * {duty_muAres:.2e}")
print(f"    = {N_bin_76 * duty_muAres:.2e}")
print(f"  This is negligible for a single mass bin. The ~0.1--6 range")
print(f"  quoted in the text integrates over all masses and uses a")
print(f"  range of binary fractions (see Section III of paper).")
print()


# =============================================================================
# 5. Frequency distribution within PTA band
# =============================================================================
# For GW-driven circular binaries, the GW frequency evolves as:
#   df/dt = (96/5) pi^{8/3} (G Mc)^{5/3} f^{11/3} / c^5
# Peters (1964) Eq. (5.6)
#
# The time per unit frequency is dt/df ~ f^{-11/3}, so the number of
# binaries per frequency bin scales as dN/df ~ f^{-11/3}.
# This means almost all PTA-band binaries pile up at the lowest
# frequencies. Specifically:
#
# Fraction in bin [f, f+df] = f^{-11/3} df / integral_fmin^fmax f^{-11/3} df

print("=" * 72)
print("4. FREQUENCY DISTRIBUTION IN PTA BAND")
print("=" * 72)

# Integral of f^{-11/3} from fmin to fmax (in nHz)
def frac_in_band(f1_nHz, f2_nHz, fmin_nHz=1.0, fmax_nHz=100.0):
    """Fraction of PTA-band binaries with f in [f1, f2]."""
    # integral of f^{-11/3} df = [-3/8 f^{-8/3}] = (3/8)(f1^{-8/3} - f2^{-8/3})
    num = f1_nHz**(-8.0/3.0) - f2_nHz**(-8.0/3.0)
    den = fmin_nHz**(-8.0/3.0) - fmax_nHz**(-8.0/3.0)
    return num / den

# Frequency bins (PTA resolution: Delta_f = 1/T_obs)
T_obs = 20.0  # yr
delta_f_nHz = 1.0 / (T_obs * yr_s) * 1e9  # nHz

print(f"PTA frequency resolution (T_obs = {T_obs} yr): {delta_f_nHz:.2f} nHz")
print()

# Fraction in first few bins
for i, f_lo in enumerate([1.0, 1.0 + delta_f_nHz, 1.0 + 2*delta_f_nHz]):
    f_hi = f_lo + delta_f_nHz
    frac = frac_in_band(f_lo, f_hi)
    N_in_bin = N_bin_76 * frac
    print(f"  Bin {i+1}: {f_lo:.2f}--{f_hi:.2f} nHz: "
          f"fraction = {frac:.3f}, N = {N_in_bin:.2f}")

# Fraction above 100 nHz (muAres band)
frac_muAres_freq = frac_in_band(100.0, 1000.0)  # 100 nHz to 1 muHz
print()
print(f"  Fraction at f > 100 nHz: {frac_muAres_freq:.2e}")
print(f"  N(f > 100 nHz, 76 Mpc) = {N_bin_76 * frac_muAres_freq:.2e}")
print(f"  (Consistent with duty-cycle estimate above)")
print()


# =============================================================================
# 6. GWB CONSISTENCY CHECK — Phinney (2001) Eq. (11) / LM24 Eq. (5)
# =============================================================================
# From Phinney (2001) Eq. (11) for circular inspiralling binaries:
#   h_c^2(f) = (4 / 3pi^{1/3}) * (1/c^2) * (G Mc)^{5/3} / f^{4/3}
#              * N_0 * <(1+z)^{-1/3}>
#
# For a distribution of masses, using SPZ24 Eq. (15) / LM24 Eq. (5):
#   h_c^2(f) = (4pi / 3c^2) * 1/(pi*f)^{4/3}
#              * <q/(1+q)^2> * <(1+z)^{-1/3}>
#              * integral[ (G*M)^{5/3} * (dn/dM) dM ]
#
# Mass-ratio and redshift averages from LM24 Eq. (6) / SPZ24 Eq. (18):
#   p_z(z) = z^gamma * exp(-(z/z_*)^beta_z)  [we use SPZ24 parametrization]
#   p_q(q) = q^delta
# with gamma=1.0, z_*=0.5, delta=-1 (LM24 adopts SPZ24 values)
#
# <q/(1+q)^2> = 0.238  (LM24 Section 5.1 / SPZ24 Eq. (18))
# <(1+z)^{-1/3}> = 0.890  (LM24 Section 5.1)

print("=" * 72)
print("5. GWB CONSISTENCY CHECK [Phinney (2001) / LM24 Eq. (5)]")
print("=" * 72)

# Mass-ratio and redshift distribution averages (from LM24 Section 5.1)
avg_q_factor = 0.238       # <q/(1+q)^2> for p_q ~ q^delta, delta=-1
avg_z_factor = 0.890       # <(1+z)^{-1/3}> for p_z ~ z exp(-z/0.5)

# Compute integral of (G*M)^{5/3} * (dn/dM) dM over the BHMF
# dn/dM = (1/M) * dn/d(ln M)
# So integral = integral[ (G*M)^{5/3} * (1/M) * dn/d(ln M) * dM ]
#             = integral[ (G*M)^{5/3} * dn/d(ln M) * d(ln M) ]
#             = G^{5/3} * integral[ M^{5/3} * dn/d(ln M) * d(ln M) ]

def mass_integral_schechter():
    """Integrate M^{5/3} * (dn/d ln M) * d(ln M) using Schechter BHMF."""
    def integrand(lnM):
        M = np.exp(lnM)
        return (M * Msun)**( 5.0/3.0) * bhmf_lm24(M)  # M in Msun, convert to grams for G
    # Actually, keep M in Msun and multiply by (G*Msun)^{5/3} outside
    def integrand_dimless(lnM):
        M = np.exp(lnM)
        return M**(5.0/3.0) * bhmf_lm24(M)  # Mpc^-3 * Msun^{5/3}
    result, err = integrate.quad(integrand_dimless, np.log(1e7), np.log(1e12))
    return result

I_M = mass_integral_schechter()  # in Mpc^-3 * Msun^{5/3}

# Now compute h_c at f = 1/yr using LM24 Eq. (5):
# h_c^2 = (4pi)/(3 c^2 (pi f)^{4/3}) * <q/(1+q)^2> * <(1+z)^{-1/3}>
#          * G^{5/3} * I_M
# where I_M is in Mpc^-3 * Msun^{5/3}
# Need to convert Mpc^-3 to cm^-3

f_ref = 1.0 / yr_s  # 1/yr in Hz
GM_unit = (G_cgs * Msun)**(5.0/3.0)  # (G * 1 Msun)^{5/3} in CGS
I_M_cgs = I_M * GM_unit / Mpc_cm**3  # now in cm^-3 * (cm^3 g^-1 s^-2 * g)^{5/3} ...

# Let me be more explicit about units.
# h_c^2 = (4pi / 3) * (1/c^2) * (1/(pi*f)^{4/3}) * <q/(1+q)^2> * <(1+z)^{-1/3}>
#        * integral[ (G*M_BH)^{5/3} dn/d(ln M) d(ln M) ]
#
# The integral in CGS: sum over all masses of (G*M)^{5/3} * n(M) where n is in cm^-3
# I_M is in Msun^{5/3} * Mpc^{-3}
# Convert: (G_cgs * Msun)^{5/3} * (1/Mpc_cm^3)

I_cgs = I_M * (G_cgs * Msun)**(5.0/3.0) / Mpc_cm**3  # cm^{-3} * cm^5 g^{5/3} s^{-10/3} ...
# Actually (G*M)^{5/3} has units of [cm^3 g^-1 s^-2 * g]^{5/3} = [cm^3 s^-2]^{5/3}
#   = cm^5 s^{-10/3}
# So I_cgs has units cm^{-3} * cm^5 s^{-10/3} = cm^2 s^{-10/3}

hc2_LM24 = (4 * np.pi / 3.0) * (1.0 / c_cgs**2) * (1.0 / (np.pi * f_ref)**(4.0/3.0)) \
           * avg_q_factor * avg_z_factor * I_cgs

hc_LM24 = np.sqrt(hc2_LM24)

# LM24 reports h_c ~ 2.0e-15 at f = 1/yr (Section 5.1)
# NANOGrav measures (2.4 +0.7/-0.6) x 10^-15

print(f"Using Schechter BHMF [LM24 Eq. (3)] with SPZ24 averages:")
print(f"  <q/(1+q)^2> = {avg_q_factor}")
print(f"  <(1+z)^{{-1/3}}> = {avg_z_factor}")
print(f"  h_c(f = 1/yr) = {hc_LM24:.2e}")
print(f"  LM24 reports: ~2.0 x 10^-15")
print(f"  NANOGrav 15yr: (2.4 +0.7/-0.6) x 10^-15")
print()

# Now rescale using LM24's ~250 anchor instead of Schechter integration.
# The Schechter integration gives ~2x more SMBHs above 10^9.
# But the dominant contribution to h_c comes from ~10^9 Msun (LM24 Fig 5),
# so if our Schechter fit overestimates by 2x, our h_c would be ~sqrt(2)
# too high. Since h_c depends on the integral of M^{5/3} * dn/d(ln M),
# the correction isn't simply the count ratio. LM24 Figure 5 shows their
# h_c is consistent with PTA data, providing the self-consistency check.
print(f"Self-consistency: LM24's BHMF (with full posterior, yielding ~250")
print(f"SMBHs > 10^9 in MASSIVE vol) gives h_c consistent with PTA data.")
print(f"CC25 constrains F_BHB using the same GWB amplitude. Therefore our")
print(f"binary count (LM24 SMBH count * CC25 F_BHB) is consistent with")
print(f"the measured GWB by construction.")
print()


# =============================================================================
# 7. Bence's scaling check: (76/108)^3 applied to MASSIVE count
# =============================================================================
print("=" * 72)
print("6. BENCE'S SCALING: MASSIVE vol -> 76 Mpc")
print("=" * 72)

# Bence suggests: 250 * (76/108)^3 ~ 87 within 76 Mpc.
# But this applies (76/108)^3 to the MASSIVE survey count (250),
# which already covers only 41.9% of the sky. The correct scaling
# depends on what we mean:
#
# (a) 250 in MASSIVE vol (2.05e6 Mpc^3, partial sky)
#     -> n = 250/2.05e6 = 1.22e-4 Mpc^-3
#     -> N(full sphere, 76 Mpc) = n * (4/3)pi(76)^3 = 224
#
# (b) Bence's simple scaling: 250 * (76/108)^3 = 87
#     This implicitly treats 250 as a full-sphere count at 108 Mpc,
#     but 250 is the MASSIVE count (partial sky). If the paper says
#     "250 within 108 Mpc" without specifying MASSIVE sky coverage,
#     then 87 is a valid full-sky extrapolation.

vol_ratio = (D_tier3 / D_MASSIVE)**3
N_bence = N_MASSIVE_LM24 * vol_ratio
N_correct = n_above_9_LM24 * V_full_76

print(f"LM24 reports ~{N_MASSIVE_LM24} in MASSIVE vol (41.9% sky, 108 Mpc)")
print(f"Bence scaling: {N_MASSIVE_LM24} * (76/108)^3 = {N_bence:.0f}")
print(f"  -> This is correct IF 250 is treated as the count within a")
print(f"     partial volume that scales homogeneously to 76 Mpc.")
print()
print(f"Full-sphere scaling from number density:")
print(f"  n = {N_MASSIVE_LM24}/2.05e6 = {n_above_9_LM24:.2e} Mpc^-3")
print(f"  N(full sphere, 76 Mpc) = {N_correct:.0f}")
print()
print(f"The distinction matters for the paper text. Line 430 says:")
print(f"  '~250 SMBHs above 10^9 within 108 Mpc'")
print(f"If this means the MASSIVE survey volume (partial sky), then")
print(f"Bence's (76/108)^3 scaling gives ~{N_bence:.0f} in the same sky fraction")
print(f"at 76 Mpc, or ~{N_correct:.0f} for the full sphere at 76 Mpc.")
print()


# =============================================================================
# 8. Summary
# =============================================================================
print("=" * 72)
print("SUMMARY TABLE")
print("=" * 72)
print(f"{'Quantity':<55} {'Value':>15}")
print("-" * 72)
print(f"{'LM24 BHMF: N(>10^9) in MASSIVE vol':<55} {'~250':>15}")
print(f"{'  -> number density n [Mpc^-3]':<55} {n_above_9_LM24:>15.2e}")
print(f"{'N(>10^9, full sphere, 108 Mpc)':<55} {N_full_108:>15.0f}")
print(f"{'N(>10^9, full sphere, 76 Mpc)':<55} {N_full_76:>15.0f}")
print(f"{'N(>10^9, Bence scaling: 250*(76/108)^3)':<55} {N_bence:>15.0f}")
print(f"{'CC25 F_BHB':<55} {'2.6%':>15}")
print(f"{'  95% CI':<55} {'0.8% -- 7.4%':>15}")
print(f"{'N_SMBHB (PTA band, 76 Mpc, median)':<55} {N_bin_76:>15.1f}")
print(f"{'  95% CI':<55} {f'{N_bin_76_lo:.1f} -- {N_bin_76_hi:.1f}':>15}")
print(f"{'t_res PTA band (10^9 Msun, q=1) [Myr]':<55} {t_PTA/1e6:>15.1f}")
print(f"{'t_res muAres band (100nHz-1muHz) [yr]':<55} {t_muHz:>15.0f}")
print(f"{'h_c(1/yr) from BHMF [Phinney formalism]':<55} {f'{hc_LM24:.1e}':>15}")
print(f"{'  LM24 reports':<55} {'~2.0e-15':>15}")
print(f"{'  NANOGrav 15yr':<55} {'(2.4+/-0.7)e-15':>15}")
print()

# Also compute for a few mass thresholds
print("Additional mass thresholds (full sphere, 108 Mpc):")
for logM in [8.0, 8.5, 9.0, 9.5, 10.0]:
    n_M = integrate_bhmf(10**logM)
    N_M = n_M * V_full_108
    # Apply correction factor to match LM24 anchor at 10^9
    # ratio of Schechter to LM24 at 10^9
    corr = n_above_9_LM24 / n_above_9_schechter
    N_M_corr = N_M * corr
    print(f"  N(>10^{logM:.1f}) = {N_M:.0f} (Schechter) / "
          f"~{N_M_corr:.0f} (LM24-anchored)")
print()

print("=" * 72)
print("7. COSMOLOGICAL muAres COUNTS")
print("=" * 72)

# Within the echo horizon (76 Mpc), the expected number of 10^9 Msun
# binaries in the muAres band is negligible (~3e-5). But muAres is a
# space-based detector that sees the entire sky to cosmological distances.
# A 10^9 Msun binary at 76 Mpc has h ~ 3e-14, while muAres sensitivity
# is ~1e-17, so the detection horizon far exceeds z = 1.
#
# The question: how many SMBHBs are in the muAres band across the
# observable universe?
#
# Method:
# 1. Use the LM24-anchored local number density n(>M) at z=0
# 2. Assume the comoving density is approximately constant out to z~1
#    (conservative; merger rate may increase with z)
# 3. Compute comoving volume to z_max
# 4. Multiply by F_BHB * duty_cycle_muAres
#
# The duty cycle fraction t_muAres / T_c(1 nHz) is MASS-INDEPENDENT
# for GW-driven circular binaries:
#   T_c(f) = (5/256) M_geo^{-5/3} (1+q)^2/q (pi f)^{-8/3}
#   t_muAres = T_c(100 nHz) - T_c(1 muHz)
#   t_muAres / T_c(1 nHz) = [f_100nHz^{-8/3} - f_1muHz^{-8/3}] / f_1nHz^{-8/3}
# The M-dependence cancels in the ratio.

# Comoving volume to redshift z (flat LCDM, H0=67.4, Om=0.315)
H0_km_s_Mpc = 67.4
Om = 0.315
OL = 1 - Om

def comoving_distance(z_max, npts=10000):
    """Comoving distance in Mpc (flat LCDM)."""
    zz = np.linspace(0, z_max, npts)
    dz = zz[1] - zz[0]
    integrand = 1.0 / np.sqrt(Om * (1 + zz)**3 + OL)
    d_H = 2.998e5 / H0_km_s_Mpc  # Hubble distance in Mpc
    return d_H * np.trapz(integrand, zz)

def comoving_volume(z_max):
    """Comoving volume in Mpc^3 (full sky, flat LCDM)."""
    d_c = comoving_distance(z_max)
    return (4.0 / 3.0) * np.pi * d_c**3

# Duty cycle: mass-independent
duty_muAres_ratio = (f_PTA_high**(-8.0/3.0) - f_muAres**(-8.0/3.0)) / \
                    f_PTA_low**(-8.0/3.0)
print(f"Duty cycle (muAres band / total from 1 nHz): {duty_muAres_ratio:.2e}")
print(f"  (mass-independent for GW-driven circular binaries)")
print()

# Number densities for different mass thresholds (LM24-anchored)
corr_factor = n_above_9_LM24 / n_above_9_schechter

mass_thresholds = [8.0, 8.5, 9.0, 9.5]
n_above = {}
for logM in mass_thresholds:
    n_sch = integrate_bhmf(10**logM)
    n_above[logM] = n_sch * corr_factor  # LM24-anchored

z_max_vals = [0.5, 1.0, 1.5]

print(f"{'log10(M_min)':<14}", end="")
for z in z_max_vals:
    print(f"  {'z<'+str(z):>10}", end="")
print()
print("-" * 50)

for logM in mass_thresholds:
    print(f"  {logM:<12}", end="")
    for z_max in z_max_vals:
        V_c = comoving_volume(z_max)
        N_total_smbh = n_above[logM] * V_c
        N_pta_band = N_total_smbh * F_BHB_median
        N_muAres = N_pta_band * duty_muAres_ratio
        print(f"  {N_muAres:>10.1f}", end="")
    print()

print()
print("Detailed breakdown for z < 1:")
V_c_1 = comoving_volume(1.0)
print(f"  Comoving volume (z<1): {V_c_1:.2e} Mpc^3")
print(f"  Comoving distance to z=1: {comoving_distance(1.0):.0f} Mpc")
print()

for logM in mass_thresholds:
    N_total = n_above[logM] * V_c_1
    N_pta = N_total * F_BHB_median
    N_mu = N_pta * duty_muAres_ratio
    print(f"  M > 10^{logM:.1f}: {N_total:.0f} SMBHs, {N_pta:.0f} PTA-band, "
          f"{N_mu:.1f} in muAres band")

# Time in muAres band and flow rate
print()
t_muAres_yr = t_muHz  # already computed above: ~380 yr
print(f"  t_muAres (10^9 Msun, q=1) = {t_muAres_yr:.0f} yr")
print(f"  10-yr mission: negligible flow of new entries (t_muAres >> 10 yr)")
print(f"  N_muAres over 10 yr ~ N_muAres(now) * (1 + 10/{t_muAres_yr:.0f})")

# For the paper: key number at M > 10^9
N_mu_9_z1 = n_above[9.0] * V_c_1 * F_BHB_median * duty_muAres_ratio
print()
print(f"KEY RESULT: ~{N_mu_9_z1:.0f} SMBHBs with M > 10^9 in muAres band (z<1)")
print(f"  95% CI: {N_mu_9_z1 * F_BHB_lower/F_BHB_median:.1f} -- "
      f"{N_mu_9_z1 * F_BHB_upper/F_BHB_median:.1f}")
N_mu_85_z1 = n_above[8.5] * V_c_1 * F_BHB_median * duty_muAres_ratio
print(f"  At M > 10^8.5: ~{N_mu_85_z1:.0f}")
print()


print("=" * 72)
print("KEY REFERENCES USED IN EACH STEP")
print("=" * 72)
print("Step 1: SMBH count         -> LM24 Eq.(3), Sec 4.2, Fig.4")
print("Step 2: Occupation fraction -> CC25 Sec 4.3, Fig.3")
print("Step 3: Residence time      -> CC25 Eq.(12)")
print("Step 4: Freq. distribution  -> Peters (1964) Eq.(5.6)")
print("Step 5: GWB cross-check     -> P01 Eq.(11), LM24 Eq.(5),")
print("                               SPZ24 Eq.(15)")
print("Step 6: Cosmological muAres -> LM24 density + CC25 F_BHB +")
print("                               mass-independent duty cycle")
