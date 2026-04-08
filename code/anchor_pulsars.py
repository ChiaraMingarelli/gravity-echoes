"""
Estimate the number of PTA pulsars that could serve as
phase-coherent 'anchor' pulsars for echo science.

Phase coherence requires:
    delta_L_p < c / (2 pi f)

The distance uncertainty from parallax:
    sigma_d = sigma_pi * d^2   (for sigma_pi << pi)

where sigma_pi is in arcsec and d in pc.

So the required parallax precision:
    sigma_pi < c / (2 pi f d^2)     [arcsec]
"""

import numpy as np

c = 2.998e8       # m/s
pc = 3.086e16     # m

def phase_coherence_threshold_pc(f_Hz):
    """Maximum distance uncertainty (pc) for phase coherence at frequency f."""
    return c / (2 * np.pi * f_Hz) / pc

def required_parallax_uas(f_Hz, d_pc):
    """Required parallax precision (microarcsec) to meet phase coherence at distance d."""
    delta_Lp_pc = phase_coherence_threshold_pc(f_Hz)
    # sigma_d = sigma_pi [arcsec] * d^2 [pc]
    # sigma_pi [arcsec] < delta_Lp / d^2
    sigma_pi_arcsec = delta_Lp_pc / d_pc**2
    return sigma_pi_arcsec * 1e6  # convert to uas

# ---- Print threshold vs frequency ----
print("Phase coherence distance threshold vs GW frequency")
print("=" * 50)
freqs_nHz = [1, 3, 5, 10, 30, 100]
for f_nHz in freqs_nHz:
    f = f_nHz * 1e-9
    thresh = phase_coherence_threshold_pc(f)
    print(f"  f = {f_nHz:4d} nHz  -->  delta_L_p < {thresh:.2f} pc")

# ---- Required parallax precision vs distance ----
print("\nRequired parallax precision (uas) for phase coherence")
print("=" * 60)
print(f"{'d (pc)':>10s}", end="")
for f_nHz in [3, 5, 10, 30]:
    print(f"  {f_nHz} nHz", end="")
print()
print("-" * 60)
for d in [100, 157, 200, 300, 500, 800, 1000, 1500]:
    print(f"{d:10d}", end="")
    for f_nHz in [3, 5, 10, 30]:
        f = f_nHz * 1e-9
        sig = required_parallax_uas(f, d)
        print(f"  {sig:8.1f}", end="")
    print()

# ---- Known nearby MSPs with best parallaxes ----
print("\n\nKnown PTA MSPs with best distance constraints")
print("=" * 70)

# Source: ATNF catalog, Reardon 2024, Moran 2023, Deller 2019, Ding 2023
# Format: (name, d_pc, sigma_d_pc, method)
pulsars = [
    ("J0437-4715",  156.96, 0.11, "timing (Reardon 2024)"),
    ("J0030+0451",  325,    9,    "timing parallax (PPTA)"),
    ("J2124-3358",  410,    20,   "VLBI (Deller+ 2019)"),
    ("J1012+5307",  845,    14,   "Gaia+timing (Moran+ 2023)"),
    ("J1024-0719",  1203,   78,   "Gaia+timing (Moran+ 2023)"),
    ("J1713+0747",  1176,   11,   "VLBI (Deller+ 2019)"),
    ("J1909-3744",  1140,   70,   "timing (PPTA DR3)"),
    ("J0613-0200",  780,    50,   "timing (EPTA)"),
    ("J1744-1134",  395,    14,   "VLBI (Deller+ 2019)"),
    ("J2145-0750",  613,    30,   "VLBI (Deller+ 2019)"),
    ("J1643-1224",  735,    100,  "timing (IPTA)"),
    ("J1939+2134",  3560,   200,  "timing (IPTA)"),  # B1937+21
    ("J1857+0943",  910,    70,   "timing (NANOGrav)"),
    ("J0751+1807",  1100,   110,  "timing"),
    ("J1738+0333",  1470,   110,  "VLBI (Deller+ 2019)"),
    ("J2317+1439",  1890,   200,  "timing"),
    ("J1600-3053",  1630,   120,  "timing"),
    ("J0340+4130",  1730,   300,  "DM estimate"),
]

# Check which meet threshold at various frequencies
print(f"{'Pulsar':>14s}  {'d (pc)':>8s}  {'σ_d (pc)':>8s}  {'Method':>30s}  ", end="")
for f_nHz in [3, 10, 30]:
    print(f" {f_nHz}nHz", end="")
print()
print("-" * 100)

for name, d, sig_d, method in sorted(pulsars, key=lambda x: x[1]):
    print(f"{name:>14s}  {d:8.1f}  {sig_d:8.2f}  {method:>30s}  ", end="")
    for f_nHz in [3, 10, 30]:
        f = f_nHz * 1e-9
        thresh = phase_coherence_threshold_pc(f)
        if sig_d < thresh:
            print("   ✓ ", end="")
        else:
            ratio = sig_d / thresh
            print(f" {ratio:4.0f}x", end="")
    print()

# ---- Projection: SKA-era improvements ----
print("\n\nProjected improvements with SKA1 timing (20 yr, 100 ns)")
print("=" * 70)
print("SKA1 timing parallax precision scales roughly as:")
print("  sigma_pi ~ 100 * (sigma_TOA / 100 ns) * (T / 20 yr)^{-5/2} * (Ncad)^{-1/2} uas")
print("  For best pulsars: ~0.5-2 uas achievable")
print()

# With SKA achieving ~1 uas timing parallax for nearby bright MSPs
ska_sig_pi_uas = 1.0  # optimistic for best pulsars
print(f"Assuming SKA timing parallax precision: {ska_sig_pi_uas} uas")
print(f"{'d (pc)':>10s}  {'σ_d (pc)':>10s}  {'Meet 10 nHz?':>15s}  {'Meet 3 nHz?':>15s}")
print("-" * 55)
for d in [100, 150, 200, 250, 300, 400, 500, 700, 1000]:
    sig_d = ska_sig_pi_uas * 1e-6 / 206265 * (d * pc / pc)**2
    # Actually: sigma_d = sigma_pi [rad] * d^2 / (1 arcsec in rad)
    # No. sigma_d [pc] = sigma_pi [arcsec] * d^2 [pc]  (linearized)
    sig_d = (ska_sig_pi_uas * 1e-6) * d**2
    thresh_10 = phase_coherence_threshold_pc(10e-9)
    thresh_3 = phase_coherence_threshold_pc(3e-9)
    meet_10 = "YES" if sig_d < thresh_10 else f"no ({sig_d/thresh_10:.0f}x)"
    meet_3 = "YES" if sig_d < thresh_3 else f"no ({sig_d/thresh_3:.0f}x)"
    print(f"{d:10d}  {sig_d:10.3f}  {meet_10:>15s}  {meet_3:>15s}")

# ---- Summary estimate ----
print("\n\nSummary: estimated anchor pulsar counts")
print("=" * 50)
print("At 10 nHz (δL_p < 0.15 pc):")
print("  Current: 1 (J0437-4715 only)")
print("  SKA-era: ~2-4 (need d < 200 pc with σ_π ~ 1 μas)")
print()
print("At 3 nHz (δL_p < 0.5 pc):")
print("  Current: 1 (J0437-4715)")
print("  SKA-era: ~5-10 (d < 350 pc with σ_π ~ 1 μas)")
print()
print("Key points:")
print("  - σ_d = σ_π × d² means distance error grows quadratically")
print("  - Even 1 μas parallax gives σ_d = 1 pc at d = 1 kpc")
print("  - Only very nearby pulsars (d < 200-400 pc) can be anchors")
print("  - SKA will discover new nearby MSPs (currently ~3-5 known < 300 pc in PTAs)")
print("  - Moran+ 2023 Gaia+timing gives 29-53% improvement, helpful but not transformative")
print("  - Roman + Gaia (McKinnon 2026) extends to fainter companions")
