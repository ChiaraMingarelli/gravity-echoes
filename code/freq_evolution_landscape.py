"""
freq_evolution_landscape.py

Compute and visualize the GW frequency evolution over Earth-pulsar
baselines for non-spinning SMBHBs.

Two-panel figure:
  Top: Pulsar-term frequency f_P vs total mass M, for LISA and muAres
       Earth-term frequencies. Shows where f_P falls relative to the
       PTA band. Both detector classes produce f_P in the nHz regime
       for sufficiently high masses.
  Bottom: Pulsar-term timing residual amplitude vs M. This is why
       only muAres sources are detectable: residuals scale as M^{5/3}.

No spins (chi1 = chi2 = 0), equal mass.

Author: Chiara Mingarelli / generated code
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smbhb_evolution import SMBHBEvolution, PC, C_SI, YR, M_SUN, G_SI

# ======================================================================
# Physical functions
# ======================================================================

def isco_frequency(M_msun):
    """GW frequency at ISCO (Schwarzschild) [Hz]."""
    M_kg = M_msun * M_SUN
    return C_SI**3 / (6**1.5 * np.pi * G_SI * M_kg)


def light_travel_time_yr(L_kpc, geo_factor=1.0):
    """Look-back time [yr] for a pulsar at L_kpc with geometric factor."""
    L_m = L_kpc * 1e3 * PC
    return L_m * geo_factor / C_SI / YR


def newtonian_fP(M_msun, f_E, tau_yr):
    """
    Leading-order (Newtonian) pulsar-term frequency.

    f_P = f_E * (1 + 256/5 * pi^(8/3) * Mc_s^(5/3) * f_E^(8/3) * tau)^{-3/8}

    Equal mass assumed (eta = 0.25).
    """
    eta = 0.25
    M_kg = np.asarray(M_msun, dtype=float) * M_SUN
    Mc_kg = M_kg * eta**0.6
    Mc_s = G_SI * Mc_kg / C_SI**3
    tau_s = tau_yr * YR

    x = 256.0 / 5 * np.pi**(8.0/3) * Mc_s**(5.0/3) * f_E**(8.0/3) * tau_s
    f_P = f_E * (1 + x)**(-3.0/8)
    return f_P


def pulsar_term_residual_ns(M_msun, f_P, D_L_Mpc=100.0):
    """
    Order-of-magnitude pulsar-term timing residual [ns].

    r_P ~ h / (2 pi f_P)  where  h ~ 2 (pi f_P)^{2/3} Mc^{5/3} / (c^4 D_L)

    This gives r_P ~ Mc^{5/3} f_P^{-1/3} / (pi^{1/3} c^4 D_L).
    """
    eta = 0.25
    M_kg = np.asarray(M_msun, dtype=float) * M_SUN
    Mc_kg = M_kg * eta**0.6
    D_L = D_L_Mpc * 1e6 * PC

    # Strain amplitude at f_P
    h = 2 * (np.pi * f_P)**(2.0/3) * (G_SI * Mc_kg)**(5.0/3) / (C_SI**4 * D_L)

    # Residual ~ h / (2 pi f_P)
    r = h / (2 * np.pi * f_P)
    return r * 1e9  # convert s -> ns


# ======================================================================
# Setup
# ======================================================================

tau_fid = light_travel_time_yr(1.0, 1.0)  # ~3262 yr
print(f"Fiducial baseline: tau = {tau_fid:.0f} yr (1 kpc, geo=1)")

M_arr = np.logspace(5, 10.5, 500)

# Earth-term frequencies: three in muAres band, three in LISA band
freq_configs = [
    # (f_E [Hz], label, color, linestyle, detector)
    (1e-6,   r"$f_E = 1\;\mu$Hz",    "#1b4f72", "-",  "muAres"),
    (0.1e-6, r"$f_E = 0.1\;\mu$Hz",  "#2874a6", "-",  "muAres"),
    (10e-6,  r"$f_E = 10\;\mu$Hz",   "#5dade2", "-",  "muAres"),
    (1e-3,   r"$f_E = 1$ mHz",       "#cb4335", "--", "LISA"),
    (0.1e-3, r"$f_E = 0.1$ mHz",     "#ec7063", "--", "LISA"),
    (10e-3,  r"$f_E = 10$ mHz",      "#f1948a", "--", "LISA"),
]

# ======================================================================
# Figure: two panels
# ======================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
plt.rcParams.update({"font.size": 12})

for f_E, label, color, ls, det in freq_configs:
    fP_arr = []
    res_arr = []
    M_valid = []

    for M in M_arr:
        # Skip if f_E above ISCO
        if f_E > isco_frequency(M):
            continue
        fP = newtonian_fP(M, f_E, tau_fid)
        if fP > 0 and fP < f_E:
            r = pulsar_term_residual_ns(M, fP, D_L_Mpc=100.0)
            fP_arr.append(fP)
            res_arr.append(r)
            M_valid.append(M)

    if len(M_valid) == 0:
        continue

    M_valid = np.array(M_valid)
    fP_arr = np.array(fP_arr)
    res_arr = np.array(res_arr)

    lbl = f"{label} ({det})"
    ax1.loglog(M_valid, fP_arr * 1e9, color=color, ls=ls, lw=2, label=lbl)
    ax2.loglog(M_valid, res_arr, color=color, ls=ls, lw=2, label=lbl)

# --- Top panel: f_P ---
ax1.axhspan(1, 300, alpha=0.10, color="green", zorder=0)
ax1.text(1.5e5, 8, "PTA band (1\u2013300 nHz)", fontsize=10, color="green",
         alpha=0.8, fontweight="bold")
ax1.set_ylabel(r"Pulsar-term frequency $f_P$ [nHz]", fontsize=13)
ax1.set_title(
    rf"Non-spinning, equal-mass binaries — $\tau = {tau_fid:.0f}$ yr "
    r"(1 kpc, geo$\,=1$)" "\n"
    r"Pulsar terms probe the nHz-frequency epoch of both LISA and $\mu$Ares sources",
    fontsize=12,
)
ax1.legend(fontsize=8.5, loc="upper right", ncol=2)
ax1.grid(True, alpha=0.3, which="both")
ax1.set_ylim(1e-4, 1e8)

# --- Bottom panel: residual amplitude ---
# PTA sensitivity levels
ax2.axhline(100, color="gray", ls=":", lw=1, alpha=0.6)
ax2.text(1.5e5, 130, "Current PTA (~100 ns)", fontsize=9, color="gray")
ax2.axhline(10, color="gray", ls=":", lw=1, alpha=0.6)
ax2.text(1.5e5, 13, "SKA-era (~10 ns)", fontsize=9, color="gray")
ax2.axhline(1, color="gray", ls=":", lw=1, alpha=0.6)
ax2.text(1.5e5, 1.3, "1 ns", fontsize=9, color="gray")

ax2.set_xlabel(r"Total mass $M$ [$M_\odot$]", fontsize=13)
ax2.set_ylabel(r"Pulsar-term residual $r_P$ [ns]  ($D_L = 100$ Mpc)",
               fontsize=13)
ax2.legend(fontsize=8.5, loc="upper left", ncol=2)
ax2.grid(True, alpha=0.3, which="both")
ax2.set_xlim(1e5, 3e10)
ax2.set_ylim(1e-8, 1e6)

plt.tight_layout()
outdir = Path(__file__).resolve().parent
fig.savefig(outdir / "fig_freq_landscape.png", dpi=150, bbox_inches="tight")
fig.savefig(outdir / "fig_freq_landscape.pdf", bbox_inches="tight")
print(f"\nFigure saved to {outdir}/fig_freq_landscape.png (.pdf)")
plt.close()


# ======================================================================
# Print key numbers
# ======================================================================
print("\n" + "=" * 70)
print(f"  Key f_P values at tau = {tau_fid:.0f} yr (1 kpc, geo=1)")
print("=" * 70)
print(f"  {'M [Msun]':>12s} {'f_E':>12s} {'f_P [nHz]':>12s} "
      f"{'r_P [ns]':>12s}  {'Detector':>8s}")
print(f"  {'-'*62}")

spot_checks = [
    (1e9,  1e-6,   "muAres"),
    (1e9,  0.1e-6, "muAres"),
    (1e8,  1e-6,   "muAres"),
    (1e7,  0.1e-3, "LISA"),
    (1e6,  1e-3,   "LISA"),
    (1e6,  0.1e-3, "LISA"),
    (1e5,  10e-3,  "LISA"),
]
for M, fE, det in spot_checks:
    fP = newtonian_fP(M, fE, tau_fid)
    r = pulsar_term_residual_ns(M, fP, D_L_Mpc=100.0)
    print(f"  {M:12.0e} {fE:12.2e} {fP*1e9:12.2f} {r:12.4e}  {det:>8s}")

print("\nDone.")
