"""
bhmf_rates.py — SMBHB echo source counts and GWB amplitude
using the Liepold & Ma (2024) black hole mass function.

References
----------
[LM24]  Liepold & Ma, arXiv:2407.14595
        "Big Galaxies and Big Black Holes"
[S21]   Sesana et al., Exp. Astron. 51, 1333 (2021)
[P01]   Phinney, arXiv:astro-ph/0108028
[SP23]  Sato-Polito et al. (2023)
[A23]   Agazie et al. (2023b), NANOGrav population synthesis
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# Constants (SI)
# ======================================================================
G     = 6.67430e-11       # m^3 kg^-1 s^-2
c     = 2.99792458e8      # m/s
Msun  = 1.98892e30        # kg
pc    = 3.08567758e16     # m
Mpc   = 1e6 * pc          # m
yr    = 365.25 * 86400    # s


# ======================================================================
# 1. Black hole mass function: Liepold & Ma (2024), Eq. (3)
#
#    dn / d ln M_BH = phi * (M_BH / M_s)^(alpha+1)
#                     * exp(-(M_BH / M_s)^beta)
#
#    Parameters from their combined posterior (Sec. 4.1):
#      alpha             = -1.27 +/- 0.02
#      beta              =  0.45 +/- 0.02
#      log10(phi/Mpc^-3) = -2.00 +/- 0.07
#      log10(M_s/Msun)   =  8.09 +/- 0.09
# ======================================================================

# Central values
ALPHA = -1.27
BETA  =  0.45
LOG10_PHI = -2.00   # log10(phi / Mpc^-3)
LOG10_MS  =  8.09   # log10(M_s / Msun)

PHI = 10**LOG10_PHI              # Mpc^{-3}
MS  = 10**LOG10_MS               # Msun


def bhmf_lm24(M_bh_msun):
    """
    Liepold & Ma (2024) BHMF, Eq. (3).

    Parameters
    ----------
    M_bh_msun : float or array
        Black hole mass in solar masses.

    Returns
    -------
    dn_dlnM : float or array
        Number density per unit ln(M_BH), in Mpc^{-3}.
    """
    x = np.asarray(M_bh_msun, dtype=float) / MS
    return PHI * x**(ALPHA + 1) * np.exp(-x**BETA)


# ======================================================================
# 2. Stochastic GWB amplitude h_c
#
#    Following Phinney (2001) and LM24 Eq. (4)-(5).
#
#    h_c^2(f) = (4 pi) / (3 c^2)  *  1 / (pi f)^(4/3)
#               * integral[ dM  (G M)^{5/3}  (dn/dM) ]
#               * <q(1+q)^{-2}>  *  <(1+z)^{-1/3}>
#
#    LM24 rewrite this as Eq. (5) in convenient units:
#
#    h_c^2 = 1.18e-30  *  (yr^{-1} / f)^{4/3}
#            * <q/(1+q)^2>  *  <(1+z)^{-1/3}>
#            * integral[ dM  (M / 10^9 Msun)^{5/3}
#                        * d/dM (n / 10^{-4} Mpc^{-3}) ]
#
#    The averages over mass ratio q and redshift z are computed
#    from the adopted distributions (LM24 Eq. 6, fit to
#    NANOGrav population synthesis of Agazie et al. 2023b):
#
#      p_z(z) ~ z^gamma * exp(-(z/z_*)^beta_z)
#        with gamma = 1.0,  z_* = 0.5,  beta_z = 2.0
#
#      p_q(q) ~ q^2   for 0 < q <= 1
#
#    Yielding (LM24 Sec. 5.1):
#      <q / (1+q)^2>  = 0.238
#      <(1+z)^{-1/3}> = 0.890
# ======================================================================

Q_AVG = 0.238     # <q/(1+q)^2> from p_q ~ q^2
Z_AVG = 0.890     # <(1+z)^{-1/3}> from Eq. (6) distributions


def hc_squared(f_hz, bhmf=bhmf_lm24,
               log10M_lo=7.0, log10M_hi=11.5):
    """
    Characteristic strain squared, h_c^2(f), from LM24 Eq. (5).

    Parameters
    ----------
    f_hz : float
        GW frequency [Hz].
    bhmf : callable
        BHMF function: M_msun -> dn/d(ln M) [Mpc^{-3}].
    log10M_lo, log10M_hi : float
        Integration bounds in log10(M/Msun).

    Returns
    -------
    hc2 : float
    """
    # LM24 Eq. (5) prefactor
    prefactor = 1.18e-30 * (1.0 / (f_hz * yr))**(4.0 / 3)

    # Integral: int dM (M/10^9)^{5/3} * (dn/dM) / 10^{-4}
    # With substitution u = log10(M), dM = M ln(10) du,
    # and dn/dM = bhmf(M) / M:
    #   = int du * ln(10) * (M/10^9)^{5/3} * bhmf(M) / 10^{-4}

    def integrand(log10M):
        M = 10**log10M
        return np.log(10) * (M / 1e9)**(5.0 / 3) * bhmf(M) / 1e-4

    integral, _ = quad(integrand, log10M_lo, log10M_hi, limit=200)

    return prefactor * Q_AVG * Z_AVG * integral


def hc(f_hz, **kw):
    """Characteristic strain h_c(f)."""
    return np.sqrt(hc_squared(f_hz, **kw))


def dhc2_dlog10M(log10M, f_hz, bhmf=bhmf_lm24):
    """
    Differential contribution dh_c^2 / d(log10 M_BH).

    For reproducing LM24 Fig. 5 (left panel).
    """
    M = 10**log10M
    prefactor = 1.18e-30 * (1.0 / (f_hz * yr))**(4.0 / 3)
    return prefactor * Q_AVG * Z_AVG * np.log(10) * (M / 1e9)**(5.0 / 3) * bhmf(M) / 1e-4


# ======================================================================
# 3. Cumulative SMBH number counts
#
#    N(> M_min) = V * int_{M_min}^{infty} (dn/dM) dM
#               = V * int_{ln M_min}^{infty} bhmf(M) d(ln M)
#
#    where V = (4 pi / 3) D_max^3  is the comoving volume.
# ======================================================================

def n_above(M_min_msun, bhmf=bhmf_lm24, log10M_hi=11.5):
    """
    Comoving number density of SMBHs above M_min [Mpc^{-3}].
    """
    def integrand(log10M):
        return bhmf(10**log10M) / np.log(10)  # dn/d(log10 M) = bhmf / ln(10)

    result, _ = quad(integrand, np.log10(M_min_msun), log10M_hi, limit=200)
    return result


def N_above(M_min_msun, D_max_Mpc, **kw):
    """
    Total number of SMBHs above M_min within distance D_max.
    """
    V = (4 * np.pi / 3) * D_max_Mpc**3
    return n_above(M_min_msun, **kw) * V


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

    outdir = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # Print BHMF parameters
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Liepold & Ma (2024) BHMF parameters [Eq. 3]")
    print("=" * 60)
    print(f"  alpha             = {ALPHA}")
    print(f"  beta              = {BETA}")
    print(f"  log10(phi/Mpc^-3) = {LOG10_PHI}  ->  phi = {PHI:.4e} Mpc^-3")
    print(f"  log10(M_s/Msun)   = {LOG10_MS}  ->  M_s = {MS:.3e} Msun")
    print(f"\n  Adopted averages [Eq. 6]:")
    print(f"  <q/(1+q)^2>  = {Q_AVG}")
    print(f"  <(1+z)^-1/3> = {Z_AVG}")

    # ------------------------------------------------------------------
    # h_c at f = 1/yr
    # ------------------------------------------------------------------
    f_ref = 1.0 / yr
    hc_val = hc(f_ref)
    print(f"\n  h_c(f = 1/yr) = {hc_val:.2e}  = {hc_val/1e-15:.2f} x 10^-15")
    print(f"  [LM24 report ~2.0 x 10^-15]")
    print(f"  PTA:  NANOGrav = (2.4 +0.7/-0.6) x 10^-15")
    print(f"        PPTA     = (2.04 +0.41/-0.36) x 10^-15")

    # ------------------------------------------------------------------
    # SMBH counts
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Cumulative SMBH counts N(> M_min)")
    print(f"{'='*60}")

    mass_thresholds = [1e8, 3e8, 1e9, 3e9, 1e10]
    distances = [108, 200, 500]

    print(f"\n  {'':>10s}", end="")
    for D in distances:
        print(f"  {'D=%d Mpc' % D:>14s}", end="")
    print()
    print(f"  {'-'*56}")

    for Mmin in mass_thresholds:
        print(f"  M>{Mmin:.0e}", end="")
        for D in distances:
            N = N_above(Mmin, D)
            print(f"  {N:14.1f}", end="")
        print()

    # Check against LM24 Sec. 4.2:
    # They predict 1-14 SMBHs with M_BH > 10^{10} within 108 Mpc
    N_check = N_above(1e10, 108)
    print(f"\n  Sanity check: N(>10^10) within 108 Mpc = {N_check:.1f}")
    print(f"  [LM24 predict 1-14 (90% CI)]")

    # ------------------------------------------------------------------
    # Figure 1: BHMF and cumulative counts
    # ------------------------------------------------------------------
    M_arr = np.logspace(7, 11.5, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.rcParams.update({"font.size": 12})

    # Left: dn/d(ln M)
    dn = bhmf_lm24(M_arr)
    ax1.loglog(M_arr, dn, color="#7b2d8e", lw=2.5,
               label="Liepold & Ma (2024)")
    ax1.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]")
    ax1.set_ylabel(r"$dn/d\ln M_{\rm BH}$ [Mpc$^{-3}$]")
    ax1.set_xlim(1e7, 3e11)
    ax1.set_ylim(1e-8, 1e-1)
    ax1.axvspan(3e8, 3e9, alpha=0.10, color="red", zorder=0)
    ax1.text(4e8, 2e-2, "Echo sources\n" + r"$M \sim 10^{8.5}$–$10^{9.5}$",
             fontsize=9, color="red")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("Local BHMF [LM24 Eq. 3]")

    # Right: cumulative counts within 108 Mpc
    V_108 = (4 * np.pi / 3) * 108**3
    N_cum = np.array([n_above(M) * V_108 for M in M_arr])
    ax2.loglog(M_arr, N_cum, color="#7b2d8e", lw=2.5)
    ax2.set_xlabel(r"$M_{\rm BH, min}$ [$M_\odot$]")
    ax2.set_ylabel(r"$N(>M_{\rm BH})$ within 108 Mpc")
    ax2.set_xlim(1e7, 3e11)
    ax2.set_ylim(0.1, 1e6)
    ax2.axhspan(1, 14, alpha=0.15, color="gray", zorder=0)
    ax2.text(3e10, 3, r"LM24: 1–14 at $M>10^{10}$", fontsize=9,
             color="gray")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title(r"Cumulative counts (MASSIVE volume)")

    plt.tight_layout()
    fig.savefig(outdir / "fig_bhmf_comparison.png", dpi=150,
                bbox_inches="tight")
    fig.savefig(outdir / "fig_bhmf_comparison.pdf", bbox_inches="tight")
    print(f"\nFigure -> fig_bhmf_comparison.pdf")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 2: h_c spectrum and differential contribution
    # ------------------------------------------------------------------
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: dh_c^2 / d(log10 M)
    log10M_plot = np.linspace(7, 12, 400)
    dh = np.array([dhc2_dlog10M(lm, f_ref) for lm in log10M_plot])
    ax1.plot(10**log10M_plot, dh / 1e-30, color="#7b2d8e", lw=2.5)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]")
    ax1.set_ylabel(r"$dh_c^2/d\log_{10}M_{\rm BH}$ [$10^{-30}$]")
    ax1.set_title(r"Differential $h_c^2$ at $f = 1\,\rm yr^{-1}$ [LM24 Eq. 5]")
    ax1.set_xlim(1e7, 1e12)
    ax1.grid(True, alpha=0.3)

    # Right: h_c(f) with PTA data
    f_arr = np.logspace(-9.2, -7.0, 80)
    hc_arr = np.array([hc(f) for f in f_arr])
    ax2.loglog(f_arr * 1e9, hc_arr / 1e-15, color="#7b2d8e", lw=2.5,
               label="LM24 BHMF")

    # PTA measurements at f = 1/yr ~ 31.7 nHz
    f_pta_nHz = 1.0 / yr * 1e9
    ax2.errorbar([f_pta_nHz], [2.4], yerr=[[0.6], [0.7]], fmt="v",
                 color="red", ms=10, capsize=5, lw=1.5,
                 label="NANOGrav", zorder=5)
    ax2.errorbar([f_pta_nHz * 1.03], [2.04], yerr=[[0.36], [0.41]],
                 fmt="^", color="darkgreen", ms=10, capsize=5, lw=1.5,
                 label="PPTA", zorder=5)
    ax2.errorbar([f_pta_nHz * 0.97], [2.5], yerr=[[0.7], [0.7]],
                 fmt="s", color="royalblue", ms=8, capsize=5, lw=1.5,
                 label="EPTA+InPTA", zorder=5)

    ax2.set_xlabel(r"$f$ [nHz]")
    ax2.set_ylabel(r"$h_c$ [$10^{-15}$]")
    ax2.set_title(r"GWB characteristic strain [LM24 Eq. 5]")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig2.savefig(outdir / "fig_hc_bhmf.png", dpi=150, bbox_inches="tight")
    fig2.savefig(outdir / "fig_hc_bhmf.pdf", bbox_inches="tight")
    print(f"Figure -> fig_hc_bhmf.pdf")
    plt.close()

    print("\nDone.")
