"""
plot_residual_vs_mass.py

Pulsar-term timing residual r_P vs total mass M for several
Earth-term frequencies, with PTA detection thresholds.

Each curve is truncated at f_ISCO (the maximum physical mass
for that frequency), making it immediately clear why LISA
sources cannot produce detectable echoes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smbhb_evolution import G_SI, C_SI, M_SUN, PC, MPC, YR

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})


def h0_strain(Mc_kg, f_Hz, D_L_m):
    """Characteristic strain amplitude (factor-2 convention)."""
    return 2 * (G_SI * Mc_kg) ** (5./3) * (np.pi * f_Hz) ** (2./3) / (C_SI**4 * D_L_m)


def f_isco(M_tot_msun):
    """ISCO frequency for Schwarzschild BH [Hz]."""
    return C_SI**3 / (6**1.5 * np.pi * G_SI * M_tot_msun * M_SUN)


def residual_amplitude(M_tot_msun, f_E_Hz, D_L_Mpc, L_p_kpc,
                       geom_factor=0.5, iota=np.pi/4):
    """Approximate pulsar-term residual r_P [seconds]."""
    M = np.asarray(M_tot_msun, dtype=float) * M_SUN
    eta = 0.25
    Mc = M * eta ** 0.6
    D_L = D_L_Mpc * MPC
    Mc_s = G_SI * Mc / C_SI**3

    L_p = L_p_kpc * 1e3 * PC
    tau = L_p * geom_factor / C_SI

    coeff = (5. / 256) * Mc_s ** (-5./3) * np.pi ** (-8./3)
    f_P_inv83 = tau / coeff + f_E_Hz ** (-8./3)

    valid = f_P_inv83 > 0
    f_P = np.where(valid, f_P_inv83 ** (-3./8), np.nan)

    h_P = h0_strain(Mc, f_P, D_L)

    ci = np.cos(iota)
    F_eff = 0.4
    geom_proj = F_eff * np.sqrt((1 + ci**2)**2 + (2*ci)**2)
    r_P = h_P / (2 * np.pi * f_P) * geom_proj

    return r_P, f_P


def main():
    outdir = Path(__file__).resolve().parent

    log_M = np.linspace(5, 10.5, 800)
    M_arr = 10**log_M

    D_L = 100.0   # Mpc
    L_p = 1.0     # kpc

    # Frequencies: two μAres, two LISA
    curves = [
        (r'$1\;\mu$Hz',   1e-6,  'C0',  '-',  2.2),
        (r'$10\;\mu$Hz',  1e-5,  'C0',  '--', 1.8),
        (r'$0.1$ mHz',    1e-4,  'C3',  '-',  2.2),
        (r'$1$ mHz',      1e-3,  'C3',  '--', 1.8),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    for label, f_E, color, ls, lw in curves:
        r_P, f_P = residual_amplitude(M_arr, f_E, D_L, L_p)
        r_ns = r_P * 1e9

        # Truncate at ISCO: f_E must be < f_ISCO(M)
        M_max = C_SI**3 / (6**1.5 * np.pi * G_SI * f_E) / M_SUN
        log_M_max = np.log10(M_max)

        # Solid curve up to ISCO
        mask_phys = log_M <= log_M_max
        ax.plot(log_M[mask_phys], r_ns[mask_phys],
                color=color, ls=ls, lw=lw, label=label)

        # Endpoint marker at ISCO
        if log_M_max < 10.5:
            idx = np.searchsorted(log_M, log_M_max) - 1
            if 0 <= idx < len(r_ns) and np.isfinite(r_ns[idx]):
                ax.plot(log_M[idx], r_ns[idx], 'o', color=color,
                        ms=5, mfc='white', mew=1.5, zorder=4)

    # --- Detection thresholds ---
    ax.axhline(100, color='0.6', ls=':', lw=1, zorder=0)
    ax.axhline(1, color='0.6', ls=':', lw=1, zorder=0)

    # Shade detectable region
    ax.axhspan(1, 1e6, alpha=0.07, color='C0', zorder=0)

    # Threshold labels (right side)
    ax.text(10.4, 110, r'$\sigma_{\rm TOA}$', fontsize=8.5, color='0.45',
            ha='right', va='bottom')
    ax.text(10.4, 1.15, '1 ns', fontsize=8.5, color='0.45',
            ha='right', va='bottom')

    # Golden binary
    r_golden, _ = residual_amplitude(np.array([1e9]), 1e-6, 100.0, 1.0)
    ax.plot(9.0, r_golden[0]*1e9, 'k*', ms=14, zorder=5)
    ax.annotate('Golden binary',
                xy=(9.0, r_golden[0]*1e9),
                xytext=(7.3, r_golden[0]*1e9*15),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

    # Band labels
    ax.text(9.5, 2e-4, r'$\mu$Ares band', fontsize=10, color='C0',
            ha='center', fontweight='bold', alpha=0.7)
    ax.text(6.5, 2e-4, 'LISA band', fontsize=10, color='C3',
            ha='center', fontweight='bold', alpha=0.7)

    # Annotation: why LISA fails
    ax.annotate('ISCO cutoff:\nLISA sources\ntoo light',
                xy=(7.65, 3e-2), fontsize=8, color='C3',
                ha='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='C3', alpha=0.8))

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}\,(M_{\rm tot}/M_\odot)$')
    ax.set_ylabel(r'Pulsar-term residual $r_P$ [ns]')
    ax.set_xlim(5, 10.5)
    ax.set_ylim(1e-5, 1e5)

    # Custom legend: group by band
    custom_handles = [
        Line2D([0], [0], color='C0', ls='-', lw=2),
        Line2D([0], [0], color='C0', ls='--', lw=1.8),
        Line2D([0], [0], color='C3', ls='-', lw=2),
        Line2D([0], [0], color='C3', ls='--', lw=1.8),
    ]
    custom_labels = [c[0] for c in curves]
    ax.legend(custom_handles, custom_labels, fontsize=8.5,
              loc='upper left', title=r'$f_E$', title_fontsize=9,
              framealpha=0.9, handlelength=2.2)

    # Subtitle
    ax.set_title(r'$D_L = 100$ Mpc, $L_p = 1$ kpc (equal mass, non-spinning)',
                 fontsize=9, color='0.4', pad=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig_residual_vs_mass.pdf", bbox_inches='tight')
    fig.savefig(outdir / "fig_residual_vs_mass.png", dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir}/fig_residual_vs_mass.pdf")


if __name__ == "__main__":
    main()
