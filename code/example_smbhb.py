"""
example_smbhb.py

Demonstrates the SMBHB evolution code using the fiducial parameters
from Table I of Mingarelli et al. (2012), PRL 109, 081104.

Produces three figures:
  1. Frequency evolution from pulsar to Earth
  2. GW phase decomposition by pN order
  3. Timing residuals (Earth term, pulsar term, total)

Also prints a comparison with Table I cycle counts.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smbhb_evolution import SMBHBEvolution, Pulsar

# ======================================================================
# 1.  Fiducial binary: m1 = m2 = 1e9 Msun, f_E = 100 nHz
#     (Table I, row 1 of Mingarelli et al. 2012)
# ======================================================================
binary = SMBHBEvolution(
    m1=1e9,
    m2=1e9,
    chi1=0.5,
    chi2=0.5,
    kappa1=0.3,            # moderate misalignment
    kappa2=0.3,
    f_gw_earth=100e-9,     # 100 nHz
    D_L=100.0,             # 100 Mpc
    iota=np.pi / 4,
    psi=0.2,
    theta_s=np.pi / 3,
    phi_s=np.pi / 4,
)
binary.summary()

# ======================================================================
# 2.  Fiducial pulsar at 1 kpc
# ======================================================================
psr = Pulsar(name="J0437-4715", theta=2.0, phi=1.0, dist_kpc=1.0)
tau = binary.light_travel_time(psr)
print(f"\nPulsar: {psr.name}")
print(f"  Distance        = {psr.dist_kpc} kpc")
print(f"  Light-travel τ  = {tau:.0f} yr")
print(f"  Geometric factor= {1 + np.dot(binary.Omega_hat, psr.p_hat):.3f}")

# ======================================================================
# 3.  pN phase decomposition over the Earth-pulsar baseline
# ======================================================================
print("\n--- pN Decomposition ---")
decomp = binary.pn_decomposition(abs(tau), n_points=20000)

print(f"\nGW cycles over {abs(tau):.0f} yr baseline (Pulsar → Earth):")
print(f"  {'Order':12s} {'Cycles':>12s}")
print(f"  {'-' * 26}")
for key in ["Newtonian", "1pN", "1.5pN", "SO", "2pN", "Thomas", "Total"]:
    val = decomp["cycles"][key]
    print(f"  {key:12s} {val:12.1f}")

# ======================================================================
# 4.  Table I comparison (non-spinning, for clean mass-only terms)
# ======================================================================
print("\n" + "=" * 58)
print("  Comparison with Table I (Mingarelli et al. 2012)")
print("  Non-spinning: chi1 = chi2 = 0, 1 kpc baseline")
print("=" * 58)
binary_ns = SMBHBEvolution(
    m1=1e9, m2=1e9,
    chi1=0.0, chi2=0.0,
    f_gw_earth=100e-9,
    D_L=100.0,
)
# Exact 1 kpc light travel time with geometric factor (1+Omega.p)=1
from smbhb_evolution import PC, C_SI, YR as YR_s
tau_1kpc = 1000 * PC / C_SI / YR_s
print(f"  Exact 1 kpc light-travel time: {tau_1kpc:.1f} yr")
decomp_ns = binary_ns.pn_decomposition(tau_1kpc, n_points=30000)

# TaylorT2 analytic cycle counts — direct comparison with Table I
print(f"\n  {'':12s} {'Code':>10s} {'Table I':>10s}")
print(f"  {'-' * 34}")
table_vals = {
    "Newtonian": 4267.8,
    "1pN": 77.3,
    "1.5pN": -45.8,
    "2pN": 2.2,
    "Total": 4305.1,
}
# Table I "1.5pN" column = mass/tail only (-10 pi coefficient)
# Table I "Spin orbit/Θ" column includes the SO with Θ=1
# For non-spinning, SO = 0, so 1.5pN = mass/tail only
for key in ["Newtonian", "1pN", "1.5pN", "2pN", "Total"]:
    code_val = decomp_ns["cycles"][key]
    tbl = table_vals.get(key, "—")
    print(f"  {key:12s} {code_val:10.1f}  {tbl:>10}")
print(f"\n  v_E = {decomp_ns['cycles']['v_E']:.6f}")
print(f"  v_P = {decomp_ns['cycles']['v_P']:.6f}")
print(f"  f_P = {decomp_ns['cycles']['f_P_nHz']:.2f} nHz")
print(f"  Δf  = {decomp_ns['cycles']['delta_f_nHz']:.2f} nHz")

# ======================================================================
# 5.  Additional Table I rows for validation
# ======================================================================
print("\n--- Additional configurations ---")
configs = [
    {"m1": 1e9, "m2": 1e9, "f": 50e-9, "span": tau_1kpc,
     "label": "1e9+1e9, 50nHz, ~1kpc"},
    {"m1": 1e8, "m2": 1e8, "f": 100e-9, "span": tau_1kpc,
     "label": "1e8+1e8, 100nHz, ~1kpc"},
    {"m1": 1e9, "m2": 1e9, "f": 100e-9, "span": 10,
     "label": "1e9+1e9, 100nHz, 10yr"},
]
for cfg in configs:
    b = SMBHBEvolution(m1=cfg["m1"], m2=cfg["m2"],
                        chi1=0, chi2=0,
                        f_gw_earth=cfg["f"], D_L=100)
    d = b.pn_decomposition(cfg["span"])
    print(f"\n  {cfg['label']}:")
    for k in ["Newtonian", "1pN", "1.5pN", "2pN", "Total"]:
        print(f"    {k:12s} {d['cycles'][k]:10.1f} cycles")

# ======================================================================
# 6.  Timing residual
# ======================================================================
print("\n--- Timing Residual ---")
res = binary.timing_residual(psr, T_obs_yr=10.0, n_obs=5000)
print(f"  Δf (Earth − Pulsar) = {res['delta_f_nHz']:.2f} nHz")
print(f"  F+ = {res['Fp']:.4f},  F× = {res['Fc']:.4f}")
print(f"  Max |residual| = {np.max(np.abs(res['residual_ns'])):.1f} ns")

# ======================================================================
# 7.  FIGURES
# ======================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 13))
plt.rcParams.update({"font.size": 12})

# --- Panel 1: Frequency evolution ---
ax = axes[0]
t = decomp["t_yr"]
ax.plot(t, decomp["f"]["2pN"] * 1e9, "k-", lw=1.5, label="Full 2pN")
ax.plot(t, decomp["f"]["Newtonian"] * 1e9, "--", color="C0",
        lw=1, label="Newtonian")
ax.axvline(0, color="r", ls=":", lw=0.8, alpha=0.7)
ax.axvline(t[0], color="b", ls=":", lw=0.8, alpha=0.7)
ax.annotate("Earth", xy=(0, ax.get_ylim()[0]), xytext=(5, 5),
            textcoords="offset points", color="r", fontsize=10)
ax.annotate("Pulsar", xy=(t[0], ax.get_ylim()[0]), xytext=(5, 5),
            textcoords="offset points", color="b", fontsize=10)
ax.set_xlabel("Time [yr]  (0 = Earth epoch)")
ax.set_ylabel(r"$f_{\rm GW}$ [nHz]")
ax.set_title(
    r"GW Frequency Evolution: $m_1 = m_2 = 10^9\,M_\odot$, "
    r"$f_E = 100$ nHz, $L_p = 1$ kpc"
)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# --- Panel 2: Phase decomposition ---
ax = axes[1]
colors = {"Newtonian": "C0", "1pN": "C1", "1.5pN": "C2", "2pN": "C3"}
for key in ["Newtonian", "1pN", "1.5pN", "2pN"]:
    cyc = np.abs(decomp["dPhi"][key]) / (2 * np.pi)
    cyc[cyc < 1e-6] = 1e-6  # avoid log(0)
    ax.plot(t, cyc, color=colors[key], lw=1.5, label=key)
cyc_T = np.abs(decomp["phi_T"]) / (2 * np.pi)
cyc_T[cyc_T < 1e-6] = 1e-6
ax.plot(t, cyc_T, "C4", ls="--", lw=1.5, label="Thomas prec.")
ax.set_xlabel("Time [yr]  (0 = Earth epoch)")
ax.set_ylabel("GW cycles (cumulative)")
ax.set_title("Accumulated GW Cycles by Post-Newtonian Order")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Panel 3: Timing residual ---
ax = axes[2]
ax.plot(res["t_yr"], res["earth_term"] * 1e15, "C0", lw=0.8,
        alpha=0.7, label="Earth term")
ax.plot(res["t_yr"], res["pulsar_term"] * 1e15, "C1", lw=0.8,
        alpha=0.7, label="Pulsar term")
ax2 = ax.twinx()
ax2.plot(res["t_yr"], res["residual_ns"], "k-", lw=1.2,
         label="Timing residual")
ax.set_xlabel("Observation time [yr]")
ax.set_ylabel(r"Strain $\times\,10^{15}$", color="C0")
ax2.set_ylabel("Timing residual [ns]", color="k")
ax.set_title(
    f"Timing Residual: {psr.name}  "
    rf"($L_p = {psr.dist_kpc}$ kpc, $\tau = {res['tau_yr']:.0f}$ yr, "
    rf"$\Delta f = {res['delta_f_nHz']:.1f}$ nHz)"
)
# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
outdir = Path(__file__).resolve().parent
fig.savefig(outdir / "smbhb_evolution.png", dpi=150, bbox_inches="tight")
fig.savefig(outdir / "smbhb_evolution.pdf", bbox_inches="tight")
print(f"\nFigures saved to {outdir}/smbhb_evolution.png (.pdf)")
plt.close()

# ======================================================================
# 8.  Multi-pulsar comparison
# ======================================================================
print("\n--- Multi-pulsar frequency shift ---")
pulsars = [
    Pulsar("PSR_A", theta=1.0, phi=0.5, dist_kpc=0.5),
    Pulsar("PSR_B", theta=2.0, phi=1.0, dist_kpc=1.0),
    Pulsar("PSR_C", theta=1.5, phi=2.5, dist_kpc=2.0),
    Pulsar("PSR_D", theta=0.8, phi=3.5, dist_kpc=0.3),
]

fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
for p in pulsars:
    r = binary.timing_residual(p, T_obs_yr=10, n_obs=3000)
    ax.plot(r["t_yr"], r["residual_ns"], lw=1.2,
            label=f"{p.name} ({p.dist_kpc} kpc, "
                  f"τ={r['tau_yr']:.0f} yr, "
                  f"Δf={r['delta_f_nHz']:.1f} nHz)")
    print(f"  {p.name}: τ = {r['tau_yr']:7.0f} yr, "
          f"Δf = {r['delta_f_nHz']:6.2f} nHz, "
          f"max|r| = {np.max(np.abs(r['residual_ns'])):.1f} ns")
ax.set_xlabel("Observation time [yr]")
ax.set_ylabel("Timing residual [ns]")
ax.set_title(
    r"Timing Residuals for Multiple Pulsars  "
    r"($m_1 = m_2 = 10^9\,M_\odot$, $f_E = 100$ nHz)"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(outdir / "smbhb_multi_pulsar.png", dpi=150, bbox_inches="tight")
print(f"\nMulti-pulsar figure saved to {outdir}/smbhb_multi_pulsar.png")
plt.close()

print("\nDone.")
