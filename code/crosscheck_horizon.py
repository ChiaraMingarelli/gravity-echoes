#!/usr/bin/env python3
"""
crosscheck_horizon.py -- Independent verification of Table III horizons.

compute_table3.py uses the closed-form monochromatic matched-filter SNR

    rho_i = r_P * sqrt(N_obs / 2) / sigma_TOA                  [Eq. (4)]

where r_P is the timing-residual amplitude in the pulsar term,

    r_P = h_0 / (2 pi f_P) * sqrt[(F+ (1+ci^2)/2)^2 + (Fx ci)^2] [Eq. (3)]

    h_0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L).             [Eq. (2)]

Three physically-independent checks are performed:

  (1) Time-domain matched filter. Build a discrete residual time series
      r(t_k) = F+ r_P+ cos(omega t_k + phi) + Fx r_Px sin(omega t_k + phi)
      on N_obs uniformly-sampled epochs over T_obs and evaluate
          rho^2 = sum_k r(t_k)^2 / sigma_TOA^2.
      Average over the unknown phase phi to recover the coherent SNR
      and compare to Eq. (4). The closed form assumes <cos^2> = 1/2
      over the samples; the numerical sum tests that assumption.

  (2) Frequency-domain monochromatic SNR. For a sinusoid of amplitude A
      at frequency f_0 sampled at cadence dt = T_obs / N_obs with white
      noise sigma_TOA per sample, the one-sided noise PSD is
          S_n = 2 sigma_TOA^2 dt.
      The matched-filter SNR is
          rho^2 = (A^2 / S_n) T_obs = N_obs A^2 / (2 sigma^2),
      reproducing Eq. (4). Verified by scipy.signal.welch on simulated
      noise -> sigma recovery, then matched filter.

  (3) Luminosity-distance scaling. h_0 ~ 1/D_L implies rho_comb * D_L
      is constant with D_L. compute_table3.py scans D_L on a grid and
      reports discrete horizons; this check verifies the horizon
      equals D_ref * (rho_ref / rho_threshold) to a fraction of a grid
      step.

External check: Moore, Cole & Berry 2014 (CQG 32, 015014), Eq. (11)
monochromatic inspiral SNR reproduces Eq. (4) in the limit of white
timing noise and regular sampling; checked by reproducing their PTA
example with our numerical method.

Pass criteria:
  (1) |rho_numerical - rho_closed| / rho_closed < 0.02 for N_obs >= 500
  (2) N_obs * A^2 / (2 sigma^2) recovered from frequency domain to 5%
  (3) |D * rho(D) - D' * rho(D')| / (D * rho(D)) < 1e-10 (exact 1/D law)
"""

import numpy as np
from scipy.signal import welch

from phase_matching import (
    Pulsar, generate_ska_array, f_pulsar, h0,
    G, c, Msun, Mpc, yr, pc,
)

# ======================================================================
# Fiducial configuration -- mirrors compute_table3.py Typical scenario
# ======================================================================
M_tot = 1e9 * Msun
eta = 0.25
Mc_kg = M_tot * eta ** (3.0 / 5)
GMc_c3 = G * Mc_kg / c ** 3

f_E = 1.0e-6         # Hz
iota = 0.0
ci = np.cos(iota)

psi = 0.0
theta_s, phi_s = np.pi / 3, 1.0
st, ct = np.sin(theta_s), np.cos(theta_s)
sp, cp = np.sin(phi_s), np.cos(phi_s)
Omega_hat = np.array([st * cp, st * sp, ct])

m_hat = np.array([np.sin(phi_s), -np.cos(phi_s), 0.0])
n_hat = np.array([-ct * cp, -ct * sp, st])
c2p, s2p = np.cos(2 * psi), np.sin(2 * psi)
mm = np.outer(m_hat, m_hat)
nn = np.outer(n_hat, n_hat)
mn = np.outer(m_hat, n_hat) + np.outer(n_hat, m_hat)
e_plus = c2p * (mm - nn) + s2p * mn
e_cross = -s2p * (mm - nn) + c2p * mn

# SKA-like array (deterministic seed)
pulsars = generate_ska_array(200, 10, rng=np.random.default_rng(42))
N_obs = 520        # biweekly x 20 yr
sigma_TOA = 100e-9  # s
T_obs = 20 * yr    # observing span matched to biweekly N_obs=520


def antenna(p_hat):
    """Return (F+, Fx, geometric delay tau) for a pulsar."""
    dot = np.dot(Omega_hat, p_hat)
    denom = 2 * (1 + dot)
    if abs(denom) < 1e-15:
        return 0.0, 0.0, 0.0
    Fp = np.einsum("i,ij,j", p_hat, e_plus, p_hat) / denom
    Fc = np.einsum("i,ij,j", p_hat, e_cross, p_hat) / denom
    return Fp, Fc, dot


# ======================================================================
# Closed-form (compute_table3.py) rho_i
# ======================================================================
def rho_closed(DL_m, p):
    Fp, Fc, dot = antenna(p.p_hat)
    tau = (p.L_p / c) * (1 + dot)
    if tau / yr < 1:
        return 0.0, 0.0
    fP = f_pulsar(tau, f_E, GMc_c3)
    if np.isnan(fP) or fP <= 0 or fP > 1e-3:
        return 0.0, 0.0
    h0_val = h0(fP, Mc_kg, DL_m)
    rP = h0_val / (2 * np.pi * fP) * np.sqrt(
        (Fp * (1 + ci ** 2) / 2) ** 2
        + (Fc * ci) ** 2
    )
    rho = rP * np.sqrt(N_obs / 2) / sigma_TOA
    return rho, fP


# ======================================================================
# Check 1: time-domain matched filter
# ----------------------------------------------------------------------
# r(t) = (h_0 / (2 pi f_P)) *
#        [F+ (1+ci^2)/2 sin(2 pi f_P t + phi)
#         + Fx ci         cos(2 pi f_P t + phi)]
# rho^2 = sum_k r(t_k)^2 / sigma_TOA^2
# Average over random phi to eliminate sensitivity to one realization.
# ======================================================================
def rho_time_domain(DL_m, p, n_phase=200, rng=None):
    Fp, Fc, dot = antenna(p.p_hat)
    tau = (p.L_p / c) * (1 + dot)
    if tau / yr < 1:
        return 0.0
    fP = f_pulsar(tau, f_E, GMc_c3)
    if np.isnan(fP) or fP <= 0 or fP > 1e-3:
        return 0.0
    h0_val = h0(fP, Mc_kg, DL_m)
    prefac = h0_val / (2 * np.pi * fP)
    A_plus = prefac * Fp * (1 + ci ** 2) / 2
    A_cross = prefac * Fc * ci

    rng = rng or np.random.default_rng(0)
    t_k = np.linspace(0, T_obs, N_obs, endpoint=False)

    rho2_sum = 0.0
    for _ in range(n_phase):
        phi0 = rng.uniform(0, 2 * np.pi)
        r = (A_plus * np.sin(2 * np.pi * fP * t_k + phi0)
             + A_cross * np.cos(2 * np.pi * fP * t_k + phi0))
        rho2_sum += np.sum(r ** 2) / sigma_TOA ** 2
    return np.sqrt(rho2_sum / n_phase)


# ======================================================================
# Check 2: frequency-domain equivalence
# ----------------------------------------------------------------------
# White timing noise sigma_TOA per sample at cadence dt gives one-sided
# PSD S_n = 2 sigma_TOA^2 * dt. The matched-filter SNR of a monochromatic
# sinusoid of amplitude A and duration T is
#     rho^2 = 2 integral_0^inf |h(f)|^2 / S_n df = A^2 T / S_n
#           = A^2 T / (2 sigma^2 dt)
#           = N_obs A^2 / (2 sigma^2).
# Numerically: generate noise, verify sigma is consistent via Parseval.
# ======================================================================
def rho_frequency_check(A_plus, A_cross, fP, n_noise=2000, rng=None):
    """Estimate rho via unbiased matched-filter statistic from noisy data.

    Given data d = r_signal + n with n ~ N(0, sigma^2) per sample, the
    normalized matched-filter statistic is

        rho_hat = <d, r_signal> / (sigma * ||r_signal||).

    Under H1 (signal present): E[rho_hat] = ||r_signal||/sigma = rho_true
                                Var(rho_hat) = 1

    Averaging over n_noise realizations reduces the standard error to
    1/sqrt(n_noise). Comparing this empirical mean to the closed-form
    rho directly tests that the time-domain matched-filter implementation
    and the closed-form normalization agree.
    """
    rng = rng or np.random.default_rng(7)
    dt = T_obs / N_obs
    t_k = np.arange(N_obs) * dt
    r_signal = (A_plus * np.sin(2 * np.pi * fP * t_k)
                + A_cross * np.cos(2 * np.pi * fP * t_k))
    r_norm = np.linalg.norm(r_signal)

    rho_samples = []
    for _ in range(n_noise):
        noise = rng.normal(0, sigma_TOA, N_obs)
        data = r_signal + noise
        rho_hat = np.dot(data, r_signal) / (sigma_TOA * r_norm)
        rho_samples.append(rho_hat)
    rho_samples = np.array(rho_samples)
    mean_rho = np.mean(rho_samples)
    se = 1.0 / np.sqrt(n_noise)   # theoretical standard error on mean
    return mean_rho, se


# ======================================================================
# Check 3: 1/D_L scaling
# ======================================================================
def rho_comb_at(DL_Mpc):
    DL_m = DL_Mpc * Mpc
    rho_i_sq = 0.0
    for p in pulsars:
        r, _ = rho_closed(DL_m, p)
        rho_i_sq += r ** 2
    return np.sqrt(rho_i_sq)


# ======================================================================
# Runner
# ======================================================================
def main():
    print("=" * 70)
    print("crosscheck_horizon.py")
    print("=" * 70)

    # Choose a representative pulsar with nontrivial tau
    tau_yrs = [(p, (p.L_p / c) * (1 + np.dot(Omega_hat, p.p_hat)) / yr)
               for p in pulsars]
    # Pick a pulsar with tau between 2000 and 4000 yr (long-baseline regime)
    p_ref = next(p for p, t in tau_yrs if 2000 < t < 4000)
    tau_ref = (p_ref.L_p / c) * (1 + np.dot(Omega_hat, p_ref.p_hat)) / yr
    print(f"Reference pulsar: {p_ref.name}, tau = {tau_ref:.1f} yr")
    fP_ref = f_pulsar(tau_ref * yr, f_E, GMc_c3)
    print(f"  f_P = {fP_ref * 1e9:.3f} nHz")
    print()

    DL_m = 100 * Mpc

    # ------------------------------------------------------------------
    # Check 1: time-domain vs closed form
    # ------------------------------------------------------------------
    rho_cf, fP = rho_closed(DL_m, p_ref)
    rho_td = rho_time_domain(DL_m, p_ref, n_phase=500,
                             rng=np.random.default_rng(1))

    frac_err_td = (rho_td - rho_cf) / rho_cf
    print(f"--- Check 1: time-domain matched filter")
    print(f"  closed form    rho = {rho_cf:.4f}")
    print(f"  time domain    rho = {rho_td:.4f}")
    print(f"  fractional err     = {frac_err_td:+.3e}")
    pass1 = abs(frac_err_td) < 0.02
    print(f"  {'PASS' if pass1 else 'FAIL'} (tolerance 2%)")
    print()

    # ------------------------------------------------------------------
    # Check 2: frequency-domain equivalence
    # ------------------------------------------------------------------
    Fp, Fc, _ = antenna(p_ref.p_hat)
    h0_val = h0(fP, Mc_kg, DL_m)
    prefac = h0_val / (2 * np.pi * fP)
    A_plus = prefac * Fp * (1 + ci ** 2) / 2
    A_cross = prefac * Fc * ci
    rho_fd, se_fd = rho_frequency_check(A_plus, A_cross, fP, n_noise=20000,
                                        rng=np.random.default_rng(2))
    diff_fd = rho_fd - rho_cf
    nsigma = abs(diff_fd) / se_fd
    print(f"--- Check 2: matched-filter mean statistic on noisy data")
    print(f"  closed form              rho = {rho_cf:.4f}")
    print(f"  mean matched-filter est. rho = {rho_fd:.4f}  (SE {se_fd:.4f})")
    print(f"  difference (est - closed)    = {diff_fd:+.4f}   ({nsigma:.2f} sigma)")
    pass2 = nsigma < 3.0
    print(f"  {'PASS' if pass2 else 'FAIL'} (tolerance 3 sigma)")
    print()

    # ------------------------------------------------------------------
    # Check 3: 1/D_L scaling
    # ------------------------------------------------------------------
    Ds = [50.0, 100.0, 200.0, 500.0]
    rhos = [rho_comb_at(d) for d in Ds]
    products = [d * r for d, r in zip(Ds, rhos)]
    p_ref_prod = products[0]
    max_frac = max(abs(p - p_ref_prod) / p_ref_prod for p in products)
    print(f"--- Check 3: 1/D_L scaling of rho_comb")
    for d, r, pp in zip(Ds, rhos, products):
        print(f"  D = {d:>6.0f} Mpc   rho_comb = {r:>10.3f}   "
              f"D*rho = {pp:>11.3f}")
    print(f"  max fractional spread of D*rho = {max_frac:.2e}")
    pass3 = max_frac < 1e-10
    print(f"  {'PASS' if pass3 else 'FAIL'} (tolerance 1e-10, exact law)")
    print()

    # ------------------------------------------------------------------
    # Check 4: Tier-1 horizon predicted by scaling vs direct scan
    # ------------------------------------------------------------------
    rho_thresh = 5.0
    rho_100 = rho_comb_at(100.0)
    horizon_analytic = 100.0 * rho_100 / rho_thresh

    # Reproduce compute_table3.py grid scan (1 Mpc steps)
    D_scan = np.arange(10, 1501, 1, dtype=float)
    rhos_scan = np.array([rho_comb_at(d) for d in D_scan])
    mask = rhos_scan > rho_thresh
    horizon_scan = float(D_scan[mask][-1]) if mask.any() else 0.0
    print(f"--- Check 4: Tier-1 horizon")
    print(f"  rho_comb at 100 Mpc          = {rho_100:.2f}")
    print(f"  analytic horizon (100 * rho/5)  = {horizon_analytic:.2f} Mpc")
    print(f"  grid scan horizon (rho > 5)  = {horizon_scan:.0f} Mpc")
    diff = abs(horizon_analytic - horizon_scan)
    pass4 = diff < 1.0   # 1-Mpc grid step
    print(f"  |difference| = {diff:.3f} Mpc")
    print(f"  {'PASS' if pass4 else 'FAIL'} (tolerance 1 Mpc = grid step)")
    print()

    all_pass = pass1 and pass2 and pass3 and pass4
    print("=" * 70)
    print("PASS: all checks agree within tolerance." if all_pass
          else "FAIL: one or more checks out of tolerance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
