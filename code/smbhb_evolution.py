"""
smbhb_evolution.py

Track the post-Newtonian evolution of a supermassive black hole binary (SMBHB)
from pulsar term to Earth term in pulsar timing arrays.

Based on:
  Mingarelli et al. (2012), PRL 109, 081104
  Mingarelli et al., "Pulsar Timing Arrays: The Emerging GW Landscape"

Physics:
  - TaylorT1 frequency/phase evolution up to 2pN order
  - Spin-orbit coupling (1.5pN) and spin-spin (2pN)
  - Thomas precession from orbital plane precession
  - Earth and pulsar term timing residuals
  - Antenna pattern functions for individual pulsars

Author: Generated from Mingarelli et al. papers
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Tuple

# ======================================================================
# Physical constants (SI)
# ======================================================================
G_SI = 6.67430e-11        # m^3 kg^-1 s^-2
C_SI = 2.99792458e8       # m/s
M_SUN = 1.98892e30        # kg
PC = 3.08567758e16        # m
MPC = 1e6 * PC            # m
YR = 365.25 * 86400       # s


# ======================================================================
# Pulsar dataclass
# ======================================================================
@dataclass
class Pulsar:
    """A pulsar in the timing array.

    Parameters
    ----------
    name : str
        Pulsar designation.
    theta : float
        Colatitude on the sky [rad].
    phi : float
        Azimuthal angle [rad].
    dist_kpc : float
        Distance from Earth [kpc].
    """
    name: str
    theta: float
    phi: float
    dist_kpc: float

    @property
    def p_hat(self) -> np.ndarray:
        """Unit vector toward the pulsar."""
        return np.array([
            np.sin(self.theta) * np.cos(self.phi),
            np.sin(self.theta) * np.sin(self.phi),
            np.cos(self.theta),
        ])


# ======================================================================
# Main class
# ======================================================================
class SMBHBEvolution:
    """
    Post-Newtonian evolution of a SMBHB system for PTA observations.

    Implements the "PTA Time Machine" (Mingarelli et al. 2012): the
    pulsar term samples the binary at a retarded time ~L/c years before
    the Earth term, enabling measurement of orbital evolution over
    millennial timescales.

    The TaylorT1 approximant is used: coupled ODEs for f(t) and Phi(t)
    are integrated numerically with pN corrections to the GW energy flux.

    Parameters
    ----------
    m1, m2 : float
        Component masses [M_sun].  Convention: m1 >= m2.
    chi1, chi2 : float
        Dimensionless spin magnitudes, 0 <= chi <= 1.
    kappa1, kappa2 : float
        Spin-orbit misalignment angles [rad].  0 = aligned with L.
    f_gw_earth : float
        GW frequency at the Earth epoch [Hz].
    D_L : float
        Luminosity distance [Mpc].
    iota : float
        Orbital inclination [rad].
    psi : float
        GW polarization angle [rad].
    theta_s, phi_s : float
        Source sky position (colatitude, azimuth) [rad].
    """

    def __init__(
        self,
        m1: float,
        m2: float,
        chi1: float = 0.0,
        chi2: float = 0.0,
        kappa1: float = 0.0,
        kappa2: float = 0.0,
        f_gw_earth: float = 1e-7,
        D_L: float = 100.0,
        iota: float = np.pi / 4,
        psi: float = 0.0,
        theta_s: float = np.pi / 3,
        phi_s: float = np.pi / 4,
    ):
        # Store user-facing parameters
        self.m1_msun = m1
        self.m2_msun = m2
        self.chi1 = chi1
        self.chi2 = chi2
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.f_gw_earth = f_gw_earth
        self.D_L_mpc = D_L
        self.iota = iota
        self.psi = psi
        self.theta_s = theta_s
        self.phi_s = phi_s

        # Derived SI quantities
        self.m1 = m1 * M_SUN
        self.m2 = m2 * M_SUN
        self.M = self.m1 + self.m2
        self.mu = self.m1 * self.m2 / self.M
        self.eta = self.mu / self.M
        self.Mc = self.M * self.eta ** 0.6
        self.D_L = D_L * MPC

        # Geometrized mass [seconds]
        self.M_s = G_SI * self.M / C_SI ** 3
        self.Mc_s = G_SI * self.Mc / C_SI ** 3

        # GW propagation direction (toward the observer)
        self.Omega_hat = -np.array([
            np.sin(theta_s) * np.cos(phi_s),
            np.sin(theta_s) * np.sin(phi_s),
            np.cos(theta_s),
        ])

        self._compute_pn_coefficients()

    # ------------------------------------------------------------------
    # Post-Newtonian coefficients
    # ------------------------------------------------------------------
    def _compute_pn_coefficients(self):
        """
        Compute separate energy and flux pN coefficients up to 2pN.

        True TaylorT1 keeps the ratio F_hat/E_prime_hat as a ratio of
        series (not expanded), which is important at 2pN and above.
        """
        eta = self.eta
        m1, m2, M = self.m1, self.m2, self.M
        chi1, chi2 = self.chi1, self.chi2
        ck1 = np.cos(self.kappa1)
        ck2 = np.cos(self.kappa2)

        # ================================================================
        # GW energy FLUX coefficients: F_hat(v) = 1 + F2 v^2 + F3 v^3 + F4 v^4
        # (Blanchet 2014, Living Rev. Rel.)
        # ================================================================
        self.F2 = -(1247.0 / 336 + 35.0 * eta / 12)

        # 1.5pN: tail (4pi) + spin-orbit (-beta)
        self.beta_so = (1.0 / 12) * (
            (113 * (m1 / M) ** 2 + 75 * eta) * chi1 * ck1
            + (113 * (m2 / M) ** 2 + 75 * eta) * chi2 * ck2
        )
        self.F3 = 4 * np.pi - self.beta_so

        # 2pN flux (mass terms)
        cos12 = ck1 * ck2 + np.sin(self.kappa1) * np.sin(self.kappa2)
        self.sigma_ss = (eta / 48) * (
            -247 * chi1 * chi2 * cos12
            + 721 * chi1 * ck1 * chi2 * ck2
        )
        self.F4 = -(44711.0 / 9072 + 9271.0 * eta / 504
                     + 65.0 * eta ** 2 / 18) + self.sigma_ss

        # ================================================================
        # Binding ENERGY derivative coefficients:
        #   E(v) = -(mu/2) v^2 [1 + A2 v^2 + A3 v^3 + A4 v^4]
        #   dE/dv / (-mu v) = 1 + 2 A2 v^2 + (5/2) A3 v^3 + 3 A4 v^4
        # (Blanchet 2014; Kidder 1995 for spin terms)
        # ================================================================
        A2 = -(3.0 / 4 + eta / 12)           # 1pN energy
        A3_so = 0.0                            # 1.5pN spin-orbit energy
        if chi1 > 0 or chi2 > 0:
            # Spin-orbit contribution to energy at 1.5pN
            # From Kidder (1995): E_SO = (14 S_L/3 + 2 delta Delta_L) / M^2
            delta = (m1 - m2) / M
            chi_s = (chi1 * ck1 + chi2 * ck2) / 2
            chi_a = (chi1 * ck1 - chi2 * ck2) / 2
            A3_so = (14.0 / 3 * chi_s + 2.0 * delta * chi_a) / 3
        A4 = -(27.0 / 8 - 19.0 * eta / 8 + eta ** 2 / 24)  # 2pN energy (mass)
        # Spin-spin contribution to energy at 2pN
        A4_ss = 0.0
        if chi1 > 0 and chi2 > 0:
            A4_ss = -(1.0 / 48) * (
                -247 * chi1 * chi2 * cos12
                + 721 * chi1 * ck1 * chi2 * ck2
            )
            A4 += A4_ss

        self.Ep2 = 2 * A2                     # coeff of v^2 in dE/dv norm.
        self.Ep3 = 2.5 * A3_so                # coeff of v^3
        self.Ep4 = 3 * A4                     # coeff of v^4

        # Store combined TaylorT4 coefficients for reference
        self.c1pn = -(743.0 / 336 + 11.0 * eta / 4)
        self.c1p5n = 4 * np.pi - self.beta_so
        self.c2pn = (
            34103.0 / 18144 + 13661.0 * eta / 2016
            + 59.0 * eta ** 2 / 18 - self.sigma_ss
        )

        # ---- Thomas precession ----
        # Simple-precession approximation (Apostolatos et al. 1994, Ref [49]).
        # Valid when m1 = m2 or when one spin dominates.
        # Precession rate of L about J:
        #   Omega_p = (2 + 3 m2/(2 m1)) * (J / (2 r^3))
        # In terms of v = (pi M f)^{1/3}:
        #   Omega_p = (2 + 3q/2) * eta * v^5 / (2 M_s)
        # where q = m2/m1 (convention m1 >= m2).
        q = m2 / m1
        self.prec_prefactor = 2 + 1.5 * q

        # Spin angular momenta (geometrized: S_i = chi_i * G m_i^2 / c)
        S1 = chi1 * G_SI * m1 ** 2 / C_SI
        S2 = chi2 * G_SI * m2 ** 2 / C_SI

        # Precession cone half-angle zeta_L
        # From Apostolatos et al.: tan(zeta_L) ≈ S_perp / L
        S_perp = np.hypot(
            S1 * np.sin(self.kappa1),
            S2 * np.sin(self.kappa2),
        )
        L0 = self.mu * (G_SI * self.M) ** (2.0 / 3) / (
            np.pi * self.f_gw_earth
        ) ** (1.0 / 3)
        self.zeta_L = np.arctan2(S_perp, L0) if L0 > 0 else 0.0

        # Spin parameter Theta (Mingarelli et al. 2012, PRL 109, 081104,
        # defined below Eq. 3; max value 7.8 for chi=1 equal mass)
        self.Theta_param = (1.0 / 12) * sum(
            (113 * (mi / M) ** 2 + 75 * eta) * chi_i * abs(ck_i)
            for mi, chi_i, ck_i in [(m1, chi1, ck1), (m2, chi2, ck2)]
        )

    # ------------------------------------------------------------------
    # TaylorT1 flux factor
    # ------------------------------------------------------------------
    def _correction_factor(self, v: float, pn_order: int) -> float:
        """
        True TaylorT1 correction: ratio F_hat(v) / E_prime_hat(v).

        Keeps the energy derivative and flux as separate pN series
        and evaluates their ratio numerically (no re-expansion).
        This is the defining property of the TaylorT1 approximant
        (Damour, Iyer & Sathyaprakash 2001, PRD 63, 044023).

        df/dt = (96/5) pi^{8/3} Mc^{5/3} f^{11/3} * F_hat / E_prime_hat

        Flux coefficients F2, F3, F4: Blanchet 2006, Living Rev. Rel.
        9, 4, Eq. 227 (updated in Blanchet 2014, Eq. 314).
        Energy derivative coefficients Ep2-Ep4: Blanchet 2006 Eq. 193;
        spin-orbit terms from Kidder 1995, PRD 52, 821, Eq. 2.9.

        Parameters
        ----------
        v : float or array
            Post-Newtonian velocity parameter (pi M_s f)^{1/3}.
        pn_order : int
            0 = Newtonian, 2 = include 1pN, 3 = include 1.5pN,
            4 = include 2pN.
        """
        x = v ** 2
        one = np.ones_like(v, dtype=float) if hasattr(v, '__len__') else 1.0

        # Normalized flux: F_hat = 1 + F2 x + F3 x^{3/2} + F4 x^2
        F_hat = one.copy() if hasattr(one, 'copy') else 1.0
        if pn_order >= 2:
            F_hat = F_hat + self.F2 * x
        if pn_order >= 3:
            F_hat = F_hat + self.F3 * x ** 1.5
        if pn_order >= 4:
            F_hat = F_hat + self.F4 * x ** 2

        # Normalized energy derivative: E_prime_hat = 1 + Ep2 x + Ep3 x^{3/2} + Ep4 x^2
        Ep_hat = one.copy() if hasattr(one, 'copy') else 1.0
        if pn_order >= 2:
            Ep_hat = Ep_hat + self.Ep2 * x
        if pn_order >= 3:
            Ep_hat = Ep_hat + self.Ep3 * x ** 1.5
        if pn_order >= 4:
            Ep_hat = Ep_hat + self.Ep4 * x ** 2

        return F_hat / Ep_hat

    # ------------------------------------------------------------------
    # Precession rate
    # ------------------------------------------------------------------
    def _precession_rate(self, f: float) -> float:
        """
        Orbital precession angular frequency Omega_p [rad/s].

        Simple-precession approximation (Apostolatos et al. 1994):
            Omega_p = (2 + 3q/2) * eta * M_s^{2/3} * (pi f)^{5/3} / 2
        """
        return (
            self.prec_prefactor
            * self.eta
            * self.M_s ** (2.0 / 3)
            * (np.pi * f) ** (5.0 / 3)
            / 2
        )

    # ------------------------------------------------------------------
    # Core evolution (backward integration)
    # ------------------------------------------------------------------
    def evolve(
        self,
        t_span_yr: float,
        n_points: int = 10000,
        pn_order: int = 4,
    ) -> dict:
        """
        Evolve the binary backward from the Earth epoch using TaylorT1.

        TaylorT1 integrates the coupled ODEs df/dt and dPhi/dt
        numerically, keeping the ratio F(v)/E'(v) unexpanded
        (Damour, Iyer & Sathyaprakash 2001, PRD 63, 044023;
        Buonanno, Chen & Vallisneri 2003, PRD 67, 104025, Sec. II).

        Integration proceeds from t = 0 (Earth) into the past
        (t < 0), covering the light-travel time to a pulsar.

        Parameters
        ----------
        t_span_yr : float
            How far into the past to integrate [yr].
        n_points : int
            Number of output time samples.
        pn_order : int
            0 = Newtonian, 2 = 1pN, 3 = 1.5pN, 4 = 2pN.

        Returns
        -------
        dict
            t_yr : array, time (negative = past)
            f_gw : array, GW frequency [Hz]
            Phi  : array, accumulated GW phase [rad]
            phi_T: array, Thomas precession phase [rad]
            v    : array, pN velocity parameter
        """
        T = t_span_yr * YR

        def rhs(tau, y):
            """RHS for backward integration.  tau = -t >= 0."""
            f, Phi, phi_T = y
            if f < 1e-15:
                return [0.0, 0.0, 0.0]

            v = (np.pi * self.M_s * f) ** (1.0 / 3)

            # Frequency evolution (forward)
            dfdt = (
                (96.0 / 5)
                * np.pi ** (8.0 / 3)
                * self.Mc_s ** (5.0 / 3)
                * f ** (11.0 / 3)
                * self._correction_factor(v, pn_order)
            )

            # GW phase rate
            dPhidt = 2 * np.pi * f

            # Thomas precession phase rate
            if self.zeta_L > 1e-10:
                Omega_p = self._precession_rate(f)
                dphi_T_dt = Omega_p * (1 - np.cos(self.zeta_L))
            else:
                dphi_T_dt = 0.0

            # Backward: d/dtau = -d/dt
            return [-dfdt, -dPhidt, -dphi_T_dt]

        sol = solve_ivp(
            rhs,
            [0, T],
            [self.f_gw_earth, 0.0, 0.0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-15,
            max_step=T / max(n_points, 1000),
            dense_output=True,
        )

        tau = np.linspace(0, T, n_points)
        y = sol.sol(tau)

        return {
            "t_yr": -tau / YR,
            "f_gw": y[0],
            "Phi": y[1],
            "phi_T": y[2],
            "v": (np.pi * self.M_s * y[0]) ** (1.0 / 3),
        }

    # ------------------------------------------------------------------
    # TaylorT2 analytic phase decomposition
    # ------------------------------------------------------------------
    def pn_decomposition(self, t_span_yr: float, n_points: int = 10000) -> dict:
        """
        Decompose the GW phase into individual pN contributions using
        the TaylorT2 analytic phase formula (Blanchet 2006, Living Rev.
        Rel., Eqs. 232/234), as referenced in Mingarelli et al. (2012).

        The decomposition combines two complementary calculations:

        1. **Total** GW cycles from TaylorT2 phase formula:
              N_total = (1/(32 pi eta)) * Delta[v^{-5} (1 + phi_2 v^2 + ...)]
           where v_P is found from the TaylorT2 time equation (Eq. 232).

        2. **Individual pN corrections** from frequency-domain cycle
           counting N = integral(f/fdot, df), which yields TaylorF2
           coefficients with prefactor 3/(256 pi eta):
              Delta N_n = (3/(256 pi eta)) * psi_n * Delta[v^{2n-5}]

        3. **Newtonian** = Total - sum(all corrections).

        4. **Spin-orbit** = difference of TaylorT2 totals with/without SO.

        Also runs TaylorT1 numerical evolution (full 2pN) to provide
        time-domain arrays for plotting.

        Parameters
        ----------
        t_span_yr : float
            Light-travel baseline (pulsar to Earth) [yr].
        n_points : int
            Number of output time samples for numerical evolution.

        Returns
        -------
        dict
            t_yr  : time array (from numerical evolution)
            Phi   : dict of cumulative phase at each pN order (numerical)
            dPhi  : dict of incremental phase contributions (numerical)
            f     : dict of frequency at each cumulative order (numerical)
            df    : dict of incremental frequency shifts (numerical)
            phi_T : Thomas precession phase array (numerical)
            cycles: dict of GW cycle counts (ANALYTIC)
        """
        eta = self.eta

        # ==============================================================
        # TaylorT2 coefficients (Blanchet 2006, Eqs. 232 and 234)
        # ==============================================================

        # --- Time: t(v) = t_c - (5M_s)/(256 eta) v^{-8} [1 + sum] ---
        tau_2 = 743.0 / 252 + 11.0 * eta / 3                   # 1pN
        tau_3_mass = -(32.0 / 5) * np.pi                        # 1.5pN (tail)
        tau_3_SO = (self.beta_so * 48.0) / 5.0                  # 1.5pN (SO)
        tau_4 = (3058673.0 / 508032 + 5429.0 * eta / 504
                 + 617.0 * eta ** 2 / 72)                       # 2pN (mass)
        tau_4_SS = -self.sigma_ss * 40.0 / eta if eta > 0 else 0.0

        tau_3 = tau_3_mass + tau_3_SO
        tau_4_full = tau_4 + tau_4_SS

        # --- Phase: Phi(v) = Phi_c - (1/(32 eta)) v^{-5} [1 + sum] ---
        #     (GW phase = 2 * orbital; Blanchet Eq. 234 is orbital,
        #      so prefactor is 1/(32 eta) not 1/(16 eta))
        phi_2 = 3715.0 / 1008 + 55.0 * eta / 12                # 1pN
        phi_3_mass = -10.0 * np.pi                              # 1.5pN (tail)
        phi_3_SO = (10.0 / 3) * self.beta_so                    # 1.5pN (SO)
        phi_4 = (15293365.0 / 508032 + 27145.0 * eta / 504
                 + 3085.0 * eta ** 2 / 72)                      # 2pN (mass)
        phi_4_SS = -(10.0 / eta) * self.sigma_ss if eta > 0 else 0.0

        phi_3 = phi_3_mass + phi_3_SO
        phi_4_full = phi_4 + phi_4_SS

        # ==============================================================
        # TaylorF2 coefficients for frequency-domain cycle counting
        #   N = integral(f / fdot, df) decomposes as:
        #   N_n = (3/(256 pi eta)) * psi_n * Delta[v^{2n-5}]
        #
        # At 1pN and 1.5pN, psi_n = (8 phi_n - 5 tau_n) / 3.
        # At 2pN, psi_4 = phi_4 (standard TaylorF2 result; the naive
        # formula (8 phi_4 - 5 tau_4)/3 overcounts due to cross-terms
        # in the SPA integral that cancel at this order).
        # ==============================================================
        psi_2 = (8 * phi_2 - 5 * tau_2) / 3.0       # = (20/9)(743/336 + 11 eta/4)
        psi_3_mass = (8 * phi_3_mass - 5 * tau_3_mass) / 3.0  # = -16 pi
        psi_4 = phi_4                                 # standard TaylorF2 2pN coeff

        # ==============================================================
        # Earth-epoch velocity
        # ==============================================================
        v_E = (np.pi * self.M_s * self.f_gw_earth) ** (1.0 / 3)
        T_baseline = t_span_yr * YR  # seconds
        time_prefac = 5.0 * self.M_s / (256.0 * eta)
        phase_prefac = 1.0 / (32.0 * np.pi * eta)  # GW cycles

        from scipy.optimize import brentq

        # ==============================================================
        # Solve for v_P from full TaylorT2 time equation
        # ==============================================================
        def time_from_v(v, t2, t3, t4):
            """TaylorT2 time as function of v."""
            return time_prefac * v ** (-8) * (
                1.0 + t2 * v ** 2 + t3 * v ** 3 + t4 * v ** 4
            )

        def solve_vP(t2, t3, t4):
            """Find v_P such that t(v_P) - t(v_E) = T_baseline."""
            target = T_baseline + time_from_v(v_E, t2, t3, t4)
            def residual(v):
                return time_from_v(v, t2, t3, t4) - target
            v_lo, v_hi = v_E * 0.001, v_E * 0.9999
            if residual(v_lo) * residual(v_hi) > 0:
                v_est = (v_E ** (-8) + 256.0 * eta * T_baseline
                         / (5.0 * self.M_s)) ** (-1.0 / 8)
                v_lo = v_est * 0.5
            return brentq(residual, v_lo, v_hi, xtol=1e-15, rtol=1e-14)

        # v_P from full 2pN TaylorT2 time equation (including SO & SS)
        v_P = solve_vP(tau_2, tau_3, tau_4_full)

        # Also solve v_P without SO (for SO decomposition via differencing)
        v_P_noSO = solve_vP(tau_2, tau_3_mass, tau_4_full)

        # ==============================================================
        # Total GW cycles from TaylorT2 phase (full 2pN)
        # ==============================================================
        def phi_T2(v, p2, p3, p4):
            """TaylorT2 phase function (returns GW cycles)."""
            return phase_prefac * v ** (-5) * (
                1.0 + p2 * v ** 2 + p3 * v ** 3 + p4 * v ** 4
            )

        N_total = (phi_T2(v_P, phi_2, phi_3, phi_4_full)
                   - phi_T2(v_E, phi_2, phi_3, phi_4_full))

        # Total without SO (for SO by differencing)
        N_total_noSO = (phi_T2(v_P_noSO, phi_2, phi_3_mass, phi_4_full)
                        - phi_T2(v_E, phi_2, phi_3_mass, phi_4_full))

        # ==============================================================
        # Individual corrections: frequency-domain cycle counting
        #   N_n = (3 / (256 pi eta)) * psi_n * Delta[v^{2n-5}]
        # with v_P from full TaylorT2 time equation.
        # ==============================================================
        fd_pf = 3.0 / (256.0 * np.pi * eta)

        N_1pN = fd_pf * psi_2 * (v_P ** (-3) - v_E ** (-3))
        N_15pN = fd_pf * psi_3_mass * (v_P ** (-2) - v_E ** (-2))
        N_2pN = fd_pf * psi_4 * (v_P ** (-1) - v_E ** (-1))

        # SO contribution: difference of totals with/without SO
        N_SO = N_total - N_total_noSO

        # Newtonian = Total minus all corrections
        N_Newt = N_total - N_1pN - N_15pN - N_SO - N_2pN

        # Thomas precession (from numerical evolution)
        evol_full = self.evolve(t_span_yr, n_points, pn_order=4)
        N_Thomas = evol_full["phi_T"][-1] / (2 * np.pi)

        cycles = {
            "Newtonian": N_Newt,
            "1pN": N_1pN,
            "1.5pN": N_15pN,
            "SO": N_SO,
            "2pN": N_2pN,
            "Thomas": N_Thomas,
            "Total": N_total + N_Thomas,
        }

        # ==============================================================
        # Store useful derived quantities
        # ==============================================================
        f_P = v_P ** 3 / (np.pi * self.M_s)
        cycles["v_E"] = v_E
        cycles["v_P"] = v_P
        cycles["f_P_nHz"] = f_P * 1e9
        cycles["f_E_nHz"] = self.f_gw_earth * 1e9
        cycles["delta_f_nHz"] = (self.f_gw_earth - f_P) * 1e9

        # ==============================================================
        # Numerical TaylorT1 evolution for time-domain plots
        # ==============================================================
        orders = {"Newtonian": 0, "1pN": 2, "1.5pN": 3, "2pN": 4}
        evols = {
            name: self.evolve(t_span_yr, n_points, pn_order=order)
            for name, order in orders.items()
        }

        t = evols["Newtonian"]["t_yr"]
        Phi = {k: evols[k]["Phi"] for k in orders}
        f = {k: evols[k]["f_gw"] for k in orders}

        dPhi = {
            "Newtonian": Phi["Newtonian"],
            "1pN": Phi["1pN"] - Phi["Newtonian"],
            "1.5pN": Phi["1.5pN"] - Phi["1pN"],
            "2pN": Phi["2pN"] - Phi["1.5pN"],
        }
        df = {
            "Newtonian": f["Newtonian"],
            "1pN": f["1pN"] - f["Newtonian"],
            "1.5pN": f["1.5pN"] - f["1pN"],
            "2pN": f["2pN"] - f["1.5pN"],
        }

        phi_T = evols["2pN"]["phi_T"]

        return {
            "t_yr": t,
            "Phi": Phi,
            "dPhi": dPhi,
            "f": f,
            "df": df,
            "phi_T": phi_T,
            "cycles": cycles,
        }

    # ------------------------------------------------------------------
    # Antenna pattern functions
    # ------------------------------------------------------------------
    def antenna_pattern(self, pulsar: Pulsar) -> Tuple[float, float]:
        """
        Compute F+, Fx antenna pattern functions.

        Following the convention of Eq. 63 in the review
        (Mingarelli et al.), with polarization angle psi
        absorbed into the polarization tensors.

        Parameters
        ----------
        pulsar : Pulsar

        Returns
        -------
        (F_plus, F_cross) : tuple of float
        """
        # Principal axes of the GW frame
        m_hat = np.array([np.sin(self.phi_s), -np.cos(self.phi_s), 0.0])
        n_hat = np.array([
            -np.cos(self.theta_s) * np.cos(self.phi_s),
            -np.cos(self.theta_s) * np.sin(self.phi_s),
            np.sin(self.theta_s),
        ])

        # Polarization tensors rotated by psi
        c2p = np.cos(2 * self.psi)
        s2p = np.sin(2 * self.psi)
        mm = np.outer(m_hat, m_hat)
        nn = np.outer(n_hat, n_hat)
        mn = np.outer(m_hat, n_hat) + np.outer(n_hat, m_hat)

        e_plus = c2p * (mm - nn) + s2p * mn
        e_cross = -s2p * (mm - nn) + c2p * mn

        p = pulsar.p_hat
        denom = 2 * (1 + np.dot(self.Omega_hat, p))
        if abs(denom) < 1e-15:
            return 0.0, 0.0

        Fp = np.einsum("i,ij,j", p, e_plus, p) / denom
        Fc = np.einsum("i,ij,j", p, e_cross, p) / denom
        return Fp, Fc

    # ------------------------------------------------------------------
    # Light-travel time
    # ------------------------------------------------------------------
    def light_travel_time(self, pulsar: Pulsar) -> float:
        """
        Light-travel delay tau = L_p (1 + Omega_hat . p_hat) / c  [yr].

        This is the time separation between the Earth and pulsar terms
        (Paper Eq. 1; Anholm et al. 2009, PRD 79, 084030, Eq. 1;
        Mingarelli et al. 2012, PRL 109, 081104, Eq. 1).
        """
        L = pulsar.dist_kpc * 1e3 * PC
        geom = 1 + np.dot(self.Omega_hat, pulsar.p_hat)
        return L * geom / C_SI / YR

    # ------------------------------------------------------------------
    # GW strain
    # ------------------------------------------------------------------
    def _strain(self, f, Phi):
        """
        Compute h+(t), hx(t) at Newtonian amplitude order.

        Amplitude: h_0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L)
        (Paper Eq. 2; Maggiore 2007 Eq. 4.25.)

        Polarizations (Eq. 64-67 of Mingarelli et al. review):
          h+ = -h_0 (1 + ci^2) cos(2 Phi)
          hx = -h_0 * 2 ci sin(2 Phi)
        with polarization angle psi absorbed into the waveform.
        """
        ci = np.cos(self.iota)
        A = (
            4
            * (np.pi * f) ** (2.0 / 3)
            * (G_SI * self.Mc) ** (5.0 / 3)
            / (C_SI ** 4 * self.D_L)
        )
        hp = -A * (1 + ci ** 2) * np.cos(2 * Phi)
        hc = -A * 2 * ci * np.sin(2 * Phi)
        return hp, hc

    # ------------------------------------------------------------------
    # Timing residuals
    # ------------------------------------------------------------------
    def timing_residual(
        self,
        pulsar: Pulsar,
        T_obs_yr: float = 10.0,
        n_obs: int = 5000,
    ) -> dict:
        """
        Compute timing residuals for a single pulsar.

        s(t) = F+ [h+(t_E) - h+(t_p)] + Fx [hx(t_E) - hx(t_p)]

        where t_p = t_E - tau, and tau = L(1 + Omega . p)/c.
        The timing residual r(t) is the integral of s(t).

        Parameters
        ----------
        pulsar : Pulsar
        T_obs_yr : float
            Observation time span [yr].
        n_obs : int
            Number of output samples.

        Returns
        -------
        dict with keys:
            t_yr, residual_ns, redshift,
            earth_term, pulsar_term,
            f_earth_nHz, f_pulsar_nHz, delta_f_nHz,
            tau_yr, Fp, Fc
        """
        tau_yr = self.light_travel_time(pulsar)
        span = T_obs_yr + abs(tau_yr) + 500  # buffer
        evol = self.evolve(span, n_points=80000)

        # Interpolators over the full backward evolution
        f_interp = interp1d(
            evol["t_yr"], evol["f_gw"], kind="cubic", fill_value="extrapolate"
        )
        Phi_interp = interp1d(
            evol["t_yr"], evol["Phi"], kind="cubic", fill_value="extrapolate"
        )

        Fp, Fc = self.antenna_pattern(pulsar)

        t_obs = np.linspace(0, T_obs_yr, n_obs)

        # Earth term: near t = 0
        f_E = f_interp(-t_obs)
        Phi_E = Phi_interp(-t_obs)
        hp_E, hc_E = self._strain(f_E, Phi_E)

        # Pulsar term: further in the past
        f_P = f_interp(-(t_obs + tau_yr))
        Phi_P = Phi_interp(-(t_obs + tau_yr))
        hp_P, hc_P = self._strain(f_P, Phi_P)

        # Redshift (frequency shift)
        earth = Fp * hp_E + Fc * hc_E
        puls = Fp * hp_P + Fc * hc_P
        redshift = earth - puls

        # Integrate to get timing residual
        dt = (t_obs[1] - t_obs[0]) * YR
        residual = np.cumsum(redshift) * dt

        delta_f = np.mean(f_E) - np.mean(f_P)

        return {
            "t_yr": t_obs,
            "residual_s": residual,
            "residual_ns": residual * 1e9,
            "redshift": redshift,
            "earth_term": earth,
            "pulsar_term": puls,
            "f_earth_nHz": f_E * 1e9,
            "f_pulsar_nHz": f_P * 1e9,
            "delta_f_nHz": delta_f * 1e9,
            "tau_yr": tau_yr,
            "Fp": Fp,
            "Fc": Fc,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def orbital_velocity(self) -> float:
        """Dimensionless orbital velocity v/c at the Earth epoch."""
        return (np.pi * self.M_s * self.f_gw_earth) ** (1.0 / 3)

    @property
    def orbital_timescale_yr(self) -> float:
        """Characteristic timescale f / fdot [yr]."""
        f = self.f_gw_earth
        v = self.orbital_velocity
        dfdt = (
            (96.0 / 5)
            * np.pi ** (8.0 / 3)
            * self.Mc_s ** (5.0 / 3)
            * f ** (11.0 / 3)
            * self._correction_factor(v, 4)
        )
        return f / dfdt / YR

    @property
    def gw_amplitude(self) -> float:
        """Characteristic GW strain h0 at the Earth epoch.

        .. deprecated::
            Uses prefactor 2 (face-on, single polarization).  The paper
            convention is h0 = 4 (G Mc)^{5/3} (pi f)^{2/3} / (c^4 D_L),
            implemented in phase_matching.h0().  Use that instead for any
            calculation that must match the manuscript.
        """
        import warnings
        warnings.warn(
            "gw_amplitude uses prefactor 2; paper Eq. (2) uses 4. "
            "Use phase_matching.h0() for manuscript-consistent values.",
            DeprecationWarning,
            stacklevel=2,
        )
        f = self.f_gw_earth
        return (
            2
            * (np.pi * f) ** (2.0 / 3)
            * (G_SI * self.Mc) ** (5.0 / 3)
            / (C_SI ** 4 * self.D_L)
        )

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------
    def summary(self):
        """Print a summary of binary parameters and pN coefficients."""
        v = self.orbital_velocity
        sep = "=" * 58
        print(sep)
        print("  SMBHB System Parameters")
        print(sep)
        print(f"  m1          = {self.m1_msun:.2e} Msun")
        print(f"  m2          = {self.m2_msun:.2e} Msun")
        print(f"  Mc          = {self.Mc / M_SUN:.2e} Msun")
        print(f"  eta         = {self.eta:.4f}")
        print(f"  chi1        = {self.chi1:.2f}   kappa1 = {np.degrees(self.kappa1):.1f} deg")
        print(f"  chi2        = {self.chi2:.2f}   kappa2 = {np.degrees(self.kappa2):.1f} deg")
        print(f"  f_GW(Earth) = {self.f_gw_earth * 1e9:.1f} nHz")
        print(f"  D_L         = {self.D_L_mpc:.0f} Mpc")
        print(f"  h0          = {self.gw_amplitude:.2e}")
        print(f"  v/c         = {v:.4f}")
        print(f"  f/fdot      = {self.orbital_timescale_yr:.0f} yr")
        print(f"  iota        = {np.degrees(self.iota):.1f} deg")
        print(f"  zeta_L      = {np.degrees(self.zeta_L):.2f} deg")
        print("-" * 58)
        print("  pN coefficients (TaylorT1 flux):")
        print(f"    c_1pN     = {self.c1pn:.4f}")
        print(f"    beta_SO   = {self.beta_so:.4f}")
        print(f"    c_1.5pN   = {self.c1p5n:.4f}")
        print(f"    sigma_SS  = {self.sigma_ss:.6f}")
        print(f"    c_2pN     = {self.c2pn:.4f}")
        print(f"    Theta     = {self.Theta_param:.4f}")
        print(sep)
