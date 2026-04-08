import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import io
import os

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="GW Detector Sensitivity", layout="wide")

# =============================================================================
# Constants & Unicode tick formatter
# =============================================================================
c_SI = 299792458.0  # m/s
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
MSUN = 1.98892e30   # kg
PC_SI = 3.08568e16  # m
YR_S = 365.25 * 24 * 3600  # seconds per year

# Cosmological constants
H0_km_s_Mpc = 67.4
H0 = H0_km_s_Mpc * 1000.0 / 3.08567758e22

# Fiducial reservoir densities (Msun/Mpc^3)
# SMBH density updated to Liepold & Ma (2024) value
RHO_SMBH_FID = 1.8e6  # Liepold & Ma (2024), ApJL 971, L29
RHO_STELLAR_FID = 5.9e8
RHO_NSC_FID = 3.0e6

# Population parameters from Table I of Mingarelli (2026), arXiv:2601.18859
# SMBHB A_bench updated for L&M (2024) density: A = 1.6e-15 at f_ref = 1/yr
# IMBH-SMBH A_bench scaled by sqrt(1.8e6/4.2e5) = 2.07
POPULATIONS = {
    'SMBHB': {
        'reservoir': 'SMBH',
        'f_ref': 3.2e-8,
        'A_bench': 1.6e-15,  # Updated for L&M (2024)
        'f_min': 1e-9,
        'f_max': 4e-7,
        'f_merge_fid': 0.1,
        'epsilon_gw': 0.02,
        'color': '#0072B2',
    },
    'IMBH-SMBH': {
        'reservoir': 'SMBH',
        'f_ref': 3e-3,
        'A_bench': 2.3e-20,  # Scaled from 1.1e-20 by sqrt(1.8e6/4.2e5)
        'f_min': 1e-5,
        'f_max': 4e-2,
        'f_merge_fid': 0.05,
        'epsilon_gw': 0.05,
        'color': '#D55E00',
    },
    'EMRI': {
        'reservoir': 'NSC',
        'f_ref': 1e-2,
        'A_bench': 1.1e-20,
        'f_min': 1e-5,
        'f_max': 1e-2,
        'f_merge_fid': 0.1,
        'epsilon_gw': 0.05,
        'color': '#009E73',
    },
}

POPULATION_DISPLAY_NAMES = {
    'SMBHB': 'SMBHBs',
    'IMBH-SMBH': 'AGN-IMRI',
    'EMRI': 'EMRI',
}

_SUP = str.maketrans('-0123456789', '\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079')

def _log_fmt(x, pos):
    """Format log-scale tick labels using Unicode superscripts (no mathtext)."""
    if x <= 0:
        return ''
    exp = round(np.log10(x))
    if abs(x - 10**exp) / max(x, 1e-30) < 0.01:
        return f'10{str(exp).translate(_SUP)}'
    return ''

# =============================================================================
# Cached detector functions
# LISA/muAres: inlined physics from gwent/detector.py
# PTA: hasasia DeterSensitivityCurve (Hazboun, Romano & Smith 2019)
# =============================================================================

@st.cache_data
def get_muares_hc(T_obs_yr, acc_noise_level, f_min=1e-7, f_max=1e-1, nfreqs=2000):
    """muAres characteristic strain sensitivity with galactic confusion noise.

    Instrument noise inlined from gwent/detector.py (Sesana et al. 2021).
    acc_noise_level: acceleration noise in units of 1e-15 m/s^2/sqrt(Hz).
                     Default (Sesana et al.) is 1.0.
    Galactic WD confusion noise (Cornish & Robson 2017) added in the mHz band.
    """
    L = 3.95e11       # m (~2.6 AU)
    S_pos = 2.5e-21   # m^2/Hz  (50 pm/sqrt(Hz))^2
    S_acc_amp = (acc_noise_level * 1e-15) ** 2  # m^2/s^4/Hz
    # Sesana+2021 Sec 4: acceleration noise flat at 1e-15 m/s^2/sqrt(Hz)
    # "down to 1e-7 Hz" — no low-frequency reddening specified.

    f = np.logspace(np.log10(f_min), np.log10(f_max), nfreqs)
    f_star = c_SI / (2.0 * np.pi * L)

    S_acc = S_acc_amp  # flat (no LISA-style f_knee reddening)
    Sn = (20.0 / 3.0) * (1.0 / L**2) * (
        4.0 * S_acc / (2.0 * np.pi * f) ** 4 + S_pos
    ) * (1.0 + (f / f_star) ** 2)

    # Galactic binary confusion noise (Cornish & Robson 2017, gwent model 1)
    # WD binaries emit above ~10^-5 Hz (orbital periods < ~1 day).
    # The f^{-7/3} power law is only valid in the mHz band; applying it
    # at μHz would grossly overestimate the foreground.  We smoothly
    # suppress S_c below f_conf_min = 3×10^-5 Hz using an exponential
    # cutoff, consistent with the low-frequency edge of the DWD
    # population (see Sesana+2021 Fig 1).
    A_conf = 1.4e-44
    f_k = 0.0016 * T_obs_yr ** (-2.0 / 9.0)
    gamma_conf = 1100.0 * T_obs_yr ** (3.0 / 10.0)
    S_c = A_conf * f ** (-7.0 / 3.0) * (1.0 + np.tanh(gamma_conf * (f_k - f)))
    f_conf_min = 3e-5  # Hz — low-frequency edge of DWD foreground
    S_c = S_c * np.exp(-((f_conf_min / np.maximum(f, 1e-20)) ** 4))

    h_n_inst = np.sqrt(f * Sn)            # instrument only
    h_n_conf = np.sqrt(f * S_c)           # confusion only
    return f, h_n_inst, h_n_conf


@st.cache_data
def get_lisa_hc(T_obs_yr, f_min=1e-5, f_max=1.0, nfreqs=1000):
    """LISA characteristic strain sensitivity with galactic confusion noise.

    Instrument noise inlined from gwent/detector.py SpaceBased class with
    analytic transfer function (Robson, Cornish & Liu 2019).
    Galactic binary confusion noise from Cornish & Robson (2017), parameterised
    as a function of T_obs (gwent model 1).
    """
    L = 2.5e9              # m (2.5 Gm)
    A_acc = 3e-15          # m/s^2
    f_acc_break_low = 0.4e-3   # Hz
    f_acc_break_high = 8.0e-3  # Hz
    A_IFO = 1.5e-11        # m
    f_IFO_break = 2.0e-3   # Hz

    f = np.logspace(np.log10(f_min), np.log10(f_max), nfreqs)
    f_trans = c_SI / (2.0 * np.pi * L)

    # Acceleration noise
    P_acc = (A_acc**2
             * (1.0 + (f_acc_break_low / f) ** 2)
             * (1.0 + (f / f_acc_break_high) ** 4)
             / (2.0 * np.pi * f) ** 4)
    # Interferometer position noise
    P_IMS = A_IFO**2 * (1.0 + (f_IFO_break / f) ** 4)

    # Single-link power spectral density
    P_n = (P_IMS + 2.0 * (1.0 + np.cos(f / f_trans) ** 2) * P_acc) / L**2

    # Sky-averaged response (analytic approximation)
    R_f = (3.0 / 10.0) / (1.0 + 0.6 * (f / f_trans) ** 2)

    # Effective noise PSD (instrument only)
    S_n = P_n / R_f

    # Galactic binary confusion noise (Cornish & Robson 2017, gwent model 1)
    # Longer missions resolve more individual binaries, reducing confusion
    # Parameters calibrated for T_obs in years (gwent astropy convention)
    A_conf = 1.4e-44
    f_k = 0.0016 * T_obs_yr ** (-2.0 / 9.0)
    gamma_conf = 1100.0 * T_obs_yr ** (3.0 / 10.0)
    S_c = A_conf * f ** (-7.0 / 3.0) * (1.0 + np.tanh(gamma_conf * (f_k - f)))

    h_n_inst = np.sqrt(f * S_n)            # instrument only
    h_n_conf = np.sqrt(f * S_c)           # confusion only
    return f, h_n_inst, h_n_conf


# ── Precomputed PTA sensitivity curves ──
# NANOGrav 15yr: hasasia built-in NG11 DeterSensitivityCurve (real noise models).
# IPTA 2050: hasasia sim_pta (131 pulsars, 50yr, 200ns, 26/yr).
#   - WN-only: white noise only
#   - WN+RN: with per-pulsar red noise from NG12.5 chromatic noise analysis (arXiv:2511.22597)
# Precomputed with pta_cw_sensitivity.ipynb; see notebook for full derivation.
_PTA_CACHE = os.path.join(os.path.dirname(__file__), "pta_sensitivity_curves.npz")


@st.cache_data
def _load_pta_curves():
    """Load precomputed PTA sensitivity curves from disk.

    Returns dict with (f, hc, h0) tuples for each PTA.
    """
    d = np.load(_PTA_CACHE)
    return {
        'ng15': (d['ng15_f'], d['ng15_hc'], d['ng15_h0']),
        # Default IPTA 2050 = weekly cadence
        'ipta': (d['ipta_weekly_rn_f'], d['ipta_weekly_rn_hc'], d['ipta_weekly_rn_h0']),
        'ipta_wn': (d['ipta_weekly_f'], d['ipta_weekly_hc'], d['ipta_weekly_h0']),
        # Cadence variants
        'ipta_biweekly': (d['ipta_rn_f'], d['ipta_rn_hc'], d['ipta_rn_h0']),
        'ipta_biweekly_wn': (d['ipta_f'], d['ipta_hc'], d['ipta_h0']),
        'ipta_daily': (d['ipta_daily_rn_f'], d['ipta_daily_rn_hc'], d['ipta_daily_rn_h0']),
        'ipta_daily_wn': (d['ipta_daily_f'], d['ipta_daily_hc'], d['ipta_daily_h0']),
    }


@st.cache_data
def get_custom_pta_hc(n_pulsars, timespan, sigma_ns, cadence, nfreqs=200):
    """Custom PTA deterministic CW sensitivity curve via hasasia.

    Only imported/run when the user explicitly enables the custom PTA.
    Uses hasasia.sim.sim_pta + DeterSensitivityCurve.
    """
    import hasasia.sensitivity as hsen
    import hasasia.sim as hsim

    sigma_sec = sigma_ns * 1e-9
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, n_pulsars)
    theta = np.arccos(rng.uniform(-1, 1, n_pulsars))

    T_sec = timespan * YR_S
    f_min = 1.0 / T_sec
    f_max = cadence / (2.0 * YR_S)
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), nfreqs)

    psrs = hsim.sim_pta(timespan=timespan, cad=cadence, sigma=sigma_sec,
                        phi=phi, theta=theta)
    spectra = [hsen.Spectrum(psr, freqs=freqs) for psr in psrs]
    sc = hsen.DeterSensitivityCurve(spectra)
    return sc.freqs, sc.h_c



def scale_amplitude(A_bench, reservoir, rho_smbh, rho_stellar, rho_nsc):
    """Scale amplitude based on reservoir density relative to fiducial."""
    if reservoir == 'SMBH':
        return A_bench * np.sqrt(rho_smbh / RHO_SMBH_FID)
    elif reservoir == 'STELLAR':
        return A_bench * np.sqrt(rho_stellar / RHO_STELLAR_FID)
    elif reservoir == 'NSC':
        return A_bench * np.sqrt(rho_nsc / RHO_NSC_FID)
    return A_bench


def get_population_hc(pop_name, rho_smbh, rho_stellar, rho_nsc, nfreqs=500):
    """Compute characteristic strain ceiling for a GWB population.

    Returns (freqs, h_c) over the population's frequency band.
    h_c(f) = A_scaled * (f / f_ref)^(-2/3)
    """
    p = POPULATIONS[pop_name]
    A_scaled = scale_amplitude(p['A_bench'], p['reservoir'],
                               rho_smbh, rho_stellar, rho_nsc)
    f = np.logspace(np.log10(p['f_min']), np.log10(p['f_max']), nfreqs)
    hc = A_scaled * (f / p['f_ref']) ** (-2.0 / 3.0)
    return f, hc


# =============================================================================
# Echo source computation (Mingarelli et al. 2012 framework)
# =============================================================================

def f_isco_schwarzschild(M_total_msun):
    """Schwarzschild ISCO frequency for total mass M [Hz]."""
    M_kg = M_total_msun * MSUN
    return c_SI**3 / (6**1.5 * np.pi * G_SI * M_kg)


def compute_echo_sources(M_total_msun, q, D_L_Mpc, f_earth_Hz,
                         n_pulsars=20, T_pta_yr=25.0, T_muares_yr=10.0,
                         seed=42):
    """Compute earth term + pulsar term positions for an SMBHB on the h_c plot.

    Parameters
    ----------
    M_total_msun : float  — total mass in solar masses
    q : float             — mass ratio (<=1)
    D_L_Mpc : float       — luminosity distance in Mpc
    f_earth_Hz : float    — earth-term GW frequency (Hz), in muAres band
    n_pulsars : int       — number of pulsar terms to generate
    T_pta_yr : float      — PTA observation time (yr), for h_c = h_0 sqrt(f T)
    T_muares_yr : float   — muAres observation time (yr)
    seed : int            — random seed for reproducibility

    Returns
    -------
    earth : dict with keys 'f', 'hc', 'f_isco'
    pulsars : list of dicts with keys 'f', 'hc', 'd_kpc', 'tau_yr'
    warning : str or None — warning message if f_earth clamped to f_ISCO
    """
    rng = np.random.default_rng(seed)

    # Chirp mass
    eta = q / (1.0 + q)**2
    Mc_kg = M_total_msun * MSUN * eta**0.6

    D_L_m = D_L_Mpc * 1e6 * PC_SI

    # ISCO frequency clamp
    f_isco = f_isco_schwarzschild(M_total_msun)
    warning = None
    if f_earth_Hz > f_isco:
        warning = (f"f_earth = {f_earth_Hz:.1e} Hz exceeds f_ISCO = {f_isco:.2e} Hz "
                   f"for M = {M_total_msun:.0e} M☉. Clamping to 0.9 × f_ISCO.")
        f_earth_Hz = 0.9 * f_isco

    # GW strain amplitude h_0(f)
    def h0(f):
        return (4.0 / D_L_m) * (G_SI * Mc_kg / c_SI**2)**(5./3) * (np.pi * f / c_SI)**(2./3)

    # Frequency at look-back time tau (seconds) from f_earth
    # f_P = f_E * (1 + (256/5) pi^{8/3} (G Mc/c^3)^{5/3} f_E^{8/3} tau)^{-3/8}
    coeff = (256.0 / 5.0) * np.pi**(8./3) * (G_SI * Mc_kg / c_SI**3)**(5./3)

    def f_pulsar(tau_s):
        return f_earth_Hz * (1.0 + coeff * f_earth_Hz**(8./3) * tau_s)**(-3./8)

    # Earth term: h_c = h_0 * sqrt(min(f*T_obs, f^2/fdot))
    # At muHz the binary may chirp through the bin in < T_obs,
    # so use the smaller of monochromatic and chirping cycle counts.
    h0_earth = h0(f_earth_Hz)
    fdot_earth = (96.0/5.0) * np.pi**(8./3) * (G_SI*Mc_kg/c_SI**3)**(5./3) * f_earth_Hz**(11./3)
    N_mono_earth = f_earth_Hz * T_muares_yr * YR_S
    N_chirp_earth = f_earth_Hz**2 / fdot_earth
    hc_earth = h0_earth * np.sqrt(min(N_mono_earth, N_chirp_earth))
    earth = {'f': f_earth_Hz, 'h0': h0_earth, 'hc': hc_earth, 'f_isco': f_isco,
             'Mc_kg': Mc_kg, 'D_L_m': D_L_m}

    # Generate 20 pulsars with realistic distances and sky angles
    # Distances: 0.2 – 5 kpc (log-uniform), angles: uniform on sphere
    d_kpc = 10**rng.uniform(np.log10(0.2), np.log10(5.0), n_pulsars)
    cos_theta = rng.uniform(-1, 1, n_pulsars)  # angle between source & pulsar

    pulsars = []
    for i in range(n_pulsars):
        d_m = d_kpc[i] * 1e3 * PC_SI
        # Geometric delay: tau = d_p (1 - cos theta) / c
        tau_s = d_m * (1.0 - cos_theta[i]) / c_SI
        if tau_s < 1.0:
            # Nearly aligned — negligible delay, skip
            continue
        tau_yr = tau_s / YR_S
        fp = f_pulsar(tau_s)
        if fp < 1e-10 or fp > 1e0:
            continue
        h0_p = h0(fp)
        N_mono_p = fp * T_pta_yr * YR_S
        fdot_p = (96.0/5.0) * np.pi**(8./3) * (G_SI*Mc_kg/c_SI**3)**(5./3) * fp**(11./3)
        N_chirp_p = fp**2 / fdot_p
        hc_p = h0_p * np.sqrt(min(N_mono_p, N_chirp_p))
        pulsars.append({'f': fp, 'h0': h0_p, 'hc': hc_p, 'd_kpc': d_kpc[i], 'tau_yr': tau_yr})

    return earth, pulsars, warning


# =============================================================================
# PTA population defaults
# =============================================================================
# Fixed PTA arrays (always shown, not user-configurable)
PTA_FIXED = {
    'NANOGrav 15yr': {
        'n_pulsars': 67, 'timespan': 15.0, 'sigma_ns': 300, 'cadence': 26,
        'color': '#DDCC77', 'ls': '-',
        'description': '67 pulsars, 15 yr (Agazie+ 2023)',
    },
    'IPTA 2050': {
        'n_pulsars': 131, 'timespan': 50.0, 'sigma_ns': 200, 'cadence': 52,
        'color': '#88CCEE', 'ls': '-',
        'description': '131 pulsars, 50 yr, 200 ns, weekly cadence + per-pulsar red noise (Larsen+ 2026)',
    },
}

# Custom PTA defaults (user-adjustable)
CUSTOM_PTA_DEFAULTS = {
    'n_pulsars': 200, 'timespan': 20.0, 'sigma_ns': 100, 'cadence': 52,
}

SIGMA_OPTIONS = [10, 20, 50, 100, 200, 300, 500]

# =============================================================================
# Session state initialisation
# =============================================================================
def _init_defaults():
    """Set all session state keys to defaults (only if missing)."""
    st.session_state.setdefault('show_lisa', True)
    st.session_state.setdefault('lisa_T', 4)
    st.session_state.setdefault('show_muares', True)
    st.session_state.setdefault('muares_T', 10)
    st.session_state.setdefault('muares_acc', 1.0)
    st.session_state.setdefault('show_gwb_ceilings', True)
    st.session_state.setdefault('rho_smbh', RHO_SMBH_FID)
    st.session_state.setdefault('rho_stellar', RHO_STELLAR_FID)
    st.session_state.setdefault('rho_nsc', RHO_NSC_FID)
    for pop_name in POPULATIONS:
        _pop_default = True
        st.session_state.setdefault(f'show_pop_{pop_name}', _pop_default)

    # Fixed PTAs are always on
    st.session_state.setdefault('show_nanograv', True)
    st.session_state.setdefault('show_ipta2050', True)

    # Custom PTA
    st.session_state.setdefault('show_custom_pta', False)
    st.session_state.setdefault('custom_pta_n', CUSTOM_PTA_DEFAULTS['n_pulsars'])
    st.session_state.setdefault('custom_pta_T', CUSTOM_PTA_DEFAULTS['timespan'])
    st.session_state.setdefault('custom_pta_sig', CUSTOM_PTA_DEFAULTS['sigma_ns'])
    st.session_state.setdefault('custom_pta_cad', CUSTOM_PTA_DEFAULTS['cadence'])

_init_defaults()


def _reset_defaults():
    """Reset everything to paper defaults and rerun."""
    keys_to_delete = [k for k in st.session_state if k not in ('_streamlit',)]
    for k in keys_to_delete:
        del st.session_state[k]
    _init_defaults()


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    strain_convention = st.radio(
        "Strain convention",
        ["hc (characteristic strain)", "h₀ (strain amplitude)"],
        index=0,
        key='strain_convention',
        help="hc = h₀√(fT) is the characteristic strain — standard for GWB measurements. "
             "h₀ is the GW amplitude at the detector (matches Sesana+ 2021 Fig. 2)."
    )
    strain_key = 'h0' if '₀' in strain_convention else 'hc'
    st.caption(
        "**hc**: characteristic strain — standard for PTA/GWB results.  \n"
        "**h₀**: GW strain amplitude — direct observable, matches Sesana+ 2021."
    )
    show_labels = st.checkbox("Show curve labels", value=True, key='show_labels',
                               help="Toggle text labels on detector/PTA curves")
    show_legends = st.checkbox("Show legends", value=True, key='show_legends',
                                help="Toggle legend boxes on figures")
    st.markdown("---")

    st.header("Space-Based Detectors")

    show_lisa = st.checkbox("LISA", value=st.session_state['show_lisa'], key='show_lisa')
    if show_lisa:
        lisa_T = st.slider("LISA obs. time (yr)", 1, 10, st.session_state['lisa_T'], key='lisa_T')

    show_muares = st.checkbox("\u03bcAres", value=st.session_state['show_muares'], key='show_muares')
    if show_muares:
        muares_T = st.slider("\u03bcAres obs. time (yr)", 1, 20,
                              st.session_state['muares_T'], key='muares_T')
        muares_acc = st.slider(
            "\u03bcAres accel. noise (10\u207b\u00b9\u2075 m/s\u00b2/\u221aHz)",
            0.1, 10.0, st.session_state['muares_acc'], step=0.1,
            key='muares_acc',
            help="Default 1.0 = Sesana et al. (2021). Lower = better technology."
        )

    st.markdown("---")
    st.header("PTA Sensitivity")

    st.checkbox("NANOGrav 15yr", value=st.session_state['show_nanograv'],
                key='show_nanograv',
                help=PTA_FIXED['NANOGrav 15yr']['description'])
    st.checkbox("IPTA 2050", value=st.session_state['show_ipta2050'],
                key='show_ipta2050',
                help=PTA_FIXED['IPTA 2050']['description'])
    st.checkbox("IPTA 2050 (WN only)", value=st.session_state.get('show_ipta2050_wn', False),
                key='show_ipta2050_wn',
                help='131 pulsars, 50 yr, 200 ns white noise only (no red noise)')

    st.markdown("**Cadence variants**")
    st.checkbox("IPTA 2050 biweekly", value=st.session_state.get('show_ipta_biweekly', False),
                key='show_ipta_biweekly',
                help='Biweekly cadence (26/yr), WN + RN')
    st.checkbox("IPTA 2050 biweekly (WN only)", value=st.session_state.get('show_ipta_biweekly_wn', False),
                key='show_ipta_biweekly_wn',
                help='Biweekly cadence (26/yr), white noise only')
    st.checkbox("IPTA 2050 daily", value=st.session_state.get('show_ipta_daily', False),
                key='show_ipta_daily',
                help='Daily cadence (365/yr), WN + RN')
    st.checkbox("IPTA 2050 daily (WN only)", value=st.session_state.get('show_ipta_daily_wn', False),
                key='show_ipta_daily_wn',
                help='Daily cadence (365/yr), white noise only')

    st.markdown("---")
    st.subheader("Custom PTA")
    try:
        import hasasia  # noqa: F401
        _hasasia_available = True
    except ImportError:
        _hasasia_available = False
    show_custom_pta = st.checkbox("Show custom PTA", value=st.session_state['show_custom_pta'],
                                   key='show_custom_pta',
                                   help="Build your own PTA sensitivity curve",
                                   disabled=not _hasasia_available)
    if not _hasasia_available:
        st.caption("Requires `hasasia` (pip install hasasia)")
    if show_custom_pta and _hasasia_available:
        st.slider("Pulsars", 20, 1000,
                   st.session_state['custom_pta_n'], key='custom_pta_n')
        st.slider("Timespan (yr)", 5.0, 50.0,
                   st.session_state['custom_pta_T'], step=1.0, key='custom_pta_T')
        st.select_slider("RMS residual (ns)", SIGMA_OPTIONS,
                          value=st.session_state['custom_pta_sig'], key='custom_pta_sig')
        _cad_options = [("Daily (365/yr)", 365), ("Weekly (52/yr)", 52),
                        ("Biweekly (26/yr)", 26), ("Monthly (12/yr)", 12)]
        _cad_labels = [c[0] for c in _cad_options]
        _cad_map = {c[0]: c[1] for c in _cad_options}
        _cad_rev = {c[1]: c[0] for c in _cad_options}
        _cur_cad = st.session_state['custom_pta_cad']
        _cur_label = _cad_rev.get(_cur_cad, _cad_labels[0])
        _sel = st.select_slider("Cadence", options=_cad_labels,
                                value=_cur_label, key='custom_pta_cad_label')
        st.session_state['custom_pta_cad'] = _cad_map[_sel]


    st.markdown("---")
    st.header("Echo Sources")
    show_inspiral_tracks = st.checkbox("Show inspiral tracks",
                                        value=False, key='show_inspiral_tracks',
                                        help="Overlay the full inspiral h_c track "
                                             "(h₀√(f²/ḟ)) behind each echo source. "
                                             "Terminates at f_ISCO.")

    # --- Conservative binary (10⁸, 2 Gpc, 10 μHz) ---
    show_echo_conservative = st.checkbox("Conservative (10⁸ M☉, 2 Gpc)", value=True,
                                          key='show_echo_conservative',
                                          help="Most common mass, far away — weakest but guaranteed. Green markers.")
    if show_echo_conservative:
        with st.expander("Conservative binary parameters"):
            echo3_M = st.select_slider("Total mass (M☉)  ★",
                                        options=[3e7, 1e8, 3e8, 5e8],
                                        value=1e8, format_func=lambda x: f"{x:.0e}",
                                        key='echo3_M')
            _f_isco3 = f_isco_schwarzschild(echo3_M)
            st.caption(f"f_ISCO = {_f_isco3:.2e} Hz  ({_f_isco3*1e6:.1f} μHz)")
            echo3_q = st.slider("Mass ratio q  ★", 0.1, 1.0, 1.0, step=0.1, key='echo3_q')
            echo3_DL = st.slider("D_L (Mpc)  ★", 100, 5000, 2000, step=50, key='echo3_DL')
            echo3_fE = st.select_slider("f_earth (Hz)  ★",
                                         options=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
                                         value=1e-5, format_func=lambda x: f"{x:.0e}",
                                         key='echo3_fE')

    # --- Typical binary (5×10⁸, 200 Mpc, 1 μHz) ---
    show_echo_typical = st.checkbox("Typical (5×10⁸ M☉, 200 Mpc)", value=True,
                                      key='show_echo_typical',
                                      help="At the echo detection boundary (Tier 1). Orange markers.")
    if show_echo_typical:
        with st.expander("Typical binary parameters"):
            echo2_M = st.select_slider("Total mass (M☉) ",
                                        options=[1e8, 3e8, 5e8, 1e9],
                                        value=5e8, format_func=lambda x: f"{x:.0e}",
                                        key='echo2_M')
            _f_isco2 = f_isco_schwarzschild(echo2_M)
            st.caption(f"f_ISCO = {_f_isco2:.2e} Hz  ({_f_isco2*1e6:.1f} μHz)")
            echo2_q = st.slider("Mass ratio q ", 0.1, 1.0, 1.0, step=0.1, key='echo2_q')
            echo2_DL = st.slider("D_L (Mpc) ", 10, 2000, 200, step=10, key='echo2_DL')
            echo2_fE = st.select_slider("f_earth (Hz) ",
                                         options=[1e-7, 3e-7, 1e-6, 3e-6, 1e-5],
                                         value=1e-6, format_func=lambda x: f"{x:.0e}",
                                         key='echo2_fE')

    # --- Optimistic binary (10⁹, 100 Mpc, 1 μHz) ---
    show_echo_optimistic = st.checkbox("Optimistic (10⁹ M☉, 100 Mpc)", value=True,
                                        key='show_echo_optimistic',
                                        help="Golden binary — rare but spectacular. Blue markers.")
    if show_echo_optimistic:
        with st.expander("Optimistic binary parameters"):
            echo1_M = st.select_slider("Total mass (M☉)",
                                        options=[1e8, 3e8, 5e8, 1e9, 2e9, 3e9, 5e9, 1e10],
                                        value=1e9, format_func=lambda x: f"{x:.0e}",
                                        key='echo1_M')
            _f_isco1 = f_isco_schwarzschild(echo1_M)
            st.caption(f"f_ISCO = {_f_isco1:.2e} Hz  ({_f_isco1*1e6:.1f} μHz)")
            echo1_q = st.slider("Mass ratio q", 0.1, 1.0, 1.0, step=0.1, key='echo1_q')
            echo1_DL = st.slider("D_L (Mpc)", 10, 1000, 100, step=10, key='echo1_DL')
            echo1_fE = st.select_slider("f_earth (Hz)",
                                         options=[1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
                                         value=1e-6, format_func=lambda x: f"{x:.0e}",
                                         key='echo1_fE')

    # --- LISA-band binary (10⁶, 100 Mpc, 0.1 mHz) ---
    show_echo_lisa = st.checkbox("LISA-band (10⁶ M☉, 100 Mpc)", value=False,
                                   key='show_echo_lisa',
                                   help="LISA-band source — echoes completely undetectable. Red markers.")
    if show_echo_lisa:
        with st.expander("LISA-band binary parameters"):
            echo4_M = st.select_slider("Total mass (M☉)  ◆",
                                        options=[1e5, 3e5, 1e6, 3e6, 1e7],
                                        value=1e6, format_func=lambda x: f"{x:.0e}",
                                        key='echo4_M')
            _f_isco4 = f_isco_schwarzschild(echo4_M)
            st.caption(f"f_ISCO = {_f_isco4:.2e} Hz ({_f_isco4*1e3:.1f} mHz)")
            echo4_q = st.slider("Mass ratio q  ◆", 0.1, 1.0, 1.0, step=0.1, key='echo4_q')
            echo4_DL = st.slider("D_L (Mpc)  ◆", 10, 1000, 100, step=10, key='echo4_DL')
            echo4_fE = st.select_slider("f_earth (Hz)  ◆",
                                         options=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
                                         value=1e-4, format_func=lambda x: f"{x:.0e}",
                                         key='echo4_fE')

    st.markdown("---")
    st.header("GWB Ceilings")
    st.caption("Energetic ceilings from Mingarelli (2026), arXiv:2601.18859")

    show_gwb_ceilings = st.checkbox("Show GWB ceilings",
                                     value=st.session_state['show_gwb_ceilings'],
                                     key='show_gwb_ceilings',
                                     help="Astrophysical GWB amplitude ceilings "
                                          "constrained by mass density reservoirs")
    if show_gwb_ceilings:
        st.subheader("Populations")
        for pop_name in POPULATIONS:
            st.checkbox(POPULATION_DISPLAY_NAMES.get(pop_name, pop_name),
                        value=st.session_state[f'show_pop_{pop_name}'],
                        key=f'show_pop_{pop_name}')

        st.subheader("Reservoir Densities")
        rho_smbh = st.select_slider(
            "\u03c1_SMBH (M\u2609/Mpc\u00b3)",
            options=[4.2e5, 1e6, 1.8e6, 3e6, 5e6, 1e7],
            value=st.session_state['rho_smbh'],
            format_func=lambda x: f"{x:.1e}",
            key='rho_smbh',
            help="SMBH mass density. Fiducial: 1.8\u00d710\u2076 (Liepold & Ma 2024)"
        )
        rho_stellar = st.select_slider(
            "\u03c1_stellar (M\u2609/Mpc\u00b3)",
            options=[1e8, 3e8, 5.9e8, 1e9, 3e9],
            value=st.session_state['rho_stellar'],
            format_func=lambda x: f"{x:.1e}",
            key='rho_stellar',
            help="Stellar mass density. Fiducial: 5.9\u00d710\u2078"
        )
        rho_nsc = st.select_slider(
            "\u03c1_NSC (M\u2609/Mpc\u00b3)",
            options=[1e5, 5e5, 1.4e6, 3e6, 1e7],
            value=st.session_state['rho_nsc'],
            format_func=lambda x: f"{x:.1e}",
            key='rho_nsc',
            help="Nuclear star cluster mass density. Fiducial: 1.4\u00d710\u2076"
        )

    st.markdown("---")
    if st.button("Reset to defaults"):
        _reset_defaults()
        st.rerun()


# =============================================================================
# Compute curves
# =============================================================================
pta_curves = {}  # name -> (f, hc, h0)
_pta_data = _load_pta_curves()

# NANOGrav 15yr — precomputed from hasasia built-in (real noise models)
if st.session_state.get('show_nanograv', True):
    pta_curves['NANOGrav 15yr'] = _pta_data['ng15']

# IPTA 2050 — precomputed from hasasia sim_pta (WN + per-pulsar red noise)
if st.session_state.get('show_ipta2050', True):
    pta_curves['IPTA 2050'] = _pta_data['ipta']

# IPTA 2050 (WN only) — white noise only baseline
if st.session_state.get('show_ipta2050_wn', False):
    pta_curves['IPTA 2050 (WN only)'] = _pta_data['ipta_wn']

# IPTA 2050 cadence variants
if st.session_state.get('show_ipta_biweekly', False):
    pta_curves['IPTA 2050 biweekly'] = _pta_data['ipta_biweekly']
if st.session_state.get('show_ipta_biweekly_wn', False):
    pta_curves['IPTA 2050 biweekly (WN)'] = _pta_data['ipta_biweekly_wn']
if st.session_state.get('show_ipta_daily', False):
    pta_curves['IPTA 2050 daily'] = _pta_data['ipta_daily']
if st.session_state.get('show_ipta_daily_wn', False):
    pta_curves['IPTA 2050 daily (WN)'] = _pta_data['ipta_daily_wn']

# Custom PTA — computed on demand with hasasia (lazy import)
if st.session_state.get('show_custom_pta', False):
    f, hc = get_custom_pta_hc(
        n_pulsars=st.session_state['custom_pta_n'],
        timespan=st.session_state['custom_pta_T'],
        sigma_ns=st.session_state['custom_pta_sig'],
        cadence=st.session_state['custom_pta_cad'],
    )
    T_custom = st.session_state['custom_pta_T'] * YR_S
    h0 = hc / np.sqrt(f * T_custom)
    pta_curves['Custom PTA'] = (f, hc, h0)


lisa_curve = None
lisa_conf_curve = None
if st.session_state['show_lisa']:
    _fl, _hl_inst, _hl_conf = get_lisa_hc(st.session_state['lisa_T'])
    lisa_curve = (_fl, _hl_inst)       # instrument only
    lisa_conf_curve = (_fl, _hl_conf)  # confusion only

muares_curve = None
muares_conf_curve = None
if st.session_state['show_muares']:
    _fm, _hm_inst, _hm_conf = get_muares_hc(st.session_state['muares_T'],
                                              st.session_state['muares_acc'])
    muares_curve = (_fm, _hm_inst)       # instrument only
    muares_conf_curve = (_fm, _hm_conf)  # confusion only

# GWB population ceilings
gwb_ceiling_curves = {}
if st.session_state.get('show_gwb_ceilings', False):
    _rho_smbh = st.session_state.get('rho_smbh', RHO_SMBH_FID)
    _rho_stellar = st.session_state.get('rho_stellar', RHO_STELLAR_FID)
    _rho_nsc = st.session_state.get('rho_nsc', RHO_NSC_FID)
    for _pop_name in POPULATIONS:
        if st.session_state.get(f'show_pop_{_pop_name}', True):
            _f, _hc = get_population_hc(_pop_name, _rho_smbh, _rho_stellar, _rho_nsc)
            gwb_ceiling_curves[_pop_name] = (_f, _hc)

# Echo sources — optimistic (golden) binary
echo1_earth, echo1_pulsars, echo1_warning = None, [], None
# Echo sources — typical binary
echo2_earth, echo2_pulsars, echo2_warning = None, [], None
# Echo sources — LISA-band binary
echo4_earth, echo4_pulsars, echo4_warning = None, [], None
# Echo sources — conservative binary
echo3_earth, echo3_pulsars, echo3_warning = None, [], None

_T_pta = 25.0  # use longest active PTA timespan if available
if st.session_state.get('show_nanograv', True):
    _T_pta = max(_T_pta, PTA_FIXED['NANOGrav 15yr']['timespan'])
if st.session_state.get('show_ipta2050', True):
    _T_pta = max(_T_pta, PTA_FIXED['IPTA 2050']['timespan'])
if st.session_state.get('show_custom_pta', False):
    _T_pta = max(_T_pta, st.session_state.get('custom_pta_T', 20.0))
_T_mu = st.session_state.get('muares_T', 10)

if st.session_state.get('show_echo_optimistic', True):
    echo1_earth, echo1_pulsars, echo1_warning = compute_echo_sources(
        M_total_msun=st.session_state.get('echo1_M', 1e9),
        q=st.session_state.get('echo1_q', 1.0),
        D_L_Mpc=st.session_state.get('echo1_DL', 100),
        f_earth_Hz=st.session_state.get('echo1_fE', 1e-6),
        n_pulsars=20, T_pta_yr=_T_pta, T_muares_yr=_T_mu, seed=42,
    )

if st.session_state.get('show_echo_typical', True):
    echo2_earth, echo2_pulsars, echo2_warning = compute_echo_sources(
        M_total_msun=st.session_state.get('echo2_M', 6e8),
        q=st.session_state.get('echo2_q', 1.0),
        D_L_Mpc=st.session_state.get('echo2_DL', 200),
        f_earth_Hz=st.session_state.get('echo2_fE', 1e-6),
        n_pulsars=20, T_pta_yr=_T_pta, T_muares_yr=_T_mu, seed=137,
    )

if st.session_state.get('show_echo_conservative', True):
    echo3_earth, echo3_pulsars, echo3_warning = compute_echo_sources(
        M_total_msun=st.session_state.get('echo3_M', 1e8),
        q=st.session_state.get('echo3_q', 1.0),
        D_L_Mpc=st.session_state.get('echo3_DL', 2000),
        f_earth_Hz=st.session_state.get('echo3_fE', 1e-5),
        n_pulsars=20, T_pta_yr=_T_pta, T_muares_yr=_T_mu, seed=271,
    )

if st.session_state.get('show_echo_lisa', False):
    echo4_earth, echo4_pulsars, echo4_warning = compute_echo_sources(
        M_total_msun=st.session_state.get('echo4_M', 1e6),
        q=st.session_state.get('echo4_q', 1.0),
        D_L_Mpc=st.session_state.get('echo4_DL', 100),
        f_earth_Hz=st.session_state.get('echo4_fE', 1e-4),
        n_pulsars=20, T_pta_yr=_T_pta, T_muares_yr=_T_mu, seed=314,
    )


# =============================================================================
# Title
# =============================================================================
st.title("Gravity Echoes from Supermassive Black Hole Binaries")
st.caption("PTAs \u2192 \u03bcAres \u2192 LISA  |  nHz to Hz")


# =============================================================================
# Plot
# =============================================================================
# ── Modern color palette ──
# Tol qualitative palette (colorblind-safe)
_COL_MUARES = '#332288'    # indigo (μAres)
_COL_LISA = '#CC6677'      # rose (LISA)
_COL_OPT = '#CCBB44'       # olive/gold (optimistic) — distinct from SMBHB blue bg
_COL_TYP = '#AA3377'       # wine/magenta (typical) — distinct from AGN-IMRI orange bg
_COL_CON = '#EE7733'       # orange (conservative) — distinct from EMRI green bg
_COL_LISA_SRC = '#882255'  # dark plum (LISA-band)

# PTA curve styles
_pta_styles = {
    'NANOGrav 15yr': {'color': PTA_FIXED['NANOGrav 15yr']['color'],
                      'ls': PTA_FIXED['NANOGrav 15yr']['ls']},
    'IPTA 2050': {'color': PTA_FIXED['IPTA 2050']['color'],
                         'ls': PTA_FIXED['IPTA 2050']['ls']},
    'IPTA 2050 (WN only)': {'color': PTA_FIXED['IPTA 2050']['color'],
                             'ls': '--'},
    'IPTA 2050 biweekly': {'color': '#44AA99', 'ls': '-'},
    'IPTA 2050 biweekly (WN)': {'color': '#44AA99', 'ls': '--'},
    'IPTA 2050 daily': {'color': '#999933', 'ls': '-'},
    'IPTA 2050 daily (WN)': {'color': '#999933', 'ls': '--'},
    'Custom PTA': {'color': '#882255', 'ls': '-'},
}

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

ax.set_xscale('log')
ax.set_yscale('log')

# ── PTA curves (thin, clean) ──
for name, (f, hc, h0) in pta_curves.items():
    sty = _pta_styles.get(name, {'color': '#888888', 'ls': '-'})
    y = h0 if strain_key == 'h0' else hc
    ax.plot(f, y, color=sty['color'], ls=sty['ls'], lw=1.8, label=name)


# ── GWB population ceilings ──
for _pop_name, (_pf, _phc) in gwb_ceiling_curves.items():
    _pcol = POPULATIONS[_pop_name]['color']
    _plbl = POPULATION_DISPLAY_NAMES.get(_pop_name, _pop_name)
    ax.plot(_pf, _phc, color=_pcol, ls='--', lw=1.5, label=_plbl, zorder=4)
    ax.fill_between(_pf, _phc * 0.01, _phc, color=_pcol, alpha=0.10, zorder=2)

# ── Galactic WD confusion noise (one grey curve + fill) ──
# Use μAres if available (wider freq range), else LISA. Plot ONCE.
_conf_to_plot = muares_conf_curve if muares_conf_curve is not None else lisa_conf_curve
if _conf_to_plot is not None:
    _fc, _hc_conf = _conf_to_plot
    _mask = _hc_conf > 1e-25
    if np.any(_mask):
        ax.fill_between(_fc[_mask], 1e-25, _hc_conf[_mask],
                        color='#DDDDDD', alpha=0.25, zorder=0)
        ax.plot(_fc[_mask], _hc_conf[_mask], color='#AAAAAA', ls='-',
                lw=1.5, alpha=0.6, zorder=1, label='Galactic WD foreground')

# ── μAres (instrument only, solid line) ──
if muares_curve is not None:
    fm, hm = muares_curve
    ax.plot(fm, hm, color=_COL_MUARES, ls='-', lw=2.0, label='\u03bcAres', zorder=3)

# ── LISA (instrument only, solid line) ──
if lisa_curve is not None:
    fl, hl = lisa_curve
    ax.plot(fl, hl, color=_COL_LISA, ls='-', lw=2.0, label='LISA', zorder=3)

# ── Inspiral track helper ──
def _plot_inspiral_track(earth, color):
    """Plot the full inspiral h_c(f) track from ~1e-10 Hz up to f_ISCO."""
    if earth is None:
        return
    Mc_kg = earth['Mc_kg']
    D_L_m = earth['D_L_m']
    f_isco = earth['f_isco']
    f_arr = np.logspace(-10, np.log10(f_isco), 300)
    h0_arr = (4.0 / D_L_m) * (G_SI * Mc_kg / c_SI**2)**(5./3) * (np.pi * f_arr / c_SI)**(2./3)
    fdot_arr = (96.0/5.0) * np.pi**(8./3) * (G_SI * Mc_kg / c_SI**3)**(5./3) * f_arr**(11./3)
    if strain_key == 'hc':
        track = h0_arr * np.sqrt(f_arr**2 / fdot_arr)
    else:
        track = h0_arr
    ax.plot(f_arr, track, color=color, ls='-.', lw=1.0, alpha=0.35, zorder=1)

# ── Echo source helper ──
def _plot_echo(earth, pulsars, color, earth_label, psr_label, star_marker='*', psr_marker='o'):
    if earth is not None:
        ax.scatter(earth['f'], earth[strain_key],
                   marker=star_marker, s=250, color=color, zorder=10,
                   edgecolors='white', linewidths=0.8, label=earth_label)
    if pulsars:
        fp = np.array([p['f'] for p in pulsars])
        hp = np.array([p[strain_key] for p in pulsars])
        ax.scatter(fp, hp, marker=psr_marker, s=35, color=color, zorder=10,
                   edgecolors='white', linewidths=0.4, alpha=0.85, label=psr_label)
        if earth is not None:
            for p in pulsars:
                ax.plot([p['f'], earth['f']], [p[strain_key], earth[strain_key]],
                        color=color, ls='-', lw=0.3, alpha=0.15, zorder=1)

# ── Inspiral tracks (behind everything else) ──
if st.session_state.get('show_inspiral_tracks', False):
    _plot_inspiral_track(echo1_earth, _COL_OPT)
    _plot_inspiral_track(echo2_earth, _COL_TYP)
    _plot_inspiral_track(echo3_earth, _COL_CON)
    _plot_inspiral_track(echo4_earth, _COL_LISA_SRC)

# ── Echo sources ──
_plot_echo(echo4_earth, echo4_pulsars, _COL_LISA_SRC,
           'Earth term (LISA-band)', 'Pulsar terms (LISA-band)', psr_marker='s')
_plot_echo(echo3_earth, echo3_pulsars, _COL_CON,
           'Earth term (conservative)', 'Pulsar terms (conservative)', psr_marker='^')
_plot_echo(echo2_earth, echo2_pulsars, _COL_TYP,
           'Earth term (typical)', 'Pulsar terms (typical)', psr_marker='D')
_plot_echo(echo1_earth, echo1_pulsars, _COL_OPT,
           'Earth term (optimistic)', 'Pulsar terms (optimistic)', psr_marker='o')

# ── Axis limits & labels ──
ax.set_xlim(1e-10, 1e-3)
ax.set_ylim(1e-23, 1e-11)
ax.set_xlabel('Frequency (Hz)', fontsize=16, labelpad=8)
_ylabel = 'Strain Amplitude, $h_0$' if strain_key == 'h0' else 'Characteristic Strain, $h_c$'
ax.set_ylabel(_ylabel, fontsize=16, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=0.8)
ax.tick_params(axis='both', which='minor', length=3, width=0.5)
ax.grid(True, which='major', alpha=0.15, ls='-', lw=0.5)
ax.grid(True, which='minor', alpha=0.06, ls='-', lw=0.3)

# ── Clean detector labels (placed on the curves) ──
if st.session_state.get('show_labels', True):
    if muares_curve is not None:
        fm, hm = muares_curve
        # Place label at a fixed frequency well inside the plot
        idx = np.argmin(np.abs(fm - 3e-5))
        ax.text(fm[idx], hm[idx]*0.35, '\u03bcAres', fontsize=15,
                color=_COL_MUARES, fontweight='bold', ha='center', va='top')
    if lisa_curve is not None:
        fl, hl = lisa_curve
        # Place label at a fixed frequency well inside the plot
        idx = np.argmin(np.abs(fl - 3e-4))
        ax.text(fl[idx], hl[idx]*0.35, 'LISA', fontsize=15,
                color=_COL_LISA, fontweight='bold', ha='center', va='top')

    # PTA labels — place near the minimum of each curve
    for name, (f, hc, h0) in pta_curves.items():
        y = h0 if strain_key == 'h0' else hc
        idx_min = np.argmin(y)
        ax.text(f[idx_min], y[idx_min] * 0.4, name,
                fontsize=11, color=_pta_styles.get(name, {'color': '#888888'})['color'], ha='center', va='top',
                fontweight='bold', alpha=0.9)

# ── Tick formatting (MUST come AFTER all plot() calls) ──
x_decades = np.arange(-10, -2)
y_decades = np.arange(-23, -10)
ax.xaxis.set_major_locator(FixedLocator([10.0**e for e in x_decades]))
ax.yaxis.set_major_locator(FixedLocator([10.0**e for e in y_decades]))
ax.xaxis.set_major_formatter(FuncFormatter(_log_fmt))
ax.yaxis.set_major_formatter(FuncFormatter(_log_fmt))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# ── Legend: Echo Sources only ──
if st.session_state.get('show_legends', True):
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()

    src_idx = [i for i, l in enumerate(labels) if 'term' in l.lower()]

    if src_idx:
        src_handles = []
        for i in src_idx:
            h = handles[i]
            c = h.get_facecolors()[0] if hasattr(h, 'get_facecolors') else h.get_color()
            m = '*' if 'earth' in labels[i].lower() else 'o'
            ms = 10 if m == '*' else 6
            src_handles.append(Line2D([0], [0], marker=m, color='none',
                                       markerfacecolor=c, markersize=ms,
                                       markeredgecolor='white', markeredgewidth=0.5))
        # Add inspiral track entry if tracks are shown
        if st.session_state.get('show_inspiral_tracks', False):
            src_handles.append(Line2D([0], [0], color='grey', ls='-.', lw=1.0, alpha=0.5))
            src_labels = [labels[i] for i in src_idx] + ['Inspiral track']
        else:
            src_labels = [labels[i] for i in src_idx]

        ax.legend(src_handles, src_labels,
                  fontsize=13, loc='lower left',
                  ncol=2, framealpha=0.85, edgecolor='0.8',
                  borderpad=0.4, columnspacing=1.0, handlelength=1.8,
                  title='Echo Sources', title_fontsize=14)

fig.tight_layout()
st.pyplot(fig)

# =============================================================================
# Download button (PDF with PNG fallback) — saves the displayed figure directly
# =============================================================================
img = io.BytesIO()
try:
    fig.savefig(img, format='pdf', dpi=300, bbox_inches='tight')
    img.seek(0)
    st.download_button(
        label="Download Figure as PDF",
        data=img,
        file_name="gw_sensitivity.pdf",
        mime="application/pdf",
    )
except ValueError:
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    st.download_button(
        label="Download Figure as PNG",
        data=img,
        file_name="gw_sensitivity.png",
        mime="image/png",
    )
plt.close(fig)


# =============================================================================
# Key numbers table
# =============================================================================
st.markdown("---")
st.subheader("Sensitivity Summary")

rows = []
if lisa_curve is not None:
    fl, hl = lisa_curve
    idx = np.argmin(hl)
    rows.append(("LISA", f"{hl[idx]:.2e}", f"{fl[idx]:.2e}"))
if muares_curve is not None:
    fm, hm = muares_curve
    idx = np.argmin(hm)
    rows.append(("\u03bcAres", f"{hm[idx]:.2e}", f"{fm[idx]:.2e}"))
for name, (f, hc, h0) in pta_curves.items():
    y = h0 if strain_key == 'h0' else hc
    idx = np.argmin(y)
    rows.append((name, f"{y[idx]:.2e}", f"{f[idx]:.2e}"))
if rows:
    _col_label = "Min h₀" if strain_key == 'h0' else "Min hc"
    header = f"| Detector | {_col_label} | Optimal Freq (Hz) |\n|---|---|---|\n"
    body = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows)
    st.markdown(header + body)


# =============================================================================
# Echo source info
# =============================================================================
_any_echoes = (echo1_earth is not None) or (echo2_earth is not None) or (echo3_earth is not None) or (echo4_earth is not None)
if _any_echoes:
    st.markdown("---")
    st.subheader("Echo Source Parameters")

    for label, e_earth, e_pulsars, e_warn, pfx_key in [
        ("LISA-band (red)", echo4_earth, echo4_pulsars, echo4_warning, 'echo4'),
        ("Conservative (green)", echo3_earth, echo3_pulsars, echo3_warning, 'echo3'),
        ("Typical (orange)", echo2_earth, echo2_pulsars, echo2_warning, 'echo2'),
        ("Optimistic (blue)", echo1_earth, echo1_pulsars, echo1_warning, 'echo1'),
    ]:
        if e_earth is None:
            continue
        if e_warn:
            st.warning(e_warn)
        _defaults = {'echo1': (1e9, 1.0, 100), 'echo2': (6e8, 1.0, 200), 'echo3': (1e8, 1.0, 2000), 'echo4': (1e6, 1.0, 100)}
        _dM, _dq, _dDL = _defaults.get(pfx_key, (1e9, 1.0, 100))
        _M = st.session_state.get(f'{pfx_key}_M', _dM)
        _q = st.session_state.get(f'{pfx_key}_q', _dq)
        _DL = st.session_state.get(f'{pfx_key}_DL', _dDL)
        st.markdown(
            f"**{label}**: M = {_M:.0e} M\u2609, q = {_q:.1f}, "
            f"D_L = {_DL} Mpc  |  "
            f"f_ISCO = {e_earth['f_isco']:.2e} Hz ({e_earth['f_isco']*1e6:.1f} \u03bcHz)"
        )
        _strain_label = 'h₀' if strain_key == 'h0' else 'h_c'
        st.markdown(
            f"  Earth term: f = {e_earth['f']:.1e} Hz, "
            f"{_strain_label} = {e_earth[strain_key]:.2e}"
        )
        if e_pulsars:
            fp_all = [p['f'] for p in e_pulsars]
            hc_all = [p[strain_key] for p in e_pulsars]
            tau_all = [p['tau_yr'] for p in e_pulsars]
            st.markdown(
                f"  {len(e_pulsars)} pulsar terms: "
                f"f = {min(fp_all):.1e}\u2013{max(fp_all):.1e} Hz, "
                f"{_strain_label} = {min(hc_all):.2e}\u2013{max(hc_all):.2e}, "
                f"\u03c4 = {min(tau_all):.0f}\u2013{max(tau_all):.0f} yr lookback"
            )


# =============================================================================
# References
# =============================================================================
st.markdown("---")
st.subheader("PTA Red Noise Assumptions")
st.markdown("""
The IPTA 2050 curves include per-pulsar intrinsic
red noise drawn from the ranges measured in the NANOGrav 12.5-year chromatic
noise analysis ([arXiv:2511.22597](https://arxiv.org/abs/2511.22597)): amplitudes log₁₀A uniformly
distributed in [−15, −12] and spectral indices γ uniformly distributed in [1, 5].
Red noise is injected into the pulsar noise covariance matrix via hasasia's
`sim_pta`. The "WN only" curves use white noise alone (σ = 200 ns) for comparison.
Cadence variants (biweekly, weekly, daily) are scaled from the biweekly baseline
by √(cadence ratio), as the matched-filter SNR scales as √N_obs.
""")

st.markdown("---")
st.subheader("References")
st.markdown("""
- **LISA**: Amaro-Seoane et al. (2017), [arXiv:1702.00786](https://arxiv.org/abs/1702.00786)
- **\u03bcAres**: Sesana et al. (2021), Exp. Astron. 51, 1333, [arXiv:1908.11391](https://arxiv.org/abs/1908.11391)
- **PTA sensitivity formalism**: Hazboun, Romano & Smith (2019), PRD 100, 104028;
  NANOGrav curve via [hasasia](https://github.com/Hazboun6/hasasia) (deterministic CW sensitivity)
- **NANOGrav noise models**: NANOGrav 12.5-year chromatic noise analysis, [arXiv:2511.22597](https://arxiv.org/abs/2511.22597)
- **gwent**: [github.com/ark0015/gwent](https://github.com/ark0015/gwent)
""")
