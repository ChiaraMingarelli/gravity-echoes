# gravity-echoes

Code and data for **"Multiband Gravity Echoes from Supermassive Black Hole Binaries"** by Qinyuan Zheng, Bence Bécsy, and Chiara M. F. Mingarelli (2026).

This repository contains the post-Newtonian evolution library, analysis scripts, and interactive Streamlit dashboard used to produce the results in the paper.

## Repository structure

```
gravity-echoes/
├── paper/                          LaTeX source and bibliography
│   ├── restructured-echo.tex
│   ├── references.bib
│   └── (RevTeX 4.2 style files)
├── figures/                        Figures included in the paper
├── code/
│   ├── smbhb_evolution.py          Core post-Newtonian SMBHB evolution library
│   ├── phase_matching.py           PTA sensitivity and phase-matching utilities
│   ├── anchor_pulsars.py           Pulsar distance catalog
│   │
│   ├── app.py                      Fig. 1: interactive multiband sensitivity dashboard
│   ├── mc_error_bars_q2prior.py    Fig. 3: expected SMBHB count vs distance
│   ├── echo_horizon.py             Fig. 4: echo detectability contours
│   ├── plot_pn_phases.py           Fig. 5: cumulative GW cycles by pN order
│   │
│   ├── compute_table2.py           Table II: echo parameters for fiducial scenarios
│   ├── compute_table3.py           Table III: combined-SNR horizons by tier
│   ├── compute_delta_f.py          Table II frequency-shift columns
│   │
│   ├── compute_rho_E.py            Earth-term SNR and sigma_chi^(E) Fisher calc
│   ├── compute_sigma_chi_pulsar.py Single-pulsar sigma_chi (Eq. sigma_chi_prior)
│   ├── compute_sigma_chi_network.py Network Fisher-sum sigma_chi (Appendix C)
│   ├── compute_fp_shift.py         Pulsar-term frequency shift Delta f_P
│   ├── compute_beta.py             Spin-orbit parameter beta
│   ├── compute_disk_dephasing.py   Circumbinary disk dephasing
│   ├── compute_binary_population.py Appendix C: binary population estimate
│   │
│   ├── verify_section4b.py         Verification: spin-distance degeneracy
│   ├── verify_equations.py         Verification: all equations in the paper
│   ├── verify_paper_numbers.py     Verification: all numerical claims
│   ├── crosscheck_cycle_counts.py  Cross-check: pN cycle counts
│   ├── crosscheck_horizon.py       Cross-check: horizon distances
│   └── crosscheck_sigma_chi.py     Cross-check: sigma_chi calculations
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The core library depends on `numpy` and `scipy`. Figure scripts additionally require `matplotlib`. The interactive dashboard requires `streamlit`.

## Interactive dashboard

A hosted version is available at [gravityecho.streamlit.app](https://gravityecho.streamlit.app).

To run locally:

```bash
cd code
streamlit run app.py
```

The dashboard generates the multiband sensitivity landscape shown in Fig. 1.

## Reproducing paper figures

All figure-generating scripts are in `code/`. Each can be run standalone:

```bash
cd code
python mc_error_bars_q2prior.py    # Fig. 3
python echo_horizon.py             # Fig. 4
python plot_pn_phases.py           # Fig. 5
```

Figures 1 (multiband sensitivity landscape) and 2 (sky localization) are produced by the Streamlit app and the `phase_matching.py` library. Figure 6 (geometry) is generated separately.

## Reproducing paper tables

```bash
cd code
python compute_table2.py           # Table II
python compute_table3.py           # Table III
python compute_delta_f.py          # Table II frequency-shift columns
```

Table I is an input-configuration table and has no generator.

## Verification scripts

The `verify_*` and `crosscheck_*` scripts re-derive every numerical claim in the paper from first principles:

```bash
cd code
python verify_paper_numbers.py     # all quoted numbers
python verify_equations.py         # all equations
python verify_section4b.py         # spin-distance degeneracy
python crosscheck_cycle_counts.py  # pN cycle counts
python crosscheck_horizon.py       # horizon distances
python crosscheck_sigma_chi.py     # sigma_chi
```

## Compiling the paper

```bash
cd paper
pdflatex restructured-echo
bibtex restructured-echo
pdflatex restructured-echo
pdflatex restructured-echo
```

Requires a LaTeX distribution with RevTeX 4.2 (style files included in `paper/`).

## License

This code is released under the MIT License. See `LICENSE` for details.
