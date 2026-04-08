# Gravity Echoes

Code and data for **"Gravity Echoes from Supermassive Black Hole Binaries"** by Qinyuan Zheng, Bence Bécsy, and Chiara M. F. Mingarelli (2026).

This repository contains the post-Newtonian evolution library, analysis scripts, and interactive Streamlit dashboard used to produce the results in the paper.

## Repository structure

```
echo-smbhb/
├── paper/                  LaTeX source and bibliography
│   ├── restructured-echo.tex
│   ├── references.bib
│   └── (RevTeX 4.2 style files)
├── figures/                Figures included in the paper
├── code/
│   ├── smbhb_evolution.py          Core post-Newtonian SMBHB evolution library
│   ├── phase_matching.py           PTA sensitivity and phase-matching utilities
│   ├── anchor_pulsars.py           Pulsar distance catalog
│   ├── app.py                      Streamlit interactive dashboard
│   ├── echo_horizon.py             Fig. 4: echo detectability contours
│   ├── plot_pn_phases.py           Fig. 5: cumulative GW cycles by pN order
│   ├── mc_error_bars_q2prior.py    Fig. 3: expected SMBHB count vs distance
│   ├── compute_table2.py           Table II: anchor pulsar properties
│   ├── compute_table3.py           Table III: echo parameters for fiducial scenarios
│   ├── compute_binary_population.py  Appendix D: binary population estimate
│   ├── compute_disk_dephasing.py   Circumbinary disk dephasing calculation
│   ├── compute_delta_f.py          Frequency shifts by pN order
│   ├── compute_beta.py             Spin-orbit parameter calculation
│   ├── verify_section4b.py         Verification: spin-distance degeneracy
│   ├── verify_equations.py         Verification: all equations in the paper
│   ├── verify_paper_numbers.py     Verification: all numerical claims
│   ├── bhmf_rates.py               Black hole mass function and merger rates
│   ├── muares_echo_population.py   Echo population synthesis
│   ├── freq_evolution_landscape.py Frequency evolution landscape
│   ├── plot_geometry.py            Precession geometry diagrams
│   ├── plot_residual_vs_mass.py    Timing residual vs total mass
│   ├── example_smbhb.py           Example usage of smbhb_evolution library
│   └── phase_error_scaling.py      Phase error from distance uncertainty
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The code depends on `numpy`, `scipy`, and `matplotlib`. The interactive dashboard additionally requires `streamlit`.

## Running the Streamlit dashboard

```bash
cd code
streamlit run app.py
```

## Reproducing paper figures

All figure-generating scripts are in `code/`. Each can be run standalone:

```bash
cd code
python echo_horizon.py          # Fig. 4
python plot_pn_phases.py        # Fig. 5
python mc_error_bars_q2prior.py # Fig. 3
```

Figures 1 (sensitivity landscape), 2 (SNR vs angle), and 6 (sky localization) were generated with the Streamlit app and the `phase_matching.py` library.

## Compiling the paper

```bash
cd paper
pdflatex restructured-echo
bibtex restructured-echo
pdflatex restructured-echo
pdflatex restructured-echo
```

Requires a LaTeX distribution with RevTeX 4.2 (included in this repository).

## Citation

Citation information will be added once the paper is available on the arXiv.

## License

This code is released under the MIT License.
