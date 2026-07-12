# Spin-spin convention audit — 2026-07-11

## Summary

Three inconsistent 2PN spin-spin (SS) treatments coexisted in this repo. An
external review flagged the disagreement; `lalsimulation` (LIGO's reference
post-Newtonian implementation) was used as the arbiter. **Verdict: the main
module (`code/smbhb_evolution.py`) was correct all along; the two verification
scripts and one helper function carried a wrong σ/η-style normalization and
have been fixed. The paper's published numbers are unaffected.**

## The disagreement

All parties use the same η-inclusive Poisson & Will (1995) spin-spin parameter
σ = (η/48)[−247 χ₁χ₂ cos(κ₁−κ₂) + 721 χ₁cosκ₁ χ₂cosκ₂]. They disagreed on how
σ enters the TaylorT2 time/phase coefficients:

| Location | τ₄_SS | φ₄_SS | Status before audit |
|---|---|---|---|
| `smbhb_evolution.py` (authoritative; generated all paper tables) | −2σ | −10σ | **correct** |
| `crosscheck_cycle_counts.py` Method B | −(40/η)σ | −(10/η)σ | wrong (×~5–20 at η=1/4) |
| `crosscheck_cycle_counts.py` Method A | SS added to T1 flux/energy (η-free form) | — | inconsistent with the library's documented SS-free T1 design |
| `verify_equations.py` check 18 | expected SS in binding energy, −σ/η form | — | wrong expectation; **was failing (exit 1)** |
| `phase_matching.py::pn_cycles` | — | −10σ/η, and SO term missing the 75η piece of β_SO | wrong (non-authoritative helper) |

Observable symptoms before the fix: `verify_equations.py` exited 1 (its single
red check), and `crosscheck_cycle_counts.py` reported 17–83-cycle disagreements
on every χ=0.98 scenario.

## The arbiter

`code/audit_ss_convention.py` compares against `lalsimulation` 6.2.0
(`SimInspiralTaylorF2AlignedPhasing`). PN phasing coefficients are
dimensionless functions of (η, χᵢ) only, so LIGO-scale and SMBHB-scale
binaries are equivalent tests. The χ₁χ₂ bilinear of the 2PN coefficient —
D = [ψ₄(χ,χ) − ψ₄(χ,0) − ψ₄(0,χ) + ψ₄(0,0)] — isolates the disputed spin1-spin2
term, cancelling self-spin/quadrupole-monopole contributions. Two mass ratios
(η = 0.25 and 0.1875) separate the module's −10σ (∝ η·χ₁χ₂) from the
crosscheck's −(10/η)σ (η-free) by their η-scaling.

**Result: LAL matches −10σ to relative deviation ≲ 4×10⁻¹⁶ at both mass
ratios; the −(10/η)σ form is off by 75–81%.** The audit also confirms, to
machine precision: the 1.5PN spin-orbit term (4β_SO with the full
113(mᵢ/M)²+75η structure), the 1PN and 2PN mass sectors, and the ψ₄ = φ₄ = 5τ₄
structural identity that the wrong convention violated.

Structural cross-evidence: the exact mass-sector identity φ₄ = 5τ₄
(15293365 = 5×3058673, 27145 = 5×5429, 3085 = 5×617) is satisfied by the
module's SS pair (−10σ = 5·(−2σ)) and violated by the crosscheck's
(−(10/η)σ ≠ 5·(−40/η)σ).

## Fixes applied (all in `code/`)

1. `crosscheck_cycle_counts.py` — Method B: τ₄_SS → −2σ, φ₄_SS → −10σ;
   Method A: SS removed from TaylorT1 flux/energy to mirror the library's
   documented aligned-spin simplification (the Kidder SS flux/energy terms
   belong to the package roadmap's `include_ss` option).
2. `verify_equations.py` check 18 — now asserts the mass-only Ep4 that the
   module deliberately implements, plus a new check pinning σ to the
   η-inclusive Poisson & Will definition.
3. `phase_matching.py::pn_cycles` — 2PN SS: −10σ/η → −10σ; 1.5PN SO:
   (113/6)χ → 4β_SO = (113−76η)χ/3 (the 75η piece had been dropped). This
   helper feeds `analyze_echo`'s per-pulsar cycle diagnostics only; at η=1/4
   its N_1.5PN was low by ×1.66 and its N_2PN SS piece high by ×4.

## Paper impact: none on published tables

Tables II/III, the Δf columns, Fig. 5, and the §4b σ_χ numbers were generated
by `pn_decomposition` / `compute_delta_f.py` / `plot_pn_phases.py` /
`verify_section4b.py`, all of which already carried the LAL-confirmed
−2σ/−10σ convention (verified by grep during the audit). `verify_paper_numbers.py`
passes unchanged after the fixes.

## Post-fix status

Full battery green: `verify_equations.py` 25 passed / 0 failed,
`verify_paper_numbers.py` all-pass, `verify_section4b.py` pass,
`crosscheck_cycle_counts.py` PASS on all scenarios (including χ=0.98),
`crosscheck_sigma_chi.py` pass. The pre-fix paper state is preserved at the
local git tag `v1.0-paper` (commit 7643c20).

Adopted convention, normative for the `pulsarterm_physics` package:
**σ_SS is η-inclusive (Poisson & Will 1995); τ₄_SS = −2σ, φ₄_SS = ψ₄_SS = −10σ;
SS is excluded from the TaylorT1 flux/energy path by design (aligned-spin
simplification) and carried entirely by the TaylorT2/F2 phase.**
