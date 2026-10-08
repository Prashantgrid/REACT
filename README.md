# REACT: probabilistic multi-domain regional energy resilience

Reproducibility code for the manuscript **"Beyond Composite Rankings: A Probabilistic Multi-Domain
Assessment of Regional Energy Resilience"** (Energy Strategy Reviews).

REACT combines compound-hazard simulation, climate-driven derating of energy-system indicators,
uncertain-weight TOPSIS, three spatial stress-and-support scenarios, and a relative-weakness
diagnostic based on leave-one-state-out (PRESS) residuals and a selection-stability criterion.

## Repository structure

- `REACT_v4.py` — embedded hazard and structural data and the Monte Carlo simulation engine.
- `REACT_ESR_reproducible.py` — manuscript specification: load-relative Energy System indicators,
  the relative-weakness diagnostic, all robustness analyses, and the synthetic recovery check.
- `verify_headline.py` — headline verification at the manuscript settings (about one minute).
- `requirements.txt` — Python dependencies.
- `reference/classification_audit.csv` — all 48 state–scenario cases: provisional domain,
  `P_sel`, median residual, `T`, permutation `p`, BH `q`, identification, and expected rank.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the manuscript

```bash
python REACT_ESR_reproducible.py results --mode all
```

Settings: `M=2000` hazard iterations, `Mw=10000` profile draws, `R=5000` permutation replicates,
seed `12345`. Runtime is about 15 minutes. `--mode run` produces the main results only;
`--mode robustness` and `--mode recovery` produce the appendix analyses. `--fast` runs a reduced
smoke test (M=250, Mw=1000, R=100); its selection frequencies differ slightly from the manuscript.

| Manuscript element | Output file |
|---|---|
| Table 4, Fig. 5, Table 6 (regional pattern) | `results/classification_audit.csv` |
| Fig. 3 (M0, M1, M2) | `results/m0_m1_m2_vulnerability.csv`, `results/figure_data/m012.json` |
| Fig. 4 (rank acceptability) | `results/figure_data/rank_accept.json` |
| Table 5 (indicator context) | `results/indicator_ranks.csv`, `results/structural_indicator_panel.csv` |
| Domain-score correlations | `results/domain_score_correlations.csv`, `results/domain_scores_baseline.csv` |
| Scenario rank correlations | `results/scenario_rank_correlations.csv` |
| Copula selection | `results/copula_fit.csv` |
| Specification sensitivity (Table A.8) | `results/robustness/robustness_specifications.csv` |
| Threshold sweep (Table A.7) | `results/robustness/robustness_threshold_sweep.csv` |
| Leave-one-state-out reruns | `results/robustness/robustness_panel_deletion.csv` |
| Monte Carlo replicas (Table A.4) | `results/robustness/robustness_seed_summary.csv` |
| Indicator redundancy (Table A.5, VIF) | `results/robustness/robustness_indicator_reduction.csv`, `robustness_vif_*.csv` |
| Regression leverage | `results/robustness/regression_leverage.csv` |
| Planted-deviation recovery | `results/robustness/recovery_planted_deviation.csv` |

## Headline verification

```bash
python verify_headline.py
```

Expected identified relative domain weaknesses:

- **Baseline:** Bayern and Baden-Württemberg — Climate Exposure.
- **National Pool:** Bayern, Baden-Württemberg, Brandenburg — Climate Exposure; Hamburg — Energy System.
- **Adjacency Flow:** Bayern, Baden-Württemberg, Brandenburg — Climate Exposure.

The selected compound-hazard family is a **Student-t copula with ν = 8**; Gaussian, Student-t,
Clayton, and Gumbel candidates are compared by AIC.

## Energy System construct

Energy System capability is measured relative to the load each state serves:

- wind, PV, and thermal capacity: MW per MW of average load (`D_annual / 8760`);
- storage energy capacity: MWh per MW of average load (hours of average demand);
- grid infrastructure: line-length index per 1000 km² of state area (network density);
- interstate connectivity `C_i` and the external-support score `S(r_i)`: unchanged.

The support layer multiplies the ratios back by average load, so supply, demand, and interstate
flows stay in MW. Sensitivity specifications:

```bash
python REACT_ESR_reproducible.py results_absolute --energy-construct absolute
python REACT_ESR_reproducible.py results_grid_demand --grid-normalisation demand
python REACT_ESR_reproducible.py results_grid_geomean --grid-normalisation geomean
```

State areas are from Destatis GENESIS table 33111-0008 (31 December 2023).

## Diagnostic definition

For each Monte Carlo iteration, each domain score is regressed on the other two, and the
leave-one-state-out residual `e_i / (1 - h_i)` is formed. Across `Mw` profile draws, with the
associated prediction error added, the residuals are standardised across states within each
domain. `P_sel` is the share of draws in which a domain is the state's lowest standardised
residual, and `T` measures its separation from zero and from the other two domains. A relative
domain weakness is reported when

`P_sel >= 0.70` and `T > 0`.

The code also reports a within-state label-permutation check with Benjamini–Hochberg adjustment;
it is satisfied by all identified cases and does not alter any classification.

Random-number generation is deterministic. Each spatial scenario starts from a scenario-specific
generator (`seed + 0`, `+101`, `+202`); hazard draws, derating perturbations, and weight draws use
that stream sequentially, and the diagnostic continues from the captured post-simulation state.

## Figures

`figure_sources/` contains the scripts for the manuscript figures. Figures 3–5 read the JSON
files written to `results/figure_data/`; copy them to `figure_sources/data/` and run
`fig_rankings.py` and `fig_relative_weakness_map.py`.

## Data

The state-year hazard panel and the structural state inputs are embedded in `REACT_v4.py`. The
original public sources are documented in the manuscript: DWD climate records, MaStR, the BNetzA
power-plant list, OpenStreetMap/TSO grid data, state energy balances, Destatis area statistics,
and regional statistics.

## Scope

REACT is a regional screening and prioritisation framework. The spatial-support model is a reduced
adequacy and deliverability representation rather than an AC power-flow model; higher-resolution
climate-risk, power-system, or socio-economic analysis is intended as a second stage.

## Changes in this version

- Energy System indicators measured relative to regional load (principal specification);
  absolute capacities and alternative grid normalisations retained as sensitivity options.
- Released code for all robustness analyses: specification sensitivity, threshold sweep,
  leave-one-state-out reruns, seed replicas, indicator reduction and VIF, regression leverage,
  and planted-deviation recovery.
- Classification rule stated as `P_sel >= 0.70` and `T > 0`; the permutation check is reported.
- Fixed a missing line break in `REACT_v4.py` that prevented import, and a copula-cache key that
  ignored the state set in panel-deletion reruns.
