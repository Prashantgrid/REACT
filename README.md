# REACT: probabilistic multi-domain regional energy resilience

Reproducibility code for the manuscript **“Beyond Composite Rankings: A Probabilistic Multi-Domain Framework for Regional Energy Resilience.”**

REACT combines compound-hazard simulation, screening-level energy-asset derating, uncertain-weight TOPSIS, three spatial-support scenarios, and a conditional domain-profile diagnostic. The final diagnostic uses leave-one-state-out (PRESS) residuals, prediction-error perturbation, selection stability, a residual-separation statistic, state-specific label permutation, and Benjamini–Hochberg correction.

## Repository structure

- `REACT_v4.py` — embedded hazard/structural data and the Monte Carlo simulation engine.
- `REACT_ESR_reproducible.py` — manuscript-aligned entry point implementing the final PRESS + permutation-calibrated ESR diagnostic.
- `verify_headline.py` — fast deterministic test for the headline state-domain classifications.
- `requirements.txt` — Python dependencies.

The separation is intentional: `REACT_v4.py` preserves the computational/data provenance of the analysis, while `REACT_ESR_reproducible.py` contains the final diagnostic specification used for the ESR manuscript.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the manuscript-aligned numerical results

Default settings are `M=2000` hazard iterations, `Mw=10000` profile draws, `R=500` permutation replicates, and seed `12345`.

```bash
python REACT_ESR_reproducible.py results
```

The main outputs are:

- `results/results_baseline.csv`
- `results/results_national_pool.csv`
- `results/results_adjacency_flow.csv`
- `results/cross_scenario_summary.csv`
- `results/copula_fit.csv`

## Fast verification

```bash
python verify_headline.py
```

Expected calibrated classifications:

- **Baseline:** Bayern and Baden-Württemberg — Climate; Hamburg — Energy.
- **National Pool:** the same three plus Mecklenburg-Vorpommern — Adaptive/Socio-economic Capacity.
- **Adjacency Flow:** the same four as National Pool.

The selected compound-hazard family is a **Student-t copula with ν = 8**. Gaussian, Student-t, Clayton, and Gumbel candidates are compared by AIC.

## Diagnostic definition

For each retained Monte Carlo iteration, the three domain scores are residualised against the other two domains. The public default implements the manuscript's leave-one-state-out residual

`e_i / (1-h_i)`

and propagates the associated prediction error. Across `Mw` profile draws it estimates selection stability `Psel`, median standardised residuals, and the joint separation statistic `T`. State-specific domain-label permutations provide one-sided Monte Carlo p-values, and Benjamini–Hochberg correction is applied across the sixteen states within each scenario.

A state-domain profile is reported as calibrated differentiation when:

`q <= 0.10`, `Psel >= 0.70`, and `T > 0`.

## Reproducibility note

During final code–manuscript synchronization, an earlier analysis build was found to use ordinary OLS residuals with leverage-scaled perturbation, while the current manuscript states the PRESS residual above. The GitHub entry point therefore defaults to the equation stated in the manuscript. This preserves the headline calibrated classifications (3 Baseline, 4 National Pool, 4 Adjacency Flow), although some selection-frequency values can shift slightly. The manuscript table should use the public-code outputs before final submission.

For provenance only, the entry point supports:

```bash
python REACT_ESR_reproducible.py legacy_results --residual-mode legacy
```

## Data

The state-year hazard panel and the structural state inputs used by the analysis are embedded in `REACT_v4.py`. The original public sources are documented in the manuscript, including DWD climate records, MaStR, the BNetzA power-plant list, OpenStreetMap/TSO grid data, state energy balances, and regional statistics.

## Scope

REACT is a regional screening and referral framework. The spatial-support model is a reduced adequacy/deliverability representation rather than an AC power-flow model; higher-resolution climate-risk, power-system, or socio-economic analysis is intended as a second stage.
