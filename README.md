# REACT: code and data

Supplementary material for “Beyond Composite Rankings: A Probabilistic Multi-Domain Assessment
of Regional Energy Resilience” (Energy Strategy Reviews).

## Contents

| Folder | Content |
|---|---|
| `code/` | REACT model and analysis scripts. `REACT_v4.py` holds the processed hazard panel, the structural inputs, and the model engine; `REACT_ESR_reproducible.py` sets the manuscript specification (load-relative Energy System indicators, PRESS residuals, selection rule). |
| `code/data/` | `manuscript_storage_panel.csv`: usable energy of registered operating battery systems, including household systems, from the Market Master Data Registry snapshot of 1 October 2025 after the quality screen. `dwd_regional_averages_txbs_year.txt`: official DWD regional hot-day averages. |
| `code/react_pf/` | Helper modules (state definitions, warming deltas, classification helpers) imported by `react_r1_analyses.py` and `react_r3_configs.py`. |
| `results/` | Archived outputs of all runs reported in the manuscript. |
| `figure_sources/` | Scripts and processed data for the figures. |

## Requirements

Python 3.10 or later and the packages in `code/requirements.txt`:

```bash
pip install -r code/requirements.txt
```

## Main analysis and sensitivity analyses

From `code/`:

```bash
python verify_headline.py                              # identified cases of the main run (~1 min)
python REACT_ESR_reproducible.py ../results --mode all # main run and robustness analyses
python react_r3_configs.py ../results                  # lowest score, M0, M1, M2
python react_r3_analyses.py ../results/r3              # Energy System tests, warming
python react_r3_mechanism.py ../results                # load-density mechanism
python react_r1_analyses.py ../results/r1 copula 30    # bootstrap, split sample, trends, copula family
python react_r1_analyses.py ../results/r1 params       # support layer, weights, derating, omissions
python react_r1_analyses.py ../results/r1 noise        # prediction-perturbation scale
python react_r1_weights.py ../results/r1               # domain-emphasis weight profiles
```

Main settings: 2,000 hazard realisations per scenario, 10,000 diagnostic draws, seed 12345. The
main classification audit reports 5,000 label permutations; the supporting analyses use fewer,
because the classification depends on the selection frequency and the separation statistic only.

Identified relative weaknesses in the main run (all Climate Exposure):

- Baseline: Bayern, Baden-Württemberg.
- National Pool and Adjacency Flow: Bayern, Baden-Württemberg, Brandenburg, Sachsen.

The archived hot-day response curves (`results/r1/`, `results/r3/`) and the physical-corridor
ratings (`results/r3/`) are processed inputs to the warming and corridor comparisons; they were
derived from a daily maximum-temperature model and from the ENTSO-E grid extract used by
PyPSA-Eur. Keep them when regenerating the other results.

## Results and manuscript content

| Manuscript content | Archived output |
|---|---|
| Main diagnoses, Fig. 5, Tables 5 and A.8 | `results/classification_audit.csv` |
| Fig. 6: relative domain profiles (National Pool) | `results/results_national_pool.csv` |
| Table 6: model configurations | `results/r3/classification_across_configurations.csv` |
| Table 7: Energy System tests | `results/r3/r3_variants.csv` |
| Table 8: evidence levels | `results/r3/evidence_levels.csv` |
| Table A.9: alternative specifications | `results/robustness/robustness_specifications.csv` |
| Table A.10: hazard record | `results/r1/r1_copula.csv`, `results/r1/r1_copula_fits.csv` |
| Table A.10: support layer, weights, derating, omissions, observed inputs, warming | `results/r1/r1_params.csv`, `results/r1/r1_weights.csv` |
| Table A.10: prediction perturbation | `results/r1/r1_noise.csv` |
| Fig. 7: warming | `results/r3/climate_delta_classifications.csv`, `results/r3/climate_differential_classifications.csv` |
| Physical-corridor comparison | `results/r3/lite_vs_physical_corridors.csv`, `results/r3/lite_adjacency_physical_corridors.csv` |

For the fixed-input diagnostic, M0 identifies 13 states; adding within-domain weighting uncertainty (M1) retains four. The complete REACT run identifies 2, 4, and 4 state–scenario cases under Baseline, National Pool, and Adjacency Flow, respectively. These are diagnostic classifications, not predictions of actual outages.

The table bodies and the numerical values quoted in the text are written from these files by:

```bash
mkdir -p ../tables
python make_main_numbers.py ../results ../tables
python make_r3_numbers.py ../results ../tables/numbers.tex
python make_r3_tables.py ../results ../tables
python make_r1_outputs.py ../results ../tables
```

The figures are written to `figure_sources/out/` by the scripts in `figure_sources/`
(`python fig_rankings.py`, and likewise for the other scripts).

## Interpretation

A relative weakness is identified when the selection frequency is at least 0.70 and the separation
statistic is positive. The selection frequency describes stability under the specified hazard,
weight, and prediction uncertainty; it is not a calibrated probability. The label-permutation check
is exploratory and does not enter the classification. Indicator-omission tests assess dependence on
individual inputs; they do not harmonise reference years. The support layer does not test import
reliability.
