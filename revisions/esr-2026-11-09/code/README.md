# REACT analysis files for the 9 November 2026 ESR manuscript

These files accompany “Beyond Composite Rankings: A Probabilistic Multi-Domain
Assessment of Regional Energy Resilience.” They use the corrected storage data.
They preserve the manuscript's 30-replicate bootstrap and deterministic M0
comparison (13 states flagged); a different workflow is present at the root of
the public repository. Use this dated snapshot to reproduce this manuscript.

## Inputs

`REACT_v4.py` contains the processed hazard panel and structural inputs. Its loader
replaces the legacy embedded storage column with `data/manuscript_storage_panel.csv`.
This is the quality-screened usable energy of registered operating battery systems,
including household systems, in the MaStR snapshot of 1 October 2025. The full
state-level extract, screening documentation, and aggregation scripts are at:
https://github.com/Prashantgrid/REACT/tree/main/data/mastr_storage/2025-10-01

The official DWD hot-day comparison uses `data/dwd_regional_averages_txbs_year.txt`.
The archived hot-day response curves in `results/r1/` and `results/r3/` are inputs to
the warming sensitivity runs. Keep them when regenerating the other results.

## Main runs and sensitivity analyses

Install `requirements.txt` in a virtual environment. From this directory:

```bash
python verify_headline.py
python REACT_ESR_reproducible.py ../results --mode all
python react_r3_configs.py ../results
python react_r3_analyses.py ../results/r3
python react_r3_mechanism.py ../results
python react_r1_analyses.py ../results/r1 copula 30
python react_r1_analyses.py ../results/r1 params
python react_r1_weights.py ../results/r1
```

Main settings are 2,000 hazard realisations, 10,000 diagnostic draws and seed
12345. The main audit reports 5,000 exploratory label permutations. Supporting
analyses use fewer permutations because classification depends on selection
frequency and the separation statistic, not the permutation p-value.

Expected identified Climate Exposure weaknesses:

- Baseline: Bayern and Baden-Württemberg.
- National Pool and Adjacency Flow: Bayern, Baden-Württemberg, Brandenburg, Sachsen.

Hamburg and Mecklenburg-Vorpommern are not identified weaknesses in the main run.

## Archived results and manuscript tables

| Manuscript content | Archived output |
|---|---|
| Main diagnoses | `results/classification_audit.csv` |
| Table 7: deterministic and probabilistic comparisons | `results/r3/classification_across_configurations.csv` |
| Table 8: Energy System tests | `results/r3/r3_variants.csv` |
| Table A.9: alternative specifications | `results/robustness/robustness_specifications.csv` |
| Table A.10: hazard/bootstrap tests | `results/r1/r1_copula.csv`, `r1_copula_fits.csv` |
| Table A.10: support, weights, omissions, overlap | `results/r1/r1_params.csv`, `r1_weights.csv` |
| Uniform and heterogeneous warming | `results/r3/climate_delta_classifications.csv`, `climate_differential_classifications.csv` |
| Physical-corridor comparison | `results/r3/lite_vs_physical_corridors.csv`, `lite_adjacency_physical_corridors.csv` |

From this directory, regenerate numerical macros and table bodies:

```bash
python make_main_numbers.py ../results ../manuscript/revision_r3
python make_r3_numbers.py ../results ../manuscript/revision_r3/numbers.tex
python make_r3_tables.py ../results ../manuscript/revision_r3
python make_r1_outputs.py ../results ../manuscript/revision_r3
```

These commands reproduce the supplied macros and tables byte for byte. In the
full manuscript package, `check_r3_numbers.py ../results ../manuscript` additionally
checks headline numbers against the manuscript and standalone response.

`REACT_PF_revision.py` contains the original warming and physical-corridor helper
routines; those routines require the external GridKit/weather intermediate model.
The corresponding processed response curves, corridor capacities, and result
outputs are archived here. The PF dispatch/adequacy experiments are not results
of this manuscript. `react_pf/mcda.py` and `react_pf/climate_delta.py` also supply
helpers imported by the structural workflow above.

## Interpretation

An identified relative weakness meets `P_sel >= 0.70` and `T > 0`. The label
permutation check is exploratory and is not a calibrated significance test.
Indicator omissions assess dependence on inputs; they do not harmonise reference
years. The reduced support model does not establish import reliability.
