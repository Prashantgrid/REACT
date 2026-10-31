# REACT: revised Energy Strategy Reviews manuscript

Reproducibility code for **Beyond Composite Rankings: A Probabilistic Multi-Domain
Assessment of Regional Energy Resilience**, ESR-D-26-03728.

The main battery indicator now uses **one archived MaStR export, 1 October 2025,
for all 16 states**. No state uses an interpolated or PV-proportional battery value.
The later 31 October 2026 export is a separately dated sensitivity.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_revision.sh results
```

In the manuscript ZIP, these files are under `code/`; invoke
`bash code/run_revision.sh results` from the package root. In the GitHub repository
the code is at the repository root. The script resolves its input paths itself.
It runs the three main scenarios, all appendix and reviewer analyses, 40 full-model
year-block bootstrap samples, summaries, data checks, and Figures 3–6. It also
updates numerical LaTeX tables when the manuscript source is present. Runtime
depends on hardware. Logs are kept beside the outputs. `JOBS=2` reduces concurrent
bootstrap workers; the default is four. `PYTHON=/path/to/python` selects an interpreter.

Main settings: 2,000 hazard iterations, 10,000 diagnostic profile draws, 5,000
permutation audit replicates, seed 12345. Robustness classifications use fewer
permutation replicates because that audit does not enter the classification rule.
A smoke test (`python REACT_ESR_reproducible.py smoke --fast`) uses smaller
simulation counts and must not be quoted as manuscript results.

## Main input and findings

`data/mastr_storage/2025-10-01/storage_by_state.csv` is the default input. REACT
reads `battery_energy_MWh_excluding_duration_flags` as `Storage_MWh` for every
state and rejects missing states, duplicate states, invalid or negative values.
The dataset keeps the raw energy column, net and gross power separately, missing
energy counts, duration flags, and source-member fingerprints. See its README for
selection, joining, units, and limitations.

The screened national 2025 battery stock is 23,871.648430 MWh; raw stock is
23,982.567730 MWh. The 21 flagged plants are a diagnostic exclusion, not an
independently verified correction. Registered stock does not establish state of
charge, dispatchability, household-battery controllability, or firm support.

The manuscript rule is `P_sel >= 0.70 and T > 0`. The default panel identifies:

- Baseline: Bayern and Baden-Württemberg, Climate Exposure.
- National Pool and Adjacency Flow: Bayern, Baden-Württemberg, Brandenburg,
  and Sachsen, Climate Exposure.

These are **10 state–scenario cases**. Sachsen is just above the threshold under
coherent hazards and is sensitive to input and specification changes. Hamburg
remains below the main threshold. The 2022-only socio-economic subset removes
Baden-Württemberg's identification; it also removes indicators and changes the
construct. The results are comparative screening findings, not validated outage
or operational-resilience predictions.

```bash
python verify_headline.py
python -m unittest discover -s tests -v
```

The fixed-input diagnostic (M0) identifies five states; within-domain weight
uncertainty (M1) retains four. M0 still propagates regression prediction error.
The selected historical hazard copula is Student-t with nu=8.

## Output map

| Content | File under `results/` |
|---|---|
| All 48 state–scenario diagnoses | `classification_audit.csv` |
| Exact input panel after normalization | `structural_indicator_panel.csv` |
| Input and code fingerprints, settings | `run_manifest.json` |
| Main expected ranks and rank acceptability | `results_*.csv`, `figure_data/` |
| M0/M1/M2 and conventional diagnoses | `revision/m012_classification.csv` |
| Storage panels, including raw and later snapshots | `revision/storage_correction.csv` |
| All robustness checks summarized | `revision/robustness_matrix.csv` |
| Full-model 40-sample year-block bootstrap | `revision/bootstrap_full_model.csv` |
| Copula dependence, trends, held-out-year likelihood | `revision/copula_*.csv`, `revision/hazard_*.csv` |
| Indicator and construct checks, five seeds, 16 panel deletions | `robustness/` |
| Numbers for checking manuscript prose | `manuscript_numbers.txt` |
| Input completeness and plausibility checks | `data_checks.md` |

`reference/classification_audit.csv` records the complete main audit. Figures use
the same output data, through `finalize_results.py` and `figure_sources/`.
`update_manuscript_tables.py` updates numerical tables; narrative interpretations
and dates still require author review when changing model inputs.

## Storage source and sensitivities

The aggregation script reads the unit, plant, and catalogue XML members of one
Bundesnetzagentur MaStR ZIP. It joins `SpeMastrNummer` to plant `MaStRNummer`,
counts usable energy once per eligible plant, converts kW to MW and kWh to MWh,
and excludes foreign, inactive, nonoperating, and known future units. Missing
capacities are counted, never estimated. Batteries are separated from pumped
storage and other technologies. Source URLs, filters, complete XML-member
SHA-256 fingerprints and ZIP CRCs are in each snapshot's `provenance.json`.

```bash
python tools/aggregate_mastr_storage.py official_export.zip \
  --snapshot-date 2025-10-01 \
  --source-url https://download.marktstammdatenregister.de/Stichtag/Gesamtdatenexport_20251001_25.2.zip \
  --output-dir data/mastr_storage/2025-10-01
```

Main data: [MaStR download portal](https://www.marktstammdatenregister.de/MaStR/Datendownload),
provider Bundesnetzagentur, Datenlizenz Deutschland attribution 2.0.
This is a derived dataset, not an official state statistical release.

The storage sensitivity replaces **all 16 values** with raw 2025 totals, the
original submitted panel, the historical interim panel, or raw/screened 2026
totals. The interim HH=145, BE=217, ST=605 MWh values are retained only for this
comparison. To use another complete dated aggregate, pass it explicitly:

```bash
bash run_revision.sh other_results path/to/main.csv path/to/later.csv
```

The embedded structural and hazard data are in `REACT_v4.py`; the default battery
panel overrides its legacy storage values. Energy capability is measured per MW
of average regional load; battery energy per load is hours of average demand.
Grid density is per state area. Absolute capacities, alternative grid
normalisations, and thermal capacity coded as a cost or excluded are reported
as construct sensitivities. National Pool and Adjacency Flow remain simplified
support assumptions, not physical dispatch or observed corridor-limit models.
