# Validation of the corrected ESR revision

## Numerical and data checks completed

- Full main run: M=2,000, diagnostic profile draws=10,000, permutation audit
  replicates=5,000, seed=12345; all three spatial scenarios.
- All specification, redundancy, five-seed, 16-state-deletion, synthetic-recovery,
  weight, support, overlap, socio-economic, storage, and climate-shift analyses
  completed at the manuscript settings.
- Forty complete year-block bootstrap samples, 48 rows per sample: 1,920 rows.
- `python verify_headline.py`: PASS, reproducing 2/4/4 identified cases under
  Baseline/National Pool/Adjacency Flow, respectively.
- Three automated tests: PASS. They cover shared-plant energy deduplication,
  missing values, source-status/date/geography exclusions, unallocated energy,
  duration flags, complete-state input validation, and default-panel loading.
- A separate sequential/parallel bootstrap comparison with identical resamples
  and seeds produced exactly equal model output frames.
- Reduced- and full-permutation runs gave exactly equal primary numerical
  outputs and classifications. Permutation values do not enter identification.
- Numerical table updates, summaries, data checks, and Figures 3–6 completed.
  `bash -n run_revision.sh` and Python compilation pass.

## Source evidence

The single archived export is dated 1 October 2025, schema 25.2. All 45 required
XML members (catalogue, 22 unit files, 22 plant files) were read fully with ZIP
CRC verification and uncompressed SHA-256 fingerprints. The snapshot has
2,137,717 eligible battery units in the 16-state aggregate. The national raw
usable-energy sum is 23,982.567730 MWh; the screened sum is 23,871.648430 MWh.

Twenty-one duration flags remove 110.919300 MWh, 0.46% of the raw total. Missing
usable energy affects 37 allocated battery plants; 49 units have missing or
unallocated plant energy. No eligible linked plant spans states or technologies
in this snapshot. Raw and screened 2025 inputs give the same ten classifications.

The later 2026 export retains its raw NRW anomaly and a separate screened column.
Non-battery storage totals include implausible pumped-storage energy values;
these are flagged in the data checks and are excluded from the battery indicator.

## Interpretation limits retained

Registry plausibility checks are not independent measurement validation.
Registered battery stock, particularly household stock, is not proven available
for interstate support. The duration screen is explicit and reversible, with raw
sensitivity outputs. The two dates are separate stock snapshots, not a time series
reconstructed from a current register.

REACT is a comparative screening model. It does not establish outage rates,
chronological adequacy, restoration performance, or physical import reliability.
Sachsen is near the threshold and varies across seeds, historical samples,
input panels, and specifications. Socio-economic and correlation-based indicator
reductions can change the southern diagnoses. The manuscript and response letter
report these qualifications.
