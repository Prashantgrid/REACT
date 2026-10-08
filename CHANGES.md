# REACT ESR package — changes in this version

## Specification decision

Energy System capability is now measured relative to the load each state serves (principal
specification); absolute capacities are a sensitivity specification.

| Indicator | Principal specification |
|---|---|
| Wind, PV, thermal capacity | MW per MW of average load (`D_annual / 8760`) |
| Storage energy capacity | MWh per MW of average load (hours of average demand) |
| Grid infrastructure | line-length index per 1000 km² (network density) |
| Interstate connectivity `C_i`, external support `S(r_i)` | unchanged |

Grid infrastructure is area-normalised because meshing and redundancy are spatial properties and
demand-relative transfer capability is already carried by `C_i`. Normalising by average load or by
the geometric mean of area and load gives expected-rank correlations of 0.96–1.00 with the main
specification and the same Climate Exposure classifications (Table A.8). Dividing by average load
or by annual demand gives identical scores after min–max normalisation.

## Headline results (M=2000, Mw=10000, R=5000, seed 12345)

- Identified relative weaknesses (9 of 48 cases): Bayern and Baden-Württemberg — Climate Exposure
  (all three scenarios); Brandenburg — Climate Exposure (both coherent scenarios); Hamburg — Energy
  System (National Pool).
- Baseline expected ranks: Bremen, Hamburg, Niedersachsen first; Sachsen-Anhalt last.
- Energy System and Socio-economic Capacity domain scores: Pearson r = −0.79.
- Absolute capacities: rank correlation 0.63–0.78 with the main specification; reproduce the
  previous headline cases (Hamburg Energy ×3, Mecklenburg-Vorpommern Socio-economic ×2).

## Manuscript

Rewritten: abstract, Table 2 (Energy units), construct paragraph (Section 2), Section 3 (all
results), Section 4 (strategic agendas, scope), Section 5, Data availability, appendix
robustness section (Tables A.4–A.8), support-layer MW conversion (A.2.3). New reference:
Destatis land area (GENESIS 33111-0008). Figures 3–5 regenerated from model output.
Unchanged: introduction structure and Table 1, hazard model, event checks, heatwave validation,
hazard parameters, interdependence and connectivity definitions, Figures 1, 2, A.1–A.3.

## Code

- `REACT_ESR_reproducible.py`: load-relative construct with `--energy-construct` and
  `--grid-normalisation`; `--mode run | robustness | recovery | all`.
- Released code for every appendix analysis (previously unreleased: panel deletion, seed
  replicas, indicator reduction, recovery).
- Fixes: missing line break in `REACT_v4.py` (import failed); copula cache key ignored the state
  set during panel-deletion reruns.
- `verify_headline.py` checks the new headline set at full settings (~15 s).

## Manuscript reference update (8 October 2026)

- Added Pant et al. (2025), IJEPES 172:111324, in the limitations discussion on converter/controller dynamics.
- Clarified the conclusion claim about consistency across spatial scenarios.

## Before submission

1. **Storage data.** `Storage_MWh` is 0 for Hamburg, Berlin, and Sachsen-Anhalt in the embedded
   panel. MaStR registers home and large-scale batteries in all states, so these zeros are likely
   a filtering artefact. They lower all three states' Energy System scores. Please check against
   MaStR and rerun if corrected.
2. Confirm the volume number (86) added to the two DIW Wochenbericht 2019 references.
3. Confirm the wording of the generative-AI declaration.
4. Push `code/` (and optionally `figure_sources/`) to the GitHub repository so the Data
   availability statement matches.
