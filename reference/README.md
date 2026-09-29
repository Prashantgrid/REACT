# Manuscript reference output

This directory provides a compact audit of the manuscript-aligned REACT run.

- `classification_audit.csv` reports all 48 state-scenario cells with the provisional domain, selection stability `P_sel`, candidate sign consistency `W_candidate`, candidate median standardized residual, joint separation statistic `T_joint`, raw label-permutation `p`, Benjamini-Hochberg `q`, and final classification.

The manuscript defaults are `M=2000` hazard iterations, `Mw=10000` profile draws, `R=500` permutation replicates, and seed `12345`.

Reproduce the numerical run with:

```bash
python REACT_ESR_reproducible.py results --no-figures
```

Random-number generation is deterministic. Each spatial scenario starts from a scenario-specific NumPy generator seed derived from the manuscript seed (baseline +0, national pool +101, adjacency flow +202). Hazard draws, derating-parameter perturbations, and weight draws use that scenario stream sequentially; the diagnostic continues from the captured post-simulation generator state, including residual perturbations and label permutations.
