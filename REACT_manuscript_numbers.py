#!/usr/bin/env python3
"""Print the numbers quoted in the revised manuscript, grouped by section.
Usage: python REACT_manuscript_numbers.py <results_dir>
Use it after a rerun (e.g. with new storage data) to update the text."""
import json
import os
import sys

import pandas as pd

res = sys.argv[1] if len(sys.argv) > 1 else "results"
rv, rb = os.path.join(res, "revision"), os.path.join(res, "robustness")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 70)


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string() if hasattr(df, "to_string") else df)


a = pd.read_csv(os.path.join(res, "classification_audit.csv"))
show("Identified cases (Table 4)", a[a.identified][["Scenario", "State", "Provisional_domain", "P_sel", "Median_Z_candidate"]].round(3))
show("All provisional domains (Table 6)", a.pivot(index="State", columns="Scenario", values="P_sel").round(2).join(
    a.pivot(index="State", columns="Scenario", values="Median_Z_candidate").round(2), rsuffix=" Z"))
b = pd.read_csv(os.path.join(res, "results_baseline.csv"), index_col=0)
show("Baseline expected ranks (Section 3.1)", b.Expected_rank.sort_values().round(2))
ra = json.load(open(os.path.join(res, "figure_data", "rank_accept.json")))
show("P(rank 1-4), %", {k: round(sum(v), 1) for k, v in ra.items()})
m = pd.read_csv(os.path.join(res, "m0_m1_m2_vulnerability.csv"), index_col=0)
m["max_diff"] = m.max(axis=1) - m.min(axis=1)
show("M0/M1/M2 (Fig. 3)", m.round(3).assign(rM0=m.M0.rank(), rM1=m.M1.rank(), rM2=m.M2.rank()))
show("Scenario rank correlations", pd.read_csv(os.path.join(res, "scenario_rank_correlations.csv")).round(3))
show("Domain-score correlations", pd.read_csv(os.path.join(res, "domain_score_correlations.csv")).round(2))
show("M0/M1/M2 diagnosis (Table 5)", pd.read_csv(os.path.join(rv, "m012_classification.csv")).round(2))
r = pd.read_csv(os.path.join(rv, "m2_rank_intervals.csv"))
show("Rank intervals (Baseline)", r[r.scenario == "Baseline"].sort_values("expected_rank").round(3))
show("Support shortfall", pd.read_csv(os.path.join(rv, "support_shortfall_by_state.csv")).round(3))
show("Robustness matrix (Fig. 6)", pd.read_csv(os.path.join(rv, "robustness_matrix.csv")))
for f in ["robustness_seed_summary.csv", "robustness_threshold_sweep.csv", "robustness_indicator_reduction.csv",
          "robustness_vif_energy_by_construct.csv", "regression_leverage.csv"]:
    show(f, pd.read_csv(os.path.join(rb, f)).round(3))
pdl = pd.read_csv(os.path.join(rb, "robustness_panel_deletion.csv"))
show("Panel deletion: same provisional", f"{(pdl.provisional == pdl.provisional_full).sum()} of {len(pdl)}")
show("Permutation behaviour", pd.read_csv(os.path.join(rv, "permutation_behaviour.csv")))
