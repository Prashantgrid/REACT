#!/usr/bin/env python3
"""Classifications across model configurations (Reviewer 1.8, Reviewer 3.5), REACT only.

Conventional lowest domain score, deterministic diagnosis (M0), weight uncertainty with a fixed
indicator matrix (M1), and full REACT (M2, read from the classification audit of the main run).
Usage: python react_r3_configs.py <results_dir>   (writes <results_dir>/r3/classification_across_configurations.csv)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import REACT_v4 as core  # noqa: E402
import REACT_ESR_reproducible as R  # noqa: E402
from react_pf import mcda as MC  # noqa: E402

RES = sys.argv[1]
SCEN = ("baseline", "national_pool", "adjacency_flow")
LABEL = {"baseline": "Baseline", "national_pool": "National Pool", "adjacency_flow": "Adjacency Flow"}
static = R._configured_static()
hazard = core.load_hazard_panel()
audit = pd.read_csv(os.path.join(RES, "classification_audit.csv"))
m0, _ = MC.lite_m0(static, hazard)
m1 = MC.lite_m1(static, hazard, M=2000, Mw=10000, n_perm=200)
states = core.STATE_ORDER
cmp = pd.DataFrame(index=states)
cmp["conventional_lowest_domain"] = m0.lowest_domain_score
cmp["M0_provisional"] = m0.provisional; cmp["M0_T"] = m0["T"]; cmp["M0_class"] = m0.classification
cmp["M1_provisional"] = m1.provisional; cmp["M1_Psel"] = m1.P_sel; cmp["M1_T"] = m1["T"]
cmp["M1_class"] = m1.classification
for sc in SCEN:
    la = audit[audit.Scenario == LABEL[sc]].set_index("State").loc[states]
    cmp[f"M2_{sc}_provisional"] = la.Provisional_domain; cmp[f"M2_{sc}_Psel"] = la.P_sel
    cmp[f"M2_{sc}_class"] = la.Classification; cmp[f"M2_{sc}_rank"] = la.Expected_rank
os.makedirs(os.path.join(RES, "r3"), exist_ok=True)
cmp.to_csv(os.path.join(RES, "r3", "classification_across_configurations.csv"))
print(cmp[["conventional_lowest_domain", "M0_class", "M1_class"]])
