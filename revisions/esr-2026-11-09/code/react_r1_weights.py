#!/usr/bin/env python3
"""Reviewer 1.6: asymmetric domain-weight profiles (Dirichlet mean 0.5 on one domain, 0.25 on the others,
concentration 8). Appends to <out>/r1_weights.csv. Usage: python react_r1_weights.py <out_dir>"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import REACT_v4 as core  # noqa: E402
import REACT_ESR_reproducible as R  # noqa: E402

OUT = sys.argv[1]
st, hz = R._configured_static(), core.load_hazard_panel()
rows = []
for name, a in (("climate_emphasis", [4.0, 2.0, 2.0]), ("energy_emphasis", [2.0, 4.0, 2.0]),
                ("socio_emphasis", [2.0, 2.0, 4.0])):
    core.DOMAIN_ALPHA = a
    try:
        for sc in R.SCENARIOS:
            r = R.run_scenario(st, hz, sc, verbose=False, n_perm=20)
            for s in core.STATE_ORDER:
                rows.append(dict(variant=name, group="weights", scenario=R.SCENARIO_LABEL[sc], state=s,
                                 provisional=r["dominant"][s], P_sel=r["P_sel"][s], T=r["T_joint"][s],
                                 identified=bool(r["identified"][s]), expected_rank=r["expected_rank"][s]))
    finally:
        core.DOMAIN_ALPHA = None
    x = pd.DataFrame(rows); x = x[(x.variant == name) & x.identified]
    print(name, list(zip(x.state, x.provisional, x.scenario)), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "r1_weights.csv"), index=False)
print("done")
