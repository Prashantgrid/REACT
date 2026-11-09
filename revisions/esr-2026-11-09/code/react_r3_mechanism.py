#!/usr/bin/env python3
"""Load density as a mechanism of the inverse Energy System / Socio-economic relation (Reviewer 3.2).
Writes <results_dir>/r3/load_density_mechanism.csv. Usage: python react_r3_mechanism.py <results_dir>"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import REACT_v4 as core  # noqa: E402
import REACT_ESR_reproducible as R  # noqa: E402

RES = sys.argv[1]
st = R._configured_static().loc[core.STATE_ORDER]
area = np.array([R.STATE_AREA_KM2[s] for s in core.STATE_ORDER], float)
load = R.average_load_mw(st)
ell = load / area                                        # load density, MW per km2
kap = (st.Wind_Onshore_MW + st.Solar_PV_MW).to_numpy(float) / area   # wind + PV per km2
ds = pd.read_csv(os.path.join(RES, "domain_scores_baseline.csv"), index_col=0).loc[core.STATE_ORDER]
X = np.column_stack([np.ones(16), np.log(ell)])
res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]  # noqa: E731
terr = ~np.isin(core.STATE_ORDER, ["Hamburg", "Berlin", "Bremen"])
out = dict(load_density_ratio=ell.max() / ell.min(), ren_area_ratio=kap.max() / kap.min(),
           sd_log_load_density=np.log(ell).std(), sd_log_ren_area=np.log(kap).std(),
           corr_logKL_neglogell=np.corrcoef(np.log(kap / ell), -np.log(ell))[0, 1],
           rho_E_ell=spearmanr(ell, ds.Energy).statistic, rho_S_ell=spearmanr(ell, ds.Adaptive).statistic,
           partial_r_ES_given_logell=np.corrcoef(res(ds.Energy.to_numpy()), res(ds.Adaptive.to_numpy()))[0, 1],
           r_ES_territorial=np.corrcoef(ds.Energy[terr], ds.Adaptive[terr])[0, 1])
pd.Series(out).to_csv(os.path.join(RES, "r3", "load_density_mechanism.csv"), header=["value"])
print(pd.Series(out).round(3))
