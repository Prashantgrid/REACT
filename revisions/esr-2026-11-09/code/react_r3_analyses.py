#!/usr/bin/env python3
"""Supporting analyses for the response to Reviewer 3 (REACT only, no power-flow model).

1. What drives Hamburg's Energy System score (indicator profile).
2. Which Energy System indicators carry the inverse relation with Socio-economic Capacity.
3. How much of each domain the diagnostic expects from the other two (leave-one-out prediction).
4. Energy System specification variants, three scenarios each, full settings:
     no_import_penalty   Energy domain without the external-support score S(r_i)
     local_supply_only   wind, PV, thermal, storage relative to load
     network_access_only grid density and interstate connectivity
     absolute_capacities installed capacities in absolute terms (asset-stock construct)
Writes to <out_dir>: hamburg_energy_profile.csv, energy_indicator_vs_socio.csv,
loo_domain_prediction.csv, r3_variants.csv, r3_variant_summary.csv.
Usage: python react_r3_analyses.py <out_dir> [n_perm]
"""
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import REACT_v4 as core  # noqa: E402
import REACT_ESR_reproducible as R  # noqa: E402

OUT = sys.argv[1]
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
os.makedirs(OUT, exist_ok=True)
SC = R.SCENARIOS
LAB = R.SCENARIO_LABEL
static0 = R._configured_static()
hazard = core.load_hazard_panel()
R._configure_core()
ENERGY = list(core.BLOCK_INDICATORS["Energy"])
print("Energy indicators:", ENERGY, flush=True)


def run3(static, blocks=None):
    def f():
        return {sc: R.run_scenario(static, hazard, sc, verbose=False, n_perm=N_PERM) for sc in SC}
    return R._with_spec(f, blocks=blocks) if blocks is not None else f()


# ---------------------------------------------------------------------------------------------
main = run3(static0)
panel = core.build_baseline_panel(static0, hazard)
B = main["baseline"]["B_blocks"]                       # mean domain scores, Baseline

# 1. Hamburg's Energy System indicator profile (baseline structural panel)
rows = []
for n in ENERGY:
    x = panel[n].astype(float)
    cost = core.IS_COST[n]
    z = (x - x.min()) / (x.max() - x.min())
    z = 1 - z if cost else z
    rows.append(dict(indicator=n, hamburg_value=x["Hamburg"], panel_median=x.median(),
                     hamburg_normalised=z["Hamburg"], hamburg_rank_of_16=int(z.rank(ascending=False)["Hamburg"])))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "hamburg_energy_profile.csv"), index=False)

# 2. Correlation of each Energy indicator with the Socio-economic domain score
rows = []
for n in ENERGY:
    x = panel[n].astype(float)
    x = -x if core.IS_COST[n] else x
    rows.append(dict(indicator=n, pearson_r=np.corrcoef(x, B["Adaptive"])[0, 1],
                     spearman=spearmanr(x, B["Adaptive"]).statistic))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "energy_indicator_vs_socio.csv"), index=False)

# 3. Leave-one-out prediction of each domain from the other two (mean Baseline domain scores)
rows = []
cols = ["Climate", "Energy", "Adaptive"]
Bv = B[cols].to_numpy(float)
for i, s in enumerate(B.index):
    for b, name in enumerate(cols):
        oth = [k for k in range(3) if k != b]
        keep = np.arange(len(Bv)) != i
        X = np.column_stack([np.ones(keep.sum()), Bv[keep][:, oth]])
        beta = np.linalg.lstsq(X, Bv[keep, b], rcond=None)[0]
        pred = beta[0] + Bv[i, oth] @ beta[1:]
        rows.append(dict(state=s, domain=name, observed=Bv[i, b], predicted_loo=pred, residual=Bv[i, b] - pred,
                         median_z=main["baseline"]["Z_resid"].loc[s, name]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "loo_domain_prediction.csv"), index=False)
print("profile, correlations, LOO prediction written", flush=True)

# 4. Variants
def blocks_with(energy):
    b = deepcopy(core.BLOCK_INDICATORS); b["Energy"] = energy; return b


load = R.average_load_mw(static0)
hours = static0.loc[core.STATE_ORDER, "Storage_MWh"].astype(float).to_numpy() / load
nz = hours[hours > 0]
zero_states = [s for s, h in zip(core.STATE_ORDER, hours) if h == 0]


def static_storage_fill(h_fill):
    st = static0.copy()
    for i, s in enumerate(core.STATE_ORDER):
        if s in zero_states:
            st.loc[s, "Storage_MWh"] = h_fill * load[i]
    return st


variants = {
    "main": (static0, None),
    "no_import_penalty": (static0, blocks_with([n for n in ENERGY if n != "R_ri"])),
    "local_supply_only": (static0, blocks_with(["Wind_MW", "Solar_MW", "Thermal_MW", "Storage_MWh"])),
    "network_access_only": (static0, blocks_with(["WLINES", "C_i"])),
    "absolute_capacities": (static0, "absolute"),
}
rows, summ = [], []
for name, (st, blocks) in variants.items():
    if name == "main":
        res = main
    elif blocks == "absolute":
        res = R._with_spec(lambda: {sc: R.run_scenario(st, hazard, sc, verbose=False, n_perm=N_PERM) for sc in SC},
                           construct="absolute")
    else:
        res = run3(st, blocks)
    Bb = res["baseline"]["B_blocks"]
    r_ea = np.corrcoef(Bb["Energy"], Bb["Adaptive"])[0, 1]
    ids = []
    for sc in SC:
        r = res[sc]
        rho = spearmanr(r["expected_rank"], main[sc]["expected_rank"]).statistic
        for s in core.STATE_ORDER:
            rows.append(dict(variant=name, scenario=LAB[sc], state=s, provisional=r["dominant"][s],
                             P_sel=r["P_sel"][s], T=r["T_joint"][s], identified=bool(r["identified"][s]),
                             expected_rank=r["expected_rank"][s], rank_rho_vs_main=rho))
            if r["identified"][s]:
                ids.append(f"{R.ABBR[s]}-{r['dominant'][s][0]}({LAB[sc][0]})")
    hh = {LAB[sc]: (res[sc]["dominant"]["Hamburg"], float(res[sc]["P_sel"]["Hamburg"]),
                    bool(res[sc]["identified"]["Hamburg"])) for sc in SC}
    summ.append(dict(variant=name, r_energy_socio=r_ea, n_identified=len(ids), identified="; ".join(ids),
                     hamburg=str(hh), fill_h=(float(nz.min()) if name == "storage_fill_low" else
                                              float(np.median(nz)) if name == "storage_fill_median" else np.nan)))
    print(f"  {name}: r(E,A)={r_ea:.2f}; {len(ids)} identified: {ids}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "r3_variants.csv"), index=False)
pd.DataFrame(summ).to_csv(os.path.join(OUT, "r3_variant_summary.csv"), index=False)
print("zero-storage states:", zero_states, "fill low/median (h):", nz.min(), np.median(nz))
print("done")
