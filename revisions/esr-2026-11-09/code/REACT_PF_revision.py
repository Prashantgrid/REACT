#!/usr/bin/env python3
"""Revision analyses built on REACT-Lite and REACT-PF (run after REACT_PF.py).

  --mode climate      REACT-Lite under stylised +1 K / +2 K warming deltas (Reviewer 3.4)
  --mode sensitivity  REACT-PF MCDA under alternative KPI specifications (value function of the
                      deliverability ratio, N-0 instead of N-1, local firm capacity added,
                      neighbour support halved, forward grid with planned HVDC corridors)
  --mode all
Outputs are written to <pf_dir>/revision/.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import REACT_v4 as core
import REACT_ESR_reproducible as R
from react_pf import network as N, mcda as MC, climate_delta as CD, deliverability as D

SCEN = ("baseline", "national_pool", "adjacency_flow")
LABEL = {"baseline": "Baseline", "national_pool": "National Pool", "adjacency_flow": "Adjacency Flow"}


def climate(pf_dir, out, n_perm=200):
    model = pickle.load(open(os.path.join(pf_dir, "pf_model.pkl"), "rb"))
    hazard, static, wg = model["hazard"], R._configured_static(), model["wg"]  # corrected storage panel
    curve = CD.hot_day_response(wg)
    pd.DataFrame(curve[1], index=curve[0], columns=N.STATES).to_csv(os.path.join(out, "hot_day_response_curves.csv"))
    fit = R.estimate_copula(hazard, MC.HCOLS)
    ranges = CD.historical_ranges(hazard, fit)
    rows, panel_rows = [], []
    R.N_PERMUTATION = n_perm
    for dT in (0.0, 1.0, 2.0):
        hz = hazard if dT == 0 else CD.delta_panel(hazard, curve, dT)
        g = hz.groupby("State").mean(numeric_only=True)
        for s in N.STATES:
            panel_rows.append(dict(dT=dT, state=s, **{c: g.loc[s, c] for c in MC.HCOLS}))
        R._COPULA_CACHE.clear()
        core.HAZARD_RANGE_OVERRIDE = None if dT == 0 else ranges
        try:
            for sc in SCEN:
                t0 = time.time()
                r = R.run_scenario(static, hz, sc, verbose=False, n_perm=n_perm)
                for s in N.STATES:
                    rows.append(dict(dT=dT, scenario=LABEL[sc], state=s, provisional=r["dominant"][s],
                                     P_sel=r["P_sel"][s], T=r["T_joint"][s], identified=bool(r["identified"][s]),
                                     classification=r["classification"][s], expected_rank=r["expected_rank"][s]))
                print(f"  climate dT={dT:+.0f} {sc}: {[(s, c) for s, c in r['classification'].items() if c != 'Multidomain']}"
                      f" ({time.time() - t0:.0f} s)", flush=True)
        finally:
            core.HAZARD_RANGE_OVERRIDE = None
            R._COPULA_CACHE.clear()
    pd.DataFrame(panel_rows).to_csv(os.path.join(out, "climate_delta_hazard_means.csv"), index=False)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "climate_delta_classifications.csv"), index=False)
    return df


def climate_differential(pf_dir, out, n_real=12, dT=2.0, n_perm=50, seed=777):
    """Differential warming: state-specific scaling of the +2 K changes (Reviewer 3.4)."""
    model = pickle.load(open(os.path.join(pf_dir, "pf_model.pkl"), "rb"))
    hazard, static, wg = model["hazard"], R._configured_static(), model["wg"]  # corrected storage panel
    curve = CD.hot_day_response(wg)
    fit = R.estimate_copula(hazard, MC.HCOLS); ranges = CD.historical_ranges(hazard, fit)
    rng = np.random.default_rng(seed); rows = []
    R.N_PERMUTATION = n_perm
    for k in range(n_real):
        hz, f = CD.delta_panel_differential(hazard, curve, dT, rng)
        R._COPULA_CACHE.clear(); core.HAZARD_RANGE_OVERRIDE = ranges
        try:
            for sc in SCEN:
                r = R.run_scenario(static, hz, sc, verbose=False, n_perm=n_perm)
                for s in N.STATES:
                    rows.append(dict(realisation=k, scenario=LABEL[sc], state=s, provisional=r["dominant"][s],
                                     P_sel=r["P_sel"][s], identified=bool(r["identified"][s]),
                                     classification=r["classification"][s], f_hot=f["hot"][s], f_p30=f["p30"][s]))
        finally:
            core.HAZARD_RANGE_OVERRIDE = None; R._COPULA_CACHE.clear()
        print(f"  differential warming realisation {k}: done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "climate_differential_classifications.csv"), index=False)
    return df


def _pf_variant(model, kt, tables, name, out, M, Mw, n_perm, **kw):
    rows = []
    for sc in SCEN:
        spec = MC.pf_energy_spec(kt, tables, sc, **kw)
        r = MC.run_react(model["static"], model["hazard"], sc, spec, M=M, Mw=Mw, n_perm=n_perm)
        d = r["diag"]
        for s in N.STATES:
            rows.append(dict(variant=name, scenario=LABEL[sc], state=s, provisional=d.provisional[s],
                             P_sel=d.P_sel[s], T=d["T"][s], identified=bool(d.identified[s]),
                             classification=d.classification[s], expected_rank=d.expected_rank[s]))
    print(f"  PF variant {name}: {[(r['state'], r['classification'], r['scenario'][:3]) for r in rows if r['identified']]}",
          flush=True)
    return rows


def sensitivity(pf_dir, out, M=2000, Mw=10000, n_perm=200, part=None):
    """part: None (all), 'a' (value-function and indicator-set variants), 'b' (security, neighbours,
    HVDC, grid reserve); the two parts can run in parallel and are merged afterwards."""
    model = pickle.load(open(os.path.join(pf_dir, "pf_model.pkl"), "rb"))
    kt = pd.read_csv(os.path.join(pf_dir, "pf_kpis.csv"), index_col=0)     # nodal structural KPIs
    tables = pickle.load(open(os.path.join(pf_dir, "pf_mc_tables.pkl"), "rb"))["mc"]
    rows = []
    variants = {"main": {}, "DR_sat_0.25": dict(dr_scale=0.25), "DR_sat_1.0": dict(dr_scale=1.0),
                "DR_cap_3": dict(dr_mode="cap", dr_cap=3.0), "DR_log": dict(dr_mode="log"),
                "DR_linear": dict(dr_mode="linear"), "with_local_firm": dict(local_firm=True),
                "without_EENS_LOLE": dict(drop=("EENS_ppm", "LOLE_h")),
                "without_storm": dict(drop=("Storm_loss_h",))}
    if part in (None, "a"):
        for name, kw in variants.items():
            rows += _pf_variant(model, kt, tables, name, out, M, Mw, n_perm, **kw)
    if part == "a":
        pd.DataFrame(rows).to_csv(os.path.join(out, "pf_sensitivity_part_a.csv"), index=False)
        return
    # N-0 deliverability instead of N-1
    k0 = kt.copy(); k0["DR_winter_N1"] = kt.DR_winter_N0
    rows += _pf_variant(model, k0, tables, "DR_N0", out, M, Mw, n_perm)
    # structural reruns (nodal): neighbours halved, planned HVDC corridors, grid-reserve plants included
    g, fl = model["grid"], model["fleet"]
    extra = {}
    for name in ("neighbours_x0.5", "hvdc_2028", "with_grid_reserve", "load_nearest_substation", "load_spread_5"):
        if name == "with_grid_reserve":
            # powerplantmatching thermal totals (market plants plus grid reserve) instead of REACT's
            # market-active totals; record both so that the text can state the difference
            fl2 = N.build_fleet(g, model["static"], with_grid_reserve=True)
            pd.DataFrame({"REACT_MW": fl.cap[N.THERMAL].sum(1), "ppm_MW": fl2.cap[N.THERMAL].sum(1)}).to_csv(
                os.path.join(out, "thermal_totals_variant.csv"))
        elif name == "load_nearest_substation":
            fl2 = N.build_fleet(g, model["static"], load_spread_k=1)
        elif name == "load_spread_5":
            fl2 = N.build_fleet(g, model["static"], load_spread_k=5)
        else:
            fl2 = copy.deepcopy(fl)
        scale = 0.5 if name == "neighbours_x0.5" else 1.0
        if name == "hvdc_2028":
            N.add_hvdc(g, fl2)
        dw = D.deliverable_supply_nodal(g, fl2, "winter_dunkelflaute", n1=True, scale_nb=scale)
        dh = D.deliverable_supply_nodal(g, fl2, "summer_heatwave", n1=True, scale_nb=scale)
        k2 = kt.copy(); k2["DR_winter_N1"] = dw.DR; k2["DR_heat_N1"] = dh.DR
        extra[name] = pd.DataFrame({"DR_winter_N1": dw.DR, "DR_heat_N1": dh.DR, "import_N1_MW": dw.import_MW,
                                    "lambda_sys_winter": dw.system_loadability})
        print(f"  structural variant {name}: lambda_sys winter {dw.system_loadability.iloc[0]:.3f}", flush=True)
        rows += _pf_variant(model, k2, tables, name, out, M, Mw, n_perm)
    pd.concat(extra, axis=1).to_csv(os.path.join(out, "deliverability_variants.csv"))
    pa = os.path.join(out, "pf_sensitivity_part_a.csv")
    if part == "b" and os.path.exists(pa):
        rows = pd.read_csv(pa).to_dict("records") + rows
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "pf_sensitivity_classifications.csv"), index=False)
    return df


def lite_physical(pf_dir, out, n_perm=200):
    """REACT-Lite Adjacency Flow with physical N-1 corridor ratings instead of Eq. (A.9) (R1.4, R3.2)."""
    model = pickle.load(open(os.path.join(pf_dir, "pf_model.pkl"), "rb"))
    br = model["grid"].branches
    cap = np.zeros((16, 16))
    for (a, b), gb in br[br.interstate].groupby(["z0", "z1"]):
        i, j = core.STATE_ORDER.index(a), core.STATE_ORDER.index(b)
        cap[i, j] += gb.s_nom.sum(); cap[j, i] = cap[i, j]
    largest = np.zeros((16, 16))
    for r in br[br.interstate].itertuples():
        i, j = core.STATE_ORDER.index(r.z0), core.STATE_ORDER.index(r.z1)
        largest[i, j] = largest[j, i] = max(largest[i, j], r.s_nom / r.circuits)
    n1 = np.where(cap > 0, cap - largest, 0.0)
    rows = []
    R.N_PERMUTATION = n_perm
    for name, M_ in (("Eq_A9", None), ("physical_N1", n1), ("physical_N1_x0.5", 0.5 * n1)):
        core.CORRIDOR_CAP_OVERRIDE = M_
        try:
            r = R.run_scenario(R._configured_static(), model["hazard"], "adjacency_flow", verbose=False, n_perm=n_perm)
        finally:
            core.CORRIDOR_CAP_OVERRIDE = None
        for s in N.STATES:
            rows.append(dict(variant=name, state=s, provisional=r["dominant"][s], P_sel=r["P_sel"][s],
                             T=r["T_joint"][s], identified=bool(r["identified"][s]),
                             classification=r["classification"][s], expected_rank=r["expected_rank"][s]))
        print(f"  Lite AF corridors={name}: {[(s, c) for s, c in r['classification'].items() if c != 'Multidomain']}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "lite_adjacency_physical_corridors.csv"), index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pf_dir"); p.add_argument("--mode", choices=("climate", "climate_diff", "sensitivity", "lite_physical", "all"), default="all")
    p.add_argument("--mc", type=int, default=2000); p.add_argument("--diagnostic", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--part", choices=("a", "b"), default=None)
    a = p.parse_args()
    out = os.path.join(a.pf_dir, "revision"); os.makedirs(out, exist_ok=True)
    if a.mode in ("climate", "all"):
        climate(a.pf_dir, out, a.permutations)
    if a.mode in ("climate_diff", "all"):
        climate_differential(a.pf_dir, out)
    if a.mode in ("lite_physical", "all"):
        lite_physical(a.pf_dir, out, a.permutations)
    if a.mode in ("sensitivity", "all"):
        sensitivity(a.pf_dir, out, a.mc, a.diagnostic, a.permutations, part=a.part)


if __name__ == "__main__":
    main()
