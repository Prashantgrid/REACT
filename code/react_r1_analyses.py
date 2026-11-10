#!/usr/bin/env python3
"""Supporting analyses for the response to Reviewer 1 (REACT only).

Part "copula" (Comment 1.3): year-block bootstrap of hazard dependence and marginals, split-sample fits
(1990-2007, 2008-2024), trend adjustment to 2024 conditions, Gaussian instead of Student-t copula.
Part "params" (Comments 1.4, 1.5, 1.6, 1.9, 1.10, 1.1, 3.4): support-model parameters one at a time,
thermal capacity excluded, Dirichlet concentration, no derating (double counting), leave one
socio-economic indicator out, official DWD hot-day values, warming-delta assumptions.
Writes <out>/r1_<part>.csv (one row per variant x scenario x state) and r1_copula_fits.csv.
Part "noise": prediction-perturbation scale 0, 0.5 and 2 (main specification 1).
Usage: python react_r1_analyses.py <out_dir> copula|params|noise [n_boot]
"""
import os
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import REACT_v4 as core  # noqa: E402
import REACT_ESR_reproducible as R  # noqa: E402

OUT, PART = sys.argv[1], sys.argv[2]
N_BOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 30
N_PERM = 20
os.makedirs(OUT, exist_ok=True)
SC = R.SCENARIOS
COH = ("national_pool", "adjacency_flow")
LAB = R.SCENARIO_LABEL
HC = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean", "Rx1day_mm_max",
      "Storm_Days_mean", "low_wind_days_mean"]
static0 = R._configured_static()
hazard0 = core.load_hazard_panel()
rows, fits = [], []


def record(name, res, extra=None):
    for sc, r in res.items():
        for s in core.STATE_ORDER:
            d = dict(variant=name, scenario=LAB[sc], state=s, provisional=r["dominant"][s], P_sel=r["P_sel"][s],
                     T=r["T_joint"][s], identified=bool(r["identified"][s]), expected_rank=r["expected_rank"][s])
            if extra:
                d.update(extra)
            rows.append(d)
    ids = [f"{R.ABBR[s]}-{r['dominant'][s][0]}({LAB[sc][0]})" for sc, r in res.items() for s in core.STATE_ORDER
           if r["identified"][s]]
    print(f"  {name}: {ids}", flush=True)


def run(hazard=None, static=None, scen=SC, **spec):
    hz = hazard0 if hazard is None else hazard
    st = static0 if static is None else static
    R._COPULA_CACHE.clear()
    f = lambda: {sc: R.run_scenario(st, hz, sc, verbose=False, n_perm=N_PERM) for sc in scen}  # noqa: E731
    try:
        return R._with_spec(f, **spec) if spec else f()
    finally:
        R._COPULA_CACHE.clear()


def with_core(fn, **attrs):
    saved = {k: deepcopy(getattr(core, k)) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(core, k, v)
        return fn()
    finally:
        for k, v in saved.items():
            setattr(core, k, v)


def save():
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"r1_{PART}.csv"), index=False)
    if fits:
        pd.DataFrame(fits).to_csv(os.path.join(OUT, "r1_copula_fits.csv"), index=False)


if PART == "copula_tail":
    rows += pd.read_csv(os.path.join(OUT, "r1_copula.csv")).to_dict("records")
    fits += pd.read_csv(os.path.join(OUT, "r1_copula_fits.csv")).to_dict("records")
    PART = "copula"; TAIL = True
else:
    TAIL = False
if PART == "copula" and not TAIL:
    years = np.sort(hazard0.Year.unique())
    fit0 = R.estimate_copula(hazard0, HC)
    fits.append(dict(variant="main", model=fit0["model"], df=fit0["df"]))
    # 1. year-block bootstrap: years resampled with replacement, all 16 states of a year kept together
    rng = np.random.default_rng(2026)
    for b in range(N_BOOT):
        pick = rng.choice(years, size=len(years), replace=True)
        hz = pd.concat([hazard0[hazard0.Year == y].assign(Year=1990 + k) for k, y in enumerate(pick)],
                       ignore_index=True)
        R._COPULA_CACHE.clear(); f = R.estimate_copula(hz, HC)
        fits.append(dict(variant=f"boot_{b:02d}", model=f["model"], df=f["df"]))
        record(f"boot_{b:02d}", run(hz), dict(group="bootstrap"))
        save()
if PART == "copula":
    # 2. split-sample fits
    done_ = {r["variant"] for r in rows}
    for name, (a, z) in (("split_1990_2007", (1990, 2007)), ("split_2008_2024", (2008, 2024))):
        if name in done_:
            continue
        hz = hazard0[(hazard0.Year >= a) & (hazard0.Year <= z)].copy()
        R._COPULA_CACHE.clear(); f = R.estimate_copula(hz, HC)
        fits.append(dict(variant=name, model=f["model"], df=f["df"]))
        record(name, run(hz), dict(group="split"))
if PART == "copula":
    # 3. trend adjustment: Theil-Sen trend per state and hazard removed, record shifted to 2024 level
    hz = hazard0.copy(); trend_rows = []
    for s in core.STATE_ORDER:
        m = hz.State == s
        for c in HC:
            y, x = hz.loc[m, c].to_numpy(float), hz.loc[m, "Year"].to_numpy(float)
            ok = ~np.isnan(y)
            slope = theilslopes(y[ok], x[ok])[0]; tau, p = kendalltau(x[ok], y[ok])
            trend_rows.append(dict(state=s, hazard=c, slope_per_decade=10 * slope, kendall_tau=tau, p=p))
            hz.loc[m, c] = np.where(ok, np.maximum(y + slope * (2024 - x), 0.0), np.nan)
    pd.DataFrame(trend_rows).to_csv(os.path.join(OUT, "r1_hazard_trends.csv"), index=False)
    record("trend_adjusted_2024", run(hz), dict(group="trend"))
    # 4. Gaussian copula instead of Student-t
    orig = R.estimate_copula

    def gaussian(hz_, hc_):
        f = dict(orig(hz_, hc_)); f["model"] = "gaussian"; f["df"] = None
        return f
    R.estimate_copula = gaussian
    try:
        record("gaussian_copula", run(), dict(group="copula_family"))
    finally:
        R.estimate_copula = orig
    save()

elif PART == "params":
    # 1.4 support-model parameters, one at a time (coherent scenarios only: Baseline has no support layer)
    cc = core.CAPACITY_CREDIT
    oat = [("credit_wind_0.05", dict(CAPACITY_CREDIT={**cc, "Wind": 0.05})),
           ("credit_wind_0.20", dict(CAPACITY_CREDIT={**cc, "Wind": 0.20})),
           ("credit_pv_0.00", dict(CAPACITY_CREDIT={**cc, "Solar": 0.00})),
           ("credit_pv_0.10", dict(CAPACITY_CREDIT={**cc, "Solar": 0.10})),
           ("credit_thermal_0.70", dict(CAPACITY_CREDIT={**cc, "Thermal": 0.70})),
           ("credit_thermal_0.95", dict(CAPACITY_CREDIT={**cc, "Thermal": 0.95})),
           ("storage_window_2h", dict(H_DIS=2.0)), ("storage_window_8h", dict(H_DIS=8.0)),
           ("flow_scale_0.125", dict(FLOW_SCALE=0.125)), ("flow_scale_0.50", dict(FLOW_SCALE=0.50)),
           ("penalty_cap_0.25", dict(PEN_CAP=0.25)), ("penalty_cap_1.00", dict(PEN_CAP=1.00)),
           ("af_loss_0.50", dict(AF_SUPPORT_LOSS=0.50, AF_GRID_LOSS=0.20)),
           ("af_loss_0.90", dict(AF_SUPPORT_LOSS=0.90, AF_GRID_LOSS=0.40))]
    for name, attrs in oat:
        record(name, with_core(lambda: run(scen=COH), **attrs), dict(group="support"))
    save()
    # 1.6 Dirichlet concentration
    for a in (0.5, 2.0, 5.0):
        record(f"dirichlet_{a:g}", with_core(lambda: run(), DIRICHLET_ALPHA=a), dict(group="weights"))
    save()
    # 1.9 double counting: no derating of Energy System indicators
    zero = {"PV": dict(beta=0.0, delta_max=0.0), "Thermal": dict(alpha=0.0, beta=0.0, delta_max=0.0),
            "Grid": dict(beta=0.0, delta_max=0.0, gamma=0.0), "Wind": dict(beta=0.0, delta_max=0.0)}
    record("no_derating", run(derating=zero), dict(group="double_counting"))
    # 1.5 thermal capacity excluded from the Energy System domain
    b = deepcopy(core.BLOCK_INDICATORS); b["Energy"] = [n for n in b["Energy"] if n != "Thermal_MW"]
    record("thermal_excluded", run(blocks=b), dict(group="thermal"))
    save()
    # 1.10 leave one socio-economic indicator out
    for n in ("GDP_pc", "HDI", "Patents", "ADR"):
        b = deepcopy(core.BLOCK_INDICATORS); b["Adaptive"] = [k for k in b["Adaptive"] if k != n]
        record(f"without_{n}", run(blocks=b), dict(group="socio_years"))
    save()
    # 1.1 official DWD regional hot-day averages replace the panel values (13 territorial states)
    dwd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dwd_regional_averages_txbs_year.txt")
    if os.path.exists(dwd_path):
        d = pd.read_csv(dwd_path, sep=";", skiprows=1).rename(columns={"Jahr": "Year"})
        d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
        name_map = {"Baden-Wuerttemberg": "Baden-Württemberg", "Thueringen": "Thüringen"}
        hz = hazard0.copy(); cmp_rows = []
        for col in d.columns[2:]:
            s = name_map.get(col, col)
            if s not in core.STATE_ORDER:
                continue
            for _, r_ in d.iterrows():
                y = int(r_["Year"])
                m = (hz.State == s) & (hz.Year == y)
                if m.any():
                    cmp_rows.append(dict(state=s, year=y, panel=float(hz.loc[m, "hot_days_mean"].iloc[0]),
                                         dwd=float(r_[col])))
                    hz.loc[m, "hot_days_mean"] = float(r_[col])
        pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT, "r1_dwd_hotdays_comparison.csv"), index=False)
        record("dwd_hot_days", run(hz), dict(group="observed_inputs"))
        save()
    # 3.4 warming-delta assumptions at +2 K (low-wind 0 or 4 %/K; heavy-precipitation frequency 10 %/K)
    from react_pf import climate_delta as CD
    cv = pd.read_csv(os.path.join(OUT, "hot_day_response_curves.csv"), index_col=0)
    shifts, E = cv.index.to_numpy(float), cv[core.STATE_ORDER].to_numpy(float)
    fit = R.estimate_copula(hazard0, HC); ranges = CD.historical_ranges(hazard0, fit)

    def delta(h, dT, lw=0.02, p30=1.25):
        h = h.copy()
        for i, s in enumerate(core.STATE_ORDER):
            m = h.State == s
            x = h.loc[m, "hot_days_mean"].to_numpy(float)
            e = np.maximum.accumulate(E[:, i])
            sh = np.interp(np.maximum(x, e[0] + 1e-6), e, shifts)
            h.loc[m, "hot_days_mean"] = np.interp(sh + dT, shifts, e)
        h["Rx1day_mm_mean"] *= 1 + 0.07 * dT; h["Rx1day_mm_max"] *= 1 + 0.07 * dT
        h["p30_days_mean"] *= p30 ** dT
        h["low_wind_days_mean"] = np.minimum(h["low_wind_days_mean"] * (1 + lw * dT), 365)
        return h
    for name, kw in (("delta2K_lowwind_0", dict(lw=0.0)), ("delta2K_lowwind_4", dict(lw=0.04)),
                     ("delta2K_p30_10", dict(p30=1.10)), ("delta2K_reference", dict())):
        hz = delta(hazard0, 2.0, **kw)
        record(name, with_core(lambda: run(hz), HAZARD_RANGE_OVERRIDE=ranges), dict(group="warming"))
        save()
elif PART == "noise":
    # scale of the prediction perturbation added to the PRESS residuals (main specification: 1)
    saved = R.REGRESSION_NOISE_SCALE
    try:
        for s in (0.0, 0.5, 2.0):
            R.REGRESSION_NOISE_SCALE = s
            record(f"noise_{s:g}", run(), dict(group="prediction_error"))
            save()
    finally:
        R.REGRESSION_NOISE_SCALE = saved
print("done")
