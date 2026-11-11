"""REACT-PF resilience KPIs per state (Energy System domain of the full model).

Six indicators, each with a physical definition (direction in brackets):
  E1 EENS_ppm       expected energy not served / demand, hazard ensemble, current system   [cost]
  E2 LOLE_h         expected loss-of-load hours per year, same ensemble                     [cost]
  E3 DR_winter      N-1 secure deliverability ratio, winter Dunkelflaute snapshot (cap 3)   [benefit]
  E4 DR_heat        N-1 secure deliverability ratio, summer heatwave snapshot (cap 3)       [benefit]
  E5 Storm_loss_h   unserved energy over restoration after a compound grid-damage event,
                    hours of average load                                                   [cost]
  E6 Exposure       share of national shortfall borne by the state relative to its demand
                    share when the system is stressed to a reliability-standard level
                    (national LOLE ~ 3 h/a); 1 = pro-rata, > 1 = transmission-induced exposure [cost]
  (Local_firm: local firm capacity / peak load, reported for comparison with REACT-Lite only)
E1, E2 and E6 depend on the spatial-support scenario; E3-E5 are network properties.
Deliverability ratios enter through a saturating value function (mcda.dr_value).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import pandas as pd

from .network import STATES, ABBR
from . import adequacy as A
from . import deliverability as D

PF_INDICATORS = [("EENS_ppm", "cost"), ("LOLE_h", "cost"), ("DR_winter", "benefit"), ("DR_heat", "benefit"),
                 ("Storm_loss_h", "cost"), ("Exposure", "cost")]
# Local firm capacity (E7) is reported but not used in the PF Energy domain: in REACT-PF the role of
# local resources is already contained in E3-E5 (deliverable supply includes local resources;
# grid-damage losses depend on them). It is the asset-stock construct of REACT-Lite.
DR_CAP = 3.0


def structural_kpis(grid, fleet, out_dir, storm_samples=30, storm=True):
    t0 = time.time()
    res = {}
    for snap, n1 in (("winter_dunkelflaute", False), ("winter_dunkelflaute", True), ("summer_heatwave", True),
                     ("summer_nominal", True)):
        res[(snap, n1)] = D.deliverable_supply(grid, fleet, snap, n1=n1)
    st_ = D.storm_event_test(grid, fleet, n_samples=storm_samples) if storm else None
    lf = D.local_firm_ratio(fleet)
    df = pd.DataFrame(index=STATES)
    df["peak_MW"] = res[("winter_dunkelflaute", True)].peak_MW
    df["DS_winter_N0_MW"] = res[("winter_dunkelflaute", False)].DS_MW
    df["DS_winter_N1_MW"] = res[("winter_dunkelflaute", True)].DS_MW
    df["IC_winter_N0_MW"] = res[("winter_dunkelflaute", False)].import_capability_MW
    df["IC_winter_N1_MW"] = res[("winter_dunkelflaute", True)].import_capability_MW
    df["DR_winter_N0"] = res[("winter_dunkelflaute", False)].DR
    df["DR_winter_N1"] = res[("winter_dunkelflaute", True)].DR
    df["DR_heat_N1"] = res[("summer_heatwave", True)].DR
    df["DR_summer_nominal_N1"] = res[("summer_nominal", True)].DR
    df["heat_retention"] = df.DR_heat_N1 / df.DR_summer_nominal_N1
    df["Local_firm"] = lf
    if st_ is not None:
        for c in st_.columns:
            df[f"storm_{c}"] = st_[c]
    df.to_csv(os.path.join(out_dir, "pf_structural_kpis.csv"))
    print(f"  structural KPIs done in {time.time() - t0:.0f} s", flush=True)
    return df


# ------------------------------------------------------------------------------------------------
# Monte Carlo (parallel over years)
# ------------------------------------------------------------------------------------------------
_G = {}


def _init(payload):
    _G.update(pickle.loads(payload))


def _one_year(args):
    scenario, seed, thermal_scale, p_fail = args
    am = _G.setdefault(("am", scenario, thermal_scale, p_fail), None)
    if am is None:
        am = A.AdequacyModel(_G["grid"], _G["fleet"], _G["wg"], _G["fit"], _G["hazard"], scenario, _G["ic"],
                             p_fail=p_fail)
        am.thermal_scale = thermal_scale
        _G[("am", scenario, thermal_scale, p_fail)] = am
    r = am.simulate_year(np.random.default_rng(seed))
    return dict(seed=seed, eens=r["eens"], lole=r["lole"], eens_event=r["eens_event"], demand=r["demand"],
                lole_nat=r["lole_nat"], eens_nat=r["eens_nat"], n_days=r["n_days"], lp_fail=r["lp_fail"],
                n_events=len(r["events"]), draw=r["draw"])


def run_mc(payload, scenario, n_years, seed0, thermal_scale=1.0, p_fail=0.20, processes=2):
    args = [(scenario, seed0 + k, thermal_scale, p_fail) for k in range(n_years)]
    with mp.get_context("fork").Pool(processes, initializer=_init, initargs=(payload,)) as pool:
        out = pool.map(_one_year, args, chunksize=max(1, n_years // (4 * processes)))
    return out


def mc_table(out):
    eens = np.array([o["eens"] for o in out]); lole = np.array([o["lole"] for o in out])
    dem = np.array([o["demand"] for o in out]); ev = np.array([o["eens_event"] for o in out])
    return dict(eens=eens, lole=lole, demand=dem, eens_event=ev,
                lole_nat=np.array([o["lole_nat"] for o in out]), eens_nat=np.array([o["eens_nat"] for o in out]),
                n_days=np.array([o["n_days"] for o in out]), lp_fail=np.array([o["lp_fail"] for o in out]),
                n_events=np.array([o["n_events"] for o in out]))


def summarise_mc(tab, label):
    e = tab["eens"].mean(0); d = tab["demand"].mean(0)
    df = pd.DataFrame(index=STATES)
    df["EENS_MWh"] = e; df["EENS_ppm"] = 1e6 * e / d; df["LOLE_h"] = tab["lole"].mean(0)
    df["EENS_event_share"] = np.where(e > 0, tab["eens_event"].mean(0) / np.maximum(e, 1e-9), np.nan)
    df["years"] = len(tab["eens"])
    df["scenario"] = label
    return df


def exposure_ratio(tab):
    """Share of national unserved energy borne by each state / its demand share."""
    e = tab["eens"].sum(0); d = tab["demand"].sum(0)
    if e.sum() <= 0:
        return pd.Series(1.0, index=STATES)
    return pd.Series((e / e.sum()) / (d / d.sum()), index=STATES)


def structural_kpis_nodal(grid, fleet, out_dir, storm_samples=30, tag="", scale_nb=1.0, storm=True):
    """Nodal (optimal intra-state redispatch) version of the structural KPIs; main specification."""
    t0 = time.time()
    res = {}
    for snap, n1 in (("winter_dunkelflaute", False), ("winter_dunkelflaute", True), ("summer_heatwave", True),
                     ("summer_nominal", True)):
        res[(snap, n1)] = D.deliverable_supply_nodal(grid, fleet, snap, n1=n1, scale_nb=scale_nb)
        print(f"    nodal {snap} N-{int(n1)}: lambda_sys={res[(snap, n1)].system_loadability.iloc[0]:.3f}", flush=True)
    lf = D.local_firm_ratio(fleet)
    df = pd.DataFrame(index=STATES)
    w1 = res[("winter_dunkelflaute", True)]
    df["peak_MW"] = w1.peak_MW
    df["DS_winter_N0_MW"] = res[("winter_dunkelflaute", False)].DS_MW
    df["DS_winter_N1_MW"] = w1.DS_MW
    df["IC_winter_N0_MW"] = res[("winter_dunkelflaute", False)].import_MW
    df["IC_winter_N1_MW"] = w1.import_MW
    df["DR_winter_N0"] = res[("winter_dunkelflaute", False)].DR
    df["DR_winter_N1"] = w1.DR
    df["DR_heat_N1"] = res[("summer_heatwave", True)].DR
    df["DR_summer_nominal_N1"] = res[("summer_nominal", True)].DR
    df["heat_retention"] = df.DR_heat_N1 / df.DR_summer_nominal_N1
    df["Local_firm"] = lf
    for (snap, n1), r in res.items():
        df[f"lambda_sys_{snap}_N{int(n1)}"] = r.system_loadability.iloc[0]
    if storm:
        st = D.storm_event_test_nodal(grid, fleet, n_samples=storm_samples)
        for c in st.columns:
            df[f"storm_{c}"] = st[c]
    df.to_csv(os.path.join(out_dir, f"pf_structural_kpis_nodal{tag}.csv"))
    print(f"  nodal structural KPIs{tag} done in {time.time() - t0:.0f} s", flush=True)
    return df
