#!/usr/bin/env python3
"""REACT-PF: power-flow-based resilience KPIs as the Energy System domain of REACT.

REACT is released in two tiers that share the hazard model, the Climate Exposure and
Socio-economic Capacity domains, the weighting scheme and the relative-weakness diagnostic:

  REACT-Lite  (REACT_ESR_reproducible.py)  Energy System = structural indicators relative to load
              plus the reduced support layer (S(r_i), C_i, Eq. A.9). Data-light screening tier.
  REACT-PF    (this file)                  Energy System = seven performance KPIs from a nodal
              220/380 kV DC power-flow model (react_pf/): N-1 deliverability, chronological
              adequacy Monte Carlo under the REACT hazard ensemble, grid-damage restoration,
              reliability-standard stress exposure, and local firm capacity.

Usage
-----
  python REACT_PF.py results_pf --mode all          # structural + MC + MCDA + comparisons
  python REACT_PF.py results_pf --mode kpis         # power-flow KPIs only
  python REACT_PF.py results_pf --mode mcda         # MCDA from saved KPIs
  python REACT_PF.py results_pf --fast              # smoke test (few years, small M)
Options: --years N (MC years per scenario), --stress-years N, --processes P, --weather-dir DIR
(historical replay from external hourly data, see react_pf/weather.py).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import REACT_v4 as core
import REACT_ESR_reproducible as R
from react_pf import network as N, weather as W, deliverability as D, kpis as K, mcda as MC, adequacy as A

SCEN = ("baseline", "national_pool", "adjacency_flow")
LABEL = {"baseline": "Baseline", "national_pool": "National Pool", "adjacency_flow": "Adjacency Flow"}
STRESS_SCALE = 0.60     # thermal fleet scale giving national LOLE ~ 3 h/a in the copper plate


def build(out_dir, weather_dir=None):
    path = os.path.join(out_dir, "pf_model.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    t0 = time.time()
    hazard = core.load_hazard_panel(); static = R._configured_static()
    grid = N.build_grid(); fleet = N.build_fleet(grid, static); N.mark_cnecs(grid, fleet)
    polys = N.state_polygons(); cen = N.state_centroids_km(polys)
    wg = W.ExternalWeather(weather_dir) if weather_dir else W.WeatherGenerator(cen, hazard)
    fit = R.estimate_copula(hazard, MC.HCOLS)
    model = dict(grid=grid, fleet=fleet, wg=wg, fit=fit, hazard=hazard, static=static, centroids=cen)
    pickle.dump(model, open(path, "wb"))
    print(f"[build] {grid.notes[0]}; fleet notes: {fleet.notes}; {time.time() - t0:.0f} s", flush=True)
    return model


def write_network_tables(model, out_dir):
    g, fl = model["grid"], model["fleet"]
    br = g.branches.copy()
    cor = br[br.interstate].copy()
    cor["pair"] = [" - ".join(sorted((N.ABBR[a], N.ABBR[b]))) for a, b in zip(cor.z0, cor.z1)]
    cor["circuit_MVA"] = cor.s_nom / cor.circuits
    tab = cor.groupby("pair").agg(circuits=("circuits", "sum"), branches=("s_nom", "size"),
                                  thermal_MVA=("s_nom", "sum"),
                                  largest_circuit_MVA=("circuit_MVA", "max")).sort_values("thermal_MVA", ascending=False)
    tab["N1_MVA"] = tab.thermal_MVA - tab.largest_circuit_MVA          # single-circuit N-1
    tab.to_csv(os.path.join(out_dir, "pf_interstate_corridors.csv"))
    caps = fl.cap.copy(); caps[["PHS_MW", "BAT_MW"]] = fl.stor_p.values; caps[["PHS_MWh", "BAT_MWh"]] = fl.stor_e.values
    caps["annual_TWh"] = fl.annual_twh
    caps.to_csv(os.path.join(out_dir, "pf_fleet_by_state.csv"))
    # REACT-Lite Eq. (A.9) corridor capacities versus physical ratings (Reviewer 1.4)
    static = model["static"]; hazard = model["hazard"]
    R._configure_core(); panel = core.build_baseline_panel(static, hazard)
    cols = [n for n, _, _ in core.INDICATOR_SYSTEM]; cti = {c: i for i, c in enumerate(cols)}
    Dproxy = core.baseline_demand(static, panel[cols].to_numpy(float), cti)
    Ci = static.loc[N.STATES, "C_i"].to_numpy(float)
    rows = []
    A_ = core.adjacency_matrix(core.STATE_ORDER)
    for i in range(16):
        for j in range(i + 1, 16):
            if A_[i, j] == 0:
                continue
            pair = " - ".join(sorted((N.ABBR[N.STATES[i]], N.ABBR[N.STATES[j]])))
            Tij = core.FLOW_SCALE * 0.5 * (Dproxy[i] + Dproxy[j]) * np.sqrt(Ci[i] * Ci[j])
            phys = tab.loc[pair] if pair in tab.index else None
            rows.append(dict(pair=pair, lite_Tij_MW=Tij,
                             physical_thermal_MVA=float(phys.thermal_MVA) if phys is not None else 0.0,
                             physical_N1_MVA=float(phys.N1_MVA) if phys is not None else 0.0))
    for p in tab.index:
        if p not in [r["pair"] for r in rows]:
            rows.append(dict(pair=p, lite_Tij_MW=0.0, physical_thermal_MVA=float(tab.loc[p, "thermal_MVA"]),
                             physical_N1_MVA=float(tab.loc[p, "N1_MVA"])))
    lt = pd.DataFrame(rows)
    lt["ratio_lite_to_N1"] = lt.lite_Tij_MW / lt.physical_N1_MVA.replace(0, np.nan)
    lt.to_csv(os.path.join(out_dir, "pf_vs_lite_corridors.csv"), index=False)
    return tab, lt


def run_kpis(model, out_dir, years, stress_years, processes, seed=4242):
    g, fl = model["grid"], model["fleet"]
    struct_z = K.structural_kpis(g, fl, out_dir, storm=False)   # zonal: screening bound + comparison
    ic = struct_z.IC_winter_N0_MW.to_numpy(float)
    struct = K.structural_kpis_nodal(g, fl, out_dir)      # nodal: main specification of E3-E5
    payload = pickle.dumps(dict(grid=g, fleet=fl, wg=model["wg"], fit=model["fit"], hazard=model["hazard"], ic=ic))
    tables = {}
    for k, sc in enumerate(SCEN):
        t0 = time.time()
        out = K.run_mc(payload, sc, years, seed + 10000 * k, processes=processes)
        tables[sc] = K.mc_table(out)
        print(f"  MC {sc}: {years} years, {time.time() - t0:.0f} s; flagged days/yr "
              f"{tables[sc]['n_days'].mean():.1f}; LP failures {tables[sc]['lp_fail'].sum()}", flush=True)
    stress = {}
    for k, sc in enumerate(SCEN):
        t0 = time.time()
        # Baseline stress only documents that the stressed system is not short under state-specific
        # hazards and nominal neighbour support (E6 uses the coherent runs), so fewer years suffice
        ny = max(10, stress_years // 4) if sc == "baseline" else stress_years
        out = K.run_mc(payload, sc, ny, seed + 50000 + 10000 * k, thermal_scale=STRESS_SCALE,
                       p_fail=0.0, processes=processes)
        stress[sc] = K.mc_table(out)
        print(f"  stress {sc}: national LOLE {stress[sc]['lole_nat'].mean():.2f} h/a, "
              f"{time.time() - t0:.0f} s", flush=True)
    pickle.dump(dict(mc=tables, stress=stress), open(os.path.join(out_dir, "pf_mc_tables.pkl"), "wb"))
    return finalise_kpis(struct, tables, stress, out_dir)


def finalise_kpis(struct, tables, stress, out_dir):
    summ = pd.concat([K.summarise_mc(tables[sc], LABEL[sc]) for sc in SCEN])
    summ.to_csv(os.path.join(out_dir, "pf_mc_summary.csv"))
    ssum = pd.concat([K.summarise_mc(stress[sc], LABEL[sc]) for sc in SCEN])
    ssum.to_csv(os.path.join(out_dir, "pf_stress_summary.csv"))
    kt = struct.copy()
    for sc in SCEN:
        kt[f"EENS_ppm_{sc}"] = summ[summ.scenario == LABEL[sc]].EENS_ppm.values
        kt[f"LOLE_h_{sc}"] = summ[summ.scenario == LABEL[sc]].LOLE_h.values
        # E6 is a property of the physical network under coherent scarcity: Baseline and Adjacency Flow
        # use the coherent network stress run; National Pool (no internal constraints) its copper plate
        src = "national_pool" if sc == "national_pool" else "adjacency_flow"
        kt[f"Exposure_{sc}"] = K.exposure_ratio(stress[src]).values
        kt[f"stress_EENS_ppm_{sc}"] = ssum[ssum.scenario == LABEL[sc]].EENS_ppm.values
    kt.to_csv(os.path.join(out_dir, "pf_kpis.csv"))
    nat = pd.DataFrame({LABEL[sc]: dict(years=len(tables[sc]["eens"]), lole_nat=tables[sc]["lole_nat"].mean(),
                                         eens_nat_MWh=tables[sc]["eens_nat"].mean(),
                                         share_years_with_shedding=(tables[sc]["eens_nat"] > 1).mean(),
                                         adequacy_EENS_MWh=(tables[sc]["eens"] - tables[sc]["eens_event"]).sum(1).mean(),
                                         stress_years=len(stress[sc]["eens"]),
                                         stress_lole_nat=stress[sc]["lole_nat"].mean(),
                                         stress_eens_nat_MWh=stress[sc]["eens_nat"].mean()) for sc in SCEN}).T
    nat.to_csv(os.path.join(out_dir, "pf_national_summary.csv"))
    return kt, tables


def run_mcda(model, kt, tables, out_dir, M, Mw, n_perm):
    static, hazard = model["static"], model["hazard"]
    res = {}
    rows = []
    for sc in SCEN:
        t0 = time.time()
        spec = MC.pf_energy_spec(kt, tables, sc)
        r = MC.run_react(static, hazard, sc, spec, M=M, Mw=Mw, n_perm=n_perm)
        res[sc] = r
        d = r["diag"].copy(); d["Scenario"] = LABEL[sc]; d["State"] = d.index
        rows.append(d)
        print(f"  REACT-PF MCDA {sc}: identified {list(d[d.identified].classification.items())} "
              f"({time.time() - t0:.0f} s)", flush=True)
        r["B"].to_csv(os.path.join(out_dir, f"pf_domain_scores_{sc}.csv"))
    audit = pd.concat(rows).reset_index(drop=True)
    audit.to_csv(os.path.join(out_dir, "pf_classification_audit.csv"), index=False)
    return res, audit


def comparisons(model, kt, tables, pf_res, pf_audit, out_dir, M, Mw, n_perm, lite_dir):
    static, hazard = model["static"], model["hazard"]
    from scipy.stats import pearsonr, spearmanr
    lite_audit = pd.read_csv(os.path.join(lite_dir, "classification_audit.csv"))
    # M0 / M1 / M2 / PF classification matrix (R3.5, R1.8)
    m0, B0 = MC.lite_m0(static, hazard)
    m1 = MC.lite_m1(static, hazard, M=M, Mw=Mw, n_perm=n_perm)
    pfm0, _ = MC.pf_m0(static, hazard, kt, tables, "baseline")
    cmp = pd.DataFrame(index=N.STATES)
    cmp["conventional_lowest_domain"] = m0.lowest_domain_score
    cmp["M0_provisional"] = m0.provisional; cmp["M0_T"] = m0["T"]; cmp["M0_class"] = m0.classification
    cmp["M1_provisional"] = m1.provisional; cmp["M1_Psel"] = m1.P_sel; cmp["M1_T"] = m1["T"]; cmp["M1_class"] = m1.classification
    for sc in SCEN:
        la = lite_audit[lite_audit.Scenario == LABEL[sc]].set_index("State").loc[N.STATES]
        cmp[f"M2_{sc}_provisional"] = la.Provisional_domain; cmp[f"M2_{sc}_Psel"] = la.P_sel
        cmp[f"M2_{sc}_class"] = la.Classification
        pa = pf_audit[pf_audit.Scenario == LABEL[sc]].set_index("State").loc[N.STATES]
        cmp[f"PF_{sc}_provisional"] = pa.provisional; cmp[f"PF_{sc}_Psel"] = pa.P_sel; cmp[f"PF_{sc}_class"] = pa.classification
        cmp[f"PF_{sc}_rank"] = pa.expected_rank
        cmp[f"M2_{sc}_rank"] = la.Expected_rank
    cmp["PF_M0_provisional"] = pfm0.provisional; cmp["PF_M0_class"] = pfm0.classification
    cmp.to_csv(os.path.join(out_dir, "classification_across_configurations.csv"))
    # domain-score correlations (R3.2)
    lite_B = pd.read_csv(os.path.join(lite_dir, "domain_scores_baseline.csv"), index_col=0).loc[N.STATES]
    pfB = pf_res["baseline"]["B"].loc[N.STATES]
    corr = []
    for name, B in (("REACT-Lite", lite_B), ("REACT-PF", pfB)):
        for a, b in (("Climate", "Energy"), ("Climate", "Adaptive"), ("Energy", "Adaptive")):
            corr.append(dict(model=name, pair=f"{a}~{b}", pearson=pearsonr(B[a], B[b]).statistic,
                             spearman=spearmanr(B[a], B[b]).statistic))
    corr.append(dict(model="Lite-vs-PF", pair="Energy~Energy", pearson=pearsonr(lite_B.Energy, pfB.Energy).statistic,
                     spearman=spearmanr(lite_B.Energy, pfB.Energy).statistic))
    pd.DataFrame(corr).to_csv(os.path.join(out_dir, "domain_correlations_lite_vs_pf.csv"), index=False)
    # rank correlations Lite vs PF
    rk = [dict(scenario=LABEL[sc], spearman=spearmanr(cmp[f"M2_{sc}_rank"], cmp[f"PF_{sc}_rank"]).statistic) for sc in SCEN]
    pd.DataFrame(rk).to_csv(os.path.join(out_dir, "rank_correlation_lite_vs_pf.csv"), index=False)
    # evidence tiers (R3.6)
    tiers = MC.evidence_tiers(lite_audit, pf_audit, kt)
    tiers.to_csv(os.path.join(out_dir, "evidence_tiers.csv"), index=False)
    # PF indicator VIF and correlations
    st = kt.loc[N.STATES]
    E = pd.DataFrame({"EENS_ppm": st.EENS_ppm_baseline, "LOLE_h": st.LOLE_h_baseline,
                      "DR_winter": MC.dr_value(st.DR_winter_N1), "DR_heat": MC.dr_value(st.DR_heat_N1),
                      "Storm_loss_h": st.storm_EENS_event_h, "Exposure": st.Exposure_adjacency_flow,
                      "Local_firm": st.Local_firm})
    E.corr(method="spearman").to_csv(os.path.join(out_dir, "pf_indicator_correlations.csv"))
    return cmp, tiers


def main():
    p = argparse.ArgumentParser(description="REACT-PF")
    p.add_argument("output_dir", nargs="?", default="results_pf")
    p.add_argument("--mode", choices=("all", "kpis", "mcda"), default="all")
    p.add_argument("--years", type=int, default=250)
    p.add_argument("--stress-years", type=int, default=120)
    p.add_argument("--processes", type=int, default=2)
    p.add_argument("--mc", type=int, default=2000); p.add_argument("--diagnostic", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--lite-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    p.add_argument("--weather-dir", default=None)
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    if a.fast:
        a.years, a.stress_years, a.mc, a.diagnostic, a.permutations = 8, 6, 200, 1000, 50
    os.makedirs(a.output_dir, exist_ok=True)
    model = build(a.output_dir, a.weather_dir)
    write_network_tables(model, a.output_dir)
    if a.mode in ("all", "kpis"):
        kt, tables = run_kpis(model, a.output_dir, a.years, a.stress_years, a.processes)
    else:
        mt = pickle.load(open(os.path.join(a.output_dir, "pf_mc_tables.pkl"), "rb"))
        nod = os.path.join(a.output_dir, "pf_structural_kpis_nodal.csv")
        struct = pd.read_csv(nod if os.path.exists(nod) else os.path.join(a.output_dir, "pf_structural_kpis.csv"),
                             index_col=0)
        kt, tables = finalise_kpis(struct, mt["mc"], mt["stress"], a.output_dir)
    if a.mode in ("all", "mcda"):
        pf_res, pf_audit = run_mcda(model, kt, tables, a.output_dir, a.mc, a.diagnostic, a.permutations)
        comparisons(model, kt, tables, pf_res, pf_audit, a.output_dir, a.mc, a.diagnostic, a.permutations, a.lite_dir)


if __name__ == "__main__":
    main()
