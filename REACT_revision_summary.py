#!/usr/bin/env python3
"""Collect the robustness results into one matrix (Fig. 6 / Table A.9 of the revised manuscript).

Usage: python REACT_revision_summary.py <results_dir>
Reads <results_dir>/robustness/*.csv and <results_dir>/revision/*.csv and writes
<results_dir>/revision/robustness_matrix.csv and robustness_matrix.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

CASES = [("Bayern", "Climate", "BY Climate"), ("Baden-Württemberg", "Climate", "BW Climate"),
         ("Brandenburg", "Climate", "BB Climate"), ("Hamburg", "Energy", "HH Energy"),
         ("Sachsen", "Climate", "SN Climate"), ("Mecklenburg-Vorpommern", "Adaptive", "MV Socio-econ.")]
SCEN = ["Baseline", "National Pool", "Adjacency Flow"]


def counts(df, main=None):
    """Number of scenarios (of 3) in which each case is identified; missing scenarios taken from main."""
    out = {}
    for st, dom, lab in CASES:
        k = 0
        for sc in SCEN:
            g = df[(df.state == st) & (df.scenario == sc)]
            if g.empty and main is not None:
                g = main[(main.state == st) & (main.scenario == sc)]
            if g.empty:
                continue
            r = g.iloc[0]
            k += int(bool(r.identified) and r.provisional == dom)
        out[lab] = k
    return out


def psel_range(df, main=None):
    out = {}
    for st, dom, lab in CASES:
        vals = []
        for sc in SCEN:
            g = df[(df.state == st) & (df.scenario == sc)]
            if g.empty and main is not None:
                g = main[(main.state == st) & (main.scenario == sc)]
            if not g.empty and g.iloc[0].provisional == dom:
                vals.append(float(g.iloc[0].P_sel))
        out[lab] = (min(vals), max(vals)) if vals else None
    return out


def main(res):
    rv = os.path.join(res, "revision")
    rb = os.path.join(res, "robustness")
    rows = []

    def add(group, label, df, main_df=None, rho=None, extra=None):
        c = counts(df, main_df)
        p = psel_range(df, main_df)
        if rho is None and "rank_rho_vs_main" in df:
            rho = (float(df.rank_rho_vs_main.min()), float(df.rank_rho_vs_main.max()))
        rows.append(dict(group=group, label=label, **{k: v for k, v in c.items()},
                         **{f"P_{k}": (None if v is None else f"{v[0]:.2f}-{v[1]:.2f}") for k, v in p.items()},
                         rho_min=None if rho is None else round(rho[0], 2),
                         rho_max=None if rho is None else round(rho[1], 2),
                         n_identified=int(df.identified.sum()) if "identified" in df else None,
                         **(extra or {})))

    spec = pd.read_csv(os.path.join(rb, "robustness_specifications.csv"))
    main_df = spec[spec.specification == "main"]
    add("Main", "Main specification", main_df, rho=(1.0, 1.0))

    # numerical and panel
    seeds = pd.read_csv(os.path.join(rb, "robustness_seed_replicas.csv"))
    per_seed = [counts(g) for _, g in seeds.groupby("seed")]
    rows.append(dict(group="Numerical and panel", label="Monte Carlo seeds (minimum over 5)",
                     **{lab: min(c[lab] for c in per_seed) for _, _, lab in CASES}))
    pdl = pd.read_csv(os.path.join(rb, "robustness_panel_deletion.csv"))
    per_del = {}
    for d, g in pdl.groupby("deleted"):
        per_del[d] = counts(g)
    rows.append(dict(group="Numerical and panel", label="Leave one state out (minimum over 15)",
                     **{lab: min(c[lab] for d, c in per_del.items() if d != st) for st, _, lab in CASES}))

    # construct
    names = {"absolute_capacities": "Absolute capacities", "grid_per_demand": "Grid per average load",
             "grid_geometric_mean": "Grid per geometric mean", "thermal_as_cost": "Thermal capacity as cost"}
    for k, lab in names.items():
        add("Construct", lab, spec[spec.specification == k])
    ov = pd.read_csv(os.path.join(rv, "overlap_sensitivity.csv"))
    add("Construct", "Thermal capacity excluded", ov[ov.specification == "thermal_excluded"])
    so = pd.read_csv(os.path.join(rv, "socio_sensitivity.csv"))
    for k, lab in [("socio_2022_only (GDP, HDI)", "Socio-economic: 2022 indicators only"),
                   ("socio_without_innovation_2019", "Socio-economic: without innovation (2019)"),
                   ("socio_without_age_dependency_2025", "Socio-economic: without age dependency (2025)")]:
        add("Construct", lab, so[so.specification == k])
    stg = pd.read_csv(os.path.join(rv, "storage_correction.csv"))
    for k, lab in [("storage_registry_raw", "Storage data: 1 Oct 2025 raw registry"),
                   ("storage_submitted_panel", "Storage data: original submitted panel"),
                   ("storage_legacy_interim", "Storage data: historical interim panel"),
                   ("storage_snapshot_excl_flags", "Storage data: 31 Oct 2026 screened"),
                   ("storage_snapshot_raw", "Storage data: 31 Oct 2026 raw registry")]:
        if (stg.specification == k).any():
            add("Construct", lab, stg[stg.specification == k])

    # hazard
    add("Hazard", "Low hazard response", spec[spec.specification == "low_hazard_response"])
    add("Hazard", "High hazard response", spec[spec.specification == "high_hazard_response"])
    for k, lab in [("no_derating (hazards in Climate Exposure only)", "No derating"),
                   ("climate_without_hot_days", "Climate Exposure without hot days"),
                   ("climate_without_low_wind", "Climate Exposure without low-wind days"),
                   ("climate_without_HD_WSM_LWD (continuous derating drivers via Energy only)",
                    "Climate Exposure without HD, WSM, LWD")]:
        add("Hazard", lab, ov[ov.specification == k])
    cl = pd.read_csv(os.path.join(rv, "climate_shift_sensitivity.csv"))
    for k, lab in [("trend_adjusted_2024", "Hazards trend-adjusted to 2024"),
                   ("trend_extrapolated_2050", "Hazards trend-extrapolated to 2050"),
                   ("trend_2050_heat_moisture_precip_only", "2050 trends, wind hazards unchanged"),
                   ("uniform_intensification", "Uniform hazard intensification")]:
        add("Hazard", lab, cl[cl.specification == k])
    bpath = os.path.join(rv, "bootstrap_full_model.csv")
    if os.path.exists(bpath):
        bt = pd.read_csv(bpath)
        per_rep = [counts(g) for _, g in bt.groupby("rep")]
        n = len(per_rep)
        rows.append(dict(group="Hazard", label=f"Year-block bootstrap (median over {n})",
                         **{lab: int(np.median([c[lab] for c in per_rep])) for _, _, lab in CASES},
                         **{f"share_any_{lab}": round(np.mean([c[lab] > 0 for c in per_rep]), 3) for _, _, lab in CASES},
                         **{f"share_all3_{lab}": round(np.mean([c[lab] == 3 for c in per_rep]), 3) for _, _, lab in CASES},
                         **{f"share_ge2_{lab}": round(np.mean([c[lab] >= 2 for c in per_rep]), 3) for _, _, lab in CASES},
                         rho_min=round(float(bt.rank_rho_vs_main.min()), 2), rho_max=round(float(bt.rank_rho_vs_main.max()), 2)))

    # weights
    add("Weights", "Equal weights", spec[spec.specification == "equal_weights"])
    w = pd.read_csv(os.path.join(rv, "weights_sensitivity.csv"))
    for k, lab in [("alpha_0.5", r"Dirichlet $\alpha=0.5$"), ("alpha_2", r"Dirichlet $\alpha=2$"),
                   ("alpha_5", r"Dirichlet $\alpha=5$"), ("alpha_20", r"Dirichlet $\alpha=20$")]:
        add("Weights", lab, w[w.specification == k])
    emph = w[w.specification.str.startswith("emphasis")]
    per = [counts(g) for _, g in emph.groupby("specification")]
    rows.append(dict(group="Weights", label="Domain emphasis (3 distributions)",
                     **{lab: f"{min(c[lab] for c in per)}-{max(c[lab] for c in per)}" if min(c[lab] for c in per) != max(c[lab] for c in per)
                        else min(c[lab] for c in per) for _, _, lab in CASES},
                     rho_min=round(float(emph.rank_rho_vs_main.min()), 2), rho_max=round(float(emph.rank_rho_vs_main.max()), 2)))

    # support
    sp = pd.read_csv(os.path.join(rv, "support_sensitivity.csv"))
    oat = sp[~sp.specification.isin(["main", "joint_structural_uncertainty"])]
    per = [counts(g, main_df) for _, g in oat.groupby("specification")]
    rows.append(dict(group="Support layer", label="Support parameters, one at a time (12)",
                     **{lab: (f"{min(c[lab] for c in per)}-{max(c[lab] for c in per)}" if min(c[lab] for c in per) != max(c[lab] for c in per)
                              else min(c[lab] for c in per)) for _, _, lab in CASES},
                     rho_min=round(float(oat.rank_rho_vs_main.min()), 2), rho_max=round(float(oat.rank_rho_vs_main.max()), 2)))
    add("Support layer", "Support parameters, jointly sampled", sp[sp.specification == "joint_structural_uncertainty"], main_df)
    se = pd.read_csv(os.path.join(rv, "support_extra_sensitivity.csv"))
    add("Support layer", "Max-min fair flow allocation", se[se.specification == "flow_allocation_fair"], main_df)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rv, "robustness_matrix.csv"), index=False)
    print(df[["group", "label"] + [c[2] for c in CASES] + ["rho_min", "rho_max"]].to_string())
    return df


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
