#!/usr/bin/env python3
"""Additional analyses for the revised REACT manuscript (response to reviewers).

Every analysis reuses the manuscript specification in ``REACT_ESR_reproducible.py``
and changes one element at a time. Outputs are written to ``<out>/revision/``.

  m012          relative-weakness classification under M0, M1, M2 and
                conventional domain diagnostics (R1.8, R3.5)
  weights       Dirichlet concentration and domain-emphasis weight distributions (R1.6)
  support       support-layer parameters, one at a time and jointly sampled (R1.4, R1.6)
  overlap       hazard double-counting variants and thermal exclusion (R1.9, R1.5)
  dependence    inter-state dependence, trend tests, year-blocked copula CV,
                year-block bootstrap of the copula parameters (R1.3)
  bootstrap     year-block bootstrap propagated through the full model (R1.3)
  climate       trend-adjusted and intensified hazard distributions (R3.4)
  storage       complete dated registry panels and historical-input sensitivities
  support_diag  frequency of unresolved shortfall by state (R3.1, R3.2)
  permutation   behaviour of the within-state label-permutation check (R1.7)

Usage:  python REACT_revision.py <out_dir> --analysis all
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, theilslopes

import REACT_ESR_reproducible as react

core = react.core
SC = react.SCENARIOS
LBL = react.SCENARIO_LABEL
HCOLS = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean",
         "Rx1day_mm_max", "Storm_Days_mean", "low_wind_days_mean"]
HEADLINE = [("Bayern", "Climate"), ("Baden-Württemberg", "Climate"),
            ("Brandenburg", "Climate"), ("Hamburg", "Energy")]
DOMAINS = ("Climate", "Energy", "Adaptive")
SNAPSHOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "mastr_storage", "2026-10-31", "storage_by_state.csv")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _data():
    return react._configured_static(), core.load_hazard_panel()


def run3(static, hazard, scenarios=SC, n_perm=20):
    return {sc: react.run_scenario(static, hazard, sc, verbose=False, n_perm=n_perm) for sc in scenarios}


def rows_from(res, main, spec, extra=None):
    rows = []
    for sc, r in res.items():
        rho = spearmanr(r["expected_rank"], main[sc]["expected_rank"]).statistic
        for s in core.STATE_ORDER:
            d = dict(specification=spec, scenario=LBL[sc], state=s, provisional=r["dominant"][s],
                     P_sel=float(r["P_sel"][s]), T=float(r["T_joint"][s]), identified=bool(r["identified"][s]),
                     expected_rank=float(r["expected_rank"][s]), rank_rho_vs_main=float(rho))
            if extra:
                d.update(extra)
            rows.append(d)
    return rows


def headline_summary(df):
    """Compact table: P_sel range and number of identified scenarios for the headline cases."""
    out = []
    for spec, g in df.groupby("specification", sort=False):
        row = dict(specification=spec)
        for st, dom in HEADLINE:
            h = g[g.state == st]
            same = h[h.provisional == dom]
            ps = h.apply(lambda r: r.P_sel if r.provisional == dom else np.nan, axis=1)
            row[f"{st}"] = (f"{np.nanmin(ps):.2f}-{np.nanmax(ps):.2f} ({int((same.identified).sum())})"
                            if len(same) else f"other ({h.provisional.mode().iat[0]})")
        others = g[g.identified & ~g.apply(lambda r: (r.state, r.provisional) in HEADLINE, axis=1)]
        row["additional"] = "; ".join(sorted({f"{react.ABBR[s]}:{d}" for s, d in zip(others.state, others.provisional)}))
        row["n_identified"] = int(g.identified.sum())
        row["rho_min"] = round(g.rank_rho_vs_main.min(), 3)
        row["rho_max"] = round(g.rank_rho_vs_main.max(), 3)
        out.append(row)
    return pd.DataFrame(out)


class patched:
    """Temporarily set attributes on modules; restores them on exit."""

    def __init__(self, **kw):
        self.kw = kw
        self.saved = {}

    def __enter__(self):
        for key, val in self.kw.items():
            mod, attr = key.split("__", 1)
            m = core if mod == "core" else react
            self.saved[key] = deepcopy(getattr(m, attr))
            setattr(m, attr, val)
        return self

    def __exit__(self, *a):
        for key, val in self.saved.items():
            mod, attr = key.split("__", 1)
            m = core if mod == "core" else react
            setattr(m, attr, val)
        react._COPULA_CACHE.clear()


# -----------------------------------------------------------------------------
# R1.8 / R3.5: M0, M1, M2 relative-weakness classifications and conventional diagnostics
# -----------------------------------------------------------------------------
def _diag_from_blocks(block_iter, seed):
    rng = np.random.default_rng(seed)
    P, W, med, lo, hi, T, p, q, cal = react.diagnose_blocks(block_iter, rng, n_perm=20)
    cand = P.argmax(1)
    psel = P.max(1)
    ident = (psel >= 0.70) & (T > 0)
    return pd.DataFrame(dict(provisional=[DOMAINS[c] for c in cand], P_sel=psel, T=T,
                             median_Z=[med[i, c] for i, c in enumerate(cand)], identified=ident),
                        index=core.STATE_ORDER)


def analysis_m012(out, main):
    static, hazard = _data()
    react._configure_core()
    panel = core.build_baseline_panel(static, hazard)
    cols = [n for n, _, _ in core.INDICATOR_SYSTEM]
    is_cost = np.array([core.IS_COST[c] for c in cols])
    Yn = core.normalised_indicators(panel[cols].to_numpy(float), is_cost)
    bidx = {b: np.array([cols.index(c) for c in core.BLOCK_INDICATORS[b]]) for b in DOMAINS}
    # M0: observed matrix, equal within-domain weights; only prediction error is propagated
    B0 = np.column_stack([Yn[:, bidx[b]].mean(1) for b in DOMAINS])
    m0 = _diag_from_blocks(B0[None], react.RNG_SEED)
    # M0 without prediction error: deterministic residual ordering
    R0, _ = react._residualise(B0)
    Z0 = (R0 - R0.mean(0)) / R0.std(0)
    # M1: observed matrix, Dirichlet within-domain weights
    rng = np.random.default_rng(react.RNG_SEED)
    B1 = np.zeros((react.N_MONTE_CARLO, len(Yn), 3))
    for m in range(react.N_MONTE_CARLO):
        for k, b in enumerate(DOMAINS):
            ww = core.sample_dirichlet_floor([1.0] * len(bidx[b]), rng)
            B1[m, :, k] = Yn[:, bidx[b]] @ ww
    m1 = _diag_from_blocks(B1, react.RNG_SEED + 1)

    # conventional diagnostics on the M0 matrix
    w0 = np.zeros(len(cols))
    for b in DOMAINS:
        w0[bidx[b]] = 1 / 3 / len(bidx[b])
    gap = w0 * (1 - Yn)
    obstacle = gap / gap.sum(1, keepdims=True)
    obst_dom = np.column_stack([obstacle[:, bidx[b]].sum(1) for b in DOMAINS])
    zdom = (B0 - B0.mean(0)) / B0.std(0)

    rows = []
    for i, s in enumerate(core.STATE_ORDER):
        d = dict(state=s,
                 lowest_domain_score=DOMAINS[int(B0[i].argmin())],
                 obstacle_degree_main=DOMAINS[int(obst_dom[i].argmax())],
                 obstacle_share=round(float(obst_dom[i].max()), 3),
                 lowest_domain_z=DOMAINS[int(zdom[i].argmin())],
                 lowest_domain_z_value=round(float(zdom[i].min()), 2),
                 M0_residual_lowest=DOMAINS[int(Z0[i].argmin())],
                 M0_residual_z=round(float(Z0[i].min()), 2),
                 M0_provisional=m0.provisional[s], M0_Psel=round(m0.P_sel[s], 3), M0_identified=bool(m0.identified[s]),
                 M1_provisional=m1.provisional[s], M1_Psel=round(m1.P_sel[s], 3), M1_identified=bool(m1.identified[s]))
        for sc in SC:
            r = main[sc]
            d[f"M2_{sc}_provisional"] = r["dominant"][s]
            d[f"M2_{sc}_Psel"] = round(float(r["P_sel"][s]), 3)
            d[f"M2_{sc}_identified"] = bool(r["identified"][s])
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "m012_classification.csv"), index=False)
    pd.DataFrame(B0, index=core.STATE_ORDER, columns=DOMAINS).to_csv(os.path.join(out, "m0_domain_scores.csv"))

    # rank-acceptability summaries that M0 cannot provide
    acc = []
    for sc in SC:
        ri = main[sc]["rank_iter"]
        for k, s in enumerate(core.STATE_ORDER):
            acc.append(dict(scenario=LBL[sc], state=s, expected_rank=float(ri[:, k].mean()),
                            p_top3=float((ri[:, k] <= 3).mean()), p_bottom3=float((ri[:, k] >= 14).mean()),
                            rank_q05=float(np.quantile(ri[:, k], 0.05)), rank_q95=float(np.quantile(ri[:, k], 0.95))))
    pd.DataFrame(acc).to_csv(os.path.join(out, "m2_rank_intervals.csv"), index=False)
    print("  m012: done", flush=True)
    return df


# -----------------------------------------------------------------------------
# R1.6: weight distributions
# -----------------------------------------------------------------------------
def _domain_emphasis_sampler(alpha_dom, alpha_within=1.0):
    base = core.sample_dirichlet_floor

    def f(alpha, rng, **k):
        if len(alpha) == 3:
            return base(list(alpha_dom), rng, **k)
        return base([alpha_within] * len(alpha), rng, **k)
    return f


def analysis_weights(out, main):
    static, hazard = _data()
    specs = {
        "alpha_0.5": dict(core__DIRICHLET_ALPHA=0.5),
        "alpha_2": dict(core__DIRICHLET_ALPHA=2.0),
        "alpha_5": dict(core__DIRICHLET_ALPHA=5.0),
        "alpha_20": dict(core__DIRICHLET_ALPHA=20.0),
        "emphasis_climate": dict(core__sample_dirichlet_floor=_domain_emphasis_sampler((4, 2, 2))),
        "emphasis_energy": dict(core__sample_dirichlet_floor=_domain_emphasis_sampler((2, 4, 2))),
        "emphasis_socioeconomic": dict(core__sample_dirichlet_floor=_domain_emphasis_sampler((2, 2, 4))),
    }
    rows = rows_from(main, main, "main")
    for name, kw in specs.items():
        t = time.time()
        with patched(**kw):
            res = run3(static, hazard)
        rows += rows_from(res, main, name)
        print(f"  weights {name}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "weights_sensitivity.csv"), index=False)
    headline_summary(df).to_csv(os.path.join(out, "weights_summary.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# R1.4 / R1.6: support-layer parameters
# -----------------------------------------------------------------------------
SUPPORT_RANGES = dict(c_w=(0.05, 0.20), c_pv=(0.02, 0.10), c_th=(0.75, 0.95), h_dis=(2.0, 8.0),
                      phi=(0.10, 0.50), af_access=(0.50, 0.90), af_grid=(0.15, 0.45), np_cap=(0.30, 0.70))


def _joint_sampler(rng):
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SUPPORT_RANGES.items()}


def analysis_support(out, main):
    static, hazard = _data()
    sc2 = ("national_pool", "adjacency_flow")
    specs = {
        "credits_low": dict(core__CAPACITY_CREDIT={"Wind": 0.05, "Solar": 0.02, "Thermal": 0.75}),
        "credits_high": dict(core__CAPACITY_CREDIT={"Wind": 0.20, "Solar": 0.10, "Thermal": 0.95}),
        "storage_window_2h": dict(core__H_DIS=2.0),
        "storage_window_8h": dict(core__H_DIS=8.0),
        "transfer_phi_0.10": dict(core__FLOW_SCALE=0.10),
        "transfer_phi_0.50": dict(core__FLOW_SCALE=0.50),
        "af_access_penalty_0.50": dict(core__AF_PENALTY_ACCESS=0.50),
        "af_access_penalty_0.90": dict(core__AF_PENALTY_ACCESS=0.90),
        "af_grid_penalty_0.15": dict(core__AF_PENALTY_GRID=0.15),
        "af_grid_penalty_0.45": dict(core__AF_PENALTY_GRID=0.45),
        "np_cap_0.30": dict(core__NP_PENALTY_CAP=0.30),
        "np_cap_0.70": dict(core__NP_PENALTY_CAP=0.70),
        "joint_structural_uncertainty": dict(core__STRUCTURAL_SAMPLER=_joint_sampler),
    }
    main2 = {sc: main[sc] for sc in sc2}
    rows = rows_from(main2, main2, "main")
    for name, kw in specs.items():
        t = time.time()
        with patched(**kw):
            res = run3(static, hazard, scenarios=sc2)
        rows += rows_from(res, main2, name)
        print(f"  support {name}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "support_sensitivity.csv"), index=False)
    headline_summary(df).to_csv(os.path.join(out, "support_summary.csv"), index=False)
    return df


def analysis_support_extra(out, main, corridor_file=None):
    """Adjacency Flow with max-min fair shortfall sharing and, if supplied, physical corridor capacities."""
    static, hazard = _data()
    main2 = {"adjacency_flow": main["adjacency_flow"]}
    specs = {"flow_allocation_fair": dict(core__FLOW_ALLOCATION="fair")}
    if corridor_file:
        K = pd.read_csv(corridor_file)   # columns: state_a, state_b, capacity_MW
        idx = {s: i for i, s in enumerate(core.STATE_ORDER)}
        for scale in (0.3, 0.5, 0.7):
            F = np.zeros((len(idx), len(idx)))
            for a, b, c in K[["state_a", "state_b", "capacity_MW"]].itertuples(index=False):
                F[idx[a], idx[b]] = F[idx[b], idx[a]] = scale * float(c)
            specs[f"physical_corridors_x{scale}"] = dict(core__FLOW_CAPACITY=F)
    rows = rows_from(main2, main2, "main")
    for name, kw in specs.items():
        t = time.time()
        with patched(**kw):
            res = run3(static, hazard, scenarios=("adjacency_flow",))
        rows += rows_from(res, main2, name)
        print(f"  support_extra {name}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "support_extra_sensitivity.csv"), index=False)
    headline_summary(df).to_csv(os.path.join(out, "support_extra_summary.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# R1.9 / R1.5: overlapping hazard pathways and thermal capacity
# -----------------------------------------------------------------------------
def _no_derating():
    return {"PV": dict(beta=0.0, delta_max=0.0), "Thermal": dict(alpha=0.0, beta=0.0, delta_max=0.0),
            "Grid": dict(beta=0.0, delta_max=0.0, gamma=0.0), "Wind": dict(beta=0.0, delta_max=0.0)}


def _blocks_without(drop):
    b = deepcopy(core.BLOCK_INDICATORS)
    for k in b:
        b[k] = [n for n in b[k] if n not in drop]
    return b


def analysis_overlap(out, main):
    static, hazard = _data()
    specs = {
        "no_derating (hazards in Climate Exposure only)": dict(core__PARAMS_DERATING=_no_derating()),
        "climate_without_HD_WSM_LWD (continuous derating drivers via Energy only)":
            dict(core__BLOCK_INDICATORS=_blocks_without({"Hot_Days", "Soil_Moist", "Low_Wind"})),
        "climate_without_low_wind": dict(core__BLOCK_INDICATORS=_blocks_without({"Low_Wind"})),
        "climate_without_hot_days": dict(core__BLOCK_INDICATORS=_blocks_without({"Hot_Days"})),
        "thermal_excluded": dict(core__BLOCK_INDICATORS=_blocks_without({"Thermal_MW"})),
    }
    rows = rows_from(main, main, "main")
    for name, kw in specs.items():
        t = time.time()
        with patched(**kw):
            res = run3(static, hazard)
        rows += rows_from(res, main, name)
        print(f"  overlap {name}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "overlap_sensitivity.csv"), index=False)
    headline_summary(df).to_csv(os.path.join(out, "overlap_summary.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# R1.10 / R1.11: socio-economic reference years and indicator scope
# -----------------------------------------------------------------------------
def analysis_socio(out, main):
    static, hazard = _data()
    specs = {
        "socio_2022_only (GDP, HDI)": dict(core__BLOCK_INDICATORS=_blocks_without({"Patents", "ADR"})),
        "socio_without_innovation_2019": dict(core__BLOCK_INDICATORS=_blocks_without({"Patents"})),
        "socio_without_age_dependency_2025": dict(core__BLOCK_INDICATORS=_blocks_without({"ADR"})),
    }
    rows = rows_from(main, main, "main")
    for name, kw in specs.items():
        t = time.time()
        with patched(**kw):
            res = run3(static, hazard)
        rows += rows_from(res, main, name)
        print(f"  socio {name}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "socio_sensitivity.csv"), index=False)
    headline_summary(df).to_csv(os.path.join(out, "socio_summary.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# R1.3: dependence structure, trends, copula CV, parameter bootstrap
# -----------------------------------------------------------------------------
def _pit_panel(hazard, col):
    """years x states matrix of within-state PIT values."""
    rows = {}
    for s in sorted(hazard.State.unique()):
        d = hazard[hazard.State == s].set_index("Year")[col].dropna()
        rows[s] = d.rank() / (len(d) + 1)
    return pd.DataFrame(rows)


def analysis_dependence(out, n_boot=200, seed=2026):
    _, hazard = _data()
    names = {"hot_days_mean": "HD", "soil_moist_winter_mean": "WSM", "p30_days_mean": "P30",
             "Rx1day_mm_mean": "Rx1_mean", "Rx1day_mm_max": "Rx1_max", "Storm_Days_mean": "SD",
             "low_wind_days_mean": "LWD"}
    # (a) inter-state dependence within a year and design effect
    dep = []
    for c in HCOLS:
        P = _pit_panel(hazard, c)
        Cm = P.corr(method="spearman").to_numpy()
        rbar = Cm[np.triu_indices_from(Cm, 1)].mean()
        m = P.shape[1]
        n_years = P.dropna().shape[0]
        deff = 1 + (m - 1) * rbar
        dep.append(dict(hazard=names[c], mean_interstate_spearman=round(rbar, 3), design_effect=round(deff, 2),
                        n_state_years=n_years * m, n_effective=round(n_years * m / deff, 0)))
    pd.DataFrame(dep).to_csv(os.path.join(out, "interstate_dependence.csv"), index=False)

    # (b) monotonic trends per state and hazard (Mann-Kendall via Kendall's tau; Theil-Sen slope)
    tr = []
    for c in HCOLS:
        for s in sorted(hazard.State.unique()):
            d = hazard[hazard.State == s][["Year", c]].dropna()
            tau, p = kendalltau(d.Year, d[c])
            sl = theilslopes(d[c], d.Year)[0]
            tr.append(dict(hazard=names[c], state=s, kendall_tau=tau, p_value=p, theil_sen_per_decade=10 * sl,
                           mean=d[c].mean()))
    trd = pd.DataFrame(tr)
    trd.to_csv(os.path.join(out, "hazard_trends_by_state.csv"), index=False)
    summ = trd.groupby("hazard").agg(n_sig_increase=("p_value", lambda p: int(((p < 0.05) & (trd.loc[p.index, "kendall_tau"] > 0)).sum())),
                                     n_sig_decrease=("p_value", lambda p: int(((p < 0.05) & (trd.loc[p.index, "kendall_tau"] < 0)).sum())),
                                     median_slope_per_decade=("theil_sen_per_decade", "median"),
                                     median_mean=("mean", "median")).reset_index()
    summ["median_relative_change_per_decade_pct"] = 100 * summ.median_slope_per_decade / summ.median_mean
    summ.to_csv(os.path.join(out, "hazard_trends_summary.csv"), index=False)

    # (c) year-blocked 5-fold cross-validation of copula families
    from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula
    U_all, years_all = [], []
    for s in sorted(hazard.State.unique()):
        d = hazard[hazard.State == s].sort_values("Year")
        U = np.full((len(d), len(HCOLS)), np.nan)
        for j, c in enumerate(HCOLS):
            x = d[c].to_numpy(float)
            mk = ~np.isnan(x)
            U[mk, j] = pd.Series(x[mk]).rank().to_numpy() / (mk.sum() + 1)
        U_all.append(U)
        years_all.append(d.Year.to_numpy())
    U_all = np.vstack(U_all)
    years_all = np.concatenate(years_all)
    ok = ~np.isnan(U_all).any(1)
    U_all, years_all = np.clip(U_all[ok], 1e-7, 1 - 1e-7), years_all[ok]
    uyears = np.unique(years_all)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(uyears), 5)
    k = len(HCOLS)
    cv = {"gaussian": 0.0, "student_t": 0.0, "clayton": 0.0, "gumbel": 0.0}
    df_sel = []
    for f in folds:
        te = np.isin(years_all, f)
        Utr, Ute = U_all[~te], U_all[te]
        R = core._nearest_corr(GaussianCopula(k_dim=k).fit_corr_param(Utr))
        cv["gaussian"] += float(GaussianCopula(corr=R, k_dim=k, allow_singular=True).logpdf(Ute).sum())
        best = max(((float(StudentTCopula(corr=R, df=float(d), k_dim=k).logpdf(Utr).sum()), d)
                    for d in np.arange(3.0, 30.5, 0.5)))
        df_sel.append(best[1])
        cv["student_t"] += float(StudentTCopula(corr=R, df=float(best[1]), k_dim=k).logpdf(Ute).sum())
        for fam, fn in (("clayton", react._clayton_logpdf), ("gumbel", react._gumbel_logpdf)):
            th = react._fit_archimedean(Utr, fam)["theta"]
            cv[fam] += float(np.sum(fn(Ute, th))) if np.isfinite(th) else -np.inf
    pd.DataFrame([dict(family=fam, heldout_loglik=v) for fam, v in cv.items()] +
                 [dict(family="student_t_df_selected_per_fold", heldout_loglik=";".join(str(d) for d in df_sel))]
                 ).to_csv(os.path.join(out, "copula_year_blocked_cv.csv"), index=False)

    # (d) year-block bootstrap of the copula parameters
    yrs = np.sort(hazard.Year.unique())
    boot = []
    for b in range(n_boot):
        pick = rng.choice(yrs, size=len(yrs), replace=True)
        hz = pd.concat([hazard[hazard.Year == y].assign(Year=1000 + i) for i, y in enumerate(pick)], ignore_index=True)
        react._COPULA_CACHE.clear()
        fit = react.estimate_copula(hz, HCOLS)
        R = fit["R"]
        boot.append(dict(rep=b, model=fit["model"], df=fit.get("student_t_df"),
                         aic_t_minus_gauss=fit["aic_student_t"] - fit["aic_gaussian"],
                         r_P30_Rx1=R[2, 3], r_SD_LWD=R[5, 6], r_HD_WSM=R[0, 1], r_HD_LWD=R[0, 6]))
    react._COPULA_CACHE.clear()
    bd = pd.DataFrame(boot)
    bd.to_csv(os.path.join(out, "copula_parameter_bootstrap.csv"), index=False)
    print("  dependence: done", flush=True)
    return bd


def _bootstrap_hazard(hazard, rng):
    yrs = np.sort(hazard.Year.unique())
    pick = rng.choice(yrs, size=len(yrs), replace=True)
    return pd.concat([hazard[hazard.Year == y].assign(Year=1000 + i) for i, y in enumerate(pick)], ignore_index=True)


def _bootstrap_worker(payload):
    b, years, storage_file, main_ranks, mc, diagnostic = payload
    react.STORAGE_FILE = storage_file
    react.N_MONTE_CARLO, react.N_DIAGNOSTIC = mc, diagnostic
    static, hazard = _data()
    hz = pd.concat([hazard[hazard.Year == year].assign(Year=1000 + i)
                    for i, year in enumerate(years)], ignore_index=True)
    react._COPULA_CACHE.clear()
    res = run3(static, hz)
    rows = rows_from(res, main_ranks, f"boot_{b:03d}",
                     extra=dict(rep=b, copula=res["baseline"]["copula_fit"]["model"],
                                df=res["baseline"]["copula_fit"].get("student_t_df")))
    return b, rows


def analysis_bootstrap(out, main, n_boot=40, seed=7, jobs=4):
    """Year-block bootstrap of the hazard record propagated through the full model."""
    static, hazard = _data()
    rng = np.random.default_rng(seed)
    years = np.sort(hazard.Year.unique())
    main_ranks = {sc: {"expected_rank": main[sc]["expected_rank"]} for sc in SC}
    # Generate exactly the original sequential resamples before distributing work.
    tasks = [(b, rng.choice(years, size=len(years), replace=True), react.STORAGE_FILE,
              main_ranks, react.N_MONTE_CARLO, react.N_DIAGNOSTIC) for b in range(n_boot)]
    completed = {}
    if jobs == 1:
        iterator = map(_bootstrap_worker, tasks)
        for b, result in iterator:
            completed[b] = result
            print(f"  bootstrap rep {b}: complete", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_bootstrap_worker, task) for task in tasks]
            for future in as_completed(futures):
                b, result = future.result()
                completed[b] = result
                print(f"  bootstrap rep {b}: complete ({len(completed)}/{n_boot})", flush=True)
                checkpoint = [r for key in sorted(completed) for r in completed[key]]
                pd.DataFrame(checkpoint).to_csv(os.path.join(out, "bootstrap_full_model.csv"), index=False)
    rows = [r for key in sorted(completed) for r in completed[key]]
    pd.DataFrame(rows).to_csv(os.path.join(out, "bootstrap_full_model.csv"), index=False)
    react._COPULA_CACHE.clear()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# R3.4: non-stationary hazards
# -----------------------------------------------------------------------------
TREND_COLS = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean",
              "Rx1day_mm_max", "Storm_Days_mean", "low_wind_days_mean"]


THERMO_COLS = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean", "Rx1day_mm_max"]


def trend_shifted_hazard(hazard, target_year, cols=None):
    """Shift each state's record along its Theil-Sen trend to the target year (non-negative)."""
    hz = hazard.copy()
    for s in hz.State.unique():
        m = hz.State == s
        for c in (cols or TREND_COLS):
            d = hz.loc[m, ["Year", c]].dropna()
            sl = theilslopes(d[c], d.Year)[0]
            hz.loc[m, c] = np.maximum(hz.loc[m, c] + sl * (target_year - hz.loc[m, "Year"]), 0.0)
    return hz


def scaled_hazard(hazard, factors):
    hz = hazard.copy()
    for c, f in factors.items():
        hz[c] = hz[c] * f
    return hz


def analysis_climate(out, main):
    static, hazard = _data()
    fixed = {c: (float(hazard[c].min()), float(hazard[c].max())) for c in HCOLS}
    variants = {
        "trend_adjusted_2024": trend_shifted_hazard(hazard, 2024),
        "trend_extrapolated_2050": trend_shifted_hazard(hazard, 2050),
        "trend_2050_heat_moisture_precip_only": trend_shifted_hazard(hazard, 2050, THERMO_COLS),
        "uniform_intensification": scaled_hazard(hazard, {"hot_days_mean": 1.5, "soil_moist_winter_mean": 0.9,
                                                          "p30_days_mean": 1.2, "Rx1day_mm_mean": 1.1,
                                                          "Rx1day_mm_max": 1.1}),
    }
    rows = rows_from(main, main, "main")
    for name, hz in variants.items():
        t = time.time()
        with patched(core__HAZARD_RANGES=fixed):
            react._COPULA_CACHE.clear()
            res = run3(static, hz)
        rows += rows_from(res, main, name)
        print(f"  climate {name}: {time.time() - t:.0f}s", flush=True)
    react._COPULA_CACHE.clear()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "climate_shift_sensitivity.csv"), index=False)
    headline_summary(df).to_csv(os.path.join(out, "climate_shift_summary.csv"), index=False)
    # mean hazard levels by variant for reporting
    lv = []
    for name, hz in [("historical", hazard)] + list(variants.items()):
        g = hz.groupby("State")[HCOLS].mean().mean()
        lv.append(dict(variant=name, **{k: round(v, 2) for k, v in g.items()}))
    pd.DataFrame(lv).to_csv(os.path.join(out, "climate_shift_levels.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# Storage-data correction
# -----------------------------------------------------------------------------
def analysis_storage(out, main):
    _, hazard = _data()
    rows = rows_from(main, main, "main")
    base = react._configured_static
    submitted = core.load_static_panel().loc[core.STATE_ORDER, "Storage_MWh"].to_dict()
    interim = dict(submitted, **react.STORAGE_CORRECTIONS)
    variants = {"storage_submitted_panel": submitted,
                "storage_legacy_interim": interim,
                "storage_registry_raw": react.load_storage_file(
                    react.STORAGE_FILE or react.DEFAULT_STORAGE_FILE,
                    "battery_registered_usable_energy_MWh").to_dict()}
    for name, vals in variants.items():
        def cfg(vals=vals):
            st = base()
            for s, v in vals.items():
                st.loc[s, "Storage_MWh"] = v
            return st
        t = time.time()
        res = run3(cfg(), hazard)
        rows += rows_from(res, main, name)
        print(f"  storage {name}: {time.time() - t:.0f}s", flush=True)
    if SNAPSHOT_FILE:
        for col, name in (("battery_energy_MWh_excluding_duration_flags", "storage_snapshot_excl_flags"),
                          ("battery_registered_usable_energy_MWh", "storage_snapshot_raw")):
            vals = react.load_storage_file(SNAPSHOT_FILE, col)
            st = base()
            for s in core.STATE_ORDER:
                st.loc[s, "Storage_MWh"] = float(vals[s])
            t = time.time()
            res = run3(st, hazard)
            rows += rows_from(res, main, name)
            print(f"  storage {name}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "storage_correction.csv"), index=False)
    headline_summary(df).to_csv(os.path.join(out, "storage_summary.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# R3.1 / R3.2: how often does the support layer leave a state short?
# -----------------------------------------------------------------------------
def analysis_support_diag(out):
    static, hazard = _data()
    rec = []
    with patched(core__RECORD_SUPPORT=rec):
        for sc in ("national_pool", "adjacency_flow"):
            react.run_scenario(static, hazard, sc, verbose=False, n_perm=20)
            rec_sc = [r for r in core.RECORD_SUPPORT if r["scenario"] == sc]
            if sc == "national_pool":
                scar = np.array([r["scarcity"] for r in rec_sc])
                pen = np.array([r["pen"] for r in rec_sc])
                for k, s in enumerate(core.STATE_ORDER):
                    rec.append(dict(_row=True, scenario=LBL[sc], state=s, p_national_scarcity=float((scar > 0).mean()),
                                    mean_national_scarcity=float(scar.mean()), p95_national_scarcity=float(np.quantile(scar, 0.95)),
                                    p_penalised=float((pen[:, k] > 0).mean()), mean_penalty=float(pen[:, k].mean())))
            else:
                th = np.array([r["theta"] for r in rec_sc])
                S = np.array([r["S"] for r in rec_sc])
                D = np.array([r["D"] for r in rec_sc])
                for k, s in enumerate(core.STATE_ORDER):
                    deficit = (D[:, k] - S[:, k]) > 0
                    rec.append(dict(_row=True, scenario=LBL[sc], state=s, p_local_deficit=float(deficit.mean()),
                                    p_unresolved=float((th[:, k] > 1e-6).mean()), mean_theta=float(th[:, k].mean()),
                                    mean_theta_given_deficit=float(th[deficit, k].mean()) if deficit.any() else 0.0,
                                    supply_to_demand=float(np.median(S[:, k] / D[:, k]))))
        out_rows = [r for r in rec if r.get("_row")]
    df = pd.DataFrame(out_rows).drop(columns="_row")
    df.to_csv(os.path.join(out, "support_shortfall_by_state.csv"), index=False)
    print("  support_diag: done", flush=True)
    return df


# -----------------------------------------------------------------------------
# R1.7: permutation check behaviour
# -----------------------------------------------------------------------------
def analysis_permutation(out, audit_path):
    a = pd.read_csv(audit_path)
    a["T_positive"] = a.T_joint > 0
    s = a.groupby(["identified", "T_positive"]).agg(n=("State", "size"), n_p_le_0_001=("p_permutation", lambda p: int((p <= 0.001).sum())),
                                                     max_p=("p_permutation", "max")).reset_index()
    s.to_csv(os.path.join(out, "permutation_behaviour.csv"), index=False)
    return s


# -----------------------------------------------------------------------------
def main():
    global SNAPSHOT_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="results")
    ap.add_argument("--analysis", default="all")
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4, help="independent full-model bootstrap workers")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--storage-file", default=None,
                    help="16-state MaStR storage CSV used as the main storage data (see REACT_ESR_reproducible.py)")
    ap.add_argument("--snapshot-file", default=SNAPSHOT_FILE,
                    help="additional 16-state MaStR storage CSV evaluated as a robustness row")
    ap.add_argument("--corridor-file", default=None,
                    help="CSV with state_a,state_b,capacity_MW (physical interstate corridor ratings)")
    a = ap.parse_args()
    if a.fast:
        react.N_MONTE_CARLO, react.N_DIAGNOSTIC = 300, 2000
    react.STORAGE_FILE = a.storage_file
    SNAPSHOT_FILE = a.snapshot_file
    out = os.path.join(a.out_dir, "revision")
    os.makedirs(out, exist_ok=True)
    static, hazard = _data()
    todo = a.analysis.split(",")
    allx = "all" in todo
    main_res = None
    if allx or set(todo) & {"m012", "weights", "support", "support_extra", "overlap", "bootstrap", "climate",
                            "storage", "socio"}:
        t = time.time()
        main_res = run3(static, hazard)
        print(f"main: {time.time() - t:.0f}s", flush=True)
        for sc in SC:
            print(sc, sorted((s, main_res[sc]["dominant"][s]) for s in core.STATE_ORDER if main_res[sc]["identified"][s]))
    if allx or "m012" in todo:
        analysis_m012(out, main_res)
    if allx or "permutation" in todo:
        audit = os.path.join(a.out_dir, "classification_audit.csv")
        if os.path.exists(audit):
            analysis_permutation(out, audit)
        else:
            print("  permutation: run REACT_ESR_reproducible.py first (needs classification_audit.csv)")
    if allx or "support_diag" in todo:
        analysis_support_diag(out)
    if allx or "dependence" in todo:
        analysis_dependence(out)
    if allx or "overlap" in todo:
        analysis_overlap(out, main_res)
    if allx or "socio" in todo:
        analysis_socio(out, main_res)
    if allx or "weights" in todo:
        analysis_weights(out, main_res)
    if allx or "support" in todo:
        analysis_support(out, main_res)
    if allx or "support_extra" in todo:
        analysis_support_extra(out, main_res, a.corridor_file)
    if allx or "storage" in todo:
        analysis_storage(out, main_res)
    if allx or "climate" in todo:
        analysis_climate(out, main_res)
    if allx or "bootstrap" in todo:
        analysis_bootstrap(out, main_res, n_boot=a.n_boot, jobs=max(1, a.jobs))


if __name__ == "__main__":
    main()
