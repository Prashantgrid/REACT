#!/usr/bin/env python3
"""Manuscript-aligned REACT reproducibility entry point.

This module reuses the embedded data and simulation engine in ``REACT_v4.py``
and defines the specification of "Beyond Composite Rankings: A Probabilistic
Multi-Domain Assessment of Regional Energy Resilience" (Energy Strategy Reviews).

Principal specification
-----------------------
* Load-relative Energy System indicators:
    wind, PV and thermal capacity   -> MW per MW of average load (D_annual/8760)
    storage energy capacity         -> MWh per MW of average load (hours)
    grid infrastructure (W_LINES)   -> index per km2 of state area (network density)
    interstate connectivity C_i and external-support score S(r_i) unchanged
  The support layer converts ratios back to MW, so deficit/flow physics are
  construct-invariant. Absolute capacities and alternative grid normalisations
  are sensitivity specifications (--energy-construct, --grid-normalisation).
* M=2000 hazard iterations, Mw=10000 profile draws, seed=12345
* PRESS/leave-one-state-out residual e_i/(1-h_i)
* relative domain weakness when P_sel >= 0.70 and T > 0
  (a within-state label-permutation check with BH adjustment is reported)

Modes
-----
  run         main results for the three scenarios (default)
  robustness  specification, threshold, panel-deletion, seed and redundancy checks
  recovery    synthetic planted-deviation recovery
  all         run + robustness + recovery
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import REACT_v4 as core

N_MONTE_CARLO = 2000
N_DIAGNOSTIC = 10000
N_PERMUTATION = 5000
RNG_SEED = 12345
REGRESSION_NOISE_SCALE = 1.0
RESIDUAL_MODE = "press"
_COPULA_CACHE = {}


def _clayton_logpdf(U, theta):
    U = np.asarray(U, float)
    d = U.shape[1]
    if theta <= 0:
        return np.full(U.shape[0], -np.inf)
    log_const = sum(np.log(1.0 / theta + k) for k in range(d)) + d * np.log(theta)
    log_prod = (-theta - 1.0) * np.log(U).sum(axis=1)
    one_plus_t = np.power(U, -theta).sum(axis=1) - d + 1.0
    if np.any(one_plus_t <= 0):
        return np.full(U.shape[0], -np.inf)
    return log_const + log_prod + (-1.0 / theta - d) * np.log(one_plus_t)


def _gumbel_poly_coeff(alpha, d):
    c = np.array([1.0])
    for n in range(d):
        deriv = np.array([k * c[k] for k in range(1, len(c))], dtype=float)
        term_der = np.concatenate([[0.0], alpha * deriv]) if len(deriv) else np.zeros(1)
        term_neg = np.concatenate([[0.0], -alpha * c])
        L = max(len(term_der), len(term_neg), len(c))
        out = np.zeros(L)
        out[: len(term_der)] += term_der
        out[: len(term_neg)] += term_neg
        out[: len(c)] -= n * c
        c = out
    return c


def _gumbel_logpdf(U, theta):
    U = np.asarray(U, float)
    d = U.shape[1]
    if theta < 1:
        return np.full(U.shape[0], -np.inf)
    alpha = 1.0 / theta
    lu = -np.log(U)
    t = np.power(lu, theta).sum(axis=1)
    x = np.power(t, alpha)
    coeff = _gumbel_poly_coeff(alpha, d)
    P = np.polynomial.polynomial.polyval(x, coeff)
    signed = ((-1.0) ** d) * P
    if np.any(signed <= 0) or np.any(t <= 0):
        return np.full(U.shape[0], -np.inf)
    log_psi_d = -x - d * np.log(t) + np.log(signed)
    log_phi_prime = d * np.log(theta) + (theta - 1.0) * np.log(lu).sum(axis=1) - np.log(U).sum(axis=1)
    return log_psi_d + log_phi_prime


def _fit_archimedean(U, family):
    if family == "clayton":
        fun = lambda th: -float(np.sum(_clayton_logpdf(U, th)))
        bounds = (1e-4, 20.0)
    elif family == "gumbel":
        fun = lambda th: -float(np.sum(_gumbel_logpdf(U, th)))
        bounds = (1.000001, 20.0)
    else:
        raise ValueError(family)
    opt = minimize_scalar(fun, bounds=bounds, method="bounded", options={"xatol": 1e-5})
    if not opt.success or not np.isfinite(opt.fun):
        return dict(theta=np.nan, ll=-np.inf, aic=np.inf)
    ll = -float(opt.fun)
    return dict(theta=float(opt.x), ll=ll, aic=2 - 2 * ll)


def estimate_copula(hazard, hcols):
    """Compare Gaussian, Student-t, Clayton and Gumbel candidates by AIC."""
    from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula

    key = (tuple(hcols), tuple(sorted(hazard["State"].unique())), len(hazard),
           int(hazard["Year"].min()), int(hazard["Year"].max()))
    if key in _COPULA_CACHE:
        return _COPULA_CACHE[key]

    U = core._pseudo_observations(hazard, hcols)
    k_dim = len(hcols)
    k_corr = k_dim * (k_dim - 1) // 2
    base = GaussianCopula(k_dim=k_dim)
    R = core._nearest_corr(base.fit_corr_param(U))
    g = GaussianCopula(corr=R, k_dim=k_dim, allow_singular=True)
    ll_g = float(np.sum(g.logpdf(U)))
    candidates = [(2 * k_corr - 2 * ll_g, "gaussian", None, ll_g)]

    for df in np.arange(3.0, 30.5, 0.5):
        tc = StudentTCopula(corr=R, df=float(df), k_dim=k_dim)
        ll = float(np.sum(tc.logpdf(U)))
        aic = 2 * (k_corr + 1) - 2 * ll
        if np.isfinite(aic):
            candidates.append((aic, "student_t", float(df), ll))

    cl = _fit_archimedean(U, "clayton")
    gu = _fit_archimedean(U, "gumbel")
    candidates += [(cl["aic"], "clayton", cl["theta"], cl["ll"]),
                   (gu["aic"], "gumbel", gu["theta"], gu["ll"])]
    best = min(candidates, key=lambda x: x[0])

    states = sorted(hazard["State"].unique())
    marginals = {}
    for st in states:
        d = hazard[hazard["State"] == st].sort_values("Year")
        marginals[st] = {c: np.sort(d[c].dropna().to_numpy(float)) for c in hcols}

    aic_by_family, param_by_family = {}, {}
    for aic, name, param, _ in candidates:
        if name not in aic_by_family or aic < aic_by_family[name]:
            aic_by_family[name] = aic
            param_by_family[name] = param

    fit = {
        "model": best[1],
        "df": best[2] if best[1] == "student_t" else None,
        "theta": best[2] if best[1] in ("clayton", "gumbel") else None,
        "R": R,
        "marginals": marginals,
        "states": states,
        "n_complete": len(U),
        "aic_gaussian": aic_by_family.get("gaussian", np.nan),
        "aic_student_t": aic_by_family.get("student_t", np.nan),
        "aic_clayton": aic_by_family.get("clayton", np.nan),
        "aic_gumbel": aic_by_family.get("gumbel", np.nan),
        "aic_selected": best[0],
        "loglik_selected": best[3],
        "student_t_df": param_by_family.get("student_t"),
        "clayton_theta": param_by_family.get("clayton"),
        "gumbel_theta": param_by_family.get("gumbel"),
    }
    _COPULA_CACHE[key] = fit
    return fit


def _residualise(B):
    """Eq. (5) leave-one-state-out/PRESS residual and prediction scale."""
    N, K = B.shape
    R = np.zeros_like(B, dtype=float)
    SE = np.zeros_like(B, dtype=float)
    for b in range(K):
        others = [k for k in range(K) if k != b]
        X = np.column_stack([np.ones(N), B[:, others]])
        y = B[:, b]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        XtX_inv = np.linalg.pinv(X.T @ X)
        lev = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        one_minus_h = np.maximum(1.0 - lev, 1e-8)
        sigma = np.sqrt(np.sum(resid**2) / max(N - X.shape[1], 1))
        if RESIDUAL_MODE == "press":
            R[:, b] = resid / one_minus_h
            SE[:, b] = sigma / np.sqrt(one_minus_h)
        elif RESIDUAL_MODE == "legacy":
            R[:, b] = resid
            SE[:, b] = sigma * np.sqrt(one_minus_h)
        else:
            raise ValueError(RESIDUAL_MODE)
    return R, SE


def _joint_statistic(med_z, cand):
    others = [b for b in range(len(med_z)) if b != cand]
    return float(min(-med_z[cand], min(med_z[b] - med_z[cand] for b in others)))


def _bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def _permutation_pvalues(z_store, observed_T, rng, n_perm=None, chunk=50):
    """State-specific label-permutation null, repeating selection in each null replicate."""
    if n_perm is None:
        n_perm = N_PERMUTATION
    Q, N, K = z_store.shape
    perms = np.asarray([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], dtype=np.int8)
    pvals = np.ones(N, dtype=float)
    for i in range(N):
        z = z_store[:, i, :]
        t_obs = float(observed_T[i])
        exceed = done = 0
        while done < n_perm:
            c = min(chunk, n_perm - done)
            pick = rng.integers(0, 6, size=(c, Q), dtype=np.int8)
            idx = perms[pick]
            zp = np.take_along_axis(z[None, :, :], idx, axis=2)
            winners = zp.argmin(axis=2)
            probs = np.stack([(winners == b).mean(axis=1) for b in range(K)], axis=1)
            cand = probs.argmax(axis=1)
            med = np.median(zp, axis=1)
            rows = np.arange(c)
            tc = med[rows, cand]
            diff = med - tc[:, None]
            diff[rows, cand] = np.inf
            t_null = np.minimum(-tc, diff.min(axis=1))
            exceed += int(np.count_nonzero(t_null >= t_obs - 1e-15))
            done += c
        pvals[i] = (exceed + 1) / (n_perm + 1)
    return pvals


def diagnose_blocks(block_iter, rng, n_perm=None):
    """Uncertainty-propagated, permutation-calibrated domain-profile screen."""
    M, N, K = block_iter.shape
    R_cache = np.empty_like(block_iter, dtype=float)
    SE_cache = np.empty_like(block_iter, dtype=float)
    for m in range(M):
        R_cache[m], SE_cache[m] = _residualise(block_iter[m])

    draw_idx = rng.integers(0, M, size=N_DIAGNOSTIC)
    R = R_cache[draw_idx].copy()
    R += rng.normal(0.0, 1.0, size=R.shape) * (REGRESSION_NOISE_SCALE * SE_cache[draw_idx])
    sd = R.std(axis=1, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z_store = (R - R.mean(axis=1, keepdims=True)) / sd

    winners = z_store.argmin(axis=2)
    P = np.zeros((N, K), dtype=float)
    for i in range(N):
        P[i] = np.bincount(winners[:, i], minlength=K) / N_DIAGNOSTIC
    W = (z_store < 0).mean(axis=0)
    med = np.median(z_store, axis=0)
    lo = np.quantile(z_store, 0.05, axis=0)
    hi = np.quantile(z_store, 0.95, axis=0)
    cand = P.argmax(axis=1)
    psel = P.max(axis=1)
    T = np.array([_joint_statistic(med[i], int(cand[i])) for i in range(N)])
    p_perm = _permutation_pvalues(z_store, T, rng, n_perm=n_perm)
    q_bh = _bh_adjust(p_perm)
    calibrated = (q_bh <= 0.10) & (psel >= 0.70) & (T > 0)
    return P, W, med, lo, hi, T, p_perm, q_bh, calibrated



# =============================================================================
# Energy System construct
# =============================================================================
ENERGY_CONSTRUCT = "load_relative"      # or "absolute"
GRID_NORMALISATION = "area"              # "area" | "demand" | "geomean" (load_relative only)

# State land area, km2, 31.12.2023. Statistisches Bundesamt, GENESIS table 33111-0008
# (Bodenflaeche nach Art der tatsaechlichen Nutzung, Bundeslaender).
STATE_AREA_KM2 = {
    "Baden-Württemberg": 35748, "Bayern": 70542, "Berlin": 891, "Brandenburg": 29654,
    "Bremen": 420, "Hamburg": 755, "Hessen": 21116, "Mecklenburg-Vorpommern": 23293,
    "Niedersachsen": 47710, "Nordrhein-Westfalen": 34113, "Rheinland-Pfalz": 19858,
    "Saarland": 2572, "Sachsen": 18450, "Sachsen-Anhalt": 20555, "Schleswig-Holstein": 15804,
    "Thüringen": 16202,
}

_ENGINE_BUILD = core.build_baseline_panel


def average_load_mw(static):
    """Average electricity load (MW) from annual consumption (TWh)."""
    return static.loc[core.STATE_ORDER, "Cons_TWh"].astype(float).to_numpy() * 1e6 / 8760.0


def build_panel(static, hazard):
    """Structural indicator panel under the selected Energy System construct."""
    core.ENERGY_CONSTRUCT = "absolute"
    p = _ENGINE_BUILD(static, hazard)          # absolute values; engine sets CONS_VEC = 1
    if ENERGY_CONSTRUCT == "absolute":
        core.CONS_VEC = np.ones(len(core.STATE_ORDER))
        return p
    load = average_load_mw(static)
    for c in ("Wind_MW", "Solar_MW", "Thermal_MW", "Storage_MWh"):
        p[c] = p[c] / load                     # MW/MW and MWh/MW (= hours)
    area = np.array([STATE_AREA_KM2[s] for s in core.STATE_ORDER], float)
    if GRID_NORMALISATION == "area":
        p["WLINES"] = p["WLINES"] / area * 1e3  # index per 1000 km2
    elif GRID_NORMALISATION == "demand":
        p["WLINES"] = p["WLINES"] / load * 1e3  # index per GW of average load
    elif GRID_NORMALISATION == "geomean":
        p["WLINES"] = p["WLINES"] / np.sqrt(area * load) * 1e3
    else:
        raise ValueError(GRID_NORMALISATION)
    core.CONS_VEC = load                       # support layer converts ratios back to MW
    return p.astype(float)


def _configured_static():
    static = core.load_static_panel().loc[core.STATE_ORDER].copy()
    # Table H.10 value used in the ESR manuscript.
    static.loc["Hamburg", "Solar_PV_MW"] = 250.0
    return static


def _configure_core():
    core.N_MONTE_CARLO = N_MONTE_CARLO
    core.RNG_SEED = RNG_SEED
    core.NORMALISATION = "minmax"
    core.REGRESSION_NOISE_SCALE = REGRESSION_NOISE_SCALE
    core._residualise = _residualise
    core.estimate_copula = estimate_copula
    core.build_baseline_panel = build_panel


def run_scenario(static, hazard, scenario, verbose=True, n_perm=None):
    """Run the simulation engine, then apply the ESR relative-weakness diagnostic."""
    _configure_core()
    capture = {}
    original_diag = core.diagnose_blocks

    def _capture_only(block_iter, rng):
        capture["state"] = deepcopy(rng.bit_generator.state)
        N, K = block_iter.shape[1], block_iter.shape[2]
        z = np.zeros((N, K), dtype=float)
        return z, z.copy(), z.copy(), z.copy(), z.copy()

    core.diagnose_blocks = _capture_only
    try:
        result = core.run_scenario(static, hazard, scenario, verbose=False)
    finally:
        core.diagnose_blocks = original_diag

    rng = np.random.default_rng()
    rng.bit_generator.state = capture["state"]
    P, W, med, lo, hi, T, p_perm, q_bh, calibrated = diagnose_blocks(result["block_iter"], rng, n_perm=n_perm)
    cols = ["Climate", "Energy", "Adaptive"]
    idx = core.STATE_ORDER
    Pdf = pd.DataFrame(P, index=idx, columns=cols)
    result["P_dominance"] = Pdf
    result["P_negative"] = pd.DataFrame(W, index=idx, columns=cols)
    result["Z_resid"] = pd.DataFrame(med, index=idx, columns=cols)
    result["Z_resid_lo"] = pd.DataFrame(lo, index=idx, columns=cols)
    result["Z_resid_hi"] = pd.DataFrame(hi, index=idx, columns=cols)
    result["dominant"] = Pdf.idxmax(axis=1)
    result["P_sel"] = Pdf.max(axis=1)
    result["T_joint"] = pd.Series(T, index=idx)
    result["p_permutation"] = pd.Series(p_perm, index=idx)
    result["q_bh"] = pd.Series(q_bh, index=idx)
    identified = (result["P_sel"].to_numpy() >= 0.70) & (T > 0)
    result["identified"] = pd.Series(identified, index=idx)
    result["calibrated"] = pd.Series(calibrated, index=idx)
    result["classification"] = pd.Series(np.where(identified, result["dominant"].to_numpy(), "Multidomain"), index=idx)
    result["expected_rank"] = pd.Series(result["expected_rank"], index=idx)
    result["cc_mean"] = pd.Series(result["cc_mean"], index=idx)
    if verbose:
        print(f"  {scenario}: identified = {list(result['classification'][identified].items())}")
    return result


# =============================================================================
# Outputs
# =============================================================================
SCENARIOS = ("baseline", "national_pool", "adjacency_flow")
SCENARIO_LABEL = {"baseline": "Baseline", "national_pool": "National Pool", "adjacency_flow": "Adjacency Flow"}
ABBR = {"Baden-Württemberg": "BW", "Bayern": "BY", "Berlin": "BE", "Brandenburg": "BB", "Bremen": "HB",
        "Hamburg": "HH", "Hessen": "HE", "Mecklenburg-Vorpommern": "MV", "Niedersachsen": "NI",
        "Nordrhein-Westfalen": "NRW", "Rheinland-Pfalz": "RP", "Saarland": "SL", "Sachsen": "SN",
        "Sachsen-Anhalt": "ST", "Schleswig-Holstein": "SH", "Thüringen": "TH"}


def write_scenario_results(result, scenario, out_dir):
    P = result["P_dominance"]
    df = pd.DataFrame(index=core.STATE_ORDER)
    df["CC_mean"] = result["cc_mean"]
    df["Expected_rank"] = result["expected_rank"]
    df["Provisional_domain"] = result["dominant"]
    df["Classification"] = result["classification"]
    for b in P.columns:
        df[f"P_{b}"] = P[b]
        df[f"P_negative_{b}"] = result["P_negative"][b]
        df[f"Median_Z_{b}"] = result["Z_resid"][b]
    df["P_sel"] = result["P_sel"]
    df["T_joint"] = result["T_joint"]
    df["p_permutation"] = result["p_permutation"]
    df["q_BH"] = result["q_bh"]
    df["identified"] = result["identified"]
    df.to_csv(os.path.join(out_dir, f"results_{scenario}.csv"))


def classification_audit(results):
    rows = []
    for sc in SCENARIOS:
        r = results[sc]
        for s in core.STATE_ORDER:
            d = r["dominant"][s]
            rows.append(dict(Scenario=SCENARIO_LABEL[sc], State=s, Provisional_domain=d,
                             P_sel=r["P_sel"][s], W_candidate=r["P_negative"].loc[s, d],
                             Median_Z_candidate=r["Z_resid"].loc[s, d], T_joint=r["T_joint"][s],
                             p_permutation=r["p_permutation"][s], q_BH=r["q_bh"][s],
                             identified=bool(r["identified"][s]),
                             Classification=r["classification"][s], Expected_rank=r["expected_rank"][s]))
    return pd.DataFrame(rows)


def m0_m1_vulnerability(static, hazard, n_draws=2000):
    """Deterministic TOPSIS (M0, equal domain weights) and weight uncertainty only (M1)."""
    _configure_core()
    panel = core.build_baseline_panel(static, hazard)
    cols = [n for n, _, _ in core.INDICATOR_SYSTEM]
    Yn = core.normalised_indicators(panel[cols].to_numpy(float), np.array([core.IS_COST[c] for c in cols]))
    w0 = np.zeros(len(cols))
    for b in ("Climate", "Energy", "Adaptive"):
        idx = [i for i, c in enumerate(cols) if core.BLOCK_OF[c] == b]
        w0[idx] = 1 / 3 / len(idx)
    cc0 = core.topsis_cc(Yn, w0)
    rng = np.random.default_rng(RNG_SEED)
    cc1 = []
    for _ in range(n_draws):
        wb = core.sample_dirichlet_floor([1] * 3, rng)
        w = np.zeros(len(cols))
        for bi, b in enumerate(("Climate", "Energy", "Adaptive")):
            idx = np.array([i for i, c in enumerate(cols) if core.BLOCK_OF[c] == b])
            w[idx] = wb[bi] * core.sample_dirichlet_floor([1] * len(idx), rng)
        cc1.append(core.topsis_cc(Yn, w))
    return pd.Series(1 - cc0, index=core.STATE_ORDER), pd.Series(1 - np.mean(cc1, 0), index=core.STATE_ORDER), panel


def indicator_ranks(panel):
    """Direction-adjusted ranks of the structural indicators (1 = most favourable; ties averaged)."""
    from scipy.stats import rankdata
    out = pd.DataFrame(index=panel.index)
    for n, b, g in core.INDICATOR_SYSTEM:
        v = panel[n].to_numpy(float)
        out[n] = rankdata(v if g == "cost" else -v, method="average")
    return out


def write_figure_data(results, m0, m1, out_dir):
    fd = os.path.join(out_dir, "figure_data")
    os.makedirs(fd, exist_ok=True)
    base = results["baseline"]
    m2 = 1 - base["cc_mean"]
    order = m2.sort_values().index
    json.dump({ABBR[s]: {"M0": round(float(m0[s]), 4), "M1": round(float(m1[s]), 4), "M2": round(float(m2[s]), 4)}
               for s in order}, open(os.path.join(fd, "m012.json"), "w"), indent=1)
    acc = pd.DataFrame(base["rank_accept"], index=core.STATE_ORDER)
    order = base["expected_rank"].sort_values().index
    json.dump({ABBR[s]: [round(100 * float(acc.loc[s, k]), 2) for k in range(4)] for s in order},
              open(os.path.join(fd, "rank_accept.json"), "w"), indent=1)
    dom_name = {"Climate": "Climate", "Energy": "Energy", "Adaptive": "Socio-econ."}
    prof = {}
    for s in order:
        prof[ABBR[s]] = {SCENARIO_LABEL[sc]: dict(domain=dom_name[results[sc]["dominant"][s]],
                                                  psel=round(float(results[sc]["P_sel"][s]), 3),
                                                  z=round(float(results[sc]["Z_resid"].loc[s, results[sc]["dominant"][s]]), 3),
                                                  identified=bool(results[sc]["identified"][s]))
                         for sc in SCENARIOS}
    json.dump(prof, open(os.path.join(fd, "domain_profiles.json"), "w"), indent=1)


def run_all(out_dir="REACT_ESR_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    hazard = core.load_hazard_panel()
    static = _configured_static()
    results = {}
    print(f"[spec] construct={ENERGY_CONSTRUCT} grid={GRID_NORMALISATION} M={N_MONTE_CARLO} "
          f"Mw={N_DIAGNOSTIC} R={N_PERMUTATION} seed={RNG_SEED}")
    for scenario in SCENARIOS:
        results[scenario] = run_scenario(static, hazard, scenario)
        write_scenario_results(results[scenario], scenario, out_dir)
    classification_audit(results).to_csv(os.path.join(out_dir, "classification_audit.csv"), index=False)

    fit = results["baseline"]["copula_fit"]
    pd.DataFrame([{
        "selected_model": fit["model"], "df": fit.get("student_t_df"), "n_complete": fit["n_complete"],
        "AIC_Gaussian": fit["aic_gaussian"], "AIC_Student_t": fit["aic_student_t"],
        "AIC_Clayton": fit["aic_clayton"], "AIC_Gumbel": fit["aic_gumbel"],
        "AIC_selected": fit["aic_selected"],
    }]).to_csv(os.path.join(out_dir, "copula_fit.csv"), index=False)

    summary = pd.DataFrame(index=core.STATE_ORDER)
    for sc, r in results.items():
        summary[f"rank_{sc}"] = r["expected_rank"]
        summary[f"provisional_{sc}"] = r["dominant"]
        summary[f"class_{sc}"] = r["classification"]
        summary[f"Psel_{sc}"] = r["P_sel"]
        summary[f"T_{sc}"] = r["T_joint"]
    summary.to_csv(os.path.join(out_dir, "cross_scenario_summary.csv"))

    from scipy.stats import spearmanr, pearsonr
    rows = []
    for a, b in (("baseline", "national_pool"), ("baseline", "adjacency_flow"), ("national_pool", "adjacency_flow")):
        rows.append(dict(pair=f"{a}~{b}", spearman=spearmanr(results[a]["expected_rank"], results[b]["expected_rank"]).statistic))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "scenario_rank_correlations.csv"), index=False)

    m0, m1, panel = m0_m1_vulnerability(static, hazard)
    pd.DataFrame({"M0": m0, "M1": m1, "M2": 1 - results["baseline"]["cc_mean"]}).to_csv(os.path.join(out_dir, "m0_m1_m2_vulnerability.csv"))
    panel.to_csv(os.path.join(out_dir, "structural_indicator_panel.csv"))
    indicator_ranks(panel).to_csv(os.path.join(out_dir, "indicator_ranks.csv"))
    B = results["baseline"]["B_blocks"]
    B.to_csv(os.path.join(out_dir, "domain_scores_baseline.csv"))
    corr = []
    for a, b in (("Climate", "Energy"), ("Climate", "Adaptive"), ("Energy", "Adaptive")):
        corr.append(dict(pair=f"{a}~{b}", pearson=pearsonr(B[a], B[b]).statistic, spearman=spearmanr(B[a], B[b]).statistic))
    pd.DataFrame(corr).to_csv(os.path.join(out_dir, "domain_score_correlations.csv"), index=False)
    write_figure_data(results, m0, m1, out_dir)
    return results


# =============================================================================
# Robustness analyses
# =============================================================================
HEADLINE_RULE = 0.70


def _with_spec(fn, **spec):
    """Run fn() under temporary module-level specification changes."""
    global ENERGY_CONSTRUCT, GRID_NORMALISATION
    saved = dict(construct=ENERGY_CONSTRUCT, grid=GRID_NORMALISATION,
                 derating=deepcopy(core.PARAMS_DERATING), system=list(core.INDICATOR_SYSTEM),
                 dirichlet=core.sample_dirichlet_floor, blocks=deepcopy(core.BLOCK_INDICATORS))
    try:
        if "construct" in spec:
            ENERGY_CONSTRUCT = spec["construct"]
        if "grid" in spec:
            GRID_NORMALISATION = spec["grid"]
        if "derating" in spec:
            core.PARAMS_DERATING.clear(); core.PARAMS_DERATING.update(spec["derating"])
        if spec.get("thermal_cost"):
            core.INDICATOR_SYSTEM[:] = [(n, b, ("cost" if n == "Thermal_MW" else g)) for n, b, g in core.INDICATOR_SYSTEM]
            core._rebuild_indicator_derivatives()
        if spec.get("equal_weights"):
            core.sample_dirichlet_floor = lambda alpha, rng, **k: np.full(len(alpha), 1.0 / len(alpha))
        if "blocks" in spec:
            core.BLOCK_INDICATORS = spec["blocks"]
        return fn()
    finally:
        ENERGY_CONSTRUCT, GRID_NORMALISATION = saved["construct"], saved["grid"]
        core.PARAMS_DERATING.clear(); core.PARAMS_DERATING.update(saved["derating"])
        core.INDICATOR_SYSTEM[:] = saved["system"]; core._rebuild_indicator_derivatives()
        core.sample_dirichlet_floor = saved["dirichlet"]
        core.BLOCK_INDICATORS = saved["blocks"]


LOW_RESPONSE = {"PV": dict(beta=0.05, delta_max=0.05), "Thermal": dict(alpha=0.10, beta=0.05, delta_max=0.15),
                "Grid": dict(beta=0.05, delta_max=0.05, gamma=0.10), "Wind": dict(beta=0.20, delta_max=0.20)}
HIGH_RESPONSE = {"PV": dict(beta=0.15, delta_max=0.15), "Thermal": dict(alpha=0.20, beta=0.15, delta_max=0.35),
                 "Grid": dict(beta=0.15, delta_max=0.15, gamma=0.30), "Wind": dict(beta=0.40, delta_max=0.40)}
SPECIFICATIONS = {
    "main": {},
    "absolute_capacities": dict(construct="absolute"),
    "grid_per_demand": dict(grid="demand"),
    "grid_geometric_mean": dict(grid="geomean"),
    "equal_weights": dict(equal_weights=True),
    "low_hazard_response": dict(derating=LOW_RESPONSE),
    "high_hazard_response": dict(derating=HIGH_RESPONSE),
    "thermal_as_cost": dict(thermal_cost=True),
}


def _three_scenarios(static, hazard, n_perm=20):
    return {sc: run_scenario(static, hazard, sc, verbose=False, n_perm=n_perm) for sc in SCENARIOS}


def specification_sensitivity(static, hazard, main_results, out_dir):
    from scipy.stats import spearmanr
    rows = []
    for name, spec in SPECIFICATIONS.items():
        res = main_results if name == "main" else _with_spec(lambda: _three_scenarios(static, hazard), **spec)
        for sc in SCENARIOS:
            r = res[sc]
            rho = spearmanr(r["expected_rank"], main_results[sc]["expected_rank"]).statistic
            for s in core.STATE_ORDER:
                rows.append(dict(specification=name, scenario=SCENARIO_LABEL[sc], state=s,
                                 provisional=r["dominant"][s], P_sel=r["P_sel"][s], T=r["T_joint"][s],
                                 identified=bool(r["identified"][s]), expected_rank=r["expected_rank"][s],
                                 rank_rho_vs_main=rho))
        print(f"  spec {name}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "robustness_specifications.csv"), index=False)
    return df


def threshold_sweep(audit, out_dir):
    rows = []
    for tau in (0.60, 0.65, 0.70, 0.75, 0.80):
        f = audit[(audit.P_sel >= tau) & (audit.T_joint > 0)]
        for (state, dom), g in f.groupby(["State", "Provisional_domain"]):
            rows.append(dict(threshold=tau, state=state, domain=dom, n_scenarios=len(g),
                             scenarios="; ".join(g.Scenario)))
        rows.append(dict(threshold=tau, state="__total__", domain="", n_scenarios=len(f), scenarios=""))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "robustness_threshold_sweep.csv"), index=False)
    return df


def panel_deletion(static, hazard, main_results, out_dir, scenarios=SCENARIOS):
    full = list(core.STATE_ORDER)
    rows = []
    try:
        for d in full:
            core.STATE_ORDER = [s for s in full if s != d]
            st = static.loc[core.STATE_ORDER]
            hz = hazard[hazard["State"] != d].reset_index(drop=True)
            for sc in scenarios:
                r = run_scenario(st, hz, sc, verbose=False, n_perm=20)
                for s in core.STATE_ORDER:
                    rows.append(dict(deleted=d, scenario=SCENARIO_LABEL[sc], state=s,
                                     provisional=r["dominant"][s], provisional_full=main_results[sc]["dominant"][s],
                                     P_sel=r["P_sel"][s], T=r["T_joint"][s], identified=bool(r["identified"][s]),
                                     identified_full=bool(main_results[sc]["identified"][s])))
            print(f"  panel deletion {d}: done", flush=True)
    finally:
        core.STATE_ORDER = full
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "robustness_panel_deletion.csv"), index=False)
    return df


def seed_replicas(static, hazard, out_dir, seeds=(12345, 999, 7, 42, 2025)):
    global RNG_SEED
    from scipy.stats import spearmanr
    saved = RNG_SEED
    rows = []
    try:
        for seed in seeds:
            RNG_SEED = seed
            for sc in SCENARIOS:
                r = run_scenario(static, hazard, sc, verbose=False, n_perm=20)
                for s in core.STATE_ORDER:
                    rows.append(dict(seed=seed, scenario=SCENARIO_LABEL[sc], state=s, expected_rank=r["expected_rank"][s],
                                     provisional=r["dominant"][s], P_sel=r["P_sel"][s], T=r["T_joint"][s],
                                     identified=bool(r["identified"][s])))
            print(f"  seed {seed}: done", flush=True)
    finally:
        RNG_SEED = saved
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "robustness_seed_replicas.csv"), index=False)
    summ = []
    for sc in SCENARIOS:
        g = df[df.scenario == SCENARIO_LABEL[sc]]
        er = g.pivot(index="state", columns="seed", values="expected_rank")
        ps = g.pivot(index="state", columns="seed", values="P_sel")
        dm = g.pivot(index="state", columns="seed", values="provisional")
        idf = g.pivot(index="state", columns="seed", values="identified")
        rhos = [spearmanr(er[a], er[b]).statistic for i, a in enumerate(seeds) for b in seeds[i + 1:]]
        summ.append(dict(scenario=SCENARIO_LABEL[sc], sd_max_expected_rank=er.std(axis=1, ddof=1).max(),
                         sd_max_Psel=ps.std(axis=1, ddof=1).max(), min_rank_rho=min(rhos),
                         same_provisional=int((dm.nunique(axis=1) == 1).sum()),
                         same_classification=int((idf.nunique(axis=1) == 1).sum())))
    pd.DataFrame(summ).to_csv(os.path.join(out_dir, "robustness_seed_summary.csv"), index=False)
    return df


def redundancy(static, hazard, main_results, out_dir):
    """Within-domain VIF and correlation-cluster indicator reduction."""
    from scipy.stats import spearmanr
    _configure_core()
    panel = core.build_baseline_panel(static, hazard)
    vif_rows = []
    for b, names in core.BLOCK_INDICATORS.items():
        X = panel[names].to_numpy(float)
        X = (X - X.mean(0)) / X.std(0)
        for j, n in enumerate(names):
            others = [k for k in range(len(names)) if k != j]
            A = np.column_stack([np.ones(len(X)), X[:, others]])
            beta = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
            res = X[:, j] - A @ beta
            r2 = 1 - res.var() / X[:, j].var()
            vif_rows.append(dict(domain=b, indicator=n, VIF=1 / max(1 - r2, 1e-9)))
    pd.DataFrame(vif_rows).to_csv(os.path.join(out_dir, "robustness_vif_within_domain.csv"), index=False)

    # Energy System VIF under the load-relative and absolute constructs
    def _energy_vif():
        _configure_core()
        pe = core.build_baseline_panel(static, hazard)
        names = core.BLOCK_INDICATORS["Energy"]
        X = pe[names].to_numpy(float); X = (X - X.mean(0)) / X.std(0)
        out = {}
        for j, n in enumerate(names):
            A = np.column_stack([np.ones(len(X)), np.delete(X, j, 1)])
            beta = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
            r2 = 1 - (X[:, j] - A @ beta).var() / X[:, j].var()
            out[n] = 1 / max(1 - r2, 1e-9)
        return pd.Series(out)
    pd.DataFrame({"load_relative": _with_spec(_energy_vif, construct="load_relative"),
                  "absolute": _with_spec(_energy_vif, construct="absolute")}).to_csv(
        os.path.join(out_dir, "robustness_vif_energy_by_construct.csv"))

    # Maximum leverage in the three relative-weakness regressions (Baseline mean domain scores)
    B = main_results["baseline"]["B_blocks"].to_numpy(float)
    lev = []
    for b, name in enumerate(("Climate", "Energy", "Adaptive")):
        X = np.column_stack([np.ones(len(B)), np.delete(B, b, 1)])
        h = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
        lev.append(dict(regression=name, max_leverage=h.max(), state=main_results["baseline"]["B_blocks"].index[h.argmax()]))
    pd.DataFrame(lev).to_csv(os.path.join(out_dir, "regression_leverage.csv"), index=False)

    base_rank = main_results["baseline"]["expected_rank"]
    rows = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        kept = {}
        for b, names in core.BLOCK_INDICATORS.items():
            rho = panel[names].corr(method="spearman").abs().to_numpy()
            remaining, keep = list(range(len(names))), []
            while remaining:                         # connected components of |rho| > tau
                comp, stack = set(), [remaining[0]]
                while stack:
                    i = stack.pop()
                    if i in comp:
                        continue
                    comp.add(i)
                    stack += [k for k in remaining if k not in comp and rho[i, k] > tau]
                keep.append(min(comp))                # retain the first-listed member
                remaining = [k for k in remaining if k not in comp]
            kept[b] = [names[k] for k in sorted(keep)]
        n_kept = sum(len(v) for v in kept.values())
        r = _with_spec(lambda: run_scenario(static, hazard, "baseline", verbose=False, n_perm=20), blocks=kept)
        rk_full = base_rank.rank(); rk_red = r["expected_rank"].rank()
        shift = (rk_full - rk_red).abs()
        rows.append(dict(tau=tau, n_kept=n_kept, kept="; ".join(sum(kept.values(), [])),
                         rho=spearmanr(base_rank, r["expected_rank"]).statistic,
                         max_shift=shift.max(), mean_shift=shift.mean(),
                         identified="; ".join(f"{s}:{r['dominant'][s]}" for s in core.STATE_ORDER if r["identified"][s])))
        print(f"  redundancy tau={tau}: n_kept={n_kept}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "robustness_indicator_reduction.csv"), index=False)
    return df


def run_robustness(out_dir, main_results=None):
    os.makedirs(out_dir, exist_ok=True)
    hazard = core.load_hazard_panel()
    static = _configured_static()
    if main_results is None:
        main_results = _three_scenarios(static, hazard)
    audit = classification_audit(main_results)
    threshold_sweep(audit, out_dir)
    specification_sensitivity(static, hazard, main_results, out_dir)
    redundancy(static, hazard, main_results, out_dir)
    seed_replicas(static, hazard, out_dir)
    panel_deletion(static, hazard, main_results, out_dir)


# =============================================================================
# Synthetic recovery of planted domain deviations
# =============================================================================
def run_recovery(out_dir, deltas=(0.10, 0.20, 0.40, 0.80, 1.20), n_panels=200, n_draws=2000,
                 n_shocked=12, loading=0.7, noise=0.5, seed=202):
    """Planted negative deviations in 12 of 16 regions; PRESS residuals and the P_sel/T rule."""
    rng = np.random.default_rng(seed)
    N, K = 16, 3
    rows = []
    for delta in deltas:
        correct = ident_correct = total = 0
        for _ in range(n_panels):
            f = rng.normal(0, 1, N)
            B = np.column_stack([loading * f + noise * rng.normal(0, 1, N) for _ in range(K)])
            idx = rng.choice(N, n_shocked, replace=False)
            tgt = rng.integers(0, K, n_shocked)
            B[idx, tgt] -= delta * B.std(axis=0)[tgt]
            R, SE = _residualise(B)
            Rq = R[None] + rng.normal(0, 1, (n_draws, N, K)) * SE[None]
            sd = Rq.std(axis=1, keepdims=True)
            z = (Rq - Rq.mean(axis=1, keepdims=True)) / np.where(sd < 1e-12, 1, sd)
            win = z.argmin(axis=2)
            P = np.stack([(win == b).mean(axis=0) for b in range(K)], axis=1)
            cand = P.argmax(axis=1); psel = P.max(axis=1); med = np.median(z, axis=0)
            T = np.array([_joint_statistic(med[i], int(cand[i])) for i in range(N)])
            ok = cand[idx] == tgt
            correct += int(ok.sum())
            ident_correct += int((ok & (psel[idx] >= HEADLINE_RULE) & (T[idx] > 0)).sum())
            total += n_shocked
        rows.append(dict(delta=delta, provisional_correct=correct / total, identified_correct=ident_correct / total))
        print(f"  recovery delta={delta}: provisional {correct / total:.1%}, identified {ident_correct / total:.1%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "recovery_planted_deviation.csv"), index=False)
    return df


def main():
    global N_MONTE_CARLO, N_DIAGNOSTIC, N_PERMUTATION, RNG_SEED, RESIDUAL_MODE, ENERGY_CONSTRUCT, GRID_NORMALISATION
    p = argparse.ArgumentParser(description="REACT ESR reproducibility pipeline")
    p.add_argument("output_dir", nargs="?", default="REACT_ESR_outputs")
    p.add_argument("--mode", choices=("run", "robustness", "recovery", "all"), default="run")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--mc", type=int)
    p.add_argument("--diagnostic", type=int)
    p.add_argument("--permutations", type=int)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--energy-construct", choices=("load_relative", "absolute"), default="load_relative")
    p.add_argument("--grid-normalisation", choices=("area", "demand", "geomean"), default="area")
    p.add_argument("--residual-mode", choices=("press", "legacy"), default="press")
    args = p.parse_args()
    if args.fast:
        N_MONTE_CARLO, N_DIAGNOSTIC, N_PERMUTATION = 250, 1000, 100
    if args.mc:
        N_MONTE_CARLO = args.mc
    if args.diagnostic:
        N_DIAGNOSTIC = args.diagnostic
    if args.permutations:
        N_PERMUTATION = args.permutations
    RNG_SEED = args.seed
    RESIDUAL_MODE = args.residual_mode
    ENERGY_CONSTRUCT = args.energy_construct
    GRID_NORMALISATION = args.grid_normalisation
    main_results = None
    if args.mode in ("run", "all"):
        main_results = run_all(args.output_dir)
    if args.mode in ("robustness", "all"):
        run_robustness(os.path.join(args.output_dir, "robustness"), main_results)
    if args.mode in ("recovery", "all"):
        run_recovery(os.path.join(args.output_dir, "robustness"))


if __name__ == "__main__":
    main()
