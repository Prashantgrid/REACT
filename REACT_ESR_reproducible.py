#!/usr/bin/env python3
"""Manuscript-aligned REACT reproducibility entry point.

This module reuses the embedded data and simulation engine in ``REACT_v4.py``
and replaces the final diagnostic/calibration stage with the specification in
"Beyond Composite Rankings: A Probabilistic Multi-Domain Framework for
Regional Energy Resilience".

Default specification
---------------------
* absolute installed-capacity Energy indicators (demand-relative = sensitivity)
* M=2000 hazard iterations
* Mw=10000 profile draws
* R=500 state-specific label-permutation replicates
* seed=12345
* PRESS/leave-one-state-out residual e_i/(1-h_i)
* calibrated differentiation when q<=0.10, Psel>=0.70, and T>0
"""
from __future__ import annotations

import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import REACT_v4 as core

N_MONTE_CARLO = 2000
N_DIAGNOSTIC = 10000
N_PERMUTATION = 500
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

    key = (tuple(hcols), len(hazard), int(hazard["Year"].min()), int(hazard["Year"].max()))
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


def _configured_static():
    static = core.load_static_panel().loc[core.STATE_ORDER].copy()
    # Table H.10 value used in the ESR manuscript.
    static.loc["Hamburg", "Solar_PV_MW"] = 250.0
    return static


def _configure_core():
    core.N_MONTE_CARLO = N_MONTE_CARLO
    core.RNG_SEED = RNG_SEED
    core.ENERGY_CONSTRUCT = "absolute"
    core.NORMALISATION = "minmax"
    core.REGRESSION_NOISE_SCALE = REGRESSION_NOISE_SCALE
    core._residualise = _residualise
    core.estimate_copula = estimate_copula


def run_scenario(static, hazard, scenario, verbose=True):
    """Run the simulation engine, then replace the legacy classifier with the ESR diagnostic."""
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
    P, W, med, lo, hi, T, p_perm, q_bh, calibrated = diagnose_blocks(result["block_iter"], rng)
    cols = ["Climate", "Energy", "Adaptive"]
    Pdf = pd.DataFrame(P, index=core.STATE_ORDER, columns=cols)
    Wdf = pd.DataFrame(W, index=core.STATE_ORDER, columns=cols)
    Zdf = pd.DataFrame(med, index=core.STATE_ORDER, columns=cols)
    result["P_dominance"] = Pdf
    result["P_negative"] = Wdf
    result["Z_resid"] = Zdf
    result["Z_resid_lo"] = pd.DataFrame(lo, index=core.STATE_ORDER, columns=cols)
    result["Z_resid_hi"] = pd.DataFrame(hi, index=core.STATE_ORDER, columns=cols)
    result["dominant"] = Pdf.idxmax(axis=1)
    result["T_joint"] = pd.Series(T, index=core.STATE_ORDER)
    result["p_permutation"] = pd.Series(p_perm, index=core.STATE_ORDER)
    result["q_bh"] = pd.Series(q_bh, index=core.STATE_ORDER)
    result["calibrated"] = pd.Series(calibrated, index=core.STATE_ORDER)
    result["classification"] = pd.Series(np.where(calibrated, result["dominant"].to_numpy(), "Multidomain"), index=core.STATE_ORDER)
    result["B_resid"] = pd.DataFrame(_residualise(result["B_blocks"].to_numpy())[0], index=core.STATE_ORDER, columns=cols)
    if verbose:
        print(f"{scenario}: {int(calibrated.sum())} calibrated state-domain profiles")
    return result


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
    df["P_sel"] = P.max(axis=1)
    df["T_joint"] = result["T_joint"]
    df["p_permutation"] = result["p_permutation"]
    df["q_BH"] = result["q_bh"]
    df["calibrated"] = result["calibrated"]
    df.to_csv(os.path.join(out_dir, f"results_{scenario}.csv"))


def run_all(out_dir="REACT_ESR_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    hazard = core.load_hazard_panel()
    static = _configured_static()
    results = {}
    for scenario in ("baseline", "national_pool", "adjacency_flow"):
        results[scenario] = run_scenario(static, hazard, scenario)
        write_scenario_results(results[scenario], scenario, out_dir)

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
        summary[f"Psel_{sc}"] = r["P_dominance"].max(axis=1)
        summary[f"T_{sc}"] = r["T_joint"]
        summary[f"q_{sc}"] = r["q_bh"]
    summary.to_csv(os.path.join(out_dir, "cross_scenario_summary.csv"))
    return results


def main():
    global N_MONTE_CARLO, N_DIAGNOSTIC, N_PERMUTATION, RNG_SEED, RESIDUAL_MODE
    p = argparse.ArgumentParser(description="REACT ESR reproducibility pipeline")
    p.add_argument("output_dir", nargs="?", default="REACT_ESR_outputs")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--mc", type=int)
    p.add_argument("--diagnostic", type=int)
    p.add_argument("--permutations", type=int)
    p.add_argument("--seed", type=int, default=12345)
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
    run_all(args.output_dir)


if __name__ == "__main__":
    main()
