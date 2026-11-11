"""MCDA layers: REACT-PF (power-flow KPIs in the Energy System domain) and REACT-Lite comparisons.

* ``run_react(...)`` is a generic REACT Monte Carlo (copula hazards -> Climate Exposure,
  structural Socio-economic Capacity, and an Energy System block given by ``energy_spec``),
  followed by the manuscript's relative-weakness diagnostic (PRESS residuals, P_sel >= 0.70 and
  T > 0, label-permutation check). With ``energy_spec='lite'`` it reproduces REACT-Lite; with the
  PF specification the seven indicators of ``kpis.PF_INDICATORS`` replace the Lite Energy block.
* ``lite_m0`` / ``lite_m1``: deterministic (M0) and weights-only (M1) versions of the relative-
  weakness diagnostic, to show how classifications differ across model configurations (R3.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import REACT_v4 as core
import REACT_ESR_reproducible as R
from .network import STATES

HCOLS = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean", "Rx1day_mm_max",
         "Storm_Days_mean", "low_wind_days_mean"]
CLIMATE = [("Hot_Days", "hot_days_mean", "cost"), ("Soil_Moist", "soil_moist_winter_mean", "benefit"),
           ("P30_Days", "p30_days_mean", "cost"), ("Rx1day", "Rx1day_mm_mean", "cost"),
           ("Storm_Days", "Storm_Days_mean", "cost"), ("Low_Wind", "low_wind_days_mean", "cost")]
ADAPTIVE = [("GDP_pc", "benefit"), ("HDI", "benefit"), ("Patents", "benefit"), ("ADR", "cost")]
BLOCKS = ("Climate", "Energy", "Adaptive")
SC_OFFSET = {"baseline": 0, "national_pool": 101, "adjacency_flow": 202}


def _diagnose(block_iter, rng, n_perm):
    P, W, med, lo, hi, T, p_perm, q_bh, _ = R.diagnose_blocks(block_iter, rng, n_perm=n_perm)
    cand = P.argmax(1); psel = P.max(1)
    ident = (psel >= 0.70) & (T > 0)
    dom = np.array(BLOCKS)[cand]
    return pd.DataFrame(dict(provisional=dom, P_sel=psel, T=T, Z_med=med[np.arange(16), cand],
                             p_perm=p_perm, q_BH=q_bh, identified=ident,
                             classification=np.where(ident, dom, "Multidomain")), index=STATES), med


def run_react(static, hazard, scenario, energy_spec, M=2000, Mw=10000, n_perm=5000, seed=12345):
    """energy_spec: list of (name, direction, sampler) with sampler(rng, m) -> (16,) values."""
    R.N_DIAGNOSTIC = Mw
    fit = R.estimate_copula(hazard, HCOLS)
    R._configure_core()
    panel = core.build_baseline_panel(static, hazard)
    coherent = scenario != "baseline"
    rng = np.random.default_rng(seed + SC_OFFSET[scenario])
    names = [c[0] for c in CLIMATE] + [e[0] for e in energy_spec] + [a[0] for a in ADAPTIVE]
    is_cost = np.array([c[2] == "cost" for c in CLIMATE] + [e[1] == "cost" for e in energy_spec]
                       + [a[1] == "cost" for a in ADAPTIVE])
    blk = {"Climate": np.arange(len(CLIMATE)),
           "Energy": len(CLIMATE) + np.arange(len(energy_spec)),
           "Adaptive": len(CLIMATE) + len(energy_spec) + np.arange(len(ADAPTIVE))}
    J = len(names); N = 16
    adaptive = np.column_stack([panel.loc[STATES, a[0]].to_numpy(float) for a in ADAPTIVE])
    cc_iter = np.zeros((M, N)); rank_iter = np.zeros((M, N), int); block_iter = np.zeros((M, N, 3))
    for m in range(M):
        hz = core.sample_hazard_year(fit, HCOLS, coherent, rng)
        clim = np.column_stack([[hz[s][c] for s in STATES] for _, c, _ in CLIMATE])
        en = np.column_stack([e[2](rng, m) for e in energy_spec])
        Y = np.column_stack([clim, en, adaptive])
        Yn = core.normalised_indicators(Y, is_cost)
        wb = core.sample_dirichlet_floor([1.0] * 3, rng); w = np.zeros(J)
        for bi, b in enumerate(BLOCKS):
            ww = core.sample_dirichlet_floor([1.0] * len(blk[b]), rng)
            w[blk[b]] = wb[bi] * ww
            block_iter[m, :, bi] = Yn[:, blk[b]] @ ww
        cc = core.topsis_cc(Yn, w); cc_iter[m] = cc; rank_iter[m] = (-cc).argsort().argsort() + 1
    diag, med = _diagnose(block_iter, rng, n_perm)
    diag["expected_rank"] = rank_iter.mean(0); diag["CC_mean"] = cc_iter.mean(0)
    B = pd.DataFrame(block_iter.mean(0), index=STATES, columns=BLOCKS)
    acc = np.column_stack([(rank_iter == r).mean(0) for r in range(1, N + 1)])
    return dict(diag=diag, B=B, rank_accept=acc, block_iter=block_iter)


def dr_value(dr, mode="sat", scale=0.5, cap=3.0):
    """Value of deliverable supply. 'sat' (main): u = 1 - exp(-(DR-1)/scale), i.e. full value is
    approached once a state can serve its peak with a margin; surplus deliverability that only
    supports exports adds little resilience for the state's own consumers. Alternatives:
    'cap' = min(DR, cap); 'log' = log(DR); 'linear' = DR."""
    dr = np.asarray(dr, float)
    if mode == "sat":
        return 1.0 - np.exp(-(dr - 1.0) / scale)
    if mode == "cap":
        return np.minimum(dr, cap)
    if mode == "log":
        return np.log(dr)
    return dr


def pf_energy_spec(kpi_table, mc_tables, scenario, boot=True, dr_mode="sat", dr_scale=0.5, dr_cap=3.0,
                   local_firm=False, drop=()):
    """Samplers for the PF Energy System indicators. E1/E2 are bootstrap means over Monte Carlo years."""
    tab = mc_tables[scenario]
    eens, lole, dem = tab["eens"], tab["lole"], tab["demand"]
    ny = len(eens)
    st = kpi_table.loc[STATES]

    def boot_idx(rng):
        return rng.integers(0, ny, ny) if boot else np.arange(ny)

    cache = {}

    def e1(rng, m):
        idx = boot_idx(rng); cache["idx"] = idx
        return 1e6 * eens[idx].mean(0) / dem[idx].mean(0)

    def e2(rng, m):
        return lole[cache["idx"]].mean(0)

    fixed = lambda v: (lambda rng, m: v)
    v = lambda c: dr_value(st[c].to_numpy(float), dr_mode, dr_scale, dr_cap)
    spec = [("EENS_ppm", "cost", e1), ("LOLE_h", "cost", e2),
            ("DR_winter", "benefit", fixed(v("DR_winter_N1"))),
            ("DR_heat", "benefit", fixed(v("DR_heat_N1"))),
            ("Storm_loss_h", "cost", fixed(st.storm_EENS_event_h.to_numpy(float))),
            ("Exposure", "cost", fixed(st[f"Exposure_{scenario}"].to_numpy(float)))]
    if local_firm:
        spec.append(("Local_firm", "benefit", fixed(st.Local_firm.to_numpy(float))))
    return [x for x in spec if x[0] not in drop]


def lite_energy_spec(static, hazard, scenario):
    """Not used for the main Lite run (R.run_scenario is used); kept for symmetry."""
    raise NotImplementedError


# ------------------------------------------------------------------------------------------------
# REACT-Lite model configurations M0 / M1 for the relative-weakness diagnostic (R3.5, R1.8)
# ------------------------------------------------------------------------------------------------
def _lite_matrix(static, hazard):
    R._configure_core()
    panel = core.build_baseline_panel(static, hazard)
    cols = [n for n, _, _ in core.INDICATOR_SYSTEM]
    Yn = core.normalised_indicators(panel[cols].to_numpy(float), np.array([core.IS_COST[c] for c in cols]))
    blk = {b: np.array([cols.index(c) for c in core.BLOCK_INDICATORS[b]]) for b in BLOCKS}
    return Yn, blk


def deterministic_diagnosis(B):
    """Single-profile relative weakness: lowest standardised PRESS residual, adverse and separated (T>0)."""
    Rr, _ = R._residualise(B)
    Z = (Rr - Rr.mean(0)) / Rr.std(0)
    cand = Z.argmin(1)
    T = np.array([R._joint_statistic(Z[i], int(cand[i])) for i in range(16)])
    dom = np.array(BLOCKS)[cand]
    lowest_score = np.array(BLOCKS)[B.argmin(1)]
    return pd.DataFrame(dict(provisional=dom, Z=Z[np.arange(16), cand], T=T, identified=T > 0,
                             classification=np.where(T > 0, dom, "Multidomain"), lowest_domain_score=lowest_score),
                        index=STATES)


def lite_m0(static, hazard):
    Yn, blk = _lite_matrix(static, hazard)
    B = np.column_stack([Yn[:, blk[b]].mean(1) for b in BLOCKS])
    return deterministic_diagnosis(B), B


def lite_m1(static, hazard, M=2000, Mw=10000, n_perm=5000, seed=12345):
    R.N_DIAGNOSTIC = Mw
    Yn, blk = _lite_matrix(static, hazard)
    rng = np.random.default_rng(seed)
    block_iter = np.zeros((M, 16, 3))
    for m in range(M):
        for bi, b in enumerate(BLOCKS):
            ww = core.sample_dirichlet_floor([1.0] * len(blk[b]), rng)
            block_iter[m, :, bi] = Yn[:, blk[b]] @ ww
    diag, _ = _diagnose(block_iter, rng, n_perm)
    return diag


def pf_m0(static, hazard, kpi_table, mc_tables, scenario, **kw):
    R._configure_core()
    panel = core.build_baseline_panel(static, hazard)
    spec = pf_energy_spec(kpi_table, mc_tables, scenario, boot=False, **kw)
    rng = np.random.default_rng(0)
    clim = np.column_stack([panel.loc[STATES, c[0]].to_numpy(float) for c in CLIMATE])
    en = np.column_stack([e[2](rng, 0) for e in spec])
    ad = np.column_stack([panel.loc[STATES, a[0]].to_numpy(float) for a in ADAPTIVE])
    cost = np.array([c[2] == "cost" for c in CLIMATE] + [e[1] == "cost" for e in spec] + [a[1] == "cost" for a in ADAPTIVE])
    Yn = core.normalised_indicators(np.column_stack([clim, en, ad]), cost)
    nC, nE = len(CLIMATE), len(spec)
    B = np.column_stack([Yn[:, :nC].mean(1), Yn[:, nC:nC + nE].mean(1), Yn[:, nC + nE:].mean(1)])
    return deterministic_diagnosis(B), B


# ------------------------------------------------------------------------------------------------
# Evidence tiers (R3.6)
# ------------------------------------------------------------------------------------------------
def evidence_tiers(lite_audit, pf_diag, structural):
    """Combine REACT-Lite, REACT-PF and the physical tests into planning-evidence tiers (R3.6).

    A  demonstrated  - relative weakness identified (P_sel>=0.70, T>0) in BOTH tiers in at least one
                       scenario each (robust to how the Energy System is measured), or, for the
                       Energy System, a physical N-1 deficit (deliverability ratio < 1 in a snapshot).
    B  indicated     - identified in one tier only; or provisional in both tiers with max P_sel>=0.60.
                       Socio-economic Capacity is capped at B because implementation and institutional
                       capacity are not measured.
    C  investigate   - provisional direction (max P_sel >= 0.50 in either tier) without the above.
    The shortfall exposure of the network stress test (E6 >= 1.5) is reported as information.
    """
    rows = []
    for s in STATES:
        la = lite_audit[lite_audit.State == s]; pa = pf_diag[pf_diag.State == s]
        for dom in BLOCKS:
            lid = int(((la.Provisional_domain == dom) & la.identified).sum())
            pid = int(((pa.provisional == dom) & pa.identified).sum())
            lpv = int((la.Provisional_domain == dom).sum()); ppv = int((pa.provisional == dom).sum())
            lmax = float(la.loc[la.Provisional_domain == dom, "P_sel"].max()) if lpv else 0.0
            pmax = float(pa.loc[pa.provisional == dom, "P_sel"].max()) if ppv else 0.0
            dr_min = float(structural.loc[s, ["DR_winter_N1", "DR_heat_N1"]].min())
            expo = float(structural.loc[s, "Exposure_adjacency_flow"])
            phys_deficit = dom == "Energy" and dr_min < 1.0
            phys_exposure = dom == "Energy" and expo >= 1.5
            if lid > 0 and pid > 0 or phys_deficit:
                tier = "A"
            elif lid > 0 or pid > 0 or (lmax >= 0.60 and pmax >= 0.60):
                tier = "B"
            elif max(lmax, pmax) >= 0.50:
                tier = "C"
            else:
                continue
            if dom == "Adaptive" and tier == "A":
                tier = "B"
            rows.append(dict(state=s, domain=dom, lite_identified=lid, pf_identified=pid, lite_provisional=lpv,
                             pf_provisional=ppv, lite_Psel_max=lmax, pf_Psel_max=pmax, DR_min=dr_min,
                             exposure=expo, physical_deficit=phys_deficit, physical_exposure=phys_exposure,
                             tier=tier))
    return pd.DataFrame(rows)
