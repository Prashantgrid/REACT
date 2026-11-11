"""Zonal flow-based DC-OPF on the nodal network (PTDF x GSK), with lazy line and N-1 constraints.

One linear programme serves three uses:
  * deliverability tests (maximum load a state can serve in a stress snapshot, N-0 or N-1);
  * hourly adequacy dispatch with load shedding in stress windows (chronological, with storage);
  * storm-event restoration (time-varying post-outage PTDF).
Variables are zonal (state x technology), injections are distributed to buses with fixed
generation-shift keys, and flows on every 220/380 kV branch are enforced with the nodal PTDF,
so Kirchhoff's voltage law and physical thermal ratings hold on the full network. Branch
constraints are added lazily (only violated branch-hours enter the LP), which is exact at
convergence.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .network import TECHS, STORAGE, STATES

COST = {"LIG": 30.0, "COAL": 60.0, "GAS": 90.0, "OIL": 150.0, "BIO": 40.0, "ROR": 0.0, "WON": 0.0,
        "WOFF": 0.0, "PV": 0.0}
IMPORT_COST, EXPORT_VALUE = 85.0, 45.0
VOLL = 10000.0
# Convex shedding segments (shares of zonal load): shortfalls that the network allows to be shared
# are shared in proportion to demand to within 0.5 % of load; any deviation from pro-rata sharing is
# therefore caused by network constraints (E6).
SHED_SEG = np.r_[np.full(10, 0.005), np.full(5, 0.01), [0.10, 0.80]]
SHED_COST = VOLL * (1.0 + 0.002 * np.arange(len(SHED_SEG)))
ETA_C = {"PHS": 0.87, "BAT": 0.95}; ETA_D = {"PHS": 0.87, "BAT": 0.95}


class ZonalModel:
    """Holds zonal PTDFs for a given topology. ``Z`` from network.zonal_ptdf."""

    def __init__(self, grid, fleet, Z, rating=None, lodf=None, radial=None):
        self.grid, self.fleet, self.Z = grid, fleet, Z
        self.nbr = Z["LOAD"].shape[0]
        self.F = grid.branches.s_nom.values.copy() if rating is None else rating
        self.lodf, self.radial = lodf, radial
        self.nnb = len(fleet.neighbours)
        self.cnec = grid.branches["cnec"].values if "cnec" in grid.branches else np.ones(self.nbr, bool)
        self.hvdc_cap = getattr(fleet, "hvdc_cap", None)


# ------------------------------------------------------------------------------------------------
# Generic hourly-block LP
# ------------------------------------------------------------------------------------------------
def solve_block(zm, avail, load, nb_imp, nb_exp, rating_pu, *, stor_soc0=None, network=True,
                maximise_load_zone=None, fixed_load_zone_cap=None, n1=False, Zt=None, island_load=None,
                max_iter=30, tol=1e-3, curtail_cost=0.0, seed_lines=None):
    """Solve a T-hour block.

    avail : (T,16,K) available generation MW per tech;  load : (T,16) MW
    nb_imp, nb_exp : (T,nnb) MW;  rating_pu : (T,nbr) multiplier on s_nom
    stor_soc0 : (16,2) initial state of charge MWh (None -> full)
    maximise_load_zone : z -> deliverability test (single hour), returns served load
    Zt : optional list of per-hour zonal PTDF dicts (time-varying topology)
    island_load : (T,16) load that is islanded (unservable) and excluded from the balance
    """
    T = load.shape[0]
    fl = zm.fleet
    K = len(TECHS)
    Pst = fl.stor_p[STORAGE].values; Est = fl.stor_e[STORAGE].values
    # ---- variable layout per hour
    gen_idx = [(z, k) for z in range(16) for k in range(K) if fl.cap.values[z, k] > 0]
    st_idx = [(z, s) for z in range(16) for s in range(2) if Pst[z, s] > 0]
    nG, nS, nN = len(gen_idx), len(st_idx), zm.nnb
    nSh = 16 * len(SHED_SEG)
    hv = zm.Z.get("HVDC")
    nH = 0 if hv is None else hv.shape[1]
    nLz = 1 if maximise_load_zone is not None else 0
    per = nG + 2 * nS + nS + nSh + nN + nLz + nH
    oG, oD, oC, oE, oSh, oN = 0, nG, nG + nS, nG + 2 * nS, nG + 3 * nS, nG + 3 * nS + nSh
    oL = oN + nN
    oH = oL + nLz
    nv = per * T
    c = np.zeros(nv); lb = np.zeros(nv); ub = np.zeros(nv)
    gz = np.array([z for z, _ in gen_idx], int); gk = np.array([k for _, k in gen_idx], int)
    sz = np.array([z for z, _ in st_idx], int); ss = np.array([s for _, s in st_idx], int)
    shz = np.repeat(np.arange(16), len(SHED_SEG)); shj = np.tile(np.arange(len(SHED_SEG)), 16)
    gcost = np.array([COST[TECHS[k]] for k in gk])
    for t in range(T):
        b = t * per
        ub[b + oG:b + oG + nG] = avail[t, gz, gk]; c[b + oG:b + oG + nG] = gcost
        ub[b + oD:b + oD + nS] = Pst[sz, ss]; c[b + oD:b + oD + nS] = 0.5
        ub[b + oC:b + oC + nS] = Pst[sz, ss]; c[b + oC:b + oC + nS] = 0.0
        ub[b + oE:b + oE + nS] = Est[sz, ss]
        lshed = load[t] if island_load is None else load[t] - island_load[t]
        ub[b + oSh:b + oSh + nSh] = SHED_SEG[shj] * lshed[shz]; c[b + oSh:b + oSh + nSh] = SHED_COST[shj]
        lb[b + oN:b + oN + nN] = -nb_exp[t]; ub[b + oN:b + oN + nN] = nb_imp[t]
        c[b + oN:b + oN + nN] = IMPORT_COST
        if nH:
            lb[b + oH:b + oH + nH] = -zm.hvdc_cap; ub[b + oH:b + oH + nH] = zm.hvdc_cap; c[b + oH:b + oH + nH] = 0.01
        if maximise_load_zone is not None:
            ub[b + oL] = np.inf if fixed_load_zone_cap is None else fixed_load_zone_cap
            c[b + oL] = -1e5
            c[b + oSh:b + oSh + nSh] = 1e7       # no shedding elsewhere in a deliverability test
    # export value: asymmetric -> split not needed for adequacy; small linear cost keeps LP bounded
    # ---- equality rows: national balance (T) and storage dynamics (nS*T)
    rows, cols, vals, beq = [], [], [], []
    r = 0
    for t in range(T):
        b = t * per
        idx = np.r_[b + oG:b + oG + nG, b + oD:b + oD + nS, b + oSh:b + oSh + nSh, b + oN:b + oN + nN]
        v = np.r_[np.ones(nG), np.ones(nS), np.ones(nSh), np.ones(nN)]
        idx = np.r_[idx, b + oC + np.arange(nS)]; v = np.r_[v, -np.ones(nS)]
        L = load[t].sum() - (0 if island_load is None else island_load[t].sum())
        if maximise_load_zone is not None:
            idx = np.r_[idx, b + oL]; v = np.r_[v, -1.0]
        rows += [r] * len(idx); cols += list(idx); vals += list(v); beq.append(L); r += 1
    soc0 = Est[sz, ss] if stor_soc0 is None else stor_soc0[sz, ss]
    eta_c = np.array([ETA_C[STORAGE[s]] for s in ss]); eta_d = np.array([ETA_D[STORAGE[s]] for s in ss])
    for t in range(T):
        b = t * per
        for j in range(nS):
            # E_t - E_{t-1} - eta_c*C_t + D_t/eta_d = 0
            ii = [b + oE + j, b + oC + j, b + oD + j]; vv = [1.0, -eta_c[j], 1.0 / eta_d[j]]
            if t > 0:
                ii.append((t - 1) * per + oE + j); vv.append(-1.0); rhs = 0.0
            else:
                rhs = soc0[j]
            rows += [r] * len(ii); cols += ii; vals += vv; beq.append(rhs); r += 1
    Aeq = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv)); beq = np.array(beq)

    # ---- flow sensitivities per hour: f_t = H_t x_t + f0_t
    def H_of(t):
        Zh = zm.Z if Zt is None else Zt[t]
        Hg = np.column_stack([Zh[TECHS[k]][:, z] for z, k in gen_idx]) if nG else np.zeros((zm.nbr, 0))
        Hs = np.column_stack([Zh[STORAGE[s]][:, z] for z, s in st_idx]) if nS else np.zeros((zm.nbr, 0))
        Hsh = Zh["LOAD"][:, shz]                    # shedding = negative load
        Hn = Zh["NB"]
        parts = [Hg, Hs, -Hs, np.zeros((zm.nbr, nS)), Hsh, Hn]
        if maximise_load_zone is not None:
            parts.append(-Zh["LOAD"][:, [maximise_load_zone]])
        if nH:
            parts.append(Zh["HVDC"])
        H = np.hstack(parts)
        lt = load[t] if island_load is None else load[t] - island_load[t]
        f0 = -Zh["LOAD"] @ lt
        return H, f0
    Hs_, f0s = zip(*[H_of(t) for t in range(T)]) if network else ([], [])
    Fmax = [zm.F * rating_pu[t] for t in range(T)] if network else []

    A_ub_rows, b_ub = [], []
    active = set()
    res = None
    if network and seed_lines is not None and len(seed_lines):
        # warm active set: lines binding on the previous stress day, both directions
        for t in range(T):
            for l in seed_lines:
                for sgn in (1.0, -1.0):
                    active.add((t, l, -1, sgn))
                    row = np.zeros(nv); row[t * per:(t + 1) * per] = Hs_[t][l]
                    A_ub_rows.append(sp.csr_matrix(sgn * row)); b_ub.append(np.array([Fmax[t][l] - sgn * f0s[t][l]]))
    for it in range(max_iter):
        if A_ub_rows:
            Aub = sp.vstack(A_ub_rows).tocsr(); bub = np.concatenate(b_ub)
        else:
            Aub, bub = None, None
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=np.column_stack([lb, ub]),
                      method="highs")
        if res.status != 0:
            break
        if not network:
            break
        x = res.x
        new = 0
        for t in range(T):
            xt = x[t * per:(t + 1) * per]
            f = Hs_[t] @ xt + f0s[t]
            viol = np.where((np.abs(f) > Fmax[t] * (1 + tol)) & zm.cnec)[0]
            for l in viol:
                sgn = np.sign(f[l])
                if (t, l, -1, sgn) in active:
                    continue
                active.add((t, l, -1, sgn)); new += 1
                row = np.zeros(nv); row[t * per:(t + 1) * per] = Hs_[t][l]
                A_ub_rows.append(sp.csr_matrix(sgn * row)); b_ub.append(np.array([Fmax[t][l] - sgn * f0s[t][l]]))
            if n1 and zm.lodf is not None:
                Ld = zm.lodf
                cont = ~zm.radial & zm.cnec
                cidx = np.where(cont)[0]
                fpost = f[:, None] + Ld[:, cont] * f[None, cont]
                fpost[~zm.cnec] = 0.0
                Fpost = np.repeat((zm.F * rating_pu[t])[:, None], len(cidx), 1)  # post-contingency: 100 % rating
                rem = getattr(zm.grid, "n1_remaining", None)
                if rem is not None:                       # remaining circuits of the outaged branch
                    Fpost[cidx, np.arange(len(cidx))] = zm.F[cidx] * rating_pu[t][cidx] * rem[cidx]
                bad = np.argwhere(np.abs(fpost) > Fpost * (1 + tol) + 1e-6)
                if len(bad) > 400:   # add the worst ones first
                    sev = np.abs(fpost[bad[:, 0], bad[:, 1]]) / np.maximum(Fpost[bad[:, 0], bad[:, 1]], 1e-6)
                    bad = bad[np.argsort(-sev)[:400]]
                for l, cj in bad:
                    k = cidx[cj]
                    sgn = np.sign(fpost[l, cj])
                    if (t, l, k, sgn) in active:
                        continue
                    active.add((t, l, k, sgn)); new += 1
                    hl = Hs_[t][l] + Ld[l, k] * Hs_[t][k]
                    f0l = f0s[t][l] + Ld[l, k] * f0s[t][k]
                    row = np.zeros(nv); row[t * per:(t + 1) * per] = hl
                    A_ub_rows.append(sp.csr_matrix(sgn * row)); b_ub.append(np.array([Fpost[l, cj] - sgn * f0l]))
        if new == 0:
            break
    out = dict(status=res.status if res is not None else -1, it=it + 1, n_active=len(active))
    if res is None or res.status != 0:
        out["shed"] = np.full((T, 16), np.nan)
        return out
    x = res.x.reshape(T, per)
    shed = np.zeros((T, 16))
    for t in range(T):
        np.add.at(shed[t], shz, x[t, oSh:oSh + nSh])
    out["shed"] = shed if island_load is None else shed + island_load
    out["gen"] = x[:, oG:oG + nG]; out["gen_idx"] = gen_idx
    out["imports"] = x[:, oN:oN + nN]
    out["soc_end"] = np.zeros((16, 2)); out["soc_end"][sz, ss] = x[-1, oE:oE + nS]
    np_z = np.zeros((T, 16))
    for t in range(T):
        np.add.at(np_z[t], gz, x[t, oG:oG + nG])
        np.add.at(np_z[t], sz, x[t, oD:oD + nS] - x[t, oC:oC + nS])
    np_z += shed - load
    out["net_position"] = np_z
    if maximise_load_zone is not None:
        out["served"] = x[:, oL]
    if network:
        out["flows"] = np.array([Hs_[t] @ x[t] + f0s[t] for t in range(T)])
        Fm = np.array(Fmax)
        out["binding"] = np.where((np.abs(out["flows"]) >= Fm * (1 - 1e-3)).any(0) & zm.cnec)[0]
    return out
