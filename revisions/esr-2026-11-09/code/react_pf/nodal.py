"""Nodal security-constrained snapshot OPF (single hour) on the 590-bus network.

Used for the structural REACT-PF KPIs (deliverability ratios, grid-damage restoration). Each
(bus, technology) pair is its own dispatch variable, so intra-state redispatch is optimal (the
TSO view) rather than fixed by zonal shift keys. Loads are fixed at their nodal shares; in a
deliverability test the load of the tested state is scaled uniformly at its buses. Branch limits
are enforced on all CNECs, optionally with single-circuit N-1 post-contingency limits (LODF),
both added lazily.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .network import TECHS, STORAGE, STATES, ZIDX
from .flow import COST, IMPORT_COST, VOLL


class NodalModel:
    def __init__(self, grid, fleet, P=None, lodf=None, radial=None, island=None):
        self.grid, self.fleet = grid, fleet
        self.P = grid.ptdf if P is None else P
        self.lodf, self.radial = lodf, radial
        self.F = grid.branches.s_nom.values
        self.cnec = grid.branches["cnec"].values
        zb = np.array([ZIDX[s] for s in grid.buses.state])
        self.zb = zb
        island = np.zeros(len(zb), bool) if island is None else island
        self.island = island
        # nodal capacities per technology: cap[z,k] * gsk_k[b,z]
        units = []
        for k, tech in enumerate(TECHS + STORAGE):
            g = fleet.gsk[tech]
            capz = fleet.cap[tech].values if tech in TECHS else fleet.stor_p[tech].values
            for b in np.where(g.sum(1) > 0)[0]:
                c = capz[zb[b]] * g[b, zb[b]]
                if c > 1e-6 and not island[b]:
                    units.append((b, k, c))
        self.ub_b = np.array([u[0] for u in units]); self.ub_k = np.array([u[1] for u in units])
        self.ub_c = np.array([u[2] for u in units])
        self.nnb = len(fleet.neighbours)
        self.Pg = self.P[:, self.ub_b]                       # branch x units
        self.Pn = self.P @ fleet.nsk                         # branch x neighbours
        self.lsk = fleet.lsk.copy(); self.lsk[island] = 0.0
        self.Pl = self.P @ self.lsk                          # branch x zones (load shares)
        self.hvdc = None
        if getattr(fleet, "hvdc_ends", None):
            self.hvdc = np.column_stack([self.P[:, t] - self.P[:, f] for f, t in fleet.hvdc_ends])
            self.hvdc_cap = fleet.hvdc_cap

    def solve(self, avail_factor, load, nb_imp, nb_exp, rating, maximise_zone=None, n1=False, max_iter=40, tol=1e-3):
        """avail_factor: dict tech -> (16,) availability multipliers (or scalar); load: (16,) MW."""
        nU, nN = len(self.ub_c), self.nnb
        nH = 0 if self.hvdc is None else self.hvdc.shape[1]
        tech_names = TECHS + STORAGE
        af = np.array([np.broadcast_to(np.asarray(avail_factor.get(tech_names[k], 1.0), float), (16,))[self.zb[b]]
                       for b, k in zip(self.ub_b, self.ub_k)])
        ub_g = self.ub_c * af
        cost_g = np.array([COST.get(tech_names[k], 1.0) for k in self.ub_k])
        islanded_load = (self.fleet.lsk[self.island] * load[None, :]).sum(0)
        L = np.asarray(load, float) - islanded_load
        # variables: g (nU) | m (nN) | shed (16) | Lz (0/1) | h (nH)
        allz = isinstance(maximise_zone, str) and maximise_zone == "all"
        nLz = 1 if maximise_zone is not None else 0
        nv = nU + nN + 16 + nLz + nH
        c = np.r_[cost_g, np.full(nN, IMPORT_COST), np.full(16, VOLL if maximise_zone is None else 1e7),
                  np.full(nLz, -1e5), np.full(nH, 0.01)]
        lb = np.r_[np.zeros(nU), -nb_exp, np.zeros(16), np.zeros(nLz), -(self.hvdc_cap if nH else np.zeros(0))]
        ub = np.r_[ub_g, nb_imp, np.maximum(L, 0), np.full(nLz, np.inf), (self.hvdc_cap if nH else np.zeros(0))]
        Lfull = L.copy()
        if allz:                     # uniform loadability: lambda * (all zonal loads), no shedding
            L = np.zeros(16); ub[nU + nN:nU + nN + 16] = 0.0
            c[nU + nN + 16] = -1e5 * Lfull.sum()
        elif maximise_zone is not None:
            L = L.copy(); L[maximise_zone] = 0.0; ub[nU + nN + maximise_zone] = 0.0
        lz_bal = -Lfull.sum() if allz else -1.0
        Aeq = np.r_[np.ones(nU), np.ones(nN), np.ones(16), np.full(nLz, lz_bal), np.zeros(nH)][None, :]
        beq = np.array([L.sum()])
        lz_col = [-(self.Pl @ Lfull)[:, None]] if allz else ([-self.Pl[:, [maximise_zone]]] if nLz else [])
        H = np.hstack([self.Pg, self.Pn, self.Pl] + lz_col + ([self.hvdc] if nH else []))
        f0 = -self.Pl @ L
        F = self.F * rating
        rows, rhs, active = [], [], set()
        res = None
        for it in range(max_iter):
            Aub = sp.csr_matrix(np.array(rows)) if rows else None
            res = linprog(c, A_ub=Aub, b_ub=np.array(rhs) if rows else None, A_eq=Aeq, b_eq=beq,
                          bounds=np.column_stack([lb, ub]), method="highs")
            if res.status != 0:
                break
            x = res.x; f = H @ x + f0
            new = 0
            for l in np.where((np.abs(f) > F * (1 + tol)) & self.cnec)[0]:
                sg = np.sign(f[l])
                if (l, -1, sg) in active:
                    continue
                active.add((l, -1, sg)); rows.append(sg * H[l]); rhs.append(F[l] - sg * f0[l]); new += 1
            if n1 and self.lodf is not None:
                cont = ~self.radial & self.cnec
                cidx = np.where(cont)[0]
                fp = f[:, None] + self.lodf[:, cont] * f[None, cont]
                fp[~self.cnec] = 0.0
                Fp = np.repeat(F[:, None], len(cidx), 1)
                rem = self.grid.n1_remaining
                Fp[cidx, np.arange(len(cidx))] = F[cidx] * rem[cidx]
                bad = np.argwhere(np.abs(fp) > Fp * (1 + tol) + 1e-6)
                if len(bad) > 300:
                    sev = np.abs(fp[bad[:, 0], bad[:, 1]]) / np.maximum(Fp[bad[:, 0], bad[:, 1]], 1e-6)
                    bad = bad[np.argsort(-sev)[:300]]
                for l, cj in bad:
                    k = cidx[cj]; sg = np.sign(fp[l, cj])
                    if (l, k, sg) in active:
                        continue
                    active.add((l, k, sg))
                    rows.append(sg * (H[l] + self.lodf[l, k] * H[k]))
                    rhs.append(Fp[l, cj] - sg * (f0[l] + self.lodf[l, k] * f0[k])); new += 1
            if new == 0:
                break
        out = dict(status=res.status, it=it + 1, n_active=len(active))
        if res.status != 0:
            out.update(served=np.nan, shed=np.full(16, np.nan))
            return out
        x = res.x
        out["shed"] = x[nU + nN:nU + nN + 16] + islanded_load
        out["served"] = x[nU + nN + 16] if nLz else np.nan
        out["flows"] = H @ x + f0
        out["local_dispatch"] = np.bincount(self.zb[self.ub_b], weights=x[:nU], minlength=16)
        return out
