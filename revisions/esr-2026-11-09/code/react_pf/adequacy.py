"""Chronological adequacy-and-deliverability Monte Carlo (REACT-PF scenarios).

For each Monte Carlo year:
  1. REACT's copula draws one hazard year (state-specific in Baseline, a shared quantile vector
     in the two spatially coherent scenarios) -- the *same* hazard model as REACT-Lite;
  2. the weather generator produces hourly wind, PV, temperature and load consistent with it;
  3. availability: thermal units follow two-state Markov forced-outage processes, are derated in
     hot hours and after dry winters (REACT Table A.3 relationships applied to daily Tmax), and
     carry summer maintenance; run-of-river follows season and soil moisture; line ratings use
     an N-1 security margin (s_max_pu = 0.7) and the heat-dependent ampacity factor;
  4. a compound precipitation-storm year (Xi* > Xi_crit, REACT trigger) damages a random 20 % of
     the struck state's 220/380 kV lines, restored with log-normal repair times;
  5. stress days are screened and solved as daily load-shedding DC-OPFs on the full network
     (Network / Adjacency Flow scenario) or as a copper plate (National Pool), with storage
     state of charge carried across consecutive stress days.
Outputs per state and year: unserved energy (MWh), loss-of-load hours, and the event part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import TECHS, THERMAL, STORAGE, STATES, ZIDX, NEIGHBOURS, zonal_ptdf, outage_ptdf
from .flow import ZonalModel, solve_block
from .deliverability import ampacity_factor, avg_load_mw
from .weather import DOY_H, H, DAYS

FOR = {"LIG": 0.08, "COAL": 0.07, "GAS": 0.05, "OIL": 0.05}           # forced outage rates
UNIT_MW = {"LIG": 600.0, "COAL": 500.0, "GAS": 250.0, "OIL": 100.0}
MTTR_H = 50.0
S_MAX_PU = 0.70
XI_CRIT = 0.883
SCENARIO_CFG = {
    "baseline": dict(coherent=False, network=True, kappa_nb=0.25),
    "national_pool": dict(coherent=True, network=False, kappa_nb=0.50),
    "adjacency_flow": dict(coherent=True, network=True, kappa_nb=0.50),
}


def markov_availability(n_units, for_, mttr, rng, hours=H):
    """Fraction of units available per hour; two-state Markov chain per unit."""
    if n_units == 0:
        return np.ones(hours)
    mu = 1.0 / mttr; lam = for_ * mu / (1 - for_)
    up = rng.random(n_units) > for_
    out = np.empty(hours)
    for t in range(hours):
        r = rng.random(n_units)
        up = np.where(up, r > lam, r < mu)
        out[t] = up.mean()
    return out


class AdequacyModel:
    def __init__(self, grid, fleet, wg, fit, hazard_panel, scenario, import_cap_mw, s_max_pu=S_MAX_PU,
                 kappa_nb=None, p_fail=0.20, nb_scale=1.0, screen_nat_mw=5000.0, screen_ic_share=0.5):
        self.grid, self.fleet, self.wg, self.fit = grid, fleet, wg, fit
        self.cfg = dict(SCENARIO_CFG[scenario]); self.scenario = scenario
        if kappa_nb is not None:
            self.cfg["kappa_nb"] = kappa_nb
        self.Z = zonal_ptdf(grid, fleet)
        self.zm = ZonalModel(grid, fleet, self.Z)
        self.s_max_pu, self.p_fail, self.nb_scale = s_max_pu, p_fail, nb_scale
        self.avg = avg_load_mw(fleet)
        self.ic = np.asarray(import_cap_mw, float)
        self.screen_nat, self.screen_ic = screen_nat_mw, screen_ic_share
        self.units = {k: np.where(fleet.cap[k].values > 0, np.maximum(1, np.round(fleet.cap[k].values / UNIT_MW[k])), 0).astype(int)
                      for k in THERMAL}
        self.thermal_scale = 1.0          # < 1 in the reliability-standard stress test
        hp = hazard_panel
        self.sm_lo, self.sm_hi = float(hp.soil_moist_winter_mean.min()), float(hp.soil_moist_winter_mean.max())
        xi_cols = ("p30_days_mean", "Rx1day_mm_mean", "Storm_Days_mean")
        self.xi_lo = {c: np.array([fit["marginals"][s][c].min() for s in STATES]) for c in xi_cols}
        self.xi_hi = {c: np.array([fit["marginals"][s][c].max() for s in STATES]) for c in xi_cols}
        br = grid.branches
        self.b_z0 = np.array([ZIDX[s] for s in br.z0]); self.b_z1 = np.array([ZIDX[s] for s in br.z1])
        self.imp_ntc = np.array([NEIGHBOURS[n]["imp"] for n in fleet.neighbours], float) * nb_scale
        self.exp_ntc = np.array([NEIGHBOURS[n]["exp"] for n in fleet.neighbours], float)

    # ------------------------------------------------------------------ hazards
    def draw_hazard(self, rng):
        from REACT_v4 import sample_hazard_year
        hcols = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean", "Rx1day_mm_max",
                 "Storm_Days_mean", "low_wind_days_mean"]
        hz = sample_hazard_year(self.fit, hcols, self.cfg["coherent"], rng)
        return pd.DataFrame(hz).T.loc[STATES]

    def xi(self, draw):
        n = lambda c: np.clip((draw[c].to_numpy(float) - self.xi_lo[c]) / np.maximum(self.xi_hi[c] - self.xi_lo[c], 1e-12), 0, 1)
        Hp, Hr, Hs = n("p30_days_mean"), n("Rx1day_mm_mean"), n("Storm_Days_mean")
        return 0.40 * Hp + 0.20 * Hr + 0.20 * Hs + 0.20 * np.maximum(Hp, Hr)

    # ------------------------------------------------------------------ one year
    def build_year(self, rng, draw, weather=None):
        fl = self.fleet
        w = self.wg.sample_year(rng, draw) if weather is None else weather
        pert = {k: rng.uniform(0.8, 1.2) for k in ("Thermal", "Grid")}
        S = np.clip((draw.soil_moist_winter_mean.to_numpy(float) - self.sm_lo) / (self.sm_hi - self.sm_lo), 0, 1)
        tmax_h = w["tmax_daily"][DOY_H]                                   # (H,16)
        Hh = np.clip((tmax_h - 27.0) / 11.0, 0, 1)
        summer = ((DOY_H >= 120) & (DOY_H < 273)).astype(float)[:, None]
        d_th = np.minimum(pert["Thermal"] * (0.15 * Hh + 0.10 * (1 - S)[None, :] * summer), 0.25)
        maint = 1.0 - 0.10 * summer
        avail = np.zeros((H, 16, len(TECHS)), dtype=np.float32)
        A = self.thermal_availability(rng)
        for k in THERMAL:
            avail[:, :, TECHS.index(k)] = fl.cap[k].values[None, :] * self.thermal_scale * A[k] * (1 - d_th) * maint
        avail[:, :, TECHS.index("BIO")] = fl.cap["BIO"].values[None, :] * 0.85
        ror = (0.50 + 0.15 * np.cos(2 * np.pi * (DOY_H - 160) / 365.25))[:, None] * (0.85 + 0.15 * S)[None, :]
        avail[:, :, TECHS.index("ROR")] = fl.cap["ROR"].values[None, :] * ror
        avail[:, :, TECHS.index("WON")] = fl.cap["WON"].values[None, :] * w["wind"]
        avail[:, :, TECHS.index("PV")] = fl.cap["PV"].values[None, :] * w["pv"]
        off = np.zeros((H, 16))
        for s, cf in (w["offshore"] or {}).items():
            off[:, ZIDX[s]] = cf
        avail[:, :, TECHS.index("WOFF")] = fl.cap["WOFF"].values[None, :] * off
        load = self.avg[None, :] * w["load_factor"]
        # correlated neighbour support: stress index from national load and wind
        nat_load = load.sum(1); wcf = (w["wind"] * fl.cap["WON"].values).sum(1) / fl.cap["WON"].sum()
        s_idx = np.clip((nat_load / np.quantile(nat_load, 0.999) - 0.75) / 0.25, 0, 1) * np.clip((0.25 - wcf) / 0.25, 0, 1)
        phi = np.clip(0.9 - self.cfg["kappa_nb"] * s_idx, 0.0, 1.0)
        imp = phi[:, None] * self.imp_ntc[None, :]
        exp = np.repeat(self.exp_ntc[None, :], H, 0)
        xi = self.xi(draw)
        return dict(w=w, avail=avail, load=load, imp=imp, exp=exp, xi=xi, S=S, pert=pert, phi=phi)

    def thermal_availability(self, rng):
        """Vectorised two-state Markov forced-outage simulation for all thermal units at once."""
        owners = [(k, z) for k in THERMAL for z in range(16) for _ in range(self.units[k][z])]
        n = len(owners)
        f = np.array([FOR[k] for k, _ in owners]); mu = 1.0 / MTTR_H; lam = f * mu / (1 - f)
        up = rng.random(n) > f
        R = rng.random((H, n)).astype(np.float32)
        hist = np.empty((H, n), dtype=bool)
        for t in range(H):
            up = np.where(up, R[t] > lam, R[t] < mu)
            hist[t] = up
        out = {}
        keys = np.array([THERMAL.index(k) * 16 + z for k, z in owners])
        for k in THERMAL:
            a = np.ones((H, 16))
            for z in range(16):
                m = keys == THERMAL.index(k) * 16 + z
                if m.any():
                    a[:, z] = hist[:, m].mean(1)
            out[k] = a
        return out

    def screen(self, yr):
        # storage is not credited in the screen (energy-limited); the LP adds it on flagged days
        loc = yr["avail"].sum(2)
        nat = loc.sum(1) + yr["imp"].sum(1) - yr["load"].sum(1)
        zon = loc + self.screen_ic * self.ic[None, :] - yr["load"]
        flag_h = (nat < self.screen_nat) | (zon < 0).any(1)
        return flag_h.reshape(DAYS, 24).any(1), nat, zon

    def rating(self, yr, hours):
        T = yr["w"]["temp"][hours]
        tb = 0.5 * (T[:, self.b_z0] + T[:, self.b_z1])
        return self.s_max_pu * ampacity_factor(tb)

    def solve_days(self, yr, days):
        shed = np.zeros((H, 16)); soc = None; prev = -2; n_lp = 0; status_bad = 0; seed = set()
        for d in sorted(days):
            hrs = np.arange(d * 24, d * 24 + 24)
            if d != prev + 1:
                soc = None                                       # full storage after a non-stress day
            r = solve_block(self.zm, yr["avail"][hrs].astype(float), yr["load"][hrs], yr["imp"][hrs], yr["exp"][hrs],
                            self.rating(yr, hrs), stor_soc0=soc, network=self.cfg["network"],
                            seed_lines=sorted(seed)[:60])
            n_lp += 1
            if r["status"] != 0:
                status_bad += 1; prev = d; soc = None
                continue
            shed[hrs] = r["shed"]; soc = r["soc_end"]; prev = d
            if "binding" in r:
                seed |= set(int(l) for l in r["binding"])
        return shed, n_lp, status_bad

    def storm_events(self, yr, rng, step_h=12, max_h=168):
        # National Pool (copper plate): grid damage only strands load on islanded substations
        """Grid-damage events in states with Xi* > Xi_crit; returns (H,16) shed and event log."""
        br = self.grid.branches
        hit = np.where(yr["xi"] > XI_CRIT)[0]
        shed = np.zeros((H, 16)); log = []
        if len(hit) == 0:
            return shed, log
        def start_hour():
            day = rng.integers(274, 365 + 90) % 365 if rng.random() < 0.5 else rng.integers(120, 243)
            return int(day) * 24 + int(rng.integers(0, 24))
        common = start_hour()
        failed_all, rep_all = [], []
        starts = {}
        for z in hit:
            s = STATES[z]
            t0 = common if self.cfg["coherent"] else start_hour()
            cand = np.where((br.kind.values == "line") & ((br.z0.values == s) | (br.z1.values == s)))[0]
            p = self.p_fail * yr["pert"]["Grid"]
            f = cand[rng.random(len(cand)) < p]
            rep = np.minimum(rng.lognormal(np.log(36.0), 0.7, len(f)), max_h)
            failed_all.append((t0, f, rep)); starts[s] = t0
        # restoration stages on a common clock per start time
        for t0 in sorted({x[0] for x in failed_all}):
            group = [(f, rep) for (tt, f, rep) in failed_all if tt == t0]
            for k in range(0, max_h, step_h):
                out = np.concatenate([f[rep > k] for f, rep in group]) if group else np.array([], int)
                if len(out) == 0:
                    break
                h = (t0 + k + step_h // 2) % H
                hrs = (t0 + k + np.arange(step_h)) % H
                P, island = outage_ptdf(self.grid, np.unique(out))
                if not self.cfg["network"]:
                    if island.any():
                        isl = (self.fleet.lsk[island] * yr["load"][[h]][0][None, :]).sum(0)
                        shed[hrs] = np.maximum(shed[hrs], isl[None, :])
                    continue
                Zs = zonal_ptdf(self.grid, self.fleet, P)
                isl = np.zeros((1, 16)); ld = yr["load"][[h]]
                if island.any():
                    isl[0] = (self.fleet.lsk[island] * ld[0][None, :]).sum(0)
                r = solve_block(self.zm, yr["avail"][[h]].astype(float), ld, yr["imp"][[h]], yr["exp"][[h]],
                                self.rating(yr, [h]), network=True, Zt=[Zs], island_load=isl)
                if r["status"] == 0:
                    shed[hrs] = np.maximum(shed[hrs], r["shed"][0][None, :])
            log.append(dict(start=t0, states=[s for s, v in starts.items() if v == t0],
                            n_failed=int(sum(len(f) for f, _ in group))))
        return shed, log

    def simulate_year(self, rng, draw=None, full_year=False, weather=None):
        draw = self.draw_hazard(rng) if draw is None else draw
        yr = self.build_year(rng, draw, weather)
        flag_d, nat, zon = self.screen(yr)
        days = np.arange(DAYS) if full_year else np.where(flag_d)[0]
        shed, n_lp, bad = self.solve_days(yr, days)
        ev_shed, ev_log = self.storm_events(yr, rng)
        total = np.maximum(shed, ev_shed)
        return dict(eens=total.sum(0), lole=(total > 1.0).sum(0), lole_nat=int((total.sum(1) > 1.0).sum()),
                    eens_nat=float(total.sum()), eens_adequacy=shed.sum(0),
                    eens_event=ev_shed.sum(0), n_days=len(days), n_lp=n_lp, lp_fail=bad, events=ev_log,
                    demand=yr["load"].sum(0), draw=draw, min_nat_margin=float(nat.min()),
                    phi_min=float(yr["phi"].min()))
