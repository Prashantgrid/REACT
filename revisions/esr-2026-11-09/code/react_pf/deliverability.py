"""Physical deliverability tests on the nodal network (structural REACT-PF KPIs).

Deliverable supply DS_z: the largest load state z can serve in a stress snapshot from its own
available resources plus imports, with every other state serving its own coincident load and all
branch flows within their thermal ratings (N-0) or within ratings after any single non-radial
branch outage (N-1, post-contingency flows via LODF). The concept follows load-deliverability
tests in capacity planning (e.g. PJM CETL/CETO), applied here to federal states.

The ratio DR_z = DS_z / peak load_z is the physical counterpart of REACT-Lite's local capacity
and connectivity indicators: a state with little local generation but secure import capability
obtains a high DR_z, which is the interconnected-system view requested by Reviewer 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import TECHS, STATES, ZIDX, NEIGHBOURS, zonal_ptdf, outage_ptdf, lodf
from .flow import ZonalModel, solve_block

PEAK_FACTOR = 1.44            # national 2019 load shape: max / mean (PyPSA model-energy example)

SNAPSHOTS = {
    # winter evening Dunkelflaute: low wind, no PV, cold (static line ratings), forced outages
    "winter_dunkelflaute": dict(load=1.00, avail={"LIG": .90, "COAL": .90, "GAS": .90, "OIL": .90, "BIO": .85,
                                                  "ROR": .45, "WON": .05, "WOFF": .10, "PV": 0.0},
                                nb=0.50, temp=0.0),
    # summer heatwave evening residual peak: thermal cooling-water and drought derating (REACT
    # Table A.3 at full heat and dry winter: 0.15 + 0.10*0.8), low river flow, PV temperature loss,
    # heat-derated overhead-line ampacity at 38 C
    "summer_heatwave": dict(load=0.85, avail={"LIG": .69, "COAL": .69, "GAS": .69, "OIL": .69, "BIO": .85,
                                              "ROR": .35, "WON": .05, "WOFF": .08, "PV": .13},
                            nb=0.50, temp=38.0),
    "summer_nominal": dict(load=0.85, avail={"LIG": .85, "COAL": .85, "GAS": .85, "OIL": .85, "BIO": .85,
                                             "ROR": .50, "WON": .10, "WOFF": .15, "PV": .15},
                           nb=0.75, temp=25.0),
}
T_CMAX, T_REF = 80.0, 35.0     # conductor limit and ambient temperature of the static rating


def ampacity_factor(temp_c, dlr=False):
    """Overhead-line rating multiplier from the steady-state heat balance, I ~ sqrt(T_c - T_a).
    Static ratings refer to 35 C ambient; without dynamic line rating there is no credit below."""
    f = np.sqrt(np.clip(T_CMAX - np.asarray(temp_c, float), 0, None) / (T_CMAX - T_REF))
    return f if dlr else np.minimum(f, 1.0)


def avg_load_mw(fleet):
    return fleet.annual_twh * 1e6 / 8760.0


def snapshot_arrays(fleet, snap, scale_nb=1.0):
    sp_ = SNAPSHOTS[snap]
    avail = np.zeros((1, 16, len(TECHS)))
    for k, tech in enumerate(TECHS):
        avail[0, :, k] = fleet.cap[tech].values * sp_["avail"][tech]
    load = (avg_load_mw(fleet) * PEAK_FACTOR * sp_["load"])[None, :]
    imp = np.array([NEIGHBOURS[n]["imp"] for n in fleet.neighbours]) * sp_["nb"] * scale_nb
    exp = np.array([NEIGHBOURS[n]["exp"] for n in fleet.neighbours]) * 1.0
    return avail, load, imp[None, :], exp[None, :], sp_["temp"]


def deliverable_supply(grid, fleet, snap="winter_dunkelflaute", n1=False, zm=None, branch_temp=None,
                       scale_nb=1.0):
    if zm is None:
        Z = zonal_ptdf(grid, fleet)
        L, radial = lodf(grid) if n1 else (None, None)
        zm = ZonalModel(grid, fleet, Z, lodf=L, radial=radial)
    avail, load, imp, exp, temp = snapshot_arrays(fleet, snap, scale_nb)
    rating = ampacity_factor(temp if branch_temp is None else branch_temp) * np.ones(zm.nbr)
    out = {}
    for z, s in enumerate(STATES):
        ld = load.copy(); ld[0, z] = 0.0
        r = solve_block(zm, avail, ld, imp, exp, rating[None, :], maximise_load_zone=z, n1=n1)
        served = float(r["served"][0]) if r["status"] == 0 else np.nan
        out[s] = dict(DS_MW=served, peak_MW=float(load[0, z]), DR=served / float(load[0, z]),
                      local_avail_MW=float(avail[0, z].sum() + fleet.stor_p.values[z].sum()),
                      lp_iter=r["it"], n_active=r["n_active"])
    df = pd.DataFrame(out).T
    df["import_capability_MW"] = df.DS_MW - df.local_avail_MW.clip(upper=df.DS_MW)
    return df


def local_firm_ratio(fleet, snap="winter_dunkelflaute"):
    avail, load, *_ = snapshot_arrays(fleet, snap)
    loc = avail[0].sum(1) + fleet.stor_p.values.sum(1)
    return pd.Series(loc / load[0], index=STATES)


# ------------------------------------------------------------------------------------------------
# Compound storm/flood event: fragility-based line failures and restoration
# ------------------------------------------------------------------------------------------------
def storm_event_test(grid, fleet, n_samples=30, p_fail=0.20, repair_median_h=36.0, repair_sigma=0.7,
                     step_h=12, max_h=168, seed=2024, states=None):
    """Expected unserved energy in the struck state over restoration, in hours of its average load.

    Each 220/380 kV overhead line with at least one terminal in the struck state fails
    independently with probability ``p_fail`` (the expected 20 % loss matches REACT-Lite's
    extreme-event step loss gamma_grid). Repair times are log-normal (median 36 h, sigma 0.7; cf.
    Panteli & Mancarella 2015 and the 2005 Muensterland pylon collapses), discretised to 12-h
    restoration stages. Each stage is a winter storm hour (onshore wind 0.30 in the struck state
    because of cut-outs, 0.50 elsewhere), solved as a load-shedding DC-OPF on the post-outage
    topology; load on islanded buses is unserved.
    """
    rng = np.random.default_rng(seed)
    br = grid.branches
    avg = avg_load_mw(fleet)
    load = (avg * 1.15)[None, :]
    base_av = {"LIG": .90, "COAL": .90, "GAS": .90, "OIL": .90, "BIO": .85, "ROR": .55, "WON": .50,
               "WOFF": .50, "PV": 0.0}
    imp = (np.array([NEIGHBOURS[n]["imp"] for n in fleet.neighbours]) * 0.75)[None, :]
    exp = np.array([NEIGHBOURS[n]["exp"] for n in fleet.neighbours])[None, :]
    rows = []
    for s in (states or STATES):
        z = ZIDX[s]
        cand = np.where((br.kind.values == "line") & ((br.z0.values == s) | (br.z1.values == s)))[0]
        avail = np.zeros((1, 16, len(TECHS)))
        for k, tech in enumerate(TECHS):
            avail[0, :, k] = fleet.cap[tech].values * base_av[tech]
        avail[0, z, TECHS.index("WON")] = fleet.cap["WON"].values[z] * 0.30
        ens = []; n_failed = []; peak_short = []
        for _ in range(n_samples):
            failed = cand[rng.random(len(cand)) < p_fail]
            rep = np.minimum(rng.lognormal(np.log(repair_median_h), repair_sigma, len(failed)), max_h)
            n_failed.append(len(failed))
            e = 0.0; ps = 0.0
            for t0 in range(0, max_h, step_h):
                out = failed[rep > t0]
                if len(out) == 0:
                    break
                P, island = outage_ptdf(grid, out)
                fz = dict(grid=grid)
                Z = zonal_ptdf(grid, fleet, P)
                zm = ZonalModel(grid, fleet, Z)
                isl = np.zeros((1, 16))
                if island.any():
                    isl[0] = (fleet.lsk[island] * load[0][None, :]).sum(0)
                r = solve_block(zm, avail, load, imp, exp, np.ones((1, zm.nbr)), Zt=[Z], island_load=isl)
                sh = r["shed"][0, z]
                e += sh * step_h; ps = max(ps, sh)
            ens.append(e); peak_short.append(ps)
        rows.append(dict(state=s, lines_exposed=len(cand), mean_failed=np.mean(n_failed),
                         EENS_event_MWh=np.mean(ens), EENS_event_h=np.mean(ens) / avg[z],
                         P_any_shed=np.mean(np.array(ens) > 1.0),
                         peak_shortfall_share=np.mean(peak_short) / load[0, z]))
    return pd.DataFrame(rows).set_index("state")


# ------------------------------------------------------------------------------------------------
# Nodal versions (main specification for the structural KPIs)
# ------------------------------------------------------------------------------------------------
def deliverable_supply_nodal(grid, fleet, snap="winter_dunkelflaute", n1=True, scale_nb=1.0, branch_temp=None):
    from .nodal import NodalModel
    L, radial = lodf(grid) if n1 else (None, None)
    nm = NodalModel(grid, fleet, lodf=L, radial=radial)
    sp_ = SNAPSHOTS[snap]
    load = avg_load_mw(fleet) * PEAK_FACTOR * sp_["load"]
    imp = np.array([NEIGHBOURS[n]["imp"] for n in fleet.neighbours]) * sp_["nb"] * scale_nb
    exp = np.array([NEIGHBOURS[n]["exp"] for n in fleet.neighbours]) * 1.0
    rating = ampacity_factor(sp_["temp"] if branch_temp is None else branch_temp) * np.ones(len(grid.branches))
    af = dict(sp_["avail"]); af.update({"PHS": 1.0, "BAT": 1.0})
    # system secure loadability: largest uniform scaling of all coincident loads (N-0 / N-1)
    rs = nm.solve(af, load, imp, exp, rating, maximise_zone="all", n1=n1)
    lam = float(rs["served"]) if rs["status"] == 0 else np.nan
    base = load * min(1.0, lam)            # other states held at a securely servable level
    out = {}
    for z, s in enumerate(STATES):
        r = nm.solve(af, base, imp, exp, rating, maximise_zone=z, n1=n1)
        loc = sum(fleet.cap[t].values[z] * sp_["avail"][t] for t in TECHS) + fleet.stor_p.values[z].sum()
        out[s] = dict(DS_MW=float(r["served"]), peak_MW=float(load[z]), DR=float(r["served"]) / float(load[z]),
                      local_avail_MW=float(loc), local_dispatch_MW=float(r["local_dispatch"][z]),
                      lp_iter=r["it"], n_active=r["n_active"], status=r["status"])
    df = pd.DataFrame(out).T
    df["import_MW"] = df.DS_MW - df.local_dispatch_MW
    df["system_loadability"] = lam
    return df


def storm_event_test_nodal(grid, fleet, n_samples=30, p_fail=0.20, repair_median_h=36.0, repair_sigma=0.7,
                           step_h=12, max_h=168, seed=2024, states=None):
    """Nodal version of ``storm_event_test`` (optimal redispatch on the damaged network)."""
    from .nodal import NodalModel
    rng = np.random.default_rng(seed)
    br = grid.branches
    avg = avg_load_mw(fleet)
    load = avg * 1.15
    af = {"LIG": .90, "COAL": .90, "GAS": .90, "OIL": .90, "BIO": .85, "ROR": .55, "WON": .50,
          "WOFF": .50, "PV": 0.0, "PHS": 1.0, "BAT": 1.0}
    imp = np.array([NEIGHBOURS[n]["imp"] for n in fleet.neighbours]) * 0.75
    exp = np.array([NEIGHBOURS[n]["exp"] for n in fleet.neighbours]) * 1.0
    rows = []
    for s in (states or STATES):
        z = ZIDX[s]
        a = dict(af); won = np.full(16, 0.50); won[z] = 0.30; a["WON"] = won
        cand = np.where((br.kind.values == "line") & ((br.z0.values == s) | (br.z1.values == s)))[0]
        ens, nf, ps = [], [], []
        for _ in range(n_samples):
            failed = cand[rng.random(len(cand)) < p_fail]
            rep = np.minimum(rng.lognormal(np.log(repair_median_h), repair_sigma, len(failed)), max_h)
            nf.append(len(failed)); e = 0.0; pk = 0.0
            for t0 in range(0, max_h, step_h):
                out = failed[rep > t0]
                if len(out) == 0:
                    break
                P, island = outage_ptdf(grid, out)
                nm = NodalModel(grid, fleet, P=P, island=island)
                r = nm.solve(a, load, imp, exp, np.ones(len(br)))
                sh = float(r["shed"][z]); e += sh * step_h; pk = max(pk, sh)
            ens.append(e); ps.append(pk)
        rows.append(dict(state=s, lines_exposed=len(cand), mean_failed=np.mean(nf), EENS_event_MWh=np.mean(ens),
                         EENS_event_h=np.mean(ens) / avg[z], P_any_shed=np.mean(np.array(ens) > 1.0),
                         peak_shortfall_share=np.mean(ps) / load[z]))
    return pd.DataFrame(rows).set_index("state")
