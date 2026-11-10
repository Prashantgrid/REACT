"""REACT-PF network layer: nodal 220/380 kV model of Germany reduced to 16 federal-state zones.

Data (cached in ``data_cache/``, SHA-256 in ``SHA256SUMS``; see ``fetch_data.py``):
  * ENTSO-E GridKit extract distributed with PyPSA-Eur (Wiegmans 2016; Hoersch et al. 2018):
    buses, AC lines (voltage, circuits, route length), 380/220 kV transformers.
  * SciGRID-DE (PyPSA example network; Matke et al. 2016): bus-level load and onshore-wind/PV
    capacity used only as *spatial keys* to distribute state totals within each state.
  * powerplantmatching (Gotzens et al. 2019): conventional, hydro, bioenergy and pumped-storage
    units with coordinates; used for the fuel split and bus location of state thermal capacity.
  * REACT structural panel (REACT_v4.py): state totals of onshore wind, PV, fossil thermal
    capacity, battery storage energy and annual consumption, so REACT-Lite and REACT-PF use
    the same asset base.

Electrical parameters follow PyPSA standard line types used by PyPSA-Eur for GridKit lines:
380 kV  Al/St 240/40 4-bundle (x = 0.246 ohm/km, 2.58 kA -> 1698 MVA per circuit);
220 kV  Al/St 240/40 2-bundle (x = 0.301 ohm/km, 1.29 kA -> 492 MVA per circuit);
380/220 kV transformers x = 0.1 pu on 2000 MVA. DC load-flow (lossless, flat voltage).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache")
GEOJSON = os.path.join(HERE, "..", "..", "figure_sources", "data", "de_states_ne10m.geojson")

STATES = ["Hamburg", "Bayern", "Hessen", "Baden-Württemberg", "Niedersachsen", "Bremen",
          "Nordrhein-Westfalen", "Berlin", "Schleswig-Holstein", "Rheinland-Pfalz", "Saarland",
          "Thüringen", "Brandenburg", "Mecklenburg-Vorpommern", "Sachsen", "Sachsen-Anhalt"]
ABBR = {"Baden-Württemberg": "BW", "Bayern": "BY", "Berlin": "BE", "Brandenburg": "BB", "Bremen": "HB",
        "Hamburg": "HH", "Hessen": "HE", "Mecklenburg-Vorpommern": "MV", "Niedersachsen": "NI",
        "Nordrhein-Westfalen": "NRW", "Rheinland-Pfalz": "RP", "Saarland": "SL", "Sachsen": "SN",
        "Sachsen-Anhalt": "ST", "Schleswig-Holstein": "SH", "Thüringen": "TH"}
ZIDX = {s: i for i, s in enumerate(STATES)}

S_BASE = 100.0                                   # MVA
LINE_TYPE = {380: dict(x_km=0.246, s_circ=np.sqrt(3) * 380 * 2.58),
             220: dict(x_km=0.301, s_circ=np.sqrt(3) * 220 * 1.29)}
TRAFO = dict(s_nom=2000.0, x_pu_own=0.10)

# Generation technologies of the zonal model
THERMAL = ["LIG", "COAL", "GAS", "OIL"]
TECHS = THERMAL + ["BIO", "ROR", "WON", "WOFF", "PV"]
STORAGE = ["PHS", "BAT"]

# Offshore wind in operation 31.12.2024 by state of grid connection (BWE/BWO/VDMA/WAB/WindGuard,
# "Status of Offshore Wind Energy Development, Year 2024"): NI 5,257 MW; SH 2,135 MW; MV 1,831 MW.
# Landing substations (lon, lat) and shares of each state's capacity.
OFFSHORE = {
    "Niedersachsen": (5257.0, [((7.31, 53.12), 0.25), ((7.30, 52.97), 0.35), ((7.22, 53.36), 0.20),
                               ((8.04, 53.57), 0.10), ((8.06, 52.84), 0.10)]),
    "Schleswig-Holstein": (2135.0, [((9.23, 53.92), 1.00)]),
    "Mecklenburg-Vorpommern": (1831.0, [((13.64, 54.14), 0.60), ((12.20, 54.11), 0.40)]),
}

# Cross-border exchange capacities (MW; import to DE, export from DE). Rounded reference-grid
# values of the order reported for 2024/25 (ENTSO-E TYNDP reference grid, JAO/TSO publications).
# HVDC landing points
# are fixed by coordinates; AC borders are distributed over the GridKit border buses in
# proportion to the thermal rating of the cross-border circuits terminating there.
NEIGHBOURS = {
    "NL": dict(imp=5000, exp=5000), "BE": dict(imp=1000, exp=1000, hvdc=[(6.43, 50.86)]),
    "LU": dict(imp=200, exp=1200), "FR": dict(imp=3000, exp=3000), "CH": dict(imp=4000, exp=4000),
    "AT": dict(imp=5000, exp=5000), "CZ": dict(imp=2600, exp=2100), "PL": dict(imp=1500, exp=1500),
    "DK1": dict(imp=3500, exp=3500), "DK2": dict(imp=1000, exp=1000, hvdc=[(12.20, 54.11)]),
    "SE": dict(imp=600, exp=600, hvdc=[(10.80, 53.92)]), "NO": dict(imp=1400, exp=1400, hvdc=[(9.37, 53.92)]),
}
GRIDKIT_COUNTRY = {"NL": "NL", "BE": "BE", "LU": "LU", "FR": "FR", "CH": "CH", "AT": "AT", "CZ": "CZ",
                   "PL": "PL", "DK": "DK1"}

# Area of each state (km2, Destatis 33111-0008) is held in REACT_ESR_reproducible.STATE_AREA_KM2.


# ----------------------------------------------------------------------------------------------
# Geography
# ----------------------------------------------------------------------------------------------
def state_polygons():
    geo = json.load(open(GEOJSON))
    out = {f["properties"]["name"]: shape(f["geometry"]).buffer(0) for f in geo["features"]}
    for p in out.values():
        shapely.prepare(p)
    return out


def assign_states(lon, lat, polys, max_deg=0.25):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    res = np.array([None] * len(lon), dtype=object)
    for name, p in polys.items():
        res[shapely.contains_xy(p, lon, lat) & (res == None)] = name  # noqa: E711
    miss = np.where(res == None)[0]  # noqa: E711
    if len(miss):
        pts = shapely.points(lon[miss], lat[miss])
        names = list(polys)
        d = np.column_stack([shapely.distance(polys[n], pts) for n in names])
        j = d.argmin(1)
        ok = d[np.arange(len(miss)), j] <= max_deg
        res[miss[ok]] = [names[k] for k in j[ok]]
    return res


def state_centroids_km(polys):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    out = {}
    for s in STATES:
        p = polys[s].representative_point() if s not in ("Berlin", "Hamburg", "Bremen") else polys[s].centroid
        x, y = tr.transform(p.x, p.y)
        out[s] = (x / 1e3, y / 1e3, p.x, p.y)
    return out


# ----------------------------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------------------------
@dataclass
class Grid:
    buses: pd.DataFrame
    branches: pd.DataFrame
    nb_attach: dict
    ptdf: np.ndarray = None
    slack: int = 0
    notes: list = field(default_factory=list)


def _tags_country(tag):
    if not isinstance(tag, str):
        return None
    i = tag.find('"country"=>"')
    return tag[i + 12:i + 14] if i >= 0 else None


def build_grid(include_under_construction=True):
    polys = state_polygons()
    de = shapely.union_all(list(polys.values())); deb = de.buffer(0.03); shapely.prepare(deb)
    b = pd.read_csv(os.path.join(CACHE, "gridkit_buses.csv"), quotechar="'")
    l = pd.read_csv(os.path.join(CACHE, "gridkit_lines.csv"), quotechar="'")
    t = pd.read_csv(os.path.join(CACHE, "gridkit_transformers.csv"), quotechar="'")
    b = b[(b.dc == "f") & b.voltage.isin([220.0, 380.0])].copy()
    b["inDE"] = shapely.contains_xy(deb, b.x.values, b.y.values)
    b["country"] = b.tags.map(_tags_country)
    deb_ids = set(b.loc[b.inDE, "bus_id"])
    l = l[l.voltage.isin([220, 380])].copy()
    if not include_under_construction:
        l = l[~l.under_construction]
    # cross-border AC circuits -> neighbour attachment shares
    cross = l[(l.bus0.isin(deb_ids)) ^ (l.bus1.isin(deb_ids))].copy()
    cross["de_bus"] = np.where(cross.bus0.isin(deb_ids), cross.bus0, cross.bus1)
    cross["fb"] = np.where(cross.bus0.isin(deb_ids), cross.bus1, cross.bus0)
    bc = b.set_index("bus_id")["country"]
    cross["nb"] = cross.fb.map(bc).map(GRIDKIT_COUNTRY)
    cross["s"] = cross.circuits * cross.voltage.map(lambda v: LINE_TYPE[int(v)]["s_circ"])
    # internal AC network
    l = l[l.bus0.isin(deb_ids) & l.bus1.isin(deb_ids)].copy()
    t = t[t.bus0.isin(deb_ids) & t.bus1.isin(deb_ids)].copy()
    buses = b[b.inDE].set_index("bus_id")[["x", "y", "voltage", "station_id"]].copy()
    rows = []
    for r in l.itertuples():
        lt = LINE_TYPE[int(r.voltage)]
        km = r.length / 1e3
        x_ohm = lt["x_km"] * km / r.circuits
        rows.append(dict(bus0=r.bus0, bus1=r.bus1, kind="line", v=int(r.voltage), circuits=int(r.circuits),
                         length_km=km, s_nom=lt["s_circ"] * r.circuits,
                         x_pu=x_ohm / (r.voltage ** 2 / S_BASE), uc=bool(r.under_construction),
                         cable=bool(r.underground), gid=int(r.line_id)))
    for r in t.itertuples():
        rows.append(dict(bus0=r.bus0, bus1=r.bus1, kind="trafo", v=380, circuits=1, length_km=0.0,
                         s_nom=TRAFO["s_nom"], x_pu=TRAFO["x_pu_own"] * S_BASE / TRAFO["s_nom"], uc=False,
                         cable=False, gid=int(r.transformer_id)))
    br = pd.DataFrame(rows)
    br = br[br.bus0 != br.bus1]
    # keep the main connected component
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(buses.index); G.add_edges_from(zip(br.bus0, br.bus1))
    comp = max(nx.connected_components(G), key=len)
    dropped = len(buses) - len(comp)
    buses = buses.loc[sorted(comp)]
    br = br[br.bus0.isin(comp) & br.bus1.isin(comp)].reset_index(drop=True)
    buses["state"] = assign_states(buses.x.values, buses.y.values, polys)
    # The 1:10m boundaries are coarse at city-state scale: substations serving Bremen
    # (Mittelsbueren/Niedervieland, Farge) lie a few km outside the generalised polygon.
    pts = shapely.points(buses.x.values, buses.y.values)
    for city in ("Bremen", "Hamburg", "Berlin"):
        if (buses.state == city).sum() < 3:
            d = shapely.distance(polys[city], pts)
            sel = (d < 0.06) & ~buses.state.isin(["Bremen", "Hamburg", "Berlin"]).values
            buses.loc[sel, "state"] = city
            grid_note = f"{city}: {int(sel.sum())} substations within 0.06 deg of the generalised boundary reassigned"
            print("  [network]", grid_note)
    buses = buses[buses.state.notna()].copy()
    br = br[br.bus0.isin(buses.index) & br.bus1.isin(buses.index)].reset_index(drop=True)
    buses["i"] = np.arange(len(buses))
    br["i0"] = br.bus0.map(buses.i).astype(int); br["i1"] = br.bus1.map(buses.i).astype(int)
    br["z0"] = buses.state.values[br.i0]; br["z1"] = buses.state.values[br.i1]
    br["interstate"] = br.z0 != br.z1
    # neighbour attachment: bus index -> share
    nb_attach = {}
    cross = cross[cross.de_bus.isin(buses.index) & cross.nb.notna()]
    for nb, g in cross.groupby("nb"):
        s = g.groupby("de_bus").s.sum()
        nb_attach[nb] = {int(buses.at[k, "i"]): float(v / s.sum()) for k, v in s.items()}
    for nb, spec in NEIGHBOURS.items():
        if "hvdc" in spec:
            nb_attach[nb] = {}
            for lon, lat in spec["hvdc"]:
                j = nearest_bus(buses, lon, lat, v=380)
                nb_attach[nb][j] = nb_attach[nb].get(j, 0) + 1.0 / len(spec["hvdc"])
    missing = [n for n in NEIGHBOURS if n not in nb_attach]
    grid = Grid(buses=buses, branches=br, nb_attach=nb_attach)
    grid.notes.append(f"GridKit 220/380 kV: {len(buses)} buses, {int((br.kind=='line').sum())} lines, "
                      f"{int((br.kind=='trafo').sum())} transformers; {dropped} isolated buses dropped; "
                      f"{int(br.uc.sum())} lines flagged under construction in the extract and included")
    if missing:
        grid.notes.append(f"neighbours without GridKit border buses: {missing}")
    compute_ptdf(grid)
    return grid


def nearest_bus(buses, lon, lat, v=None, state=None):
    d = buses
    if v is not None:
        d = d[d.voltage == v]
    if state is not None and (d.state == state).any():
        d = d[d.state == state]
    k = ((d.x - lon) ** 2 + ((d.y - lat) * 1.6) ** 2).idxmin()
    return int(buses.at[k, "i"])


def compute_ptdf(grid, branch_mask=None):
    """PTDF (branches x buses), single slack; flows invariant to the slack for balanced injections."""
    nbus = len(grid.buses)
    br = grid.branches if branch_mask is None else grid.branches[branch_mask]
    nbr = len(br)
    A = np.zeros((nbr, nbus))
    A[np.arange(nbr), br.i0.values] = 1.0
    A[np.arange(nbr), br.i1.values] = -1.0
    bvec = 1.0 / br.x_pu.values
    B = A.T @ (bvec[:, None] * A)
    keep = np.r_[0:grid.slack, grid.slack + 1:nbus]
    Binv = np.zeros((nbus, nbus))
    Binv[np.ix_(keep, keep)] = np.linalg.inv(B[np.ix_(keep, keep)])
    P = (bvec[:, None] * A) @ Binv
    if branch_mask is None:
        grid.ptdf = P
        grid.inc = A
    return P


def lodf(grid, P=None, single_circuit=True):
    """Outage distribution factors for N-1 analysis (branches x branches).

    With ``single_circuit`` (default) the contingency is the loss of ONE circuit of a GridKit branch
    with n circuits (the usual N-1 element), i.e. a parallel element with susceptance b/n carrying
    f/n. Column k then gives the post-contingency flow change per unit of pre-contingency flow on
    branch k:  dF_l = M_lk/n_k / (1 - M_kk/n_k) * f_k ;  for l = k the remaining n_k-1 circuits
    carry (n_k-1)/n_k * (f_k + M_kk/n_k/(1 - M_kk/n_k) f_k) against (n_k-1)/n_k of the rating.
    Radial single-circuit branches (M_kk -> 1) are flagged and excluded as contingencies.
    """
    P = grid.ptdf if P is None else P
    br = grid.branches
    M = P[:, br.i0.values] - P[:, br.i1.values]          # branch-to-branch PTDF
    n = br.circuits.values.astype(float) if single_circuit else np.ones(len(br))
    d = 1.0 - np.diag(M) / n
    radial = np.abs(d) < 1e-6
    L = (M / n[None, :]) / np.where(radial, np.nan, d)[None, :]
    k = np.arange(len(br))
    L[k, k] = -1.0 / n + (n - 1) / n * np.diag(M) / n / np.where(radial, np.nan, d)
    grid.n1_remaining = np.where(n > 1, (n - 1) / n, 0.0)
    return L, radial


def outage_ptdf(grid, out_idx):
    """PTDF after removing branches ``out_idx``; returns (ptdf_post, islanded_bus_mask)."""
    import networkx as nx
    br = grid.branches
    keep = np.ones(len(br), bool); keep[list(out_idx)] = False
    G = nx.Graph(); G.add_nodes_from(range(len(grid.buses)))
    G.add_edges_from(zip(br.i0.values[keep], br.i1.values[keep]))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    main = comps[0]
    island = np.ones(len(grid.buses), bool); island[list(main)] = False
    nbus = len(grid.buses)
    A = grid.inc[keep]
    bvec = 1.0 / br.x_pu.values[keep]
    B = A.T @ (bvec[:, None] * A)
    keepb = np.array(sorted(main)); keepb = keepb[keepb != grid.slack] if grid.slack in main else keepb[1:]
    Binv = np.zeros((nbus, nbus))
    Binv[np.ix_(keepb, keepb)] = np.linalg.inv(B[np.ix_(keepb, keepb)])
    Pk = (bvec[:, None] * A) @ Binv
    Pfull = np.zeros((len(br), nbus)); Pfull[keep] = Pk
    Pfull[:, island] = 0.0
    return Pfull, island


# ----------------------------------------------------------------------------------------------
# Fleet and load allocation
# ----------------------------------------------------------------------------------------------
@dataclass
class Fleet:
    cap: pd.DataFrame          # zones x TECHS (MW)
    stor_p: pd.DataFrame       # zones x STORAGE (MW)
    stor_e: pd.DataFrame       # zones x STORAGE (MWh)
    gsk: dict                  # tech -> (nbus x 16) bus shares per zone
    lsk: np.ndarray            # (nbus x 16) load shares
    nsk: np.ndarray            # (nbus x n_neighbours)
    neighbours: list
    annual_twh: np.ndarray
    reserve_extra: pd.DataFrame = None
    notes: list = field(default_factory=list)


def _keys_from_points(grid, lon, lat, w, states, polys, v=None):
    """Distribute weights w located at (lon, lat) to buses of the same state; returns nbus x 16."""
    st = assign_states(lon, lat, polys) if states is None else states
    K = np.zeros((len(grid.buses), 16))
    bx = grid.buses
    for z, s in enumerate(STATES):
        m = (st == s)
        bz = bx[bx.state == s]
        if v is not None and (bz.voltage == v).any():
            bz = bz[bz.voltage == v]
        if len(bz) == 0:
            continue
        if m.sum() == 0 or np.nansum(np.asarray(w)[m]) <= 0:
            K[bz.i.values, z] = 1.0 / len(bz)
            continue
        X = np.column_stack([bz.x.values, bz.y.values * 1.6])
        P = np.column_stack([np.asarray(lon)[m], np.asarray(lat)[m] * 1.6])
        j = ((P[:, None, :] - X[None, :, :]) ** 2).sum(2).argmin(1)
        np.add.at(K[:, z], bz.i.values[j], np.asarray(w, float)[m])
        K[:, z] /= K[:, z].sum()
    return K


KEY_VOLTAGE = 380   # injections attach to 380 kV substations (PyPSA-Eur style mapping to 380 kV);
                    # 220 kV lines remain as parallel paths. None -> nearest bus of any voltage.


def spread_load_keys(grid, lon, lat, w, polys, k=3, v=380):
    """Load keys emulating 110 kV meshing: each SciGRID load is shared among the k nearest substations
    (380 kV by default) of its state with inverse-distance weights (sensitivity for load pockets)."""
    st = assign_states(lon, lat, polys)
    K = np.zeros((len(grid.buses), 16))
    bx = grid.buses if v is None else grid.buses[grid.buses.voltage == v]
    for z, s in enumerate(STATES):
        m = st == s
        bz = bx[bx.state == s]
        if m.sum() == 0 or len(bz) == 0:
            if len(bz):
                K[bz.i.values, z] = 1.0 / len(bz)
            continue
        X = np.column_stack([bz.x.values, bz.y.values * 1.6])
        P = np.column_stack([np.asarray(lon)[m], np.asarray(lat)[m] * 1.6])
        d = np.sqrt(((P[:, None, :] - X[None, :, :]) ** 2).sum(2)) + 0.02
        kk = min(k, len(bz))
        nn = np.argsort(d, 1)[:, :kk]
        wd = 1.0 / np.take_along_axis(d, nn, 1); wd /= wd.sum(1, keepdims=True)
        ww = np.asarray(w, float)[m][:, None] * wd
        np.add.at(K[:, z], bz.i.values[nn.ravel()], ww.ravel())
        K[:, z] /= K[:, z].sum()
    return K


LOAD_SPREAD_K = 3  # each SciGRID load is shared among the 3 nearest 380 kV substations of its state
                   # (proxy for the meshed 110 kV supply of urban load); 1 = nearest substation only


def build_fleet(grid, static, with_grid_reserve=False, battery_fill=None, key_voltage=KEY_VOLTAGE,
                load_spread_k=LOAD_SPREAD_K):
    """Allocate REACT state totals (and supplementary techs) to buses.

    ``static`` is REACT's structural panel (index = state). Thermal state totals follow REACT
    (BNetzA power-plant list, market-active fossil capacity); powerplantmatching provides the
    fuel split and plant locations. ``with_grid_reserve`` instead uses the powerplantmatching
    totals (which include plants held in the grid reserve).
    """
    import pypsa, warnings
    warnings.filterwarnings("ignore")
    polys = state_polygons()
    ppm = pd.read_csv(os.path.join(CACHE, "powerplants.csv"), low_memory=False)
    ppm = ppm[(ppm.Country == "Germany") & (ppm.DateOut.isna() | (ppm.DateOut > 2024.5))
              & (ppm.DateIn.fillna(1900) <= 2024.5)].dropna(subset=["lat", "lon"]).copy()
    ppm["state"] = assign_states(ppm.lon.values, ppm.lat.values, polys)
    ppm = ppm[ppm.state.notna()]
    fmap = {"Lignite": "LIG", "Hard Coal": "COAL", "Natural Gas": "GAS", "Oil": "OIL",
            "Biogas": "BIO", "Solid Biomass": "BIO", "Waste": "BIO", "Other": "BIO", "Geothermal": "BIO"}
    ppm["tech"] = ppm.Fueltype.map(fmap)
    ppm.loc[(ppm.Fueltype == "Hydro") & ppm.Technology.isin(["Run-Of-River", "Reservoir"]), "tech"] = "ROR"
    ppm.loc[(ppm.Fueltype == "Hydro") & (ppm.Technology == "Pumped Storage"), "tech"] = "PHS"
    cap = pd.DataFrame(0.0, index=STATES, columns=TECHS)
    gsk = {}
    th = ppm[ppm.tech.isin(THERMAL)].pivot_table(index="state", columns="tech", values="Capacity",
                                                  aggfunc="sum").reindex(STATES).fillna(0.0)
    for k in THERMAL:
        if k not in th:
            th[k] = 0.0
    th = th[THERMAL]
    notes = []
    for s in STATES:
        tot = float(static.at[s, "Thermal_Fossil_MW"]) if not with_grid_reserve else float(th.loc[s].sum())
        shares = th.loc[s] / th.loc[s].sum() if th.loc[s].sum() > 0 else pd.Series([0, 0, 1, 0], index=THERMAL)
        cap.loc[s, THERMAL] = tot * shares.values
    notes.append("thermal by state: REACT totals (BNetzA list) split by powerplantmatching fuel shares; "
                 f"ppm active total {th.values.sum():.0f} MW vs REACT {static['Thermal_Fossil_MW'].sum():.0f} MW")
    for k in ("BIO", "ROR"):
        cap[k] = ppm[ppm.tech == k].groupby("state").Capacity.sum().reindex(STATES).fillna(0.0)
    for k in THERMAL + ["BIO", "ROR"]:
        sub = ppm[ppm.tech == k]
        gsk[k] = _keys_from_points(grid, sub.lon.values, sub.lat.values, sub.Capacity.values, sub.state.values, polys, v=key_voltage)
    # pumped hydro
    phs = ppm[ppm.tech == "PHS"].copy()
    # Daily-cycle PHS: usable energy = P x duration, duration from powerplantmatching capped at 8 h
    # (Schluchsee-type cascades report seasonal reservoir volumes that are not usable as daily
    # storage); 6 h where missing. Germany total is then of the order of the ~40 GWh usually cited.
    dur = phs.Duration.where(phs.Duration > 0).fillna(6.0).clip(upper=8.0)
    phs["E"] = phs.Capacity * dur
    stor_p = pd.DataFrame(0.0, index=STATES, columns=STORAGE); stor_e = stor_p.copy()
    stor_p["PHS"] = phs.groupby("state").Capacity.sum().reindex(STATES).fillna(0.0)
    stor_e["PHS"] = phs.groupby("state").E.sum().reindex(STATES).fillna(0.0)
    gsk["PHS"] = _keys_from_points(grid, phs.lon.values, phs.lat.values, phs.Capacity.values, phs.state.values, polys, v=key_voltage)
    # batteries: REACT storage energy (MaStR); E/P = 1.5 h
    bat_e = static.loc[STATES, "Storage_MWh"].astype(float).copy()
    if battery_fill is not None:
        bat_e = bat_e.where(bat_e > 0, battery_fill)
    stor_e["BAT"] = bat_e.values; stor_p["BAT"] = bat_e.values / 1.5
    # onshore wind and PV: REACT totals, SciGRID-DE spatial keys
    n = pypsa.Network(os.path.join(CACHE, "scigrid-de.nc"))
    sb = n.buses[["x", "y"]]
    for k, carrier, col in (("WON", "Wind Onshore", "Wind_Onshore_MW"), ("PV", "Solar", "Solar_PV_MW")):
        g = n.generators[n.generators.carrier == carrier]
        xy = sb.loc[g.bus]
        gsk[k] = _keys_from_points(grid, xy.x.values, xy.y.values, g.p_nom.values, None, polys, v=key_voltage)
        cap[k] = static.loc[STATES, col].astype(float).values
    # offshore
    cap["WOFF"] = 0.0
    K = np.zeros((len(grid.buses), 16))
    for s, (mw, sites) in OFFSHORE.items():
        cap.loc[s, "WOFF"] = mw
        for (lon, lat), sh in sites:
            K[nearest_bus(grid.buses, lon, lat, v=key_voltage, state=s), ZIDX[s]] += sh
    gsk["WOFF"] = K
    gsk["BAT"] = gsk["PV"].copy()
    # load: SciGRID-DE bus loads as spatial key, scaled to REACT annual consumption
    ld = n.loads_t.p_set.mean()
    xy = sb.loc[n.loads.bus.loc[ld.index]]
    lsk = (_keys_from_points(grid, xy.x.values, xy.y.values, ld.values, None, polys, v=key_voltage)
           if load_spread_k in (None, 1) else spread_load_keys(grid, xy.x.values, xy.y.values, ld.values, polys, load_spread_k,
                                                          v=key_voltage))
    # neighbours
    nbs = list(NEIGHBOURS)
    nsk = np.zeros((len(grid.buses), len(nbs)))
    for j, nb in enumerate(nbs):
        for i, sh in grid.nb_attach.get(nb, {}).items():
            nsk[i, j] = sh
    annual = static.loc[STATES, "Cons_TWh"].astype(float).values
    for k in TECHS:
        if k not in gsk:
            gsk[k] = lsk.copy()
    fl = Fleet(cap=cap, stor_p=stor_p, stor_e=stor_e, gsk=gsk, lsk=lsk, nsk=nsk, neighbours=nbs,
               annual_twh=annual, notes=notes)
    return fl


CNEC_Z2Z = 0.05   # Core CCR CNEC selection: maximum zone-to-zone PTDF >= 5 %


def mark_cnecs(grid, fleet, threshold=CNEC_Z2Z):
    """Flag critical network elements: branches whose maximum zone-to-zone PTDF (states and
    neighbours, load-proportional keys) reaches ``threshold``. Branches below it carry
    predominantly intra-zonal flows whose loading in a zonal model is an artefact of fixed
    shift keys (110 kV support and local redispatch are not represented)."""
    PZ = grid.ptdf @ np.hstack([fleet.lsk, fleet.nsk])
    grid.branches["z2z"] = PZ.max(1) - PZ.min(1)
    grid.branches["cnec"] = grid.branches.z2z >= threshold
    return grid.branches.cnec.values


# Planned HVDC corridors (Netzentwicklungsplan; commissioning expected 2026-2028), used only in the
# forward-grid sensitivity: (name, from (lon, lat), to (lon, lat), MW)
HVDC_PROJECTS = [("Ultranet", (6.62, 51.27), (8.44, 49.25), 2000.0),
                 ("A-Nord", (7.20, 53.35), (6.62, 51.27), 2000.0),
                 ("SuedLink-BW", (9.17, 53.90), (9.15, 49.15), 2000.0),
                 ("SuedLink-BY", (9.37, 53.92), (10.17, 50.00), 2000.0),
                 ("SuedOstLink", (11.63, 52.25), (12.20, 48.60), 2000.0)]


def add_hvdc(grid, fleet, projects=HVDC_PROJECTS):
    ends = [(nearest_bus(grid.buses, *a, v=380), nearest_bus(grid.buses, *b, v=380)) for _, a, b, _ in projects]
    fleet.hvdc_ends = ends
    fleet.hvdc_cap = np.array([p[3] for p in projects])
    fleet.hvdc_names = [p[0] for p in projects]
    return fleet


def zonal_ptdf(grid, fleet, P=None):
    """Zonal PTDFs per technology: dict key -> (nbranch x 16); neighbours -> (nbranch x nnb)."""
    P = grid.ptdf if P is None else P
    Z = {k: P @ fleet.gsk[k] for k in TECHS + STORAGE}
    Z["LOAD"] = P @ fleet.lsk
    Z["NB"] = P @ fleet.nsk
    if getattr(fleet, "hvdc_ends", None):
        # transfer h from 'from' to 'to': injection -h at from, +h at to
        Z["HVDC"] = np.column_stack([P[:, t] - P[:, f] for f, t in fleet.hvdc_ends])
    return Z
