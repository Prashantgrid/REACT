#!/usr/bin/env python3
"""Numbers and appendix tables from the main REACT run, written as LaTeX macros and table bodies.

Usage: python make_main_numbers.py <results_dir> <manuscript revision dir>
Writes numbers_main.tex and tab_main_*.tex, so that no value from the main run is typed by hand.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

RES, OUT = sys.argv[1], sys.argv[2]
RB = os.path.join(RES, "robustness")
AB = {"Baden-Württemberg": "BW", "Bayern": "BY", "Berlin": "BE", "Brandenburg": "BB", "Bremen": "HB",
      "Hamburg": "HH", "Hessen": "HE", "Mecklenburg-Vorpommern": "MV", "Niedersachsen": "NI",
      "Nordrhein-Westfalen": "NRW", "Rheinland-Pfalz": "RP", "Saarland": "SL", "Sachsen": "SN",
      "Sachsen-Anhalt": "ST", "Schleswig-Holstein": "SH", "Thüringen": "TH"}
TEXN = {"Baden-Württemberg": 'Baden-W\\"urttemberg', "Thüringen": 'Th\\"uringen'}
LETTER = {"Climate": "C", "Energy": "E", "Adaptive": "S"}
DLONG = {"Climate": "Climate Exposure", "Energy": "Energy System", "Adaptive": "Socio-economic Capacity"}
SC = ["Baseline", "National Pool", "Adjacency Flow"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
       "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth"]
WORD = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]
M = {}


def put(k, v):
    M[k] = v if isinstance(v, str) else str(v)


def tx(s):
    return TEXN.get(s, s)


def andlist(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


# pooled 1990--2024 hazard ranges used to normalise the derating drivers (basis of the caps)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import REACT_v4 as core  # noqa: E402
hz_ = core.load_hazard_panel()
lw_ = hz_.low_wind_days_mean.dropna()
put("LowWindMin", f"{lw_.min():.0f}"); put("LowWindMax", f"{lw_.max():.0f}")
put("LowWindShareMin", f"{100 * lw_.min() / 365:.0f}"); put("LowWindShareMax", f"{100 * lw_.max() / 365:.0f}")
put("HotDaysPoolMax", f"{hz_.hot_days_mean.max():.0f}")

au = pd.read_csv(os.path.join(RES, "classification_audit.csv"))
ident = au[au.identified]
cases = {}
for r in ident.itertuples():
    cases.setdefault((r.State, r.Provisional_domain), []).append(r.Scenario)
HEAD = sorted(cases, key=lambda k: (-len(cases[k]), {"Bayern": 0, "Baden-Württemberg": 1}.get(k[0], 2), k[0]))
put("NIdentStates", WORD[len({s for s, _ in HEAD})])
put("NIdentClimate", WORD[sum(len(v) for (s, d), v in cases.items() if d == "Climate")])
put("NMultidomain", int((~au.identified).sum()))

# ---- domain correlations, M0/M1/M2, ranks ---------------------------------------------------------
dc = pd.read_csv(os.path.join(RES, "domain_score_correlations.csv")).set_index("pair")
put("rESSpearman", f"{dc.loc['Energy~Adaptive', 'spearman']:.2f}")
m = pd.read_csv(os.path.join(RES, "m0_m1_m2_vulnerability.csv"), index_col=0)
put("MaxDiffMZeroTwo", f"{float((m.max(axis=1) - m.min(axis=1)).max()):.3f}")
rk = {c: pd.Series(rankdata(m[c]), index=m.index) for c in m.columns}       # 1 = least vulnerable
sh = (rk["M0"] - rk["M2"]).abs()
put("MaxShiftMZeroTwo", WORD[int(sh.max())])
m1 = (rk["M0"] - rk["M1"]).abs().sort_values(ascending=False)
put("MOneMovers", andlist([tx(s) for s in m1.index[:2]]))
put("MOneShift", WORD[int(m1.iloc[0])])
b = pd.read_csv(os.path.join(RES, "results_baseline.csv"), index_col=0).sort_values("Expected_rank")
put("TopThree", f"{tx(b.index[0])}, {tx(b.index[1])}, and {tx(b.index[2])}")
put("LastState", tx(b.index[-1]))
er = b.Expected_rank.to_numpy()
best = max(int(((er >= er[i]) & (er <= er[i] + 1.0)).sum()) for i in range(len(er)))
put("MidGroupN", WORD[best])
sr = pd.read_csv(os.path.join(RES, "scenario_rank_correlations.csv")).set_index("pair").spearman
put("RhoBNP", f"{sr['baseline~national_pool']:.3f}"); put("RhoBAF", f"{sr['baseline~adjacency_flow']:.3f}")
put("RhoScenMin", f"{sr.min():.2f}")
prov = au.pivot(index="State", columns="Scenario", values="Provisional_domain")
same = prov.nunique(axis=1) == 1
put("NSameProv", WORD[int(same.sum())])
put("ChangeProvStates", andlist([tx(s) for s in prov.index[~same]]))

# ---- panel deletion -----------------------------------------------------------------------------
pdl = pd.read_csv(os.path.join(RB, "robustness_panel_deletion.csv"))
put("NDelKept", int((pdl.provisional == pdl.provisional_full).sum())); put("NDelAll", len(pdl))
parts = []
for s, d in HEAD:
    msc = cases[(s, d)]
    y = pdl[(pdl.state == s) & pdl.scenario.isin(msc)]
    per = [int(y[y.scenario == sc].identified.sum()) for sc in msc]
    n = y[y.scenario == msc[0]].deleted.nunique()
    where = "in all three scenarios" if len(msc) == 3 else (
        "under both coherent scenarios" if set(msc) == {"National Pool", "Adjacency Flow"} else "under " + andlist(msc))
    if all(p == n for p in per):
        parts.append(f"{tx(s)} after every deletion {where}")
    else:
        parts.append(f"{tx(s)} in {min(per)}--{max(per)} of {n} deletions {where}" if min(per) != max(per)
                     else f"{tx(s)} in {per[0]} of {n} deletions {where}")
put("DelSummary", andlist(parts))
put("BYAllDel", "yes" if all(int(pdl[(pdl.state == "Bayern") & (pdl.scenario == sc)].identified.sum())
                             == pdl.deleted.nunique() - 0 for sc in cases.get(("Bayern", "Climate"), [])) else "no")

# ---- leverage, permutation ------------------------------------------------------------------------
lev = pd.read_csv(os.path.join(RB, "regression_leverage.csv")).set_index("regression").max_leverage
put("LevC", f"{lev['Climate']:.3f}"); put("LevE", f"{lev['Energy']:.3f}"); put("LevS", f"{lev['Adaptive']:.3f}")
pmin = au.p_permutation.min()
put("NPermMin", int(np.isclose(au.p_permutation, pmin).sum()))
put("PermIdAll", "yes" if np.isclose(ident.p_permutation, pmin).all() else "no")

# ---- Monte Carlo replicas -----------------------------------------------------------------------
ss = pd.read_csv(os.path.join(RB, "robustness_seed_summary.csv")).set_index("scenario")
put("MCsdRank", f"{ss.sd_max_expected_rank.max():.2f}"); put("MCsdPsel", f"{ss.sd_max_Psel.max():.3f}")
put("MCrhoMin", f"{ss.min_rank_rho.min():.3f}")
put("MCAllSame", "yes" if (ss.same_provisional == 16).all() and (ss.same_classification == 16).all() else "no")
rows = [f"{sc} & {ss.loc[sc, 'sd_max_expected_rank']:.2f} & {ss.loc[sc, 'sd_max_Psel']:.3f} & "
        f"{ss.loc[sc, 'min_rank_rho']:.3f} & {int(ss.loc[sc, 'same_classification'])}/16 \\\\" for sc in SC]
open(os.path.join(OUT, "tab_main_mc.tex"), "w").write("\n".join(rows) + "\n")

# ---- VIF and indicator reduction -------------------------------------------------------------------
v = pd.read_csv(os.path.join(RB, "robustness_vif_within_domain.csv")).set_index("indicator").VIF
put("VIFPthirty", f"{v['P30_Days']:.1f}"); put("VIFRx", f"{v['Rx1day']:.1f}"); put("VIFHDI", f"{v['HDI']:.1f}")
en = ["Wind_MW", "Solar_MW", "R_ri", "Thermal_MW", "WLINES", "C_i", "Storage_MWh"]
put("VIFEnergyMax", f"{v[en].max():.1f}")
vc = pd.read_csv(os.path.join(RB, "robustness_vif_energy_by_construct.csv"), index_col=0)
top = vc.absolute.sort_values(ascending=False).head(3)
NM = {"WLINES": "grid", "Storage_MWh": "storage", "Solar_MW": "PV", "Wind_MW": "wind", "Thermal_MW": "thermal",
      "C_i": "connectivity", "R_ri": "external-support"}
put("VIFAbsText", andlist([f"{NM[k]} ({x:.1f})" for k, x in top.items()]))
put("VIFAbsMax", f"{top.iloc[0]:.1f}")
ir = pd.read_csv(os.path.join(RB, "robustness_indicator_reduction.csv"))
rows = [f"{r.tau:.1f} & {int(r.n_kept)} & {r.rho:.3f} & {int(r.max_shift)} & {r.mean_shift:.2f} \\\\" for r in ir.itertuples()]
open(os.path.join(OUT, "tab_main_reduction.tex"), "w").write("\n".join(rows) + "\n")
hi = ir[(ir.tau >= 0.7) & (ir.tau < 0.9)]
put("RedRhoMin", f"{hi.rho.min():.2f}"); put("RedRhoMax", f"{hi.rho.max():.2f}")
_t = []
for r in hi.sort_values("tau", ascending=False).itertuples():
    ids = [x.split(":") for x in str(r.identified).split("; ") if ":" in x]
    lab = andlist([f"{tx(a)}'s {DLONG[{'Climate': 'Climate', 'Energy': 'Energy', 'Adaptive': 'Adaptive'}[b]]}"
                   for a, b in ids]) if ids else "no case"
    _t.append(f"at $\\tau={r.tau:.1f}$ only {lab} is identified" if len(ids) == 1 else
              f"at $\\tau={r.tau:.1f}$, {lab} are identified")
put("RedIdText", "; ".join(_t))
south = [sum(x in str(r.identified) for x in ("Bayern:Climate", "Baden-Württemberg:Climate")) for r in hi.itertuples()]
put("RedSouthMin", WORD[min(south)])

# ---- threshold sweep -------------------------------------------------------------------------------
th = pd.read_csv(os.path.join(RB, "robustness_threshold_sweep.csv"))
th = th[th.state != "__total__"]
rows = []
for t, g in th.groupby("threshold"):
    order = {s: i for i, (s, _) in enumerate(HEAD)}
    g = g.assign(o=[order.get(x, 99) for x in g.state]).sort_values(["n_scenarios", "o", "state"],
                                                                      ascending=[False, True, True])
    lst = ", ".join(f"{AB[r.state]} ({int(r.n_scenarios)})" for r in g.itertuples())
    n = int(g.n_scenarios.sum())
    if abs(t - 0.70) < 1e-9:
        rows.append(f"\\textbf{{{t:.2f}}} & \\textbf{{{n}}} & \\textbf{{{lst}}} \\\\")
    else:
        rows.append(f"{t:.2f} & {n} & {lst} \\\\")
open(os.path.join(OUT, "tab_main_threshold.tex"), "w").write("\n".join(rows) + "\n")
base = {(r.state, r.domain): r.n_scenarios for r in th[np.isclose(th.threshold, 0.70)].itertuples()}
for t, tag in ((0.75, "Up"), (0.65, "Down")):
    g = {(r.state, r.domain): r.n_scenarios for r in th[np.isclose(th.threshold, t)].itertuples()}
    lost = [tx(s) for (s, d) in base if (s, d) not in g]
    gained = [tx(s) for (s, d) in g if (s, d) not in base]
    put(f"Thr{tag}Lost", andlist(lost) if lost else "none")
    put(f"Thr{tag}Gained", andlist(gained) if gained else "none")
up = {(r.state, r.domain) for r in th[np.isclose(th.threshold, 0.75)].itertuples()}
put("ThrUpKept", andlist([tx(s) for s, d in sorted(up)]) if up else "none")

# ---- specification sensitivity table ----------------------------------------------------------------
sp = pd.read_csv(os.path.join(RB, "robustness_specifications.csv"))
SPN = [("main", "Main specification"), ("grid_per_demand", "Grid per average load"),
       ("grid_geometric_mean", "Grid per geometric mean"), ("absolute_capacities", "Absolute capacities"),
       ("equal_weights", "Equal weights"), ("low_hazard_response", "Low hazard response"),
       ("high_hazard_response", "High hazard response"), ("thermal_as_cost", "Thermal capacity as cost")]
hdr = "Specification & " + " & ".join(f"{tx(s)} \\newline ({DLONG[d].split()[0]})" for s, d in HEAD) + \
      " & Other identified & Rank $\\rho$ vs main \\\\"
rows = [hdr, "\\midrule"]
for var, lab in SPN:
    x = sp[sp.specification == var]
    cells = []
    for s, d in HEAD:
        y = x[x.state == s]
        yd = y[y.provisional == d]
        nid = int(yd.identified.sum())
        if len(yd) == 0:
            rng_ = "--$^{a}$"
        else:
            lo_, hi_ = yd.P_sel.min(), yd.P_sel.max()
            rng_ = f"{lo_:.2f}" if abs(hi_ - lo_) < 0.005 else f"{lo_:.2f}--{hi_:.2f}"
            if len(yd) < 3:
                rng_ += "$^{a}$"
        cells.append(f"{rng_} ({nid})")
    oth = sorted({f"{AB[r.state]}-{LETTER[r.provisional]}" for r in x[x.identified].itertuples()
                  if (r.state, r.provisional) not in HEAD})
    rho = "--" if var == "main" else (f"{x.rank_rho_vs_main.min():.2f}--{x.rank_rho_vs_main.max():.2f}"
                                       if x.rank_rho_vs_main.max() - x.rank_rho_vs_main.min() > 0.005
                                       else f"{x.rank_rho_vs_main.min():.2f}")
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {', '.join(oth) or '--'} & {rho} \\\\")
open(os.path.join(OUT, "tab_main_specs.tex"), "w").write("\n".join(rows) + "\n")
put("SpecCols", "Y" + "*{%d}{>{\\centering\\arraybackslash}p{2.05cm}}" % len(HEAD) + ">{\\centering\\arraybackslash}p{1.5cm}>{\\centering\\arraybackslash}p{1.5cm}")
ab = sp[sp.specification == "absolute_capacities"]
put("AbsRhoBase", f"{ab[ab.scenario == 'Baseline'].rank_rho_vs_main.iloc[0]:.2f}")
put("AbsRhoNP", f"{ab[ab.scenario == 'National Pool'].rank_rho_vs_main.iloc[0]:.2f}")
put("AbsRhoAF", f"{ab[ab.scenario == 'Adjacency Flow'].rank_rho_vs_main.iloc[0]:.2f}")
mb = sp[(sp.specification == "main") & (sp.scenario == "Baseline")].set_index("state").expected_rank
abb = ab[ab.scenario == "Baseline"].set_index("state").expected_rank
r0 = mb.rank(); r1 = abb.rank()
mv = (r0 - r1).sort_values()
up = [s for s in mv.index[::-1][:3]]
dn = [s for s in mv.index[:2]]
put("AbsUp", andlist([f"{tx(s)} rises from {ORD[int(r0[s]) - 1]} to {ORD[int(r1[s]) - 1]}" for s in up]))
put("AbsDown", andlist([f"{tx(s)} falls from {ORD[int(r0[s]) - 1]} to {ORD[int(r1[s]) - 1]}" for s in dn]))
ida = {(r.state, r.provisional, r.scenario) for r in ab[ab.identified].itertuples()}


def scen_text(scs):
    if len(scs) == 3:
        return "in all three scenarios"
    if set(scs) == {"National Pool", "Adjacency Flow"}:
        return "under both coherent scenarios"
    return "under " + andlist(scs)


both, only_abs, lost = [], [], []
for s, d in HEAD:
    a = [sc for (ss, dd, sc) in ida if ss == s and dd == d]
    (both if a else lost).append(tx(s))
extra = {}
for (s, d, sc) in ida:
    if (s, d) not in HEAD:
        extra.setdefault((s, d), []).append(sc)
put("AbsBoth", andlist(both) if both else "none")
put("AbsLost", andlist(lost) if lost else "none")
put("AbsExtra", andlist([f"{tx(s)}'s {DLONG[d]} weakness {scen_text(v)}" for (s, d), v in sorted(extra.items())])
    if extra else "no further weaknesses")

with open(os.path.join(OUT, "numbers_main.tex"), "w") as fh:
    fh.write("% generated by make_main_numbers.py -- do not edit by hand\n")
    for k, val in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(f"wrote {len(M)} macros and 4 tables; headline cases: {HEAD}")

# ---- identified relative weaknesses and their least favourable indicators (Table 5) --------------------
ir_ = pd.read_csv(os.path.join(RES, "indicator_ranks.csv"), index_col=0)
IND = {"Hot_Days": "hot days", "Soil_Moist": "winter soil moisture", "P30_Days": "heavy-precipitation days",
       "Rx1day": "maximum 1-day precipitation", "Storm_Days": "storm days", "Low_Wind": "low-wind days",
       "Wind_MW": "wind per load", "Solar_MW": "PV per load", "R_ri": "external support",
       "Thermal_MW": "thermal capacity per load", "WLINES": "grid density", "C_i": "interstate connectivity",
       "Storage_MWh": "storage hours", "GDP_pc": "GDP per capita", "HDI": "human development",
       "Patents": "patents", "ADR": "age dependency"}
BLK = {"Climate": ["Hot_Days", "Soil_Moist", "P30_Days", "Rx1day", "Storm_Days", "Low_Wind"],
       "Energy": ["Wind_MW", "Solar_MW", "R_ri", "Thermal_MW", "WLINES", "C_i", "Storage_MWh"],
       "Adaptive": ["GDP_pc", "HDI", "Patents", "ADR"]}
rows = []
for s, d in HEAD:
    scs = cases[(s, d)]
    y = au[(au.State == s) & au.Scenario.isin(scs)]
    lo_, hi_ = y.P_sel.min(), y.P_sel.max()
    ps = f"{lo_:.2f}" if abs(hi_ - lo_) < 0.005 else f"{lo_:.2f}--{hi_:.2f}"
    rr_ = ir_.loc[s, BLK[d]].sort_values(ascending=False)
    worst = [f"{IND[k]} ({x:g})" for k, x in rr_.items() if x >= 11][:4]
    worst[0] = worst[0][0].upper() + worst[0][1:]
    where = "All three scenarios" if len(scs) == 3 else ", ".join(scs)
    rows.append(f"{tx(s)} & {DLONG[d]} & {where} & {ps} & {', '.join(worst)} \\\\")
open(os.path.join(OUT, "tab_main_identified.tex"), "w").write("\n".join(rows) + "\n")

# ---- seed agreement -----------------------------------------------------------------------------------
sr_ = pd.read_csv(os.path.join(RB, "robustness_seed_replicas.csv"))
gg = sr_.groupby(["scenario", "state"]).agg(nprov=("provisional", "nunique"), nid=("identified", "sum"),
                                              ns=("seed", "nunique"), pmin=("P_sel", "min"), pmax=("P_sel", "max"))
put("MCSameProv", int((gg.nprov == 1).sum())); put("MCSameClass", int(((gg.nid == 0) | (gg.nid == gg.ns)).sum()))
flip = gg[(gg.nid > 0) & (gg.nid < gg.ns)]
put("MCFlipText", andlist([f"{tx(s)} under {sc} ($P^{{\\mathrm{{sel}}}}={r.pmin:.2f}$--${r.pmax:.2f}$, identified in "
                           f"{int(r.nid)} of {int(r.ns)} replicas)" for (sc, s), r in flip.iterrows()]) if len(flip) else "none")
pv = gg[gg.nprov > 1]
put("MCProvText", andlist([tx(s) for (_, s) in pv.index]) if len(pv) else "none")
put("MCProvPmax", f"{pv.pmax.max():.2f}" if len(pv) else "--")
with open(os.path.join(OUT, "numbers_main.tex"), "w") as fh:
    fh.write("% generated by make_main_numbers.py -- do not edit by hand\n")
    for k, val in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(f"wrote {len(M)} macros (final)")

# ---- regional pattern of provisional domains (Table 6): groups fixed, values computed ------------------
GROUPS = [("South", [["Bayern", "Baden-Württemberg"]], ["Persistent relative climate weakness"]),
          ("East", [["Brandenburg"], ["Sachsen"]],
           ["Heat and dryness; coherent-scenario weakness", "Storms and extreme precipitation; coherent-scenario weakness"]),
          ("Load centres", [["Hamburg"], ["Berlin", "Nordrhein-Westfalen", "Hessen", "Rheinland-Pfalz"]],
           ["Strongest Energy direction; largely expected for a load centre", "Shared Energy direction"]),
          ("Socio-economic", [["Mecklenburg-Vorpommern"], ["Thüringen", "Sachsen-Anhalt", "Saarland"]],
           ["Strongest socio-economic direction", "Shared direction; compound in Sachsen-Anhalt"]),
          ("Balanced", [["Niedersachsen", "Bremen", "Schleswig-Holstein"]], ["Balanced profiles"])]
PD = {"Climate": "Climate Exposure", "Energy": "Energy System", "Adaptive": "Socio-economic"}
rows = []
for gi, (glab, members, reads) in enumerate(GROUPS):
    if gi:
        rows.append("\\addlinespace")
    for k, (sts, rd) in enumerate(zip(members, reads)):
        y = au[au.State.isin(sts)]
        doms = set(y.Provisional_domain)
        adverse = (y.T_joint > 0).any()
        dom = PD[doms.pop()] if len(doms) == 1 and adverse else ("No adverse domain" if not adverse else "Mixed")
        nid = int(y.identified.sum())
        rows.append(f"{glab if k == 0 else ''} & {', '.join(tx(x) for x in sts)} & {dom} & "
                    f"{y.P_sel.min():.2f}--{y.P_sel.max():.2f} & {nid} of {len(y)} & {rd} \\\\")
assert sorted(sum([sum(m, []) for _, m, _ in GROUPS], [])) == sorted(au.State.unique())
open(os.path.join(OUT, "tab_main_typology.tex"), "w").write("\n".join(rows) + "\n")
print("wrote tab_main_typology.tex")
