#!/usr/bin/env python3
"""Tables and macros for the Reviewer 1 analyses (REACT only).

Usage: python make_r1_outputs.py <results_dir> <manuscript revision dir>
Reads <results_dir>/r1/r1_copula.csv, r1_params.csv, r1_weights.csv, r1_copula_fits.csv,
r1_hazard_trends.csv, r1_dwd_hotdays_comparison.csv and the main classification audit.
Writes tab_r1_sens.tex and numbers_r1.tex.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES, OUT = sys.argv[1], sys.argv[2]
R1 = os.path.join(RES, "r1")
AB = {"Baden-Württemberg": "BW", "Bayern": "BY", "Berlin": "BE", "Brandenburg": "BB", "Bremen": "HB",
      "Hamburg": "HH", "Hessen": "HE", "Mecklenburg-Vorpommern": "MV", "Niedersachsen": "NI",
      "Nordrhein-Westfalen": "NRW", "Rheinland-Pfalz": "RP", "Saarland": "SL", "Sachsen": "SN",
      "Sachsen-Anhalt": "ST", "Schleswig-Holstein": "SH", "Thüringen": "TH"}
LETTER = {"Climate": "C", "Energy": "E", "Adaptive": "S"}
M = {}


def put(k, v):
    M[k] = v if isinstance(v, str) else str(v)


au = pd.read_csv(os.path.join(RES, "classification_audit.csv"))
main = au.rename(columns={"Scenario": "scenario", "State": "state", "Provisional_domain": "provisional",
                          "Expected_rank": "expected_rank"})
main_id = {(r.state, r.provisional, r.scenario) for r in main.itertuples() if r.identified}
# headline cases: identified in the main specification, ordered by number of scenarios, then name
_cnt = {}
for st_, dm_, sc_ in main_id:
    _cnt[(st_, dm_)] = _cnt.get((st_, dm_), 0) + 1
HEAD = sorted(_cnt, key=lambda k: (-_cnt[k], ["Bayern", "Baden-Württemberg"].index(k[0])
                                   if k[0] in ("Bayern", "Baden-Württemberg") else 2, k[0]))
frames = [pd.read_csv(os.path.join(R1, f)) for f in ("r1_copula.csv", "r1_params.csv", "r1_weights.csv")
          if os.path.exists(os.path.join(R1, f))]
v = pd.concat(frames, ignore_index=True)


def rho_min(x):
    out = []
    for sc, g in x.groupby("scenario"):
        b = main[main.scenario == sc].set_index("state").expected_rank
        out.append(spearmanr(g.set_index("state").expected_rank, b.reindex(g.state)).statistic)
    return min(out)


def cell(x, s, d):
    y = x[(x.state == s) & (x.provisional == d) & x.identified]
    return f"{y.scenario.nunique()}/{x.scenario.nunique()}"


def others(x):
    y = x[x.identified]
    ids = sorted({f"{AB[r.state]}-{LETTER[r.provisional]}" for r in y.itertuples()
                  if (r.state, r.provisional) not in HEAD})
    return ", ".join(ids) or "--"


LABELS = {
    "split_1990_2007": "Fit to 1990--2007", "split_2008_2024": "Fit to 2008--2024",
    "trend_adjusted_2024": "Trends removed, record at 2024 level", "gaussian_copula": "Gaussian copula",
    "credit_wind_0.05": "Wind credit 0.05", "credit_wind_0.20": "Wind credit 0.20",
    "credit_pv_0.00": "PV credit 0", "credit_pv_0.10": "PV credit 0.10",
    "credit_thermal_0.70": "Thermal credit 0.70", "credit_thermal_0.95": "Thermal credit 0.95",
    "storage_window_2h": "Storage window 2~h", "storage_window_8h": "Storage window 8~h",
    "flow_scale_0.125": "$\\phi=0.125$", "flow_scale_0.50": "$\\phi=0.50$",
    "penalty_cap_0.25": "Penalty cap 0.25", "penalty_cap_1.00": "Penalty cap 1.0",
    "af_loss_0.50": "Adjacency losses $\\times0.7$", "af_loss_0.90": "Adjacency losses $\\times1.3$",
    "dirichlet_0.5": "Dirichlet $\\alpha=0.5$", "dirichlet_2": "Dirichlet $\\alpha=2$", "dirichlet_5": "Dirichlet $\\alpha=5$",
    "climate_emphasis": "Climate emphasis (0.5/0.25/0.25)", "energy_emphasis": "Energy emphasis (0.25/0.5/0.25)",
    "socio_emphasis": "Socio-economic emphasis (0.25/0.25/0.5)",
    "no_derating": "No derating of Energy System indicators", "thermal_excluded": "Thermal capacity excluded",
    "without_GDP_pc": "Without GDP per capita (2022)", "without_HDI": "Without human development (2022)",
    "without_Patents": "Without patents (2019)", "without_ADR": "Without age dependency (2025)",
    "dwd_hot_days": "Official DWD hot-day series", "delta2K_reference": "$+2$~K, reference assumptions",
    "delta2K_lowwind_0": "$+2$~K, low-wind days 0\\%/K", "delta2K_lowwind_4": "$+2$~K, low-wind days 4\\%/K",
    "delta2K_p30_10": "$+2$~K, heavy-precipitation days 10\\%/K"}
GROUPS = [("bootstrap", "Hazard record"), ("split", "Hazard record"), ("trend", "Hazard record"),
          ("copula_family", "Hazard record"), ("support", "Support layer"), ("thermal", "Thermal capacity"),
          ("weights", "Weights"), ("double_counting", "Derating"),
          ("socio_years", "Socio-economic indicators"), ("observed_inputs", "Observed inputs"),
          ("warming", "Warming assumptions")]

rows = ["\\begin{tabular}{@{}p{2.9cm}p{5.4cm}" + "c" * len(HEAD) + "p{3.1cm}c@{}}", "\\toprule",
        "Group & Specification & " + " & ".join(f"{AB[s_]}-{LETTER[d_]}" for s_, d_ in HEAD) +
        " & Other identified & Rank $\\rho$ \\\\", "\\midrule",
        "Main & Main specification & " + " & ".join(f"{_cnt[k]}/3" for k in HEAD) + " & -- & 1.00 \\\\"]
last = "Main"
for grp, glab in GROUPS:
    x = v[v.group == grp]
    if x.empty:
        continue
    if glab != last:
        rows.append("\\midrule"); lab = glab; last = glab
    else:
        lab = ""
    if grp == "bootstrap":
        nb = x.variant.nunique()
        cells = []
        for s, d in HEAD:
            msc = [sc for (ss, dd, sc) in main_id if ss == s and dd == d]
            y = x[(x.state == s) & (x.provisional == d) & x.scenario.isin(msc)]
            cells.append(f"{100 * y.identified.mean():.0f}\\%")
        share = x[x.identified].groupby(["state", "provisional"]).variant.nunique() / nb
        oth = ", ".join(f"{AB[s]}-{LETTER[d]} ({100 * f:.0f}\\%)" for (s, d), f in share.items()
                        if (s, d) not in HEAD and f >= 0.10) or "--"
        rhos = [rho_min(g) for _, g in x.groupby("variant")]
        rows.append(f"{lab} & Year-block bootstrap ({nb} replicates) & " + " & ".join(cells) +
                    f" & {oth} & {np.median(rhos):.2f} \\\\")
        lab = ""
        continue
    for var in x.variant.unique():
        y = x[x.variant == var]
        rows.append(f"{lab} & {LABELS.get(var, var)} & " + " & ".join(cell(y, s, d) for s, d in HEAD) +
                    f" & {others(y)} & {rho_min(y):.2f} \\\\")
        lab = ""
rows += ["\\bottomrule", "\\end{tabular}",
         "\\par\\smallskip\\raggedright\\scriptsize Cells: scenarios in which the case is identified, of the scenarios run "
         "(support-layer variants affect only the two spatially coherent scenarios). Bootstrap: share of replicate--scenario "
         "runs in which the case is identified, over the scenarios in which it is identified in the main specification; "
         "other cases listed if identified in at least 10\\% of replicates. Rank $\\rho$: minimum over scenarios of the "
         "expected-rank correlation with the main specification (bootstrap: median over replicates). Emphasis profiles: "
         "Dirichlet domain weights with the stated means and concentration 8."]
open(os.path.join(OUT, "tab_r1_sens.tex"), "w").write("\n".join(rows) + "\n")
print("wrote tab_r1_sens.tex")

# ---- macros ---------------------------------------------------------------------------------------
b = v[v.group == "bootstrap"]
if not b.empty:
    put("NBoot", b.variant.nunique())
    for (s, d), tag in zip(HEAD, [AB[s_] for s_, _ in HEAD]):
        msc = [sc for (ss, dd, sc) in main_id if ss == s and dd == d]
        y = b[(b.state == s) & (b.provisional == d) & b.scenario.isin(msc)]
        put(f"Boot{tag}", f"{100 * y.identified.mean():.0f}")
    put("BootRhoMed", f"{np.median([rho_min(g) for _, g in b.groupby('variant')]):.2f}")
    fits = pd.read_csv(os.path.join(R1, "r1_copula_fits.csv"))
    bf = fits[fits.variant.str.startswith("boot")]
    put("BootNt", int((bf.model == "student_t").sum()))
    t = bf[bf.model == "student_t"].df
    put("BootDfMin", f"{t.min():.1f}" if len(t) else "--"); put("BootDfMax", f"{t.max():.1f}" if len(t) else "--")
    for var, tag in (("split_1990_2007", "SplitA"), ("split_2008_2024", "SplitB")):
        if (fits.variant == var).any():
            f = fits[fits.variant == var].iloc[0]
            put(f"{tag}Model", "Student-$t$" if f.model == "student_t" else f.model.capitalize())
tr = os.path.join(R1, "r1_hazard_trends.csv")
if os.path.exists(tr):
    t = pd.read_csv(tr)
    put("NTrendSig", int((t.p < 0.05).sum())); put("NTrendAll", len(t))
    h = t[t.hazard == "hot_days_mean"]
    put("NTrendHot", int((h.p < 0.05).sum()))
    put("HotTrendMin", f"{h.slope_per_decade.min():.1f}"); put("HotTrendMax", f"{h.slope_per_decade.max():.1f}")


def ids_of(var):
    y = v[(v.variant == var) & v.identified]
    return {(r.state, r.provisional, r.scenario) for r in y.itertuples()}


for var, tag in (("no_derating", "NoDer"), ("dwd_hot_days", "Dwd"), ("trend_adjusted_2024", "Trend"),
                 ("gaussian_copula", "Gauss"), ("thermal_excluded", "ThEx"), ("split_1990_2007", "SplitA"),
                 ("split_2008_2024", "SplitB")):
    if var in set(v.variant):
        s_ = ids_of(var)
        put(f"Same{tag}", "yes" if s_ == main_id else "no")
        put(f"NId{tag}", len(s_))
        y = v[v.variant == var].merge(main[["scenario", "state", "P_sel"]], on=["scenario", "state"], suffixes=("", "_m"))
        put(f"DPsel{tag}", f"{(y.P_sel - y.P_sel_m).abs().max():.3f}")
if "no_derating" in set(v.variant):
    put("RhoNoDer", f"{rho_min(v[v.variant == 'no_derating']):.2f}")
    # selection frequencies of main cases that drop out without the derating
    y = v[(v.variant == "no_derating")]
    lost_nd = main_id - ids_of("no_derating")
    ps = [float(y[(y.state == s_) & (y.provisional == d_) & (y.scenario == sc)].P_sel.iloc[0]) for s_, d_, sc in lost_nd]
    if ps:
        put("NoDerLostPsel", f"{min(ps):.2f}" if min(ps) == max(ps) else f"{min(ps):.2f}--{max(ps):.2f}")
# size of the derating relative to the cross-state spread of the derated indicators
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from REACT_v4 import PARAMS_DERATING  # noqa: E402
dmax = [p_["delta_max"] for p_ in PARAMS_DERATING.values()]
dmax.append(PARAMS_DERATING["Grid"]["delta_max"] + PARAMS_DERATING["Grid"]["gamma"])  # compound-event step
put("DerMaxMin", f"{100 * min(dmax):.0f}"); put("DerMaxMax", f"{100 * max(dmax):.0f}")
panel = pd.read_csv(os.path.join(RES, "structural_indicator_panel.csv"), index_col=0)
der = panel[["Wind_MW", "Solar_MW", "Thermal_MW", "WLINES"]]
cv = der.std() / der.mean()
put("DerCVMin", f"{cv.min():.2f}"); put("DerCVMax", f"{cv.max():.2f}")
sup = v[v.group == "support"]
if not sup.empty:
    put("NSupport", sup.variant.nunique())
    bybw = sup[sup.state.isin(["Bayern", "Baden-Württemberg"]) & (sup.provisional == "Climate") & sup.identified]
    put("SupportSouthAll", "yes" if bybw.groupby("variant").size().min() == 4 and bybw.variant.nunique() == sup.variant.nunique() else "no")
    bb = sup[(sup.state == "Brandenburg") & (sup.provisional == "Climate") & sup.identified]
    put("SupportBBAll", "yes" if bb.variant.nunique() == sup.variant.nunique() and bb.groupby("variant").size().min() == 2 else "no")
    for s_, d_ in HEAD:
        if s_ in ("Bayern", "Baden-Württemberg", "Brandenburg"):
            continue
        y = sup[(sup.state == s_) & (sup.provisional == d_)].groupby("variant").identified.sum()
        put(f"Support{AB[s_]}Min", int(y.min())); put(f"Support{AB[s_]}Max", int(y.max()))
    put("SupportRhoMin", f"{min(rho_min(g) for _, g in sup.groupby('variant')):.2f}")
dw = os.path.join(R1, "r1_dwd_hotdays_comparison.csv")
if os.path.exists(dw):
    d = pd.read_csv(dw)
    put("DwdMAD", f"{(d.panel - d.dwd).abs().mean():.3f}"); put("DwdMax", f"{(d.panel - d.dwd).abs().max():.2f}")
    put("DwdN", len(d)); put("DwdStates", d.state.nunique())
# ---- prose summaries built from the results (used in the response letter) -----------------------------
TEXN = {"Baden-Württemberg": "Baden-W\\\"urttemberg", "Thüringen": "Th\\\"uringen"}
DLONG = {"Climate": "Climate Exposure", "Energy": "Energy System", "Adaptive": "Socio-economic Capacity"}
SCS = {"Baseline": "Baseline", "National Pool": "National Pool", "Adjacency Flow": "Adjacency Flow"}


SC_ORDER = {"Baseline": 0, "National Pool": 1, "Adjacency Flow": 2}


def case_list(cases):
    by = {}
    for st_, dm, sc in sorted(cases):
        by.setdefault((st_, dm), []).append(sc)
    parts = []
    for (st_, dm), scs in by.items():
        scs = sorted(scs, key=SC_ORDER.get)
        parts.append(f"{TEXN.get(st_, st_)}'s {DLONG[dm]} ({', '.join(SCS[x] for x in scs)})")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def of(k, n):
    return f"all {n}" if k == n else f"{k} of {n}"


def diff_text(var, ref=None):
    ref = main_id if ref is None else ref
    s_ = ids_of(var)
    scen = set(v[v.variant == var].scenario)
    ref = {c for c in ref if c[2] in scen}
    lost, gained = ref - s_, s_ - ref
    if not lost and not gained:
        return "reproduces the identified cases"
    out = []
    if lost:
        out.append("no longer identifies " + case_list(lost))
    if gained:
        out.append("adds " + case_list(gained))
    return ", and ".join(out)


def south_count(vars_):
    k = n = 0
    for var in vars_:
        s_ = ids_of(var); scen = set(v[v.variant == var].scenario)
        for st_ in ("Bayern", "Baden-Württemberg"):
            for sc in scen:
                if (st_, "Climate", sc) in main_id:
                    n += 1; k += int((st_, "Climate", sc) in s_)
    return k, n


names = {"split_1990_2007": "the fit to 1990--2007", "split_2008_2024": "the fit to 2008--2024",
         "trend_adjusted_2024": "the trend-adjusted record", "gaussian_copula": "the Gaussian copula"}
have = [k for k in names if k in set(v.variant)]
if have:
    k, n = south_count(have)
    txt = "Relative to the main specification, " + "; ".join(f"{names[k_]} {diff_text(k_)}" for k_ in have) + \
          f". Across these {len(have)} fits, the southern Climate Exposure weaknesses are identified in {of(k, n)} of their " \
          "main-specification scenarios."
    put("SplitTrendGaussSummary", txt)
supv = list(dict.fromkeys(v[v.group == "support"].variant))
if supv:
    k, n = south_count(supv)
    bbk = sum(int(("Brandenburg", "Climate", sc) in ids_of(var)) for var in supv for sc in ("National Pool", "Adjacency Flow"))
    extra = []
    for s_, d_ in HEAD:
        if s_ in ("Bayern", "Baden-Württemberg", "Brandenburg"):
            continue
        cnt = [sum(int((s_, d_, sc) in ids_of(var)) for sc in ("National Pool", "Adjacency Flow")) for var in supv]
        extra.append(f"{TEXN.get(s_, s_)}'s {DLONG[d_]} weakness in {of(sum(cnt), 2 * len(supv))}")
    changed = [var for var in supv if ids_of(var) != {c for c in main_id if c[2] != "Baseline"}]
    put("SupportSummary", f"The southern Climate Exposure weaknesses are identified in {of(k, n)} variant--scenario "
        f"cases, Brandenburg's in {of(bbk, 2 * len(supv))}" + "".join(f", and {e}" for e in extra)
        + (f"; none of the {len(supv)} variants changes a classification" if not changed else
           f"; {len(changed)} of the {len(supv)} variants change a classification") +
        f" (expected-rank correlations at least "
        f"{M.get('SupportRhoMin', '--')}).")
wv = [x for x in ("dirichlet_0.5", "dirichlet_2", "dirichlet_5") if x in set(v.variant)]
WN = {"dirichlet_0.5": "$\\alpha=0.5$", "dirichlet_2": "$\\alpha=2$", "dirichlet_5": "$\\alpha=5$",
      "climate_emphasis": "the climate-emphasis profile", "energy_emphasis": "the energy-emphasis profile",
      "socio_emphasis": "the socio-economic-emphasis profile"}
if wv:
    k, n = south_count(wv)
    by3 = all(all(("Bayern", "Climate", sc) in ids_of(x) for sc in SCS) for x in wv)
    put("WeightSummary", "For the concentrations, relative to the main specification, " + "; ".join(f"{WN[x]} {diff_text(x)}" for x in wv) +
        f". The southern Climate Exposure weaknesses are identified in {of(k, n)} variant--scenario cases"
        + (", and Bayern's in every scenario under every distribution." if by3 else "."))
sv_ = [x for x in ("without_GDP_pc", "without_HDI", "without_Patents", "without_ADR") if x in set(v.variant)]
SN = {"without_GDP_pc": "without GDP per capita", "without_HDI": "without human development",
      "without_Patents": "without patents", "without_ADR": "without age dependency"}
if sv_:
    k, n = south_count(sv_)
    put("SocioSummary", "Relative to the main specification, the analysis " + "; ".join(f"{SN[x]} {diff_text(x)}" for x in sv_) +
        f". The southern Climate Exposure weaknesses are identified in {of(k, n)} cases"
        + (lambda dep: (", whereas " + " and ".join(f"{TEXN.get(x, x)}'s" for x in dep) +
                        (" results depend" if len(dep) > 1 else " result depends") +
                        " on individual socio-economic indicators, consistent with "
                        + ("their" if len(dep) > 1 else "its") + " conditional status.") if dep else ".")(
            sorted({c[0] for x in sv_ for c in ({c for c in main_id} - ids_of(x))
                    if c[0] not in ("Bayern", "Baden-Württemberg")})))
emph = [x for x in ("climate_emphasis", "energy_emphasis", "socio_emphasis") if x in set(v.variant)]
if emph:
    rr_ = [rho_min(v[v.variant == x]) for x in emph]
    put("EmphRhoMin", f"{min(rr_):.2f}"); put("EmphRhoMax", f"{max(rr_):.2f}")
    put("EmphSame", "yes" if all(ids_of(x) == main_id for x in emph) else "no")
    put("EmphSummary", "; ".join(f"{WN[x]} {diff_text(x)}" for x in emph))
    y = v[v.variant.isin(emph)].merge(main[["scenario", "state", "P_sel"]], on=["scenario", "state"], suffixes=("", "_m"))
    put("EmphDPsel", f"{(y.P_sel - y.P_sel_m).abs().max():.3f}")
for var, tag in (("no_derating", "NoDer"), ("thermal_excluded", "ThEx"), ("dwd_hot_days", "Dwd")):
    if var in set(v.variant):
        put(f"{tag}Summary", diff_text(var))
cs = pd.read_csv(os.path.join(R1, "cross_state_rank_corr.csv")).set_index("hazard").mean_cross_state_spearman
put("CrossMin", f"{cs.min():.2f}"); put("CrossMax", f"{cs.max():.2f}")
put("CrossHot", f"{cs.hot_days_mean:.2f}"); put("CrossRx", f"{cs.Rx1day_mm_mean:.2f}")
with open(os.path.join(OUT, "numbers_r1.tex"), "w") as fh:
    fh.write("% generated by make_r1_outputs.py -- do not edit by hand\n")
    for k, val in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(f"wrote {len(M)} macros")
