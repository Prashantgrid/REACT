#!/usr/bin/env python3
"""LaTeX tables of the Reviewer 3 revision (REACT only), written from result files.

Usage: python make_r3_tables.py <results_dir> <out_dir>
Tables: tab_r3_energy.tex (Energy System specification tests), tab_r3_configs.tex (classifications
across model configurations), tab_r3_evidence.tex (evidence levels), tab_r3_climate.tex (warming deltas).
"""
import os
import sys

import pandas as pd

RES, OUT = sys.argv[1], sys.argv[2]
R3 = os.path.join(RES, "r3")
os.makedirs(OUT, exist_ok=True)
AB = {"Baden-Württemberg": "BW", "Bayern": "BY", "Berlin": "BE", "Brandenburg": "BB", "Bremen": "HB",
      "Hamburg": "HH", "Hessen": "HE", "Mecklenburg-Vorpommern": "MV", "Niedersachsen": "NI",
      "Nordrhein-Westfalen": "NRW", "Rheinland-Pfalz": "RP", "Saarland": "SL", "Sachsen": "SN",
      "Sachsen-Anhalt": "ST", "Schleswig-Holstein": "SH", "Thüringen": "TH"}
TEX = {"Baden-Württemberg": "Baden-W\\\"urttemberg", "Thüringen": "Th\\\"uringen"}
DOM = {"Climate": "Climate", "Energy": "Energy", "Adaptive": "Socio-econ.", "Multidomain": "--"}
LONG = {"Climate": "Climate Exposure", "Energy": "Energy System", "Adaptive": "Socio-economic Capacity"}
SCN = ["Baseline", "National Pool", "Adjacency Flow"]


def w(name, s):
    open(os.path.join(OUT, name), "w").write(s + "\n")
    print("wrote", name)


def ids(df):
    out = []
    for sc in SCN:
        y = df[(df.scenario == sc) & df.identified]
        out.append(", ".join(f"{AB[r.state]}-{r.provisional[0] if r.provisional != 'Adaptive' else 'S'}"
                             for r in y.itertuples()) or "--")
    return " / ".join(out)


# ---- 1. Energy System specification tests --------------------------------------------------------
sv = pd.read_csv(os.path.join(R3, "r3_variants.csv"))
summ = pd.read_csv(os.path.join(R3, "r3_variant_summary.csv")).set_index("variant")
names = [("main", "Main specification (seven indicators)"),
         ("no_import_penalty", "Without external-support score $S(r_i)$"),
         ("local_supply_only", "Local supply only (wind, PV, thermal, storage per load)"),
         ("network_access_only", "Network access only (grid density, $C_i$)"),
         ("absolute_capacities", "Absolute capacities (asset stock)")]
rows = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.0cm}>{\\raggedright\\arraybackslash}p{7.4cm}ccc@{}}", "\\toprule",
        "Energy System specification & Identified cases (Base / NP / AF) & \\multicolumn{2}{c}{Hamburg} & $r_{E,S}$ \\\\",
        "\\cmidrule(lr){3-4}", " & & $P^{\\mathrm{sel}}$ & Ident. & \\\\", "\\midrule"]
for v, lab in names:
    if v not in summ.index:
        continue
    x = sv[sv.variant == v]
    hh = x[x.state == "Hamburg"]
    rows.append(f"{lab} & {ids(x)} & {hh.P_sel.min():.3f}--{hh.P_sel.max():.3f} & {int(hh.identified.sum())}/3 & "
                f"${summ.loc[v, 'r_energy_socio']:.2f}$ \\\\")
lp = pd.read_csv(os.path.join(R3, "lite_adjacency_physical_corridors.csv"))
ph = lp[lp.variant == "physical_N1"]
hhp = ph[ph.state == "Hamburg"].iloc[0]
af = ", ".join(f"{AB[r.state]}-{r.provisional[0] if r.provisional != 'Adaptive' else 'S'}" for r in ph[ph.identified].itertuples()) or "--"
rows.append(f"Physical N-1 corridor ratings in Adjacency Flow & -- / -- / {af} & {hhp.P_sel:.3f} (AF) & "
            f"{int(hhp.identified)}/1 & -- \\\\")
rows += ["\\bottomrule", "\\end{tabular}",
         "\\par\\smallskip\\raggedright\\scriptsize Cases: C Climate Exposure, E Energy System, S Socio-economic "
         "Capacity. $r_{E,S}$: correlation of the Baseline Energy System and Socio-economic Capacity scores. Physical "
         "ratings: N-1 ratings of the 220/380~kV lines between adjacent states, in Adjacency Flow only."]
w("tab_r3_energy.tex", "\n".join(rows))

# ---- 2. Classifications across configurations -------------------------------------------------------
cmp = pd.read_csv(os.path.join(R3, "classification_across_configurations.csv"), index_col=0)
cmp = cmp.sort_values("M2_baseline_rank")


def cell(prov, cls):
    if cls != "Multidomain":
        return f"\\textbf{{{DOM[cls]}}}"
    return f"({DOM[prov]})" if prov in DOM and prov != "Multidomain" else "--"


rows = ["\\begin{tabular}{@{}lllllll@{}}", "\\toprule",
        "State & Lowest & M0 & M1 & \\multicolumn{3}{c}{M2 (full REACT)} \\\\", "\\cmidrule(l){5-7}",
        " & score & & & Base & NP & AF \\\\", "\\midrule"]
for s, r in cmp.iterrows():
    c = [DOM[r.conventional_lowest_domain], cell(r.M0_provisional, r.M0_class), cell(r.M1_provisional, r.M1_class)]
    c += [cell(r[f"M2_{k}_provisional"], r[f"M2_{k}_class"]) for k in ("baseline", "national_pool", "adjacency_flow")]
    rows.append(f"{AB[s]} & " + " & ".join(c) + " \\\\")
rows += ["\\midrule",
         "Identified & -- & " + " & ".join(str(int((cmp[k] != "Multidomain").sum())) for k in
                                           ("M0_class", "M1_class", "M2_baseline_class", "M2_national_pool_class",
                                            "M2_adjacency_flow_class")) + " \\\\",
         "\\bottomrule", "\\end{tabular}"]
w("tab_r3_configs.tex", "\n".join(rows))

# ---- 3. Evidence levels ---------------------------------------------------------------------------
au = pd.read_csv(os.path.join(RES, "classification_audit.csv")).rename(
    columns={"Scenario": "scenario", "State": "state", "Provisional_domain": "provisional"})
sp = pd.read_csv(os.path.join(RES, "robustness", "robustness_specifications.csv")).rename(columns={"specification": "variant"})
allv = pd.concat([sp[sp.variant != "main"], sv[sv.variant != "main"]], ignore_index=True)
alt = [v for v in allv.variant.unique() if v != "network_access_only"]
cd = pd.read_csv(os.path.join(R3, "climate_delta_classifications.csv"))
dd = pd.read_csv(os.path.join(R3, "climate_differential_classifications.csv"))
rows_out = []
for (s, dmn), g in au.groupby(["state", "provisional"]):
    n_main = int(g.identified.sum())
    psel_max = float(g.P_sel.max())
    main_sc = list(g[g.identified].scenario)
    sub = allv[(allv.state == s) & (allv.provisional == dmn)]
    n_any = sum(bool(sub[sub.variant == v].identified.any()) for v in alt)
    n_all = sum(int(sub[(sub.variant == v) & sub.identified].scenario.nunique() >= max(n_main, 1)) for v in alt) if n_main else 0
    warm = {dt: int(cd[(cd.dT == dt) & (cd.state == s) & (cd.provisional == dmn)].identified.sum()) for dt in (1.0, 2.0)}
    het = [float(dd[(dd.state == s) & (dd.scenario == sc)].identified.mean()) for sc in main_sc]
    if n_main == 3 and n_any == len(alt) and warm[1.0] == 3 and warm[2.0] == 3:
        tier = "Persistent"
    elif n_main > 0:
        tier = "Conditional"
    elif psel_max >= 0.5:
        tier = "Further investigation"
    else:
        continue
    rows_out.append(dict(state=s, domain=dmn, tier=tier, n_main=n_main, n_alt=n_any, n_alt_all=n_all,
                         warm1=warm[1.0], warm=warm[2.0], het_min=min(het) if het else float("nan"),
                         het_max=max(het) if het else float("nan"), psel=psel_max, psel_min=float(g.P_sel.min())))
ev = pd.DataFrame(rows_out)
ev.to_csv(os.path.join(R3, "evidence_levels.csv"), index=False)
order = {"Persistent": 0, "Conditional": 1, "Further investigation": 2}
ev = ev.sort_values(["tier", "domain", "psel"], key=lambda c: c.map(order) if c.name == "tier" else c,
                    ascending=[True, True, False])
rows = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{1.6cm}>{\\raggedright\\arraybackslash}p{2.9cm}ccccc@{}}",
        "\\toprule",
        "Evidence & State, domain & Main & \\multicolumn{2}{c}{Alt. specs} & $+2$~K & Het. \\\\",
        "\\cmidrule(lr){4-5}", " & & & all & $\\ge$1 & & $+2$~K \\\\", "\\midrule"]
last = None
for r in ev.itertuples():
    lab = {"Further investigation": "Further inv."}.get(r.tier, r.tier) if r.tier != last else ""
    if last is not None and r.tier != last:
        rows.append("\\midrule")
    het = "--" if r.n_main == 0 else (f"{r.het_min * 100:.0f}\\%" if abs(r.het_max - r.het_min) < 1e-9
                                      else f"{r.het_min * 100:.0f}--{r.het_max * 100:.0f}\\%")
    alls = f"{r.n_alt_all}" if r.n_main else "--"
    dom = LONG[r.domain].split()[0] if r.domain != "Adaptive" else "Socio-econ."
    rows.append(f"{lab} & {AB[r.state]}, {dom} & {r.n_main}/3 & {alls} & {r.n_alt} & {r.warm}/3 & {het} \\\\")
    last = r.tier
rows += ["\\bottomrule", "\\end{tabular}",
         "\\par\\smallskip\\raggedright\\scriptsize Main: scenarios identified in the main specification. Alt. specs: "
         f"number of the {len(alt)} alternative specifications (Tables~\\ref{{tab:spec_sensitivity}} "
         "and~\\ref{tab:energy_construct}) under which the case is identified in all or at least one of its scenarios. "
         "$+2$~K: scenarios identified under uniform warming. Het.: share of heterogeneous-warming realisations in which "
         "the case remains identified."]
w("tab_r3_evidence.tex", "\n".join(rows))

# ---- 4. Warming deltas ------------------------------------------------------------------------------
hm = pd.read_csv(os.path.join(R3, "climate_delta_hazard_means.csv"))
hd = hm.pivot(index="state", columns="dT", values="hot_days_mean")
c2 = cd[cd.scenario == "Adjacency Flow"].pivot(index="state", columns="dT", values="P_sel")
idf = cd[cd.scenario == "Adjacency Flow"].pivot(index="state", columns="dT", values="identified")
rows = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
        "State & \\multicolumn{3}{c}{Hot days (days/yr)} & \\multicolumn{3}{c}{$P^{\\mathrm{sel}}$, Adjacency Flow} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}", " & 1990--2024 & $+1$ K & $+2$ K & 1990--2024 & $+1$ K & $+2$ K \\\\",
        "\\midrule"]
for s in sorted(hd.index, key=lambda x: AB[x]):
    ps = [(f"\\textbf{{{c2.loc[s, dt]:.2f}}}" if idf.loc[s, dt] else f"{c2.loc[s, dt]:.2f}") for dt in (0.0, 1.0, 2.0)]
    rows.append(f"{AB[s]} & {hd.loc[s, 0.0]:.1f} & {hd.loc[s, 1.0]:.1f} & {hd.loc[s, 2.0]:.1f} & " + " & ".join(ps) + " \\\\")
rows += ["\\bottomrule", "\\end{tabular}",
         "\\par\\smallskip\\raggedright\\scriptsize Bold: identified relative weakness ($P^{\\mathrm{sel}}\\ge0.70$, $T>0$)."]
w("tab_r3_climate.tex", "\n".join(rows))
