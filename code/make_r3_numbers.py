#!/usr/bin/env python3
"""Numbers quoted in the Reviewer 3 revision (REACT only), written as LaTeX macros.

Usage: python make_r3_numbers.py <results_dir> <out .tex>
Reads <results_dir>/classification_audit.csv, robustness/robustness_specifications.csv and the
files in <results_dir>/r3/ written by react_r3_analyses.py and the climate-delta runs.
"""
import ast
import os
import sys

import numpy as np
import pandas as pd

RES, OUT = sys.argv[1], sys.argv[2]
R3 = os.path.join(RES, "r3")
AB = {"Baden-Württemberg": "BW", "Bayern": "BY", "Berlin": "BE", "Brandenburg": "BB", "Bremen": "HB",
      "Hamburg": "HH", "Hessen": "HE", "Mecklenburg-Vorpommern": "MV", "Niedersachsen": "NI",
      "Nordrhein-Westfalen": "NRW", "Rheinland-Pfalz": "RP", "Saarland": "SL", "Sachsen": "SN",
      "Sachsen-Anhalt": "ST", "Schleswig-Holstein": "SH", "Thüringen": "TH"}
DOM = {"Climate": "Climate", "Energy": "Energy", "Adaptive": "Socio-economic"}
M = {}


def put(k, v):
    M[k] = v if isinstance(v, str) else str(v)


def f2(x):
    return f"{x:.2f}".replace("-", "$-$") if x < 0 else f"{x:.2f}"


def m(x):                      # signed number for math mode
    return f"{x:.2f}"


# ---- main classification ---------------------------------------------------------------------
au = pd.read_csv(os.path.join(RES, "classification_audit.csv"))
put("NIdentMain", int(au.identified.sum()))
sv = pd.read_csv(os.path.join(R3, "r3_variants.csv"))
summ = pd.read_csv(os.path.join(R3, "r3_variant_summary.csv")).set_index("variant")
for v, tag in (("main", "Main"), ("no_import_penalty", "NoImp"), ("local_supply_only", "Local"),
               ("network_access_only", "Net"), ("absolute_capacities", "Abs")):
    put(f"rEA{tag}", m(summ.loc[v, "r_energy_socio"]))
    put(f"NId{tag}", int(summ.loc[v, "n_identified"]))
    x = sv[(sv.variant == v) & (sv.state == "Hamburg")]
    put(f"HHPselMin{tag}", f"{x.P_sel.min():.3f}"); put(f"HHPselMax{tag}", f"{x.P_sel.max():.3f}")
    put(f"HHNId{tag}", int(x.identified.sum()))
    put(f"RhoMin{tag}", f"{sv[sv.variant == v].rank_rho_vs_main.min():.2f}")
    bb = sv[(sv.variant == v) & sv.state.isin(["Bayern", "Baden-Württemberg"]) & sv.identified]
    put(f"SouthNId{tag}", int(len(bb)))
net = sv[sv.variant == "network_access_only"]
put("NetPselMax", f"{net.P_sel.max():.2f}")
put("NetHHTMax", m(float(net[net.state == "Hamburg"]["T"].max())))

# ---- Hamburg profile, correlations, leave-one-out prediction ----------------------------------------
hp = pd.read_csv(os.path.join(R3, "hamburg_energy_profile.csv")).set_index("indicator")
for n, tag in (("Wind_MW", "Wind"), ("Solar_MW", "PV"), ("R_ri", "Sr"), ("Thermal_MW", "Th"), ("WLINES", "Grid"),
               ("C_i", "Ci"), ("Storage_MWh", "Sto")):
    put(f"HHRank{tag}", int(hp.loc[n, "hamburg_rank_of_16"]))
put("HHWind", f"{hp.loc['Wind_MW', 'hamburg_value']:.2f}"); put("MedWind", f"{hp.loc['Wind_MW', 'panel_median']:.2f}")
put("HHPV", f"{hp.loc['Solar_MW', 'hamburg_value']:.2f}"); put("MedPV", f"{hp.loc['Solar_MW', 'panel_median']:.2f}")
ec = pd.read_csv(os.path.join(R3, "energy_indicator_vs_socio.csv")).set_index("indicator")
for n, tag in (("Wind_MW", "Wind"), ("Solar_MW", "PV"), ("R_ri", "Sr"), ("Thermal_MW", "Th"), ("WLINES", "Grid"),
               ("C_i", "Ci"), ("Storage_MWh", "Sto")):
    put(f"rSoc{tag}", m(ec.loc[n, "pearson_r"]))
lo = pd.read_csv(os.path.join(R3, "loo_domain_prediction.csv"))
h = lo[(lo.state == "Hamburg") & (lo.domain == "Energy")].iloc[0]
put("HHEobs", f"{h.observed:.2f}"); put("HHEpred", f"{h.predicted_loo:.2f}")
e = lo[lo.domain == "Energy"].set_index("state")
put("EobsMedian", f"{e.observed.median():.2f}")

# ---- configurations (lowest score, M0, M1, M2) --------------------------------------------------------
cmp = pd.read_csv(os.path.join(R3, "classification_across_configurations.csv"), index_col=0)
put("NMZero", int((cmp.M0_class != "Multidomain").sum())); put("NMOne", int((cmp.M1_class != "Multidomain").sum()))
for d, tag in (("Energy", "Energy"), ("Adaptive", "Adaptive"), ("Climate", "Climate")):
    put(f"NLowest{tag}", int((cmp.conventional_lowest_domain == d).sum()))
put("MOneStates", ", ".join(f"{AB[s]} ({DOM[cmp.loc[s, 'M1_provisional']]})" for s in cmp.index
                             if cmp.loc[s, "M1_class"] != "Multidomain"))

# ---- physical corridor check ------------------------------------------------------------------------
lt = pd.read_csv(os.path.join(R3, "lite_vs_physical_corridors.csv"))
mm = lt[(lt.physical_thermal_MVA > 0) & (lt.lite_Tij_MW > 0)]
rat = mm.lite_Tij_MW / mm.physical_thermal_MVA
put("AnineMedian", f"{rat.median() * 100:.0f}"); put("AnineMin", f"{rat.min() * 100:.0f}"); put("AnineMax", f"{rat.max() * 100:.0f}")
put("NoLineAdj", int(((lt.lite_Tij_MW > 0) & (lt.physical_thermal_MVA == 0)).sum()))
l = pd.read_csv(os.path.join(R3, "lite_adjacency_physical_corridors.csv")).set_index(["variant", "state"])
a9, ph = l.xs("Eq_A9"), l.xs("physical_N1")
put("LitePhysRank", f"{float((a9.expected_rank - ph.expected_rank).abs().max()):.2f}")
put("LitePhysHH", f"{ph.loc['Hamburg', 'P_sel']:.2f}"); put("LiteAnineHH", f"{a9.loc['Hamburg', 'P_sel']:.2f}")
put("LitePhysSameClass", int((a9.classification == ph.classification).sum()))
chg = [AB[s] for s in a9.index if a9.loc[s, "classification"] != ph.loc[s, "classification"]]
put("LitePhysChanged", ", ".join(chg) if chg else "none")

# ---- warming deltas ---------------------------------------------------------------------------------
cd = pd.read_csv(os.path.join(R3, "climate_delta_classifications.csv"))
b = cd[cd.dT == 0].set_index(["scenario", "state"])
for dt, tag in ((1.0, "One"), (2.0, "Two")):
    x = cd[cd.dT == dt].set_index(["scenario", "state"])
    put(f"CDPsel{tag}", f"{float((x.P_sel - b.P_sel).abs().max()):.3f}")
    put(f"CDSame{tag}", int(((x.identified == b.identified) & (x.provisional == b.provisional)).sum()))
    put(f"CDRank{tag}", f"{float((x.expected_rank - b.expected_rank).abs().max()):.2f}")
hm = pd.read_csv(os.path.join(R3, "climate_delta_hazard_means.csv"))
hd = hm.pivot(index="state", columns="dT", values="hot_days_mean")
for dt, tag in ((0.0, "Zero"), (1.0, "One"), (2.0, "Two")):
    put(f"Hot{tag}Min", f"{float(hd[dt].min()):.1f}"); put(f"Hot{tag}Max", f"{float(hd[dt].max()):.1f}")
d = pd.read_csv(os.path.join(R3, "climate_differential_classifications.csv"))
d["base_prov"] = [b.loc[(s, t), "provisional"] for s, t in zip(d.scenario, d.state)]
d["base_id"] = [bool(b.loc[(s, t), "identified"]) for s, t in zip(d.scenario, d.state)]
put("CDiffN", int(d.realisation.nunique()))
put("CDiffSameProv", f"{d.groupby('realisation').apply(lambda x: (x.provisional == x.base_prov).mean()).mean() * 100:.0f}")
ret = d[d.base_id].groupby(["state", "scenario"]).identified.mean()


def rr(s, sc):
    return f"{ret.loc[(s, sc)] * 100:.0f}"


def coh(s):
    return f"{min(ret.loc[(s, 'National Pool')], ret.loc[(s, 'Adjacency Flow')]) * 100:.0f}"


put("CDiffBYc", coh("Bayern")); put("CDiffBWc", coh("Baden-Württemberg")); put("CDiffBBc", coh("Brandenburg"))
put("CDiffBYb", rr("Bayern", "Baseline")); put("CDiffBWb", rr("Baden-Württemberg", "Baseline"))
if ("Sachsen", "National Pool") in ret.index:
    put("CDiffSNc", coh("Sachsen"))
if ("Hamburg", "National Pool") in ret.index:
    put("CDiffHH", rr("Hamburg", "National Pool"))
n = d.groupby("realisation").identified.sum()
put("CDiffIdMin", int(n.min())); put("CDiffIdMax", int(n.max())); put("CDiffIdMed", f"{n.median():.0f}")

# ---- specification counts used in the evidence levels ---------------------------------------------
sp = pd.read_csv(os.path.join(RES, "robustness", "robustness_specifications.csv")).rename(columns={"specification": "variant"})
allv = pd.concat([sp, sv[sv.variant != "main"]], ignore_index=True)
specs = [v for v in allv.variant.unique() if v not in ("main", "network_access_only")]
put("NAltSpecs", len(specs))

ld = pd.read_csv(os.path.join(R3, "load_density_mechanism.csv"), index_col=0)["value"]
put("LoadDensRatio", f"{ld.load_density_ratio:.0f}"); put("RenAreaRatio", f"{ld.ren_area_ratio:.1f}")
put("CorrKLDens", f"{ld.corr_logKL_neglogell:.2f}"); put("RhoEDens", m(ld.rho_E_ell)); put("RhoSDens", m(ld.rho_S_ell))
put("PartialES", m(ld.partial_r_ES_given_logell)); put("rESTerr", m(ld.r_ES_territorial))
cmp_nf = [AB[s] for s in cmp.index if cmp.loc[s, "M0_class"] == "Multidomain"]
put("MZeroNotFlagged", cmp_nf[0] if len(cmp_nf) == 1 else ", ".join(cmp_nf[:-1]) + ", and " + cmp_nf[-1])
put("NMZeroNot", len(cmp_nf))

with open(OUT, "w") as fh:
    fh.write("% generated by make_r3_numbers.py -- do not edit by hand\n")
    for k, v in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(M)} macros to {OUT}")
