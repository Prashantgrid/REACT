#!/usr/bin/env python3
"""Update numerical manuscript tables from one completed run; prose needs author review."""
import argparse
from pathlib import Path
import re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("results", type=Path)
ap.add_argument("manuscript", type=Path)
a = ap.parse_args()
text = a.manuscript.read_text()
rv, rb = a.results / "revision", a.results / "robustness"
audit = pd.read_csv(a.results / "classification_audit.csv")
names = {"Baden-Württemberg": r'Baden-W\"urttemberg', "Thüringen": r'Th\"uringen'}
abbr = {"Bayern": "BY", "Baden-Württemberg": "BW", "Brandenburg": "BB", "Sachsen": "SN",
        "Hamburg": "HH", "Mecklenburg-Vorpommern": "MV", "Nordrhein-Westfalen": "NRW",
        "Hessen": "HE", "Berlin": "BE", "Rheinland-Pfalz": "RP", "Thüringen": "TH",
        "Sachsen-Anhalt": "ST", "Saarland": "SL", "Niedersachsen": "NI", "Bremen": "HB",
        "Schleswig-Holstein": "SH"}
dom = {"Climate": "Climate Exposure", "Energy": "Energy System", "Adaptive": "Socio-economic"}
cases = [("Bayern", "Climate"), ("Baden-Württemberg", "Climate"), ("Brandenburg", "Climate"),
         ("Hamburg", "Energy"), ("Sachsen", "Climate")]

def rows(label, lines):
    global text
    pos = text.index(r"\label{" + label + "}")
    start = text.index(r"\midrule", pos) + len(r"\midrule")
    stop = text.index(r"\bottomrule", start)
    text = text[:start] + "\n" + "\n".join(lines) + "\n" + text[stop:]

def interval(values, digits=2):
    lo, hi = values.min(), values.max()
    return f"{lo:.{digits}f}" if round(lo, digits) == round(hi, digits) else f"{lo:.{digits}f}--{hi:.{digits}f}"

def rho(frame):
    return interval(frame.rank_rho_vs_main)

def z_interval(values):
    return f"${values.min():+.2f}$ to ${values.max():+.2f}$"

def case_cell(frame, state, domain):
    g = frame[(frame.state == state) & (frame.provisional == domain)]
    if g.empty:
        return "--"
    return interval(g.P_sel) + f" ({int(g.identified.sum())})"

def spec_line(label, frame, selected=cases, rank=True):
    return " & ".join([label] + [case_cell(frame, *case) for case in selected]
                      + ([rho(frame)] if rank else [])) + r" \\"

found = []
reading = {"Bayern": r"Core southern result; see Fig.~\ref{fig:robustness}",
           "Baden-Württemberg": r"Core southern result; see Fig.~\ref{fig:robustness}",
           "Brandenburg": "Conditional on scenario and indicator specification",
           "Sachsen": "Near threshold; sensitive to input and model specification"}
for state in ("Bayern", "Baden-Württemberg", "Brandenburg", "Sachsen"):
    g = audit[(audit.State == state) & audit.identified]
    if g.empty:
        continue
    scenarios = "All three" if len(g) == 3 else "; ".join(g.Scenario)
    found.append(" & ".join([names.get(state, state), dom[g.Provisional_domain.iloc[0]], scenarios,
                             interval(g.P_sel), z_interval(g.Median_Z_candidate),
                             reading[state]]) + r" \\")
rows("tab:binding_constraints", found)

m = pd.read_csv(rv / "m012_classification.csv").set_index("state")
lines = []
columns = ["M0", "M1", "M2_baseline", "M2_national_pool", "M2_adjacency_flow"]
short = {"Climate": "C", "Energy": "E", "Adaptive": "S"}
for state in ("Bayern", "Baden-Württemberg", "Brandenburg", "Sachsen", "Hamburg"):
    r = m.loc[state]
    cells = [names.get(state, state)] + [dom[r[c]].replace("Climate Exposure", "Climate").replace("Energy System", "Energy")
                                       for c in ("obstacle_degree_main", "lowest_domain_score")]
    for c in columns:
        cell = f"{short[r[c+'_provisional']]} {r[c+'_Psel']:.3f}"
        cells.append(r"\textbf{" + cell + "}" if r[c+"_identified"] else cell)
    lines.append(" & ".join(cells) + r" \\")
lines += [r"\midrule", "Identified cases & -- & -- & " + " & ".join(str(int(m[c+"_identified"].sum())) for c in columns) + r" \\"]
rows("tab:m012_diagnosis", lines)

groups = [("South", ["Bayern", "Baden-Württemberg"], "Climate Exposure", "Core southern climate signal"),
          ("East", ["Brandenburg"], "Climate Exposure", "Scenario-dependent heat and dryness signal"),
          ("", ["Sachsen"], "Climate Exposure", "Climate signal near the threshold"),
          ("Load centres", ["Hamburg"], "Energy System", "Energy direction below threshold"),
          ("", ["Berlin", "Nordrhein-Westfalen", "Hessen", "Rheinland-Pfalz"], "Energy System", "Shared Energy direction"),
          ("Socio-economic", ["Mecklenburg-Vorpommern"], "Socio-economic", "Strongest socio-economic direction"),
          ("", ["Thüringen", "Sachsen-Anhalt", "Saarland"], "Socio-economic", "Shared direction; compound in Sachsen-Anhalt"),
          ("Balanced", ["Niedersachsen", "Bremen", "Schleswig-Holstein"], r"Mixed (not adverse)\tnote{a}", "Balanced profiles")]
lines = []
for group, states, domain, explanation in groups:
    g = audit[audit.State.isin(states)]
    state_text = ", ".join(names.get(s, s) for s in states)
    lines.append(" & ".join([group, state_text, domain, interval(g.P_sel),
                             z_interval(g.Median_Z_candidate), f"{int(g.identified.sum())} of {len(g)}", explanation]) + r" \\")
rows("tab:typology", lines)

support = pd.read_csv(rv / "support_sensitivity.csv")
def hh_cell(key):
    frame = support[(support.specification == key) & (support.state == "Hamburg")]
    return interval(frame.P_sel)

support_rows = []
for label, main, tested, keys in [
    (r"Credits $(c_W,c_{PV},c_{TH})$", "0.10, 0.05, 0.85", r"low\tnote{a}; high\tnote{a}", ["credits_low", "credits_high"]),
    (r"Storage window $h_{\mathrm{dis}}$", "4 h", "2 h; 8 h", ["storage_window_2h", "storage_window_8h"]),
    (r"Transfer scaling $\phi$", "0.25", "0.10; 0.50", ["transfer_phi_0.10", "transfer_phi_0.50"]),
    ("AF access penalty", "0.70", "0.50; 0.90", ["af_access_penalty_0.50", "af_access_penalty_0.90"]),
    ("AF grid penalty", "0.30", "0.15; 0.45", ["af_grid_penalty_0.15", "af_grid_penalty_0.45"]),
    ("NP penalty cap", "0.50", "0.30; 0.70", ["np_cap_0.30", "np_cap_0.70"]),
    ("All eight, sampled jointly", "--", r"ranges above\tnote{b}", ["joint_structural_uncertainty"])]:
    support_rows.append(" & ".join([label, main, tested, "; ".join(hh_cell(key) for key in keys)]) + r" \\")
fair = pd.read_csv(rv / "support_extra_sensitivity.csv")
g = fair[(fair.specification == "flow_allocation_fair") & (fair.state == "Hamburg")]
support_rows.append("Flow allocation (AF) & max-flow & max-min fair & " + interval(g.P_sel) + r" \\")
rows("tab:support_sensitivity", support_rows)

mc = pd.read_csv(rb / "robustness_seed_summary.csv")
rows("tab:mc_replicas", [f"{r.scenario} & {r.sd_max_expected_rank:.3f} & {r.sd_max_Psel:.4f} & {r.min_rank_rho:.3f} & {r.same_provisional}/16 " + r"\\" for r in mc.itertuples()])
red = pd.read_csv(rb / "robustness_indicator_reduction.csv")
red_rows = []
for r in red.itertuples():
    identified = ", ".join(abbr[v.split(":")[0]] for v in r.identified.split("; ")) if isinstance(r.identified, str) else "--"
    red_rows.append(f"{r.tau:.1f} & {r.n_kept} & {r.rho:.3f} & {r.max_shift:.0f} & {r.mean_shift:.2f} & {identified} " + r"\\")
rows("tab:multicollinearity", red_rows)

lines = []
for threshold in (0.60, 0.65, 0.70, 0.75, 0.80):
    g = audit[(audit.P_sel >= threshold) & (audit.T_joint > 0)]
    counts = g.groupby("State").size().to_dict()
    state_text = ", ".join(f"{abbr[s]} ({counts[s]})" for s in abbr if s in counts)
    cells = [f"{threshold:.2f}", str(len(g)), state_text]
    if threshold == .70:
        cells = [r"\textbf{" + c + "}" for c in cells]
    lines.append(" & ".join(cells) + r" \\")
rows("tab:threshold", lines)

spec = pd.read_csv(rb / "robustness_specifications.csv")
overlap = pd.read_csv(rv / "overlap_sensitivity.csv")
labels = [("main", "Main specification"), ("grid_per_demand", "Grid per average load"),
          ("grid_geometric_mean", "Grid per geometric mean"), ("absolute_capacities", r"Absolute capacities\tnote{a}"),
          ("equal_weights", r"Equal weights\tnote{b}"), ("low_hazard_response", "Low hazard response"),
          ("high_hazard_response", "High hazard response"), ("thermal_as_cost", "Thermal capacity as cost")]
lines = [spec_line(label, spec[spec.specification == key]) for key, label in labels]
lines.append(spec_line("Thermal capacity excluded", overlap[overlap.specification == "thermal_excluded"]))
rows("tab:spec_sensitivity", lines)

lines = [spec_line("Main specification", spec[spec.specification == "main"])]
for filename, mapping in [
    ("overlap_sensitivity.csv", [("thermal_excluded", "Thermal capacity excluded")]),
    ("socio_sensitivity.csv", [("socio_2022_only (GDP, HDI)", "Socio-economic: 2022 indicators only"),
                              ("socio_without_innovation_2019", "Socio-economic: without innovation (2019)"),
                              ("socio_without_age_dependency_2025", "Socio-economic: without age dependency (2025)")]),
    ("storage_correction.csv", [("storage_registry_raw", "Storage: raw registry 1 Oct 2025"),
                                ("storage_submitted_panel", "Storage: original submitted panel"),
                                ("storage_legacy_interim", "Storage: historical interim panel"),
                                ("storage_snapshot_excl_flags", "Storage: screened registry 31 Oct 2026"),
                                ("storage_snapshot_raw", "Storage: raw registry 31 Oct 2026")]),
    ("overlap_sensitivity.csv", [("no_derating (hazards in Climate Exposure only)", "No derating"),
                                ("climate_without_hot_days", "Climate Exposure without hot days"),
                                ("climate_without_low_wind", "Climate Exposure without low-wind days"),
                                ("climate_without_HD_WSM_LWD (continuous derating drivers via Energy only)", "Climate Exposure without HD, WSM, LWD")]),
    ("climate_shift_sensitivity.csv", [("trend_adjusted_2024", "Hazards trend-adjusted to 2024"),
                                      ("trend_extrapolated_2050", "Hazards trend-extrapolated to 2050"),
                                      ("trend_2050_heat_moisture_precip_only", "2050 trends, wind hazards unchanged"),
                                      ("uniform_intensification", r"Uniform hazard intensification\tnote{a}")]),
    ("weights_sensitivity.csv", [(f"alpha_{k}", r"Dirichlet $\alpha=" + str(k) + "$" ) for k in (.5, 2, 5, 20)])]:
    frame = pd.read_csv(rv / filename)
    for key, label in mapping:
        lines.append(spec_line(label, frame[frame.specification == key]))
bt = pd.read_csv(rv / "bootstrap_full_model.csv")
cells = []
for state, domain in cases:
    counts = bt[(bt.state == state) & (bt.provisional == domain)].groupby("rep").identified.sum().reindex(range(40), fill_value=0)
    cells.append(f"{int((counts>=2).sum())}/40 ($\\ge$2)")
lines.append(" & ".join([r"Year-block bootstrap (40 samples)\tnote{b}"] + cells + [rho(bt)]) + r" \\")
emph = pd.read_csv(rv / "weights_sensitivity.csv")
emph = emph[emph.specification.str.startswith("emphasis")]
cell_values = []
for state, domain in cases:
    per = [g[(g.state == state) & (g.provisional == domain)] for _, g in emph.groupby("specification")]
    values = pd.concat(per).P_sel
    counts = [int(g.identified.sum()) for g in per]
    count = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}--{max(counts)}"
    cell_values.append(interval(values) + f" ({count})")
lines.append(" & ".join([r"Domain emphasis (3 distributions)\tnote{c}"] + cell_values + [rho(emph)]) + r" \\")
rows("tab:extra_checks", lines)

a.manuscript.write_text(text)
print("Updated numerical tables from", a.results)
