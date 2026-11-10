"""Warming tests (fig_warming): hot days per state and selection frequencies of the four identified
cases under uniform warming of +1 and +2 K and under heterogeneous warming at +2 K.

Data: results/r3/climate_delta_hazard_means.csv, climate_delta_classifications.csv,
climate_differential_classifications.csv (written by code/react_r3_analyses.py).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

import style as S

S.apply()
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
R3 = Path(__file__).resolve().parents[1] / "results" / "r3"
hm = pd.read_csv(R3 / "climate_delta_hazard_means.csv")
uni = pd.read_csv(R3 / "climate_delta_classifications.csv")
het = pd.read_csv(R3 / "climate_differential_classifications.csv")


def mix(c, a):
    r, g, b = to_rgb(c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


fig = plt.figure(figsize=(S.FULL_W, 2.45))
gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1, 1, 1, 1], wspace=0.38, left=0.075, right=0.995,
                      bottom=0.15, top=0.80)

# (a) hot days
ax = fig.add_subplot(gs[0])
hd = hm.pivot(index="state", columns="dT", values="hot_days_mean")
hd = hd.sort_values(0.0)
y = np.arange(len(hd))
ax.hlines(y, hd[0.0], hd[2.0], color=S.GRID, lw=2.2, zorder=1)
ax.scatter(hd[0.0], y, s=12, facecolor="white", edgecolor=S.CLIM, lw=0.8, zorder=3, label="1990–2024")
ax.scatter(hd[1.0], y, s=9, color=mix(S.CLIM, 0.55), lw=0, zorder=3, label="+1 K")
ax.scatter(hd[2.0], y, s=12, color=S.CLIM, lw=0, zorder=3, label="+2 K")
ax.set_yticks(y); ax.set_yticklabels([S.ABBR[s] for s in hd.index], fontsize=6)
ax.set_ylim(-0.7, len(hd) - 0.3)
ax.set_xlabel("Hot days per year (mean)")
ax.set_title("(a) Hot days", loc="left", fontsize=7.5, fontweight="bold")
ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)

# (b-e) selection frequency of the identified cases
ev = pd.read_csv(R3 / "evidence_levels.csv")
ev = ev[ev.tier.isin(["Persistent", "Conditional"])].copy()
ev["o"] = [{"Bayern": 0, "Baden-Württemberg": 1}.get(s, 2) for s in ev.state]
ev = ev.sort_values(["tier", "o", "state"], ascending=[False, True, True])
cases = [(r.state, r.domain, r.tier.lower()) for r in ev.itertuples()][:4]
scen = ["Baseline", "National Pool", "Adjacency Flow"]
short = {"Baseline": "Base", "National Pool": "NP", "Adjacency Flow": "AF"}
dom_col = {"Climate": S.CLIM, "Energy": S.ENER}
rng = np.random.default_rng(1)
for k, (st, dom, tier) in enumerate(cases):
    a = fig.add_subplot(gs[k + 1])
    c = dom_col[dom]
    a.axhline(0.70, color=S.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for i, sc in enumerate(scen):
        u = uni[(uni.state == st) & (uni.scenario == sc)].set_index("dT").P_sel
        a.scatter([i - 0.22], [u[0.0]], s=16, facecolor="white", edgecolor=c, lw=0.8, zorder=3)
        a.scatter([i - 0.07], [u[1.0]], s=12, color=mix(c, 0.55), lw=0, zorder=3)
        a.scatter([i + 0.08], [u[2.0]], s=16, color=c, lw=0, zorder=3)
        h = het[(het.state == st) & (het.scenario == sc)]
        p = np.where(h.provisional == dom, h.P_sel, np.nan)
        a.scatter(i + 0.27 + rng.uniform(-0.05, 0.05, len(p)), p, s=5, color=mix(c, 0.45), lw=0, zorder=2)
    a.set_xticks(range(3)); a.set_xticklabels([short[x] for x in scen])
    a.set_xlim(-0.5, 2.55); a.set_ylim(0.50, 0.92)
    if k == 0:
        a.set_ylabel("Selection frequency $P^{\\mathrm{sel}}$")
        a.text(2.52, 0.705, "0.70", fontsize=6, ha="right", va="bottom")
    else:
        a.set_yticklabels([])
    lab = "Climate" if dom == "Climate" else "Energy"
    a.set_title(f"({'bcde'[k]}) {S.ABBR[st]}, {lab}", loc="left", fontsize=7.5, fontweight="bold")
    a.text(0.02, 0.97, tier, transform=a.transAxes, fontsize=6, va="top", color=S.MUTED, style="italic")

handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK, mew=0.8, ms=4, label="1990–2024"),
           Line2D([], [], marker="o", ls="", mfc=mix(S.INK, 0.45), mec="none", ms=3.6, label="+1 K uniform"),
           Line2D([], [], marker="o", ls="", mfc=S.INK, mec="none", ms=4, label="+2 K uniform"),
           Line2D([], [], marker="o", ls="", mfc=mix(S.INK, 0.35), mec="none", ms=2.6,
                  label="+2 K heterogeneous (12 realisations)"),
           Line2D([], [], color=S.INK, lw=0.6, ls=(0, (3, 2)), label="identification threshold")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.29, 1.0), ncol=5, fontsize=6.5,
           handletextpad=0.3, columnspacing=1.0, frameon=False)
S.save(fig, "fig_warming")
print("wrote", S.OUT / "fig_warming.pdf")
