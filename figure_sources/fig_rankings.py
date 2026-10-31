"""Aggregate-ranking figures (fig03_M0_M1_M2.pdf and fig04_baseline.pdf).
Data: data/m012.json and data/rank_accept.json, written by
REACT_ESR_reproducible.py (results/figure_data/).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style as S

S.apply()

# ------------------------------------------------------------------ M0 / M1 / M2 dot plot
v = S.load("m012.json")
order = list(v.keys())                      # already ordered by mean M2 vulnerability
y = np.arange(len(order))[::-1]

fig, ax = plt.subplots(figsize=(S.COL_W - 0.36, 3.3))
for yi, s in zip(y, order):
    m0, m1, m2 = v[s]["M0"], v[s]["M1"], v[s]["M2"]
    lo, hi = min(m0, m1, m2), max(m0, m1, m2)
    ax.plot([lo, hi], [yi, yi], color="#C9CED6", lw=2.2, solid_capstyle="round", zorder=1)
    ax.scatter(m0, yi, s=20, facecolors="white", edgecolors="#6B7280", linewidths=0.9, zorder=3)
    ax.scatter(m1, yi, s=16, marker="D", facecolors="#9CA3AF", edgecolors="none", zorder=3)
    ax.scatter(m2, yi, s=24, facecolors=S.ENER, edgecolors="white", linewidths=0.5, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([S.STATES[s] for s in order])
allv = [v[s][k] for s in order for k in ("M0", "M1", "M2")]
ax.set_xlim(0.01 * np.floor(100 * min(allv) - 2), 0.01 * np.ceil(100 * max(allv) + 2))
ax.set_ylim(-0.7, len(order) - 0.3)
ax.set_xlabel(r"Vulnerability $1-CC$ (lower is more favourable)")
ax.xaxis.grid(True, color=S.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
h = [Line2D([], [], marker="o", ls="", ms=4.6, mfc="white", mec="#6B7280", mew=0.9,
            label="M0 deterministic TOPSIS"),
     Line2D([], [], marker="D", ls="", ms=3.8, mfc="#9CA3AF", mec="none", label="M1 weight uncertainty"),
     Line2D([], [], marker="o", ls="", ms=5.2, mfc=S.ENER, mec="white", label="M2 full REACT")]
ax.legend(handles=h, loc="upper right", fontsize=6.6, handletextpad=0.2, borderaxespad=0.2,
          labelspacing=0.3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
S.save(fig, "fig03_M0_M1_M2")

# ------------------------------------------------------------------ rank acceptability
ra = S.load("rank_accept.json")
order = list(ra.keys())                     # ordered by expected Baseline rank
y = np.arange(len(order))[::-1]
ramp = ["#0B3C6E", "#1F6FB2", "#6FA8DC", "#C4DCF0"]   # rank 1 (dark) -> rank 4 (light)
fig, ax = plt.subplots(figsize=(S.COL_W - 0.36, 3.3))
for yi, s in zip(y, order):
    left = 0.0
    for k, p in enumerate(ra[s]):
        ax.barh(yi, p, left=left, height=0.72, color=ramp[k], edgecolor="white", linewidth=0.5)
        left += p
    ax.text(left + 0.8, yi, f"{left:.0f}%", va="center", fontsize=6.3, color=S.MUTED)
ax.set_yticks(y)
ax.set_yticklabels([S.STATES[s] for s in order])
ax.set_xlim(0, 5 * np.ceil(max(sum(ra[s]) for s in order) / 5) + 8)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel("Probability of occupying ranks 1–4 (%)")
ax.xaxis.grid(True, color=S.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ramp]
ax.legend(h, ["Rank 1", "Rank 2", "Rank 3", "Rank 4"], loc="lower right", ncol=1, fontsize=6.6,
          handlelength=1.1, handleheight=0.9, handletextpad=0.35, labelspacing=0.3, borderaxespad=0.3,
          frameon=True, facecolor="white", edgecolor="none")
S.save(fig, "fig04_baseline")
print("ok")
