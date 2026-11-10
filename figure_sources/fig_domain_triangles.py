"""Relative domain profiles (National Pool): median standardised residual of each domain per state.
Reads ../results/results_national_pool.csv. Writes out/fig_domain_triangles.pdf.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as S

S.apply()
r = pd.read_csv(S.HERE.parent / "results" / "results_national_pool.csv", index_col=0)
DOM = ["Climate", "Energy", "Adaptive"]
COL = {"Climate": S.CLIM, "Energy": S.ENER, "Adaptive": S.SOCI}
ANG = np.deg2rad([90, 330, 210])                 # Climate top, Energy right, Socio-economic left
ZMIN, ZMAX = -2.0, 2.0
ORDER = ["BY", "BW", "BB", "SN", "HH", "BE", "NRW", "HE",
         "RP", "MV", "TH", "ST", "SL", "NI", "HB", "SH"]


def radius(z):
    return (np.clip(z, ZMIN, ZMAX) - ZMIN) / (ZMAX - ZMIN)     # -2 -> centre, 0 -> 0.5, +2 -> 1


def xy(v):
    v = np.asarray(v, float)
    return v * np.cos(ANG), v * np.sin(ANG)


fig, axes = plt.subplots(2, 8, figsize=(S.FULL_W, 2.3))
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.05, hspace=0.38)
for ax, ab in zip(axes.flat, ORDER):
    row = r.loc[S.STATES[ab]]
    z = np.array([row[f"Median_Z_{d}"] for d in DOM])
    x, y = xy([1.0] * 3)
    ax.fill(x, y, facecolor="none", edgecolor="#C9CED6", lw=0.5, zorder=1)
    x, y = xy([0.5] * 3)
    ax.fill(x, y, facecolor="#F3F4F6", edgecolor=S.MUTED, lw=0.7, ls=(0, (2, 1.5)), zorder=2)
    for a in ANG:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=S.GRID, lw=0.5, zorder=1)
    xo, yo = xy(radius(z))
    dm = row["Provisional_domain"]
    ident = bool(row["identified"])
    adverse = dm in DOM and row[f"Median_Z_{dm}"] < 0          # provisional domain below expectation
    c = COL[dm] if adverse else "#9AA5B1"
    ax.fill(xo, yo, facecolor=c, alpha=0.55 if ident else 0.20, edgecolor="none", zorder=3)
    ax.fill(xo, yo, facecolor="none", edgecolor=c, lw=1.0 if ident else 0.8, zorder=3.5)
    if dm in DOM:
        k = DOM.index(dm)
        if bool(row["identified"]):
            ax.plot(xo[k], yo[k], "o", ms=4.6, mfc=COL[dm], mec="white", mew=0.6, zorder=6)
        elif adverse and row["P_sel"] >= 0.5:
            ax.plot(xo[k], yo[k], "o", ms=3.6, mfc="white", mec=COL[dm], mew=0.9, zorder=6)
    ax.set_title(f"{ab}  {row['P_sel']:.2f}", fontsize=7, pad=4,
                 fontweight="bold" if bool(row["identified"]) else "normal")
    for a, t in zip(ANG, ("C", "E", "S")):
        dx, dy = {"C": (0.13, 0.0), "E": (0.06, -0.09), "S": (-0.06, -0.09)}[t]
        ax.text(np.cos(a) + dx, np.sin(a) + dy, t, ha="center", va="center", fontsize=5.6, color=S.MUTED)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.62, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
handles = [Patch(facecolor=S.CLIM, alpha=0.55, edgecolor=S.CLIM, label="Climate Exposure"),
           Patch(facecolor=S.ENER, alpha=0.55, edgecolor=S.ENER, label="Energy System"),
           Patch(facecolor=S.SOCI, alpha=0.55, edgecolor=S.SOCI, label="Socio-economic Capacity"),
           Patch(facecolor="#9AA5B1", alpha=0.20, edgecolor="#9AA5B1", label="No adverse domain"),
           Line2D([], [], color=S.MUTED, lw=0.7, ls=(0, (2, 1.5)), label="Expected level"),
           Line2D([], [], marker="o", ls="", mfc=S.INK, mec="white", ms=5, label="Identified"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK, mew=0.9, ms=4.5, label="Provisional")]
fig.legend(handles=handles, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.0), fontsize=6.3,
           handlelength=1.4, columnspacing=0.9)
S.save(fig, "fig_domain_triangles")
