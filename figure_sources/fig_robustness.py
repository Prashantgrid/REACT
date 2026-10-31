"""Robustness matrix of the relative-weakness classifications (fig06_robustness.pdf).
Data: data/robustness_matrix.csv, written by REACT_revision_summary.py (results/revision/).
Cell value = number of the three spatial scenarios in which the case is identified.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import style as S

S.apply()
df = pd.read_csv(S.DATA / "robustness_matrix.csv")
cases = ["BY Climate", "BW Climate", "BB Climate", "HH Energy", "SN Climate", "MV Socio-econ."]
hue = {"Climate": S.CLIM, "Energy": S.ENER, "Socio-econ.": S.SOCI}
col_lab = ["Bayern\nClimate", "Baden-W.\nClimate", "Brandenb.\nClimate", "Hamburg\nEnergy",
           "Sachsen\nClimate", "Meckl.-V.\nSocio-econ."]


def shade(base, k):
    """k in [0, 3] -> mix of white and the domain hue."""
    a = [0.0, 0.30, 0.62, 1.0][int(round(k))]
    b = np.array(to_rgb(base))
    return tuple(1 - a * (1 - b))


n = len(df)
fig_h = 0.155 * n + 0.75
fig, ax = plt.subplots(figsize=(S.FULL_W * 0.78, fig_h))
ax.set_xlim(-0.5, len(cases) + 1.25)
ax.set_ylim(n - 0.5, -1.25)
groups = df.group.tolist()
for i, r in df.iterrows():
    for j, c in enumerate(cases):
        v = r[c]
        if isinstance(v, str) and "-" in v:
            lo, hi = (int(x) for x in v.split("-"))
            k, txt = (lo + hi) / 2, v.replace("-", "–")
        else:
            k = int(float(v))
            txt = f"{k}"
        col = shade(hue[c.split(" ", 1)[1]], k)
        ax.add_patch(plt.Rectangle((j - 0.47, i - 0.42), 0.94, 0.84, color=col, lw=0))
        ax.text(j, i, txt, ha="center", va="center", fontsize=6.4,
                color="white" if k >= 2.5 else S.INK)
    rho = "" if pd.isna(r.rho_min) else (f"{r.rho_min:.2f}" if r.rho_min == r.rho_max else f"{r.rho_min:.2f}–{r.rho_max:.2f}")
    ax.text(len(cases) + 0.3, i, rho, ha="center", va="center", fontsize=6.2, color=S.MUTED)
# group separators and labels
prev = None
for i, g in enumerate(groups):
    if g != prev and i > 0:
        ax.axhline(i - 0.5, color=S.INK, lw=0.5)
    prev = g
ax.set_yticks(range(n))
ax.set_yticklabels(df.label, fontsize=6.6)
ax.set_xticks(list(range(len(cases))) + [len(cases) + 0.3])
ax.set_xticklabels(col_lab + [r"Rank $\rho$"], fontsize=6.4)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
# group names on the right margin
starts = {}
for i, g in enumerate(groups):
    starts.setdefault(g, [i, i])
    starts[g][1] = i
short = {"Numerical and panel": "Panel", "Construct": "Construct", "Hazard": "Hazard", "Weights": "Weights",
         "Support layer": "Support"}
for g, (a, b) in starts.items():
    if g == "Main":
        continue
    ax.plot([len(cases) + 0.98] * 2, [a - 0.35, b + 0.35], color=S.MUTED, lw=0.6, clip_on=False)
    ax.text(len(cases) + 1.12, (a + b) / 2, short[g], rotation=90, ha="center", va="center", fontsize=6.2,
            color=S.MUTED, clip_on=False)
S.save(fig, "fig06_robustness")
print("ok")
