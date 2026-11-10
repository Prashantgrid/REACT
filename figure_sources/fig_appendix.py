"""Appendix figures: hazard dependence diagnostics, external-support score function, r_i vs C_i.
Replace fig02_correlations.pdf, Resilience_Function.pdf, fig11_ri_vs_Ci.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import style as S

S.apply()

# ------------------------------------------------------------------ correlation diagnostics
M = np.array(S.load("corr.json"))
for k in range(4):                           # fill cells hidden by overlapping labels, by symmetry
    M[k] = np.where(np.isnan(M[k]), M[k].T, M[k])
labels = ["HD", "WSM", "P30", r"Rx1$_{\mu}$", r"Rx1$_{\max}$", "SD", "LWD"]
titles = ["(a) Pooled rank", "(b) Pooled linear", "(c) State-wise PIT (copula input)", "(d) Fitted copula"]
cmap = LinearSegmentedColormap.from_list(
    "div", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"])
fig, axes = plt.subplots(1, 4, figsize=(S.FULL_W, 2.05))
fig.subplots_adjust(left=0.05, right=0.915, top=0.9, bottom=0.17, wspace=0.08)
n = 7
for k, ax in enumerate(axes):
    A = M[k].copy()
    mask = np.triu(np.ones_like(A, bool), k=0)      # show strict lower triangle only
    Am = np.ma.array(A, mask=mask)
    cm = cmap.copy(); cm.set_bad((1, 1, 1, 0))
    im = ax.imshow(Am, cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
    for i in range(n):
        for j in range(i):
            val = A[i, j]
            ax.text(j, i, f"{val:.2f}".replace("-0.00", "0.00").replace("-", "−"), ha="center",
                    va="center", fontsize=5.6, color="white" if abs(val) > 0.6 else S.INK)
    ax.set_xticks(range(n - 1))
    ax.set_xticklabels(labels[:-1], rotation=90, fontsize=6.3)
    ax.set_yticks(range(1, n))
    ax.set_yticklabels(labels[1:] if k == 0 else [], fontsize=6.3)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(titles[k], fontsize=7.2, pad=3, loc="left")
cax = fig.add_axes([0.93, 0.22, 0.012, 0.62])
cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=6.3, length=2)
cb.outline.set_visible(False)
cb.set_label("Correlation", fontsize=7)
S.save(fig, "fig02_correlations")

# ------------------------------------------------------------------ resilience value function
r = np.linspace(-1, 1, 801)


def R(x):
    return np.select([x <= -0.5, x < -0.1, np.abs(x) <= 0.1, x <= 0.3],
                     [0.8, 1.0 + 0.5 * (x + 0.1), 1.0, 1.0 - 4 * (x - 0.1)], 0.2)


ric = S.load("ri_ci.json")
fig, ax = plt.subplots(figsize=(S.COL_W + 0.32, 2.3))
for a, b, c in [(-1, -0.1, "#F3F0F9"), (-0.1, 0.1, "#F4F4F4"), (0.1, 1, "#FDF3E1")]:
    ax.axvspan(a, b, color=c, lw=0, zorder=0)
ax.plot(r, R(r), color=S.INK, lw=1.5, zorder=3)
for xk in (-0.5, -0.1, 0.1, 0.3):
    ax.axvline(xk, color="#B8BEC7", lw=0.5, ls=(0, (2, 2)), zorder=1)
ax.text(-0.55, 1.07, "Net exporters\nslope 0.5, floor 0.8", ha="center", va="bottom", fontsize=6.4, color=S.EXPORTER)
ax.text(0.0, 0.55, "Balanced", ha="center", va="center", rotation=90, fontsize=6.4, color=S.MUTED)
ax.text(0.55, 1.07, "Net importers\nslope 4, floor 0.2", ha="center", va="bottom", fontsize=6.4, color="#9A6700")
# states as rug
for ab, (ri, ci) in ric.items():
    col = S.EXPORTER if ri < -0.1 else (S.IMPORTER if ri > 0.1 else S.BALANCED)
    ax.plot([ri, ri], [-0.02, 0.06], color=col, lw=1.2, solid_capstyle="butt", zorder=4)
ax.text(-0.98, 0.1, "States (rug)", fontsize=6.0, color=S.MUTED, va="bottom")
ax.set_xlim(-1.02, 1.02)
ax.set_ylim(-0.02, 1.32)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel(r"Relative net-import indicator $r_i=N_i/\max_j|N_j|$ (imports positive)")
ax.set_ylabel(r"External-support score $S(r_i)$")
ax.spines["left"].set_bounds(0, 1.0)
S.save(fig, "Resilience_Function")

# ------------------------------------------------------------------ r_i vs C_i
fig, ax = plt.subplots(figsize=(S.COL_W + 0.38, 2.6))
ax.axvspan(-0.1, 0.1, color="#F1F1F1", lw=0, zorder=0)
ax.axvline(0, color="#B8BEC7", lw=0.5, zorder=0)
OFF = {"TH": (5, -1), "BB": (5, -1), "HE": (5, 0), "RP": (5, 0), "NI": (5, 0), "SN": (5, 0),
       "ST": (-5, 0), "HH": (5, 0), "SH": (-5, 0), "MV": (5, 0), "HB": (5, 0), "NRW": (-5, 0),
       "BY": (5, 0), "SL": (5, 0), "BE": (5, 0), "BW": (-5, 0)}
for ab, (ri, ci) in ric.items():
    col = S.EXPORTER if ri < -0.1 else (S.IMPORTER if ri > 0.1 else S.BALANCED)
    ax.scatter(ri, ci, s=34, c=col, edgecolors=S.INK, linewidths=0.6, zorder=3)
    dx, dy = OFF[ab]
    ax.annotate(ab, (ri, ci), xytext=(dx, dy), textcoords="offset points", fontsize=6.6,
                ha="left" if dx > 0 else "right", va="center", color=S.INK)
ax.set_xlim(-1.08, 1.08)
ax.set_ylim(0.2, 1.0)
ax.set_xlabel(r"Relative net-import indicator $r_i$ (imports positive)")
ax.set_ylabel(r"Composite interstate connectivity $C_i$")
ax.grid(True, color=S.GRID, lw=0.5)
ax.set_axisbelow(True)
h = [Line2D([], [], marker="o", ls="", ms=5, mfc=S.EXPORTER, mec=S.INK, mew=0.6, label="Net exporter"),
     Line2D([], [], marker="o", ls="", ms=5, mfc=S.BALANCED, mec=S.INK, mew=0.6, label=r"Balanced ($|r_i|\leq0.1$)"),
     Line2D([], [], marker="o", ls="", ms=5, mfc=S.IMPORTER, mec=S.INK, mew=0.6, label="Net importer")]
ax.legend(handles=h, loc="lower left", fontsize=6.4, handletextpad=0.2, labelspacing=0.3,
          borderaxespad=0.3, frameon=True, facecolor="white", edgecolor="none")
S.save(fig, "fig11_ri_vs_Ci")
print("ok")
