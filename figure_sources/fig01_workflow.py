"""Fig. 1 -- REACT workflow with panels drawn from the model and its results (fig01_workflow).

Step 1: direction-adjusted indicator matrix (results/structural_indicator_panel.csv).
Step 2: Student-t copula draws against the 1990-2024 record (REACT_v4 hazard panel), the capped
        derating functions (REACT_v4.PARAMS_DERATING), and the three support scenarios on the
        state adjacency network (REACT_v4.ADJACENCY).
Step 3: domain-weight draws from Dir(1,1,1) with the 0.01 floor (REACT_v4.sample_dirichlet_floor).
Step 4: selection shares in the National Pool scenario (results/results_national_pool.csv).
Step 5: evidence levels (results/r3/evidence_levels.csv) on Natural Earth state boundaries.
Usage: python fig01_workflow.py   (writes out/fig01_workflow.pdf/.png)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

import style as S

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import REACT_v4 as R  # noqa: E402

S.apply()
plt.rcParams.update({"font.size": 6, "axes.labelsize": 5.6, "xtick.labelsize": 5.2,
                     "ytick.labelsize": 5.2, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 1.8, "ytick.major.size": 1.8, "xtick.major.pad": 1.2,
                     "ytick.major.pad": 1.2, "axes.labelpad": 1.5})
NAVY = "#1F3A5F"
PANEL = "#F5F7FA"
EDGE = "#C9D2DC"
DOM = {"Climate": S.CLIM, "Energy": S.ENER, "Adaptive": S.SOCI}
RES = ROOT / "results"

W, H = S.FULL_W, 3.30
fig = plt.figure(figsize=(W, H))


def ax_in(x, y, w, h, **kw):
    """Axes placed in inches from the lower-left corner of the figure."""
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def mix(c, a):
    r, g, b = to_rgb(c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


# ---------------------------------------------------------------------------------------------- frame
cols = [(0.02, 1.42), (1.50, 1.62), (3.18, 1.16), (4.40, 1.44), (5.90, 1.24)]
titles = ["Indicator system\nand data", "Hazard simulation and\nimpact modelling", "Uncertainty\npropagation",
          "Aggregate comparison and\nrelative-weakness diagnostic", "German application"]
top, bot = 3.27, 0.27
bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off(); bg.set_xlim(0, W); bg.set_ylim(0, H)
for k, ((x0, w), t) in enumerate(zip(cols, titles)):
    app = k == 4
    bg.add_patch(FancyBboxPatch((x0, bot), w, top - bot, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=PANEL, ec=S.SOCI if app else NAVY, lw=0.7))
    bg.add_patch(FancyBboxPatch((x0, top - 0.36), w, 0.36, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=S.SOCI if app else NAVY, ec="none"))
    bg.add_patch(Rectangle((x0, top - 0.36), w, 0.06, fc=S.SOCI if app else NAVY, ec="none"))
    bg.text(x0 + 0.15, top - 0.18, str(k + 1), ha="center", va="center", color=NAVY if not app else S.SOCI,
            fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="none"))
    bg.text(x0 + 0.30, top - 0.18, t, ha="left", va="center", color="white", fontsize=6.6,
            fontweight="bold", linespacing=1.05)
    if k < 4:
        xa = x0 + w + 0.005
        bg.annotate("", xy=(cols[k + 1][0] - 0.005, 1.95), xytext=(xa, 1.95),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.25", color=NAVY, lw=0.9))
# brackets
for (xa, xb, lab, c) in ((0.02, 5.84, "REACT method (Section 2): Steps 1–4 transfer to other regions", NAVY),
                         (5.90, 7.14, "Sections 3–4", S.SOCI)):
    bg.plot([xa + 0.03, xa + 0.03, xb - 0.03, xb - 0.03], [0.20, 0.13, 0.13, 0.20], color=c, lw=0.6)
    bg.text((xa + xb) / 2, 0.06, lab, ha="center", va="center", fontsize=6, color=c, style="italic")


def note(x, y, s, w=None, **kw):
    kw.setdefault("fontsize", 5.4)
    bg.text(x, y, s, ha="left", va="top", color=S.INK, linespacing=1.15, **kw)


# ---------------------------------------------------------------------------------------------- step 1
base = pd.read_csv(RES / "results_baseline.csv", index_col=0)
x0, w1 = cols[0]
groups = [("Climate", "Climate Exposure", ["Hot days", "Winter soil moisture", "Heavy-precipitation days",
                                           "Max. 1-day precipitation", "Storm days", "Low-wind days"],
           "1990–2024 record, simulated"),
          ("Energy", "Energy System", ["Wind capacity per load", "PV capacity per load", "Thermal capacity per load",
                                       "Storage, hours of load", "Grid density", "Interstate connectivity",
                                       "External-support score"], "reduced by simulated hazards"),
          ("Adaptive", "Socio-economic Capacity", ["GDP per capita", "Human development", "Patents",
                                                   "Age dependency"], "structural")]
yy = top - 0.55
lh = 0.094
for b, title, items, sub in groups:
    hgt = 0.17 + lh * len(items) + 0.04
    bg.add_patch(FancyBboxPatch((x0 + 0.07, yy - hgt), w1 - 0.14, hgt, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=mix(DOM[b], 0.10), ec=DOM[b], lw=0.6))
    bg.add_patch(Rectangle((x0 + 0.07, yy - hgt), 0.035, hgt, fc=DOM[b], ec="none"))
    bg.text(x0 + 0.16, yy - 0.075, f"{title} ({len(items)})", fontsize=6.0, fontweight="bold", color=DOM[b],
            va="center")
    for k, it in enumerate(items):
        bg.text(x0 + 0.19, yy - 0.18 - k * lh, it, fontsize=5.3, va="center", color=S.INK)
    yy -= hgt + 0.06

# ---------------------------------------------------------------------------------------------- step 2
x0, w2 = cols[1]
hazard = R.load_hazard_panel()
hcols = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean", "Rx1day_mm_max",
         "Storm_Days_mean", "low_wind_days_mean"]
fit = R.estimate_copula(hazard, hcols)
rng = np.random.default_rng(12345)
st = "Baden-Württemberg"
sim = np.array([[d[st]["p30_days_mean"], d[st]["Rx1day_mm_mean"]]
                for d in (R.sample_hazard_year(fit, hcols, False, rng) for _ in range(700))])
obs = hazard[hazard.State == st][["p30_days_mean", "Rx1day_mm_mean"]].dropna().to_numpy()
a2 = ax_in(x0 + 0.30, 2.13, w2 - 0.42, 0.53)
a2.scatter(sim[:, 0], sim[:, 1], s=1.2, color=mix(S.CLIM, 0.45), lw=0, label="simulated")
a2.scatter(obs[:, 0], obs[:, 1], s=3.2, color=S.CLIM, lw=0, label="1990–2024")
a2.set_xlabel("Heavy-precipitation days per year", labelpad=0.5); a2.set_ylabel("Max. 1-day (mm)")
a2.legend(loc="lower right", fontsize=4.8, handletextpad=0.1, borderaxespad=0.1, markerscale=2.2,
          labelspacing=0.15)
a2.set_title("Correlated hazards of one state", fontsize=5.6, pad=2)

a3 = ax_in(x0 + 0.30, 1.38, w2 - 0.42, 0.42)
hz = np.linspace(0, 1, 101)
p = R.PARAMS_DERATING
GRAY = "#7F8C9A"
lines = [("Wind", p["Wind"]["beta"], p["Wind"]["delta_max"], S.ENER),
         ("Thermal", p["Thermal"]["alpha"] + p["Thermal"]["beta"], p["Thermal"]["delta_max"], "#56B4E9"),
         ("PV, grid\n(heat)", p["PV"]["beta"], p["PV"]["delta_max"], GRAY)]
for lab, b_, cap, c in lines:
    lo, hi = np.minimum(0.8 * b_ * hz, cap), np.minimum(1.2 * b_ * hz, cap)
    a3.fill_between(hz, 100 * lo, 100 * hi, color=c, alpha=0.25, lw=0)
    a3.plot(hz, 100 * np.minimum(b_ * hz, cap), color=c, lw=0.9)
    a3.text(1.03, 100 * min(b_, cap), lab, fontsize=4.6, color=c, va="center", ha="left")
# grid only: step loss triggered by its own driver, the compound precipitation-storm score Xi*,
# when Xi* exceeds XI_CRIT (pooled 99th percentile); it adds to the gradual heat loss in that year
g, xc = p["Grid"], R.XI_CRIT
a3.plot([xc, xc, 1.0], [0, 100 * g["gamma"], 100 * g["gamma"]], color=GRAY, lw=0.9, ls=(0, (2.2, 1.4)))
a3.plot([0.02, 0.10], [33.2, 33.2], color=GRAY, lw=0.9, ls=(0, (2.2, 1.4)))
a3.text(0.12, 33.2, f"grid, compound event ($\\Xi^*>{xc:.2f}$)", fontsize=4.4,
        color=GRAY, va="center", ha="left")
a3.set_xlim(0, 1); a3.set_xticks([0, 0.5, 1]); a3.set_ylim(0, 40); a3.set_yticks([0, 10, 20, 30, 40])
a3.set_xlabel("Normalised hazard driver", labelpad=0.5); a3.set_ylabel("Derating (%)")
a3.set_title("Capped derating, $Y_{\\mathrm{eff}}=(1-\\delta)Y$", fontsize=5.6, pad=2)
a3.set_position([(x0 + 0.30) / W, 1.38 / H, (w2 - 0.66) / W, 0.42 / H])

# scenarios on the adjacency network
geo = json.loads((S.DATA / "de_states_ne10m.geojson").read_text())
tr = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
geoms = {}
for f in geo["features"]:
    ab = [k for k, v in S.ISO.items() if v == f["properties"]["iso_3166_2"]][0]
    geoms[ab] = shp_transform(lambda x, y, z=None: tr.transform(x, y), shape(f["geometry"])).buffer(0)
cen = {ab: (g.representative_point().x, g.representative_point().y) for ab, g in geoms.items()}
adj = {(S.ABBR[a], S.ABBR[b]) for a, nb in R.ADJACENCY.items() for b in nb if S.ABBR[a] < S.ABBR[b]}
xs = np.array([c[0] for c in cen.values()]); ys = np.array([c[1] for c in cen.values()])
mx, my = xs.mean(), ys.mean()
q_ind = rng.uniform(size=16)
for i, (lab, sub) in enumerate((("Baseline", "own draws"), ("National Pool", "shared draw,\npooled support"),
                                 ("Adjacency Flow", "shared draw,\ncorridors"))):
    aw = (w2 - 0.12) / 3
    a = ax_in(x0 + 0.06 + i * aw, 0.50, aw - 0.03, 0.50)
    a.set_axis_off(); a.set_aspect("equal")
    if i == 1:
        for ab, (x, y) in cen.items():
            a.plot([x, mx], [y, my], color="#9AA5B1", lw=0.35, zorder=1)
        a.scatter([mx], [my], s=16, marker="s", color=NAVY, zorder=3)
    if i == 2:
        for u, v in adj:
            a.plot([cen[u][0], cen[v][0]], [cen[u][1], cen[v][1]], color="#9AA5B1", lw=0.45, zorder=1)
    col = [mix(S.CLIM, 0.25 + 0.75 * q) for q in q_ind] if i == 0 else [mix(S.CLIM, 0.75)] * 16
    a.scatter(xs, ys, s=5, c=col, ec=S.INK, lw=0.25, zorder=2)
    a.text(0.5, -0.04, lab, transform=a.transAxes, ha="center", va="top", fontsize=5.0,
           fontweight="bold")
    a.text(0.5, -0.20, sub, transform=a.transAxes, ha="center", va="top", fontsize=4.6,
           color=S.MUTED, linespacing=1.0)
bg.text(x0 + w2 / 2, 1.04, "Three stress-and-support scenarios", ha="center", va="bottom",
        fontsize=5.6)

# ---------------------------------------------------------------------------------------------- step 3
x0, w3 = cols[2]
a4 = ax_in(x0 + 0.06, 1.82, w3 - 0.12, 0.84)
a4.set_axis_off(); a4.set_aspect("equal")
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
wd = np.array([R.sample_dirichlet_floor([1, 1, 1], rng) for _ in range(900)])
pts = wd @ V
a4.add_patch(Polygon(V, closed=True, fc="white", ec=S.INK, lw=0.5))
a4.scatter(pts[:, 0], pts[:, 1], s=0.9, color=NAVY, alpha=0.55, lw=0)
a4.scatter(*(np.ones(3) / 3 @ V), s=10, color="white", ec=NAVY, lw=0.7, zorder=3)
for (vx, vy), lab, c, ha, dy in zip(V, ("Climate", "Energy", "Socio-econ."), (S.CLIM, S.ENER, S.SOCI),
                                     ("right", "left", "center"), (-0.02, -0.02, 0.03)):
    a4.text(vx + (-0.02 if ha == "right" else 0.02 if ha == "left" else 0), vy + dy, lab, color=c,
            fontsize=5.2, fontweight="bold", ha=ha, va="top" if dy < 0 else "bottom")
a4.set_xlim(-0.42, 1.42); a4.set_ylim(-0.16, 1.0)
bg.text(x0 + w3 / 2, 2.72, "Random domain weights", ha="center", va="bottom", fontsize=5.6)
note(x0 + 0.08, 1.79, "Within-domain weights are sampled\nthe same way.\n"
     "Domain weights: aggregate ranking.\nWithin-domain weights: domain\nscores and relative diagnosis.", fontsize=5.1)
loop = [("hazard year", S.CLIM), ("response multipliers", "#56B4E9"), ("scenario support", NAVY),
        ("weights", NAVY), ("scores, TOPSIS, residuals", NAVY)]
bx, bw, bh, yy0, dy = x0 + 0.10, w3 - 0.42, 0.12, 1.25, 0.155
for k, (t, c) in enumerate(loop):
    yy = yy0 - k * dy
    bg.add_patch(FancyBboxPatch((bx, yy - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.025",
                                fc="white", ec=c, lw=0.5))
    bg.text(bx + bw / 2, yy, t, ha="center", va="center", fontsize=4.9)
    if k < len(loop) - 1:
        bg.annotate("", xy=(bx + bw / 2, yy - dy + bh / 2), xytext=(bx + bw / 2, yy - bh / 2),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.07,head_length=0.08", color=S.MUTED,
                                    lw=0.4, shrinkA=0, shrinkB=0))
ylo, yhi, xr = yy0 - 4 * dy, yy0, bx + bw + 0.07
bg.plot([bx + bw, xr, xr], [ylo, ylo, yhi], color=NAVY, lw=0.6)
bg.annotate("", xy=(bx + bw, yhi), xytext=(xr, yhi),
            arrowprops=dict(arrowstyle="-|>,head_width=0.08,head_length=0.09", color=NAVY, lw=0.6,
                            shrinkA=0, shrinkB=0))
bg.text(xr + 0.06, (ylo + yhi) / 2, "repeated", rotation=90, ha="center",
        va="center", fontsize=5.0, color=NAVY)
note(x0 + 0.08, 0.50, "Gives distributions of scores,\nranks, and residuals.")

# ---------------------------------------------------------------------------------------------- step 4
x0, w4 = cols[3]
def box(x, y, w, h, txt, ec=NAVY, fs=5.3, bold=False):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec=ec, lw=0.5))
    bg.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.1,
            fontweight="bold" if bold else "normal")


def down(x, y0, y1):
    bg.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>,head_width=0.07,head_length=0.08",
                                                                 color=S.MUTED, lw=0.4, shrinkA=0, shrinkB=0))


bg.text(x0 + w4 / 2, 2.72, "Aggregate comparison", ha="center", va="bottom", fontsize=5.6)
box(x0 + 0.12, 2.42, w4 - 0.24, 0.24, "TOPSIS on all 17 indicators")
down(x0 + w4 / 2, 2.42, 2.34)
box(x0 + 0.12, 2.08, w4 - 0.24, 0.26, "Expected rank and rank\nacceptability of each state")
bg.text(x0 + 0.12, 1.93, "Relative weakness", ha="left", va="bottom", fontsize=5.6)
np_res = pd.read_csv(RES / "results_national_pool.csv", index_col=0)
# identification flags from the classification audit used for the manuscript tables
aud = pd.read_csv(RES / "classification_audit.csv")
aud = aud[aud.Scenario == "National Pool"].set_index("State")
assert (aud.identified.reindex(np_res.index) == np_res.identified.astype(bool)).all(), "audit and NP results differ"
a6 = ax_in(x0 + 0.30, 1.12, w4 - 0.42, 0.78)
show = ["Bayern", "Baden-Württemberg", "Brandenburg", "Sachsen", "Hamburg"]
for i, s in enumerate(show):
    left = 0
    prov = np_res.loc[s, "Provisional_domain"]
    for d in ("Climate", "Energy", "Adaptive"):
        v = np_res.loc[s, f"P_{d}"]
        a6.barh(i, v, left=left, height=0.62, color=DOM[d] if d == prov else mix(DOM[d], 0.35), lw=0)
        left += v
    ok = bool(aud.loc[s, "identified"])
    a6.text(1.02, i, "✓" if ok else "–", fontsize=6, ha="left", va="center",
            color=S.INK, fontfamily="DejaVu Sans")
a6.axvline(0.70, color=S.INK, lw=0.6, ls=(0, (2, 1.2)))
a6.set_yticks(range(len(show))); a6.set_yticklabels([S.ABBR[s] for s in show])
a6.set_ylim(len(show) - 0.5, -0.9); a6.set_xlim(0, 1); a6.set_xticks([0, 0.7, 1]); a6.set_xticklabels(["0", "0.70", "1"])
a6.set_xlabel("Share of draws in which each domain\nis the state's weakest (National Pool)", labelpad=0.5,
              linespacing=1.0)
note(x0 + 0.08, 0.79, "Each domain is compared with the level\nthe state's other two domains predict.\n"
     "A relative weakness is identified when the\nsame domain is lowest in at least 70% of\n"
     "draws and its median residual is adverse\nand separated from the others (T > 0).", fontsize=5.1)
a6.text(1.10, -0.78, "✓ identified", fontsize=4.8, ha="right", va="bottom", fontfamily="DejaVu Sans",
        color=S.MUTED)

# ---------------------------------------------------------------------------------------------- step 5
x0, w5 = cols[4]
ev = pd.read_csv(RES / "r3" / "evidence_levels.csv").set_index("state")
a7 = ax_in(x0 + 0.04, 1.02, w5 - 0.08, 1.64)
a7.set_axis_off(); a7.set_aspect("equal")


def patch(g, **kw):
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    verts, codes = [], []
    for pg in polys:
        for ring in [pg.exterior] + list(pg.interiors):
            xy = np.asarray(ring.coords)
            verts += list(xy); codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw)


LEVEL_A = {"Persistent": 1.0, "Conditional": 0.62, "Further investigation": 0.24}
for ab, g in geoms.items():
    s = S.STATES[ab]
    if s in ev.index:
        fc = mix(DOM[ev.loc[s, "domain"]], LEVEL_A[ev.loc[s, "tier"]])
    else:
        fc = "#ECEEF1"
    a7.add_patch(patch(g, fc=fc, ec="white", lw=0.4))
lab_states = [ab for ab in geoms if S.STATES[ab] in ev.index
              and ev.loc[S.STATES[ab], "tier"] in ("Persistent", "Conditional")] + ["HH"]
for ab in lab_states:
    x, y = cen[ab]
    dx, dy = {"HH": (0, 28e3), "BB": (25e3, -40e3)}.get(ab, (0, 0))
    a7.text(x + dx, y + dy, ab, fontsize=5, ha="center", va="center",
            color="white" if ab in ("BY", "BW") else S.INK, fontweight="bold")
a7.autoscale_view()
yy = 0.92
for lab, a_ in (("Persistent", 1.0), ("Conditional", 0.62), ("Further investigation", 0.24)):
    for k, d in enumerate(("Climate", "Energy", "Adaptive")):
        bg.add_patch(Rectangle((x0 + 0.08 + k * 0.085, yy - 0.075), 0.075, 0.065, fc=mix(DOM[d], a_), ec="none"))
    bg.text(x0 + 0.08 + 3 * 0.085 + 0.03, yy - 0.042, lab, fontsize=5.2, va="center")
    yy -= 0.105
bg.text(x0 + 0.08, 0.565, "Columns: Climate, Energy,\nSocio-econ. Grey: no\nconsistent weaker domain.", fontsize=4.8, va="top",
        color=S.MUTED, linespacing=1.1)
bg.text(x0 + w5 / 2, 2.72, "Evidence level of each result", ha="center", va="bottom", fontsize=5.6)

S.save(fig, "fig01_workflow")
print("wrote", S.OUT / "fig01_workflow.pdf")
