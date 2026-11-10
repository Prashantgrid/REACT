"""Relative domain weaknesses under the three stress-and-support scenarios (fig05_relative_weakness_map).

Data: data/domain_profiles.json, written by REACT_ESR_reproducible.py (results/figure_data/).
Boundaries: Natural Earth 1:10m admin-1 (public domain).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform  # noqa: deprecated alias kept for older shapely

import style as S

S.apply()
prof = S.load("domain_profiles.json")
geo = json.loads((S.DATA / "de_states_ne10m.geojson").read_text())
tr = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

geoms = {}
for f in geo["features"]:
    ab = [k for k, v in S.ISO.items() if v == f["properties"]["iso_3166_2"]][0]
    geoms[ab] = shp_transform(lambda x, y, z=None: tr.transform(x, y), shape(f["geometry"])).buffer(0)


def pale(c, a=0.30):
    r, g, b = to_rgb(c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def patch(g, **kw):
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior] + list(p.interiors):
            xy = np.asarray(ring.coords)
            verts += list(xy); codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw)


# label anchors (m) and call-outs for the small states
nudge = {"BB": (20e3, -55e3), "NI": (20e3, -25e3), "SH": (-45e3, -5e3), "MV": (10e3, -10e3), "HE": (0, 5e3)}
callout = {"HB": (-115e3, 45e3), "HH": (60e3, 85e3), "BE": (85e3, 40e3), "SL": (-85e3, -5e3)}

fig, axes = plt.subplots(1, 3, figsize=(S.FULL_W, 3.55))
for ax, sc in zip(axes, ("Baseline", "National Pool", "Adjacency Flow")):
    n_id = sum(prof[s][sc]["identified"] for s in prof)
    for ab, g in geoms.items():
        p = prof[ab][sc]; c = S.DOMAIN_COL[p["domain"]]
        ax.add_patch(patch(g, fc=c if p["identified"] else pale(c), ec="white", lw=0.45))
    for ab, g in geoms.items():
        p = prof[ab][sc]; ident = p["identified"]
        pt = g.representative_point() if ab != "BB" else g.centroid
        x, y = pt.x, pt.y
        txt = f"{ab}\n{p['psel']:.2f}"
        col = "white" if ident else S.INK
        if ab in callout:
            dx, dy = callout[ab]
            ax.plot([x, x + dx], [y, y + dy], color=S.MUTED, lw=0.4)
            ax.plot(x, y, "o", ms=1.2, color=S.MUTED)
            c = S.DOMAIN_COL[p["domain"]]
            ax.text(x + dx, y + dy, txt, ha="center", va="center", fontsize=6.0, color=col, linespacing=0.95,
                    fontweight="bold" if ident else "normal",
                    bbox=dict(boxstyle="square,pad=0.18", fc=c if ident else pale(c), ec=S.MUTED, lw=0.4))
        else:
            dx, dy = nudge.get(ab, (0, 0))
            ax.text(x + dx, y + dy, txt, ha="center", va="center", fontsize=6.0, color=col, linespacing=0.95,
                    fontweight="bold" if ident else "normal")
    ax.set_title(f"({'abc'[['Baseline', 'National Pool', 'Adjacency Flow'].index(sc)]}) {sc} · {n_id} identified",
                 fontsize=7.5)
    ax.set_aspect("equal"); ax.autoscale_view(); ax.axis("off")
    x0, x1 = ax.get_xlim(); ax.set_xlim(x0 - 60e3, x1 + 60e3)

h = [Patch(fc=S.DOMAIN_COL[k], ec="none") for k in ("Climate", "Energy", "Socio-econ.")] + \
    [Patch(fc=pale(S.DOMAIN_COL[k]), ec="none") for k in ("Climate", "Energy", "Socio-econ.")]
lab = ["Climate Exposure", "Energy System", "Socio-economic Capacity"] * 2
fig.legend(h, lab, ncol=2, loc="lower center", frameon=False, fontsize=6.6, columnspacing=1.2, handlelength=1.3,
           bbox_to_anchor=(0.5, -0.02),
           title="Relative domain weakness (saturated)  |  Multidomain, provisional domain (pale)", title_fontsize=6.6)
fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.18, wspace=0.02)
S.save(fig, "fig05_relative_weakness_map")
print("ok")
