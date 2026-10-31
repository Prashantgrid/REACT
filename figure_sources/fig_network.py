"""Interstate support network (replaces fig_network_geo.pdf).

Nodes: 16 states, coloured by net position class (r_i), area proportional to connectivity C_i.
Edges: the 29 modelled interstate corridors (extracted from the original figure).
Dashed stubs: international interconnections (enter only implicitly through r_i).
Boundaries: Natural Earth 1:10m admin-1 (public domain).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

import style as S

S.apply()
geo = json.loads((S.DATA / "de_states_ne10m.geojson").read_text())
net = S.load("network.json")
ric = S.load("ri_ci.json")
tr = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

geoms = {}
for f in geo["features"]:
    iso = f["properties"]["iso_3166_2"]
    ab = [k for k, v in S.ISO.items() if v == iso][0]
    geoms[ab] = shp_transform(lambda x, y, z=None: tr.transform(x, y), shape(f["geometry"])).buffer(0)


def poly_patch(geom, **kw):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    verts, codes = [], []
    for p in polys:
        for ring in [p.exterior] + list(p.interiors):
            xy = np.asarray(ring.coords)
            verts += xy.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw)


# node positions: representative points with small manual adjustments (km)
pos = {}
for ab, g in geoms.items():
    p = g.representative_point()
    pos[ab] = np.array([p.x, p.y])
ADJ = {"BB": (45, -20), "BE": (0, 0), "NI": (-10, -25), "HB": (0, 0), "HH": (0, 0),
       "SH": (-10, 0), "MV": (10, -10), "ST": (0, 0), "TH": (0, 0), "SN": (10, 0),
       "HE": (0, 0), "RP": (0, 0), "SL": (0, 0), "BW": (0, 0), "BY": (15, 0), "NRW": (0, 0)}
for ab, (dx, dy) in ADJ.items():
    pos[ab] = pos[ab] + np.array([dx, dy]) * 1e3
pos["BE"] = np.array(geoms["BE"].centroid.coords[0])
pos["BB"] = np.array(tr.transform(14.15, 52.15))
pos["HH"] = np.array(geoms["HH"].centroid.coords[0])


def cls(r):
    return "exp" if r < -0.1 else ("imp" if r > 0.1 else "bal")


COL = {"exp": S.EXPORTER, "bal": S.BALANCED, "imp": S.IMPORTER}
TXT = {"exp": "white", "bal": "white", "imp": S.INK}


def msize(c):  # marker area (pt^2) strictly proportional to C_i
    return 480 * c


# international interconnections: (state, label, lon, lat)
INTL = [("SH", "NOR", 8.3, 55.35), ("SH", "DNK", 9.6, 55.25), ("SH", "SWE", 11.2, 54.95),
        ("MV", "DNK", 12.9, 54.85), ("BB", "POL", 15.35, 52.75), ("SN", "POL", 15.4, 51.25),
        ("SN", "CZE", 14.3, 50.45), ("BY", "CZE", 13.35, 49.55), ("BY", "AUT", 13.1, 47.25),
        ("BW", "CHE", 8.35, 47.28), ("BW", "FRA", 7.2, 48.35), ("SL", "FRA", 6.45, 48.95),
        ("RP", "LUX", 5.95, 49.75), ("NRW", "BEL", 5.75, 50.65), ("NRW", "NLD", 5.75, 51.75),
        ("NI", "NLD", 6.5, 53.25)]

fig, ax = plt.subplots(figsize=(S.COL_W, 4.35))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0.135)
for ab, g in geoms.items():
    ax.add_patch(poly_patch(g, facecolor="#EFEDE8", edgecolor="white", linewidth=0.6, zorder=1))
de = unary_union(list(geoms.values()))
ax.add_patch(poly_patch(de, facecolor="none", edgecolor="#A8A29E", linewidth=0.5, zorder=2))

for st, lab, lon, lat in INTL:
    x, y = tr.transform(lon, lat)
    ax.plot([pos[st][0], x], [pos[st][1], y], ls=(0, (2.2, 1.6)), lw=0.7, color="#7A8CA8", zorder=3)
    ax.text(x, y, lab, ha="center", va="center", fontsize=6.3, style="italic", color="#3B4F6B",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=4)

for a, b in net["edges"]:
    ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#8C8C8C", lw=1.0,
            solid_capstyle="round", zorder=5)

for ab in pos:
    r, c = ric[ab]
    k = cls(r)
    ax.scatter(*pos[ab], s=msize(c), c=COL[k], edgecolors=S.INK, linewidths=0.6, zorder=6)
    ax.text(*pos[ab], ab, ha="center", va="center", fontsize=5.6 if len(ab) == 3 else 6.0,
            weight="bold", color=TXT[k], zorder=7)

minx, miny, maxx, maxy = de.bounds
ax.set_xlim(minx - 55e3, maxx + 60e3)
ax.set_ylim(miny - 25e3, maxy + 60e3)
ax.set_aspect("equal")
ax.axis("off")

# ---- legend --------------------------------------------------------------------
h = [Line2D([], [], marker="o", ls="", ms=6.5, mfc=S.EXPORTER, mec=S.INK, mew=0.6,
            label=r"Net exporter ($r_i<-0.1$)"),
     Line2D([], [], marker="o", ls="", ms=6.5, mfc=S.BALANCED, mec=S.INK, mew=0.6,
            label=r"Balanced ($|r_i|\leq0.1$)"),
     Line2D([], [], marker="o", ls="", ms=6.5, mfc=S.IMPORTER, mec=S.INK, mew=0.6,
            label=r"Net importer ($r_i>0.1$)")]
h2 = [Line2D([], [], color="#8C8C8C", lw=1.0, label="Interstate corridor (modelled)"),
      Line2D([], [], color="#7A8CA8", lw=0.7, ls=(0, (2.2, 1.6)),
             label="International link (via $r_i$ only)")]
leg1 = fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.6,
                  handletextpad=0.3, labelspacing=0.35, borderaxespad=0.2)
leg2 = fig.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.46, 0.058), fontsize=6.6,
                  handlelength=2.0, handletextpad=0.4, labelspacing=0.35, borderaxespad=0.2)
# size key
kx = fig.add_axes([0.47, 0.0, 0.53, 0.06])
kx.axis("off")
kx.set_xlim(0, 10)
kx.set_ylim(0, 1)
kx.text(0.05, 0.5, "Node area $\\propto C_i$", va="center", fontsize=6.6)
for i, c in enumerate([0.3, 0.6, 0.9]):
    xc = 5.0 + i * 1.75
    kx.scatter(xc, 0.5, s=msize(c), c="white", edgecolors=S.INK, linewidths=0.6)
    kx.text(xc, 0.5, f"{c:.1f}", ha="center", va="center", fontsize=5.8)

S.save(fig, "fig_network_geo")
print("ok")
