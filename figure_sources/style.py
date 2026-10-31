"""Shared figure style for the REACT manuscript.

All figures are drawn at their final printed size so that font sizes are true:
  single column = 3.5 in (88.9 mm), full width = 7.16 in (181.9 mm).
Text is Times-compatible (TeX Gyre Termes / STIX math) to match the manuscript body.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

COL_W = 3.5     # in
FULL_W = 7.16   # in

# Domain colours (Okabe-Ito; validated colour-blind safe, all pairs)
CLIM = "#D55E00"
ENER = "#0072B2"
SOCI = "#009E73"
DOMAIN_COL = {"Climate": CLIM, "Energy": ENER, "Socio-econ.": SOCI}

# Net-position colours (purple / neutral / orange; distinct from the domain hues)
EXPORTER = "#5E3C99"
BALANCED = "#9A9A9A"
IMPORTER = "#E69F00"

INK = "#1F2A37"
MUTED = "#6B7280"
GRID = "#E5E7EB"

STATES = {
    "BW": "Baden-Württemberg", "BY": "Bayern", "BE": "Berlin", "BB": "Brandenburg",
    "HB": "Bremen", "HH": "Hamburg", "HE": "Hessen", "MV": "Mecklenburg-Vorpommern",
    "NI": "Niedersachsen", "NRW": "Nordrhein-Westfalen", "RP": "Rheinland-Pfalz",
    "SL": "Saarland", "SN": "Sachsen", "ST": "Sachsen-Anhalt", "SH": "Schleswig-Holstein",
    "TH": "Thüringen",
}
ABBR = {v: k for k, v in STATES.items()}
ISO = {"BW": "DE-BW", "BY": "DE-BY", "BE": "DE-BE", "BB": "DE-BB", "HB": "DE-HB", "HH": "DE-HH",
       "HE": "DE-HE", "MV": "DE-MV", "NI": "DE-NI", "NRW": "DE-NW", "RP": "DE-RP", "SL": "DE-SL",
       "SN": "DE-SN", "ST": "DE-ST", "SH": "DE-SH", "TH": "DE-TH"}


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,       # embed TrueType (Elsevier requires embedded fonts)
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def load(name):
    return json.loads((DATA / name).read_text())
