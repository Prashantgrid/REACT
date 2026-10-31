#!/usr/bin/env python3
"""Sanity checks of the REACT input data.

Usage:  python REACT_data_checks.py [--storage-file <16-state MaStR csv>] > data_checks.md

Checks the embedded hazard panel and structural panel (completeness, ranges, internal
consistency, national totals, outliers) and, if given, a 16-state MaStR storage aggregate.
Prints a Markdown report. Reference values for national totals are order-of-magnitude
plausibility bounds, not official statistics.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import REACT_ESR_reproducible as react

core = react.core
HCOLS = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean",
         "Rx1day_mm_max", "Storm_Days_mean", "low_wind_days_mean"]
# plausible physical bounds for state-year values (generous; used only to catch unit errors).
# The German daily precipitation record (312 mm, Zinnwald-Georgenfeld, 12 Aug 2002) sets the Rx1day_max bound.
BOUNDS = {"hot_days_mean": (0, 60), "soil_moist_winter_mean": (20, 150), "p30_days_mean": (0, 15),
          "Rx1day_mm_mean": (10, 120), "Rx1day_mm_max": (15, 350), "Storm_Days_mean": (0, 80),
          "low_wind_days_mean": (0, 366)}
# plausibility ranges for national sums of the structural panel (order of magnitude only)
NATIONAL = {"Wind_Onshore_MW": (55e3, 75e3, "onshore wind, MW"),
            "Solar_PV_MW": (90e3, 130e3, "solar PV, MW"),
            "Thermal_Fossil_MW": (55e3, 90e3, "fossil thermal, MW"),
            "Cons_TWh": (430, 560, "electricity consumption, TWh"),
            "Gen_TWh": (400, 600, "electricity generation, TWh")}
GERMANY_AREA_KM2 = 357_600


def line(ok, text):
    return f"- {'PASS' if ok else 'CHECK'}: {text}"


def hazard_checks(hz):
    out = ["## Hazard panel (DWD, 1990-2024)", ""]
    n_states, years = hz.State.nunique(), sorted(hz.Year.unique())
    out.append(line(len(hz) == 560 and n_states == 16 and years == list(range(1990, 2025)),
                    f"{len(hz)} rows, {n_states} states, years {years[0]}-{years[-1]}"))
    dup = hz.duplicated(["State", "Year"]).sum()
    out.append(line(dup == 0, f"duplicate state-years: {dup}"))
    miss = hz[HCOLS].isna().sum()
    miss_txt = ", ".join(f"{c}: {int(v)}" for c, v in miss.items() if v)
    only_1990 = hz[hz.soil_moist_winter_mean.isna()].Year.unique().tolist() == [1990]
    out.append(line(only_1990 and miss.drop("soil_moist_winter_mean").sum() == 0,
                    f"missing values: {miss_txt or 'none'} (winter soil moisture missing only for 1990: {only_1990})"))
    for c, (lo, hi) in BOUNDS.items():
        x = hz[c].dropna()
        out.append(line(x.min() >= lo and x.max() <= hi, f"{c}: range {x.min():.2f}-{x.max():.2f} (bounds {lo}-{hi})"))
    bad = (hz.Rx1day_mm_max < hz.Rx1day_mm_mean).sum()
    out.append(line(bad == 0, f"station-maximum 1-day precipitation >= state-mean value in all rows (violations: {bad})"))
    # robust within-state outliers
    flags = []
    for c in HCOLS:
        for s, g in hz.groupby("State"):
            x = g[c].dropna()
            med, mad = x.median(), (x - x.median()).abs().median()
            if mad <= 0:
                continue
            z = 0.6745 * (x - med) / mad
            for idx in z[z.abs() > 5].index:
                flags.append(f"{s} {int(hz.loc[idx, 'Year'])} {c} = {hz.loc[idx, c]:.2f} (robust z = {z[idx]:.1f})")
    out.append(line(True, f"within-state outliers with |robust z| > 5: {len(flags)} (listed below; extremes are expected for "
                          "heavy-precipitation and storm years and are not errors by themselves)"))
    out += [f"  - {f}" for f in flags[:30]]
    out.append("")
    return out


def static_checks(st, storage_file=None):
    out = ["## Structural panel", ""]
    st = st.loc[core.STATE_ORDER]
    out.append(line(len(st) == 16 and st.index.is_unique, f"{len(st)} states, unique"))
    num = st.select_dtypes("number")
    na = num.isna().sum()
    out.append(line(na.sum() == 0, f"missing values: {int(na.sum())}"))
    neg = (num.drop(columns=["r_i", "r_import"], errors="ignore") < 0).sum()
    out.append(line(neg.sum() == 0, f"negative values in non-signed columns: {int(neg.sum())}"))
    zeros = {c: list(st.index[st[c] == 0]) for c in ["Wind_Onshore_MW", "Solar_PV_MW", "Thermal_Fossil_MW",
                                                    "Storage_MWh", "WLINES", "C_i"]}
    zeros = {k: v for k, v in zeros.items() if v}
    out.append(line(not zeros, f"zero capacities: {zeros or 'none'}"))
    for c, (lo, hi, lab) in NATIONAL.items():
        v = float(st[c].sum())
        out.append(line(lo <= v <= hi, f"national sum of {lab}: {v:,.0f} (plausible {lo:,.0f}-{hi:,.0f})"))
    stor = float(st.Storage_MWh.sum())
    out.append(line(15e3 <= stor <= 40e3, f"national battery storage: {stor:,.0f} MWh"))
    net = st.Gen_TWh - st.Cons_TWh
    r_chk = (net / net.abs().max() - st.r_i).abs().max()
    out.append(line(r_chk < 0.01, f"r_i equals (generation - consumption)/max|net| (max deviation {r_chk:.4f})"))
    area = sum(react.STATE_AREA_KM2.values())
    out.append(line(abs(area - GERMANY_AREA_KM2) / GERMANY_AREA_KM2 < 0.005, f"state areas sum to {area:,} km2"))
    out.append(line(st.HDI_2022.between(0.85, 1).all(), f"HDI range {st.HDI_2022.min():.3f}-{st.HDI_2022.max():.3f}"))
    out.append(line(st.ADR_2025.between(45, 90).all(), f"total age-dependency ratio range {st.ADR_2025.min():.0f}-{st.ADR_2025.max():.0f}"))
    out.append(line(st.GDP_per_Capita_EUR.between(25e3, 100e3).all(),
                    f"GDP per capita range {st.GDP_per_Capita_EUR.min():,.0f}-{st.GDP_per_Capita_EUR.max():,.0f} EUR"))
    load = react.average_load_mw(st)
    hours = st.Storage_MWh.to_numpy() / load
    out.append(line(True, "storage hours of average load: " + ", ".join(
        f"{react.ABBR[s]} {h:.2f}" for s, h in sorted(zip(core.STATE_ORDER, hours), key=lambda t: t[1]))))
    ratios = pd.DataFrame({c: st[c].to_numpy() / load for c in ["Wind_Onshore_MW", "Solar_PV_MW", "Thermal_Fossil_MW"]},
                          index=st.index)
    out.append(line(True, "largest capacity-to-load ratios: " + ", ".join(
        f"{c.split('_')[0]} {ratios[c].idxmax()} {ratios[c].max():.1f}" for c in ratios)))
    out.append("")
    if storage_file:
        out += storage_checks(st, storage_file)
    return out


def storage_checks(st, path):
    path_label = Path(path).parent.name + "/" + Path(path).name
    out = [f"## MaStR storage aggregate ({path_label})", ""]
    m = pd.read_csv(path).set_index("state")
    out.append(line(set(m.index) == set(core.STATE_ORDER) and len(m) == 16, f"{len(m)} state rows"))
    e_raw, e = m.battery_registered_usable_energy_MWh, m.battery_energy_MWh_excluding_duration_flags
    p_net, p_gross = m.battery_net_power_MW, m.battery_gross_power_MW
    out.append(line((p_net <= p_gross + 1e-6).all(), "battery net power <= gross power in every state"))
    dur = e / p_net
    out.append(line(dur.between(1.0, 3.0).all(), "battery energy/net power by state (h): " +
                    ", ".join(f"{react.ABBR[s]} {v:.2f}" for s, v in dur.sort_values().items())))
    diff = (e_raw - e)
    big = diff[diff > 1]
    out.append(line(big.empty, "energy removed by the duration screen: " +
                    (", ".join(f"{react.ABBR[s]} {v:,.0f} MWh" for s, v in big.items()) or "none")))
    out.append(line(True, f"national battery energy: raw {e_raw.sum():,.0f} MWh; screened {e.sum():,.0f} MWh"))
    if "all_storage_registered_usable_energy_MWh" in m:
        other = m.all_storage_registered_usable_energy_MWh - e_raw
        big = other[other > 5000]
        out.append(line(big.sum() < 60e3, "non-battery storage energy (mainly pumped storage) above 5 GWh: " +
                        ", ".join(f"{react.ABBR[s]} {v:,.0f} MWh" for s, v in big.items()) +
                        f"; national {other.sum():,.0f} MWh"))
    panel = st.Storage_MWh
    ratio = (panel / e).reindex(core.STATE_ORDER)
    out.append(line(True, "panel storage / screened registry energy: " +
                    ", ".join(f"{react.ABBR[s]} {v:.2f}" for s, v in ratio.sort_values().items())))
    rho = pd.Series(panel / react.average_load_mw(st), index=core.STATE_ORDER).rank().corr(
        pd.Series(e.reindex(core.STATE_ORDER).to_numpy() / react.average_load_mw(st), index=core.STATE_ORDER).rank())
    out.append(line(rho > 0.8, f"rank correlation of storage hours, panel vs registry: {rho:.2f}"))
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-file", default=None)
    ap.add_argument("--use-storage-file-in-panel", action="store_true")
    a = ap.parse_args()
    if a.use_storage_file_in_panel:
        react.STORAGE_FILE = a.storage_file
    hz = core.load_hazard_panel()
    st = react._configured_static()
    print("# REACT input data checks\n")
    print("PASS = within the stated bound or consistent; CHECK = needs a look.\n")
    print("\n".join(hazard_checks(hz)))
    print("\n".join(static_checks(st, a.storage_file)))


if __name__ == "__main__":
    main()
