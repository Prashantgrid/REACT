"""Stylised warming-level deltas for the REACT hazard panel (Reviewer 3.4).

REACT's diagnostic is a cross-state comparison: a uniform shift of every state's hazard level
cancels under min-max normalisation, so what matters for the classifications is *differential*
change. The deltas below are therefore physically state-dependent where physics makes them so:

  hot days       quantile-delta mapping through the weather generator: each state-year's
                 hot-day count is mapped to the summer temperature shift that produces it in the
                 state's own climatology, the shift is raised by dT, and the count recomputed. States
                 whose summers lie closer to the 30 C threshold gain more hot days per kelvin.
  Rx1day (mean, max)   x (1 + 0.07 dT)       Clausius-Clapeyron scaling of daily extremes
  P30 days             x 1.25**dT            exceedances of 30 mm/day under that scaling for an
                                             exponential tail with an 8-9 mm scale parameter
  low-wind days        x (1 + 0.02 dT)       small projected decline of mean near-surface wind
  storm days, winter soil moisture  unchanged (no robust projected signal at state scale)
dT is regional warming relative to the 1990-2024 climate: +1 K (roughly mid-century under
intermediate emissions) and +2 K (late century under high emissions). These are sensitivity
cases, not downscaled projections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import STATES
from .weather import DAYS


def hot_day_response(wg, years=30, shifts=np.arange(-8.0, 12.01, 0.25), seed=99):
    rng = np.random.default_rng(seed)
    E = np.zeros((len(shifts), 16))
    for _ in range(years):
        zh, zd = wg._cloud_latent(rng); an = wg._temp_anomaly(rng, zd)
        for k, m in enumerate(shifts):
            tmax = wg._daily_temp(an, zd, (wg.t_offset[None, :] + m) * wg.summer_win[:, None])[1]
            E[k] += (tmax >= 30).sum(0)
    return shifts, E / years


def delta_panel(hazard, curve, dT):
    shifts, E = curve
    h = hazard.copy()
    for i, s in enumerate(STATES):
        m = h.State == s
        x = h.loc[m, "hot_days_mean"].to_numpy(float)
        e = np.maximum.accumulate(E[:, i])
        sh = np.interp(np.maximum(x, e[0] + 1e-6), e, shifts)
        h.loc[m, "hot_days_mean"] = np.interp(sh + dT, shifts, e)
    h["Rx1day_mm_mean"] *= 1 + 0.07 * dT
    h["Rx1day_mm_max"] *= 1 + 0.07 * dT
    h["p30_days_mean"] *= 1.25 ** dT
    h["low_wind_days_mean"] = np.minimum(h["low_wind_days_mean"] * (1 + 0.02 * dT), DAYS)
    return h


def historical_ranges(hazard, fit):
    cols = ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean", "Rx1day_mm_mean", "Rx1day_mm_max",
            "Storm_Days_mean", "low_wind_days_mean"]
    pooled = {c: (float(hazard[c].min()), float(hazard[c].max())) for c in cols}
    import REACT_v4 as core
    xi = ("p30_days_mean", "Rx1day_mm_mean", "Storm_Days_mean")
    lo = {c: np.array([fit["marginals"][s][c].min() for s in core.STATE_ORDER]) for c in xi}
    hi = {c: np.array([fit["marginals"][s][c].max() for s in core.STATE_ORDER]) for c in xi}
    return dict(pooled=pooled, state_lo=lo, state_hi=hi)


def delta_panel_differential(hazard, curve, dT, rng, spread=0.5):
    """As ``delta_panel`` but each state's change is scaled by an independent factor
    U(1-spread, 1+spread) per hazard, mimicking the inter-model spread of regional change."""
    shifts, E = curve
    h = hazard.copy()
    f = {c: dict(zip(STATES, rng.uniform(1 - spread, 1 + spread, 16)))
         for c in ("hot", "rx", "p30", "lw")}
    for i, s in enumerate(STATES):
        m = h.State == s
        x = h.loc[m, "hot_days_mean"].to_numpy(float)
        e = np.maximum.accumulate(E[:, i])
        sh = np.interp(np.maximum(x, e[0] + 1e-6), e, shifts)
        h.loc[m, "hot_days_mean"] = np.interp(sh + dT * f["hot"][s], shifts, e)
        h.loc[m, "Rx1day_mm_mean"] *= 1 + 0.07 * dT * f["rx"][s]
        h.loc[m, "Rx1day_mm_max"] *= 1 + 0.07 * dT * f["rx"][s]
        h.loc[m, "p30_days_mean"] *= 1.25 ** (dT * f["p30"][s])
        h.loc[m, "low_wind_days_mean"] = np.minimum(h.loc[m, "low_wind_days_mean"] * (1 + 0.02 * dT * f["lw"][s]), DAYS)
    return h, f
