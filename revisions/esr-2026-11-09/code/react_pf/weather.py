"""Hazard-conditioned hourly weather and load for the 16 states.

Two sources are supported:

1. ``WeatherGenerator`` (default): a stochastic generator of hourly onshore/offshore wind
   capacity factors, PV capacity factors, air temperature and load. Its annual statistics are
   conditioned on REACT's copula hazard draw for that year, so the chronological power-system
   model and REACT-Lite share one hazard model:
     * the number of low-wind days in each state equals the drawn ``low_wind_days`` exactly
       (daily-mean wind latent below a state threshold calibrated to the 1990-2024 mean);
     * the number of hot days (daily Tmax >= 30 C) equals the drawn ``hot_days`` exactly
       (summer temperature shift solved per state and year);
     * winter soil moisture and the compound precipitation-storm index Xi* are passed through
       to thermal derating and to grid-damage events.
   Spatial structure: a common synoptic factor plus distance-decaying regional factors
   (exp(-d/L)), AR(1) persistence at synoptic and sub-daily scales. Marginals are calibrated to
   state full-load hours and to the national 2019 profiles in the PyPSA model-energy example.

2. ``ExternalWeather``: drop-in loader for observed or reanalysis series (renewables.ninja
   NUTS-2 aggregated to states, ERA5/atlite capacity factors, DWD/ERA5 temperature, OPSD/SMARD
   load). In this mode each Monte Carlo year replays one historical weather year and the
   hazard values of the same year from the REACT panel (``historical replay``).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import ndtr
from scipy.stats import beta as beta_dist

from .network import STATES, CACHE, ABBR

H = 8760
DAYS = 365
DOY_H = np.repeat(np.arange(DAYS), 24)            # day index of each hour (0-based)
HOUR = np.tile(np.arange(24), DAYS)

# Calibration targets (assumptions; replace with MaStR/Energy-Charts state statistics if available)
ONSHORE_FLH = {"Schleswig-Holstein": 2150, "Bremen": 2000, "Hamburg": 1900, "Niedersachsen": 2000,
               "Mecklenburg-Vorpommern": 2000, "Brandenburg": 1800, "Sachsen-Anhalt": 1850, "Berlin": 1700,
               "Nordrhein-Westfalen": 1800, "Hessen": 1650, "Thüringen": 1750, "Sachsen": 1700,
               "Rheinland-Pfalz": 1700, "Saarland": 1650, "Baden-Württemberg": 1500, "Bayern": 1500}
PV_FLH = {"Baden-Württemberg": 1030, "Bayern": 1040, "Saarland": 1000, "Rheinland-Pfalz": 990, "Hessen": 960,
          "Thüringen": 950, "Sachsen": 960, "Sachsen-Anhalt": 940, "Brandenburg": 950, "Berlin": 930,
          "Nordrhein-Westfalen": 910, "Niedersachsen": 900, "Bremen": 880, "Hamburg": 870,
          "Schleswig-Holstein": 880, "Mecklenburg-Vorpommern": 920}
OFFSHORE_CF = 0.40
# 1991-2020 state mean air temperature (C) and half annual range (K), DWD climatology (rounded)
T_MEAN = {"Baden-Württemberg": 9.2, "Bayern": 8.7, "Berlin": 10.2, "Brandenburg": 9.9, "Bremen": 9.9,
          "Hamburg": 9.8, "Hessen": 9.4, "Mecklenburg-Vorpommern": 9.1, "Niedersachsen": 9.6,
          "Nordrhein-Westfalen": 10.0, "Rheinland-Pfalz": 9.7, "Saarland": 9.8, "Sachsen": 9.2,
          "Sachsen-Anhalt": 9.7, "Schleswig-Holstein": 9.1, "Thüringen": 8.7}
T_AMP = {"Schleswig-Holstein": 8.0, "Hamburg": 8.4, "Bremen": 8.3, "Mecklenburg-Vorpommern": 8.6,
         "Niedersachsen": 8.6, "Nordrhein-Westfalen": 8.3, "Berlin": 9.6, "Brandenburg": 9.6,
         "Sachsen-Anhalt": 9.3, "Sachsen": 9.2, "Thüringen": 9.0, "Hessen": 8.8, "Rheinland-Pfalz": 8.7,
         "Saarland": 8.6, "Baden-Württemberg": 8.9, "Bayern": 9.2}

# latent-field structure
WIND = dict(w_common=0.55, w_regional=0.35, w_local=0.10, tau_c=36.0, tau_r=24.0, tau_l=4.0,
            L_km=300.0, season=0.45, kappa=2.6)
CLOUD = dict(w_common=0.50, w_regional=0.40, tau_d=1.5, L_km=250.0, kappa=3.0, k_mean=0.52, k_amp=0.13,
             hourly_sd=0.25, tau_h=3.0)
TEMP = dict(w_common=0.70, tau_d=3.0, L_km=400.0, sd_mean=3.9, sd_amp=0.0, diurnal_mean=4.0,
            diurnal_amp=2.0, cloud_coupling=0.4)
LOAD = dict(eta_heat=0.006, eta_cool=0.003, t_heat=15.0, t_cool=24.0)


def _ar1(n, cols, tau, rng, L=None):
    phi = np.exp(-1.0 / tau)
    e = rng.standard_normal((n, cols))
    if L is not None:
        e = e @ L.T
    x = lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], e, axis=0)
    x[0] = e[0]
    return x


def _spatial_chol(xy_km, L_km):
    d = np.sqrt(((xy_km[:, None, :] - xy_km[None, :, :]) ** 2).sum(-1))
    C = np.exp(-d / L_km)
    return np.linalg.cholesky(C + 1e-9 * np.eye(len(C)))


def national_load_shape():
    """Hourly national load shape (mean 1) from the 3-hourly 2019 series of the PyPSA model-energy
    example (ENTSO-E based); linear interpolation to hourly resolution."""
    import pypsa, warnings, logging
    warnings.filterwarnings("ignore"); logging.getLogger("pypsa").setLevel(logging.ERROR)
    n = pypsa.Network(os.path.join(CACHE, "model-energy.nc"))
    d = n.loads_t.p_set.iloc[:, 0].to_numpy(float)
    w = n.generators_t.p_max_pu
    x3 = np.arange(len(d)) * 3.0
    xh = np.arange(H)
    shape = np.interp(xh, x3, d, period=H); shape /= shape.mean()
    wind = np.interp(xh, x3, w["wind"].to_numpy(float), period=H)
    solar = np.interp(xh, x3, w["solar"].to_numpy(float), period=H)
    return shape, wind, solar


class _BetaLUT:
    def __init__(self, mean, kappa, n=4001):
        a, b = mean * kappa, (1 - mean) * kappa
        self.u = np.linspace(0, 1, n)
        self.q = beta_dist.ppf(np.clip(self.u, 1e-6, 1 - 1e-6), a, b)

    def __call__(self, u):
        return np.interp(u, self.u, self.q)


class WeatherGenerator:
    def __init__(self, centroids, hazard_panel, seed=777, calib_years=12):
        self.states = STATES
        self.xy = np.array([[centroids[s][0], centroids[s][1]] for s in STATES])
        self.lonlat = np.array([[centroids[s][2], centroids[s][3]] for s in STATES])
        self.Lw = _spatial_chol(self.xy, WIND["L_km"])
        self.Lc = _spatial_chol(self.xy, CLOUD["L_km"])
        self.Lt = _spatial_chol(self.xy, TEMP["L_km"])
        hz = hazard_panel.groupby("State").mean(numeric_only=True)
        self.lw_mean = np.array([hz.loc[s, "low_wind_days_mean"] for s in STATES])
        self.hd_mean = np.array([hz.loc[s, "hot_days_mean"] for s in STATES])
        self.load_shape, self.nat_wind_2019, self.nat_pv_2019 = national_load_shape()
        self._clear_sky()
        self.doy_f = DOY_H.astype(float)
        self.season_w = np.cos(2 * np.pi * (self.doy_f - 15) / 365.25)
        self.summer_win = np.exp(-0.5 * ((np.arange(DAYS) - 200) / 45.0) ** 2)
        self._calibrate(np.random.default_rng(seed), calib_years)

    # ------------------------------------------------------------------ solar geometry
    def _clear_sky(self):
        import pvlib
        times = pd.date_range("2019-01-01", periods=H, freq="h", tz="UTC")
        ghi, zen, azi, dni_extra = [], [], [], []
        for lon, lat in self.lonlat:
            loc = pvlib.location.Location(lat, lon)
            sp = loc.get_solarposition(times)
            cs = loc.get_clearsky(times, model="ineichen", solar_position=sp)
            ghi.append(cs.ghi.to_numpy()); zen.append(sp.zenith.to_numpy()); azi.append(sp.azimuth.to_numpy())
        self.ghi_cs = np.column_stack(ghi); self.zen = np.column_stack(zen); self.azi = np.column_stack(azi)
        self.times = times
        self.dni_extra = pvlib.irradiance.get_extra_radiation(times).to_numpy()

    def _pv_power(self, ghi, temp):
        import pvlib
        out = np.zeros_like(ghi)
        for i in range(ghi.shape[1]):
            erbs = pvlib.irradiance.erbs(ghi[:, i], self.zen[:, i], self.times)
            poa = pvlib.irradiance.get_total_irradiance(30, 180, self.zen[:, i], self.azi[:, i],
                                                        erbs["dni"], ghi[:, i], erbs["dhi"],
                                                        dni_extra=self.dni_extra, model="isotropic")
            g = np.nan_to_num(np.asarray(poa["poa_global"], float))
            tcell = temp[:, i] + 0.03 * g
            out[:, i] = g / 1000.0 * (1 - 0.004 * (tcell - 25.0))
        return np.clip(out, 0, None)

    # ------------------------------------------------------------------ latent fields
    def _wind_latent(self, rng):
        W = WIND
        c = _ar1(H, 1, W["tau_c"], rng)
        r = _ar1(H, 16, W["tau_r"], rng, self.Lw)
        e = _ar1(H, 16, W["tau_l"], rng)
        z = np.sqrt(W["w_common"]) * c + np.sqrt(W["w_regional"]) * r + np.sqrt(W["w_local"]) * e
        return z + W["season"] * self.season_w[:, None], c[:, 0]

    def _cloud_latent(self, rng):
        C = CLOUD
        cd = _ar1(DAYS, 1, C["tau_d"], rng); rd = _ar1(DAYS, 16, C["tau_d"], rng, self.Lc)
        zd = np.sqrt(C["w_common"]) * cd + np.sqrt(C["w_regional"]) * rd + np.sqrt(
            1 - C["w_common"] - C["w_regional"]) * rng.standard_normal((DAYS, 16))
        zh = zd[DOY_H] + C["hourly_sd"] * _ar1(H, 16, C["tau_h"], rng)
        return zh / np.sqrt(1 + C["hourly_sd"] ** 2), zd

    def _temp_anomaly(self, rng, cloud_daily):
        T = TEMP
        cd = _ar1(DAYS, 1, T["tau_d"], rng); rd = _ar1(DAYS, 16, T["tau_d"], rng, self.Lt)
        a = np.sqrt(T["w_common"]) * cd + np.sqrt(1 - T["w_common"]) * rd
        doy = np.arange(DAYS)
        summer_sign = np.where((doy >= 90) & (doy < 273), 1.0, -1.0)[:, None]
        w = T["cloud_coupling"]
        a = np.sqrt(1 - w ** 2) * a + w * summer_sign * cloud_daily
        sd = T["sd_mean"] + T["sd_amp"] * np.cos(2 * np.pi * (doy - 15) / 365.25)
        return a * sd[:, None]

    def _daily_temp(self, anom_daily, cloud_daily, shift_daily, cols=None):
        doy = np.arange(DAYS)
        st = STATES if cols is None else [STATES[c] for c in cols]
        tm = np.array([T_MEAN[s] for s in st]); ta = np.array([T_AMP[s] for s in st])
        clim = tm[None, :] - ta[None, :] * np.cos(2 * np.pi * (doy[:, None] - 20) / 365.25)
        dmean = clim + anom_daily + shift_daily
        damp = (TEMP["diurnal_mean"] + TEMP["diurnal_amp"] * np.cos(2 * np.pi * (doy - 182) / 365.25))[:, None]
        damp = damp * (1 + 0.25 * np.tanh(cloud_daily))
        return dmean, dmean + damp, clim, damp

    def _temperature(self, anom_daily, cloud_daily, shift_daily):
        dmean, tmax, clim, damp = self._daily_temp(anom_daily, cloud_daily, shift_daily)
        hourly = dmean[DOY_H] + damp[DOY_H] * np.cos(2 * np.pi * (HOUR[:, None] - 15) / 24.0)
        return hourly, dmean, tmax, clim

    # ------------------------------------------------------------------ calibration
    def _calibrate(self, rng, years):
        zs, cl, an = [], [], []
        for _ in range(years):
            z, _c = self._wind_latent(rng); zs.append(z)
            zh, zd = self._cloud_latent(rng); cl.append((zh, zd)); an.append(self._temp_anomaly(rng, zd))
        # wind: Beta mean such that the climatological CF matches the FLH target
        U = ndtr(np.vstack(zs))
        self.wind_lut = []
        for i, s in enumerate(STATES):
            target = ONSHORE_FLH[s] / H
            lo, hi = 0.05, 0.6
            for _ in range(30):
                m = 0.5 * (lo + hi)
                cf = _BetaLUT(m, WIND["kappa"])(U[:, i]).mean()
                lo, hi = (m, hi) if cf < target else (lo, m)
            self.wind_lut.append(_BetaLUT(0.5 * (lo + hi), WIND["kappa"]))
        lo, hi = 0.1, 0.8
        for _ in range(30):
            m = 0.5 * (lo + hi)
            cf = _BetaLUT(m, 1.8)(U[:, :3].mean(1)).mean()
            lo, hi = (m, hi) if cf < OFFSHORE_CF else (lo, m)
        self.off_lut = _BetaLUT(0.5 * (lo + hi), 1.8)
        # low-wind threshold on the daily-mean latent
        Dd = np.vstack([z.reshape(DAYS, 24, 16).mean(1) for z in zs])
        self.lw_theta = np.array([np.quantile(Dd[:, i], self.lw_mean[i] / DAYS) for i in range(16)])
        # temperature: state summer offset so that climatological hot days equal the 1990-2024 mean
        self.t_offset = np.zeros(16)
        for i in range(16):
            lo, hi = -6.0, 6.0
            for _ in range(30):
                m = 0.5 * (lo + hi)
                cnt = np.mean([(self._daily_temp(a[:, [i]], zd[:, [i]], m * self.summer_win[:, None], [i])[1] >= 30).sum()
                               for a, (_, zd) in zip(an, cl)])
                lo, hi = (m, hi) if cnt < self.hd_mean[i] else (lo, m)
            self.t_offset[i] = 0.5 * (lo + hi)
        # PV performance ratio to match FLH (average over four calibration years)
        acc = np.zeros(16)
        for (zh, zd), a in list(zip(cl, an))[:4]:
            temp, *_ = self._temperature(a, zd, self.t_offset[None, :] * self.summer_win[:, None])
            acc += self._pv_power(self.ghi_cs * self._clearness(zh), temp).sum(0)
        self.pv_pr = np.array([PV_FLH[s] for s in STATES]) / (acc / 4)

    def _clearness(self, zh):
        C = CLOUD
        m = C["k_mean"] + C["k_amp"] * np.cos(2 * np.pi * (self.doy_f - 182) / 365.25)
        u = ndtr(zh)
        k = np.empty_like(u)
        # Beta quantile with seasonally varying mean: 12 monthly LUTs
        month = np.minimum((self.doy_f / 30.42).astype(int), 11)
        for mo in range(12):
            sel = month == mo
            lut = _BetaLUT(float(m[sel].mean()), C["kappa"])
            k[sel] = lut(u[sel])
        return k

    # ------------------------------------------------------------------ one year
    def sample_year(self, rng, draw, return_latent=False):
        """``draw``: DataFrame indexed by state with REACT hazard columns for this year."""
        lw = draw.loc[STATES, "low_wind_days_mean"].to_numpy(float)
        hd = draw.loc[STATES, "hot_days_mean"].to_numpy(float)
        zw, zc = self._wind_latent(rng)
        Dd = zw.reshape(DAYS, 24, 16).mean(1)
        mu = np.zeros(16)
        for i in range(16):
            srt = np.sort(Dd[:, i]); L = np.clip(lw[i], 0.5, DAYS - 1.5)
            lo_i = int(np.floor(L)); fr = L - lo_i
            q = (1 - fr) * 0.5 * (srt[lo_i - 1] + srt[lo_i]) + fr * 0.5 * (srt[lo_i] + srt[lo_i + 1])
            mu[i] = self.lw_theta[i] - q
        zw = zw + mu[None, :]
        U = ndtr(zw)
        wind = np.column_stack([self.wind_lut[i](U[:, i]) for i in range(16)])
        off = {}
        for s, cols in (("Niedersachsen", ["Niedersachsen", "Bremen"]), ("Schleswig-Holstein", ["Schleswig-Holstein"]),
                        ("Mecklenburg-Vorpommern", ["Mecklenburg-Vorpommern"])):
            idx = [STATES.index(c) for c in cols]
            zo = 0.85 * zw[:, idx].mean(1) + 0.53 * _ar1(H, 1, 12.0, rng)[:, 0]
            off[s] = self.off_lut(ndtr(zo))
        zh, zd = self._cloud_latent(rng)
        anom = self._temp_anomaly(rng, zd)
        # hot-day matching: summer shift per state so that realised hot days equal the draw
        shift = np.zeros(16)
        for i in range(16):
            target = int(round(hd[i]))
            lo, hi = -8.0, 8.0
            for _ in range(28):
                m = 0.5 * (lo + hi)
                tmax = self._daily_temp(anom[:, [i]], zd[:, [i]], (self.t_offset[i] + m) * self.summer_win[:, None], [i])[1]
                cnt = int((tmax[:, 0] >= 30).sum())
                lo, hi = (m, hi) if cnt < target else (lo, m)
            shift[i] = self.t_offset[i] + 0.5 * (lo + hi)
        temp, tday, tmax, tclim = self._temperature(anom, zd, shift[None, :] * self.summer_win[:, None])
        k = self._clearness(zh)
        pv = self._pv_power(self.ghi_cs * k, temp) * self.pv_pr[None, :]
        pv = np.clip(pv, 0, 1)
        # load
        Ld = LOAD
        hdd = np.maximum(0, Ld["t_heat"] - tday) - np.maximum(0, Ld["t_heat"] - tclim)
        cdd = np.maximum(0, tday - Ld["t_cool"]) - np.maximum(0, tclim - Ld["t_cool"])
        lf = (1 + Ld["eta_heat"] * hdd + Ld["eta_cool"] * cdd)[DOY_H]
        load_factor = self.load_shape[:, None] * lf
        out = dict(wind=wind, offshore=off, pv=pv, temp=temp, tmax_daily=tmax, tday=tday,
                   load_factor=load_factor, wind_common=zc, lw_shift=mu, t_shift=shift)
        return out


class ExternalWeather:
    """Historical replay from external hourly data.

    Expected files in ``folder`` (one row per hour, UTC, 8760 or 8784 rows per year, one column per
    state using REACT state names or the two/three-letter abbreviations):
      wind_cf_<year>.csv, pv_cf_<year>.csv, temp_<year>.csv, load_<year>.csv [MW or shape]
    and optionally offshore_cf_<year>.csv with columns NI, SH, MV. Missing years are skipped.
    """

    def __init__(self, folder):
        self.folder = folder
        self.years = sorted({int(f.split("_")[-1][:4]) for f in os.listdir(folder) if f.startswith("wind_cf_")})

    def _read(self, kind, year):
        p = os.path.join(self.folder, f"{kind}_{year}.csv")
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p, index_col=0)
        inv = {v: k for k, v in ABBR.items()}
        df = df.rename(columns=lambda c: inv.get(c, c))
        return df.iloc[:H]

    def sample_year(self, rng, draw=None, year=None):
        y = year if year is not None else int(rng.choice(self.years))
        w, p, t, l = (self._read(k, y) for k in ("wind_cf", "pv_cf", "temp", "load"))
        temp = t[STATES].to_numpy(float)
        tday = temp.reshape(DAYS, 24, 16).mean(1); tmax = temp.reshape(DAYS, 24, 16).max(1)
        lf = l[STATES].to_numpy(float); lf = lf / lf.mean(0, keepdims=True)
        off = self._read("offshore_cf", y)
        offd = {"Niedersachsen": off["NI"].to_numpy(float), "Schleswig-Holstein": off["SH"].to_numpy(float),
                "Mecklenburg-Vorpommern": off["MV"].to_numpy(float)} if off is not None else None
        return dict(wind=w[STATES].to_numpy(float), offshore=offd, pv=p[STATES].to_numpy(float), temp=temp,
                    tmax_daily=tmax, tday=tday, load_factor=lf, wind_common=None, year=y)
