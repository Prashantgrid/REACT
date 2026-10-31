# REACT input data checks

PASS = within the stated bound or consistent; CHECK = needs a look.

## Hazard panel (DWD, 1990-2024)

- PASS: 560 rows, 16 states, years 1990-2024
- PASS: duplicate state-years: 0
- PASS: missing values: soil_moist_winter_mean: 16 (winter soil moisture missing only for 1990: True)
- PASS: hot_days_mean: range 0.00-30.88 (bounds 0-60)
- PASS: soil_moist_winter_mean: range 66.76-106.68 (bounds 20-150)
- PASS: p30_days_mean: range 0.00-5.11 (bounds 0-15)
- PASS: Rx1day_mm_mean: range 18.70-114.65 (bounds 10-120)
- PASS: Rx1day_mm_max: range 20.40-312.00 (bounds 15-350)
- PASS: Storm_Days_mean: range 1.00-40.00 (bounds 0-80)
- PASS: low_wind_days_mean: range 58.67-259.17 (bounds 0-366)
- PASS: station-maximum 1-day precipitation >= state-mean value in all rows (violations: 0)
- PASS: within-state outliers with |robust z| > 5: 24 (listed below; extremes are expected for heavy-precipitation and storm years and are not errors by themselves)
  - Baden-Württemberg 2019 soil_moist_winter_mean = 92.10 (robust z = -5.7)
  - Bremen 2019 soil_moist_winter_mean = 86.19 (robust z = -6.2)
  - Hamburg 2019 soil_moist_winter_mean = 78.82 (robust z = -6.9)
  - Hessen 2019 soil_moist_winter_mean = 86.41 (robust z = -6.3)
  - Mecklenburg-Vorpommern 2019 soil_moist_winter_mean = 73.60 (robust z = -5.1)
  - Niedersachsen 2019 soil_moist_winter_mean = 82.50 (robust z = -6.1)
  - Nordrhein-Westfalen 2019 soil_moist_winter_mean = 88.99 (robust z = -5.6)
  - Rheinland-Pfalz 2019 soil_moist_winter_mean = 90.94 (robust z = -5.8)
  - Saarland 2019 soil_moist_winter_mean = 92.12 (robust z = -5.3)
  - Schleswig-Holstein 2019 soil_moist_winter_mean = 87.02 (robust z = -6.0)
  - Bayern 2002 p30_days_mean = 5.00 (robust z = 5.9)
  - Mecklenburg-Vorpommern 2011 p30_days_mean = 2.88 (robust z = 5.5)
  - Sachsen 2010 p30_days_mean = 4.07 (robust z = 6.9)
  - Berlin 2017 Rx1day_mm_mean = 114.65 (robust z = 8.3)
  - Mecklenburg-Vorpommern 2021 Rx1day_mm_mean = 58.70 (robust z = 5.8)
  - Niedersachsen 1998 Rx1day_mm_mean = 23.11 (robust z = -5.9)
  - Niedersachsen 2010 Rx1day_mm_mean = 50.68 (robust z = 5.9)
  - Nordrhein-Westfalen 2021 Rx1day_mm_mean = 58.67 (robust z = 6.3)
  - Sachsen 2002 Rx1day_mm_mean = 94.30 (robust z = 7.5)
  - Berlin 2006 Rx1day_mm_max = 130.00 (robust z = 5.5)
  - Berlin 2017 Rx1day_mm_max = 196.90 (robust z = 9.8)
  - Brandenburg 2021 Rx1day_mm_max = 167.60 (robust z = 5.1)
  - Sachsen 2002 Rx1day_mm_max = 312.00 (robust z = 16.1)
  - Sachsen 2018 Rx1day_mm_max = 154.90 (robust z = 5.5)

## Structural panel

- PASS: 16 states, unique
- PASS: missing values: 0
- PASS: negative values in non-signed columns: 0
- PASS: zero capacities: none
- PASS: national sum of onshore wind, MW: 67,402 (plausible 55,000-75,000)
- PASS: national sum of solar PV, MW: 115,306 (plausible 90,000-130,000)
- PASS: national sum of fossil thermal, MW: 69,900 (plausible 55,000-90,000)
- PASS: national sum of electricity consumption, TWh: 491 (plausible 430-560)
- PASS: national sum of electricity generation, TWh: 460 (plausible 400-600)
- PASS: national battery storage: 23,872 MWh
- PASS: r_i equals (generation - consumption)/max|net| (max deviation 0.0000)
- PASS: state areas sum to 357,683 km2
- PASS: HDI range 0.921-0.975
- PASS: total age-dependency ratio range 56-78
- PASS: GDP per capita range 32,837-76,910 EUR
- PASS: storage hours of average load: HH 0.09, HB 0.14, BE 0.16, NRW 0.33, ST 0.36, HE 0.37, RP 0.39, SL 0.44, BW 0.45, NI 0.46, MV 0.51, TH 0.52, BB 0.53, BY 0.58, SH 0.60, SN 0.62
- PASS: largest capacity-to-load ratios: Wind Mecklenburg-Vorpommern 5.7, Solar Mecklenburg-Vorpommern 6.8, Thermal Brandenburg 3.8

## MaStR storage aggregate (2025-10-01/storage_by_state.csv)

- PASS: 16 state rows
- PASS: battery net power <= gross power in every state
- PASS: battery energy/net power by state (h): SL 1.41, HB 1.45, TH 1.52, MV 1.54, NRW 1.58, RP 1.59, SH 1.64, HH 1.65, NI 1.65, HE 1.66, BY 1.68, SN 1.68, BB 1.70, BW 1.71, BE 1.71, ST 1.72
- CHECK: energy removed by the duration screen: BW 5 MWh, BY 19 MWh, BB 8 MWh, HB 11 MWh, HE 3 MWh, MV 16 MWh, NI 28 MWh, NRW 8 MWh, ST 8 MWh, TH 5 MWh
- PASS: national battery energy: raw 23,983 MWh; screened 23,872 MWh
- CHECK: non-battery storage energy (mainly pumped storage) above 5 GWh: BW 110,658 MWh, TH 20,531 MWh; national 148,396 MWh
- PASS: panel storage / screened registry energy: HH 1.00, BY 1.00, HE 1.00, BW 1.00, NI 1.00, HB 1.00, NRW 1.00, BE 1.00, SH 1.00, RP 1.00, SL 1.00, TH 1.00, BB 1.00, MV 1.00, SN 1.00, ST 1.00
- PASS: rank correlation of storage hours, panel vs registry: 1.00

