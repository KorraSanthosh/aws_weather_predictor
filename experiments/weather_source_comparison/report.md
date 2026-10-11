# Which weather forecast source gives the best next-day Delhi PM2.5 prediction?

**RETROSPECTIVE experiment.** Winters 2024-25 and 2025-26 were examined in earlier experiments, so this is not an untouched test. Nothing here changes the production model, default version, dashboard, Lambda, checksums or the frozen winter 2026-27 protocol; nothing was committed, pushed or deployed. Machine-readable results: `results.json`. Declared protocol (written before any result): `config.json`. Rerun: `python experiments/weather_source_comparison/run_comparison.py` then `report.py`.

## 1. Answer in short

* **No source reliably beats the current source. Retain it.** The live Lambda calls Open-Meteo without a `models` parameter (its default `best_match` blend). Value-matching the archives shows that blend was **identical to DWD ICON in winter 2024-25 (98.7% of hours) and identical to ECMWF IFS HRES in winter 2025-26 (100.0% of hours)**. So 'the current production source' is IFS HRES, and only winter 2025-26 is a valid test against it.
* **Winter 2025-26, matched dates (n = 104), current source vs the field (Hybrid MAE µg/m³, lower is better):** production default (best_match) 28.66; NOAA GFS 0.11° 28.79; DWD ICON 30.12; ECMWF IFS 0.25° 27.90; ECMWF IFS HRES 28.66; ECMWF AIFS 0.25° 28.54. Best point estimate: IFS 0.25° (-0.76 µg/m³, -2.6%), but its 95% interval [-2.32, +1.08] (7-day blocks) and [-2.46, +1.40] (14-day) includes zero and its exact-category accuracy is -1.9 points lower. That is not evidence of an improvement.
* **The large two-winter gaps are a reference artefact, not a source effect.** Pooling both winters shows IFS 0.25° -9.0% and GFS -8.6% versus 'production_default' with intervals excluding zero, but in 2024-25 'production_default' *was ICON*, which is the weakest source in that winter. These gaps say "GFS and IFS 0.25° beat ICON in 2024-25", not "they beat what the Lambda uses today". The pre-declared rule would have mechanically recommended IFS 0.25° on this pooled set; I did not apply it because its reference condition (one constant production source) is false, and I say so rather than hide it.
* **Not manufactured:** the production-default row reproduces the earlier published V1/V2/Hybrid numbers exactly for 2025-26 and for the two-winter pooled set (Hybrid 37.21 µg/m³ pooled), so these are the same pipeline with only the weather input swapped.

## 2. What the project actually does (Step 1, from the code)

| item | fact | where |
|---|---|---|
| live weather source | Open-Meteo Forecast API, **no `models` parameter** (default `best_match`) | `core/weather.py::fetch_forecast_weather`, `backend/forecast_handler.py` |
| coordinates | `LAT`/`LON` not set in `template.yaml` → defaults 28.6139, 77.2090 | handler defaults |
| variables | temperature_2m, relative_humidity_2m, wind_speed_10m, wind_direction_10m, precipitation, surface_pressure | `core/weather.py::HOURLY_VARS` |
| daily aggregation | tomorrow's 24 hourly values (Asia/Kolkata day): mean temp/RH/wind/pressure, summed precipitation, mean wind vector (u, v); an hour missing any variable is dropped; < 12 hours → no value | `hourly_to_daily_weather` (reused unchanged here) |
| issue time / horizon | `cron(30 17 * * ? *)` = 17:30 UTC = **23:00 IST**; target = next calendar day (1 day); PM2.5 inputs up to 22:59 IST | `template.yaml` |
| today's weather (wx0) | read from the same live response; no analysis archive exists → stays ERA5 for every source here | – |
| earlier historical evaluation | `weather_prevruns_2024_2026.json` (Previous Runs, lead 1 and 2, a fixed point 28.57, 77.18) used for the forecast-training experiment; training on ERA5 | `model/forecast_training_experiment.py` |
| archives available | Previous Runs API (`<var>_previous_dayN`); Historical Forecast API (analysis-like stitched first hours: **excluded** as input); Single Runs (non-HRES only from 2026-04-02: no winter coverage) | `fetch_sources.py`, `coverage.json` |

## 3. Archive coverage (verified before downloading; `coverage.json`)

| source (Open-Meteo id) | first complete lead-1 hour | complete lead-1 days | share of hours complete (2024-01-01 → 2026-02-16) | days in the 2024-25 test window | days in the 2025-26 test window |
|---|---|---|---|---|---|
| production default (best_match) | 2024-01-19T17:00 | 759 | 97.6% | 124 / 124 | 104 / 104 |
| NOAA GFS 0.11° | 2024-01-19T17:00 | 759 | 97.6% | 124 / 124 | 104 / 104 |
| DWD ICON | 2024-01-19T17:00 | 759 | 97.6% | 124 / 124 | 104 / 104 |
| ECMWF IFS 0.25° | 2024-02-29T23:00 | 718 | 92.3% | 124 / 124 | 104 / 104 |
| ECMWF IFS HRES | 2025-10-01T05:00 | 139 | 17.8% | 0 / 124 | 104 / 104 |
| ECMWF AIFS 0.25° | 2025-02-18T05:00 | 364 | 46.8% | 0 / 124 | 104 / 104 |

Units (from the API responses): °C, %, km/h, degrees, mm, hPa. Timestamps are Asia/Kolkata hourly. Missing values are left missing (never filled); a day with fewer than 12 complete hours is dropped, exactly as in production.

* **Only winter 2025-26 can test all six sources; 2024-25 can test four** (default, GFS, ICON, IFS 0.25°). IFS HRES lead-1 starts 2025-10-01; AIFS 2025-02-18. Because IFS HRES and AIFS cover one winter, 'consistent across winters' cannot be shown for them.
* **Forecast lead**: `previous_day1` = value forecast about 24 h before the valid hour (primary, matches the live 23:00 IST → next-day setting) and `previous_day2` (48 h, strictly available). Open-Meteo does not publish the issue hour of each `previous_dayN` value, and Single Runs cannot verify it for these winters. The 12 UTC ECMWF run is published roughly 7–8 hours after initialisation, i.e. after the 17:30 UTC cutoff, so lead 1 may for some late hours of the target day lean on information a few hours newer than the Lambda would have. Lead 2 removes this doubt and is reported as a robustness arm; at lead 2 IFS 0.25° (−1.1 %) and AIFS (−1.7 %) are nominally ahead and nothing is separable from the current source (section 4).
* **GFS data-quality flag:** GFS `relative_humidity_2m` averages 32 % in the archive against 64 % in the default blend and ERA5 (bias about −37 points on the test days), and its temperature is about +3.8 °C warm. Those weather errors are large (section 7) yet GFS still predicts PM2.5 well in 2024-25, a reminder that weather-forecast skill and PM2.5 skill are different things here.

## 4. Primary result: winter 2025-26, current source (IFS HRES) as reference

Arm A (**input swap**): the same frozen, ERA5-trained V1/V2/Hybrid models; only the next-day weather input changes with the source. Identical 104 dates, same PM2.5 truth, live-view features (hours ≤ 22 IST).

| source | days | MAE | RMSE | R² | exact category | within one | Δ MAE vs production (µg/m³) | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ exact-cat (pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production default (best_match) | 104 | 28.66 | 38.8 | 0.70 | 72.1% | 99.0% | 0 (reference) | – | – | – | – |
| NOAA GFS 0.11° | 104 | 28.79 | 39.0 | 0.69 | 66.3% | 100.0% | +0.13 | +0.4% | [-2.17, +2.43] | [-2.28, +2.32] | -5.8 |
| DWD ICON | 104 | 30.12 | 39.9 | 0.68 | 76.0% | 99.0% | +1.46 | +5.1% | [-1.50, +5.01] | [-2.07, +5.59] | +3.8 |
| ECMWF IFS 0.25° | 104 | 27.90 | 38.6 | 0.70 | 70.2% | 99.0% | -0.76 | -2.6% | [-2.32, +1.08] | [-2.46, +1.40] | -1.9 |
| ECMWF IFS HRES | 104 | 28.66 | 38.8 | 0.70 | 72.1% | 99.0% | 0 (identical predictions) | – | – | – | – |
| ECMWF AIFS 0.25° | 104 | 28.54 | 39.4 | 0.69 | 70.2% | 100.0% | -0.12 | -0.4% | [-2.48, +2.50] | [-2.58, +2.77] | -1.9 |
| persistence (tomorrow = today) | 104 | 41.60 | 53.6 | 0.42 | 61.5% | 96.2% | – | – | – | – | – |

Hybrid shown. V1 and V2 tables are in `results.json`. 'production default' and 'IFS HRES' predictions are bit-for-bit identical in this winter (the default blend resolves to HRES), which is itself a result: the live system is IFS HRES.


**sudden-change days (|Δ| ≥ 30 % vs today), n = 29**

| source | Hybrid MAE | Δ vs production | 95% CI (7-d) |
|---|---|---|---|
| production default (best_match) | 45.32 | – | – |
| NOAA GFS 0.11° | 44.95 | -0.37 | [-4.33, +4.14] |
| DWD ICON | 47.09 | +1.77 | [-1.20, +4.74] |
| ECMWF IFS 0.25° | 44.33 | -0.99 | [-3.18, +2.01] |
| ECMWF IFS HRES | 45.32 | – | – |
| ECMWF AIFS 0.25° | 44.16 | -1.16 | [-5.12, +3.64] |

**stable days, n = 75**

| source | Hybrid MAE | Δ vs production | 95% CI (7-d) |
|---|---|---|---|
| production default (best_match) | 22.22 | – | – |
| NOAA GFS 0.11° | 22.54 | +0.32 | [-3.00, +3.31] |
| DWD ICON | 23.56 | +1.34 | [-2.35, +6.03] |
| ECMWF IFS 0.25° | 21.55 | -0.67 | [-2.85, +1.63] |
| ECMWF IFS HRES | 22.22 | – | – |
| ECMWF AIFS 0.25° | 22.50 | +0.28 | [-2.58, +3.20] |

### Lead 2 (strictly available forecasts), same dates

| source | days | MAE | RMSE | R² | exact category | within one | Δ MAE vs production (µg/m³) | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ exact-cat (pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production default (best_match) | 104 | 28.92 | 39.4 | 0.69 | 71.2% | 99.0% | 0 (reference) | – | – | – | – |
| NOAA GFS 0.11° | 104 | 29.20 | 39.1 | 0.69 | 64.4% | 100.0% | +0.28 | +1.0% | [-1.46, +2.31] | [-1.45, +2.13] | -6.7 |
| DWD ICON | 104 | 29.59 | 39.6 | 0.68 | 76.9% | 99.0% | +0.67 | +2.3% | [-2.14, +3.73] | [-2.57, +4.30] | +5.8 |
| ECMWF IFS 0.25° | 104 | 28.62 | 39.5 | 0.69 | 69.2% | 99.0% | -0.31 | -1.1% | [-2.38, +1.81] | [-2.54, +2.08] | -1.9 |
| ECMWF IFS HRES | 104 | 28.92 | 39.4 | 0.69 | 71.2% | 99.0% | 0 (identical predictions) | – | – | – | – |
| ECMWF AIFS 0.25° | 104 | 28.43 | 39.7 | 0.68 | 68.3% | 100.0% | -0.49 | -1.7% | [-2.89, +2.04] | [-2.89, +2.39] | -2.9 |
| persistence (tomorrow = today) | 104 | 41.60 | 53.6 | 0.42 | 61.5% | 96.2% | – | – | – | – | – |

## 5. Four-source set over both winters (reference = whatever the default was)

Valid as a comparison of GFS / ICON / IFS 0.25° with the **archived default**, but the default was ICON in 2024-25 and HRES in 2025-26, so this is **not** a test against the current source.

| source | days | MAE | RMSE | R² | exact category | within one | Δ MAE vs production (µg/m³) | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ exact-cat (pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production default (best_match) | 228 | 37.21 | 51.7 | 0.65 | 70.2% | 98.7% | 0 (reference) | – | – | – | – |
| NOAA GFS 0.11° | 228 | 34.02 | 49.3 | 0.68 | 71.5% | 99.6% | -3.18 | -8.6% | [-5.71, -0.79] | [-5.85, -0.56] | +1.3 |
| DWD ICON | 228 | 37.89 | 52.1 | 0.64 | 71.9% | 98.7% | +0.69 | +1.8% | [-0.67, +2.26] | [-0.93, +2.59] | +1.8 |
| ECMWF IFS 0.25° | 228 | 33.86 | 49.0 | 0.68 | 72.8% | 99.1% | -3.35 | -9.0% | [-4.95, -1.59] | [-4.94, -1.49] | +2.6 |
| persistence (tomorrow = today) | 228 | 46.59 | 64.4 | 0.45 | 60.5% | 96.5% | – | – | – | – | – |

Winter by winter:

| source | 2024-25 MAE (n=124) | Δ vs prod, 95% CI (7-d) | 2025-26 MAE (n=104) | Δ vs prod, 95% CI (7-d) |
|---|---|---|---|---|
| production default (best_match) | 44.38 | – | 28.66 | – |
| NOAA GFS 0.11° | 38.41 | -5.96 (-13.4%) [-9.75, -2.30] | 28.79 | +0.13 (+0.4%) [-2.17, +2.43] |
| DWD ICON | 44.41 | +0.04 (+0.1%) [-0.19, +0.31] | 30.12 | +1.46 (+5.1%) [-1.50, +5.01] |
| ECMWF IFS 0.25° | 38.86 | -5.52 (-12.4%) [-7.67, -3.08] | 27.90 | -0.76 (-2.6%) [-2.32, +1.08] |

In 2024-25 the default (ICON) is the worst of the four, and GFS and IFS 0.25° are far better. In 2025-26 the picture reverses to 'no clear difference'. Winter consistency is therefore absent for GFS (+0.4 % in 2025-26) and weak for IFS 0.25° (−2.6 %, interval includes zero).


**sudden-change days (|Δ| ≥ 30 % vs today), n = 71**

| source | Hybrid MAE | Δ vs production | 95% CI (7-d) |
|---|---|---|---|
| production default (best_match) | 55.48 | – | – |
| NOAA GFS 0.11° | 53.61 | -1.87 | [-5.88, +2.00] |
| DWD ICON | 56.10 | +0.62 | [-0.61, +1.87] |
| ECMWF IFS 0.25° | 52.41 | -3.07 | [-6.53, +0.15] |

**stable days, n = 157**

| source | Hybrid MAE | Δ vs production | 95% CI (7-d) |
|---|---|---|---|
| production default (best_match) | 28.94 | – | – |
| NOAA GFS 0.11° | 25.17 | -3.78 | [-6.88, -0.83] |
| DWD ICON | 29.66 | +0.72 | [-1.04, +2.82] |
| ECMWF IFS 0.25° | 25.47 | -3.47 | [-5.49, -1.28] |

## 6. Is it the source or the training coverage?

Arm B retrains each source's models on that source's own archived forecasts where available (ERA5 before the archive starts: 837–1440 of ~1100–1450 origins are substituted, see `winters` in `results.json`), arm B_strict uses only origins with real forecasts (needs ≥ 365).


**Hybrid MAE decomposition, winter 2025-26 (current source valid)** (µg/m³; total = source effect + training effect):

| source | arm A (input swap) | arm B (source-trained) | source effect A_src − A_prod | training effect B_src − A_src | production's own B − A (control) | training effect net of control |
|---|---|---|---|---|---|---|
| production default (best_match) | 28.66 | 28.82 | +0.00 | +0.16 | +0.16 | +0.00 |
| NOAA GFS 0.11° | 28.79 | 28.34 | +0.13 | -0.45 | +0.16 | -0.61 |
| DWD ICON | 30.12 | 28.27 | +1.46 | -1.85 | +0.16 | -2.01 |
| ECMWF IFS 0.25° | 27.90 | 27.24 | -0.76 | -0.66 | +0.16 | -0.82 |
| ECMWF IFS HRES | 28.66 | 28.62 | +0.00 | -0.04 | +0.16 | -0.20 |
| ECMWF AIFS 0.25° | 28.54 | 28.73 | -0.12 | +0.19 | +0.16 | +0.03 |

**Hybrid MAE decomposition, both winters** (µg/m³; total = source effect + training effect):

| source | arm A (input swap) | arm B (source-trained) | source effect A_src − A_prod | training effect B_src − A_src | production's own B − A (control) | training effect net of control |
|---|---|---|---|---|---|---|
| production default (best_match) | 37.21 | 35.97 | +0.00 | -1.24 | -1.24 | +0.00 |
| NOAA GFS 0.11° | 34.02 | 33.71 | -3.18 | -0.32 | -1.24 | +0.92 |
| DWD ICON | 37.89 | 35.73 | +0.69 | -2.16 | -1.24 | -0.93 |
| ECMWF IFS 0.25° | 33.86 | 33.55 | -3.35 | -0.31 | -1.24 | +0.93 |

**B_strict, winter 2025-26 (real forecasts only, ~576–617 training origins):** production default (best_match) 35.06; NOAA GFS 0.11° 30.18; DWD ICON 29.47; ECMWF IFS 0.25° 28.21. Every source loses a lot relative to arm A, including the production source (it has about 600 instead of 1450 training origins and only one winter of weather-forecast behaviour). IFS HRES and AIFS lack the ≥ 365 origins and are reported as insufficient coverage, not run. Differences between sources inside arm B_strict are therefore dominated by training size, not by the forecast source, and are not used for ranking.


**Arm B (source-trained) vs the production source's own arm B, winter 2025-26, Hybrid:**

| source | days | MAE | RMSE | R² | exact category | within one | Δ MAE vs production (µg/m³) | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ exact-cat (pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production default (best_match) | 104 | 28.82 | 40.0 | 0.68 | 66.3% | 98.1% | 0 (reference) | – | – | – | – |
| NOAA GFS 0.11° | 104 | 28.34 | 39.0 | 0.69 | 69.2% | 100.0% | -0.48 | -1.7% | [-2.47, +1.72] | [-2.24, +1.96] | +2.9 |
| DWD ICON | 104 | 28.27 | 39.3 | 0.69 | 70.2% | 99.0% | -0.55 | -1.9% | [-1.72, +0.67] | [-1.68, +0.53] | +3.8 |
| ECMWF IFS 0.25° | 104 | 27.24 | 37.6 | 0.72 | 71.2% | 99.0% | -1.58 | -5.5% | [-3.44, +0.35] | [-2.76, -0.21] | +4.8 |
| ECMWF IFS HRES | 104 | 28.62 | 38.7 | 0.70 | 72.1% | 99.0% | -0.20 | -0.7% | [-2.50, +1.94] | [-2.19, +1.83] | +5.8 |
| ECMWF AIFS 0.25° | 104 | 28.73 | 39.0 | 0.69 | 71.2% | 100.0% | -0.09 | -0.3% | [-2.18, +1.93] | [-2.03, +1.52] | +4.8 |
| persistence (tomorrow = today) | 104 | 41.60 | 53.6 | 0.42 | 61.5% | 96.2% | – | – | – | – | – |

In arm B, IFS 0.25° is the lowest in every seed and −5.5 % against the production source's own arm B; the 14-day-block interval just excludes zero while the 7-day one does not, so it fails the declared both-blocks criterion. It also mixes a source effect with the fact that the IFS 0.25° archive has 576 real-forecast training origins against 617 for the default and a different ERA5-substitution count, and it is one winter. It is the reason IFS 0.25° is named for forward logging, not evidence for a switch.


Seed-to-seed noise for one fixed source and arm is about 0.5–0.9 µg/m³ in these tables (e.g. production arm A ranges 28.66–29.34 across the five seeds, see `seed_robustness`). 
**Reading:** the input-swap arm is the clean source test, and in it nothing reliably beats the current source in 2025-26. Training on forecasts moves MAE by up to about 2 µg/m³, and moves it for the production source too; seed noise alone is 0.5–0.9 µg/m³. Arm B for the production source is not a clean control: the 617 real-forecast origins it trains on are ICON-era default forecasts (the default was ICON until Sep 2025), while it is tested on HRES, and its explicit-HRES counterpart has only 14 real origins before the winter. So arm B comparisons against production are tilted toward the other sources, which is why arm B is reported but not used for the decision. In 2025-26 the non-production sources gain from source-specific training by 0.0–1.9 µg/m³ (ICON gains most while its source effect is +1.46 worse), i.e. much of what arm B shows is a training effect. The two-winter 'improvement' of GFS and IFS 0.25° comes from the source effect against ICON in 2024-25 (the decomposition's source column), not from different training coverage.

**Seed robustness (5 seeds, hybrid MAE, number of seeds in which the source beats production):**

* A:P6_L1: NOAA GFS 0.11° 2/5, DWD ICON 0/5, ECMWF IFS 0.25° 5/5, ECMWF IFS HRES 0/5, ECMWF AIFS 0.25° 3/5
* B:P6_L1: NOAA GFS 0.11° 5/5, DWD ICON 5/5, ECMWF IFS 0.25° 5/5, ECMWF IFS HRES 4/5, ECMWF AIFS 0.25° 5/5
* A:P4_L1: NOAA GFS 0.11° 5/5, DWD ICON 0/5, ECMWF IFS 0.25° 5/5

## 7. Weather-input error against ERA5 (independent of the PM2.5 model), winter 2025-26 dates

| source | temp bias / RMSE (°C) | RH bias / RMSE (%) | wind bias / RMSE (km/h) | pressure bias / RMSE (hPa) | precip RMSE (mm) |
|---|---|---|---|---|---|
| production default (best_match) | -0.11 / 0.50 | -1.7 / 3.3 | +0.07 / 0.73 | -0.51 / 0.57 | 1.86 |
| NOAA GFS 0.11° | +3.65 / 3.80 | -36.4 / 38.4 | +1.62 / 1.98 | -0.93 / 1.01 | 1.56 |
| DWD ICON | -0.68 / 1.16 | +6.5 / 9.6 | -1.22 / 1.59 | +0.02 / 0.26 | 1.72 |
| ECMWF IFS 0.25° | -0.44 / 0.81 | +1.1 / 3.9 | +0.80 / 1.29 | -0.56 / 0.62 | 2.67 |
| ECMWF IFS HRES | -0.11 / 0.50 | -1.7 / 3.3 | +0.07 / 0.73 | -0.51 / 0.57 | 1.86 |
| ECMWF AIFS 0.25° | -0.72 / 0.94 | +7.5 / 9.1 | +1.41 / 1.85 | -0.18 / 0.32 | 1.20 |

ERA5 is a reanalysis, not ground truth for Delhi, so these are mismatch figures, not forecast skill. IFS HRES has the smallest temperature, wind and RH errors; the PM2.5 model is trained on ERA5, so a source that looks more like ERA5 is not automatically a source that predicts PM2.5 better.

## 8. Decision rule (declared in `config.json` before running)

| set : source | Δ Hybrid MAE % | 95% CI 7-d | 95% CI 14-d | worst winter % | passes all five criteria | verdict |
|---|---|---|---|---|---|---|
| P4_L1:gfs_global | -8.6 | [-5.71, -0.79] | [-5.85, -0.56] | +0.4 | False | NOT VALID EVIDENCE vs the current source: production_default was a different model in 2024-25 |
| P4_L1:icon_global | +1.9 | [-0.67, +2.26] | [-0.94, +2.59] | +5.1 | False | NOT VALID EVIDENCE vs the current source: production_default was a different model in 2024-25 |
| P4_L1:ecmwf_ifs025 | -9.0 | [-4.95, -1.59] | [-4.94, -1.49] | -2.6 | True | NOT VALID EVIDENCE vs the current source: production_default was a different model in 2024-25 |
| P6_L1:gfs_global | +0.4 | [-2.17, +2.43] | [-2.28, +2.32] | +0.4 | False | does not reliably beat the current source |
| P6_L1:icon_global | +5.1 | [-1.50, +5.01] | [-2.06, +5.59] | +5.1 | False | does not reliably beat the current source |
| P6_L1:ecmwf_ifs025 | -2.6 | [-2.32, +1.08] | [-2.46, +1.40] | -2.6 | False | does not reliably beat the current source |
| P6_L1:ecmwf_ifs_hres | +0.0 | [+0.00, +0.00] | [+0.00, +0.00] | +0.0 | False | does not reliably beat the current source |
| P6_L1:ecmwf_aifs025 | -0.4 | [-2.48, +2.50] | [-2.58, +2.77] | -0.4 | False | does not reliably beat the current source |

`passes all five criteria` is the mechanical result of the rule; the verdict column applies the reference-validity check. IFS 0.25° passes mechanically on the two-winter set only because the reference there is ICON in 2024-25.

## 9. Answers to the five questions

1. **Best-performing source on matched dates:** IFS 0.25° has the lowest Hybrid MAE at lead 1 in winter 2025-26 (27.90 vs 28.66 for the current source) and also on the four-source two-winter set, but only the latter gap is statistically separable, and that reference was ICON.
2. **Meaningful and consistent across winters?** No. The advantage over the archived default is large in 2024-25 (ICON era) and small and indistinguishable from zero in 2025-26 (current source era). AIFS (−0.4 %), GFS (+0.4 %) and ICON (+5.1 %) show nothing against the current source. IFS HRES is the current source.
3. **Usable coverage:** table in section 3. GFS, ICON and the default cover both winters; IFS 0.25° both winters (from 2024-02-29); IFS HRES lead-1 one winter (from 2025-10-01); AIFS one winter (from 2025-02-18). Single Runs and Historical Forecast cannot supply live-equivalent inputs for these winters.
4. **Limitations:** two winters only (104 + 124 days, heavily autocorrelated; block bootstrap used); retrospective, both winters already seen; the default blend changed model between winters; lead-1 publication timing cannot be verified per hour (lead-2 arm included as a check); wx0 (today's weather) is ERA5 for every source, so the experiment only changes tomorrow's weather; GFS humidity looks mis-scaled in the archive; the forecast archive point (28.6139, 77.2090) is the live point, whereas the earlier experiments used 28.57, 77.18 (the default-source values matched the earlier archive's hybrid result, so the pipeline is comparable); training-set coverage differs by source in arm B; many comparisons were made, so a lone interval that excludes zero should be read cautiously.
5. **Next source for the winter 2026-27 forward evaluation:** keep the current default as the scored source. **Log, do not score, three alternatives alongside it** on every issuance: the ECMWF IFS 0.25° forecast (best point estimate at lead 1 in 5/5 seeds; second to AIFS at lead 2), the explicit `ecmwf_ifs` (HRES) forecast, which pins today's de-facto source and detects a repeat of the ICON → HRES switch of Oct 2025, and AIFS (cheap, one winter of archive). Pinning the model explicitly instead of relying on `best_match` is worth considering, but is a production change and is not proposed until you ask. No production change follows from this experiment.

## 10. Integrity

* Training targets are always before 15 Oct of the tested winter; windows end 15 Feb; the untouched period is never read (asserted in code and tests).
* No source-specific tuning; production hyper-parameters, seeds and eligibility rules reused.
* Production code was imported read-only; protected-file hashes are checked in `integrity.json`.
* No API key is used (public Open-Meteo endpoints); no `.env` file is read.
