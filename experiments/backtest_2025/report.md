# Kal Ki Hawa: train through 2024, evaluate calendar year 2025

**RETROSPECTIVE chronological backtest, not an untouched independent test.** 2025 data had been examined in earlier experiments of this project. No feature, hyper-parameter or model choice was tuned on 2025: the production recipe (same XGBoost parameters, feature sets, hybrid router and eligibility rules) was fitted once on targets ≤ 31 Dec 2024, frozen, then scored on 2025. Nothing in the production models, default model, dashboard, Lambda or checksums was changed; nothing was committed, pushed or deployed.

## 1. Result in short

* **Test coverage: 344 of 365 days of 2025**, not the full year. The other 21 days have no valid city PM2.5 (fewer than 3 stations) on the target day (11), on the origin day (9), or no usable 23:00 IST origin (1). No eligible day was lost for missing forecast weather.
* **Best on 2025: Hybrid** (MAE 19.57 µg/m³, RMSE 30.2, R² 0.87, exact category 66.3%), then V1 (MAE 22.55), persistence (24.90), 7-day average (29.23).
* **Hybrid vs V1 is convincing on these dates:** -2.99 µg/m³ (-13.2%), 95% CI [-4.61, -1.20] (7-day blocks) and [-4.77, -1.31] (14-day blocks), both excluding zero; the strict 48-hour-forecast arm agrees (-3.22, [-4.78, -1.59]); the Hybrid beat V1 in 5/5 random seeds. Hybrid vs V2 is negligible in practice (-0.06 µg/m³, interval [-0.16, -0.01]): the Hybrid is V2 on 338 of 344 days.
* **Where the gain comes from:** sudden-change days (106 days): persistence 49.8, V1 37.1, Hybrid 28.6. On stable days (238 days) the models do *not* beat persistence (persistence 13.8, V1 16.1, Hybrid 15.6). The model's value is in anticipating changes, not in calm weather. Monsoon months are a tie (all models ≈ 7–8 µg/m³).
* **Main validity threat:** the station network changed. Training years use about 3–6 stations a day (2023–24: median 4), 2025 uses a median of 9 (up to 10), so the city-mean target in 2025 is a different (broader) average than in training. This is not corrected and the effect is unquantified; see section 8.

## 2. Data audit (from the actual files; `audit.json`)

* **PM2.5 and weather exist only from 2018-01-04.** `data/daily_opencity.csv` runs 2018-01-04 to 2026-10-07; weather (`weather_daily.csv`) 2018-01-01 to 2026-10-08. **There is no PM2.5 for 2012–2017 anywhere in the project data**, so a 2012–2024 training set is impossible; training uses 2018-01-05 → 2024-12-31.
* Minimum-coverage rule: a day counts only if ≥ 3 stations report (`core.daily.MIN_STATIONS`); 25 Mar – 30 Jun 2020 is excluded (COVID).

| year | calendar days | days with city PM2.5 (≥ 3 stations) | share | stations/day (min / median / max) | months with data (Jan…Dec day counts) |
|---|---|---|---|---|---|
| 2012 | 366 | **0** | 0% | – | no data |
| 2013 | 365 | **0** | 0% | – | no data |
| 2014 | 365 | **0** | 0% | – | no data |
| 2015 | 365 | **0** | 0% | – | no data |
| 2016 | 366 | **0** | 0% | – | no data |
| 2017 | 365 | **0** | 0% | – | no data |
| 2018 | 365 | 97 | 26.6% | 3 / 4 / 6 | 12 12 0 0 0 0 0 9 20 21 16 7 |
| 2019 | 365 | 165 | 45.2% | 3 / 6 / 6 | 1 2 0 3 2 0 20 29 30 28 19 31 |
| 2020 | 366 | 98 | 26.8% | 3 / 5 / 6 | 31 29 23 0 0 0 8 1 0 4 2 0 |
| 2021 | 365 | 93 | 25.5% | 3 / 5 / 5 | 11 3 0 0 0 0 0 0 13 18 30 18 |
| 2022 | 365 | 86 | 23.6% | 3 / 4 / 6 | 4 1 5 25 25 5 0 0 8 13 0 0 |
| 2023 | 365 | 359 | 98.4% | 3 / 4 / 4 | 31 28 31 30 31 30 25 31 30 31 30 31 |
| 2024 | 366 | 366 | 100.0% | 3 / 4 / 4 | 31 29 31 30 31 30 31 31 30 31 30 31 |
| 2025 | 365 | 354 | 97.0% | 3 / 9 / 10 | 31 28 31 25 30 27 31 31 30 30 29 31 |

**Genuine, usable data exists for 2018–2025 only, and 2018–2022 are patchy** (86–165 usable days a year, whole seasons missing). Only 2023 and 2024 are close to complete, so most of the training signal is two years of data.

**Training set actually used:** 1184 origin days with targets 2018-01-05 → 2024-12-31 (per year: 2018: 72, 2019: 155, 2020: 87, 2021: 84, 2022: 63, 2023: 357, 2024: 366). Origins with all 7 day-part (hourly) inputs, which V2 needs: 811 (per year: 2018: 69, 2019: 151, 2020: 84, 2021: 80, 2022: 62, 2023: 0, 2024: 365). 2023 has none because that year comes from CPCB daily files without hours.

**Stations and sources by period** (`audit.json → station_days_by_source_and_year`, days with any reading):

* 2018–2022: OpenAQ stations (13 ids; 6358, 6931, 6932, 8239, 10831, 5613 contribute most days; 2018 early winter also 17, 50, 235, 236, 2503, 431). OpenAQ API files add 2021–22 for station 6932.
* 2022–2025: CPCB daily files (Anand Vihar 2022–25; R.K. Puram, Punjabi Bagh, Rohini, Anand Vihar 2023–24).
* 2024 (and Jan–Feb 2025): OpenCity hourly files for four stations (10831, 17, 235, 50).
* 2025: OpenAQ again, 10 stations (10831, 17, 235, 50, 5404, 5613, 6358, 6931, 6932, 8239) and an Anand Vihar CPCB daily file also exists for 2025; this is why 2025 has 9–10 stations a day.
* Weather: ERA5 daily means for training and for today's weather; archived Open-Meteo Previous Runs forecast for tomorrow's weather in 2025.

**Archived forecasts for 2025:** the Open-Meteo Previous Runs archive (default model, point 28.6139, 77.2090 as in the Lambda) has complete 24-hour-ahead (`previous_day1`) and 48-hour-ahead (`previous_day2`) values for every scoreable 2025 day (0 scoreable days lost). The archive starts 2024-01-19, so it cannot be used to train V1/V2 on forecasts before 2025; training uses ERA5, as production does. The default blend was ICON-identical until Sep 2025 and IFS-HRES-identical afterwards (previous experiment), so the forecast model changes inside the test year.

## 3. Protocol

* Train: targets ≤ 2024-12-31; V1 (25 features) and V2 (35 features) XGBoost, the production recipe, seed 42 (+ seeds 7, 123, 2024, 99 for robustness only). Frozen models saved in `models/` (`v1_seed42.json`, `v2_seed42.json`, `hybrid_router.json`, SHA-256 in `sha256.json`).
* Test: every 2025 day that is scoreable by the project's existing eligibility ladder (`winter_backtest.classify_target`: ≥ 3 stations on target and origin day, weather present, origin usable by the 23:00 IST Lambda) **and** has the archived forecast. All models are scored on exactly the same 344 days.
* Inputs at the 23:00 IST cutoff: today's PM2.5 and day-parts from readings up to 22:59 IST, lagged PM2.5 from earlier days, tomorrow's weather from the archived forecast (lead 1 primary; lead 2 = 48 h as a strictly-available check), today's weather from ERA5 (no analysis archive exists).
* Confidence intervals: moving-block bootstrap (7-day blocks, 14-day sensitivity), 5000 resamples, shared across models.

## 4. Primary result: 2025, one-day-ahead, 24-hour-old forecast weather

| model | days | MAE | MAE 95% CI | RMSE | R² | exact category | within one | MAE vs persistence |
|---|---|---|---|---|---|---|---|---|
| persistence (tomorrow = today) | 344 | 24.90 | [19.5, 30.5] | 39.8 | 0.78 | 58.1% | 96.5% | – |
| 7-day average | 344 | 29.23 | [22.5, 35.2] | 43.9 | 0.73 | 52.6% | 96.2% | -17.4% [-31.94, -1.82] |
| V1 (25-feature XGBoost) | 344 | 22.55 | [17.9, 26.9] | 35.0 | 0.83 | 61.6% | 97.1% | +9.4% [+1.68, +17.17] |
| V2 (35 features, experimental) | 344 | 19.63 | [15.6, 23.7] | 30.2 | 0.87 | 66.0% | 98.5% | +21.2% [+13.05, +27.96] |
| Hybrid (V2 if 7 day-parts, else V1) | 344 | 19.57 | [15.5, 23.6] | 30.2 | 0.87 | 66.3% | 98.5% | +21.4% [+13.33, +28.25] |

Differences on identical days:

| comparison (second minus first, same days) | Δ MAE µg/m³ | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ RMSE | Δ R² | Δ exact-category (pts) | CI excludes 0 (both) |
|---|---|---|---|---|---|---|---|---|
| Hybrid − V1 | -2.99 | -13.2% | [-4.61, -1.20] | [-4.77, -1.31] | -4.86 | +0.044 | +4.7 | True |
| V2 − V1 | -2.92 | -13.0% | [-4.55, -1.14] | [-4.71, -1.22] | -4.85 | +0.044 | +4.4 | True |
| Hybrid − V2 | -0.06 | -0.3% | [-0.16, -0.01] | [-0.16, -0.01] | -0.02 | +0.000 | +0.3 | True |
| Hybrid − persistence | -5.33 | -21.4% | [-7.92, -2.89] | [-7.98, -2.80] | -9.58 | +0.093 | +8.1 | True |
| V1 − persistence | -2.35 | -9.4% | [-4.70, -0.39] | [-4.78, -0.08] | -4.72 | +0.049 | +3.5 | True |
| Hybrid − 7-day avg | -9.66 | -33.1% | [-13.27, -5.56] | [-13.40, -4.95] | -13.75 | +0.141 | +13.7 | True |
| V1 − 7-day avg | -6.68 | -22.8% | [-10.50, -2.49] | [-10.29, -2.19] | -8.88 | +0.097 | +9.0 | True |
| 7-day avg − persistence | +4.33 | +17.4% | [+0.48, +7.55] | [+0.52, +7.41] | +4.17 | -0.048 | -5.5 | True |

### Strictly available 48-hour-old forecasts (robustness)

| model | days | MAE | MAE 95% CI | RMSE | R² | exact category | within one | MAE vs persistence |
|---|---|---|---|---|---|---|---|---|
| persistence (tomorrow = today) | 344 | 24.90 | [19.5, 30.5] | 39.8 | 0.78 | 58.1% | 96.5% | – |
| 7-day average | 344 | 29.23 | [22.5, 35.2] | 43.9 | 0.73 | 52.6% | 96.2% | -17.4% [-31.94, -1.82] |
| V1 (25-feature XGBoost) | 344 | 23.11 | [18.4, 27.6] | 35.4 | 0.83 | 61.0% | 97.4% | +7.2% [-0.81, +15.30] |
| V2 (35 features, experimental) | 344 | 19.94 | [15.9, 23.9] | 30.4 | 0.87 | 66.6% | 98.5% | +19.9% [+11.68, +27.04] |
| Hybrid (V2 if 7 day-parts, else V1) | 344 | 19.90 | [15.8, 23.9] | 30.4 | 0.87 | 66.6% | 98.5% | +20.1% [+11.89, +27.24] |

| comparison (second minus first, same days) | Δ MAE µg/m³ | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ RMSE | Δ R² | Δ exact-category (pts) | CI excludes 0 (both) |
|---|---|---|---|---|---|---|---|---|
| Hybrid − V1 | -3.22 | -13.9% | [-4.78, -1.59] | [-4.90, -1.64] | -4.99 | +0.045 | +5.5 | True |
| V2 − V1 | -3.18 | -13.7% | [-4.74, -1.54] | [-4.88, -1.58] | -4.98 | +0.045 | +5.5 | True |
| Hybrid − V2 | -0.04 | -0.2% | [-0.11, +0.00] | [-0.11, +0.00] | -0.01 | +0.000 | +0.0 | False |
| Hybrid − persistence | -5.00 | -20.1% | [-7.64, -2.59] | [-7.70, -2.46] | -9.35 | +0.091 | +8.4 | True |
| V1 − persistence | -1.79 | -7.2% | [-4.16, +0.16] | [-4.20, +0.45] | -4.36 | +0.045 | +2.9 | False |
| Hybrid − 7-day avg | -9.33 | -31.9% | [-13.02, -5.18] | [-13.19, -4.54] | -13.51 | +0.139 | +14.0 | True |
| V1 − 7-day avg | -6.12 | -20.9% | [-10.01, -1.89] | [-9.73, -1.55] | -8.52 | +0.094 | +8.4 | True |
| 7-day avg − persistence | +4.33 | +17.4% | [+0.48, +7.55] | [+0.52, +7.41] | +4.17 | -0.048 | -5.5 | True |

## 5. Sudden-change days (|tomorrow − today| ≥ 30% of today) and stable days

| subset | days | persistence MAE | V1 MAE | Hybrid MAE | Hybrid − V1 | 95% CI (7-d blocks) | Hybrid − persistence | 95% CI |
|---|---|---|---|---|---|---|---|---|
| sudden-change | 106 | 49.8 | 37.1 | 28.6 | -8.51 | [-11.54, -4.65] | -21.24 | [-26.81, -14.01] |
| stable | 238 | 13.8 | 16.1 | 15.6 | -0.53 | [-2.58, +1.62] | +1.75 | [-0.95, +4.57] |
| days Hybrid used V2 | 338 | 25.1 | 22.8 | 19.8 | -3.04 | [-4.71, -1.23] | -5.35 | [-7.96, -2.86] |

## 6. By season and month (Hybrid − V1; negative = Hybrid better)

| season | days | persistence MAE | V1 MAE | Hybrid MAE | Hybrid − V1 (µg/m³) | 95% CI (7-d blocks) | V1 exact | Hybrid exact |
|---|---|---|---|---|---|---|---|---|
| winter (Dec-Feb) | 90 | 45.1 | 38.6 | 33.9 | -4.7 | [-8.58, -0.22] | 62.2% | 67.8% |
| pre-monsoon (Mar-May) | 81 | 21.6 | 21.4 | 16.0 | -5.4 | [-7.99, -3.45] | 49.4% | 59.3% |
| monsoon (Jun-Sep) | 116 | 7.7 | 7.9 | 7.4 | -0.5 | [-1.37, +0.35] | 69.0% | 74.1% |
| post-monsoon (Oct-Nov) | 57 | 32.6 | 28.7 | 26.8 | -2.0 | [-8.62, +4.54] | 63.2% | 57.9% |

Seasons: winter Dec–Feb (Dec 2025 plus Jan–Feb 2025), pre-monsoon Mar–May, monsoon Jun–Sep, post-monsoon Oct–Nov. Intervals with few days are wide.

| month | days | persistence MAE | V1 MAE | Hybrid MAE | Hybrid − V1 (µg/m³) | 95% CI (7-d blocks) | V1 exact | Hybrid exact |
|---|---|---|---|---|---|---|---|---|
| Jan | 31 | 52.2 | 47.4 | 46.9 | -0.5 | [-5.31, +8.54] | 54.8% | 61.3% |
| Feb | 28 | 32.1 | 32.5 | 23.1 | -9.4 | [-16.58, -5.59] | 60.7% | 64.3% |
| Mar | 31 | 17.1 | 18.3 | 12.3 | -5.9 | [-9.08, -3.09] | 51.6% | 61.3% |
| Apr | 22 | 16.9 | 16.2 | 12.5 | -3.8 | [-4.75, +0.97] | 54.5% | 63.6% |
| May | 28 | 30.3 | 29.0 | 22.9 | -6.1 | [-12.01, -2.32] | 42.9% | 53.6% |
| Jun | 25 | 9.5 | 9.2 | 8.6 | -0.6 | [-3.08, +2.36] | 72.0% | 80.0% |
| Jul | 30 | 8.4 | 7.3 | 5.9 | -1.5 | [-2.91, -0.39] | 63.3% | 70.0% |
| Aug | 31 | 7.1 | 6.8 | 5.9 | -0.9 | [-1.74, +0.49] | 64.5% | 71.0% |
| Sep | 30 | 6.3 | 8.4 | 9.3 | +0.9 | [-0.26, +1.99] | 76.7% | 76.7% |
| Oct | 29 | 24.8 | 23.1 | 17.6 | -5.6 | [-7.92, +0.05] | 72.4% | 62.1% |
| Nov | 28 | 40.7 | 34.6 | 36.4 | +1.8 | [-2.87, +12.04] | 53.6% | 53.6% |
| Dec | 31 | 49.8 | 35.2 | 30.7 | -4.6 | [-6.97, -1.38] | 71.0% | 77.4% |

## 7. Seeds and reference arm

Seed-to-seed Hybrid MAE: seed 42: 19.57 (V1 22.55), seed 7: 19.46 (V1 22.28), seed 123: 19.99 (V1 22.38), seed 2024: 19.66 (V1 22.29), seed 99: 19.68 (V1 22.58). Hybrid beat V1 in 5/5 seeds.

**Reference only, NOT a realistic forecast:** with the *observed* (ERA5) weather of the target day, which the Lambda cannot have, on 344 days: persistence (tomorrow = today) 24.90, 7-day average 29.23, V1 (25-feature XGBoost) 21.24, V2 (35 features, experimental) 18.76, Hybrid (V2 if 7 day-parts, else V1) 18.72. This is shown only to size the cost of forecast-weather error (18.72 observed vs 19.57 forecast, Hybrid). It is not used for any conclusion.

## 8. Limitations and uncertainties

* **Not an untouched test.** 2025 had been looked at before; I did not tune on it here, but earlier design choices (feature sets, hybrid rule, hyper-parameters) were made by the project while 2025 data existed in the repository.
* **Coverage:** 344/365 days, with no 2012–2017 data and patchy 2018–2022; the model effectively learns from ~2 near-complete years.
* **Station-network shift (largest threat to validity):** training city means come from ~4 stations a day (2023–24), the 2025 target from ~9–10, and the day-part features come from different station sets in different years. Part of the 2025 error, and of the ranking, may reflect this definition change. I did not rebuild a 4-station 2025 target to quantify it.
* **Forecast weather changes model mid-year:** the default blend resolved to ICON until Sep 2025 and IFS HRES afterwards. Training uses ERA5 for weather; the 2025 inputs are archived forecasts, so a train/test weather mismatch exists, as in production.
* **Publication timing:** Open-Meteo does not publish the run hour of each `previous_dayN` value; the 48-hour arm is the strictly-available check and gives the same conclusion.
* Today's weather (wx0) is ERA5 because no analysis archive exists.
* Persistence is strong in 2025 (R² 0.78, MAE 13.8 on stable days), so the margin over it comes almost entirely from sudden-change days.
* Many comparisons are reported (months, seasons, subsets); single intervals that exclude zero in sub-groups should be read cautiously.

## 9. Conclusion

On the 344 eligible days of 2025 the **Hybrid is the best model**, ahead of V1 by 2.99 µg/m³ MAE (13.2%), RMSE and R² improvements, and +4.7 points exact category accuracy. The advantage over V1 is statistically convincing under both bootstrap block lengths, under 48-hour forecasts, across all five seeds, and is concentrated on sudden-change days and clear in winter and pre-monsoon; post-monsoon favours the Hybrid in the point estimate but its interval includes zero, and in the monsoon there is no difference. The Hybrid is effectively V2 (Hybrid vs V2 differs by 0.06 µg/m³). Caveats above (retrospective, station-network shift, patchy early history) mean this supports the Hybrid as the leading candidate, not as a validated production claim; the winter 2026-27 forward evaluation remains the real test. No production change is proposed here.
