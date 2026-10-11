# Fair historical winter back-test: V1 vs V2 vs Hybrid (focus: winter 2024-25) - re-run with the added OpenCity data

> **Status: RETROSPECTIVE back-test, not an untouched test.** Winters 2023-24, 2024-25 and 2025-26 were examined in earlier experiments (`EXPERIMENTS.md`) and 2019-2022 were used to choose the features and hyper-parameters. Nothing was tuned here: the protocol below was fixed before any result was computed. The only clean test is the prospective winter 2026-27 shadow test (`model/EVALUATION_PROTOCOL.md`). No model file, dashboard output or default was changed.

Data used: production data + gap-fill from opencity_hourly (data/daily_opencity.csv). Reproduce: `python model/winter_backtest.py --daily data/daily_opencity.csv --extra-source opencity_hourly --tag opencity` then `python -m pytest tests/test_backtest.py`. Machine-readable results sit next to this file (same name, `.json`).

## 1. Answer first

* **All 124 of the 124 days of the 15 Oct 2024 - 15 Feb 2025 window are scored** (2024-10-15 to 2025-02-15), each with a city PM2.5 from at least 3 stations.
* **All 124 scored days have hourly day-part data**, so the Hybrid takes V2's branch on every day (Hybrid = V2, identical predictions). The "days without hourly features" comparison for this winter is empty.
* **Winter 2024-25 result (primary, leak-free weather).** MAE: V1 (25 features) 47.48, V2 (35 features) 42.66, Hybrid 42.66, persistence 50.78, 7-day mean 67.38 µg/m³. Exact AQI-category accuracy: V1 65.3%, V2 66.9%, Hybrid 66.9%, persistence 59.7%. V2 (35 features) − V1 (25 features): **-4.81 µg/m³ (-10.1%)**, 95% CI [-8.19, -0.95] (7-day blocks) / [-8.39, -1.56] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks).
* **Which model is best for 2024-25?** V2 (35 features) has the lowest MAE, and V2's gain over V1 is **conclusive within this sample** (interval excludes 0 for both block lengths). Highest exact-category accuracy: V2 (35 features). Versus "tomorrow = today": **-8.12 µg/m³ (-16.0%)**, 95% CI [-13.93, -2.51] (7-day blocks) / [-13.24, -4.00] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks).
* **Across the 4 winters that could be scored** (2022-23, 2023-24, 2024-25, 2025-26; 399 days, 274 with day-parts): on days with day-parts V2 − V1 is **-5.84 µg/m³ (-13.3%)**, 95% CI [-8.46, -3.26] (7-day blocks) / [-8.20, -3.48] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks); on days without, **+4.05 µg/m³ (+9.0%)**, 95% CI [+1.72, +6.68] (7-day blocks) / [+1.39, +7.08] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks). Day-part evidence comes from 3 winter(s): 2023-24, 2024-25, 2025-26.
* **Hybrid.** Pooled over all scored days it beats V1: **-4.01 µg/m³ (-9.0%)**, 95% CI [-5.92, -2.16] (7-day blocks) / [-5.87, -2.17] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks). It equals V1 where day-parts are missing, so it keeps V2's gain and avoids V2's loss on those days.
* **Recommendation: the Hybrid is the leading candidate, but keep V1 as the default setting until the winter 2026-27 forward test.** The frozen rules (`model/EVALUATION_PROTOCOL.md`) allow promotion only on unseen data; these winters were seen. Weather here is a bound, not a real forecast. One historical winter does not guarantee anything about 2026-27.

## 2. What the project does today (inspection) and why the stored models cannot be used here

* **Training recipe (both versions).** `model/train.py`: XGBoost regressor on `log1p(PM_tomorrow) − log1p(PM_today)`, 800 trees max, learning rate 0.03, depth 4, subsample 0.85, colsample 0.85, min_child_weight 3, L2 2.0, early stopping (40 rounds) on the last 60 training rows, `random_state=42`. V1 = 25 features (`FEATURES_V1`), V2 = 35 (`FEATURES_V2`, adds 7 hourly day-part features and 8 binary flags). Feature definitions live in `core/features.py` and are shared by training, evaluation and the Lambda, so the same rule builds a feature everywhere.
* **Missing features.** Day-part features come from hourly data (`core/daily.py::daypart_stats`; needs >=16 hours). Where hours are absent they are NaN; XGBoost handles NaN natively; nothing is imputed. The Hybrid (`core/hybrid.py`) uses V2 only when all 7 day-part inputs are finite, else V1.
* **Valid PM2.5.** `data/daily.csv` holds a city value only for days with >=3 stations having >=16 valid hours (or a CPCB daily value). 13 station IDs exist in total; 2018-22 and Feb 2025 onward are hourly (OpenAQ); 2023-24 are CPCB daily files for four stations and have no hourly data, so no day-part features (until the OpenCity gap-fill, section 10).
* **Stored artifacts cannot reproduce this setup.** `model/<version>/model.json` was trained through 2026-10-07, i.e. on every winter, including this one. Using them on 2024-25 would be scoring training data. The back-test therefore re-trains frozen copies **in memory** with the unchanged recipe (nothing is written to `model/`).
* **The previous winter checks were not frozen.** `model/compare_versions.py` / `model/winter_check.py` use a walk-forward fit that is refitted each month, so later months of a winter are predicted by a model that has already trained on earlier months of the same winter, with tomorrow's observed weather. This is a legitimate live-like scheme but not what was asked here.

## 3. Data coverage and missing-data report

**Data sources in this run:** OpenAQ hourly files + CPCB daily files (as in production) + OpenCity/CPCB 15-minute files for 2024-01-01 to 2025-12-31 used only to fill days a station has no OpenAQ data for (Anand Vihar: 31,026 readings kept, 2024-01-01 to 2025-04-10, Punjabi Bagh: 38,566 readings kept, 2024-01-01 to 2025-04-10, R K Puram: 37,745 readings kept, 2024-01-01 to 2025-06-15, Rohini: 34,821 readings kept, 2024-01-01 to 2025-04-10). Rules and validation: `pipeline/convert_opencity.py` and section 10 of `winter_2024_25_backtest.md`.

**Window 2024-25** (2024-10-15 to 2025-02-15, 124 target days).

| Month | Stations reporting a valid daily value |
|---|---|
| 2024-10 | 17 day(s) with 4 station(s) |
| 2024-11 | 30 day(s) with 4 station(s) |
| 2024-12 | 31 day(s) with 4 station(s) |
| 2025-01 | 29 day(s) with 4 station(s), 2 day(s) with 3 station(s) |
| 2025-02 | 13 day(s) with 4 station(s), 2 day(s) with 3 station(s) |

| Station | Valid days in window | Source | Span |
|---|---|---|---|
| 10831 (Rohini) | 124 | hourly | 2024-10-15 to 2025-02-15 |
| 17 (R K Puram) | 123 | daily-file, hourly | 2024-10-15 to 2025-02-15 |
| 235 (Anand Vihar) | 122 | daily-file, hourly | 2024-10-15 to 2025-02-15 |
| 50 (Punjabi Bagh) | 123 | daily-file, hourly | 2024-10-15 to 2025-02-15 |

| Step of the eligibility ladder (same for every model) | Days in the 2024-25 window |
|---|---|
| Calendar target days 15 Oct 2024 - 15 Feb 2025 | 124 |
| ... with a city PM2.5 (>=3 stations) on the target day | 124 |
| ... also with city PM2.5 on the forecast-origin day, weather on both days, and an origin the 23:00 Lambda could use | 124 |
| ... of which with all 7 hourly day-part features | 124 |

**Exclusions in the 2024-25 window: none.** Every calendar day has a city PM2.5 from at least 3 stations.

**What would still improve this evaluation:** (1) archived weather forecasts issued before 23:00 for each day (an archive such as Open-Meteo's historical-forecast API; the host is not reachable from the build sandbox and nothing is stored), which would turn the primary regime into a realistic one.

**All winters (same ladder):**

| Winter | Calendar days | Scored days | Excluded | Scored days with day-parts | Training origins before 15 Oct | ...of which with day-parts | Status |
|---|---|---|---|---|---|---|---|
| 2018-19 | 124 | 35 | 89 | — | 38 | — | not run: only 38 training origins before 2018-10-15 (< 365) |
| 2019-20 | 124 | 109 | 15 | — | 164 | — | not run: only 164 training origins before 2019-10-15 (< 365) |
| 2020-21 | 124 | 9 | 115 | — | 314 | — | not run: only 314 training origins before 2020-10-15 (< 365) |
| 2021-22 | 124 | 64 | 60 | — | 334 | — | not run: only 334 training origins before 2021-10-15 (< 365) |
| 2022-23 | 124 | 47 | 77 | 2 | 459 | 444 | scored |
| 2023-24 | 124 | 124 | 0 | 45 | 740 | 446 | scored |
| 2024-25 | 124 | 124 | 0 | 124 | 1106 | 733 | scored |
| 2025-26 | 124 | 104 | 20 | 103 | 1454 | 1077 | scored |

A winter is scored only with >=365 training origins and >=30 scoreable days (fixed in advance); the early winters fail the first rule because the 3-station rule removes most 2018-20 days. Winter 2022-23 has only 47 scored days (almost all Jan-Feb 2023).

## 4. Method (fixed in advance)

* **Frozen fit.** For a winter starting 15 Oct Y, each model is trained once on forecast origins whose target date is before 15 Oct Y; there is no refit inside the window, and no window outcome is used for training, early stopping or any choice. A unit test scrambles the window outcomes and checks the predictions do not move.
* **Forecast-time information only.** Features for target D+1 read data up to D (tests scramble the target day and the future). Day-of features use hours <= 22 IST because the nightly Lambda runs at 23:00.
* **Weather (no archived forecasts available).**

| Regime | Tomorrow's weather | Status |
|---|---|---|
| **PRIMARY** `live23h_wx_persistence` | today's observed weather (a no-skill forecast) | leak-free; a no-skill stand-in for a forecast (not guaranteed pessimistic, see below) |
| Diagnostic `live23h_wx_observed` | tomorrow's OBSERVED weather | uses information live would not have; optimistic; **not live performance** |
| Legacy `offline_wx_observed` | tomorrow's observed weather, all 24 h of today | what earlier experiments used; only for reconciliation |

  Neither is claimed as live performance. **Correction after a later check with real archived forecasts** (`experiments/option_screen.md`, Open-Meteo Previous Runs API, 24 h ahead): for 2025-26 the real-forecast error lay between these two regimes, but for 2024-25 it was HIGHER than the primary regime (Hybrid 44.4 vs 42.7 µg/m³), because the forecast weather differs systematically from the archive weather the models are trained on. The Hybrid's lead over V1 held with real forecasts (pooled over both winters about 11% lower MAE, interval excluding zero).
* **Baselines.** Persistence: tomorrow = today's city PM2.5. Trailing 7-day mean: mean of the available daily values D-6..D.
* **Sudden-change days.** |tomorrow − today| / today >= 30%. "Direction" = share of those days where the forecast moved the right way from today (n/a for persistence).
* **Uncertainty.** Moving-block bootstrap (overlapping blocks of 7 days; 14 days as sensitivity; 5000 resamples, seed 0) on the paired difference of absolute errors, days placed on the calendar so gaps are respected. "Conclusive" = interval excludes 0 for both block lengths. Winters are scored separately; pooled rows place winters on a synthetic timeline so blocks never span two winters.

## 5. Winter-by-winter comparison

**Primary regime (leak-free weather).** MAE in µg/m³; `*` = interval excludes 0 (7-day blocks).

| Winter | Scored / calendar days | Days with day-parts | Persist. MAE | 7-day mean MAE | V1 MAE | V2 MAE | Hybrid MAE | V2 − V1 (µg/m³) [95% CI] | Hybrid − V1 [95% CI] | Exact category V1 / V2 / Hybrid / Persist. | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-19 | 35 / 124 | — | — | — | — | — | — | — | — | — | not run: only 38 training origins before 2018-10-15 (< 365) |
| 2019-20 | 109 / 124 | — | — | — | — | — | — | — | — | — | not run: only 164 training origins before 2019-10-15 (< 365) |
| 2020-21 | 9 / 124 | — | — | — | — | — | — | — | — | — | not run: only 314 training origins before 2020-10-15 (< 365) |
| 2021-22 | 64 / 124 | — | — | — | — | — | — | — | — | — | not run: only 334 training origins before 2021-10-15 (< 365) |
| 2022-23 | 47 / 124 | 2 | 49.0 | 53.31 | 42.46 | 48.92 | 42.48 | +6.46 [+2.3, +10.6] * | +0.01 [+0.0, +0.1] | 57% / 43% / 57% / 47% | scored |
| 2023-24 | 124 / 124 | 45 | 49.91 | 59.43 | 48.64 | 47.93 | 46.16 | -0.71 [-4.2, +2.8] | -2.48 [-5.3, -0.1] * | 65% / 62% / 67% / 60% | scored |
| 2024-25 | 124 / 124 | 124 | 50.78 | 67.38 | 47.48 | 42.66 | 42.66 | -4.81 [-8.2, -1.0] * | -4.81 [-8.2, -1.0] * | 65% / 67% / 67% / 60% | scored |
| 2025-26 | 104 / 124 | 103 | 41.6 | 48.47 | 36.42 | 29.56 | 29.7 | -6.86 [-11.3, -2.3] * | -6.71 [-11.1, -2.2] * | 65% / 71% / 71% / 62% | scored |

**Diagnostic regime (tomorrow's observed weather - optimistic, not live).**

| Winter | Scored / calendar days | Days with day-parts | Persist. MAE | 7-day mean MAE | V1 MAE | V2 MAE | Hybrid MAE | V2 − V1 (µg/m³) [95% CI] | Hybrid − V1 [95% CI] | Exact category V1 / V2 / Hybrid / Persist. | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-19 | 35 / 124 | — | — | — | — | — | — | — | — | — | not run: only 38 training origins before 2018-10-15 (< 365) |
| 2019-20 | 109 / 124 | — | — | — | — | — | — | — | — | — | not run: only 164 training origins before 2019-10-15 (< 365) |
| 2020-21 | 9 / 124 | — | — | — | — | — | — | — | — | — | not run: only 314 training origins before 2020-10-15 (< 365) |
| 2021-22 | 64 / 124 | — | — | — | — | — | — | — | — | — | not run: only 334 training origins before 2021-10-15 (< 365) |
| 2022-23 | 47 / 124 | 2 | 49.0 | 53.31 | 38.2 | 44.69 | 37.92 | +6.50 [+2.0, +11.2] * | -0.28 [-0.6, +0.0] | 53% / 38% / 53% / 47% | scored |
| 2023-24 | 124 / 124 | 45 | 49.91 | 59.43 | 44.2 | 42.8 | 41.21 | -1.39 [-4.6, +2.0] | -2.98 [-5.6, -0.6] * | 68% / 68% / 69% / 60% | scored |
| 2024-25 | 124 / 124 | 124 | 50.78 | 67.38 | 45.4 | 40.64 | 40.64 | -4.75 [-8.2, -1.2] * | -4.75 [-8.2, -1.2] * | 68% / 71% / 71% / 60% | scored |
| 2025-26 | 104 / 124 | 103 | 41.6 | 48.47 | 32.77 | 28.22 | 28.36 | -4.54 [-8.7, +0.0] | -4.41 [-8.5, +0.2] | 69% / 74% / 74% / 62% | scored |

**Pooled over the scored winters** (each from its own frozen fit; these winters share the same retrospective caveats):

| Regime | Set | Days | Persist. MAE | V1 MAE | V2 MAE | Hybrid MAE | V2 − V1 | Hybrid − V1 |
|---|---|---|---|---|---|---|---|---|
| primary | All matched days | 399 | 47.91 | 44.36 | 41.62 | 40.35 | **-2.74 µg/m³ (-6.2%)**, 95% CI [-4.99, -0.52] (7-day blocks) / [-5.13, -0.29] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | **-4.01 µg/m³ (-9.0%)**, 95% CI [-5.92, -2.16] (7-day blocks) / [-5.87, -2.17] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) |
| primary | Days WITH hourly day-parts | 274 | 47.66 | 44.01 | 38.17 | 38.17 | **-5.84 µg/m³ (-13.3%)**, 95% CI [-8.46, -3.26] (7-day blocks) / [-8.20, -3.48] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | **-5.84 µg/m³ (-13.3%)**, 95% CI [-8.46, -3.26] (7-day blocks) / [-8.20, -3.48] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) |
| primary | Days WITHOUT hourly day-parts | 125 | 48.45 | 45.14 | 49.2 | 45.14 | **+4.05 µg/m³ (+9.0%)**, 95% CI [+1.72, +6.68] (7-day blocks) / [+1.39, +7.08] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | identical predictions on every day (difference exactly 0) |
| diagnostic | All matched days | 399 | 47.91 | 40.88 | 38.55 | 37.3 | **-2.33 µg/m³ (-5.7%)**, 95% CI [-4.49, -0.11] (7-day blocks) / [-4.59, +0.11] (14-day blocks) — borderline (excludes 0 for only one block length) | **-3.59 µg/m³ (-8.8%)**, 95% CI [-5.41, -1.72] (7-day blocks) / [-5.35, -1.65] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) |
| diagnostic | Days WITH hourly day-parts | 274 | 47.66 | 40.53 | 35.31 | 35.31 | **-5.22 µg/m³ (-12.9%)**, 95% CI [-7.66, -2.65] (7-day blocks) / [-7.56, -2.68] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | **-5.22 µg/m³ (-12.9%)**, 95% CI [-7.66, -2.65] (7-day blocks) / [-7.56, -2.68] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) |
| diagnostic | Days WITHOUT hourly day-parts | 125 | 48.45 | 41.67 | 45.68 | 41.67 | **+4.01 µg/m³ (+9.6%)**, 95% CI [+1.48, +6.36] (7-day blocks) / [+1.58, +6.47] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | identical predictions on every day (difference exactly 0) |

## 6. Winter 2024-25 in detail

Scored target dates: 2024-10-15 to 2025-02-15 (124 days). Training: 1106 origins with targets 2018-01-05 to 2024-10-14 (733 with day-parts, the rest with day-parts NaN). Trees after early stopping: V1 217, V2 241. Sudden-change days: 42 of 124.

### 6.1 Primary regime (leak-free weather), all matched days

| Model | MAE µg/m³ | RMSE µg/m³ | R² | Exact category | Within ±1 category | Sudden-change days: MAE / exact cat / right direction |
|---|---|---|---|---|---|---|
| V1 (25 features) | 47.48 | 65.46 | 0.551 | 65.3% | 97.6% | 81.6 / 40% / 76% |
| V2 (35 features) | 42.66 | 60.33 | 0.618 | 66.9% | 98.4% | 66.5 / 55% / 79% |
| Hybrid | 42.66 | 60.33 | 0.618 | 66.9% | 98.4% | 66.5 / 55% / 79% |
| Persistence (tomorrow = today) | 50.78 | 72.18 | 0.454 | 59.7% | 96.8% | 99.3 / 24% / n/a |
| Trailing 7-day mean | 67.38 | 87.25 | 0.202 | 59.7% | 96.0% | 79.4 / 55% / 67% |

| MAE difference (negative = second model better) | µg/m³ | % | 95% CI, 7-day blocks | 95% CI, 14-day blocks | Verdict |
|---|---|---|---|---|---|
| V2 − V1 | -4.81 | -10.1% | [-8.19, -0.95] | [-8.39, -1.56] | CONCLUSIVE |
| Hybrid − V1 | -4.81 | -10.1% | [-8.19, -0.95] | [-8.39, -1.56] | CONCLUSIVE |
| V2 − Hybrid | 0.00 (identical) | 0.0% | [0, 0] | [0, 0] | identical |
| V1 − Persist. | -3.31 | -6.5% | [-9.15, +2.01] | [-8.25, +0.92] | inconclusive |
| Hybrid − Persist. | -8.12 | -16.0% | [-13.93, -2.51] | [-13.24, -4.00] | CONCLUSIVE |
| V2 − Persist. | -8.12 | -16.0% | [-13.93, -2.51] | [-13.24, -4.00] | CONCLUSIVE |
| V1 − 7-day mean | -19.90 | -29.5% | [-32.59, -10.69] | [-33.74, -10.21] | CONCLUSIVE |

### 6.2 Dates with and without hourly features

* **Days WITH hourly day-parts: 124 days** (= all matched days; table above).
* **Days WITHOUT hourly day-parts: 0 days.** Nothing to compare in this winter.

### 6.3 Diagnostic regime (tomorrow's observed weather - NOT live performance)

| Model | MAE µg/m³ | RMSE µg/m³ | R² | Exact category | Within ±1 category | Sudden-change days: MAE / exact cat / right direction |
|---|---|---|---|---|---|---|
| V1 (25 features) | 45.4 | 62.48 | 0.591 | 68.5% | 98.4% | 74.3 / 52% / 86% |
| V2 (35 features) | 40.64 | 57.19 | 0.657 | 71.0% | 99.2% | 59.0 / 60% / 90% |
| Hybrid | 40.64 | 57.19 | 0.657 | 71.0% | 99.2% | 59.0 / 60% / 90% |
| Persistence (tomorrow = today) | 50.78 | 72.18 | 0.454 | 59.7% | 96.8% | 99.3 / 24% / n/a |
| Trailing 7-day mean | 67.38 | 87.25 | 0.202 | 59.7% | 96.0% | 79.4 / 55% / 67% |

| MAE difference (negative = second model better) | µg/m³ | % | 95% CI, 7-day blocks | 95% CI, 14-day blocks | Verdict |
|---|---|---|---|---|---|
| V2 − V1 | -4.75 | -10.5% | [-8.23, -1.16] | [-8.20, -2.00] | CONCLUSIVE |
| Hybrid − V1 | -4.75 | -10.5% | [-8.23, -1.16] | [-8.20, -2.00] | CONCLUSIVE |
| V2 − Hybrid | 0.00 (identical) | 0.0% | [0, 0] | [0, 0] | identical |
| V1 − Persist. | -5.38 | -10.6% | [-10.57, +0.66] | [-10.05, +0.28] | inconclusive |
| Hybrid − Persist. | -10.14 | -20.0% | [-15.69, -4.20] | [-15.03, -5.20] | CONCLUSIVE |
| V2 − Persist. | -10.14 | -20.0% | [-15.69, -4.20] | [-15.03, -5.20] | CONCLUSIVE |
| V1 − 7-day mean | -21.98 | -32.6% | [-34.62, -11.61] | [-35.03, -11.27] | CONCLUSIVE |

## 7. Which model is best, and is the evidence conclusive?

* **Winter 2024-25 alone: V2 and the Hybrid beat V1, conclusively within this sample.** V2 − V1 = -4.81 µg/m³ (-10.1%), interval [-8.19, -0.95] (7-day) / [-8.39, -1.56] (14-day); exact-category accuracy V1 65.3%, V2 66.9%. "Conclusive" here is about this one retrospective winter; it does not transfer automatically to 2026-27.
* **What the data can say about V2.** With hourly day-parts it has the lower MAE (pooled: V2 (35 features) − V1 (25 features): **-5.84 µg/m³ (-13.3%)**, 95% CI [-8.46, -3.26] (7-day blocks) / [-8.20, -3.48] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks)); without them it is worse (pooled, primary: V2 (35 features) − V1 (25 features): **+4.05 µg/m³ (+9.0%)**, 95% CI [+1.72, +6.68] (7-day blocks) / [+1.39, +7.08] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks); in the optimistic diagnostic regime the same gap is +4.01 µg/m³, 95% CI [+1.48, +6.36], CONCLUSIVE). That is the reason for the Hybrid; the day-part evidence comes from 3 winter(s) (274 days).
* **Hybrid.** By construction it equals V1 on days without day-parts and V2 on days with them. Historically it scores 40.35 vs V1 44.36 pooled (primary); 44% of that gain comes from the latest winter (2025-26). Hybrid − V1 pooled: -4.01 µg/m³, 95% CI [-5.92, -2.16] - CONCLUSIVE.
* **Not shown by this back-test:** whether V2 or the Hybrid beats V1 in winter 2026-27, or how they behave with a real weather forecast.

## 8. Limitations

* Retrospective: these winters were examined before; V2's design was shaped by earlier looks. Treat all results as descriptive.
* Small samples with strong serial correlation (smog episodes): 124 days is only about 17 weeks of data and neighbouring days are not independent. Wide intervals are the honest result.
* No archived weather forecasts; both weather regimes are bounds, not live performance.
* Frozen fits get up to ~4 months stale. A live system refitted monthly would do a little better.
* The 2024 and Jan-Feb 2025 hourly values come from the OpenCity feed, other periods from OpenAQ. Both are CPCB stations and the IST reading reproduces the official daily files exactly, but the hourly sources are not identical, so day-part values may differ slightly across the join.
* Winter 2022-23 has few scored days, and winters before 2022-23 could not be scored at all (too little training data under the 3-station rule).
* Day-part evidence exists for 3 scored winter(s) only.

## 9. Recommendation

Keep **V1 (`v1_25_baseline`) as the default setting for now**, and treat the **Hybrid as the leading candidate**: in this retrospective back-test it beats V1 on pooled days with an interval that excludes zero, and it equals V1 wherever hourly day-parts are missing. The frozen promotion rule (`model/EVALUATION_PROTOCOL.md`: >=5% lower MAE, 95% interval below zero, category accuracy not more than 2 points lower, on >=90 matched unseen winter days) is what should change the default, via the winter 2026-27 shadow test. Nothing here was committed, pushed or deployed.
