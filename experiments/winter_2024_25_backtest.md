# Fair historical winter back-test: V1 vs V2 vs Hybrid (focus: winter 2024-25)

> **Status: RETROSPECTIVE back-test, not an untouched test.** Winters 2023-24, 2024-25 and 2025-26 were examined in earlier experiments (`EXPERIMENTS.md`) and 2019-2022 were used to choose the features and hyper-parameters. Nothing was tuned here: the protocol below was fixed before any result was computed. The only clean test is the prospective winter 2026-27 shadow test (`model/EVALUATION_PROTOCOL.md`). No model file, dashboard output or default was changed.

> **Two runs, one protocol.** Sections 1-9 use the project's production dataset (`data/daily.csv`): only 78 of 124 days of the 2024-25 window had a city value and none had hourly data. After that report was written, 15-minute CPCB data for 2024-25 was added (section 10) as a gap-filler. With it **all 124 days are scored, each with day-part data**, and the V1-vs-V2 answer for 2024-25 changes: V2 (35 features) − V1 (25 features): **-4.81 µg/m³ (-10.1%)**, 95% CI [-8.19, -0.95] (7-day blocks) / [-8.39, -1.56] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks). Section 10 compares the runs. On the identical Oct-Dec dates the gain appears only once V2 has real day-part inputs (and more day-part rows in training); no separate control isolates those two causes. Both runs are retrospective.

Reproduce: `python model/winter_backtest.py` (about 20 s), then `python model/winter_backtest.py --daily data/daily_opencity.csv --extra-source opencity_hourly --tag opencity`, then `python -m pytest tests/test_backtest.py`. Machine-readable results: `experiments/winter_2024_25_backtest.json` and `..._opencity.json`.

## 1. Answer first (production-data run)

* **Only 78 of the 124 days of the 15 Oct 2024 - 15 Feb 2025 window could be scored** (2024-10-15 to 2024-12-31). From 1 Jan 2025 only one station (Anand Vihar) reported, below the project's 3-station rule for a city value, so those 46 days are excluded for every model (section 3). This is a data limit, not a modelling choice.
* **None of the 78 scored days has hourly day-part data** (Oct-Dec 2024 come from CPCB daily files). So the Hybrid takes V1's branch on every day: **Hybrid = V1, identical predictions**. V2 ran with its 7 day-part inputs missing (left as NaN, never imputed). The "days with hourly features" comparison for this winter is empty.
* **Winter 2024-25 result (primary, leak-free weather).** MAE: V1 (25 features) 49.05, V2 (35 features) 48.36, Hybrid 49.05, persistence 51.88, 7-day mean 71.5 µg/m³. Exact AQI-category accuracy: V1 71.8%, V2 69.2%, Hybrid 71.8%, persistence 65.4%. V2 (35 features) − V1 (25 features): **-0.69 µg/m³ (-1.4%)**, 95% CI [-2.39, +0.70] (7-day blocks) / [-2.29, +0.27] (14-day blocks) — inconclusive (interval includes 0).
* **Which model is best for 2024-25? Not established.** V2 (35 features) has the lowest point-estimate MAE and V1 (25 features) the highest exact-category accuracy, but the V2 − V1 difference is inconclusive. Both have a lower point-estimate MAE than "tomorrow = today" (5% / 7% lower), but that gap is inconclusive too (section 6.1). Both are clearly better than the trailing 7-day mean (71.5), which lags the smog surges (CONCLUSIVE).
* **Across the 4 winters that could be scored** (2022-23, 2023-24, 2024-25, 2025-26; 353 days, 105 with day-parts): on days with day-parts V2 − V1 is **-3.88 µg/m³ (-10.9%)**, 95% CI [-8.73, +1.11] (7-day blocks) / [-9.00, +1.47] (14-day blocks) — inconclusive (interval includes 0); on days without, **+1.75 µg/m³ (+3.7%)**, 95% CI [+0.30, +3.43] (7-day blocks) / [+0.20, +3.58] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks). Day-part evidence comes from 1 winter(s): 2025-26.
* **Recommendation: keep V1 as the default pending the winter 2026-27 forward test.** The evidence favours V2 only where hourly day-parts exist and only as an interval that still includes zero; it hurts without them. The Hybrid is the logical way to use V2, but this back-test cannot show it beats V1: it differs from V1 on at most 2 days in 3 of the 4 scored winters. One historical winter does not guarantee anything about 2026-27.

## 2. What the project does today (inspection) and why the stored models cannot be used here

* **Training recipe (both versions).** `model/train.py`: XGBoost regressor on `log1p(PM_tomorrow) − log1p(PM_today)`, 800 trees max, learning rate 0.03, depth 4, subsample 0.85, colsample 0.85, min_child_weight 3, L2 2.0, early stopping (40 rounds) on the last 60 training rows, `random_state=42`. V1 = 25 features (`FEATURES_V1`), V2 = 35 (`FEATURES_V2`, adds 7 hourly day-part features and 8 binary flags). Feature definitions live in `core/features.py` and are shared by training, evaluation and the Lambda, so the same rule builds a feature everywhere.
* **Missing features.** Day-part features come from hourly data (`core/daily.py::daypart_stats`; needs >=16 hours). Where hours are absent they are NaN; XGBoost handles NaN natively; nothing is imputed. The Hybrid (`core/hybrid.py`) uses V2 only when all 7 day-part inputs are finite, else V1.
* **Valid PM2.5.** `data/daily.csv` holds a city value only for days with >=3 stations having >=16 valid hours (or a CPCB daily value). 13 station IDs exist in total; 2018-22 and Feb 2025 onward are hourly (OpenAQ); 2023-24 are CPCB daily files for four stations and have no hourly data, so no day-part features (until the OpenCity gap-fill, section 10).
* **Stored artifacts cannot reproduce this setup.** `model/<version>/model.json` was trained through 2026-10-07, i.e. on every winter, including this one. Using them on 2024-25 would be scoring training data. The back-test therefore re-trains frozen copies **in memory** with the unchanged recipe (nothing is written to `model/`).
* **The previous winter checks were not frozen.** `model/compare_versions.py` / `model/winter_check.py` use a walk-forward fit that is refitted each month, so later months of a winter are predicted by a model that has already trained on earlier months of the same winter, with tomorrow's observed weather. This is a legitimate live-like scheme but not what was asked here.

## 3. Data coverage and missing-data report

**Window 2024-25** (2024-10-15 to 2025-02-15, 124 target days).

| Month | Stations reporting a valid daily value |
|---|---|
| 2024-10 | 17 day(s) with 4 station(s) |
| 2024-11 | 30 day(s) with 4 station(s) |
| 2024-12 | 31 day(s) with 4 station(s) |
| 2025-01 | 31 day(s) with 1 station(s) |
| 2025-02 | 13 day(s) with 1 station(s), 2 day(s) with 0 station(s) |

| Station | Valid days in window | Source | Span |
|---|---|---|---|
| 10831 (Rohini) | 78 | daily-file | 2024-10-15 to 2024-12-31 |
| 17 (R K Puram) | 78 | daily-file | 2024-10-15 to 2024-12-31 |
| 235 (Anand Vihar) | 122 | daily-file | 2024-10-15 to 2025-02-15 |
| 50 (Punjabi Bagh) | 78 | daily-file | 2024-10-15 to 2024-12-31 |

| Step of the eligibility ladder (same for every model) | Days in the 2024-25 window |
|---|---|
| Calendar target days 15 Oct 2024 - 15 Feb 2025 | 124 |
| ... with a city PM2.5 (>=3 stations) on the target day | 78 |
| ... also with city PM2.5 on the forecast-origin day, weather on both days, and an origin the 23:00 Lambda could use | 78 |
| ... of which with all 7 hourly day-part features | 0 |

**Every exclusion (winter 2024-25):**

| From | To | Days | Reason (same for V1, V2, Hybrid and both baselines) |
|---|---|---|---|
| 2025-01-01 | 2025-02-03 | 34 | 1_no_city_pm25_on_target_day: only 1 station(s) reported (< 3 needed for a city value) |
| 2025-02-04 | 2025-02-05 | 2 | 1_no_city_pm25_on_target_day: no station reported |
| 2025-02-06 | 2025-02-15 | 10 | 1_no_city_pm25_on_target_day: only 1 station(s) reported (< 3 needed for a city value) |

Notes: the single Anand Vihar value on those days was NOT used as a stand-in for the city mean, because it is a different quantity (one traffic hot-spot station vs the city average the models are trained on). Nothing was interpolated or imputed. Hourly data exist in this dataset only for 2018-22 and Feb 2025 onward, none inside this window, so day-part features cannot be built for it.

**What would still improve this evaluation:** (1) city PM2.5 for 1 Jan - 15 Feb 2025 from at least 3 stations (the 2025 CPCB daily files for R K Puram, Punjabi Bagh and Rohini, or the OpenCity 15-minute files, section 10; neither is reachable from the build sandbox); (2) hourly station data for the window, so day-part features can exist (the OpenCity 15-minute files provide it); (3) archived weather forecasts issued before 23:00 for each day (an archive such as Open-Meteo's historical-forecast API; the host is not reachable from the build sandbox and nothing is stored), which would turn the primary regime into a realistic one.

**All winters (same ladder):**

| Winter | Calendar days | Scored days | Excluded | Scored days with day-parts | Training origins before 15 Oct | ...of which with day-parts | Status |
|---|---|---|---|---|---|---|---|
| 2018-19 | 124 | 35 | 89 | — | 38 | — | not run: only 38 training origins before 2018-10-15 (< 365) |
| 2019-20 | 124 | 109 | 15 | — | 164 | — | not run: only 164 training origins before 2019-10-15 (< 365) |
| 2020-21 | 124 | 9 | 115 | — | 314 | — | not run: only 314 training origins before 2020-10-15 (< 365) |
| 2021-22 | 124 | 64 | 60 | — | 334 | — | not run: only 334 training origins before 2021-10-15 (< 365) |
| 2022-23 | 124 | 47 | 77 | 2 | 459 | 444 | scored |
| 2023-24 | 124 | 124 | 0 | 0 | 740 | 446 | scored |
| 2024-25 | 124 | 78 | 46 | 0 | 1106 | 446 | scored |
| 2025-26 | 124 | 104 | 20 | 103 | 1404 | 662 | scored |

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
| 2023-24 | 124 / 124 | 0 | 49.31 | 59.29 | 48.25 | 49.98 | 48.25 | +1.73 [-0.2, +3.9] | = V1 | 64% / 60% / 64% / 62% | scored |
| 2024-25 | 78 / 124 | 0 | 51.88 | 71.5 | 49.05 | 48.36 | 49.05 | -0.69 [-2.4, +0.7] | = V1 | 72% / 69% / 72% / 65% | scored |
| 2025-26 | 104 / 124 | 103 | 41.6 | 48.47 | 36.09 | 31.89 | 32.16 | -4.20 [-9.0, +0.7] | -3.93 [-8.5, +1.0] | 64% / 69% / 68% / 62% | scored |

**Diagnostic regime (tomorrow's observed weather - optimistic, not live).**

| Winter | Scored / calendar days | Days with day-parts | Persist. MAE | 7-day mean MAE | V1 MAE | V2 MAE | Hybrid MAE | V2 − V1 (µg/m³) [95% CI] | Hybrid − V1 [95% CI] | Exact category V1 / V2 / Hybrid / Persist. | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-19 | 35 / 124 | — | — | — | — | — | — | — | — | — | not run: only 38 training origins before 2018-10-15 (< 365) |
| 2019-20 | 109 / 124 | — | — | — | — | — | — | — | — | — | not run: only 164 training origins before 2019-10-15 (< 365) |
| 2020-21 | 9 / 124 | — | — | — | — | — | — | — | — | — | not run: only 314 training origins before 2020-10-15 (< 365) |
| 2021-22 | 64 / 124 | — | — | — | — | — | — | — | — | — | not run: only 334 training origins before 2021-10-15 (< 365) |
| 2022-23 | 47 / 124 | 2 | 49.0 | 53.31 | 38.2 | 44.69 | 37.92 | +6.50 [+2.0, +11.2] * | -0.28 [-0.6, +0.0] | 53% / 38% / 53% / 47% | scored |
| 2023-24 | 124 / 124 | 0 | 49.31 | 59.29 | 43.69 | 44.91 | 43.69 | +1.23 [-0.6, +3.2] | = V1 | 70% / 65% / 70% / 62% | scored |
| 2024-25 | 78 / 124 | 0 | 51.88 | 71.5 | 49.08 | 48.25 | 49.08 | -0.83 [-3.0, +0.8] | = V1 | 76% / 70% / 76% / 65% | scored |
| 2025-26 | 104 / 124 | 103 | 41.6 | 48.47 | 33.04 | 30.44 | 30.72 | -2.60 [-7.0, +1.8] | -2.32 [-6.6, +1.9] | 69% / 73% / 72% / 62% | scored |

**Pooled over the scored winters** (each from its own frozen fit; these winters share the same retrospective caveats):

| Regime | Set | Days | Persist. MAE | V1 MAE | V2 MAE | Hybrid MAE | V2 − V1 | Hybrid − V1 |
|---|---|---|---|---|---|---|---|---|
| primary | All matched days | 353 | 47.57 | 44.08 | 44.15 | 42.92 | **+0.08 µg/m³ (+0.2%)**, 95% CI [-1.83, +2.06] (7-day blocks) / [-2.06, +2.28] (14-day blocks) — inconclusive (interval includes 0) | **-1.16 µg/m³ (-2.6%)**, 95% CI [-2.60, +0.31] (7-day blocks) / [-2.70, +0.35] (14-day blocks) — inconclusive (interval includes 0) |
| primary | Days WITH hourly day-parts | 105 | 41.41 | 35.59 | 31.71 | 31.71 | **-3.88 µg/m³ (-10.9%)**, 95% CI [-8.73, +1.11] (7-day blocks) / [-9.00, +1.47] (14-day blocks) — inconclusive (interval includes 0) | **-3.88 µg/m³ (-10.9%)**, 95% CI [-8.73, +1.11] (7-day blocks) / [-9.00, +1.47] (14-day blocks) — inconclusive (interval includes 0) |
| primary | Days WITHOUT hourly day-parts | 248 | 50.17 | 47.67 | 49.42 | 47.67 | **+1.75 µg/m³ (+3.7%)**, 95% CI [+0.30, +3.43] (7-day blocks) / [+0.20, +3.58] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | identical predictions on every day (difference exactly 0) |
| diagnostic | All matched days | 353 | 47.57 | 41.01 | 41.36 | 40.29 | **+0.35 µg/m³ (+0.8%)**, 95% CI [-1.36, +2.19] (7-day blocks) / [-1.43, +2.37] (14-day blocks) — inconclusive (interval includes 0) | **-0.72 µg/m³ (-1.8%)**, 95% CI [-1.98, +0.61] (7-day blocks) / [-1.95, +0.61] (14-day blocks) — inconclusive (interval includes 0) |
| diagnostic | Days WITH hourly day-parts | 105 | 41.41 | 32.59 | 30.18 | 30.18 | **-2.42 µg/m³ (-7.4%)**, 95% CI [-6.85, +2.19] (7-day blocks) / [-6.79, +2.47] (14-day blocks) — inconclusive (interval includes 0) | **-2.42 µg/m³ (-7.4%)**, 95% CI [-6.85, +2.19] (7-day blocks) / [-6.79, +2.47] (14-day blocks) — inconclusive (interval includes 0) |
| diagnostic | Days WITHOUT hourly day-parts | 248 | 50.17 | 44.58 | 46.09 | 44.58 | **+1.52 µg/m³ (+3.4%)**, 95% CI [-0.12, +3.07] (7-day blocks) / [-0.13, +3.22] (14-day blocks) — inconclusive (interval includes 0) | identical predictions on every day (difference exactly 0) |

## 6. Winter 2024-25 in detail

Scored target dates: 2024-10-15 to 2024-12-31 (78 days). Training: 1106 origins with targets 2018-01-05 to 2024-10-14 (446 with day-parts, the rest with day-parts NaN). Trees after early stopping: V1 183, V2 232. Sudden-change days: 25 of 78.

### 6.1 Primary regime (leak-free weather), all matched days

| Model | MAE µg/m³ | RMSE µg/m³ | R² | Exact category | Within ±1 category | Sudden-change days: MAE / exact cat / right direction |
|---|---|---|---|---|---|---|
| V1 (25 features) | 49.05 | 68.82 | 0.571 | 71.8% | 98.7% | 84.8 / 60% / 76% |
| V2 (35 features) | 48.36 | 68.11 | 0.579 | 69.2% | 98.7% | 84.2 / 60% / 80% |
| Hybrid | 49.05 | 68.82 | 0.571 | 71.8% | 98.7% | 84.8 / 60% / 76% |
| Persistence (tomorrow = today) | 51.88 | 74.49 | 0.497 | 65.4% | 98.7% | 101.7 / 36% / n/a |
| Trailing 7-day mean | 71.5 | 95.6 | 0.171 | 59.0% | 97.4% | 88.3 / 56% / 64% |

| MAE difference (negative = second model better) | µg/m³ | % | 95% CI, 7-day blocks | 95% CI, 14-day blocks | Verdict |
|---|---|---|---|---|---|
| V2 − V1 | -0.69 | -1.4% | [-2.39, +0.70] | [-2.29, +0.27] | inconclusive |
| Hybrid − V1 | 0.00 (identical) | 0.0% | [0, 0] | [0, 0] | identical |
| V2 − Hybrid | -0.69 | -1.4% | [-2.39, +0.70] | [-2.29, +0.27] | inconclusive |
| V1 − Persist. | -2.83 | -5.5% | [-10.32, +4.99] | [-7.49, +4.41] | inconclusive |
| Hybrid − Persist. | -2.83 | -5.5% | [-10.32, +4.99] | [-7.49, +4.41] | inconclusive |
| V2 − Persist. | -3.52 | -6.8% | [-10.57, +3.21] | [-8.01, +2.69] | inconclusive |
| V1 − 7-day mean | -22.45 | -31.4% | [-38.16, -7.72] | [-37.20, -6.86] | CONCLUSIVE |

### 6.2 Dates with and without hourly features

* **Days WITH hourly day-parts: 0 days.** Nothing to compare in this winter.
* **Days WITHOUT hourly day-parts: 78 days** (= all matched days; table above).
* V2 is run with its 7 day-part inputs missing on every day; the Hybrid routes all of them to V1.
* For the with/without split on days that do have day-parts, see the winter-by-winter and pooled rows in section 5.

### 6.3 Diagnostic regime (tomorrow's observed weather - NOT live performance)

| Model | MAE µg/m³ | RMSE µg/m³ | R² | Exact category | Within ±1 category | Sudden-change days: MAE / exact cat / right direction |
|---|---|---|---|---|---|---|
| V1 (25 features) | 49.08 | 67.18 | 0.591 | 75.6% | 98.7% | 80.5 / 68% / 76% |
| V2 (35 features) | 48.25 | 66.24 | 0.602 | 70.5% | 98.7% | 79.9 / 64% / 84% |
| Hybrid | 49.08 | 67.18 | 0.591 | 75.6% | 98.7% | 80.5 / 68% / 76% |
| Persistence (tomorrow = today) | 51.88 | 74.49 | 0.497 | 65.4% | 98.7% | 101.7 / 36% / n/a |
| Trailing 7-day mean | 71.5 | 95.6 | 0.171 | 59.0% | 97.4% | 88.3 / 56% / 64% |

| MAE difference (negative = second model better) | µg/m³ | % | 95% CI, 7-day blocks | 95% CI, 14-day blocks | Verdict |
|---|---|---|---|---|---|
| V2 − V1 | -0.83 | -1.7% | [-2.96, +0.81] | [-3.12, +0.42] | inconclusive |
| Hybrid − V1 | 0.00 (identical) | 0.0% | [0, 0] | [0, 0] | identical |
| V2 − Hybrid | -0.83 | -1.7% | [-2.96, +0.81] | [-3.12, +0.42] | inconclusive |
| V1 − Persist. | -2.80 | -5.4% | [-9.29, +5.38] | [-7.58, +5.48] | inconclusive |
| Hybrid − Persist. | -2.80 | -5.4% | [-9.29, +5.38] | [-7.58, +5.48] | inconclusive |
| V2 − Persist. | -3.63 | -7.0% | [-9.25, +3.17] | [-7.82, +2.89] | inconclusive |
| V1 − 7-day mean | -22.42 | -31.4% | [-38.20, -6.36] | [-36.99, -5.90] | CONCLUSIVE |

### 6.4 Reconciliation with the earlier (non-frozen) check

| Setting (winter 2024-25, 78 days) | V1 MAE | V2 MAE | Note |
|---|---|---|---|
| Earlier `winter_check.py`: monthly refit, observed weather, 24 h (`experiments/winter_check.json`) | 48.9 | 47.33 | later months are predicted by models that trained on earlier months of the same winter |
| This back-test, legacy regime: frozen fit, observed weather, 24 h | 49.08 | 48.25 | same inputs as above, but nothing from the winter in training |
| This back-test, primary: frozen fit, weather persistence, hours <= 22 | 49.05 | 48.36 | leak-free weather |

The frozen fit is slightly worse than the monthly refit, as expected, and the V2-vs-V1 ordering is unchanged. The earlier check's -3.2% (interval -7.6% to +1.1%) was also inconclusive.

## 7. Which model is best, and is the evidence conclusive?

* **Winter 2024-25 alone: no conclusive winner.** V2 − V1 = -0.69 µg/m³ (-1.4%), interval [-2.39, +0.70], which includes 0. V1 has the higher exact-category accuracy (71.8% vs 69.2%), but that is a handful of days. The Hybrid cannot be distinguished from V1 here because it is V1.
* **What the data can say about V2.** With hourly day-parts it has the lower MAE (pooled: V2 (35 features) − V1 (25 features): **-3.88 µg/m³ (-10.9%)**, 95% CI [-8.73, +1.11] (7-day blocks) / [-9.00, +1.47] (14-day blocks) — inconclusive (interval includes 0)); without them it is worse (pooled, primary: V2 (35 features) − V1 (25 features): **+1.75 µg/m³ (+3.7%)**, 95% CI [+0.30, +3.43] (7-day blocks) / [+0.20, +3.58] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks); in the optimistic diagnostic regime the same gap is +1.52 µg/m³, 95% CI [-0.12, +3.07], inconclusive). That is the reason for the Hybrid; the day-part evidence comes from 1 winter(s) (105 days).
* **Hybrid.** By construction it equals V1 on days without day-parts and V2 on days with them. Historically it scores 42.92 vs V1 44.08 pooled (primary); 100% of that gain comes from the latest winter (2025-26). Hybrid − V1 pooled: -1.16 µg/m³, 95% CI [-2.60, +0.31] - inconclusive.
* **Not shown by this back-test:** whether V2 or the Hybrid beats V1 in winter 2026-27, or how they behave with a real weather forecast, or any result for Jan-Feb 2025.

## 8. Limitations

* Retrospective: these winters were examined before; V2's design was shaped by earlier looks. Treat all results as descriptive.
* Small samples with strong serial correlation (smog episodes): 78 days is only about 11 weeks of data and neighbouring days are not independent. Wide intervals are the honest result.
* No archived weather forecasts; both weather regimes are bounds, not live performance.
* Frozen fits get up to ~4 months stale. A live system refitted monthly would do a little better.
* 2024-25 covers Oct-Dec only. It misses the Jan-Feb part of the season and contains no day-part days.
* Winter 2022-23 has few scored days, and winters before 2022-23 could not be scored at all (too little training data under the 3-station rule).
* Day-part evidence exists for 1 scored winter(s) only.

## 9. Recommendation

Keep **V1 (`v1_25_baseline`) as the default**; keep V2 as the experimental candidate and the Hybrid as the pre-declared candidate for the prospective winter 2026-27 shadow test (frozen rules in `model/EVALUATION_PROTOCOL.md`; promotion needs >=5% lower MAE with the 95% interval below zero on >=90 matched days). Nothing in this back-test justifies changing the default, and nothing here was committed, pushed or deployed.

## 10. Re-run with the added OpenCity 15-minute data

### 10.1 What was added and how it was checked

* **Source:** [Delhi Hourly Air Quality Reports (OpenCity)](https://data.opencity.in/dataset/delhi-hourly-air-quality-reports), CPCB 15-minute station files "2024-25" (2024-01-01 to 2025-12-31) for R K Puram, Punjabi Bagh, Rohini and Anand Vihar. Downloaded through your in-app browser, because the build sandbox is blocked from that host. Not committed anywhere.
* **Time zone.** The timestamps carry a `+0000` suffix but are Delhi wall-clock time. Read as IST, 24-hour means reproduce the official CPCB daily files for 2024 exactly at all four stations (mean absolute difference 0.0 µg/m³ vs 8-11 µg/m³ when read as UTC); `experiments/opencity_timezone_check.json`, `pipeline/check_opencity_timezone.py`.
* **Gap-fill only.** A reading is used only when that station has no OpenAQ reading at all on that day (kept: Anand Vihar 31,026, Punjabi Bagh 38,566, R K Puram 37,745, Rohini 34,821 readings). Days OpenAQ already covers are unchanged. Blank values are dropped, nothing is interpolated. Rules written before use: `pipeline/convert_opencity.py`.
* **Effect on the dataset** (separate file `data/daily_opencity.csv`; production `data/daily.csv` untouched): 50 new city days (2025-01-01 to 2025-04-10), and 360 existing days (almost all in 2024) change value by 0.46 µg/m³ on average (max 6.8), because they are now built from the 15-minute readings instead of the CPCB daily file. Those 2024 days now also have day-part features, which is what gives V2 real inputs before and during the 2024-25 winter.

### 10.2 Winter-by-winter, with the added data

**Primary regime (leak-free weather).**

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

**Diagnostic regime (observed weather, not live).**

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

### 10.3 Where the 2024-25 change comes from (primary regime)

| Run / slice | Days | Days with day-parts | Persist. MAE | V1 MAE | V2 MAE | Hybrid MAE | V2 − V1 µg/m³ (%) [95% CI] | Hybrid − V1 |
|---|---|---|---|---|---|---|---|---|
| Production data, 15 Oct - 31 Dec (all that was scoreable) | 78 | 0 | 51.88 | 49.05 | 48.36 | 49.05 | -0.70 (-1.4%) [-2.39, +0.70] | = V1 |
| Added data, the SAME 15 Oct - 31 Dec dates | 78 | 78 | 53.07 | 49.60 | 43.70 | 43.70 | -5.90 (-11.9%) [-11.18, -1.06] | -5.90 (-11.9%) [-11.18, -1.06] |
| Added data, 1 Jan - 15 Feb (the 46 newly scoreable days) | 46 | 46 | 46.90 | 43.87 | 40.91 | 40.91 | -2.96 (-6.8%) [-5.87, +1.20] | -2.96 (-6.8%) [-5.87, +1.20] |
| Added data, whole window | 124 | 124 | 50.78 | 47.48 | 42.66 | 42.66 | -4.81 (-10.1%) [-8.18, -0.95] | -4.81 (-10.1%) [-8.18, -0.95] |

Reading: on the same Oct-Dec dates, V2 goes from "about equal to V1" to clearly better because it now has real day-part inputs (in the production run it had none and was run with NaNs). The 46 extra January-February days point the same way but, taken alone, their interval includes zero. V1's own error changes little between rows, so this is not V1 getting worse. The persistence column differs slightly between the first two rows only because the 2024 target values are now the 15-minute-built daily means.

### 10.4 Winter 2024-25 in detail, with the added data (primary regime)

Training: 1106 origins (733 with day-parts); trees after early stopping V1 217, V2 241; sudden-change days 42 of 124.

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

**Diagnostic regime (observed weather, not live):**

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

### 10.5 Pooled over scored winters, with the added data (primary regime)

| Set | Days | Persist. MAE | V1 MAE | V2 MAE | Hybrid MAE | V2 − V1 | Hybrid − V1 |
|---|---|---|---|---|---|---|---|
| All matched days | 399 | 47.91 | 44.36 | 41.62 | 40.35 | **-2.74 µg/m³ (-6.2%)**, 95% CI [-4.99, -0.52] (7-day blocks) / [-5.13, -0.29] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | **-4.01 µg/m³ (-9.0%)**, 95% CI [-5.92, -2.16] (7-day blocks) / [-5.87, -2.17] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) |
| Days WITH hourly day-parts | 274 | 47.66 | 44.01 | 38.17 | 38.17 | **-5.84 µg/m³ (-13.3%)**, 95% CI [-8.46, -3.26] (7-day blocks) / [-8.20, -3.48] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | **-5.84 µg/m³ (-13.3%)**, 95% CI [-8.46, -3.26] (7-day blocks) / [-8.20, -3.48] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) |
| Days WITHOUT hourly day-parts | 125 | 48.45 | 45.14 | 49.2 | 45.14 | **+4.05 µg/m³ (+9.0%)**, 95% CI [+1.72, +6.68] (7-day blocks) / [+1.39, +7.08] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks) | identical predictions on every day (difference exactly 0) |

### 10.6 What this changes, and what it does not

* With complete data, winter 2024-25 supports V2/Hybrid over V1: **-4.81 µg/m³ (-10.1%)**, 95% CI [-8.19, -0.95] (7-day blocks) / [-8.39, -1.56] (14-day blocks) — CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks). Pooled over the scored winters the Hybrid is -4.01 µg/m³ (-9.0%) vs V1, 95% CI [-5.92, -2.16], CONCLUSIVE; on days with day-parts V2 is -5.84 µg/m³ (-13.3%), 95% CI [-8.46, -3.26], CONCLUSIVE.
* The earlier verdict "not conclusive" most plausibly came from missing data rather than from V2 failing: V2 had no day-part inputs in the production run. (No separate control isolates this from the extra day-part rows V2 now has in training.)
* What it does not change: these winters were all seen before, the weather is a bound rather than a real forecast, and the hourly values for 2024 and early 2025 come from a different feed than 2018-22 and later 2025. A model that wins retrospectively can still lose next winter, so the default stays V1 until the shadow test, with the Hybrid as the leading candidate.
