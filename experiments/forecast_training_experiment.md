# Training on archived weather forecasts - controlled experiment

> **Retrospective, exploratory.** Winters 2024-25 and 2025-26 were examined before, so this is not an untouched hold-out. The arms and the decision rule were fixed before any result was looked at (`model/forecast_training_experiment.py`). Nothing after 15 Feb 2026 is read, so the winter 2026-27 shadow test is untouched. No model file, default, checksum or dashboard output was changed.

## 1. The two earlier numbers are different comparisons

| Result | What was compared | Training | Test weather input | Test days | Baseline | Where | Numbers |
|---|---|---|---|---|---|---|---|
| **-11.3 %** (conclusive) | Hybrid vs **V1** (a different feature set / router) | both trained on observed (ERA5) weather, identical recipe | real archived 24 h forecast | 2024-25 (124 d) + 2025-26 (104 d) = 228 | V1 itself (not persistence) | `option_screen.py` section 1 | 41.93 → 37.21 MAE; CI [-7.55, -1.70] / [-7.54, -1.67] |
| **-3.3 %** (inconclusive) | Hybrid **trained on forecast weather** vs the same Hybrid trained on observed weather | same models and recipe; only the TRAINING weather source differs | real archived 24 h forecast | same 228 days | Hybrid trained on ERA5 | `option_screen.py` O5 | 37.21 → 35.97 MAE; CI [-3.10, +0.78] / [-3.18, +0.88] |

They answer different questions: the first says the 35-feature Hybrid beats the 25-feature V1 when both see real forecasts; the second asks whether changing the TRAINING weather helps one model. They are not equivalent and neither implies the other. This experiment re-runs the second question under a controlled design and reproduces the O5 number exactly (arm A2 below: -3.3 %).

## 2. What the archive can and cannot support

* Archived forecasts (Open-Meteo Previous Runs): 24 h ahead for 759 days (2024-01-20 → 2026-02-16), 48 h ahead for 758 days. Nothing earlier exists, so winters 2018-19 → 2023-24 can only supply ERA5 weather.
* **Issue time / lead.** The live Lambda runs at 23:00 IST and uses the latest run (its hours of tomorrow are roughly 1-30 h ahead of the run, depending on run availability; not verified here). The archive gives each hour as forecast 24 h (or 48 h) earlier, i.e. the same age or older than live. It can not leak future data and, if anything, slightly understates live skill. The exact live run can not be reproduced.
* **Today's weather (wx0).** Live reads it from the forecast API's analysis; no analysis archive is available, so the main arms keep ERA5 for wx0. Arm S1 replaces it by the 24 h forecast of today as a stress test.
* **No silent substitution.** A1 uses only origins with a forecast. A2 does substitute ERA5 for older origins - openly, with the counts below. No test day lacks a forecast: all arms use the same days.
* Winters scored: only 2024-25 and 2025-26 (the only ones with forecast test weather). Results are by winter, not by calendar season; the archive has no autumn/summer window that the frozen-fit protocol could score.

| Winter | Matched test days | Days dropped (no forecast) | Training origins (ERA5 / A2) | Origins with a forecast (A1) | ...of which Nov-Feb | Nov-Feb origins in ERA5 history |
|---|---|---|---|---|---|---|
| 2024-25 | 124 | 0 | 1106 | 269 | 41 | 378 |
| 2025-26 | 104 | 0 | 1454 | 617 | 161 | 498 |

* **A1 train ONLY on forecast-weather origins not run for 2024-25:** only 269 training origins with the required weather before 2024-10-15 (< 365).

## 3. Is there a forecast-vs-ERA5 mismatch inside the test windows?

Mean difference (forecast − ERA5) and RMSE on the matched test days, tomorrow's weather:

| Winter | Input | temp °C | humidity % | wind km/h | rain mm | pressure hPa |
|---|---|---|---|---|---|---|
| 2024-25 | raw forecast | -0.26 / 1.02 | +3.20 / 7.25 | -1.74 / 2.30 | -0.06 / 1.92 | +0.33 / 0.46 |
| 2024-25 | calibrated forecast | -1.10 / 1.50 | +1.95 / 6.83 | +2.24 / 2.70 | +0.10 / 1.92 | +0.20 / 0.37 |
| 2025-26 | raw forecast | -0.11 / 0.50 | -1.67 / 3.26 | +0.07 / 0.73 | -0.01 / 1.86 | -0.51 / 0.57 |
| 2025-26 | calibrated forecast | +0.01 / 0.48 | -6.64 / 7.59 | +2.64 / 2.76 | +0.29 / 1.49 | -0.78 / 0.83 |
(bias / RMSE)

The mismatch is **not stable**: in 2024-25 the forecast wind is lower and humidity higher than ERA5, while in 2025-26 the bias is close to zero. The earlier statement that forecasts are "wind lower by 2-4 km/h" came from all-year pre-window averages and does not describe the winter test windows; this report supersedes it. A correction fitted on all seasons before a window (A3) does not match winter conditions: it moves wind by +2.2 km/h (2024-25) and +2.6 km/h (2025-26) (raw: -1.7 and +0.1) and humidity by +1.9 pts (2024-25) and -6.6 pts (2025-26), i.e. it overshoots. A3's per-winter results below follow from that.

## 4. Results (pooled over the scored winters, frozen fit before each winter)

MAE and RMSE in µg/m³; exact AQI-category accuracy; improvement over persistence = (MAE_persistence − MAE_model) / MAE_persistence. 95 % intervals: moving-block bootstrap (7-day blocks, 5000 resamples).

| Arm | Winters (matched days) | Model | MAE [95 %] | RMSE [95 %] | Exact category [95 %] | Better than persistence |
|---|---|---|---|---|---|---|
| A0 reference: train ERA5, test forecast | 2024-25, 2025-26 (228 d) | V1 (25 features) | 41.93 [36.19, 49.74] | 57.81 [48.24, 70.72] | 64.9 % [58.0, 72.2] | +10.0 % [+1.8, +17.8] |
| A0 reference: train ERA5, test forecast | 2024-25, 2025-26 (228 d) | V2 (35 features) | 37.17 [31.50, 45.28] | 51.67 [41.29, 66.16] | 70.2 % [63.2, 76.8] | +20.2 % [+10.5, +28.9] |
| A0 reference: train ERA5, test forecast | 2024-25, 2025-26 (228 d) | Hybrid | 37.21 [31.56, 45.35] | 51.68 [41.32, 66.16] | 70.2 % [63.2, 76.8] | +20.1 % [+10.4, +28.9] |
| A1 train ONLY on forecast-weather origins | 2025-26 (104 d) | V1 (25 features) | 42.20 [35.15, 52.78] | 53.61 [43.72, 65.72] | 50.0 % [38.6, 63.2] | -1.4 % [-23.7, +15.5] |
| A1 train ONLY on forecast-weather origins | 2025-26 (104 d) | V2 (35 features) | 35.62 [28.71, 46.46] | 47.32 [37.26, 59.82] | 57.7 % [45.9, 70.4] | +14.4 % [-7.0, +29.5] |
| A1 train ONLY on forecast-weather origins | 2025-26 (104 d) | Hybrid | 35.06 [27.88, 46.06] | 46.92 [36.57, 59.55] | 56.7 % [45.2, 68.6] | +15.7 % [-6.0, +31.3] |
| A2 train on forecast where it exists, ERA5 elsewhere | 2024-25, 2025-26 (228 d) | V1 (25 features) | 41.31 [35.80, 48.94] | 56.72 [47.47, 68.90] | 64.9 % [58.2, 72.0] | +11.3 % [+6.5, +15.9] |
| A2 train on forecast where it exists, ERA5 elsewhere | 2024-25, 2025-26 (228 d) | V2 (35 features) | 35.94 [29.80, 44.96] | 51.74 [40.29, 66.81] | 69.7 % [62.9, 76.3] | +22.9 % [+13.6, +30.7] |
| A2 train on forecast where it exists, ERA5 elsewhere | 2024-25, 2025-26 (228 d) | Hybrid | 35.97 [29.84, 44.98] | 51.75 [40.32, 66.83] | 69.7 % [62.9, 76.3] | +22.8 % [+13.6, +30.6] |
| A3 train ERA5, forecast input calibrated | 2024-25, 2025-26 (228 d) | V1 (25 features) | 43.73 [37.19, 53.00] | 60.94 [49.20, 76.37] | 60.1 % [53.5, 68.2] | +6.1 % [-4.8, +15.8] |
| A3 train ERA5, forecast input calibrated | 2024-25, 2025-26 (228 d) | V2 (35 features) | 37.56 [30.75, 47.52] | 55.70 [41.72, 73.44] | 70.6 % [62.9, 77.9] | +19.4 % [+6.8, +29.7] |
| A3 train ERA5, forecast input calibrated | 2024-25, 2025-26 (228 d) | Hybrid | 37.63 [30.78, 47.63] | 55.74 [41.75, 73.50] | 70.6 % [62.9, 77.9] | +19.2 % [+6.7, +29.5] |
| S1 (sensitivity) like A2, today's weather also forecast | 2024-25, 2025-26 (228 d) | V1 (25 features) | 40.32 [34.79, 47.80] | 55.85 [46.21, 68.54] | 66.7 % [60.5, 73.9] | +13.5 % [+8.0, +18.8] |
| S1 (sensitivity) like A2, today's weather also forecast | 2024-25, 2025-26 (228 d) | V2 (35 features) | 36.04 [29.90, 45.27] | 52.24 [39.80, 68.75] | 71.9 % [64.3, 78.4] | +22.7 % [+12.4, +31.1] |
| S1 (sensitivity) like A2, today's weather also forecast | 2024-25, 2025-26 (228 d) | Hybrid | 36.00 [29.85, 45.21] | 52.22 [39.79, 68.70] | 71.9 % [64.3, 78.4] | +22.7 % [+12.4, +31.2] |
| REF oracle: tomorrow's OBSERVED weather (not live) | 2024-25, 2025-26 (228 d) | V1 (25 features) | 39.64 [34.37, 47.30] | 55.07 [45.92, 67.32] | 68.9 % [62.3, 75.6] | +14.9 % [+7.2, +22.2] |
| REF oracle: tomorrow's OBSERVED weather (not live) | 2024-25, 2025-26 (228 d) | V2 (35 features) | 34.98 [29.53, 43.11] | 49.46 [38.90, 63.87] | 72.4 % [65.9, 77.6] | +24.9 % [+15.6, +33.0] |
| REF oracle: tomorrow's OBSERVED weather (not live) | 2024-25, 2025-26 (228 d) | Hybrid | 35.04 [29.58, 43.13] | 49.49 [38.95, 63.87] | 72.4 % [65.9, 77.6] | +24.8 % [+15.5, +32.8] |
| REF 48 h forecast instead of 24 h | 2024-25, 2025-26 (228 d) | V1 (25 features) | 41.70 [35.90, 49.66] | 57.53 [47.91, 70.62] | 66.7 % [60.4, 73.6] | +10.5 % [+2.4, +18.6] |
| REF 48 h forecast instead of 24 h | 2024-25, 2025-26 (228 d) | V2 (35 features) | 37.06 [31.49, 45.33] | 51.70 [41.37, 66.17] | 70.2 % [63.6, 76.4] | +20.5 % [+10.6, +29.2] |
| REF 48 h forecast instead of 24 h | 2024-25, 2025-26 (228 d) | Hybrid | 37.09 [31.54, 45.34] | 51.71 [41.41, 66.20] | 70.2 % [63.6, 76.4] | +20.4 % [+10.6, +29.1] |
| baseline | 2024-25, 2025-26 (228 d) | Persistence | 46.59 [40.07, 55.11] | 64.36 [53.67, 76.96] | 60.5 % [53.3, 68.1] | - |
| baseline | 2024-25, 2025-26 (228 d) | 7-day mean | 58.75 [50.45, 69.99] | 77.18 [64.34, 91.88] | 57.9 % [49.8, 65.9] | - |

A1 covers only 2025-26 (see section 2), so its row is **not comparable** to the pooled rows of the other arms; its comparison with A0 below uses the same 2025-26 days.

## 5. Difference from the previous model (same model under the reference arm A0)

Negative MAE/RMSE = better than A0. Intervals are for the paired difference.

| Arm | Model | ΔMAE | 95 % 7-d | 95 % 14-d | ΔRMSE [95 %] | Δ exact category | Verdict |
|---|---|---|---|---|---|---|---|
| A1 train ONLY on forecast-weather origins | V1 (25 features) | +8.94 (+26.9 %) | [+2.67, +16.39] | [+2.37, +18.17] | +8.17 [+0.68, +16.55] | -14.4 pts [-24.4, -3.8] | conclusive (interval excludes 0, both block lengths) |
| A1 train ONLY on forecast-weather origins | V2 (35 features) | +7.05 (+24.7 %) | [+3.27, +11.66] | [+2.93, +12.05] | +8.55 [+3.57, +13.51] | -14.4 pts [-25.0, -4.7] | conclusive (interval excludes 0, both block lengths) |
| A1 train ONLY on forecast-weather origins | Hybrid | +6.40 (+22.3 %) | [+2.37, +11.20] | [+2.24, +11.90] | +8.10 [+2.93, +13.17] | -15.4 pts [-25.7, -6.5] | conclusive (interval excludes 0, both block lengths) |
| A2 train on forecast where it exists, ERA5 elsewhere | V1 (25 features) | -0.62 (-1.5 %) | [-2.76, +1.72] | [-2.68, +1.87] | -1.09 [-3.32, +1.41] | +0.0 pts [-4.5, +5.0] | inconclusive |
| A2 train on forecast where it exists, ERA5 elsewhere | V2 (35 features) | -1.22 (-3.3 %) | [-3.10, +0.80] | [-3.17, +0.89] | +0.07 [-1.82, +1.82] | -0.4 pts [-6.3, +5.1] | inconclusive |
| A2 train on forecast where it exists, ERA5 elsewhere | Hybrid | -1.23 (-3.3 %) | [-3.10, +0.78] | [-3.18, +0.88] | +0.07 [-1.83, +1.81] | -0.4 pts [-6.3, +5.1] | inconclusive |
| A3 train ERA5, forecast input calibrated | V1 (25 features) | +1.80 (+4.3 %) | [-2.77, +6.72] | [-3.28, +7.16] | +3.13 [-2.04, +8.09] | -4.8 pts [-11.6, +3.2] | inconclusive |
| A3 train ERA5, forecast input calibrated | V2 (35 features) | +0.40 (+1.1 %) | [-3.35, +4.60] | [-3.61, +5.15] | +4.03 [-1.33, +8.80] | +0.4 pts [-6.8, +7.4] | inconclusive |
| A3 train ERA5, forecast input calibrated | Hybrid | +0.42 (+1.1 %) | [-3.35, +4.63] | [-3.59, +5.17] | +4.06 [-1.30, +8.84] | +0.4 pts [-6.8, +7.4] | inconclusive |
| S1 (sensitivity) like A2, today's weather also forecast | V1 (25 features) | -1.62 (-3.9 %) | [-3.89, +0.66] | [-4.12, +0.86] | -1.97 [-4.35, +0.40] | +1.8 pts [-2.4, +7.7] | inconclusive |
| S1 (sensitivity) like A2, today's weather also forecast | V2 (35 features) | -1.13 (-3.0 %) | [-3.18, +1.16] | [-3.38, +1.39] | +0.57 [-2.47, +3.19] | +1.8 pts [-4.3, +6.8] | inconclusive |
| S1 (sensitivity) like A2, today's weather also forecast | Hybrid | -1.21 (-3.3 %) | [-3.25, +1.09] | [-3.45, +1.30] | +0.53 [-2.50, +3.14] | +1.8 pts [-4.3, +6.8] | inconclusive |
| REF oracle: tomorrow's OBSERVED weather (not live) | V1 (25 features) | -2.30 (-5.5 %) | [-4.18, -0.26] | [-3.88, -0.48] | -2.74 [-5.10, -0.45] | +3.9 pts [+0.4, +8.0] | conclusive (interval excludes 0, both block lengths) |
| REF oracle: tomorrow's OBSERVED weather (not live) | V2 (35 features) | -2.19 (-5.9 %) | [-3.26, -1.00] | [-3.26, -1.09] | -2.21 [-3.41, -1.02] | +2.2 pts [-1.7, +5.3] | conclusive (interval excludes 0, both block lengths) |
| REF oracle: tomorrow's OBSERVED weather (not live) | Hybrid | -2.16 (-5.8 %) | [-3.24, -0.97] | [-3.25, -1.06] | -2.20 [-3.40, -0.99] | +2.2 pts [-1.7, +5.3] | conclusive (interval excludes 0, both block lengths) |
| REF 48 h forecast instead of 24 h | V1 (25 features) | -0.23 (-0.5 %) | [-0.91, +0.42] | [-0.87, +0.36] | -0.29 [-0.84, +0.23] | +1.8 pts [+0.0, +4.5] | inconclusive |
| REF 48 h forecast instead of 24 h | V2 (35 features) | -0.10 (-0.3 %) | [-0.95, +0.75] | [-0.89, +0.71] | +0.03 [-0.91, +0.88] | +0.0 pts [-2.8, +2.8] | inconclusive |
| REF 48 h forecast instead of 24 h | Hybrid | -0.11 (-0.3 %) | [-0.96, +0.73] | [-0.89, +0.70] | +0.03 [-0.91, +0.88] | +0.0 pts [-2.8, +2.8] | inconclusive |

### Model versus model inside each arm (Hybrid − V1 and V2 − V1, ΔMAE)

| Arm | Comparison | ΔMAE | 95 % 7-d | 95 % 14-d | Verdict |
|---|---|---|---|---|---|
| A0 reference: train ERA5, test forecast | Hybrid − V1 | -4.73 (-11.3 %) | [-7.55, -1.70] | [-7.54, -1.67] | conclusive (interval excludes 0, both block lengths) |
| A0 reference: train ERA5, test forecast | V2 − V1 | -4.77 (-11.4 %) | [-7.60, -1.74] | [-7.61, -1.67] | conclusive (interval excludes 0, both block lengths) |
| A1 train ONLY on forecast-weather origins | Hybrid − V1 | -7.14 (-16.9 %) | [-11.15, -3.53] | [-10.92, -3.39] | conclusive (interval excludes 0, both block lengths) |
| A1 train ONLY on forecast-weather origins | V2 − V1 | -6.58 (-15.6 %) | [-10.73, -2.73] | [-10.65, -3.17] | conclusive (interval excludes 0, both block lengths) |
| A2 train on forecast where it exists, ERA5 elsewhere | Hybrid − V1 | -5.34 (-12.9 %) | [-8.61, -1.73] | [-8.61, -1.66] | conclusive (interval excludes 0, both block lengths) |
| A2 train on forecast where it exists, ERA5 elsewhere | V2 − V1 | -5.37 (-13.0 %) | [-8.62, -1.76] | [-8.65, -1.69] | conclusive (interval excludes 0, both block lengths) |
| A3 train ERA5, forecast input calibrated | Hybrid − V1 | -6.10 (-14.0 %) | [-8.85, -3.18] | [-8.66, -3.33] | conclusive (interval excludes 0, both block lengths) |
| A3 train ERA5, forecast input calibrated | V2 − V1 | -6.17 (-14.1 %) | [-8.91, -3.24] | [-8.76, -3.35] | conclusive (interval excludes 0, both block lengths) |
| S1 (sensitivity) like A2, today's weather also forecast | Hybrid − V1 | -4.32 (-10.7 %) | [-7.42, -0.68] | [-7.61, -0.36] | conclusive (interval excludes 0, both block lengths) |
| S1 (sensitivity) like A2, today's weather also forecast | V2 − V1 | -4.28 (-10.6 %) | [-7.38, -0.62] | [-7.55, -0.33] | conclusive (interval excludes 0, both block lengths) |
| REF oracle: tomorrow's OBSERVED weather (not live) | Hybrid − V1 | -4.60 (-11.6 %) | [-7.31, -1.71] | [-7.21, -1.78] | conclusive (interval excludes 0, both block lengths) |
| REF oracle: tomorrow's OBSERVED weather (not live) | V2 − V1 | -4.66 (-11.8 %) | [-7.39, -1.76] | [-7.30, -1.84] | conclusive (interval excludes 0, both block lengths) |
| REF 48 h forecast instead of 24 h | Hybrid − V1 | -4.61 (-11.1 %) | [-7.27, -1.72] | [-7.15, -1.79] | conclusive (interval excludes 0, both block lengths) |
| REF 48 h forecast instead of 24 h | V2 − V1 | -4.64 (-11.1 %) | [-7.31, -1.74] | [-7.19, -1.82] | conclusive (interval excludes 0, both block lengths) |

## 6. By winter

| Arm | Winter | Days | V1 MAE | V2 MAE | Hybrid MAE [95 %] | Hybrid RMSE | Hybrid exact cat. | Hybrid vs persistence | Hybrid ΔMAE vs A0 (7-d CI, µg/m³) |
|---|---|---|---|---|---|---|---|---|---|
| A0 reference: train ERA5, test forecast | 2024-25 | 124 | 49.20 | 44.38 | 44.38 [36.39, 57.56] | 60.40 | 68.5 % | +12.6 % | - |
| A0 reference: train ERA5, test forecast | 2025-26 | 104 | 33.26 | 28.57 | 28.66 [24.27, 35.81] | 38.81 | 72.1 % | +31.1 % | - |
| A1 train ONLY on forecast-weather origins | 2025-26 | 104 | 42.20 | 35.62 | 35.06 [27.88, 46.06] | 46.92 | 56.7 % | +15.7 % | +22.3 % [+2.37, +11.20] |
| A2 train on forecast where it exists, ERA5 elsewhere | 2024-25 | 124 | 46.03 | 41.97 | 41.97 [32.40, 56.29] | 59.87 | 72.6 % | +17.3 % | -5.4 % [-5.38, +0.69] |
| A2 train on forecast where it exists, ERA5 elsewhere | 2025-26 | 104 | 35.68 | 28.75 | 28.82 [23.55, 37.18] | 39.96 | 66.3 % | +30.7 % | +0.6 % [-1.93, +2.34] |
| A3 train ERA5, forecast input calibrated | 2024-25 | 124 | 45.44 | 40.12 | 40.12 [30.17, 56.24] | 62.55 | 79.0 % | +21.0 % | -9.6 % [-8.80, +0.73] |
| A3 train ERA5, forecast input calibrated | 2025-26 | 104 | 41.70 | 34.51 | 34.65 [26.82, 45.84] | 46.34 | 60.6 % | +16.7 % | +20.9 % [+1.08, +11.58] |
| S1 (sensitivity) like A2, today's weather also forecast | 2024-25 | 124 | 44.72 | 40.41 | 40.41 [30.82, 55.41] | 59.82 | 76.6 % | +20.4 % | -8.9 % [-6.92, -0.75] |
| S1 (sensitivity) like A2, today's weather also forecast | 2025-26 | 104 | 35.06 | 30.83 | 30.74 [25.23, 39.59] | 41.37 | 66.3 % | +26.1 % | +7.2 % [-0.03, +4.60] |
| REF oracle: tomorrow's OBSERVED weather (not live) | 2024-25 | 124 | 45.40 | 40.64 | 40.64 [32.64, 54.14] | 57.19 | 71.0 % | +20.0 % | -8.4 % [-5.15, -1.91] |
| REF oracle: tomorrow's OBSERVED weather (not live) | 2025-26 | 104 | 32.77 | 28.22 | 28.36 [24.06, 35.38] | 38.34 | 74.0 % | +31.8 % | -1.0 % [-1.33, +0.55] |
| REF 48 h forecast instead of 24 h | 2024-25 | 124 | 49.02 | 43.94 | 43.94 [35.81, 57.18] | 60.12 | 69.4 % | +13.5 % | -1.0 % [-1.83, +0.69] |
| REF 48 h forecast instead of 24 h | 2025-26 | 104 | 32.98 | 28.86 | 28.92 [24.53, 36.06] | 39.41 | 71.2 % | +30.5 % | +0.9 % [-0.76, +1.46] |

## 7. Sub-seasons and sudden-change days (Hybrid, pooled)

Stubble period = 15 Oct-30 Nov; deep winter = 1 Dec-15 Feb. Sudden-change day = |tomorrow − today| / today ≥ 30 %. Small subsets give wide intervals.

| Arm | Subset | Days | Hybrid MAE [95 %] | RMSE | Exact cat. | vs persistence | ΔMAE vs A0 (µg/m³ CI) |
|---|---|---|---|---|---|---|---|
| A0 reference: train ERA5, test forecast | stubble period 15Oct-30Nov | 90 | 41.95 [29.66, 61.28] | 61.39 | 66.7 % | +10.0 % | - |
| A0 reference: train ERA5, test forecast | deep winter 1Dec-15Feb | 138 | 34.11 [29.38, 39.30] | 44.22 | 72.5 % | +26.8 % | - |
| A0 reference: train ERA5, test forecast | sudden change days ge30pct | 71 | 55.48 [46.14, 69.31] | 70.20 | 60.6 % | +39.9 % | - |
| A0 reference: train ERA5, test forecast | stable days | 157 | 28.94 [23.94, 35.78] | 40.63 | 74.5 % | -11.7 % | - |
| A0 reference: train ERA5, test forecast | days with all 7 day parts | 227 | 37.26 [31.58, 45.51] | 51.77 | 70.5 % | +20.3 % | - |
| A2 train on forecast where it exists, ERA5 elsewhere | stubble period 15Oct-30Nov | 90 | 40.32 [26.49, 62.04] | 62.51 | 72.2 % | +13.5 % | -3.9 % [-5.18, +2.15] |
| A2 train on forecast where it exists, ERA5 elsewhere | deep winter 1Dec-15Feb | 138 | 33.13 [28.31, 39.02] | 43.31 | 68.1 % | +28.9 % | -2.9 % [-2.81, +1.58] |
| A2 train on forecast where it exists, ERA5 elsewhere | sudden change days ge30pct | 71 | 59.35 [48.97, 74.87] | 74.31 | 49.3 % | +35.7 % | +7.0 % [+1.19, +7.01] |
| A2 train on forecast where it exists, ERA5 elsewhere | stable days | 157 | 25.40 [20.30, 32.50] | 37.31 | 79.0 % | +1.9 % | -12.2 % [-6.13, -0.89] |
| A2 train on forecast where it exists, ERA5 elsewhere | days with all 7 day parts | 227 | 36.02 [29.86, 45.06] | 51.84 | 70.0 % | +22.9 % | -3.3 % [-3.11, +0.79] |
| A3 train ERA5, forecast input calibrated | stubble period 15Oct-30Nov | 90 | 46.00 [31.65, 69.32] | 71.31 | 73.3 % | +1.3 % | +9.7 % [-1.55, +11.70] |
| A3 train ERA5, forecast input calibrated | deep winter 1Dec-15Feb | 138 | 32.17 [26.79, 39.18] | 42.63 | 68.8 % | +30.9 % | -5.7 % [-6.19, +3.74] |
| A3 train ERA5, forecast input calibrated | sudden change days ge30pct | 71 | 55.80 [45.13, 70.78] | 73.15 | 60.6 % | +39.6 % | +0.6 % [-4.40, +5.19] |
| A3 train ERA5, forecast input calibrated | stable days | 157 | 29.41 [22.53, 38.78] | 45.75 | 75.2 % | -13.5 % | +1.6 % [-4.44, +6.10] |
| A3 train ERA5, forecast input calibrated | days with all 7 day parts | 227 | 37.60 [30.75, 47.64] | 55.78 | 70.9 % | +19.6 % | +0.9 % [-3.38, +4.59] |
| S1 (sensitivity) like A2, today's weather also forecast | stubble period 15Oct-30Nov | 90 | 41.31 [27.20, 63.50] | 65.09 | 75.6 % | +11.4 % | -1.5 % [-4.80, +3.88] |
| S1 (sensitivity) like A2, today's weather also forecast | deep winter 1Dec-15Feb | 138 | 32.53 [28.06, 38.20] | 41.74 | 69.6 % | +30.2 % | -4.6 % [-3.62, +1.26] |
| S1 (sensitivity) like A2, today's weather also forecast | sudden change days ge30pct | 71 | 57.59 [47.62, 72.41] | 72.99 | 53.5 % | +37.6 % | +3.8 % [-0.73, +5.19] |
| S1 (sensitivity) like A2, today's weather also forecast | stable days | 157 | 26.23 [20.95, 33.75] | 39.38 | 80.3 % | -1.2 % | -9.4 % [-5.63, +0.41] |
| S1 (sensitivity) like A2, today's weather also forecast | days with all 7 day parts | 227 | 36.06 [29.91, 45.29] | 52.32 | 72.2 % | +22.8 % | -3.2 % [-3.25, +1.11] |
| REF oracle: tomorrow's OBSERVED weather (not live) | stubble period 15Oct-30Nov | 90 | 40.67 [28.84, 59.85] | 60.28 | 70.0 % | +12.7 % | -3.1 % [-2.93, +0.53] |
| REF oracle: tomorrow's OBSERVED weather (not live) | deep winter 1Dec-15Feb | 138 | 31.37 [26.87, 36.48] | 40.94 | 73.9 % | +32.7 % | -8.0 % [-4.19, -1.31] |
| REF oracle: tomorrow's OBSERVED weather (not live) | sudden change days ge30pct | 71 | 53.21 [43.70, 66.95] | 68.21 | 60.6 % | +42.4 % | -4.1 % [-5.28, +0.41] |
| REF oracle: tomorrow's OBSERVED weather (not live) | stable days | 157 | 26.83 [22.27, 33.38] | 38.11 | 77.7 % | -3.6 % | -7.3 % [-3.53, -0.54] |
| REF oracle: tomorrow's OBSERVED weather (not live) | days with all 7 day parts | 227 | 35.06 [29.58, 43.18] | 49.55 | 72.7 % | +25.0 % | -5.9 % [-3.27, -1.02] |

On sudden-change days A2 is +7.0 % vs A0 ([+1.19, +7.01] µg/m³): worse, and the interval excludes zero. The pooled gain of A2 comes from ordinary days, not from the days that matter most for a school alert.

## 8. Is the A2 gain model noise? (5 random seeds)

Production uses one seed (42). XGBoost's row/column subsampling makes any single refit slightly different, so A0 and A2 were refitted with seeds [42, 7, 123, 2024, 99].

| Seed | A0 Hybrid MAE | A2 Hybrid MAE | A2 − A0 |
|---|---|---|---|
| 42 | 37.21 | 35.97 | -1.23 |
| 7 | 37.35 | 36.25 | -1.10 |
| 123 | 37.53 | 35.89 | -1.64 |
| 2024 | 37.52 | 35.97 | -1.54 |
| 99 | 37.37 | 35.69 | -1.68 |

A2 beats A0 in 5 of 5 seeds. Averaging the 5 seeds' predictions: -1.47 µg/m³ (-4.0 %), interval [-3.01, +0.30] (7-d) / [-3.02, +0.42] (14-d) - still includes zero. The direction is not a seed accident; what remains uncertain is whether it holds across different winters (it is a different size in each, section 6). This check was added after seeing the first result, so it is supporting evidence only and is not part of the decision rule.

## 9. Decision (rule fixed in advance)

A candidate needs Hybrid pooled MAE ≥ 2 % lower than A0, an interval below zero for both 7- and 14-day blocks, and no winter more than 2 % worse.

| Candidate | Winters | Hybrid ΔMAE vs A0 | 95 % 7-d / 14-d (µg/m³) | Worst winter | Verdict |
|---|---|---|---|---|---|
| A1 train ONLY on forecast-weather origins | 2025-26 | +22.3 % | [2.37, 11.20] / [2.24, 11.90] | +22.3 % | **worse** (interval above 0) |
| A2 train on forecast where it exists, ERA5 elsewhere | 2024-25, 2025-26 | -3.3 % | [-3.10, 0.78] / [-3.18, 0.88] | +0.6 % | no evidence of improvement |
| A3 train ERA5, forecast input calibrated | 2024-25, 2025-26 | +1.1 % | [-3.35, 4.63] / [-3.59, 5.17] | +20.9 % | no evidence of improvement |

**No candidate met the rule. Keep the existing approach (A0: train on observed weather); v1 stays the default and v2/Hybrid stay experimental.**

## 10. Interpretation

* **Statistically convincing?** No. A2 (the only arm with a consistent sign: -3.3 % pooled, [-3.10, +0.78] µg/m³) is -5.4 % in 2024-25 and +0.6 % in 2025-26, and its pooled interval includes zero. Seed noise does not explain it (section 8), but two winters of day-correlated data are too few to prove a 3 % effect.
* **Training only on forecasts (A1) is clearly worse** (+22.3 % in 2025-26, interval excludes zero): the archive begins in 2024, so the model loses the six earlier winters of history (161 Nov-Feb origins instead of 498). More history matters more than matching the weather source.
* **Calibrating the forecast (A3)** gave -9.6 % in 2024-25 and +20.9 % in 2025-26: it helps when the winter has the bias the correction learned and hurts when it does not (section 3). A fixed correction is unsafe until the bias is shown to be stable.
* **Ceiling.** Even a perfect weather forecast (oracle) changes the A0 Hybrid's MAE by -5.8 % ([-3.24, -0.97] µg/m³). That is the most any better tomorrow-weather input could add at test time; A2's -3.3 % is a different lever (training) and the two should not be added.
* **Bottleneck.** The mismatch the experiment targets is small and winter-dependent. The remaining error is probably dominated by what weather inputs can not explain (emissions, smoke from fires, local events) - this experiment does not measure that, it is the reason the next steps add new information. Sudden-change days carry the largest error; candidate arms change them by A2 +7.0 %, A3 +0.6 % (none improves them).

### Will it transfer to live predictions?

* Possibly a little, not provably. Live inputs are the newest forecast run (probably slightly better than the 24 h-old archive), so the true mismatch is likely smaller than tested here.
* Training on archived forecasts bakes in the behaviour of the forecast models of 2024-26; Open-Meteo changes its models over time, so the benefit can decay.
* The nightly Lambda does not store the weather it used (only a hash of the inputs), so there is no exact-live training set yet. Logging the forecast weather inputs each night, starting now, would give 2026-27 a clean, leak-free set for the next experiment. That is a backend change and needs your approval.

### Recommendation

Keep A0 (train on observed weather). Do not retrain or deploy anything on the basis of this experiment. A2 can enter the winter 2026-27 shadow test as a pre-declared challenger only if you approve; the cheaper and more informative step is to start logging the live forecast inputs. Next, as you planned: the OpenCity hourly-data experiment (more winters with day-part data), which targets the larger source of error.
