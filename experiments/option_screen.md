# Screening of untested accuracy options (exploratory)

> **Exploratory, retrospective.** These winters were examined before. A "promising" result is a hypothesis for the winter 2026-27 shadow test, not a reason to change the default. Rules were fixed before running (see `model/option_screen.py`). Data: `data/daily_opencity.csv` (production data + OpenCity gap-fill).

## 1. How accurate is it with a REAL weather forecast?

Models trained exactly as before (observed weather); only tomorrow's weather input changes. MAE µg/m³ / exact AQI category.

| Winter | Days | Tomorrow's weather | V1 | V2 | Hybrid |
|---|---|---|---|---|---|
| 2024-25 | 124 | today's weather reused (primary bound) | 47.48 / 65.3% | 42.66 / 66.9% | 42.66 / 66.9% |
| 2024-25 | 124 | real forecast, 48 h ahead | 49.02 / 66.9% | 43.94 / 69.4% | 43.94 / 69.4% |
| 2024-25 | 124 | real forecast, 24 h ahead | 49.2 / 65.3% | 44.38 / 68.5% | 44.38 / 68.5% |
| 2024-25 | 124 | observed (optimistic bound) | 45.4 / 68.5% | 40.64 / 71.0% | 40.64 / 71.0% |
| 2025-26 | 104 | today's weather reused (primary bound) | 36.42 / 65.4% | 29.56 / 71.2% | 29.7 / 71.2% |
| 2025-26 | 104 | real forecast, 48 h ahead | 32.98 / 66.3% | 28.86 / 71.2% | 28.92 / 71.2% |
| 2025-26 | 104 | real forecast, 24 h ahead | 33.26 / 64.4% | 28.57 / 72.1% | 28.66 / 72.1% |
| 2025-26 | 104 | observed (optimistic bound) | 32.77 / 69.2% | 28.22 / 74.0% | 28.36 / 74.0% |

**Hybrid − V1 (MAE µg/m³)** under each weather input:

| Winter | Weather | Hybrid − V1 | 95% CI 7-d | 95% CI 14-d | Excludes 0 (both) |
|---|---|---|---|---|---|
| 2024-25 | weather persistence | -4.81 (-10.1%) | [-8.19, -0.95] | [-8.39, -1.56] | yes |
| 2024-25 | forecast 24h | -4.83 (-9.8%) | [-8.69, -0.49] | [-8.70, -1.05] | yes |
| 2024-25 | observed weather | -4.76 (-10.5%) | [-8.24, -1.16] | [-8.20, -2.00] | yes |
| 2025-26 | weather persistence | -6.71 (-18.4%) | [-11.06, -2.24] | [-10.61, -2.08] | yes |
| 2025-26 | forecast 24h | -4.60 (-13.8%) | [-8.51, -0.15] | [-8.57, +0.43] | no |
| 2025-26 | observed weather | -4.41 (-13.4%) | [-8.54, +0.18] | [-8.49, +0.67] | no |
| both winters | forecast 24h | -4.73 (-11.3%) | [-7.55, -1.70] | [-7.54, -1.67] | yes |

Note: "today's weather reused" was described as a pessimistic bound in the backtest report. With real forecasts that holds for 2025-26 but NOT for 2024-25, where real forecasts gave a higher error. The archived forecasts differ from the archive (ERA5) weather the models are trained on when averaged over ALL seasons (wind lower by 2-4 km/h, humidity higher by 2-5 points); inside the winter test windows the difference is smaller and differs between winters (see experiments/forecast_training_experiment.md, section 3, which supersedes this sentence).

## 2. Options screened (Hybrid, frozen per winter)

Negative = the option has lower error than the reference Hybrid.

| Option | Days | Ref MAE | Option MAE | Difference | 95% CI 7-d / 14-d blocks | Exact category | Worst winter | Verdict |
|---|---|---|---|---|---|---|---|---|
| O1 error correction | 399 | 40.35 | 40.37 | +0.01 (+0.0%) | [-0.01, +0.04] / [-0.00, +0.04] | 66.9% → 66.9% | +0.1% | no evidence |
| O2 recency weight | 399 | 40.35 | 40.17 | -0.18 (-0.5%) | [-0.73, +0.38] / [-0.72, +0.37] | 66.9% → 67.7% | +0.3% | no evidence |
| O3 monthly refit | 399 | 40.35 | 40.57 | +0.21 (+0.5%) | [-0.35, +0.80] / [-0.31, +0.70] | 66.9% → 67.2% | +1.3% | no evidence |
| O4 spatial features | 399 | 40.35 | 40.3 | -0.05 (-0.1%) | [-0.64, +0.56] / [-0.64, +0.56] | 66.9% → 69.2% | +3.3% | no evidence |
| O5 train on forecasts | 228 | 37.21 | 35.97 | -1.23 (-3.3%) | [-3.10, +0.78] / [-3.18, +0.88] | 70.2% → 69.7% | +0.6% | no evidence |
| O6 forecast bias fix | 228 | 37.21 | 36.34 | -0.87 (-2.3%) | [-3.99, +2.60] / [-3.82, +2.50] | 70.2% → 72.4% | +7.3% | no evidence |

### Per winter

| Option | Winter | Days | Ref MAE | Option MAE | Difference | 95% CI (7-d) |
|---|---|---|---|---|---|---|
| O1 error correction | 2022-23 | 47 | 42.48 | 42.48 | -0.00 (+0.0%) | [-0.00, +0.00] |
| O1 error correction | 2023-24 | 124 | 46.16 | 46.21 | +0.05 (+0.1%) | [-0.03, +0.13] |
| O1 error correction | 2024-25 | 124 | 42.66 | 42.66 | +0.00 (+0.0%) | [-0.00, +0.00] |
| O1 error correction | 2025-26 | 104 | 29.7 | 29.7 | -0.00 (-0.0%) | [-0.00, +0.00] |
| O2 recency weight | 2022-23 | 47 | 42.48 | 42.62 | +0.14 (+0.3%) | [-1.51, +2.54] |
| O2 recency weight | 2023-24 | 124 | 46.16 | 46.09 | -0.08 (-0.2%) | [-0.79, +0.80] |
| O2 recency weight | 2024-25 | 124 | 42.66 | 42.55 | -0.11 (-0.3%) | [-1.29, +1.18] |
| O2 recency weight | 2025-26 | 104 | 29.7 | 29.17 | -0.53 (-1.8%) | [-1.31, +0.38] |
| O3 monthly refit | 2022-23 | 47 | 42.48 | 42.97 | +0.49 (+1.2%) | [-1.64, +2.38] |
| O3 monthly refit | 2023-24 | 124 | 46.16 | 46.42 | +0.26 (+0.6%) | [-0.82, +1.51] |
| O3 monthly refit | 2024-25 | 124 | 42.66 | 42.59 | -0.07 (-0.2%) | [-1.26, +1.15] |
| O3 monthly refit | 2025-26 | 104 | 29.7 | 30.08 | +0.38 (+1.3%) | [-0.22, +0.96] |
| O4 spatial features | 2022-23 | 47 | 42.48 | 42.32 | -0.16 (-0.4%) | [-1.35, +1.27] |
| O4 spatial features | 2023-24 | 124 | 46.16 | 46.34 | +0.18 (+0.4%) | [-0.79, +1.19] |
| O4 spatial features | 2024-25 | 124 | 42.66 | 41.57 | -1.09 (-2.6%) | [-2.11, -0.08] |
| O4 spatial features | 2025-26 | 104 | 29.7 | 30.68 | +0.98 (+3.3%) | [-0.22, +2.28] |
| O5 train on forecasts | 2024-25 | 124 | 44.38 | 41.97 | -2.40 (-5.4%) | [-5.38, +0.69] |
| O5 train on forecasts | 2025-26 | 104 | 28.66 | 28.82 | +0.16 (+0.6%) | [-1.93, +2.34] |
| O6 forecast bias fix | 2024-25 | 124 | 44.38 | 41.04 | -3.34 (-7.5%) | [-8.01, +1.78] |
| O6 forecast bias fix | 2025-26 | 104 | 28.66 | 30.74 | +2.08 (+7.3%) | [-1.39, +5.70] |

### Fitted settings (from data before each window only)

* 2022-23: 47 days (2 with day-parts); error-correction alpha {'v1_25_baseline': 0.0, 'v2_35_candidate': 0.0}
* 2023-24: 124 days (45 with day-parts); error-correction alpha {'v1_25_baseline': 0.0, 'v2_35_candidate': 0.012}
* 2024-25: 124 days (124 with day-parts); error-correction alpha {'v1_25_baseline': 0.0, 'v2_35_candidate': 0.0}; forecast bias removed {'temp': 0.49, 'rh': 2.45, 'wind': -4.28, 'precip': -0.45, 'pressure': -0.15, 'wind_u': -2.05, 'wind_v': 1.15} (from 269 days)
* 2025-26: 104 days (103 with day-parts); error-correction alpha {'v1_25_baseline': 0.0, 'v2_35_candidate': 0.0}; forecast bias removed {'temp': -0.2, 'rh': 4.69, 'wind': -2.55, 'precip': 0.61, 'pressure': -0.03, 'wind_u': -1.4, 'wind_v': 0.51} (from 634 days)
