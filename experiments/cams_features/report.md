# Does CAMS forecast data improve the Hybrid model? (isolated experiment)

**Result: a realistic test is not possible with the data available, and the non-realistic diagnostics do not show a reliable improvement. Keep the production model unchanged.** Nothing in the project was modified, committed, pushed or deployed.

## 1. Why a realistic test could not be run

* The candidate needs the CAMS PM2.5 forecast for tomorrow *as it was issued before 23:00 IST today*. Open-Meteo's air-quality API documents no previous-run, historical-forecast or single-run option, and I probed it: `pm2_5_previous_day1/2` returned **null for every hour in all 9 periods tested** (Sep 2022, Jan 2023, Jun 2024, Jan 2025, Oct 2025, Jan/Apr/Jul/Oct 2026), while plain `pm2_5` is always filled. The previous-runs service has no air-quality route (HTTP error).
* The only CAMS history in the project (`data/raw/cams_daily.csv`, 2022-08-04 onwards) was fetched with start/end dates, which returns Open-Meteo's stitched *past* series (the early hours of each model run), not a day-ahead forecast. For a target day it carries information from runs issued during that day, which the Lambda cannot have at 23:00 the night before. It has never been used as a model input in this project; it is only a benchmark (CAMS alone: MAE 49.99 in the earlier final test).
* True archived CAMS forecasts exist at the Copernicus Atmosphere Data Store (ADS), which needs an account and API key; none is available in this session and I did not look for one.

So I did **not** build a "realistic" candidate and I am not presenting any result below as a forecast-time accuracy. I ran two clearly labelled diagnostics to find out whether CAMS carries any usable signal at all.

## 2. Diagnostics (not realistic; same eligible days, same real PM2.5, same weather forecast inputs, same fixed hyper-parameters, frozen fit before the test)

* **O_target_day (optimistic upper bound):** adds the stitched CAMS mean of the *target day*, the origin day and their log-change. It knows something about the day being predicted that a 23:00 forecast would only approximate.
* **O_lags_only (nearly realistic):** only the origin-day and previous-day CAMS means. Residual caveat: the origin-day mean includes hours after 22:59 IST.

### calendar_2025 (train <= 2024-12-31): 344 identical test days (2025-01-01 → 2025-12-31)

Training origins 1184 (only 737 have CAMS: it starts 2022-08-04; the rest are left missing, handled natively by XGBoost). CAMS alone on these days: MAE 49.1, mean bias -25.9 µg/m³, correlation with next-day PM2.5 0.72 (origin-day CAMS: 0.64).

| model | MAE | RMSE | R² | exact category |
|---|---|---|---|---|
| persistence | 24.90 | 39.8 | 0.78 | 58.1% |
| **current Hybrid (production recipe)** | 19.57 | 30.2 | 0.87 | 66.3% |
| Hybrid + CAMS, O_lags_only | 19.87 | 31.2 | 0.87 | 68.6% |
| Hybrid + CAMS, O_target_day | 18.15 | 28.0 | 0.89 | 71.8% |

| candidate − current Hybrid | Δ MAE | Δ % | 95% CI (7-d blocks) | 95% CI (14-d blocks) | Δ RMSE | Δ exact-cat (pts) | seeds better (of 5) | sudden-change MAE (Hybrid → candidate) | stable MAE |
|---|---|---|---|---|---|---|---|---|---|
| O_lags_only | +0.30 | +1.5% | [-0.27, +0.97] | [-0.15, +0.89] | +1.04 | +2.3 | 1 | 28.6 → 30.3 (n=106) | 15.6 → 15.2 |
| O_target_day | -1.42 | -7.2% | [-2.99, -0.01] | [-3.07, +0.07] | -2.22 | +5.5 | 5 | 28.6 → 26.7 (n=106) | 15.6 → 14.3 |

### winter_2025_26 (train < 2025-10-15): 104 identical test days (2025-10-15 → 2026-02-15)

Training origins 1454 (only 1007 have CAMS: it starts 2022-08-04; the rest are left missing, handled natively by XGBoost). CAMS alone on these days: MAE 87.6, mean bias -86.9 µg/m³, correlation with next-day PM2.5 0.62 (origin-day CAMS: 0.47).

| model | MAE | RMSE | R² | exact category |
|---|---|---|---|---|
| persistence | 41.60 | 53.6 | 0.42 | 61.5% |
| **current Hybrid (production recipe)** | 28.66 | 38.8 | 0.70 | 72.1% |
| Hybrid + CAMS, O_lags_only | 29.65 | 39.8 | 0.68 | 68.3% |
| Hybrid + CAMS, O_target_day | 27.85 | 37.1 | 0.72 | 69.2% |

| candidate − current Hybrid | Δ MAE | Δ % | 95% CI (7-d blocks) | 95% CI (14-d blocks) | Δ RMSE | Δ exact-cat (pts) | seeds better (of 5) | sudden-change MAE (Hybrid → candidate) | stable MAE |
|---|---|---|---|---|---|---|---|---|---|
| O_lags_only | +0.99 | +3.4% | [-0.18, +2.21] | [-0.05, +2.16] | +0.95 | -3.8 | 1 | 45.3 → 47.0 (n=29) | 22.2 → 22.9 |
| O_target_day | -0.81 | -2.8% | [-3.02, +1.39] | [-2.94, +0.86] | -1.68 | -2.9 | 5 | 45.3 → 39.1 (n=29) | 22.2 → 23.5 |

## 3. Reading

* **The nearly-realistic arm gives no improvement:** adding origin-day CAMS lags changes MAE by +0.30 (2025) and +0.99 (winter 2025-26) µg/m³, i.e. slightly worse, with the candidate better in only 1/5 and 1/5 seeds.
* **Even the optimistic target-day arm is not convincing:** -1.42 µg/m³ (-7.2%) over 2025 with intervals [-2.99, -0.01] / [-3.07, +0.07] (the 14-day interval includes zero), and -0.81 µg/m³ (-2.8%, interval [-3.02, +1.39] includes zero) in winter 2025-26, where exact-category accuracy falls 2.9 points (it rose 5.5 points over the full year, so even that direction is not consistent). This arm uses information a real forecast would only approximate, so a realistic CAMS forecast would be expected to do no better.
* CAMS on its own is a poor predictor of Delhi PM2.5 (bias −26 µg/m³ in 2025, −87 in winter 2025-26), so any benefit would have to come from the model learning a correction, which the diagnostics do not demonstrate.

## 4. Limitations
* Retrospective; 2025 and winter 2025-26 were examined before. CAMS exists only from 2022-08-04, so ~62% of training origins have it. Stitched CAMS differs from real forecasts. The training and test CAMS definitions match each other, but neither equals what the Lambda would receive. One point (28.6139, 77.2090; the API snaps to ~28.60, 77.20). No tuning on the test period.

## 5. Recommendation

1. **Do not change the production model, default, dashboard or Lambda.** There is no demonstrated improvement, and the realistic condition cannot be tested retrospectively.
2. If you still want a genuine answer: either obtain archived CAMS forecasts from the Copernicus ADS (needs your account), or, with your approval, have the Lambda *log* tomorrow's CAMS forecast at 23:00 IST during winter 2026-27 as a shadow field (no effect on predictions) and test it prospectively. That is a production change, so I have not made it.
3. Higher-value ideas are still open: the OpenCity hourly-data experiment and the forward-test freeze.
