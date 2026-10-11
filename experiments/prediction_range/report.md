# Prediction range around the Hybrid forecast (branch `feature/prediction-range`)

**RETROSPECTIVE.** The point forecast is the unchanged Hybrid. The range is added around it; it does not change the point forecast's MAE. Nothing in the production model, default, dashboard or Lambda was modified. Protocol: `config.json` (written before running).

## 1. Answer in short

* **What "85" can mean, and what the data say** (Hybrid, chosen method M3, nominal 90% range):

| reading of "85%" | calendar 2025 (344 days) | winter 2025-26 (104 days) | meets 85%? |
|---|---|---|---|
| exact AQI category of a single value (existing point forecast) | 66.3% | 72.1% | no |
| **range accuracy**: actual category inside the predicted category range | 98.5% [97.0, 99.7] | 98.1% [95.1, 100.0] | yes |
| school decision GO/CAUTION/INDOOR from the point forecast | 86.3% [80.8, 90.9] | 98.1% [95.7, 100.0] | yes in both, but this needs no range |
* **The declared gate (range accuracy ≥ 85% and fewer than 3 categories in the range, both test sets) is technically passed** (98.5% at 2.91 categories; 98.1% at 2.61). **But the gate is weak and I do not want to overstate it:** the naive range "point category ± 1" already gets 98.5% at 2.85 categories in 2025. The selected method M3 is not narrower than that naive range (it over-covers: PM2.5 coverage 96.2% against a 90% target). So the selected range adds honesty about uncertainty, not extra accuracy.
* **A better-behaved choice is M1** (simple conformal on log residuals): PM2.5 coverage 93.3% / 90.4% (target 90%), range accuracy 98.0% / 98.1% at 2.45 / 2.41 categories, i.e. narrower than the naive ±1 range. At 80% nominal: range accuracy 97.4% at 2.18 categories. M3 was chosen by the pre-declared rule (closest to nominal coverage on a 2024 split) and I kept that choice; M1 is reported alongside because it generalised better.
* **Ranges are wide.** A 90% range spans 2–3 of the 6 AQI categories; at most about 10% of days get a single-category range. In winter the mean width is about 145 µg/m³ (M1) because uncertainty grows with the pollution level.
* **Where the range is genuinely useful: the school decision.** The decision range (GO/CAUTION/INDOOR spanned by the range) contained the actual decision on 100.0% of 2025 days. Adopting the *most cautious* decision in the range cuts under-warnings (the forecast says GO/CAUTION but the day is worse) from 4.1% to 0.0%, at the cost of more over-warnings (9.6% → 43.0%) and lower plain decision accuracy (86.3% → 57.0%). That is a policy choice for you, not a free improvement.

## 2. Method

* Point forecast: Hybrid, production recipe, seed 42 (as in `experiments/backtest_2025`).
* Ranges in log space around the point forecast. **M1** global conformal quantiles of log1p residuals; **M2** per predicted-level bucket; **M3** XGBoost quantile regression (5%/95%) with conformal correction (CQR).
* **Calibration is strictly earlier and out of sample:** for 2025, residuals from 2024 (models trained to 2023-12-31; archived 24 h forecast weather; 347 days from 2024-01-20); for winter 2025-26, 2024 plus Jan–Oct 2025 residuals (617 days). Days without archived forecast weather are not used, and nothing is substituted. Calibration always ends before the test starts (checked in the tests).
* The method was chosen on a split of the 2024 calibration period only, never on test results. Nominal level 90% was declared in advance; 80% and 95% are sensitivity checks for M1/M2.
* Intervals use a moving-block bootstrap (7-day blocks; 14-day in `results.json`). Both test sets reuse days already examined in earlier experiments.

## 3a. Calendar 2025 (344 days, calibration 347 days ending 2024-12-31)

| method / nominal | PM2.5 coverage | mean width µg/m³ | range accuracy (95% CI) | categories in range | single-category days | naive ±1: accuracy / categories | decision range contains actual | under-warning point → cautious | over-warning point → cautious |
|---|---|---|---|---|---|---|---|---|---|
| M1 90% | 93.3% | 91 | 98.0% [96.2, 99.4] | 2.45 | 1.5% | 98.5% / 2.85 | 99.7% | 4.1% → 0.3% | 9.6% → 32.6% |
| M1 80% | 84.9% | 69 | 97.4% [95.4, 98.9] | 2.18 | 4.9% | 98.5% / 2.85 | 99.1% | 4.1% → 0.6% | 9.6% → 26.5% |
| M1 95% | 95.6% | 112 | 98.5% [97.0, 99.7] | 2.75 | 0.9% | 98.5% / 2.85 | 100.0% | 4.1% → 0.0% | 9.6% → 39.8% |
| M2 90% | 93.3% | 90 | 98.0% [96.2, 99.4] | 2.50 | 0.6% | 98.5% / 2.85 | 99.7% | 4.1% → 0.3% | 9.6% → 41.3% |
| M2 80% | 84.6% | 67 | 96.5% [94.5, 98.3] | 2.13 | 6.1% | 98.5% / 2.85 | 99.4% | 4.1% → 0.3% | 9.6% → 28.8% |
| M2 95% | 95.6% | 110 | 98.3% [96.5, 99.7] | 2.74 | 0.3% | 98.5% / 2.85 | 99.7% | 4.1% → 0.3% | 9.6% → 44.8% |
| M3 90% ← selected | 96.2% | 121 | 98.5% [97.0, 99.7] | 2.91 | 0.3% | 98.5% / 2.85 | 100.0% | 4.1% → 0.0% | 9.6% → 43.0% |

Selected method on sudden-change days (n=106): range accuracy 95.3%, 2.96 categories; stable days (n=238): 100.0%, 2.88 categories.

## 3b. Winter 2025-26 (104 days, calibration 617 days ending 2025-10-14)

| method / nominal | PM2.5 coverage | mean width µg/m³ | range accuracy (95% CI) | categories in range | single-category days | naive ±1: accuracy / categories | decision range contains actual | under-warning point → cautious | over-warning point → cautious |
|---|---|---|---|---|---|---|---|---|---|
| M1 90% | 90.4% | 145 | 98.1% [95.1, 100.0] | 2.41 | 0.0% | 99.0% / 2.87 | 100.0% | 0.0% → 0.0% | 1.9% → 2.9% |
| M1 80% | 83.7% | 109 | 97.1% [93.8, 100.0] | 2.11 | 9.6% | 99.0% / 2.87 | 100.0% | 0.0% → 0.0% | 1.9% → 2.9% |
| M1 95% | 94.2% | 182 | 100.0% [100.0, 100.0] | 2.75 | 0.0% | 99.0% / 2.87 | 100.0% | 0.0% → 0.0% | 1.9% → 2.9% |
| M2 90% | 91.3% | 145 | 99.0% [96.8, 100.0] | 2.44 | 0.0% | 99.0% / 2.87 | 100.0% | 0.0% → 0.0% | 1.9% → 2.9% |
| M2 80% | 83.7% | 108 | 97.1% [93.8, 100.0] | 2.13 | 7.7% | 99.0% / 2.87 | 100.0% | 0.0% → 0.0% | 1.9% → 2.9% |
| M2 95% | 94.2% | 181 | 99.0% [96.8, 100.0] | 2.71 | 0.0% | 99.0% / 2.87 | 100.0% | 0.0% → 0.0% | 1.9% → 2.9% |
| M3 90% ← selected | 96.2% | 192 | 98.1% [95.1, 100.0] | 2.61 | 0.0% | 99.0% / 2.87 | 100.0% | 0.0% → 0.0% | 1.9% → 2.9% |

Selected method on sudden-change days (n=29): range accuracy 93.1%, 2.79 categories; stable days (n=75): 100.0%, 2.53 categories.

## 4. Limitations
* Only 2024–25 residuals are available for calibration (forecast archive starts 2024-01-19), and 2024 had a 4-station network while 2025 used 9–10 stations, so calibration and test targets are not identically defined (same caveat as the 2025 backtest).
* Retrospective; the test days were examined before. 104 winter days give wide intervals.
* Ranges are not conditioned on season or weather regime beyond the predicted level. The most useful regime-specific ranges (stubble-burning season) would need crop-fire data, which the project does not have; that feature was deliberately not attempted and is deferred until you supply data.
* Coverage is marginal (averaged over days), not guaranteed on every season or on sudden-change days.

## 5. Recommendation

1. The range is **sound and honest**, and by the gate you set (85% range accuracy) it passes. Treat it as an **uncertainty display and a safety option**, not an accuracy gain: the point-forecast accuracy numbers do not change.
2. If you want it in the product, use **M1 at 90%** (or show the 80% range for a narrower band) and state plainly that ranges are wide and usually span 2–3 AQI categories. Show the cautious school decision only if you accept ~43.0% over-warning days in return for under-warnings near zero.
3. I have **not** changed the backend, dashboard or Lambda. Doing so is a production-facing change; tell me which method and level to build, and I will implement it on this branch with tests, leaving V1 as the default point forecast.
