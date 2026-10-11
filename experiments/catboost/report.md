# CatBoost vs XGBoost for next-day Delhi PM2.5 - isolated experiment

> **Retrospective and isolated.** Winters 2022-23 → 2025-26 were examined in earlier work, so none of this is an untouched hold-out; the clean test is still the winter 2026-27 forward test, which was not read. Everything lives in `experiments/catboost/`; no existing model, default, feature, checksum, Lambda, dashboard or Git state was changed. Rules were fixed in `config.json` before any result was looked at.

## Summary

**Verdict under the declared rule: no evidence of improvement: keep XGBoost; CatBoost stays a documented reference.** The evidence is mixed rather than negative: a small CatBoost edge appears in several looks but is not stable (see 'What the evidence says').

* Primary comparison (tuned CatBoost Hybrid vs XGBoost Hybrid, 399 days over 4 winters, leak-free weather): MAE 40.35 → 39.45 µg/m³ (-2.2 %), 95 % interval [-2.01, 0.14] (7-d) / [-2.05, 0.13] (14-d) - **inconclusive**.
* The untuned CatBoost baseline: -2.0 % ([-1.98, 0.33]) - inconclusive.
* Development folds (before 15 Oct 2022, used for tuning): tuned CatBoost 24.99 vs XGBoost 25.93 µg/m³ (-3.6 %, [-2.07, 0.04]).
* Rule checks: MAE at least 2 % lower than XGBoost Hybrid: yes; 95 % interval below zero for BOTH 7- and 14-day blocks: NO; no single winter more than 2 % worse: NO; exact-category accuracy not lower by more than 2 points: yes; development-fold MAE not worse than XGBoost: yes.

## What the evidence says

| Look at the data | CatBoost tuned Hybrid − XGBoost Hybrid, MAE | 95 % (7-d), µg/m³ | Verdict |
|---|---|---|---|
| Development folds (before Oct 2022), tuned CatBoost vs XGBoost | -3.6 % | [-2.07, 0.04] | inconclusive |
| T1 four winters, seed 42 (the declared primary comparison) | -2.2 % | [-2.01, 0.14] | inconclusive |
| T1 four winters, 5 seeds averaged per library (robustness) | -2.7 % | [-2.08, -0.14] | conclusive |
| T2 real 24 h forecast weather, 2024-25 + 2025-26 only | -4.3 % | [-3.08, -0.30] | conclusive |
| T3 all seasons, 2025-01 → 2026-10 | -0.5 % | [-0.51, 0.33] | inconclusive |
| S1 production data only, four winters | -1.6 % | [-1.80, 0.32] | inconclusive |

Per winter (T1): 2022-23 +3.9 %, 2023-24 -2.9 %, 2024-25 -5.0 %, 2025-26 -0.3 %. All look-ups use the same data, so they are not independent confirmations.

**Reading.** CatBoost is better than XGBoost in most looks, by about 1-5 % (largest in 2024-25), but not everywhere, and the gain is small, not stable across windows (it is worse in 2022-23, the smallest window, and about zero over all seasons in T3), and the declared primary test misses the rule on two counts (the interval touches zero; 2022-23 is more than 2 % worse). The seed-averaged and real-forecast looks are supporting evidence added to describe the noise and were not the decision test. Fairness note: CatBoost had a 16-configuration tuning budget, while XGBoost is the project's frozen production recipe (an earlier 30-configuration random search found no meaningful gain for XGBoost either). Part of the dev-fold gap may be tuning budget rather than the algorithm. On sudden-change days - the days that matter for a school alert - the two are indistinguishable.

## Answers to the five questions

1. **Existing project unchanged?** Yes - all 144 protected files (models, checksums, data, code, docs, tests) are byte-identical; nothing added or removed outside experiments/catboost/.
2. **Existing tests still pass?** Yes - all 123 of the project's tests pass; 9 experiment tests pass.
3. **Results vs V1, V2, Hybrid, persistence:** section 3 (pooled), then sections 4-8 (by winter, season, hourly-data availability, sudden changes, real forecast weather, all seasons, robustness).
4. **Convincing?** No - it does not meet the pre-declared rule (section 9); the evidence is mixed and small (section 'What the evidence says').
5. **Commands to reproduce:** section 12.

## 1. Design

* **Data.** Primary: `data/daily_opencity.csv` (production data plus OpenCity gap-fill, which supplies hourly day-part inputs for 2024-25). Secondary sensitivity: `data/daily.csv`, the data the deployed models were trained on.
* **Same rows, target, features.** Every model is trained on the same forecast origins (target before the window start), the same target (log-change from today's PM2.5), the same 25 / 35 feature lists, and the same chronological early-stopping set (last 60 training rows). CatBoost gets no extra feature, no categorical declaration and no imputation; day-part values that do not exist stay missing and both libraries handle that natively.
* **Frozen fit.** One fit per window, no refit, no early stopping on test days. Stored model files were not used or changed; frozen copies are retrained in memory.
* **Weather at test time (no future observed weather).** T1: tomorrow's weather := today's observed weather. T2: the real archived 24 h-ahead forecast (2024-25 and 2025-26 only). Tomorrow's OBSERVED weather is never used as a test input.
* **Same eligible days for every model** (eligibility ladder: target and origin PM2.5, weather, origin usable by the 23:00 Lambda).
* **Tuning on development data only.** Grid of 16 CatBoost configurations scored by blocked forward-chaining CV (6 folds, never shuffled) on 459 rows with target 2018-01-05 → 2022-10-14 (all before the first test window, 15 Oct 2022); V1 features only because the development years have no hourly day-parts, and V2 reuses the same settings. The configuration was frozen before the test stage (`--stage test` refuses to run if `tuning.json` was made from a different `config.json`).
* **Seeds.** Primary seed 42 for every model; five seeds (42, 7, 123, 2024, 99) were refitted to measure model noise.
* **Intervals.** Moving-block bootstrap (7-day blocks, 14-day sensitivity, 5000 resamples) on the same days for all models; pooled winters are placed 200 days apart so blocks never straddle two winters. Sudden-change day = |tomorrow − today| / today ≥ 30 %.
* **Coverage.** Winters 2022-23 → 2025-26 are the only ones the project's frozen-fit protocol can score (the earlier ones lack hourly origins or have fewer than 365 training rows). Seasons other than the 15 Oct → 15 Feb window are covered only by T3 (section 7).

**Sanity check - the XGBoost arm reproduces the project's existing back-test:**

| Winter | This run (XGBoost Hybrid MAE) | Existing `option_screen.json` | Match |
|---|---|---|---|
| 2022-23 | 42.48 | 42.48 | yes |
| 2023-24 | 46.16 | 46.16 | yes |
| 2024-25 | 42.66 | 42.66 | yes |
| 2025-26 | 29.7 | 29.7 | yes |

## 2. Development tuning (nothing from any test winter)

CatBoost configurations ranked by pooled out-of-fold MAE (µg/m³), V1 features, 390 scored rows. The declared baseline is `d4_lr0.03_l23_Bayesian`.

| Configuration | OOF MAE | OOF RMSE |  |
|---|---|---|---|
| d6_lr0.03_l210_Bayesian | 24.991 | 37.51 | **selected** |
| d6_lr0.06_l210_Bayesian | 25.016 | 37.88 |  |
| d6_lr0.03_l23_Bayesian | 25.109 | 37.82 |  |
| d4_lr0.03_l23_Bayesian | 25.170 | 37.70 | baseline |
| d4_lr0.03_l210_Bayesian | 25.263 | 37.93 |  |
| d4_lr0.03_l210_Bernoulli | 25.351 | 37.83 |  |
| XGBoost (production recipe, not tuned) | 25.933 | 38.75 | reference |
| Tomorrow = today | 28.594 | 42.98 | reference |

Tuned − XGBoost on the development folds: -0.94 µg/m³ (-3.6 %), 95 % [-2.07, 0.04] (7-d) / [-2.07, -0.02] (14-d). Baseline − XGBoost: -2.9 %. Tuned − baseline: -0.7 % ([-0.67, 0.27]), so tuning added little beyond the declared baseline. Picking the best of 16 configurations on the same folds is optimistic by construction, which is why the test windows decide, not these folds.

## 3. Primary test T1: four winters, frozen fit, leak-free weather

Windows scored: 2022-23, 2023-24, 2024-25, 2025-26; 399 matched days, of which 274 have all 7 day-part inputs (Hybrid routes the others to V1). Persistence = tomorrow equals today.

| Model | MAE µg/m³ [95 %] | RMSE µg/m³ [95 %] | R² [95 %] | Exact category [95 %] | Within one category | Better than persistence [95 %] |
|---|---|---|---|---|---|---|
| Tomorrow = today | 47.91 [42.87, 53.82] | 66.10 [58.56, 74.41] | 0.435 [0.271, 0.534] | 58.9 % [52.9, 64.7] | 96.7 % | - |
| 7-day mean | 58.32 [52.26, 65.36] | 76.00 [66.94, 85.51] | 0.253 [0.068, 0.366] | 53.6 % [47.5, 59.9] | 94.0 % | -21.7 % [-33.5, -10.8] |
| XGBoost V1 (25 feat.) | 44.36 [39.82, 49.77] | 60.79 [53.46, 68.87] | 0.522 [0.395, 0.600] | 64.4 % [59.5, 69.1] | 97.2 % | +7.4 % [2.2, 12.0] |
| CatBoost (baseline) V1 (25 feat.) | 44.31 [39.61, 49.78] | 61.09 [53.67, 69.07] | 0.517 [0.384, 0.599] | 63.9 % [58.6, 69.1] | 97.0 % | +7.5 % [3.2, 11.3] |
| CatBoost (tuned) V1 (25 feat.) | 43.92 [39.19, 49.48] | 60.54 [53.03, 68.68] | 0.526 [0.395, 0.604] | 63.9 % [58.5, 69.1] | 97.0 % | +8.3 % [4.1, 12.1] |
| XGBoost V2 (35 feat.) | 41.62 [36.93, 47.37] | 57.42 [49.80, 66.18] | 0.574 [0.455, 0.648] | 63.7 % [58.2, 68.6] | 97.5 % | +13.1 % [7.0, 18.9] |
| CatBoost (baseline) V2 (35 feat.) | 40.63 [36.11, 46.12] | 56.81 [49.66, 64.45] | 0.583 [0.451, 0.662] | 67.2 % [62.0, 72.2] | 98.2 % | +15.2 % [9.2, 20.7] |
| CatBoost (tuned) V2 (35 feat.) | 40.88 [36.27, 46.43] | 56.65 [49.69, 64.18] | 0.585 [0.457, 0.663] | 68.4 % [62.8, 73.2] | 98.2 % | +14.7 % [8.5, 20.4] |
| XGBoost Hybrid | 40.35 [35.91, 45.82] | 56.16 [48.51, 64.88] | 0.592 [0.485, 0.663] | 66.9 % [62.1, 71.1] | 98.0 % | +15.8 % [9.9, 21.1] |
| CatBoost (baseline) Hybrid | 39.54 [35.15, 44.86] | 55.13 [48.16, 62.67] | 0.607 [0.494, 0.678] | 68.7 % [63.5, 73.6] | 98.0 % | +17.5 % [11.8, 22.5] |
| CatBoost (tuned) Hybrid | 39.45 [34.98, 44.79] | 54.84 [47.93, 62.33] | 0.611 [0.496, 0.682] | 70.4 % [65.1, 75.1] | 98.0 % | +17.7 % [12.0, 22.8] |

### Difference from XGBoost (same version, same days)

Negative ΔMAE / ΔRMSE and positive ΔR² / Δ accuracy favour CatBoost.

| Comparison | ΔMAE | 95 % (7-d) | 95 % (14-d) | ΔRMSE [95 %] | ΔR² [95 %] | Δ exact category [95 %] | Δ within-one | Verdict |
|---|---|---|---|---|---|---|---|---|
| CatBoost (baseline) V1 (25 feat.) − XGBoost | -0.05 (-0.1 %) | [-0.98, 0.85] | [-0.97, 0.79] | +0.31 [-0.64, 1.20] | -0.005 [-0.021, 0.010] | -0.5 pts [-2.8, 1.6] | -0.3 pts | inconclusive |
| CatBoost (baseline) V2 (35 feat.) − XGBoost | -0.99 (-2.4 %) | [-2.16, 0.18] | [-2.15, 0.10] | -0.61 [-2.97, 1.49] | +0.009 [-0.025, 0.039] | +3.5 pts [0.8, 6.9] | +0.8 pts | inconclusive |
| CatBoost (baseline) Hybrid − XGBoost | -0.81 (-2.0 %) | [-1.98, 0.33] | [-1.98, 0.30] | -1.03 [-3.30, 0.81] | +0.015 [-0.013, 0.043] | +1.8 pts [-0.6, 4.7] | +0.0 pts | inconclusive |
| CatBoost (tuned) V1 (25 feat.) − XGBoost | -0.44 (-1.0 %) | [-1.38, 0.46] | [-1.40, 0.45] | -0.24 [-1.10, 0.58] | +0.004 [-0.009, 0.018] | -0.5 pts [-2.7, 1.6] | -0.3 pts | inconclusive |
| CatBoost (tuned) V2 (35 feat.) − XGBoost | -0.74 (-1.8 %) | [-1.86, 0.36] | [-1.91, 0.32] | -0.78 [-3.20, 1.24] | +0.011 [-0.021, 0.043] | +4.8 pts [2.0, 7.7] | +0.8 pts | inconclusive |
| CatBoost (tuned) Hybrid − XGBoost | -0.91 (-2.2 %) | [-2.01, 0.14] | [-2.05, 0.13] | -1.32 [-3.65, 0.60] | +0.019 [-0.010, 0.048] | +3.5 pts [0.9, 6.2] | +0.0 pts | inconclusive |

### Within the same library (does the Hybrid / V2 story change?)

| Comparison | ΔMAE | 95 % (7-d) | 95 % (14-d) | Verdict |
|---|---|---|---|---|
| XGBoost Hybrid − V1 | -4.01 (-9.0 %) | [-5.92, -2.16] | [-5.87, -2.17] | conclusive |
| XGBoost V2 − V1 | -2.74 (-6.2 %) | [-4.99, -0.52] | [-5.13, -0.29] | conclusive |
| CatBoost Hybrid − V1 | -4.48 (-10.2 %) | [-6.43, -2.54] | [-6.48, -2.41] | conclusive |
| CatBoost V2 − V1 | -3.05 (-6.9 %) | [-5.29, -0.76] | [-5.51, -0.45] | conclusive |

## 4. By winter

XGBoost vs tuned CatBoost, each model's own frozen fit.

| Winter | Days | With day-parts | Model | XGBoost MAE | CatBoost MAE | Δ | 95 % (7-d) | RMSE | R² | Exact category | Persistence MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-23 | 47 | 2 | V1 (25 feat.) | 42.46 | 44.28 | +4.3 % | [-0.87, 4.26] | 58.2 → 59.5 | 0.501 → 0.479 | 57.4 → 57.4 % | 49.00 |
| 2022-23 | 47 | 2 | V2 (35 feat.) | 48.92 | 48.48 | -0.9 % | [-4.50, 3.88] | 63.1 → 65.0 | 0.414 → 0.377 | 42.6 → 46.8 % | 49.00 |
| 2022-23 | 47 | 2 | Hybrid | 42.48 | 44.13 | +3.9 % | [-0.97, 4.24] | 58.2 → 59.4 | 0.501 → 0.479 | 57.4 → 57.4 % | 49.00 |
| 2023-24 | 124 | 45 | V1 (25 feat.) | 48.64 | 47.12 | -3.1 % | [-3.51, 0.40] | 66.1 → 65.5 | 0.422 → 0.432 | 65.3 → 62.9 % | 49.91 |
| 2023-24 | 124 | 45 | V2 (35 feat.) | 47.93 | 47.60 | -0.7 % | [-2.53, 1.78] | 64.2 → 64.5 | 0.455 → 0.449 | 62.1 → 65.3 % | 49.91 |
| 2023-24 | 124 | 45 | Hybrid | 46.16 | 44.84 | -2.9 % | [-3.49, 0.73] | 62.2 → 61.6 | 0.487 → 0.498 | 66.9 → 67.7 % | 49.91 |
| 2024-25 | 124 | 124 | V1 (25 feat.) | 47.48 | 46.95 | -1.1 % | [-1.83, 1.04] | 65.5 → 65.3 | 0.551 → 0.553 | 65.3 → 66.1 % | 50.78 |
| 2024-25 | 124 | 124 | V2 (35 feat.) | 42.66 | 40.52 | -5.0 % | [-4.45, -0.17] | 60.3 → 56.9 | 0.618 → 0.661 | 66.9 → 75.8 % | 50.78 |
| 2024-25 | 124 | 124 | Hybrid | 42.66 | 40.52 | -5.0 % | [-4.45, -0.17] | 60.3 → 56.9 | 0.618 → 0.661 | 66.9 → 75.8 % | 50.78 |
| 2025-26 | 104 | 103 | V1 (25 feat.) | 36.42 | 36.35 | -0.2 % | [-1.61, 1.49] | 48.3 → 47.7 | 0.531 → 0.542 | 65.4 → 65.4 % | 41.60 |
| 2025-26 | 104 | 103 | V2 (35 feat.) | 29.56 | 29.86 | +1.0 % | [-1.12, 1.88] | 40.1 → 39.8 | 0.677 → 0.681 | 71.2 → 73.1 % | 41.60 |
| 2025-26 | 104 | 103 | Hybrid | 29.70 | 29.62 | -0.3 % | [-1.31, 1.22] | 40.1 → 39.5 | 0.676 → 0.685 | 71.2 → 73.1 % | 41.60 |

## 5. Season, hourly-data availability and sudden changes (T1, pooled)

Hybrid, XGBoost vs tuned CatBoost.

| Set | Days | XGBoost MAE | CatBoost MAE | Δ | 95 % (7-d) | 95 % (14-d) | Exact category | Verdict |
|---|---|---|---|---|---|---|---|---|
| all matched days | 399 | 40.35 | 39.45 | -2.2 % | [-2.01, 0.14] | [-2.05, 0.13] | 66.9 → 70.4 % | inconclusive |
| days with all 7 day parts | 274 | 38.17 | 36.71 | -3.8 % | [-2.87, -0.17] | [-2.92, -0.19] | 67.5 → 73.0 % | conclusive |
| days without day parts | 125 | 45.14 | 45.44 | +0.7 % | [-1.48, 1.83] | [-1.65, 1.69] | 65.6 → 64.8 % | inconclusive |
| sudden change days ge30pct | 121 | 66.09 | 65.78 | -0.5 % | [-2.45, 1.75] | [-2.53, 1.71] | 47.9 → 51.2 % | inconclusive |
| stable days lt30pct | 278 | 29.15 | 27.99 | -4.0 % | [-2.29, -0.12] | [-2.50, -0.02] | 75.2 → 78.8 % | conclusive |
| IMD post-monsoon (15 Oct-30 Nov) | 139 | 42.16 | 41.07 | -2.6 % | [-3.30, 0.59] | [-3.14, 0.50] | 69.1 → 73.4 % | inconclusive |
| IMD winter (1 Dec-15 Feb) | 260 | 39.39 | 38.58 | -2.0 % | [-2.20, 0.41] | [-2.45, 0.46] | 65.8 → 68.8 % | inconclusive |

Hybrid in full detail:

| Subset | Model | Days | MAE | RMSE | R² | Exact | Within one | vs persistence |
|---|---|---|---|---|---|---|---|---|
| days with all 7 day parts | XGBoost Hybrid | 274 | 38.17 | 53.59 | 0.611 | 67.5 % | 98.5 % | +19.9 % |
| days with all 7 day parts | CatBoost (tuned) Hybrid | 274 | 36.71 | 51.40 | 0.642 | 73.0 % | 98.5 % | +23.0 % |
| days without day parts | XGBoost Hybrid | 125 | 45.14 | 61.42 | 0.553 | 65.6 % | 96.8 % | +6.8 % |
| days without day parts | CatBoost (tuned) Hybrid | 125 | 45.44 | 61.71 | 0.549 | 64.8 % | 96.8 % | +6.2 % |
| sudden change days ge30pct | XGBoost Hybrid | 121 | 66.09 | 82.69 | 0.354 | 47.9 % | 95.0 % | +30.5 % |
| sudden change days ge30pct | CatBoost (tuned) Hybrid | 121 | 65.78 | 81.99 | 0.364 | 51.2 % | 95.0 % | +30.8 % |
| stable days lt30pct | XGBoost Hybrid | 278 | 29.15 | 39.37 | 0.761 | 75.2 % | 99.3 % | -6.5 % |
| stable days lt30pct | CatBoost (tuned) Hybrid | 278 | 27.99 | 37.29 | 0.786 | 78.8 % | 99.3 % | -2.3 % |
| IMD post-monsoon (15 Oct-30 Nov) | XGBoost Hybrid | 139 | 42.16 | 63.88 | 0.529 | 69.1 % | 100.0 % | +11.1 % |
| IMD post-monsoon (15 Oct-30 Nov) | CatBoost (tuned) Hybrid | 139 | 41.07 | 60.89 | 0.572 | 73.4 % | 100.0 % | +13.4 % |
| IMD winter (1 Dec-15 Feb) | XGBoost Hybrid | 260 | 39.39 | 51.56 | 0.608 | 65.8 % | 96.9 % | +18.2 % |
| IMD winter (1 Dec-15 Feb) | CatBoost (tuned) Hybrid | 260 | 38.58 | 51.31 | 0.612 | 68.8 % | 96.9 % | +19.9 % |

Why the day counts differ: Hybrid uses V2 only when all 7 day-part inputs exist, so days without hourly data are predicted by V1 in both libraries; V2 on its own is still evaluated on those days with the day-part columns missing (section 3 tables).

## 6. T2: real archived 24 h weather forecast (2024-25 and 2025-26)

This is the closest retrospective match to live inputs (the live Lambda receives a forecast). Same models as T1, only tomorrow's weather input differs.

| Model | MAE µg/m³ [95 %] | RMSE µg/m³ [95 %] | R² [95 %] | Exact category [95 %] | Within one category | Better than persistence [95 %] |
|---|---|---|---|---|---|---|
| Tomorrow = today | 46.59 [40.07, 55.11] | 64.36 [53.67, 76.96] | 0.450 [0.218, 0.558] | 60.5 % [53.3, 68.1] | 96.5 % | - |
| 7-day mean | 58.75 [50.45, 69.99] | 77.18 [64.34, 91.88] | 0.209 [-0.089, 0.365] | 57.9 % [49.8, 65.9] | 95.6 % | -26.1 % [-43.5, -11.3] |
| XGBoost V1 (25 feat.) | 41.93 [36.19, 49.74] | 57.81 [48.24, 70.72] | 0.556 [0.382, 0.644] | 64.9 % [58.0, 72.2] | 98.2 % | +10.0 % [1.8, 17.8] |
| CatBoost (baseline) V1 (25 feat.) | 41.74 [36.13, 49.38] | 56.90 [47.44, 69.19] | 0.570 [0.407, 0.648] | 65.4 % [58.1, 72.5] | 98.2 % | +10.4 % [4.0, 16.7] |
| CatBoost (tuned) V1 (25 feat.) | 41.33 [35.69, 49.24] | 56.25 [46.52, 69.20] | 0.580 [0.426, 0.660] | 65.4 % [58.3, 72.2] | 98.2 % | +11.3 % [4.2, 18.1] |
| XGBoost Hybrid | 37.21 [31.56, 45.35] | 51.68 [41.32, 66.16] | 0.645 [0.529, 0.723] | 70.2 % [63.2, 76.8] | 98.7 % | +20.1 % [10.4, 28.9] |
| CatBoost (baseline) Hybrid | 36.10 [30.95, 43.35] | 49.09 [40.40, 60.70] | 0.680 [0.559, 0.744] | 70.2 % [63.3, 76.6] | 98.7 % | +22.5 % [14.2, 30.0] |
| CatBoost (tuned) Hybrid | 35.59 [30.38, 42.97] | 48.24 [39.44, 59.98] | 0.691 [0.571, 0.755] | 72.4 % [66.2, 78.1] | 98.7 % | +23.6 % [14.8, 31.5] |

| Comparison | ΔMAE | 95 % (7-d) | 95 % (14-d) | ΔRMSE [95 %] | ΔR² [95 %] | Δ exact category [95 %] | Δ within-one | Verdict |
|---|---|---|---|---|---|---|---|---|
| CatBoost (baseline) V1 (25 feat.) − XGBoost | -0.20 (-0.5 %) | [-1.55, 1.13] | [-1.42, 1.05] | -0.91 [-2.54, 0.68] | +0.014 [-0.011, 0.044] | +0.4 pts [-3.2, 4.7] | +0.0 pts | inconclusive |
| CatBoost (baseline) V2 (35 feat.) − XGBoost | -0.97 (-2.6 %) | [-2.71, 0.55] | [-2.68, 0.55] | -2.50 [-5.79, 0.47] | +0.034 [-0.008, 0.070] | +0.0 pts [-3.2, 3.0] | +0.0 pts | inconclusive |
| CatBoost (baseline) Hybrid − XGBoost | -1.11 (-3.0 %) | [-2.81, 0.37] | [-2.76, 0.36] | -2.59 [-5.81, 0.32] | +0.035 [-0.005, 0.070] | +0.0 pts [-3.2, 3.0] | +0.0 pts | inconclusive |
| CatBoost (tuned) V1 (25 feat.) − XGBoost | -0.60 (-1.4 %) | [-1.83, 0.59] | [-1.79, 0.63] | -1.56 [-3.08, -0.30] | +0.024 [0.004, 0.059] | +0.4 pts [-2.6, 4.1] | +0.0 pts | inconclusive |
| CatBoost (tuned) V2 (35 feat.) − XGBoost | -1.46 (-3.9 %) | [-2.98, -0.09] | [-2.87, -0.15] | -3.33 [-6.59, -0.55] | +0.044 [0.009, 0.078] | +2.2 pts [-1.5, 6.0] | +0.0 pts | conclusive |
| CatBoost (tuned) Hybrid − XGBoost | -1.61 (-4.3 %) | [-3.08, -0.30] | [-3.01, -0.40] | -3.44 [-6.66, -0.73] | +0.046 [0.012, 0.079] | +2.2 pts [-1.5, 6.0] | +0.0 pts | conclusive |

| Set | Days | XGBoost MAE | CatBoost MAE | Δ | 95 % (7-d) | 95 % (14-d) | Exact category | Verdict |
|---|---|---|---|---|---|---|---|---|
| 2024-25 | 124 | 44.38 | 42.05 | -5.2 % | [-4.83, -0.26] | [-4.72, -0.66] | 68.5 → 72.6 % | conclusive |
| 2025-26 | 104 | 28.66 | 27.89 | -2.7 % | [-1.81, 0.46] | [-1.73, 0.20] | 72.1 → 72.1 % | inconclusive |

## 7. T3: all seasons, 2025-01-01 → 2026-10-07 (frozen fit before 2025)

Covers the four IMD seasons; 588 days, 570 with day-parts. The fit gets up to 21 months old, which favours nothing in particular but is stricter than the winter-by-winter refit. Retrospective: this period was examined by the project's locked test.

| Model | MAE µg/m³ [95 %] | RMSE µg/m³ [95 %] | R² [95 %] | Exact category [95 %] | Within one category | Better than persistence [95 %] |
|---|---|---|---|---|---|---|
| Tomorrow = today | 21.09 [17.61, 24.72] | 33.92 [27.77, 39.57] | 0.780 [0.705, 0.834] | 58.3 % [54.3, 62.7] | 96.4 % | - |
| 7-day mean | 24.47 [20.32, 28.60] | 37.65 [30.57, 43.98] | 0.729 [0.652, 0.790] | 53.2 % [47.8, 58.5] | 95.4 % | -16.0 % [-27.0, -4.6] |
| XGBoost V1 (25 feat.) | 20.54 [17.43, 23.52] | 31.69 [26.68, 36.09] | 0.808 [0.729, 0.857] | 57.3 % [52.9, 62.0] | 96.9 % | +2.6 % [-3.2, 8.7] |
| CatBoost (baseline) V1 (25 feat.) | 20.46 [17.33, 23.45] | 31.46 [26.39, 35.99] | 0.810 [0.740, 0.856] | 58.3 % [53.7, 62.9] | 97.3 % | +3.0 % [-2.0, 8.5] |
| CatBoost (tuned) V1 (25 feat.) | 20.42 [17.32, 23.42] | 31.26 [26.24, 35.74] | 0.813 [0.742, 0.858] | 59.7 % [55.0, 64.3] | 97.3 % | +3.2 % [-1.7, 8.5] |
| XGBoost Hybrid | 17.72 [14.97, 20.32] | 27.05 [22.68, 31.06] | 0.860 [0.809, 0.893] | 62.6 % [58.6, 66.8] | 98.5 % | +16.0 % [10.5, 22.0] |
| CatBoost (baseline) Hybrid | 17.82 [15.02, 20.51] | 27.34 [22.65, 31.48] | 0.857 [0.808, 0.890] | 62.8 % [58.2, 67.0] | 98.5 % | +15.5 % [10.3, 21.5] |
| CatBoost (tuned) Hybrid | 17.63 [14.89, 20.28] | 26.96 [22.41, 31.01] | 0.861 [0.812, 0.894] | 63.1 % [58.7, 67.4] | 98.5 % | +16.4 % [11.0, 22.4] |

| Comparison | ΔMAE | 95 % (7-d) | 95 % (14-d) | ΔRMSE [95 %] | ΔR² [95 %] | Δ exact category [95 %] | Δ within-one | Verdict |
|---|---|---|---|---|---|---|---|---|
| CatBoost (baseline) V1 (25 feat.) − XGBoost | -0.09 (-0.4 %) | [-0.53, 0.41] | [-0.56, 0.38] | -0.23 [-1.07, 0.76] | +0.003 [-0.008, 0.016] | +1.0 pts [-1.2, 3.1] | +0.3 pts | inconclusive |
| CatBoost (baseline) V2 (35 feat.) − XGBoost | +0.01 (+0.0 %) | [-0.45, 0.48] | [-0.47, 0.46] | +0.25 [-0.69, 1.10] | -0.003 [-0.011, 0.008] | +0.5 pts [-1.6, 2.2] | +0.2 pts | inconclusive |
| CatBoost (baseline) Hybrid − XGBoost | +0.10 (+0.6 %) | [-0.35, 0.55] | [-0.38, 0.52] | +0.30 [-0.65, 1.15] | -0.003 [-0.011, 0.008] | +0.2 pts [-2.0, 1.8] | +0.0 pts | inconclusive |
| CatBoost (tuned) V1 (25 feat.) − XGBoost | -0.13 (-0.6 %) | [-0.57, 0.37] | [-0.59, 0.37] | -0.43 [-1.16, 0.44] | +0.005 [-0.005, 0.017] | +2.4 pts [0.2, 4.5] | +0.3 pts | inconclusive |
| CatBoost (tuned) V2 (35 feat.) − XGBoost | -0.19 (-1.1 %) | [-0.62, 0.26] | [-0.62, 0.21] | -0.15 [-0.96, 0.60] | +0.002 [-0.006, 0.011] | +0.9 pts [-1.0, 2.5] | +0.2 pts | inconclusive |
| CatBoost (tuned) Hybrid − XGBoost | -0.09 (-0.5 %) | [-0.51, 0.33] | [-0.51, 0.29] | -0.09 [-0.90, 0.65] | +0.001 [-0.006, 0.011] | +0.5 pts [-1.4, 2.2] | +0.0 pts | inconclusive |

By season, hourly-data availability and sudden changes (Hybrid):

| Set | Days | XGBoost MAE | CatBoost MAE | Δ | 95 % (7-d) | 95 % (14-d) | Exact category | Verdict |
|---|---|---|---|---|---|---|---|---|
| all days | 588 | 17.72 | 17.63 | -0.5 % | [-0.51, 0.33] | [-0.51, 0.29] | 62.6 → 63.1 % | inconclusive |
| days with all 7 day parts | 570 | 17.85 | 17.75 | -0.6 % | [-0.54, 0.32] | [-0.55, 0.26] | 63.0 → 63.5 % | inconclusive |
| days without day parts | 18 | 13.45 | 14.05 | +4.5 % | [-1.88, 3.45] | [-1.83, 3.91] | 50.0 → 50.0 % | inconclusive |
| sudden change days ge30pct | 183 | 26.33 | 26.85 | +2.0 % | [-0.47, 1.32] | [-0.44, 1.21] | 45.4 → 45.9 % | inconclusive |
| stable days lt30pct | 405 | 13.83 | 13.47 | -2.6 % | [-0.78, 0.09] | [-0.78, 0.12] | 70.4 → 70.9 % | inconclusive |
| IMD winter (Dec-Feb) | 128 | 32.97 | 32.50 | -1.4 % | [-2.11, 0.99] | [-2.31, 0.80] | 66.4 → 68.8 % | inconclusive |
| IMD pre-monsoon (Mar-May) | 169 | 15.51 | 15.34 | -1.1 % | [-0.73, 0.43] | [-0.85, 0.47] | 55.0 → 52.1 % | inconclusive |
| IMD monsoon (Jun-Sep) | 229 | 8.71 | 8.78 | +0.8 % | [-0.22, 0.40] | [-0.19, 0.39] | 66.8 → 68.6 % | inconclusive |
| IMD post-monsoon (Oct-Nov) | 62 | 25.54 | 25.90 | +1.4 % | [-1.35, 2.44] | [-1.30, 2.35] | 59.7 → 61.3 % | inconclusive |

## 8. Robustness

* **Seed noise (T1 pooled, Hybrid MAE per seed):**

| Seed | XGBoost | CatBoost baseline | CatBoost tuned | tuned − XGBoost |
|---|---|---|---|---|
| 42 | 40.35 | 39.54 | 39.45 | -0.91 |
| 7 | 40.42 | 39.60 | 39.28 | -1.14 |
| 123 | 40.54 | 39.69 | 39.47 | -1.07 |
| 2024 | 40.44 | 40.02 | 39.29 | -1.15 |
| 99 | 40.47 | 39.90 | 39.39 | -1.09 |

  Tuned CatBoost beats XGBoost in 5 of 5 seeds. Averaging each library's five seeds: -2.7 % ([-2.08, -0.14] / [-2.12, -0.14]) - conclusive.

* **Production data only (`data/daily.csv`, T1):** 2022-23, 2023-24, 2024-25, 2025-26, 353 days, 105 with day-parts (without the OpenCity gap-fill the 2024-25 winter has no day-parts, so Hybrid = V1 there).

| Set | Days | XGBoost MAE | CatBoost MAE | Δ | 95 % (7-d) | 95 % (14-d) | Exact category | Verdict |
|---|---|---|---|---|---|---|---|---|
| Hybrid | 353 | 42.92 | 42.25 | -1.6 % | [-1.80, 0.32] | [-1.81, 0.30] | 66.3 → 66.3 % | inconclusive |
| V1 | 353 | 44.08 | 44.00 | -0.2 % | [-1.15, 0.90] | [-1.14, 0.91] | 64.9 → 65.2 % | inconclusive |
| V2 | 353 | 44.15 | 43.84 | -0.7 % | [-1.37, 0.61] | [-1.36, 0.57] | 62.3 → 64.0 % | inconclusive |

* **Not covered:** different train/validation split lengths, other CatBoost loss functions, categorical encodings of month / weekday, and ensembling the two libraries. The project's earlier study found no model-family change beats XGBoost beyond noise; this experiment tests only CatBoost.

## 9. Decision (rule declared before running)

| Check | Met |
|---|---|
| MAE at least 2 % lower than XGBoost Hybrid | yes |
| 95 % interval below zero for BOTH 7- and 14-day blocks | no |
| no single winter more than 2 % worse | no |
| exact-category accuracy not lower by more than 2 points | yes |
| development-fold MAE not worse than XGBoost | yes |

Primary comparison: tuned CatBoost Hybrid vs XGBoost Hybrid, T1, primary data, seed 42: ΔMAE -0.91 µg/m³ (-2.2 %), 95 % [-2.01, 0.14] (7-d) / [-2.05, 0.13] (14-d); per winter 2022-23 +3.9 %, 2023-24 -2.9 %, 2024-25 -5.0 %, 2025-26 -0.3 %; development-fold ΔMAE -0.94 µg/m³.

**No evidence of improvement: keep XGBoost; CatBoost stays a documented reference.** Even if the rule were met, CatBoost would not be promoted: it stays a separate candidate until you review and approve further testing; the forward-test freeze, the default model and the dashboard are untouched.

## 10. Limitations

* Retrospective: every test winter was examined before; the dev-fold tuning avoids them, but the project's decisions about features and settings did not.
* Few independent periods: four winters of strongly day-correlated data; a difference of a few percent needs more than this to be proven.
* Hourly day-part data exists for part of 2022-23 and 2023-24 and for all of 2024-25 and 2025-26 only through the OpenCity gap-fill; the V2 and Hybrid comparisons rest on those winters.
* Tuning used V1 features and observed weather (the development years have neither day-parts nor forecasts); settings may not be ideal for V2 or for forecast weather.
* Test-time weather is either persistence (a no-skill bound) or a 24 h-old archived forecast (T2); live accuracy depends on the real nightly forecast.

## 11. Files

`config.json` (declared rules), `run_catboost.py` (tune and test stages), `stats.py`, `report.py`, `tuning.json`, `results.json` (machine-readable, daily predictions included), `report.md`, `tests/`, `requirements-catboost.txt`, `verify_unchanged.py`, `integrity.json`.

## 12. Reproduce

```bash
cd kal-ki-hawa
python -m venv --system-site-packages .venv-catboost            # separate environment, not committed
.venv-catboost/bin/pip install -r experiments/catboost/requirements-catboost.txt
python experiments/catboost/verify_unchanged.py --snapshot       # hash every protected file BEFORE running
export PYTHONDONTWRITEBYTECODE=1
.venv-catboost/bin/python experiments/catboost/run_catboost.py --stage tune
.venv-catboost/bin/python experiments/catboost/run_catboost.py --stage test
python experiments/catboost/verify_unchanged.py --check          # confirm nothing outside experiments/catboost changed
.venv-catboost/bin/python experiments/catboost/report.py
.venv-catboost/bin/python -m pytest experiments/catboost/tests -q   # experiment tests
python -m pytest tests -q                                         # the project's own tests
```

Data needed: `data/daily_opencity.csv`, `data/daily.csv`, `data/raw/opencity_hourly/`, `data/raw/weather_prevruns_2024_2026.json` (`pipeline/fetch_prevruns.py`), and the hourly station files under `data/raw/` (git-ignored; see README).