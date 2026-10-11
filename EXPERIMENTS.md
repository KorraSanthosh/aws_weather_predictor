# Model-improvement experiments (kal-ki-hawa)

**Result in one line (first study, below):** none of the *generic* extra features beat the 25-feature model on data it had
never seen. A follow-up study with *domain-knowledge* day-part features looked better **on days that have hourly data**, but
is **about even across the whole evaluation period and worse on days without hourly data**. It is therefore kept as an
**experimental candidate (`v2_35_candidate`)**; the **25-feature `v1_25_baseline` remains the default**. See
`model/MODEL_VERSIONS.md` and the verified paired comparison at the end of this update.

## Update 9 Oct 2026: domain-knowledge features (35) - EXPERIMENTAL CANDIDATE (not adopted)

The study below found that generic extra features did not help. A second study then added **domain-knowledge features**.
They help where hourly data exist (below) but have **not** been shown to generalise, so the 35-feature model is stored as
`model/v2_35_candidate/` and the 25-feature "v1" below stays the baseline and the default model.

**What was added (fixed in advance, `core/features.py`):**
* **Day-part PM2.5 (7 features)** from the hourly data, using Delhi's daily cycle (IST): night 00-05 (inversion
  build-up), commute 06-08 (morning rush / school run), school hours 09-15 (the sun lifts the mixing layer, so PM2.5
  is usually lowest), evening 16-19 (rush hour, boundary layer collapses at sunset), late 20-23 (night-time build-up,
  the best clue to tomorrow morning). Plus two ratios: `pm_buildup` = late / school-hours, and `pm_last3h_ratio` =
  last-3-hours / day mean (end-of-day momentum).
* **Binary domain flags (8)** about the target day: stubble-burning season (15 Oct-30 Nov), Diwali window (-1..+4 days),
  winter-inversion months (Dec-Jan), weekend, rain (>= 1 mm), stagnant air (wind < 5 and no rain), fog risk
  (humidity > 85 and temperature < 12), wind from the north-west (the stubble-smoke direction).
* The set is 8 PM2.5-history + 7 tomorrow-weather + 3 today-weather + 2 season + 7 day-part + 8 flags = **35**.

**Protocol:** same walk-forward, monthly re-fit, paired on identical days against the 25-feature model with the same
recipe (`python model/feature_study.py` -> `experiments/feature_study_35.json`). Day-part data exist only where
hourly data exist (2018-2022 and Feb 2025 onward; 2023-24 is CPCB daily files), so the day-part effect is judged on
the days that have it.

| Test (walk-forward, 2023+) | n | v1 (25) MAE / cat acc | 35 features MAE / cat acc | MAE change (95% CI) |
|---|---|---|---|---|
| Days with hourly data, Mar 2025 -> Oct 2026 | 523 | 15.47 / 66.2% | **13.61 / 70.4%** | **-12.0% (-17.9% to -5.5%)** |
| ... winter days only (15 Oct-15 Feb) | 104 | 32.81 / 67.3% | 29.97 / 71.2% | -8.7% (-19.4% to +3.9%) |
| All 2023+ days (2023-24 has no day-parts) | 1,262 | 20.71 / 64.4% | 20.54 / 64.8% | -0.8% (-3.6% to +2.0%) |

* R2 on the hourly days 0.868 -> 0.902; within one category 98.3% -> 99.2%.
* **The gain comes from the day-part features.** Flags alone (28 features) were slightly *worse* than v1 (+2.5%,
  CI +0.3% to +4.8%): the calendar/weather flags add nothing measurable beyond what the model already infers
  from weather and day-of-year. A post-hoc check "v1 + the 7 day-part features" (32 features) gave -14.1%
  (-19.4% to -8.7%), statistically the same as the 35-feature set; the pre-declared 35-feature set is the one kept as the candidate.
* Rolling winter folds (`model/train.py --version v2_35_candidate`, 306 days): MAE 41.07 vs 46.83 for "tomorrow = today" (12.3% lower,
  CI 5.8%-18.5%), category accuracy 69.0% (was 67.6%). Most of those days have no day-parts, so this understates
  the live benefit.

**Honest caveats**
* The 2023+ period was looked at before (the v1/v2 comparison and an ablation). The 35 features were written down before
  any result and nothing was tuned on it, but it is not a perfectly untouched test. The real proof is the live
  forward test this winter.
* Day-part features need **hourly** data. The nightly Lambda has it (it fetches hourly readings), but the 2023-24
  training years do not, so the model learned them from ~1,100 days (2018-22 and 2025-26).
* At the nightly run (23:00 IST) hour 23 is not in yet, so `dp_late` and `dp_last3` use the hours available. The size of this
  train/serve difference is now measured (next section): it is small.
* Tomorrow's weather is still observed in the tests and a forecast live, so live accuracy will be somewhat lower.
* `tests/test_core.py` and `tests/test_backend.py` guard the day-part hours, the no-future-leakage rule and that
  every one of the 35 features is produced live when hourly data exist.

### Verified paired comparison, v1 baseline vs v2 candidate (`python model/compare_versions.py`)

Same walk-forward protocol and identical test days for both models (1,261 days, 2023-01-01 -> 2026-10-07; full output in
`experiments/model_comparison_v1_v2.json`; the unfiltered reconciliation of the 523/1,262-day figures above is in
`experiments/model_comparison_v1_v2_all_origins.json`). MAE / RMSE in ug/m3. "Cat" = AQI-category accuracy.
Interval = paired 7-day block bootstrap on the MAE change (negative = v2 better).

**Offline (full 24 h of "today", the setting behind the numbers above)**

| Subset | Days | v1 MAE / RMSE / R2 / Cat | v2 MAE / RMSE / R2 / Cat | v2 vs v1 MAE (95% CI) |
|---|---|---|---|---|
| Full evaluation period | 1,261 | 20.6 / 33.1 / 0.85 / 63.9% | 20.5 / 33.1 / 0.85 / 64.9% | -0.7% (-3.9 to +2.4) |
| **Days with hourly data** (>= 1 Mar 2025) | 519 | 15.5 / 24.4 / 0.87 / 65.1% | 13.4 / 21.1 / 0.90 / 71.5% | **-13.8% (-19.5 to -7.6)** |
| **Days WITHOUT day-part data** (2023-24) | 733 | 24.3 / 38.2 / 0.83 / 63.0% | 25.6 / 39.6 / 0.82 / 60.3% | **+5.3% (+2.2 to +8.5)** |
| Winter 15 Oct-15 Feb, all | 351 | 40.5 / 55.0 / 0.62 / 66.1% | 41.6 / 55.5 / 0.61 / 65.5% | +2.7% (-1.5 to +7.2) |
| Winter, hourly days only | 104 | 32.7 / 43.4 / 0.62 / 64.4% | 30.0 / 38.5 / 0.70 / 72.1% | -8.2% (-19.1 to +3.6) |
| Sudden change (>= 30% day-to-day), all | 373 | 34.3 / 50.3 / 0.68 / 44.5% | 33.5 / 49.4 / 0.69 / 46.1% | -2.5% (-7.4 to +2.6) |
| Sudden change, hourly days only | 149 | 25.7 / 37.8 / 0.65 / 49.7% | 19.2 / 29.1 / 0.79 / 61.1% | -25.5% (-30.9 to -19.8) |

By season (all days / hourly days only, v2 vs v1 MAE): winter Dec-Feb +4.9% (n=258) / -9.3% (n=69); summer Mar-May
-6.7% / -23.7%; monsoon Jun-Sep -4.0% / -13.5%; post-monsoon Oct-Nov -0.4% / -4.5% (intervals for the winter and
post-monsoon rows include zero). Per-season RMSE, R2 and category accuracy are in the JSON.

**Coverage and differences, disclosed**
* 1,261 days are scored by both models (one day fewer than the earlier 1,262: 13 forecast origins in the whole history, one of them in
  the test period, cannot be reproduced at 23:00 IST and were dropped from this paired run; including them reproduces 523 hourly days,
  15.47 vs 13.61 MAE and 66.2% vs **70.4%** category accuracy exactly).
* **70.4% applies only to the 523 days that have hourly data** (Mar 2025 -> Oct 2026), where v1 scores 66.2% on the
  same days. Across all 1,262 days the figures are 64.4% (v1) and 64.8% (v2) - effectively equal.
* **v2 has 7 missing inputs on 733 of 1,261 days** (every 2023-24 day: pm_night, pm_commute, pm_school, pm_evening,
  pm_late, pm_buildup, pm_last3h_ratio). They are left as NaN, never filled in. On those days v2 is *worse* than v1.
* Hourly-data days are one year of mostly non-winter data (only 104 winter days), so the headline gain is not a winter result.

**The 23:00 IST train/serve difference, measured.** The nightly run sees hours 0-22 of "today"; training used all 24.
Re-scoring with hours > 22 removed from the test features ("live" view): v1 20.8 (all) / 15.9 (hourly), v2 20.6 / 13.7;
forecasts moved by 0.5 ug/m3 (v1) and 2.3 ug/m3 (v2, hourly days) on average; 1.2% (v1) and 5.6% (v2) of daily
category labels flipped. Retraining on the same cut-off view ("causal") did not help (v2 13.8 on hourly days), so no
third model was stored. The honest v2 figure for live use is the live view: **-13.8% MAE, 69.2% vs 64.9% category
accuracy on hourly days**, still no real change over the whole period (-0.8%).

**Not clean evidence.** The 2023+ period was examined in earlier experiments; nothing was tuned or selected on it for
this comparison (both feature lists and the training recipe were fixed beforehand), but it is not an untouched
hold-out, and tests use observed tomorrow-weather where live use relies on a forecast. A genuine forward test (winter
2026-27, frozen models, plan in `model/MODEL_VERSIONS.md`) is required before claiming v2 generalises better.

### Fresh-training study: many model types, blocked cross-validation, then predict 2025-26 (`python model/dev_cv_study.py --stage dev|locked`)

Question: can better model types or tuning (cross-validated XGBoost search, gradient-boosting variants, ExtraTrees, Ridge, a small neural
net, a direct category classifier, ensembles) lift accuracy? Protocol, fixed before running: everything is chosen on **2018-2024 only**
with 6 **blocked expanding-window** folds (never shuffled K-fold); the selection rule (best per family by pooled CV MAE; ensemble = plain
mean of the 3 best families) is written to `experiments/dev_cv_selection.json`; then the finalists are trained on <= 2024 and scored **once**
on 2025-01-01 -> 2026-10-07 (539 days, all with hourly day-parts). 2025-26 was examined before for v1/v2, but none of these models was
designed on it. It is not a clean hold-out. Output: `experiments/dev_cv_study.json`, `experiments/locked_2025_26_result.json`.

**Development CV (2018-2024, 1,020 out-of-fold days, MAE in ug/m3; persistence 29.8)**

| Model (best config per family) | CV MAE | CV category acc. |
|---|---|---|
| XGBoost default, 25 features / 35 features | 26.1 / 25.1 | 60.2% / 63.3% |
| XGBoost tuned (30 random configs), 35 features | 24.6 | 63.1% |
| Ridge / small neural net / HistGradientBoosting / ExtraTrees | 24.8 / 24.8 / 25.0 / 25.1 | 61.7-62.7% |
| Ensemble of XGB + neural net + Ridge | 24.3 | 62.7% |
| Direct 6-class XGBoost classifier (25 / 35 features) | - | 57.5% / 56.5% |

* **Shuffled K-fold would have lied:** the same default XGBoost scored MAE 22.9 (shuffled) vs 26.1 (blocked), a 12% flattering gap, because
  neighbouring days leak across shuffled folds. This is the most common reason for inflated accuracy in published air-quality work.
* No model family stood out: all lie within about 3 MAE of each other. A direct category classifier was worse than regress-then-bin.

**Locked 2025-01-01 -> 2026-10-07 (539 days), walk-forward monthly refit; "static" = trained once through 2024**

| Model | MAE | RMSE | R2 | Category acc. | Within 1 category | Winter MAE | vs v1 MAE (95% CI) |
|---|---|---|---|---|---|---|---|
| Tomorrow = today | 18.1 | 28.6 | 0.82 | 59.0% | 96.5% | - | - |
| v1 (25 features) | 15.4 | 24.4 | 0.87 | 64.6% | 98.3% | 33.5 | - |
| v2 (35 features, default params) | **13.25** | 20.6 | 0.905 | 70.5% | 99.4% | 30.1 | -13.8% (-19.2 to -8.5) |
| Tuned XGBoost, 35 features | 13.31 | 20.9 | 0.903 | **72.9%** | 99.6% | 30.6 | -13.4% (-18.9 to -8.4) |
| Small neural net | 13.98 | 21.8 | 0.895 | 67.2% | 99.3% | 29.5 | -9.1% (-15.5 to -2.8) |
| Ensemble (XGB + NN + Ridge) | 13.26 | 20.8 | 0.904 | 70.7% | 99.6% | 29.8 | -13.8% (-19.5 to -8.3) |

Trained once on <= 2024 and not refitted (static): v1 16.2 / 62.3%, v2 14.55 / 69.0%, ensemble 13.99 / 69.0%.

**What this says.** The improvement over v1 is about 13%, and it comes from the **35 day-part features**, not from the model type: tuning
(13.31 vs 13.25, no gain), a neural net (worse) and an ensemble (equal) add nothing measurable to v2 with default settings. Exact-category accuracy tops out at
about 70-73%, and every model is within a point of 99.5% for "within one category". **No model reached 90% exact-category accuracy, and
nothing in these results suggests more tuning on this data would.** Picking the single best category-accuracy cell (72.9%) after looking would
be selection on the test; the dev-CV winner by MAE (tuned XGBoost) is the pre-selected one and its MAE equals default v2.

### Winter by winter, old vs new (`python model/winter_check.py`; `experiments/winter_check.json`)

Winters are 15 Oct -> 15 Feb target dates; same walk-forward refit as above. MAE (ug/m3) / exact-category accuracy. Only the **2025-26**
winter has hourly day-parts; 2022-24 and most of 2024-25 are CPCB daily files, so v2 runs with its 7 day-part inputs missing there.

| Winter | Days | Tomorrow = today | v1 (old) | v2 (new) | v2 vs v1 MAE (95% CI) |
|---|---|---|---|---|---|
| Jan-Feb 2023 (partial) | 45 | 50.4 / 46.7% | 39.1 / 46.7% | 46.1 / 35.6% | +17.9% (+10.1 to +27.7) |
| 2023-24 | 124 | 49.3 / 62.1% | 42.7 / 68.5% | 45.6 / 68.5% | +6.8% (+2.3 to +11.1) |
| 2024-25 | 78 | 51.9 / 65.4% | 48.9 / 74.4% | 47.3 / 76.9% | -3.2% (-7.6 to +1.1) |
| **2025-26 (has day-parts)** | 104 | 40.1 / 59.6% | 32.8 / 67.3% | **30.0 / 71.2%** | -8.7% (-19.9 to +3.1) |
| All four pooled | 351 | 47.3 / 60.1% | 40.7 / 66.7% | 41.4 / 67.0% | +1.8% (-2.3 to +5.9) |

On the 2025-26 winter's 28 sudden-change days (>= 30% day-to-day) v2 is 30.5% better (MAE 42.5 vs 61.1). The hybrid equals v1 in the three
winters without day-parts and v2 in 2025-26 (pooled MAE 39.8 / 67.8%). **The only winter where v2's special inputs exist is one winter, 104
days, and its overall interval still includes zero.** That is why winter 2026-27 is the deciding test.

### Hybrid "v3_hybrid" (v2 when all 7 day-part inputs exist, else v1) - historical description only (`python model/hybrid_study.py`)

One rule, fixed before it was run (`core/hybrid.py`); no thresholds or variants were tried. It is a router between the two
stored models, not a new model. Same 1,261 days and walk-forward protocol; "live" = the 23:00 IST view. Output:
`experiments/hybrid_study_historical.json`. **Descriptive only: this period is already examined, so it is not evidence for promotion.**

| Live view, MAE / category acc. | Days | v1 | v2 | hybrid | hybrid vs v1 MAE (95% CI) |
|---|---|---|---|---|---|
| Full period | 1,261 | 20.8 / 63.9% | 20.6 / 64.0% | **19.9 / 65.7%** | -4.3% (-6.5 to -2.3) |
| Days with valid day-parts | 519 | 15.9 / 64.7% | 13.8 / 69.0% | 13.8 / 69.0% | -13.6% (-19.5 to -7.5) |
| Days without (hybrid = v1) | 742 | 24.2 / 63.3% | 25.4 / 60.5% | 24.2 / 63.3% | 0.0% |
| Winter 15 Oct-15 Feb | 351 | 40.8 / 65.8% | 41.7 / 65.5% | 39.8 / 68.4% | -2.6% (-6.0 to +0.6) |
| Winter days with day-parts | 103 | 34.0 / 64.1% | 30.4 / 72.8% | 30.4 / 72.8% | -10.5% (-22.1 to +3.1) |

The hybrid removes v2's loss on days without day-parts (v2 vs hybrid on those days: +4.9%). **Read this carefully:** the full-period
gain is mostly an artifact of history. The 742 "no day-part" days are the 2023-24 CPCB years where hourly data never existed. A live
forecast almost always has hourly data, so in the forward test the hybrid and v2 will agree except on nights when the hourly
feed has gaps. The historical -4.3% should therefore not be expected to carry over; the forward test measures exactly this.

---

## Protocol (set before looking at results)
* **Time-based only**, walk-forward with a monthly re-fit: each month is predicted by a model trained
  only on earlier days. Nothing is shuffled, nothing is interpolated, days need >= 3 Delhi stations.
* **Development period (all choices made here):** 2019-01-01 .. 2022-12-31 (397 usable days).
* **Final test (untouched until the choice was frozen):** 2023-01-01 .. 2026-10-07, 1,262 days
  (351 of them inside the 15 Oct - 15 Feb winter windows). Evaluated **once**.
* Rules fixed in advance: keep a feature group / model family / weighting only if it lowers dev MAE by >= 1%.
* Reproduce: `python model/experiments.py dev` then `python model/experiments.py final`
  (outputs in `experiments/`).

## What the code does (v1, deployed)
1. `pipeline/build_dataset.py`: hourly OpenAQ + CPCB daily files -> per-station IST daily means (>= 16 valid
   hours; values <0 or >1000 dropped) -> Delhi city mean when >= 3 stations reported (Gurugram, Faridabad,
   Noida, Ghaziabad are excluded by `find_stations.py`) -> joined with Open-Meteo daily weather.
2. `core/features.py` builds one row per day D: PM2.5 lags 0,1,2,3,6, 3/7-day mean, 7-day std, 1-day change,
   today's and tomorrow's weather, day-of-year sin/cos. Same function is used offline and in the Lambda.
3. `model/train.py`: XGBoost predicts the **log change from today's PM2.5**; the prediction is converted to
   an Indian AQI category with the CPCB PM2.5 breakpoints (`core/aqi.py`).

## Leakage audit
| Check | Finding |
|---|---|
| Future PM2.5 in features | None. Every PM2.5 feature reads days <= D only (unit-tested). |
| Rolling averages | Windows end at D. |
| Scaling / normalisation | None used (tree models). |
| Station / city averages | Built from same-day values only. |
| Tomorrow's weather | **Observed** weather of the target day in training, a **forecast** in live use. Not leakage of PM2.5, but it makes offline scores optimistic. Quantified below. |
| Uncertainty range (q10/q90) | Calibrated on the same winter folds it is reported on (slightly in-sample). Does not affect the point forecast. |
| Early stopping (v1) | Uses the last 60 *training* rows, never test rows. |

## Development-period results (2019-2022, 316 predicted days)
| Experiment | Dev MAE | Dev category acc | Decision |
|---|---|---|---|
| tomorrow = today | 32.4 | 53.5% | - |
| v1 features, v1 recipe | 27.8 | 62.7% | baseline |
| v1 features, fixed trees | 27.4 | 56.6% | - |
| + PM2.5 lag 7/14 | 26.9 | 59.8% | kept |
| + rolling / change / slope | 28.3 | 58.2% | dropped |
| + weather-derived (changes, wind direction, stagnation) | 27.7 | 58.5% | dropped |
| + seasonal / calendar (month, weekday, Diwali, season) | 26.8 | 58.5% | dropped (<1%) |
| + station spread | 26.9 | 59.5% | dropped |
| tuned XGBoost (24 random configs) | 26.4 | 59.5% | kept |
| LightGBM / Random Forest / HistGradientBoosting | 27.2 / 26.9 / 27.7 | 60.4 / 60.4 / 58.2% | none beat XGBoost by 1% |
| sample weights (winter x2, sudden change, high pollution) | 26.5 / 27.6 / 27.1 | 58.9 / 59.2 / 58.9% | none helped |
| season-aware (winter-only training) | winter MAE 39.6 vs 34.4 | - | worse |
| direct AQI classifier (approach B) | - | 53.2% vs 59.5% (A) | regression wins |

Dev set is small (316 days), so differences under about 3 category points are noise.

## Final test, evaluated once (2023-2026, 1,262 days)
| | MAE | RMSE | R2 | AQI category acc | within +-1 category |
|---|---|---|---|---|---|
| **v1 (deployed)** | 20.71 | 33.18 | 0.849 | 64.4% | 97.6% |
| v2 candidate (frozen on dev: + lag 7/14, tuned) | 20.69 | 33.28 | 0.849 | 64.1% | 97.4% |
| tomorrow = today | 23.92 | 39.04 | 0.791 | 57.6% | 96.2% |
| 7-day average | 28.53 | 45.41 | 0.718 | 52.7% | 93.7% |
| CAMS global model | 49.99 | 76.44 | 0.201 | 33.4% | 75.8% |

* v2 vs v1: MAE change 0.1% (95% interval -1.6% to +1.8%). **No improvement**, so v1 stays.
* v1 vs tomorrow = today: **13.4% lower MAE** (95% CI 9.4% to 16.9%); v2 13.5% (9.6% to 17.0%).
* Direct classifier on the test: 59.1% (regression route 64.1%) - confirms the dev choice.

### Winter windows (15 Oct - 15 Feb, 351 days) and by winter
| | MAE | category acc |
|---|---|---|
| v1 | 40.67 | 66.7% |
| tomorrow = today | 47.28 | 60.1% |
| 2022-23 (partial, 45 days) | v1 39.1 vs 50.4 | 46.7% vs 46.7% |
| 2023-24 (124 days) | v1 42.7 vs 49.3 | 68.5% vs 62.1% |
| 2024-25 (78 days, to ~31 Dec) | v1 48.9 vs 51.9 | 74.4% vs 65.4% |
| 2025-26 (104 days) | v1 32.8 vs 40.1 | 67.3% vs 59.6% |

The MAE gain over persistence holds in all four winters; the category-accuracy gain holds in the last three
and is zero in the small, partial 2022-23 window.

### By season (v1 vs tomorrow = today)
| Season | days | MAE | category acc |
|---|---|---|---|
| Winter Dec-Feb | 258 | 34.7 vs 43.5 | 63.2% vs 57.4% |
| Summer Mar-May | 354 | 17.6 vs 20.0 | 54.2% vs 46.3% |
| Monsoon Jun-Sep | 466 | 9.1 vs 10.3 | 70.2% vs 63.3% |
| Post-monsoon Oct-Nov | 184 | 36.5 vs 38.5 | 71.2% vs 65.2% |

### Sudden-change days (>= 30% day-to-day change, 373 days)
MAE 34.8 vs 47.3 for persistence (26% lower); category accuracy 44.2% vs 23.9%.

### Poor-or-worse days (PM2.5 > 90, 493 days)
Caught 91.5% of them (persistence 86.0%); MAE 35.0 vs 40.6.

### Category accuracy by true category (v2 numbers; v1 is similar)
Satisfactory 76% (persistence 68%), Moderate 57%, Poor 45%, Very Poor 69%, Severe 70%, Good 48%.
Most errors are to the neighbouring category: the model is within one category 97% of the time.

## Weather-forecast realism (the main caveat)
Training and testing use the **observed** weather of tomorrow; the live system will use a **forecast**.
Two bounds on the full test (v2 model):

| Tomorrow's weather | MAE | category acc |
|---|---|---|
| observed (best case) | 20.69 | 64.1% |
| not available (lower bound; also = "today's weather reused") | 21.65 | 60.2% |
| tomorrow = today | 23.92 | 57.6% |

A real forecast sits between the first two rows, so expect **about 60-64% category accuracy and 9-13% lower
MAE than "tomorrow = today"**. Measuring this exactly needs archived past weather *forecasts*
(Open-Meteo "historical forecast" API) - not tested here (no network access to it).

## Ablation re-scored on the test, for transparency only (NOT used to choose anything)
| Features | MAE | category acc |
|---|---|---|
| v1 | 20.67 | 64.4% |
| + lags | 20.53 | 64.5% |
| + rolling | 21.07 | 63.6% |
| + weather-derived | 20.09 | 64.8% |
| + seasonal | 19.68 | 64.8% |
| + spatial | 19.71 | 65.7% |

The calendar/season and station-spread groups *might* help by 1-2 points on the test, but the development
period did not show it, and picking them now would mean selecting on the test set. Treat this as a hypothesis
to confirm on the winter 2026-27 forward test (the live system logs every prediction and actual).

## Answers
1. **Changed:** added the experiment harness, CPCB daily loader, Delhi-only station filter, station-spread
   columns in the dataset, a feature superset (`ALL_FEATURES`), the Lambda history window (10 -> 16 days, so any
   lag up to 14 days is available live; a guard test enforces it). The deployed model/features are unchanged.
2. **Features that helped most:** today's PM2.5 and tomorrow's wind/rain; of the new ones only lag 7/14 helped on dev.
3. **Best model:** XGBoost on log-change (tuned v2 and v1 are statistically identical; v1 kept).
4. **MAE:** 20.7 overall, 40.7 in winter windows.
5. **AQI accuracy:** 64.4% overall, 66.7% in winter windows (60-64% expected with a real weather forecast).
6. **R2:** 0.85 overall, 0.61 in winter windows.
7. **Improvement over the old model:** none (0.1%, inside noise).
8. **Over tomorrow = today:** 13.4% lower MAE overall (CI 9.4-16.9%), +6.8 category points.
9. **Consistent across winters:** MAE yes (4 of 4); category accuracy 3 of 4 (the partial 2022-23 window ties).
10. **Limitations:** one weather-forecast gap (above), a small development set, partial 2022-23 and 2024-25
    winters, a city average built from different stations in different eras, 2023-24 built from only four CPCB stations (Anand Vihar, R K Puram, Punjabi Bagh, Rohini).

## Why accuracy did not go up (the bottleneck)
AQI categories are narrow (30 ug/m3 wide in the Satisfactory-Moderate range) while the day-to-day noise of
city-mean PM2.5 is about 20-30 ug/m3, so many days sit near a boundary and a small error flips the category.
97% of predictions are within one category. Information is the limit, not model capacity: a more complex model,
tuning, extra features and a classifier all landed within noise of the simple model. More real winters (and
real weather-forecast archives) are the lever that remains.

## Training on archived weather forecasts (Oct 2026, retrospective)
Question: does training on archived forecast weather (instead of observed/ERA5 weather) reduce the train/serve
mismatch? Full report: `experiments/forecast_training_experiment.md` (`model/forecast_training_experiment.py`,
data fetched by `pipeline/fetch_prevruns.py`). Winters 2024-25 and 2025-26 only (the archive starts 2024-01-20);
frozen fit before each winter, identical eligible days for every arm, rule fixed in advance.

* The earlier "-11.3 %" (Hybrid vs V1, both trained on ERA5) and "-3.3 %" (Hybrid trained on forecasts vs Hybrid
  trained on ERA5) are different comparisons on the same 228 days. A2 here reproduces the second exactly.
* Result: no candidate met the rule. Train-on-forecast-only is clearly worse (+22 % MAE in 2025-26; it throws away
  the earlier winters). Mixed training: -3.3 % MAE, interval [-3.10, +0.78] (inconclusive), better in all 5 seeds,
  but worse on sudden-change days. Calibrating the forecast helps 2024-25 and hurts 2025-26.
* The forecast-vs-ERA5 gap inside the winter windows is small and differs between winters (the earlier
  "wind lower by 2-4 km/h" was an all-year average). A perfect weather forecast would improve MAE by only ~6 %.
* Decision: keep training on observed weather. Suggested (needs approval): log the forecast weather the Lambda
  uses each night so winter 2026-27 yields an exact-live training set.
