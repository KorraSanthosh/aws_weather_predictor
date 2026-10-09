# Model-improvement experiments (kal-ki-hawa)

**Result in one line:** none of the tested changes beat the existing model on data it had never seen,
so the deployed model is unchanged. The honest, defensible performance is below.

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
