# Extra NCR / upwind city data: does it improve Delhi's next-day forecast?

**RETROSPECTIVE** test. Calendar 2025 and winter 2025-26 were examined in earlier experiments, so neither is an untouched hold-out. The protocol and pass/fail rule are in `config.json` (sha256 `a19096417020…`, stored in `metrics.json`), written before any regional model was run. No default model, Lambda, dashboard output or frozen file was changed.

## Answer

**NO EVIDENCE OF IMPROVEMENT under the pre-declared rule.**

Adding PM2.5 from 44 government stations in 4 regions around Delhi did not make the forecast measurably better. Averaged over five training seeds, exact-category accuracy of the deployed V1 model is 62.0 % (2025) and 64.4 % (winter 2025-26); with the regional data it is 62.3 % and 64.2 %, and with 12 placebo columns (real regional numbers taken from a random other day of the same month) it is 61.7 % and 64.6 %. The single run the rule calls primary (seed 42) shows 63.1 % and 67.3 % for the regional model, but in winter that is the best of its five seeds (range 62.5 % to 67.3 %); one seed is a lucky draw, not a gain. The differences are inside the noise (none of the 12 regional-versus-control intervals, 2 variants x 3 models x 2 tests, excludes zero), the seed-averaged MAE change is about zero, and in winter the Hybrid gets about 3 % worse with the regional columns, the same damage that the placebo columns do on their own. The model does use the new columns (roughly a fifth to a quarter of its split gain), but no more than it uses placebo columns, which is what fitting noise looks like.

## What was added

| Group | Stations | Where | Group-days available (2018 / 2019 / 2020 / 2021 / 2022 / **2023** / **2024** / 2025 / 2026) |
|---|---|---|---|
| ncr_near | 19 | Noida, Gurugram, Faridabad, Ghaziabad … (≤ 55 km) | 127 / 188 / 235 / 121 / 98 / 0 / 0 / 304 / 33 |
| haryana_nw | 13 | Rohtak, Panipat, Karnal, Jind, Bhiwani … (W / NW) | 0 / 184 / 220 / 109 / 101 / 0 / 0 / 130 / 33 |
| punjab | 7 | Ludhiana, Jalandhar … (stubble belt, PPCB) | 116 / 186 / 224 / 109 / 89 / 0 / 0 / 304 / 33 |
| up_east | 5 | Meerut, Hapur, Muzaffarnagar … (E, UPPCB) | 74 / 140 / 159 / 99 / 86 / 0 / 0 / 304 / 33 |

**The 2023 and 2024 hole.** The public OpenAQ archive holds no Indian government-station data for 2023 and 2024 (both layouts were checked) and only part of 2022. Regional inputs are therefore blank in those years, never filled in. The regional models learn from 433 (2025 test) and 645 (winter test) training days instead of 1184 / 1454. That can hide a real benefit; the restricted-training check below addresses it. Station files: 2.05 million hourly readings, 2018 → 2026-02-15; nothing later was read.

**Inputs.** Per region and day: the PM2.5 level, its ratio to Delhi's own value and its one-day change (12 numbers). *Previous day* = the last complete day before today (needs no new live-data assumption). *Same day* = today's hours up to 22:00 (needs the regional stations to report by 23:00 IST; not verified for the live Lambda). The Delhi target and every Delhi input are unchanged.

## Test 1: calendar 2025 (344 days, same days for every row)

| Model | Exact category (95 % CI) | Within one category | MAE µg/m³ (95 % CI) | RMSE | R² |
|---|---|---|---|---|---|
| Tomorrow = today (naive) | **58.1 %** [52.0, 63.5] | 96.5 % | 24.9 [19.5, 30.5] | 39.8 | 0.78 |
| V1 baseline (deployed default) | **61.6 %** [56.5, 66.8] | 97.1 % | 22.6 [17.9, 26.9] | 35.0 | 0.83 |
| V1 + regional, previous day (PRIMARY) | **63.1 %** [58.0, 68.3] | 97.1 % | 22.4 [17.7, 26.8] | 35.0 | 0.83 |
| V1 + regional, same day (needs data by 23:00) | **61.9 %** [56.4, 67.5] | 96.8 % | 22.6 [17.8, 27.1] | 35.0 | 0.83 |
| V1 + 12 PLACEBO columns | **62.8 %** [57.8, 67.9] | 97.1 % | 22.5 [17.8, 26.9] | 34.9 | 0.83 |
| V2 candidate | **66.0 %** [60.8, 70.4] | 98.5 % | 19.6 [15.6, 23.7] | 30.2 | 0.87 |
| Hybrid (V2 if day-parts, else V1) | **66.3 %** [61.1, 70.7] | 98.5 % | 19.6 [15.5, 23.6] | 30.2 | 0.87 |
| Hybrid + regional, previous day | **66.0 %** [60.3, 70.9] | 98.3 % | 19.5 [15.4, 23.8] | 30.8 | 0.87 |
| Hybrid + regional, same day | **67.7 %** [61.7, 72.6] | 98.3 % | 19.4 [15.4, 23.6] | 30.4 | 0.87 |

MAE intervals and the exact-category interval are 7-day moving-block bootstrap, 5000 resamples. Seed 42 shown; five seeds below.

### Paired differences

| Comparison (same days) | MAE change | 95 % interval of MAE change, µg/m³ (7-day / 14-day blocks) | Exact-category change | Seed-averaged MAE change |
|---|---|---|---|---|
| V1 + regional (prev. day) vs V1 | -0.59 % | [-0.55, 0.21] / [-0.55, 0.18] | +1.5 pts | +0.04 % |
| V1 + regional (same day) vs V1 | +0.12 % | [-0.63, 0.70] / [-0.76, 0.74] | +0.3 pts | -0.24 % |
| V1 + PLACEBO vs V1 | -0.20 % | [-0.39, 0.33] / [-0.42, 0.31] | +1.2 pts | n/a |
| Hybrid + regional (prev. day) vs Hybrid | -0.25 % | [-0.55, 0.50] / [-0.53, 0.50] | -0.3 pts | -0.42 % |
| Hybrid + regional (same day) vs Hybrid | -0.64 % | [-0.60, 0.40] / [-0.59, 0.45] | +1.5 pts | -0.68 % |

### How much the numbers move by chance (five training seeds)

The only thing that differs between seeds is which random 85 % of the training days and 85 % of the columns each tree gets to see (the production recipe samples both); data, features and test days are identical. One test day is 0.29 points of accuracy here.

| Model | Exact category: mean of 5 seeds | lowest – highest seed | MAE: mean of 5 seeds |
|---|---|---|---|
| V1 baseline (deployed default) | **62.0 %** | 61.3 % – 63.1 % | 22.4 |
| V1 + regional, previous day (PRIMARY) | **62.3 %** | 59.9 % – 63.4 % | 22.4 |
| V1 + regional, same day (needs data by 23:00) | **62.4 %** | 61.9 % – 63.1 % | 22.4 |
| V1 + 12 PLACEBO columns | **61.7 %** | 60.5 % – 62.8 % | 22.6 |
| Hybrid (V2 if day-parts, else V1) | **67.1 %** | 66.3 % – 68.3 % | 19.7 |
| Hybrid + regional, previous day | **66.8 %** | 66.0 % – 67.4 % | 19.6 |
| Hybrid + regional, same day | **67.7 %** | 66.9 % – 68.6 % | 19.5 |
| Hybrid + 12 PLACEBO columns | **67.9 %** | 66.9 % – 68.9 % | 19.7 |

A single seed can move a model's accuracy by several points by itself, which is more than any difference between the arms above. That is why a placebo that carries no information can still score a point higher in one seed.

Seeds (42, 7, 123, 2024, 99): V1 + regional beats V1 in 3 of 5 seeds; Hybrid + regional beats Hybrid in 4 of 5.

Regional inputs present (≥ 2 of 4 regions, previous day) on 284 of 344 test days; missing on 60.

## Test 2: winter 2025-26 (15 Oct 2025 → 15 Feb 2026) (104 days, same days for every row)

| Model | Exact category (95 % CI) | Within one category | MAE µg/m³ (95 % CI) | RMSE | R² |
|---|---|---|---|---|---|
| Tomorrow = today (naive) | **61.5 %** [51.0, 72.8] | 96.2 % | 41.6 [35.3, 50.4] | 53.6 | 0.42 |
| V1 baseline (deployed default) | **64.4 %** [52.8, 75.9] | 98.1 % | 33.3 [28.9, 39.6] | 45.4 | 0.58 |
| V1 + regional, previous day (PRIMARY) | **67.3 %** [56.3, 78.0] | 98.1 % | 32.8 [28.8, 39.0] | 44.4 | 0.60 |
| V1 + regional, same day (needs data by 23:00) | **66.3 %** [54.4, 77.0] | 98.1 % | 33.4 [28.6, 40.2] | 44.3 | 0.60 |
| V1 + 12 PLACEBO columns | **65.4 %** [55.1, 75.9] | 98.1 % | 33.9 [29.5, 40.2] | 45.6 | 0.58 |
| V2 candidate | **72.1 %** [61.3, 82.3] | 99.0 % | 28.6 [24.1, 35.7] | 38.8 | 0.70 |
| Hybrid (V2 if day-parts, else V1) | **72.1 %** [61.3, 82.3] | 99.0 % | 28.7 [24.3, 35.8] | 38.8 | 0.70 |
| Hybrid + regional, previous day | **68.3 %** [58.4, 79.3] | 99.0 % | 29.5 [25.2, 36.6] | 38.9 | 0.70 |
| Hybrid + regional, same day | **69.2 %** [58.9, 79.8] | 99.0 % | 30.0 [25.2, 38.1] | 40.0 | 0.68 |

MAE intervals and the exact-category interval are 7-day moving-block bootstrap, 5000 resamples. Seed 42 shown; five seeds below.

### Paired differences

| Comparison (same days) | MAE change | 95 % interval of MAE change, µg/m³ (7-day / 14-day blocks) | Exact-category change | Seed-averaged MAE change |
|---|---|---|---|---|
| V1 + regional (prev. day) vs V1 | -1.28 % | [-2.26, 1.52] / [-2.43, 1.72] | +2.9 pts | +0.59 % |
| V1 + regional (same day) vs V1 | +0.42 % | [-1.32, 1.90] / [-1.37, 1.94] | +1.9 pts | -0.75 % |
| V1 + PLACEBO vs V1 | +1.88 % | [-0.76, 2.12] / [-0.59, 2.04] | +1.0 pts | n/a |
| Hybrid + regional (prev. day) vs Hybrid | +3.02 % | [-0.38, 2.15] / [-0.17, 2.11] | -3.8 pts | +2.68 % |
| Hybrid + regional (same day) vs Hybrid | +4.62 % | [-0.23, 3.12] / [-0.07, 3.11] | -2.9 pts | +1.08 % |

### How much the numbers move by chance (five training seeds)

The only thing that differs between seeds is which random 85 % of the training days and 85 % of the columns each tree gets to see (the production recipe samples both); data, features and test days are identical. One test day is 0.96 points of accuracy here.

| Model | Exact category: mean of 5 seeds | lowest – highest seed | MAE: mean of 5 seeds |
|---|---|---|---|
| V1 baseline (deployed default) | **64.4 %** | 62.5 % – 65.4 % | 33.2 |
| V1 + regional, previous day (PRIMARY) | **64.2 %** | 62.5 % – 67.3 % | 33.4 |
| V1 + regional, same day (needs data by 23:00) | **66.2 %** | 63.5 % – 69.2 % | 33.1 |
| V1 + 12 PLACEBO columns | **64.6 %** | 63.5 % – 66.3 % | 33.9 |
| Hybrid (V2 if day-parts, else V1) | **70.0 %** | 67.3 % – 73.1 % | 29.0 |
| Hybrid + regional, previous day | **68.7 %** | 67.3 % – 70.2 % | 29.8 |
| Hybrid + regional, same day | **71.0 %** | 69.2 % – 72.1 % | 29.4 |
| Hybrid + 12 PLACEBO columns | **71.2 %** | 69.2 % – 73.1 % | 29.1 |

A single seed can move a model's accuracy by several points by itself, which is more than any difference between the arms above. That is why a placebo that carries no information can still score a point higher in one seed.

Seeds (42, 7, 123, 2024, 99): V1 + regional beats V1 in 1 of 5 seeds; Hybrid + regional beats Hybrid in 0 of 5.

Regional inputs present (≥ 2 of 4 regions, previous day) on 99 of 104 test days; missing on 5.

## Pre-declared rule

Regional data counts as HELPFUL only if ALL hold for the primary comparison v1+Rlag versus v1 (same seed 42, same days): (1) MAE at least 2 % lower on BOTH T1 and T2; (2) the paired 95 % interval of the MAE difference is below zero for BOTH the 7-day and the 14-day blocks on BOTH tests; (3) exact-category accuracy is not lower on either test; (4) the placebo arm's MAE gain is less than half of the real arm's on both tests; (5) the seed-averaged difference has the same sign. Otherwise the answer is: no evidence of improvement. Rd0 and the hybrid / v2 variants are reported but never decide anything. A pass would still only be a hypothesis for the winter 2026-27 shadow test and never a reason to change the default model.

| Test | 1. MAE gain ≥ 2 % | 2. interval below zero (7 d and 14 d) | 3. category accuracy not lower | 4. placebo gain < ½ real gain | 5. seed-averaged gain | 
|---|---|---|---|---|---|
| T1_calendar_2025 | +0.59 % → **fail** | [-0.55, 0.21] / [-0.55, 0.18] → **fail** | +1.5 pts → pass | placebo +0.04 vs real +0.13 µg → pass | +0.009 µg → **fail** |
| T2_winter_2025_26 | +1.28 % → **fail** | [-2.26, 1.52] / [-2.43, 1.72] → **fail** | +2.9 pts → pass | placebo -0.62 vs real +0.42 µg → pass | +0.195 µg → **fail** |

Result: **NO EVIDENCE OF IMPROVEMENT under the pre-declared rule**.

## Does the model actually use the regional columns?

- T1_calendar_2025: the 12 regional columns hold 22.2 % of V1's split gain; the 12 placebo columns hold 21.9 %. Using a column is not the same as being helped by it.
- T2_winter_2025_26: the 12 regional columns hold 24.0 % of V1's split gain; the 12 placebo columns hold 22.7 %. Using a column is not the same as being helped by it.
- Winter 2025-26, Hybrid: real regional columns +3.0 % MAE, placebo columns +3.2 % (14-day-block interval [0.19, 2.11] µg/m³). Merely widening the model with 12 columns costs accuracy on this short test; the regional data did not repay that cost.

## Supplementary (added after the first one-seed look; never decides anything)

Both V1 and V1 + regional trained only on days that have regional inputs, so the 2023-24 hole cannot dilute the new columns:

| Test | Training days | V1 exact category / MAE | V1 + regional exact category / MAE | MAE change | Seed-averaged | Seeds regional wins |
|---|---|---|---|---|---|---|
| T1_calendar_2025 | 433 | 62.2 % / 22.9 | 60.5 % / 24.3 | +6.01 % (interval [0.15, 2.83] µg) | +5.34 % | 0 of 5 |
| T2_winter_2025_26 | 645 | 59.6 % / 36.4 | 63.5 % / 35.9 | -1.40 % (interval [-2.63, 1.94] µg) | -1.12 % | 4 of 5 |

Even with the missing years taken out of the picture there is no benefit: in 2025 the regional model is +6.0 % MAE (interval above zero, i.e. worse); in winter it is -1.4 % (interval [-2.63, 1.94] µg/m³ spans zero; better in 4 of 5 seeds). A weak winter hint at most, and one that the 2025 result contradicts.

## Subsets (exploratory, V1 + regional vs V1, seed 42)

| Test | Subset | Days | V1 exact / MAE | V1 + regional exact / MAE | PLACEBO MAE | MAE change | interval µg (7 d) |
|---|---|---|---|---|---|---|---|
| T1_calendar_2025 | regional_available_ge2_groups | 284 | 62.7 % / 18.8 | 64.4 % / 18.7 | 18.9 | -0.12 % | [-0.44, 0.32] |
| T1_calendar_2025 | no_regional_data | 60 | 56.7 % / 40.5 | 56.7 % / 39.8 | 39.6 | -1.61 % | [-2.19, 0.50] |
| T1_calendar_2025 | stubble_period_15Oct-30Nov | 43 | 60.5 % / 34.2 | 65.1 % / 34.5 | 34.9 | +0.94 % | [-0.83, 2.22] |
| T1_calendar_2025 | sudden_change_ge30pct | 106 | 45.3 % / 37.1 | 46.2 % / 37.5 | 37.6 | +1.10 % | [-0.21, 1.00] |
| T1_calendar_2025 | stable | 238 | 68.9 % / 16.1 | 70.6 % / 15.7 | 15.8 | -2.32 % | [-0.94, 0.01] |
| T2_winter_2025_26 | regional_available_ge2_groups | 99 | 64.6 % / 33.0 | 67.7 % / 32.6 | 33.6 | -1.22 % | [-2.33, 1.68] |
| T2_winter_2025_26 | stubble_period_15Oct-30Nov | 43 | 65.1 % / 34.6 | 65.1 % / 35.4 | 36.1 | +2.32 % | [-1.67, 3.48] |
| T2_winter_2025_26 | sudden_change_ge30pct | 29 | 51.7 % / 60.7 | 51.7 % / 60.7 | 62.8 | -0.15 % | [-2.93, 3.87] |
| T2_winter_2025_26 | stable | 75 | 69.3 % / 22.6 | 73.3 % / 22.1 | 22.7 | -2.45 % | [-2.72, 1.60] |

The stubble-burning weeks (15 Oct – 30 Nov) are where Punjab / Haryana smoke should help most; there the regional model is not better either. Subsets are small and nothing here is a finding.

## Integrity checks (all passed; see tests/)

- T1_calendar_2025: 344 test rows rebuilt from a spying store, no regional date after the allowed day was read (previous-day inputs: ≤ today − 1; same-day inputs: ≤ today). The test-time 'same-day' values (hours ≤ 22) differ from full-day values on 292 rows, so the live view is really used. Canary (a column that IS tomorrow's answer) drives V1's MAE from 22.6 to 5.2, so a leak would be visible. The regional-free control reproduces the production fitting code exactly (V1: True, V2: True).
- T2_winter_2025_26: 104 test rows rebuilt from a spying store, no regional date after the allowed day was read (previous-day inputs: ≤ today − 1; same-day inputs: ≤ today). The test-time 'same-day' values (hours ≤ 22) differ from full-day values on 104 rows, so the live view is really used. Canary (a column that IS tomorrow's answer) drives V1's MAE from 33.3 to 6.9, so a leak would be visible. The regional-free control reproduces the production fitting code exactly (V1: True, V2: True).
- No regional station id appears in any Delhi station folder (44 regional vs 27 Delhi files). The Delhi target is unchanged.
- Re-running the whole experiment reproduces `metrics.json` byte for byte.

## Limits

- 2023 and 2024 regional data do not exist in the archive: the regional model learns from fewer rows, which can hide a real benefit.
- Winters 2025-26 and calendar 2025 were examined before; this is not an untouched test.
- Live use needs the regional stations to be fetched by the Lambda (OpenAQ API key, more calls, station outages). Not built and not tested here.
- Station-level government data is noisy; no outlier cleaning beyond the production 0-1000 rule.
- Crop-fire (satellite) data is not part of this experiment; the user will supply it later.
- Power: the 2025 test (344 days) can resolve an MAE change of roughly ±0.5 µg/m³ (about 2 %); the winter test (104 days) only about ±2 µg/m³ (about 6 %). A real gain smaller than that would not be seen.
- Why a gain is hard here: Delhi's own station average, yesterday's level and tomorrow's forecast wind already carry most of what upwind towns could tell a one-day-ahead model. That is an explanation, not something this experiment tested.

## Reproduce

```bash
python experiments/regional_cities/select_stations.py      # needs archive_coverage.json (rule-based station choice)
python experiments/regional_cities/download_regional.py   # public OpenAQ archive, no key; ~5 min; writes data/raw/regional/ (git-ignored)
python experiments/regional_cities/run_regional.py        # ~2 min; writes metrics.json and regional_daily.csv
python experiments/regional_cities/report.py              # writes this file
python -m pytest experiments/regional_cities/tests -q
```
