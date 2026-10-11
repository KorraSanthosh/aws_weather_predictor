# Model versions (kal-ki-hawa)

Two models are stored side by side. **Nothing here overwrites either one**: `model/CHECKSUMS.sha256` lists the SHA-256 of
every stored file and `tests/test_core.py::test_stored_artifacts_match_checksums` fails if one changes;
`model/train.py` refuses to write into an existing `model/<version>/` directory unless `--force` is given.

| | `v1_25_baseline` | `v2_35_candidate` |
|---|---|---|
| Role | **Baseline - the default model** (Lambda, dashboard, tests) | **Experimental candidate** - not deployed, not the default |
| Features | 25 | 35 |
| Needs hourly data | No | Yes: 7 day-part features (NaN when a day has < 16 hourly readings) |
| Feature list | `FEATURE_SETS["v1_25_baseline"]` in `core/features.py` | `FEATURE_SETS["v2_35_candidate"]` |
| Recipe | XGBoost, log-change target, same params and seed for both | same |
| Trained through | 2026-10-07 | 2026-10-07 |

A third, predeclared candidate **`v3_hybrid`** is a router (v2 when all 7 day-part inputs exist, else v1) with no model file of its own.

## Which file is which

| File | What it is |
|---|---|
| `model/v1_25_baseline/model.json`, `model_meta.json`, `metrics.json`, `web_{forecast,history,metrics}.json` | The baseline: the exact v1 artifacts that were committed, plus the dashboard demo data it produced |
| `model/v2_35_candidate/model.json`, `model_meta.json`, `metrics.json` | The 35-feature candidate |
| `model/model.json`, `model/model_meta.json`, `model/metrics.json`, `web/data/{forecast,history,metrics}.json` | **The default**, byte-identical to the v1 files above. These are what `scripts/build_lambda.sh` packages and the dashboard reads |
| `model/CHECKSUMS.sha256` | Checksums of every stored version file |
| `experiments/model_comparison_v1_v2.json` | Paired comparison output (3 regimes, subsets, intervals) from `model/compare_versions.py` |
| `experiments/model_comparison_v1_v2_all_origins.json` | The unfiltered run that reproduces the 523-day / 70.4% figure |
| `experiments/feature_study_35.json` | The earlier feature study |
| `experiments/hybrid_study_historical.json` | Descriptive history of the hybrid rule (not evidence) |
| `model/EVALUATION_PROTOCOL.md`, `forward_protocol.py`, `forward_eval.py`, `freeze_manifest.py`, `forward_test_manifest.json` | The forward-test rules, scoring code and frozen-file hashes |
| `core/hybrid.py` | The hybrid router (`v3_hybrid`) |

## How to run each

```bash
# Train (writes model/<version>/ only; refuses to overwrite)
python model/train.py --version v1_25_baseline
python model/train.py --version v2_35_candidate
python model/train.py --version v1_25_baseline --promote    # also replaces the DEFAULT files - only on purpose

# Lambda: the model's own model_meta.json decides which features are built
MODEL_VERSION=v2_35_candidate        # env var; unset = packaged default model/model.json
bash scripts/build_lambda.sh v1_25_baseline    # package a specific version (no argument = the default files)

# Fair comparison
python model/compare_versions.py
```
Each forecast now reports `model_version`, `n_features` and `missing_features` (inputs that were NaN, i.e. missing, never filled in).

## What was verified (9 Oct 2026)

* Training and live prediction build features with the **same function** (`core.features.build_features`) driven by the loaded
  model's feature list; the Lambda refuses a model whose features it cannot build.
* **No future data:** features for target day D+1 read only D and earlier PM2.5 and D+1's *weather* (observed in tests,
  forecast live). Tests change D+1's PM2.5 / day-parts and weather after D+1 and assert nothing moves.
* **No fabricated hourly data:** days with fewer than 16 hourly readings (all of 2023-24, which comes from CPCB daily files)
  keep the day-part features as NaN; XGBoost treats NaN as missing, same as in training.
* **Known train/serve difference:** the nightly run (23:00 IST) sees hours 0-22 of today, training used 0-23. Measured in
  `EXPERIMENTS.md`: small (v2 forecasts move 2.3 ug/m3 on average, 5.6% of category labels).
* Live weather is a forecast; every test number uses observed weather, so live error will be somewhat higher for both.

## Results in one paragraph (details and limits in `EXPERIMENTS.md`)

Over the full evaluation period (1,261 days) v2 is **not meaningfully better**: MAE 20.5 vs 20.6 (-0.7%, 95% CI -3.9 to +2.4).
On the 519 days with hourly data it is better (MAE 13.4 vs 15.5, -13.8%, CI -19.5 to -7.6; category accuracy 71.5% vs 65.1%),
including on sudden-change days (-25.5%). On the 733 days without day-part data it is **worse** (+5.3%, CI +2.2 to +8.5).
The widely quoted **70.4% category accuracy applies only to the 523 hourly-data days** (v1 scores 66.2% on the same days).
Both numbers come from a period already examined in earlier experiments, so they are evidence, not proof.

## Evaluation rules (so no number gets inflated)

1. The 2023-01-01 -> 2026-10-07 period is **burned**: it has been examined, so it must not be used to choose features,
   parameters or between versions. Nothing was tuned on it; the v2 feature list was written down before any result.
2. Do not drop, add or re-weight features because of how they scored on that period. Any new idea gets a new, clearly named
   version and is judged only on data that did not exist when it was designed.
3. Always report hourly-data days and the full period separately, with the number of days each model was scored on.
4. Quote category accuracy only together with its day count (70.4% = 523 hourly days).

## Forward evaluation: winter 2026-27 (rules now written out)

The full, machine-checked rules are in **`model/EVALUATION_PROTOCOL.md`** (constants in `model/forward_protocol.py`, scoring in
`model/forward_eval.py`, frozen-file hashes in `model/forward_test_manifest.json`). In short: v1 baseline, v2 candidate and the
**`v3_hybrid` router** (v2 when all 7 day-part inputs exist, else v1; no model file of its own) forecast every night from identical
inputs, logged first-write-wins under `pk="shadow"`; matched days only; MAE primary, AQI-category accuracy secondary; separate
reports for day-part days, no-day-part days and the complete set; at least 90 matched winter days (target dates 2026-10-15 to
2027-02-15); promote only with MAE at least 5% lower, a 95% interval below zero and category accuracy not more than 2 points worse;
no accuracy look before 2027-02-16. Shadow logging is off by default (`SHADOW_VERSIONS`), so the default model and dashboard are unchanged.

`bash scripts/validate_working_copy.sh` checks all of this on your machine without changing anything.
