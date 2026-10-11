# Forward (shadow) test protocol, winter 2026-27

Version 1.0 - **PREPARED, not yet frozen.** It becomes binding when you commit it with `model/forward_test_manifest.json`
and tag the commit (section 9). Machine-readable constants: `model/forward_protocol.py`. Scoring code: `model/forward_eval.py`.

## 1. Question
Does a candidate beat the current baseline on data that did not exist when the models and the rules were fixed?

| Name | What it is | Role |
|---|---|---|
| `v1_25_baseline` | 25-feature model | Baseline and default. Stays the default unless a candidate passes section 8 |
| `v2_35_candidate` | 35-feature model (7 day-part features from hourly data) | Candidate 1 |
| `v3_hybrid` | A **router, not a model**: use v2 when all 7 day-part inputs are present for the forecast origin day, otherwise v1 | Candidate 2, predeclared separately |

Hybrid rule, in full: "all 7 day-part inputs finite" means `pm_night, pm_commute, pm_school, pm_evening, pm_late, pm_buildup,
pm_last3h_ratio` are all real numbers (the origin day has >= 16 hourly readings and every day-part had enough hours,
`core/daily.py`). Nothing else - no threshold, no tuning, no other variant (`core/hybrid.py`). Its behaviour on past data is
described in `EXPERIMENTS.md` and is **not** evidence.

## 2. Same inputs, no influence between models (rule 1)
* One nightly run builds the history and the weather once. Every version forecasts from that **same** data for the **same**
  target date. Each version has its own model file and builds its own features from a read-only copy of the history, so no
  version sees another's output. Order of running does not matter (tested).
* Every logged row carries `inputs_sha256`, a fingerprint of every candidate input. A day whose versions disagree on it is
  excluded and counted.
* A version that fails on a night is recorded as an error and never affects the others or the primary forecast.
* **First write wins:** a shadow forecast is stored once under `pk="shadow"`, `sk="<target date>#<version>"` and is never
  overwritten, so it cannot be edited after the actual is known.
* Shadow logging is off unless `SHADOW_VERSIONS` is set. It never changes the primary forecast, the alert, the dashboard or the default model.
* Models are frozen: trained through 2026-10-07, **no retraining and no re-tuning during the test.**

## 3. Window and matching
* A forecast counts when the day it **predicts** (target date) is between **2026-10-15 and 2027-02-15** inclusive: at most 124 days.
* A day is **matched** when: all three versions were logged; none errored; all used identical inputs; the actual PM2.5 for that day is
  known (>= 3 stations, >= 16 valid hours, same city definition as training).
* Unmatched days are dropped for all versions together and **counted by reason** in the report. Days are never dropped for
  any reason that depends on a forecast's accuracy.
* The 90-day requirement therefore needs at least 73% of the window to be matched. Check operational health weekly (section 7).

## 4. Metrics (rule 2)
* **Primary: MAE** in ug/m3 on the matched days.
* **Secondary: AQI-category accuracy** (CPCB category of forecast = category of actual).
* Also reported, not gated: RMSE, R2, poor-or-worse recall, and "tomorrow = today" persistence on the same days.

## 5. Segments (rule 3)
Every metric is reported for three sets of matched days, with their day counts:
1. `complete_eligible`: all matched days. **This is the gate set.**
2. `daypart_days`: matched days with valid day-part inputs.
3. `no_daypart_days`: matched days without them (v2 runs with the gaps; the hybrid uses v1).

## 6. Uncertainty
The 95% interval for each candidate's change in MAE (percent) and in category accuracy (percentage points) versus v1 comes from a paired
**7-day** block bootstrap (2,000 resamples, seed 0), because neighbouring days are strongly correlated.

## 7. No peeking
Before 2027-02-16 only operational numbers may be looked at: nights run, rows logged, errors, share of nights with valid day-parts,
matched-day count. **Accuracy of any version is not looked at before the single final analysis.** `forward_eval.py` refuses to run
earlier (`--force-early` stamps the output "NOT A VALID ANALYSIS"). There is one analysis, no re-runs with different settings.

## 8. Decision rule (rules 4-6), applied on `complete_eligible`
Fewer than **90** matched days, or fewer than **90** matched days with valid day-parts: **INCONCLUSIVE, v1 stays**. Otherwise a
candidate is promotable only if **all** hold:
1. MAE is **at least 5% lower** than v1 (point estimate <= -5.0%);
2. the **95%** interval of that MAE change lies entirely below zero;
3. category accuracy is **not materially worse**: the change is no lower than -2.0 percentage points.

* Neither passes: **KEEP v1.**
* One passes: promote that one.
* Both pass: promote `v3_hybrid` if any matched day lacks day-parts (it is the safer serving rule), otherwise `v2_35_candidate`.
* A v2 that wins on day-part days but fails the gate set because it is worse on the days without them is **not** promoted as a standalone
  model; the hybrid is judged on its own results.
* Two candidates are each tested against v1 at 95% with no multiplicity correction; this is disclosed and the tie rule is structural.
* Promotion only means "recommend changing the default". Changing the default and deploying remain separate steps that need your approval.

## 9. Freeze and amendments (rule 7)
Frozen = the model artifacts, feature definitions, day-part and AQI-category code, hybrid router, shadow-logging handler, scoring code,
constants, and this document, all listed with SHA-256 in `model/forward_test_manifest.json`.
1. Review; if anything changes, `python model/freeze_manifest.py --write` and re-run the tests.
2. Commit the manifest with the files and tag it (suggested tag: `forward-test-freeze-2026-10`). The tag date proves nothing was altered later.
3. `python model/freeze_manifest.py --check` must pass before deployment and before the final analysis.
An amendment after the freeze needs a new protocol version in the log below. Results seen before the amendment may not be used to justify it,
and the analysis states which version it used.

Amendment log: *(none)*

## 10. Limitations we already know
* One winter, one city; at most 124 days, and neighbouring days are correlated.
* Live weather is a forecast, whereas all past tests used observed weather; both versions are affected, not necessarily equally.
* The nightly run sees hours 0-22 of today (measured earlier: small effect).
* If the OpenAQ feed has gaps, days without day-parts will exist; that is exactly what segment 3 and the hybrid are for.
* A pass is evidence for this winter. It does not prove the gain holds in other years.
