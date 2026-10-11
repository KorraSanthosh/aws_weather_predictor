# Forward test: operations guide (winter 2026-27)

How to run the **V1-serves / Hybrid-in-shadow** evaluation and know that it is really working. The *rules* of the test (window,
metrics, decision) live in `model/EVALUATION_PROTOCOL.md` and are frozen; this file is only about operating it.

**Status of this document: PREPARED, not deployed.** Nothing described here is running until the stack is deployed and
`forward_ops.py ops` says so (section 6). Do not say "the forward test has started" before then.

## 1. What runs

| | what | where it is decided |
|---|---|---|
| **Served forecast** (dashboard, school decision, alerts) | **V1 `v1_25_baseline`**, always | `MODEL_VERSION` is **not set** → `forecast_handler` default |
| **Shadow forecasts** (never shown, never alert) | V1, V2 `v2_35_candidate`, Hybrid `v3_hybrid`, all from the *same* inputs | `ShadowVersions` stack parameter → `SHADOW_VERSIONS` |
| **Provenance record** (evidence for each night) | one item per forecast night | written by `backend/forward_logger.py` |

The Lambda handler is `forward_logger.handler`. It calls the **unchanged, frozen** `forecast_handler.handler` and watches it.
It may change only two things, both protections, both recorded: (1) an hourly reading whose hour had not ended at the run time is
not passed on; (2) a shadow row generated at or after 00:00 IST of the day it predicts is not stored. With a well-behaved feed
and the 23:00 IST schedule neither ever triggers. If `ShadowVersions` is empty the wrapper simply calls the frozen handler.
**Rollback at any time:** set `Handler: forecast_handler.handler`.

DynamoDB partitions (one table): `pk="delhi"` primary forecast (what the API reads), `pk="shadow"` sk `<target>#<version>`,
`pk="provenance"` sk `<target>` (or `<date>#failed#…`, `<target>#late#…`). Shadow and provenance rows are **first write wins**:
a logged forecast can never be replaced after the actual is known. Actuals are attached to shadow rows the following nights by
the frozen handler.

## 2. Configuration

| setting | value | note |
|---|---|---|
| `Handler` | `forward_logger.handler` | already in `template.yaml` |
| `ShadowVersions` (stack parameter) | `v1_25_baseline,v2_35_candidate,v3_hybrid` | **default is empty = logging OFF**; you must pass it |
| `MODEL_VERSION` | *not set* | setting it changes what is **served** – do not |
| Package | `SHADOW=1 bash scripts/build_lambda.sh` | without `SHADOW=1` the v1/v2 folders are missing and shadow rows record an error |
| Schedule | `cron(30 17 * * ? *)` = 23:00 IST | unchanged |
| xgboost in the package | **`xgboost-cpu==3.4.1`** (≥ 3.1 is required) | the stored models need it: xgboost < 3.1 reads their `base_score` wrongly and every forecast comes out ~1.65× too high. `scripts/build_lambda.sh` pins it and fails the build otherwise; `tests/test_model_numeric.py` fails on an old xgboost |
| `Timeout` / `MemorySize` | 60 s / 512 MB | unchanged; ten sequential OpenAQ calls normally take well under 30 s, but OpenAQ retries back off up to 2+4+6+8 s each – see 7 |

## 3. The 23:00 IST cutoff and the hourly feed

* The run at 23:00 IST forecasts for the next IST day. Hour 23 does not exist yet; hours 0–22 may.
* **An hour is used only if it had ended when the run started.** The API is asked for `datetime_to = run time`, and the wrapper
  independently drops anything else (count in `hourly_feed.cutoff`; `readings_used_from_unfinished_hours` must always be `0`).
* The Hybrid uses V2 only if **all seven day-part inputs** are finite. At 23:00, `dp_late` needs 2 of hours 20–22 and `dp_last3`
  needs **both hours 21 and 22** (`core/daily.py`: `max(2, len//2)`). So the Hybrid has its inputs **only if the feed already
  holds the hour that ended at 23:00 (22:00–23:00 IST)**. If the newest hour is 21:00–22:00 the inputs are incomplete
  (`scripts/check_hourly_feed.py --selftest` shows 0 of 8 days).
* **If an hour is missing nothing is invented.** The day-part stays NaN, the Hybrid routes to V1 (that night Hybrid ≡ V1), the
  provenance record says which hours were missing and why (`hourly_feed.daypart.fallback_reasons`), and the night still counts in
  the `complete_eligible` set and in `no_daypart_days`.
* **This is the main uncertainty.** The offline studies assumed hours ≤ 22 are visible. How late OpenAQ publishes the newest hour
  is not documented, and the sandbox this was prepared in has no API key. Measure it (section 4) *before* judging the Hybrid:
  if hours 21–22 are mostly absent at 23:00, `daypart_days` will stay below the protocol's 90-day minimum and the test cannot
  conclude about the day-part model. That would be a finding about the feed, not about the model.

## 4. Commands

```bash
# A. Is the hourly feed good enough?  Read-only GETs to OpenAQ. Run it at 22:58–23:10 IST, on several nights.
read -rs OPENAQ_API_KEY && export OPENAQ_API_KEY   # paste your key when prompted (hidden); it stays in this shell only
python scripts/check_hourly_feed.py --out feed_check.json
python scripts/check_hourly_feed.py --selftest      # offline proof that the analysis works

# B. Build the package (does not deploy anything)
SHADOW=1 bash scripts/build_lambda.sh               # also writes build/lambda/build_info.json (commit, branch, dirty flag)

# C. Operations report: counts only, never accuracy (needs your AWS credentials)
python scripts/forward_ops.py ops --table <TableName> --region <region>

# D. Can every logged night be re-created from its record?  (re-runs the models from the repo files)
python scripts/forward_ops.py reproduce --table <TableName> --region <region> [--date 2026-10-15]

# E. All checks without AWS
bash scripts/validate_working_copy.sh               # includes the whole test-suite
python model/freeze_manifest.py --check
```

## 5. What each provenance record contains

`forecast_generated_at_utc/ist`, `target_date`, `origin_date`, `in_protocol_window` · `primary` (model_version, served_model, pm25,
category, school decision) · `shadow` (per version: served_model, pm25, range, AQI, category, daypart_valid, `inputs_sha256`,
missing features) · `hourly_feed` (per sensor: rows received/used, newest finished hour and its lag, which origin-day hours;
city hours present, stations per hour, hour 21/22/23 flags, per-part coverage, fallback reasons, cutoff audit) · `weather`
(Open-Meteo forecast URL, variables, call parameters, origin- and target-day values, "forecast issued before the target day") ·
`inputs` (the daily history, the target weather and the full feature row – enough to rebuild the row and re-run both models) ·
`software` (sha256 of the code and model files that ran, build info, Python/xgboost versions) · `request` (Lambda request id,
event time). **Never stored:** the OpenAQ key, AWS credentials, table name, topic ARN.

## 6. When may anyone say the forward test has started?

1. The stack is deployed from a **committed** tree (check `build_info.json`: `git_dirty: false`; `null` means git was not available and proves nothing) with `ShadowVersions` set and
   `MODEL_VERSION` unset.
2. After a 23:00 IST run, `forward_ops.py ops` shows `logging has started: YES`, three shadow rows and one provenance record
   for the night, `primary rows NOT served by V1: 0`, `readings_used_from_unfinished_hours: 0`.
3. `forward_ops.py reproduce` prints `OK` for that night.
4. Nights before **2026-10-15** (the first target date in the window) are *rehearsal*: they prove the plumbing but are not part of
   the test. The test has started when the night that forecasts **2026-10-15** (run on 14 Oct at 23:00 IST) is logged and passes 2–3.

Until 2027-02-16 look only at `ops` (operational counts). Do not compute or look at accuracy; `model/forward_eval.py` refuses
to run earlier. **Do not** invoke the Lambda from the console during the day to test it: such a run forecasts a day that has
already begun, the primary row for that day is overwritten (existing frozen behaviour), and the recorder refuses to log it as a
forecast (a `#late#` provenance record explains why). Test only after 23:00 IST, or wait for the schedule.

## 7. Known limits

* **Live hourly latency is unmeasured** (section 3) – the biggest risk to the day-part comparison.
* Existing frozen behaviour, not changed here: if the origin day has fewer than 16 hourly values at 23:00, the handler forecasts
  *today* from yesterday and overwrites today's primary row. The recorder refuses to log such a night for the test, so it shows
  up as a missing night in `ops`.
* The model was trained on full days (hour 23 included); at 23:00 it sees hours 0–22 (documented train/serve gap).
* Lambda timeout is 60 s; if OpenAQ is slow the run can fail (a `#failed#` record is written and the error is raised as before).
  Raising `Timeout` to 120 s is a one-line change you may want to review.
* One winter, one city, at most 124 daily forecasts; the protocol's minimum is 90 matched days.

## 8. Deployment steps (run only after you have reviewed and approved them; nothing here has been run)

1. Review and commit this work on `feature/forward-logging` (a clean tree makes `build_info.json` say `git_dirty: false`).
2. `bash scripts/validate_working_copy.sh` – every line must PASS.
3. Optional but recommended: `python scripts/check_hourly_feed.py` near 23:00 IST (section 4A).
4. `SHADOW=1 bash scripts/build_lambda.sh`, then open `build/lambda/build_info.json`: `shadow_package` must be `true`. The script also stops with an error unless `xgboost_cpu-3.4.1.dist-info` is in the package.
5. `sam deploy --guided --template-file template.yaml`. **Do not run `sam build`**: with no `requirements.txt` it copies the folder with an
   exclusion list that contains `*.so`, which removes the compiled parts of numpy/scipy/xgboost (upload ~15 MB instead of ~60 MB; the Lambda
   would fail on `import numpy`). Use the **new** OpenAQ key (the old one was exposed). Set
   `ShadowVersions` = `v1_25_baseline,v2_35_candidate,v3_hybrid`; do not add a `MODEL_VERSION` variable. The deploy upload line must show
   about 60 MB. Entering the key in a hidden prompt gives no feedback; type it with `read -rs KEY; echo ${#KEY}` first (must print 64) and check it with
   `curl -s -o /dev/null -w '%{http_code}' -H "X-API-Key: $KEY" https://api.openaq.org/v3/sensors/12234787` (must print 200), then pass
   `--parameter-overrides "OpenAQApiKey=$KEY" ...` to `sam deploy`.
5b. Verify the deployed package (no side effects, no key shown): `aws lambda get-function-configuration --function-name <ForecastFunction>` must show
   Handler `forward_logger.handler`, CodeSize ~60.5 MB, `SHADOW_VERSIONS` set, no `MODEL_VERSION`, and `length(OPENAQ_API_KEY)` = 64; download the zip
   (`aws lambda get-function --query Code.Location`) and import numpy, scipy, xgboost and load both models with `python3.12`.
6. The schedule fires at 23:00 IST. After the first run: `python scripts/forward_ops.py ops --table <TableName> --region <region>`
   and `... reproduce ...` (section 6). Runs on 11–13 Oct are rehearsal (targets 12–14 Oct); the window opens with the run on
   14 Oct 23:00 IST (target 15 Oct).
7. Every morning until 15 Feb 2027: `forward_ops.py ops`. Fix gaps (missing nights, fallbacks, failed runs); never edit logged rows.
