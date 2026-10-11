# Kal Ki Hawa — tomorrow's air for Delhi schools

**Environmental Hacks (AWS × WeMakeDevs) · Track: Air**

AQI apps say how bad the air is *now*. Schools need to know how bad it will be *tomorrow* to plan outdoor
activities. Kal Ki Hawa trains a machine-learning model on real ground-station PM2.5 data, forecasts
tomorrow's PM2.5 for Delhi, converts it to the official CPCB AQI category, and tells schools **GO / CAUTION /
INDOOR** every night, with an email alert when tomorrow is Poor or worse.

## Architecture (all on AWS)

```
OpenAQ open data (s3://openaq-data-archive) ─┐
Open-Meteo weather ──────────────────────────┴─► train on laptop/SageMaker ─► model.json
                                                                                 │ packaged into
EventBridge (23:00 IST) ─► Lambda: fetch last days + weather forecast ─► XGBoost ─┘
                              ├─► DynamoDB (forecast + later the actual, for live accuracy)
                              └─► SNS email alert if Poor or worse
Website (Amplify) ─► API Gateway (HTTP API) ─► Lambda ─► DynamoDB
```

## What is verified and what is not (read this first)

Status on 11 October 2026.

| Part | Status |
|---|---|
| AQI conversion, daily aggregation, features (no leakage), forecast Lambda logic, API handler, dashboard JS | **Tested.** The test suite (`python -m pytest -q`) covers them on faked inputs, including a known-answer check of the stored models and tests that future data cannot reach the features |
| Data and training | **Run on real data.** `data/daily.csv` is real OpenAQ ground-station PM2.5 (1,823 days, 2018-01-04 to 2026-10-07) and the stored models were trained on it. All accuracy figures are retrospective: three winters, each predicted by a model trained only on earlier days, using observed weather |
| SAM stack and Lambda package | **Deployed** on 11 Oct 2026 (stack `kal-ki-hawa`, ap-southeast-2, CREATE_COMPLETE). The deployed package was checked (numpy, scipy and xgboost 3.4.1 import) and the SNS subscription is confirmed |
| First live nightly run (OpenAQ and Open-Meteo calls from inside the Lambda, forecast row, alert e-mail) | **Attempted on 11 Oct 2026; the function refused to forecast.** Its freshness check stopped it: the newest valid PM2.5 day it could see was 7 Oct, too old to forecast from. OpenAQ had published nothing for these 10 stations since 7 Oct 20:00 IST, neither in its API nor in its open-data archive. No forecast has been stored, so the dashboard shows "No live forecast" until fresh data arrives |
| Forward test, winter 2026-27 | **Set up, not started.** Shadow logging of V2 and the hybrid is deployed and V1 still serves. Runs on 11-13 Oct are rehearsal; the scored window opens with the run on 14 Oct 23:00 IST (target date 15 Oct) and is not analysed before 16 Feb 2027 (`FORWARD_TEST_OPERATIONS.md`) |
| Dashboard hosting (Amplify) | Hosted on AWS Amplify at https://live.d1fte73g29q7ow.amplifyapp.com by a manual zip upload; this is not part of the stack. `web/config.js` in the repo is empty, so the repo copy runs in offline demo mode, while the hosted copy sets `KKH_API` to the stack's `ApiUrl` |

Synthetic results are only there to prove the code runs. **Never show synthetic numbers as real.** The site shows a
"DEMO DATA" banner whenever the data source is synthetic.

## Run it on real data (your laptop)

```bash
pip install -r requirements.txt
export OPENAQ_API_KEY=...            # free: https://explore.openaq.org/register
rm -f data/daily.csv data/daily.meta.json model/model*.json   # remove any synthetic leftovers

python pipeline/find_stations.py              # list Delhi PM2.5 stations; then edit config/stations.json
                                              # (keep Delhi only; the auto-pick can include Gurugram/Noida)
python pipeline/download_openaq.py --start 2023-01-01   # AWS open-data bucket (no AWS login needed)
python pipeline/backfill_openaq_api.py --start 2022-01-01   # older history via the API (the archive only
                                                            # reaches back to ~Feb 2025 for these stations)
python pipeline/download_weather.py --start 2022-01-01  # Open-Meteo weather + CAMS benchmark
python pipeline/build_dataset.py              # data-quality report -> data/daily.csv
python model/train.py --version v1_25_baseline   # writes model/v1_25_baseline/ (never overwrites; add --promote to make it the default)
# two model versions are stored side by side - see model/MODEL_VERSIONS.md (v1 baseline = default, v2 candidate = experimental)
python -m pytest -q
```

Then preview the site: `cd web && python -m http.server 8000` and open http://localhost:8000.

**Check first (day one):** the data-quality report. It prints how many *winters* (15 Oct – 15 Feb) have 90+
valid days. The model is evaluated on winters, so you want **3 or more**; one winter is not enough to test
honestly. With one winter, train.py tests the peak (1 Dec – 15 Feb) using only earlier days and says so
(`late_winter_holdout`); with even less it falls back to a non-winter test and warns loudly.
A day counts as a "city" day only if 3+ stations reported (`--min-stations`), so the average means the same
thing throughout; the Lambda applies the same rule.

**Adding more winters (recommended):** `python pipeline/find_stations.py --history` writes
`config/stations_history.json` (stations with data from before Oct 2022, even if they stopped reporting).
Then `python pipeline/download_openaq.py --stations config/stations_history.json --start 2018-01-01`,
`python pipeline/download_weather.py --start 2018-01-01`, `python pipeline/build_dataset.py`,
`python model/train.py --version v1_25_baseline --force --promote` (re-trains the default). With 2+ earlier winters train.py switches to full-winter folds on its own.
train.py also reports a 95% bootstrap interval for the gain over persistence, calm vs spike days, per-month
errors and (for short histories) a monthly walk-forward check.

**How the model is evaluated:** rolling winter folds. Each fold tests on one 15 Oct → 15 Feb window with a
model trained only on earlier days, and the results are pooled. The older `pm25_low`/`pm25_high` band in the
forecast record comes from those winter errors (the dashboard now shows the calibrated range described under
"Prediction range" instead). In winter nearly every day is Poor or worse, so "flagged Poor-or-worse" is an easy number
(even "tomorrow = today" scores high). The model predicts the *change from today's PM2.5* (trees cannot
extrapolate to winter levels they never saw), so it starts from the persistence baseline and has to beat it; judge the model mainly on MAE and AQI-category accuracy.

## Model-improvement experiments
`EXPERIMENTS.md` documents a strict development/test study (feature groups, tuning, LightGBM / Random Forest /
HistGradientBoosting, sample weighting, season-aware training, direct AQI classifier). Nothing beat the deployed
model on the untouched 2023+ test, so the model is unchanged; the document also quantifies the weather-forecast
gap (expect roughly 60-64% category accuracy live). Reproduce with `python model/experiments.py dev|final`.
The Lambda now fetches 16 days of history so any PM2.5 lag up to 14 days is available live (guarded by a test).
CPCB portal daily files go in `data/raw/cpcb/` (`build_dataset.py` reads them).

## Prediction range (90 %)

Under the forecast the dashboard shows a **likely range**, e.g. "Good to Moderate (PM2.5 30–81 µg/m³)". It shows how
uncertain the forecast is. It does **not** change the point forecast, the AQI category, the school decision or the alert.

- **Method:** split-conformal ("M1") in log1p space. The offsets come from 721 *out-of-sample* forecasts (2024-01-20 →
  2026-02-15) made by models trained only on earlier data, using archived 24 h forecast weather
  (`core/range_calibration.json`, inputs in `experiments/prediction_range/calibration_residuals.json`).
  Bounds are whole numbers rounded outward, so the range is never narrower than computed and every number matches its category.
- **Where it runs:** `core/prediction_range.py`, applied by `backend/api_handler.py` when the forecast is read (from the stored
  `pm25` and `model_version`). The nightly forecast Lambda, the stored rows and the forward-test shadow log are unchanged. If a
  range cannot be given (unknown model version, any error) the field is simply absent. The offline demo reads
  `web/data/range.json`, made by `python scripts/make_range_json.py`; `web/data/forecast.json` is not modified.
- **Retrospective evidence** (calibration strictly before each test period; `experiments/prediction_range/product_check.json`):

  | Test | Actual PM2.5 inside range | Actual category inside category range | Mean categories in range |
  |---|---|---|---|
  | Calendar 2025 (344 days) | 93.3 % computed, 95.1 % as shown (outward rounding) | 98.0 % | 2.45 / 2.48 |
  | Winter 2025-26 (104 days) | 90.4 % | 98.1 % | 2.41 |

- **What it is not:** it is not the model's accuracy. The point forecast's exact-category accuracy is 66 % (2025) and 72 %
  (winter 2025-26), and the naive "predicted category ± 1" range already contains the actual category about 98.5 % (2025) and
  99.0 % (winter) of the time at about 2.9 categories wide. So the 98 % above shows the range is calibrated, not that the model
  improved. Quote the point-forecast MAE and exact-category accuracy as the model's accuracy.
- **Limits:** ranges are wide (about 2–3 of the 6 categories); they describe past errors and are not a guarantee; the models that
  produced the calibration residuals were retrained at each window, not the final shipped models; if the live station network or
  weather source changes, coverage can change. Once live `actual_pm25` values accumulate in DynamoDB, check the share of days
  inside the shown range and rebuild with `python experiments/prediction_range/build_calibration.py` if it drifts.

## Deploy to AWS

```bash
bash scripts/build_lambda.sh
sam deploy --guided --template-file template.yaml   # params: OpenAQApiKey, SensorIds (from config/stations.json), AlertEmail
```
**Do not run `sam build`.** `build/lambda` is already a finished package (Linux wheels included). `sam build` copies a folder that has
no `requirements.txt` with an exclusion list that contains `*.so`, so it silently drops the compiled parts of numpy, scipy and xgboost
and the Lambda then fails on `import numpy`. (The upload is then ~15 MB instead of ~60 MB.) `sam deploy` zips the folder as it is.

Put the `ApiUrl` output into `web/config.js`, then host the `web/` folder on AWS Amplify (drag-and-drop deploy).
Confirm the SNS subscription email. For a normal (non forward-test) setup you may invoke the forecast function once by hand to create the
first forecast; during the forward test do not (see `FORWARD_TEST_OPERATIONS.md`).

## Things to know / improvements

- **Weather mismatch:** training uses *observed* weather for the target day; live use has a weather *forecast*.
  Real-world error will be somewhat higher. Better: train with Open-Meteo's "historical forecast" API.
- **Judge-friendly proof:** `model/metrics.json` compares the model with "tomorrow = today", a 7-day mean and
  the CAMS global model. Quote the real numbers. If RMSE is worse than persistence (big spikes missed), say so
  honestly and lead with MAE and category accuracy, or try `objective="reg:pseudohubererror"` / no log transform.
- Possible extras: 48 h forecast, more stations, Hindi UI, SageMaker for training (to show more AWS in the video).
- The guidance text is general information, not medical advice.

## Submission checklist (3 things)
1. Public GitHub repo (commit as you go; history must fall inside Oct 8–11).
2. ≤ 3 min YouTube video (public/unlisted, test signed-out): problem → who it's for → live demo → **show AWS
   (Lambda/DynamoDB/EventBridge/S3 console)** → the accuracy chart with real numbers.
3. Short writeup: problem, build, where AWS fits, **list AI coding tools used**.
