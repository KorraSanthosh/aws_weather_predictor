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

| Part | Status |
|---|---|
| AQI conversion, daily aggregation, features (no-leakage), forecast Lambda logic, API handler, dashboard JS | **Tested** (24 unit tests + browser-simulation run) on synthetic/faked data |
| Training script | **Runs end to end** — but only on SYNTHETIC data so far. Real accuracy is unknown until you run it on real data |
| OpenAQ station search, S3 archive download, OpenAQ v3 API calls, Open-Meteo calls | **Written from the public docs, never run against the live services** (the build sandbox had no access). Expect to fix small things on day one |
| SAM template, Lambda packaging, Amplify deploy | **Not deployed/tested** |

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
python model/train.py                         # metrics + model + website demo data
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
`python model/train.py`. With 2+ earlier winters train.py switches to full-winter folds on its own.
train.py also reports a 95% bootstrap interval for the gain over persistence, calm vs spike days, per-month
errors and (for short histories) a monthly walk-forward check.

**How the model is evaluated:** rolling winter folds. Each fold tests on one 15 Oct → 15 Feb window with a
model trained only on earlier days, and the results are pooled. The uncertainty range in the app comes from
those winter errors. In winter nearly every day is Poor or worse, so "flagged Poor-or-worse" is an easy number
(even "tomorrow = today" scores high). The model predicts the *change from today's PM2.5* (trees cannot
extrapolate to winter levels they never saw), so it starts from the persistence baseline and has to beat it; judge the model mainly on MAE and AQI-category accuracy.

## Model-improvement experiments
`EXPERIMENTS.md` documents a strict development/test study (feature groups, tuning, LightGBM / Random Forest /
HistGradientBoosting, sample weighting, season-aware training, direct AQI classifier). Nothing beat the deployed
model on the untouched 2023+ test, so the model is unchanged; the document also quantifies the weather-forecast
gap (expect roughly 60-64% category accuracy live). Reproduce with `python model/experiments.py dev|final`.
The Lambda now fetches 16 days of history so any PM2.5 lag up to 14 days is available live (guarded by a test).
CPCB portal daily files go in `data/raw/cpcb/` (`build_dataset.py` reads them).

## Deploy to AWS

```bash
bash scripts/build_lambda.sh
sam build && sam deploy --guided      # params: OpenAQApiKey, SensorIds (from config/stations.json), AlertEmail
```
Put the `ApiUrl` output into `web/config.js`, then host the `web/` folder on AWS Amplify (drag-and-drop deploy).
Confirm the SNS subscription email. Invoke the forecast function once by hand to create the first forecast.

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
