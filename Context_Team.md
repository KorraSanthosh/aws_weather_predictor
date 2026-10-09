# Kal Ki Hawa — Team Context

Written Fri 9 Oct 2026 (IST) so that everyone working on the project has the same picture.
Read this top to bottom once; after that use it as a checklist.

---

## 1. The one-minute version

**Kal Ki Hawa** ("tomorrow's air") forecasts **tomorrow's PM2.5 for Delhi**, converts it to an
Indian (CPCB) **AQI category**, and gives schools a plain **GO / CAUTION / INDOOR** call.
It is built for the **AWS x WeMakeDevs "Environmental Hacks"** hackathon.

- Nightly pipeline on AWS: EventBridge -> Lambda -> DynamoDB + SNS email alerts + HTTP API.
- Dashboard: map of Delhi stations, forecast, school decision, accuracy charts, Hindi/English, light/dark.
- Model: XGBoost on the *change from today's PM2.5*, using today's PM2.5 history and tomorrow's weather.

**Status today:** data + model + tests + backend code + dashboard are done and pushed to GitHub.
**Not done:** the AWS deployment, the demo video, the writeup, and the submission.

## 2. Hackathon rules and deadline

- Event: 8-11 Oct 2026. **Submit by Sunday 11 Oct** (the exact hour is not published, so aim for Saturday night).
- Must **use AWS** (or deploy on AWS). The **demo video must show AWS** (console / running services).
- Submission = **public GitHub repo** + **video of 3 minutes or less** + **writeup that lists the AI tools used**.
- **Repo history must match the event dates** — never rewrite or fake commit dates (we have not).
- Repo: https://github.com/KorraSanthosh/aws_weather_predictor (branch `main`).

## 3. Architecture

```
OpenAQ (stations) + Open-Meteo (weather)
        |
EventBridge (nightly) -> Lambda: forecast_handler  -> DynamoDB (pk "delhi", sk date)
                                      |-> SNS (email alert when forecast is Poor or worse)
API Gateway (HTTP API) -> Lambda: api_handler -> returns latest forecast + history
Amplify (static hosting) -> dashboard, which calls the API
```
Defined in `template.yaml` (AWS SAM). The OpenAQ API key is a **NoEcho SAM parameter**, never a literal.

## 4. Repo map

| Path | What it is |
|---|---|
| `core/` | Shared pure-Python logic: `aqi.py` (PM2.5 -> AQI/category/guidance), `daily.py` (hourly -> IST daily city mean), `weather.py`, `openaq.py`, `features.py` (the *same* feature builder is used in training and in the Lambda, so there is no train/serve skew) |
| `pipeline/` | Data scripts: `find_stations.py`, `download_openaq.py`, `download_weather.py`, `backfill_openaq_api.py`, `topup_recent.py`, `build_dataset.py`, `build_station_map.py` |
| `model/` | `train.py` (trains, writes `model.json`, metrics, dashboard JSON), `experiments.py` (walk-forward study), the trained model + metadata |
| `experiments/` | Saved results of the improvement study (see section 7) |
| `backend/` | `forecast_handler.py` (nightly Lambda), `api_handler.py` (HTTP API Lambda) |
| `template.yaml`, `scripts/` | SAM template, Lambda packaging, synthetic-data and robust-run helpers |
| `web/` | First dashboard (plain HTML/JS). Still works; reads `web/data/*.json` |
| `frontend/` | **React (Vite) dashboard** — the newer, nicer one (see section 8) |
| `config/` | `stations.json` (10 curated Delhi stations + sensor IDs), `stations_history.json`, `station_coords.json` (approximate map coordinates) |
| `data/daily.csv` | The built daily dataset (small, committed). Raw downloads in `data/raw/` are **git-ignored** |
| `tests/` | pytest suite (38 tests passed in our environment) |
| `README.md`, `EXPERIMENTS.md` | Setup + the full experiment write-up |

## 5. Data

- **Sources:** OpenAQ (S3 archive + v3 API), Open-Meteo (weather, archive/forecast), CAMS (global model, only used as a benchmark), CPCB portal daily files (filled the gap years).
- **Dataset rules:** IST days; a station-day needs >= 16 valid hours; values <0 or >1000 dropped; a "city" day needs **>= 3 Delhi stations**; Gurugram/Faridabad/Noida/Ghaziabad stations are excluded (they leaked into the Delhi mean once and we fixed that).
- **COVID lockdown** (2020-03-25 to 2020-06-30) can be excluded with `--exclude`.
- **Coverage is uneven:** the 2022-23 and 2024-25 winters are only partly covered; 2023-24 is built from only **four CPCB stations** (Anand Vihar, R K Puram, Punjabi Bagh, Rohini).
- Datasets we rejected: a Kaggle "Delhi NCR 2020-2025" file (synthetic) and a 2023 CAMS file (identical to the CAMS series).
- `data/raw/` is not in Git (24 MB). To recreate it, see the run order below.

### Re-creating the data and model (run from the repo root, venv active)
```
export OPENAQ_API_KEY=...            # your own key; NEVER write it into a file
python pipeline/find_stations.py     # optional: rediscover stations
python pipeline/download_openaq.py   # S3 archive (no key needed)
python pipeline/download_weather.py
python pipeline/backfill_openaq_api.py
python pipeline/topup_recent.py      # newest days from the live API
python pipeline/build_dataset.py     # -> data/daily.csv
python model/train.py                # -> model/*.json and web/data/*.json
python pipeline/build_station_map.py # -> web/data/stations.json (map data)
pytest -q
```
(Check `README.md` and each script's `--help` for flags. CPCB daily files go in `data/raw/cpcb/`.)
macOS needs `brew install libomp` for XGBoost; our venv runs Python 3.9.

## 6. The model, in plain words

- Target: `delta = log1p(PM2.5 tomorrow) - log1p(PM2.5 today)`. (A first attempt that predicted the raw level failed badly — trees cannot predict beyond levels they have seen — so we predict the *change* instead.)
- 25 features: PM2.5 lags (0,1,2,3,6 days), 3/7-day means, 7-day std, 1-day change, today's and tomorrow's weather (temperature, humidity, wind, wind components, rain, pressure), day-of-year sin/cos.
- Uncertainty range comes from pooled residual quantiles (q10/q90).
- Baselines we always compare to: **tomorrow = today**, the 7-day average, and the CAMS global model.

## 7. How accurate is it? (honest numbers)

Strict time-based testing only (no shuffling): the model for each month is trained on earlier days only.
Choices were made on a **development period (2019-2022)**; the **final test (2023 onward, 1,262 days) was run once**.

| Test | MAE (ug/m3) | AQI-category accuracy | vs "tomorrow = today" |
|---|---|---|---|
| Final test, all 1,262 days | 20.7 | 64.4% | **13.4% lower error** (95% CI 9.4-16.9%) |
| Winter folds (3 winters, 306 days) | 41.0 | 67.6% | 12.6% lower error (CI 5.7-18.7%) |

- 97% of predictions are within one AQI category of the truth.
- The model caught about 91% of Poor-or-worse days on the final test.
- **We tried to improve it** (more features, tuning, other models, classifier, sample weights) and **none beat the simple model on unseen data**, so the original model (v1) stays deployed. Details: `EXPERIMENTS.md`.
- **Main caveat:** training and testing use *observed* weather for tomorrow; live use will have a weather *forecast*. Realistic live accuracy is about **60-64% category accuracy and 9-13% lower error than "tomorrow = today"**.
- A target of "99.99% accuracy" is not realistic for this problem and we should not claim anything like it. The honest framing (beats persistence, catches bad days, within one category 97% of the time) is the strong, defensible one.

## 8. What has been built and is done

**Data/ML/backend (committed and pushed, 43 commits, one file each):** pipeline, dataset, model, experiment harness, Lambda handlers, SAM template, tests, docs.

**Dashboard (React, in `frontend/` — written and tested in a browser, but NOT yet committed):**
- Hero banner: a drawn hand holding a glass sphere with a live AQI gauge, the category, and the school decision.
- Forecast card, school-decision card, four stat tiles (stations reporting, highest, lowest, Poor-or-worse).
- **Delhi map** (Leaflet): coloured AQI pins + soft coloured zones around each station, toggle "Latest measured / Tomorrow (estimate)", **Street / Satellite** switch.
- Searchable station list; click a station to zoom the map and see an **area photo card** (photo looked up live from Wikipedia, with a credit link).
- Forecast-vs-actual chart with hover values; accuracy card.
- **Light / Dark / System** theme.
- **English / Hindi** toggle for everything (categories, school advice, dates).

Old plain-HTML dashboard in `web/` also got the map (kept as a fallback).

### Things to know about the dashboard
- The "Tomorrow (estimate)" map view scales each station by the *city* forecast; it is **not** a per-station model. The page says so — keep that wording.
- Station coordinates (`config/station_coords.json`) were entered from memory and are **approximate**.
- Area photos come from a Wikipedia search and may not match the exact monitor site; an optional exact-page override (`"wiki"`) is described in the code but **not wired up yet**.
- The Hindi text was written by us/Claude and **needs a native read** before showing to judges.
- Only 7 of 10 stations had a complete reading for 7 Oct; the other three show "n/a".
- Run locally: `cd frontend && npm install && npm run dev` (needs Node; `brew install node`). `npm run build` makes the deployable `dist/`.

## 9. Security rules (please keep these)

- **Never commit `.env`, API keys, AWS credentials, tokens, or private keys.** `.gitignore` already blocks `.env*`, keys, `.venv/`, caches, `node_modules/`, `data/raw/`.
- The OpenAQ key lives only in the `OPENAQ_API_KEY` environment variable (locally) or the SAM NoEcho parameter (AWS).
- **The OpenAQ key was pasted in plain text during development — regenerate it before deploying.**
- Never force-push or rewrite history (rules say history must match the event dates).

## 10. What is left to do (in order)

1. **Commit and push the dashboard work** (`frontend/`, map changes in `web/`, `config/station_coords.json`, `pipeline/build_station_map.py`, `web/data/stations.json`) — one file per commit, as before.
2. **Regenerate the OpenAQ key.**
3. **Deploy to AWS** (the part judges must see):
   - find the live PM2.5 **sensor IDs** (needed in `SENSOR_IDS`);
   - `scripts/build_lambda.sh`, then `sam build && sam deploy --guided` (key goes in as the hidden parameter; set an alert email);
   - invoke the forecast Lambda once; confirm a DynamoDB row and the SNS email;
   - put the API URL in the frontend (`VITE_KKH_API` in `frontend/.env`, copied from `.env.example`); host `frontend/dist` on **Amplify** (build command `npm run build`, output `dist`, app root `frontend`);
   - double-check the CORS setting on the HTTP API so the dashboard can call it.
4. **Let it run overnight** so there is at least one real, logged live prediction (a started forward test).
5. **Record the demo video (<= 3 min):** dashboard (map, forecast, school decision, Hindi toggle) -> AWS console (Lambda, DynamoDB, EventBridge, SNS, Amplify) -> an alert email.
6. **Write the writeup:** problem, approach, architecture, honest accuracy (section 7), limitations, and the **AI tools used** (Claude / Claude Code were used for coding and analysis — list this plainly).
7. **Final checks and submit:** repo public, README up to date, history intact, video link works, submit before Sunday.

## 11. Nice-to-have if time allows (cut these first if we run short)

- Wire up the `wiki` photo override and add our own station photos.
- Hindi proof-read by a native speaker; add Hindi station names.
- Test with archived weather *forecasts* (Open-Meteo "historical forecast" API) to measure the weather-forecast gap exactly.
- Tests for the frontend; a GitHub Action running pytest.
- A short README section with screenshots and the architecture diagram.
- Optional blog post for a prize.

## 12. Decisions and questions for the team

1. **Who does what?** Suggested split: one person on AWS deploy + Amplify, the other on the video + writeup + README polish; both review accuracy claims.
2. **Which dashboard do we deploy** — `frontend/` (React, recommended) or `web/` (plain)? Do we keep both?
3. **Which sensor IDs** go in `SENSOR_IDS`, and do we want fewer stations if some stay offline?
4. Do we show the map's "Tomorrow (estimate)" view in the video, given it is a scaled city estimate?
5. Who proof-reads the Hindi?
6. Realistic claims for the writeup: agree on the exact numbers and caveats (section 7) before anyone writes them.

## 13. Risks

- Deadline is Sunday with an unknown hour — leave buffer; deploy early.
- OpenAQ API limits / outages — the Lambda must fail gracefully; keep the bundled demo data as a fallback for the video.
- Live accuracy will be lower than the test numbers (forecast weather) — say so ourselves before a judge finds it.
- Data gaps: some stations offline on some days; some winters thin.
- Costs: keep everything within the AWS free tier / hackathon credits; delete resources after judging if needed.
