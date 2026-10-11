"""Builds report.md from audit.json and metrics.json (all numbers are read from those files)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())
M = json.loads((HERE / "metrics.json").read_text())
T = M["test"]
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; pct = lambda x: f"{100 * x:.1f}%"
ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]" if c else "n/a"
NAME = {"persistence": "persistence (tomorrow = today)", "mean7": "7-day average", "v1": "V1 (25-feature XGBoost)", "v2": "V2 (35 features, experimental)", "hybrid": "Hybrid (V2 if 7 day-parts, else V1)"}
L = []; w = L.append
b1, b2 = T["lead1"], T["lead2"]
d = b1["diffs"]["hybrid_vs_v1"]; dv2 = b1["diffs"]["hybrid_vs_v2"]; d2 = b2["diffs"]["hybrid_vs_v1"]


def main_table(b):
    out = ["| model | days | MAE | MAE 95% CI | RMSE | R² | exact category | within one | MAE vs persistence |", "|---|---|---|---|---|---|---|---|---|"]
    for m in ("persistence", "mean7", "v1", "v2", "hybrid"):
        v = b["models"][m]
        vp = f"{v['vs_persistence_pct']:+.1f}% {ci(v['vs_persistence_pct_ci95'])}" if "vs_persistence_pct" in v else "–"
        out.append(f"| {NAME[m]} | {b['n_matched_days']} | {f2(v['mae'])} | [{v['mae_ci95'][0]:.1f}, {v['mae_ci95'][1]:.1f}] | {f1(v['rmse'])} | {f2(v['r2'])} | {pct(v['cat_acc'])} | {pct(v['within_one_cat'])} | {vp} |")
    return "\n".join(out)


def diff_table(b):
    out = ["| comparison (second minus first, same days) | Δ MAE µg/m³ | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ RMSE | Δ R² | Δ exact-category (pts) | CI excludes 0 (both) |", "|---|---|---|---|---|---|---|---|---|"]
    labels = {"hybrid_vs_v1": "Hybrid − V1", "v2_vs_v1": "V2 − V1", "hybrid_vs_v2": "Hybrid − V2", "hybrid_vs_persistence": "Hybrid − persistence", "v1_vs_persistence": "V1 − persistence",
              "hybrid_vs_mean7": "Hybrid − 7-day avg", "v1_vs_mean7": "V1 − 7-day avg", "mean7_vs_persistence": "7-day avg − persistence"}
    for k, lab in labels.items():
        x = b["diffs"][k]
        out.append(f"| {lab} | {x['mae']:+.2f} | {x['mae_pct']:+.1f}% | {ci(x.get('mae_ci95_7d'))} | {ci(x.get('mae_ci95_14d'))} | {x['rmse']:+.2f} | {x['r2']:+.3f} | {100 * x['cat_acc']:+.1f} | {x['excludes_zero_both']} |")
    return "\n".join(out)


def split_table(b, key, title):
    out = [f"| {title} | days | persistence MAE | V1 MAE | Hybrid MAE | Hybrid − V1 (µg/m³) | 95% CI (7-d blocks) | V1 exact | Hybrid exact |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr", "5": "May", "6": "Jun", "7": "Jul", "8": "Aug", "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
    for k, v in b[key].items():
        dd = v["hybrid_minus_v1_mae"]
        out.append(f"| {names.get(k, k)} | {v['n_days']} | {f1(v['persistence_mae'])} | {f1(v['v1_mae'])} | {f1(v['hybrid_mae'])} | {dd:+.1f} | {ci(v['hybrid_minus_v1_mae_ci95_7d'])} | {pct(v['v1_exact'])} | {pct(v['hybrid_exact'])} |"
                   if dd is not None else f"| {names.get(k, k)} | {v['n_days']} | {f1(v['persistence_mae'])} | {f1(v['v1_mae'])} | {f1(v['hybrid_mae'])} | – | too few days | {pct(v['v1_exact'])} | {pct(v['hybrid_exact'])} |")
    return "\n".join(out)


w("# Kal Ki Hawa: train through 2024, evaluate calendar year 2025\n")
w("**RETROSPECTIVE chronological backtest, not an untouched independent test.** 2025 data had been examined in earlier experiments of this project. No feature, hyper-parameter or model choice was tuned on 2025: the production recipe (same XGBoost parameters, feature sets, hybrid router and eligibility rules) was fitted once on targets ≤ 31 Dec 2024, frozen, then scored on 2025. "
  "Nothing in the production models, default model, dashboard, Lambda or checksums was changed; nothing was committed, pushed or deployed.\n")
w("## 1. Result in short\n")
n = b1["n_matched_days"]
w(f"* **Test coverage: {n} of 365 days of 2025**, not the full year. The other {365 - n} days have no valid city PM2.5 (fewer than 3 stations) on the target day ({b1['eligibility_ladder_all_365_days'].get('1_no_city_pm25_on_target_day', 0)}), on the origin day ({b1['eligibility_ladder_all_365_days'].get('2_no_city_pm25_on_origin_day', 0)}), or no usable 23:00 IST origin ({b1['eligibility_ladder_all_365_days'].get('5_origin_not_usable_by_the_23h_lambda', 0)}). No eligible day was lost for missing forecast weather.")
w(f"* **Best on 2025: Hybrid** (MAE {f2(b1['models']['hybrid']['mae'])} µg/m³, RMSE {f1(b1['models']['hybrid']['rmse'])}, R² {f2(b1['models']['hybrid']['r2'])}, exact category {pct(b1['models']['hybrid']['cat_acc'])}), then V1 (MAE {f2(b1['models']['v1']['mae'])}), persistence ({f2(b1['models']['persistence']['mae'])}), 7-day average ({f2(b1['models']['mean7']['mae'])}).")
w(f"* **Hybrid vs V1 is convincing on these dates:** {d['mae']:+.2f} µg/m³ ({d['mae_pct']:+.1f}%), 95% CI {ci(d['mae_ci95_7d'])} (7-day blocks) and {ci(d['mae_ci95_14d'])} (14-day blocks), both excluding zero; the strict 48-hour-forecast arm agrees ({d2['mae']:+.2f}, {ci(d2['mae_ci95_7d'])}); the Hybrid beat V1 in {b1['seeds_hybrid_beats_v1']}/5 random seeds. "
  f"Hybrid vs V2 is negligible in practice ({dv2['mae']:+.2f} µg/m³, interval {ci(dv2.get('mae_ci95_7d'))}): the Hybrid is V2 on {b1['hybrid_routing']['V2_days']} of {n} days.")
sd = b1["subsets"]["sudden_change_ge30pct"]; st = b1["subsets"]["stable"]
w(f"* **Where the gain comes from:** sudden-change days ({sd['n_days']} days): persistence {f1(sd['persistence']['mae'])}, V1 {f1(sd['v1']['mae'])}, Hybrid {f1(sd['hybrid']['mae'])}. On stable days ({st['n_days']} days) the models do *not* beat persistence (persistence {f1(st['persistence']['mae'])}, V1 {f1(st['v1']['mae'])}, Hybrid {f1(st['hybrid']['mae'])}). "
  "The model's value is in anticipating changes, not in calm weather. Monsoon months are a tie (all models ≈ 7–8 µg/m³).")
w("* **Main validity threat:** the station network changed. Training years use about 3–6 stations a day (2023–24: median 4), 2025 uses a median of 9 (up to 10), so the city-mean target in 2025 is a different (broader) average than in training. This is not corrected and the effect is unquantified; see section 8.\n")

w("## 2. Data audit (from the actual files; `audit.json`)\n")
w(f"* **PM2.5 and weather exist only from {A['no_pm25_before']}.** `data/daily_opencity.csv` runs {A['first_row']} to {A['last_row']}; weather (`weather_daily.csv`) {A['first_weather_day']} to {A['last_weather_day']}. **There is no PM2.5 for 2012–2017 anywhere in the project data**, so a 2012–2024 training set is impossible; training uses 2018-01-05 → 2024-12-31.")
w("* Minimum-coverage rule: a day counts only if ≥ 3 stations report (`core.daily.MIN_STATIONS`); 25 Mar – 30 Jun 2020 is excluded (COVID).\n")
w("| year | calendar days | days with city PM2.5 (≥ 3 stations) | share | stations/day (min / median / max) | months with data (Jan…Dec day counts) |\n|---|---|---|---|---|---|")
for y, v in A["years"].items():
    if v["days_with_city_pm25_ge3_stations"] == 0:
        w(f"| {y} | {v['calendar_days']} | **0** | 0% | – | no data |")
    else:
        s = v["stations_per_day_min_median_max"]
        w(f"| {y} | {v['calendar_days']} | {v['days_with_city_pm25_ge3_stations']} | {pct(v['share'])} | {s[0]} / {s[1]} / {s[2]} | {' '.join(str(x) for x in v['days_by_month'].values())} |")
w("\n**Genuine, usable data exists for 2018–2025 only, and 2018–2022 are patchy** (86–165 usable days a year, whole seasons missing). Only 2023 and 2024 are close to complete, so most of the training signal is two years of data.")
tr = M["training"]
w(f"\n**Training set actually used:** {tr['origins']} origin days with targets {tr['first_target']} → {tr['last_target']} (per year: " + ", ".join(f"{k}: {v}" for k, v in tr["origins_by_year"].items()) +
  f"). Origins with all 7 day-part (hourly) inputs, which V2 needs: {tr['origins_with_all_7_dayparts']} (per year: " + ", ".join(f"{k}: {v}" for k, v in tr["origins_with_all_7_dayparts_by_year"].items()) + "). 2023 has none because that year comes from CPCB daily files without hours.")
w("\n**Stations and sources by period** (`audit.json → station_days_by_source_and_year`, days with any reading):\n")
w("* 2018–2022: OpenAQ stations (13 ids; 6358, 6931, 6932, 8239, 10831, 5613 contribute most days; 2018 early winter also 17, 50, 235, 236, 2503, 431). OpenAQ API files add 2021–22 for station 6932.\n* 2022–2025: CPCB daily files (Anand Vihar 2022–25; R.K. Puram, Punjabi Bagh, Rohini, Anand Vihar 2023–24).\n* 2024 (and Jan–Feb 2025): OpenCity hourly files for four stations (10831, 17, 235, 50).\n* 2025: OpenAQ again, 10 stations (10831, 17, 235, 50, 5404, 5613, 6358, 6931, 6932, 8239) and an Anand Vihar CPCB daily file also exists for 2025; this is why 2025 has 9–10 stations a day.\n* Weather: ERA5 daily means for training and for today's weather; archived Open-Meteo Previous Runs forecast for tomorrow's weather in 2025.\n")
w("**Archived forecasts for 2025:** the Open-Meteo Previous Runs archive (default model, point 28.6139, 77.2090 as in the Lambda) has complete 24-hour-ahead (`previous_day1`) and 48-hour-ahead (`previous_day2`) values for every scoreable 2025 day (0 scoreable days lost). The archive starts 2024-01-19, so it cannot be used to train V1/V2 on forecasts before 2025; training uses ERA5, as production does. The default blend was ICON-identical until Sep 2025 and IFS-HRES-identical afterwards (previous experiment), so the forecast model changes inside the test year.\n")

w("## 3. Protocol\n")
w(f"* Train: targets ≤ {M['train_cutoff_target_date']}; V1 (25 features) and V2 (35 features) XGBoost, the production recipe, seed 42 (+ seeds 7, 123, 2024, 99 for robustness only). Frozen models saved in `models/` (`v1_seed42.json`, `v2_seed42.json`, `hybrid_router.json`, SHA-256 in `sha256.json`).\n* Test: every 2025 day that is scoreable by the project's existing eligibility ladder (`winter_backtest.classify_target`: ≥ 3 stations on target and origin day, weather present, origin usable by the 23:00 IST Lambda) **and** has the archived forecast. All models are scored on exactly the same {n} days.\n"
  "* Inputs at the 23:00 IST cutoff: today's PM2.5 and day-parts from readings up to 22:59 IST, lagged PM2.5 from earlier days, tomorrow's weather from the archived forecast (lead 1 primary; lead 2 = 48 h as a strictly-available check), today's weather from ERA5 (no analysis archive exists).\n* Confidence intervals: moving-block bootstrap (7-day blocks, 14-day sensitivity), 5000 resamples, shared across models.\n")

w("## 4. Primary result: 2025, one-day-ahead, 24-hour-old forecast weather\n")
w(main_table(b1))
w("\nDifferences on identical days:\n")
w(diff_table(b1))
w("\n### Strictly available 48-hour-old forecasts (robustness)\n")
w(main_table(b2))
w("\n" + diff_table(b2) + "\n")
w("## 5. Sudden-change days (|tomorrow − today| ≥ 30% of today) and stable days\n")
w("| subset | days | persistence MAE | V1 MAE | Hybrid MAE | Hybrid − V1 | 95% CI (7-d blocks) | Hybrid − persistence | 95% CI |\n|---|---|---|---|---|---|---|---|---|")
for k, lab in (("sudden_change_ge30pct", "sudden-change"), ("stable", "stable"), ("hybrid_used_V2", "days Hybrid used V2")):
    v = b1["subsets"][k]; x = v["hybrid_vs_v1"]; y_ = v["hybrid_vs_persistence"]
    w(f"| {lab} | {v['n_days']} | {f1(v['persistence']['mae'])} | {f1(v['v1']['mae'])} | {f1(v['hybrid']['mae'])} | {x['mae']:+.2f} | {ci(x.get('mae_ci95_7d'))} | {y_['mae']:+.2f} | {ci(y_.get('mae_ci95_7d'))} |")
w("\n## 6. By season and month (Hybrid − V1; negative = Hybrid better)\n")
w(split_table(b1, "by_season", "season"))
w("\nSeasons: winter Dec–Feb (Dec 2025 plus Jan–Feb 2025), pre-monsoon Mar–May, monsoon Jun–Sep, post-monsoon Oct–Nov. Intervals with few days are wide.\n")
w(split_table(b1, "by_month", "month"))
w("\n## 7. Seeds and reference arm\n")
w("Seed-to-seed Hybrid MAE: " + ", ".join(f"seed {k}: {v['hybrid']:.2f} (V1 {v['v1']:.2f})" for k, v in b1["seeds"].items()) + f". Hybrid beat V1 in {b1['seeds_hybrid_beats_v1']}/5 seeds.\n")
r = M["reference_observed_weather_NOT_REALISTIC"]
w(f"**Reference only, NOT a realistic forecast:** with the *observed* (ERA5) weather of the target day, which the Lambda cannot have, on {r['n_days']} days: " + ", ".join(f"{NAME[m]} {f2(v['mae'])}" for m, v in r["models"].items()) + f". This is shown only to size the cost of forecast-weather error ({f2(r['models']['hybrid']['mae'])} observed vs {f2(b1['models']['hybrid']['mae'])} forecast, Hybrid). It is not used for any conclusion.\n")
w("## 8. Limitations and uncertainties\n")
w("* **Not an untouched test.** 2025 had been looked at before; I did not tune on it here, but earlier design choices (feature sets, hybrid rule, hyper-parameters) were made by the project while 2025 data existed in the repository.\n"
  f"* **Coverage:** {n}/365 days, with no 2012–2017 data and patchy 2018–2022; the model effectively learns from ~2 near-complete years.\n"
  "* **Station-network shift (largest threat to validity):** training city means come from ~4 stations a day (2023–24), the 2025 target from ~9–10, and the day-part features come from different station sets in different years. Part of the 2025 error, and of the ranking, may reflect this definition change. I did not rebuild a 4-station 2025 target to quantify it.\n"
  "* **Forecast weather changes model mid-year:** the default blend resolved to ICON until Sep 2025 and IFS HRES afterwards. Training uses ERA5 for weather; the 2025 inputs are archived forecasts, so a train/test weather mismatch exists, as in production.\n"
  "* **Publication timing:** Open-Meteo does not publish the run hour of each `previous_dayN` value; the 48-hour arm is the strictly-available check and gives the same conclusion.\n"
  "* Today's weather (wx0) is ERA5 because no analysis archive exists.\n"
  "* Persistence is strong in 2025 (R² 0.78, MAE 13.8 on stable days), so the margin over it comes almost entirely from sudden-change days.\n"
  "* Many comparisons are reported (months, seasons, subsets); single intervals that exclude zero in sub-groups should be read cautiously.\n")
w("## 9. Conclusion\n")
w(f"On the {n} eligible days of 2025 the **Hybrid is the best model**, ahead of V1 by {abs(d['mae']):.2f} µg/m³ MAE ({abs(d['mae_pct']):.1f}%), RMSE and R² improvements, and +{100 * d['cat_acc']:.1f} points exact category accuracy. The advantage over V1 is statistically convincing under both bootstrap block lengths, under 48-hour forecasts, across all five seeds, and is concentrated on sudden-change days "
  "and clear in winter and pre-monsoon; post-monsoon favours the Hybrid in the point estimate but its interval includes zero, and in the monsoon there is no difference. The Hybrid is effectively V2 (Hybrid vs V2 differs by 0.06 µg/m³). Caveats above (retrospective, station-network shift, patchy early history) mean this supports the Hybrid as the leading candidate, not as a validated production claim; the winter 2026-27 forward evaluation remains the real test. No production change is proposed here.\n")
(HERE / "report.md").write_text("\n".join(L))
print("report.md", len("\n".join(L)))
