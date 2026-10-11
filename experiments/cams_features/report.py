import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text()); P = json.loads((HERE / "probe.json").read_text())
f2 = lambda x: f"{x:.2f}"; pct = lambda x: f"{100 * x:.1f}%"; ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
L = []; w = L.append
w("# Does CAMS forecast data improve the Hybrid model? (isolated experiment)\n")
w("**Result: a realistic test is not possible with the data available, and the non-realistic diagnostics do not show a reliable improvement. Keep the production model unchanged.** Nothing in the project was modified, committed, pushed or deployed.\n")
w("## 1. Why a realistic test could not be run\n")
s = P["summary"]
w(f"* The candidate needs the CAMS PM2.5 forecast for tomorrow *as it was issued before 23:00 IST today*. Open-Meteo's air-quality API documents no previous-run, historical-forecast or single-run option, and I probed it: `pm2_5_previous_day1/2` returned **null for every hour in all {s['periods_answered']} periods tested** (Sep 2022, Jan 2023, Jun 2024, Jan 2025, Oct 2025, Jan/Apr/Jul/Oct 2026), while plain `pm2_5` is always filled. The previous-runs service has no air-quality route (HTTP error).")
w("* The only CAMS history in the project (`data/raw/cams_daily.csv`, 2022-08-04 onwards) was fetched with start/end dates, which returns Open-Meteo's stitched *past* series (the early hours of each model run), not a day-ahead forecast. For a target day it carries information from runs issued during that day, which the Lambda cannot have at 23:00 the night before. It has never been used as a model input in this project; it is only a benchmark (CAMS alone: MAE 49.99 in the earlier final test).")
w("* True archived CAMS forecasts exist at the Copernicus Atmosphere Data Store (ADS), which needs an account and API key; none is available in this session and I did not look for one.\n")
w("So I did **not** build a \"realistic\" candidate and I am not presenting any result below as a forecast-time accuracy. I ran two clearly labelled diagnostics to find out whether CAMS carries any usable signal at all.\n")
w("## 2. Diagnostics (not realistic; same eligible days, same real PM2.5, same weather forecast inputs, same fixed hyper-parameters, frozen fit before the test)\n")
w("* **O_target_day (optimistic upper bound):** adds the stitched CAMS mean of the *target day*, the origin day and their log-change. It knows something about the day being predicted that a 23:00 forecast would only approximate.\n* **O_lags_only (nearly realistic):** only the origin-day and previous-day CAMS means. Residual caveat: the origin-day mean includes hours after 22:59 IST.\n")
for wn, v in R["windows"].items():
    b = v["result"]; cb = b["cams_benchmark_alone"]
    w(f"### {wn}: {b['n_days']} identical test days ({b['first']} → {b['last']})\n")
    w(f"Training origins {v['train_origins']} (only {v['train_origins_with_cams_target']} have CAMS: it starts 2022-08-04; the rest are left missing, handled natively by XGBoost). CAMS alone on these days: MAE {cb['mae']:.1f}, mean bias {cb['mean_bias']:+.1f} µg/m³, correlation with next-day PM2.5 {cb['corr_with_target_day_pm25']:.2f} (origin-day CAMS: {cb['corr_origin_day_cams_with_target_day_pm25']:.2f}).\n")
    w("| model | MAE | RMSE | R² | exact category |\n|---|---|---|---|---|")
    for arm, lab in (("O_target_day", "O_target_day"), ("O_lags_only", "O_lags_only")):
        pass
    a0 = b["O_target_day"]["models"]
    for m, lab in (("persistence", "persistence"), ("hybrid", "**current Hybrid (production recipe)**")):
        x = a0[m]; w(f"| {lab} | {f2(x['mae'])} | {x['rmse']:.1f} | {f2(x['r2'])} | {pct(x['cat_acc'])} |")
    for arm in ("O_lags_only", "O_target_day"):
        x = b[arm]["models"]["hybrid_c"]; w(f"| Hybrid + CAMS, {arm} | {f2(x['mae'])} | {x['rmse']:.1f} | {f2(x['r2'])} | {pct(x['cat_acc'])} |")
    w("\n| candidate − current Hybrid | Δ MAE | Δ % | 95% CI (7-d blocks) | 95% CI (14-d blocks) | Δ RMSE | Δ exact-cat (pts) | seeds better (of 5) | sudden-change MAE (Hybrid → candidate) | stable MAE |\n|---|---|---|---|---|---|---|---|---|---|")
    for arm in ("O_lags_only", "O_target_day"):
        a = b[arm]; d = a["hybrid_c_vs_hybrid"]
        w(f"| {arm} | {d['mae']:+.2f} | {d['mae_pct']:+.1f}% | {ci(d['mae_ci95_7d'])} | {ci(d['mae_ci95_14d'])} | {d['rmse']:+.2f} | {100 * d['cat_acc']:+.1f} | {a['seeds_candidate_better']} | {a['sudden']['hybrid']['mae']:.1f} → {a['sudden']['hybrid_c']['mae']:.1f} (n={a['sudden']['n_days']}) | {a['stable']['hybrid']['mae']:.1f} → {a['stable']['hybrid_c']['mae']:.1f} |")
    w("")
w("## 3. Reading\n")
c25 = R["windows"]["calendar_2025 (train <= 2024-12-31)"]["result"]; cw = R["windows"]["winter_2025_26 (train < 2025-10-15)"]["result"]
d25, dw = c25["O_target_day"]["hybrid_c_vs_hybrid"], cw["O_target_day"]["hybrid_c_vs_hybrid"]
l25, lw = c25["O_lags_only"]["hybrid_c_vs_hybrid"], cw["O_lags_only"]["hybrid_c_vs_hybrid"]
w(f"* **The nearly-realistic arm gives no improvement:** adding origin-day CAMS lags changes MAE by {l25['mae']:+.2f} (2025) and {lw['mae']:+.2f} (winter 2025-26) µg/m³, i.e. slightly worse, with the candidate better in only {c25['O_lags_only']['seeds_candidate_better']}/5 and {cw['O_lags_only']['seeds_candidate_better']}/5 seeds.")
w(f"* **Even the optimistic target-day arm is not convincing:** {d25['mae']:+.2f} µg/m³ ({d25['mae_pct']:+.1f}%) over 2025 with intervals {ci(d25['mae_ci95_7d'])} / {ci(d25['mae_ci95_14d'])} (the 14-day interval includes zero), and {dw['mae']:+.2f} µg/m³ ({dw['mae_pct']:+.1f}%, interval {ci(dw['mae_ci95_7d'])} includes zero) in winter 2025-26, where exact-category accuracy falls {abs(100 * dw['cat_acc']):.1f} points (it rose {100 * d25['cat_acc']:.1f} points over the full year, so even that direction is not consistent). This arm uses information a real forecast would only approximate, so a realistic CAMS forecast would be expected to do no better.")
w("* CAMS on its own is a poor predictor of Delhi PM2.5 (bias −26 µg/m³ in 2025, −87 in winter 2025-26), so any benefit would have to come from the model learning a correction, which the diagnostics do not demonstrate.\n")
w("## 4. Limitations\n* Retrospective; 2025 and winter 2025-26 were examined before. CAMS exists only from 2022-08-04, so ~62% of training origins have it. Stitched CAMS differs from real forecasts. The training and test CAMS definitions match each other, but neither equals what the Lambda would receive. One point (28.6139, 77.2090; the API snaps to ~28.60, 77.20). No tuning on the test period.\n")
w("## 5. Recommendation\n")
w("1. **Do not change the production model, default, dashboard or Lambda.** There is no demonstrated improvement, and the realistic condition cannot be tested retrospectively.\n2. If you still want a genuine answer: either obtain archived CAMS forecasts from the Copernicus ADS (needs your account), or, with your approval, have the Lambda *log* tomorrow's CAMS forecast at 23:00 IST during winter 2026-27 as a shadow field (no effect on predictions) and test it prospectively. That is a production change, so I have not made it.\n3. Higher-value ideas are still open: the OpenCity hourly-data experiment and the forward-test freeze.\n")
(HERE / "report.md").write_text("\n".join(L)); print("ok")
