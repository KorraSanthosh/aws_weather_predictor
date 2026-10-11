"""Builds report.md from results.json (every number in the tables is read from the file)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
PROD = "production_default"
NAMES = {"production_default": "production default (best_match)", "ecmwf_ifs_hres": "ECMWF IFS HRES", "gfs_global": "NOAA GFS 0.11°",
         "icon_global": "DWD ICON", "ecmwf_ifs025": "ECMWF IFS 0.25°", "ecmwf_aifs025": "ECMWF AIFS 0.25°"}
f1 = lambda x: f"{x:.1f}"                                                             # noqa: E731
f2 = lambda x: f"{x:.2f}"                                                             # noqa: E731
ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]" if c else "n/a"                          # noqa: E731
pct = lambda x: f"{100 * x:.1f}%"                                                     # noqa: E731
L = []
w = L.append


def table(arm, sname, model="hybrid"):
    a = R["arms"][arm][sname]
    out = ["| source | days | MAE | RMSE | R² | exact category | within one | Δ MAE vs production (µg/m³) | Δ MAE % | 95% CI 7-d blocks | 95% CI 14-d blocks | Δ exact-cat (pts) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    p = a["pooled"]
    for s in a["sources"]:
        m = p["models"][f"{s}__{model}"]
        d = p["diff_vs_production"].get(s, {}).get(model)
        if d and not d.get("identical"):
            tail = f"{d['mae']:+.2f} | {d['mae_pct']:+.1f}% | {ci(d.get('mae_ci95_7d'))} | {ci(d.get('mae_ci95_14d'))} | {100 * d['cat_acc']:+.1f}"
        else:
            tail = "0 (reference)" if s == PROD else "0 (identical predictions)"
            tail += " | – | – | – | –"
        out.append(f"| {NAMES[s]} | {p['n_days']} | {f2(m['mae'])} | {f1(m['rmse'])} | {f2(m['r2'])} | {pct(m['cat_acc'])} | {pct(m['within_one_cat'])} | {tail} |")
    pm = p["models"][f"{PROD}__persistence"]
    out.append(f"| persistence (tomorrow = today) | {p['n_days']} | {f2(pm['mae'])} | {f1(pm['rmse'])} | {f2(pm['r2'])} | {pct(pm['cat_acc'])} | {pct(pm['within_one_cat'])} | – | – | – | – | – |")
    return "\n".join(out)


def per_winter(arm, sname, model="hybrid"):
    a = R["arms"][arm][sname]
    ws = list(a["winters"])
    out = ["| source | " + " | ".join(f"{x} MAE (n={a['winters'][x]['n_days']}) | Δ vs prod, 95% CI (7-d)" for x in ws) + " |", "|---|" + "---|---|" * len(ws)]
    for s in a["sources"]:
        cells = []
        for x in ws:
            m = a["winters"][x]["models"][f"{s}__{model}"]["mae"]
            d = a["winters"][x]["diff_vs_production"].get(s, {}).get(model)
            cells += [f2(m), (f"{d['mae']:+.2f} ({d['mae_pct']:+.1f}%) {ci(d.get('mae_ci95_7d'))}" if d and not d.get("identical") else "–")]
        out.append(f"| {NAMES[s]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def sudden(sname):
    a = R["arms"]["A"][sname]["subsets"]
    out = []
    for k, lab in (("sudden_change_days_ge30pct", "sudden-change days (|Δ| ≥ 30 % vs today)"), ("stable_days", "stable days")):
        v = a[k]
        out.append(f"\n**{lab}, n = {v['n_days']}**\n\n| source | Hybrid MAE | Δ vs production | 95% CI (7-d) |\n|---|---|---|---|")
        for s in R["arms"]["A"][sname]["sources"]:
            m = v["models"][f"{s}__hybrid"]["mae"]
            d = v["diff_vs_production"].get(s, {}).get("hybrid")
            out.append(f"| {NAMES[s]} | {f2(m)} | " + (f"{d['mae']:+.2f} | {ci(d.get('mae_ci95_7d'))}" if d and not d.get('identical') else "– | –") + " |")
    return "\n".join(out)


w("# Which weather forecast source gives the best next-day Delhi PM2.5 prediction?\n")
w("**RETROSPECTIVE experiment.** Winters 2024-25 and 2025-26 were examined in earlier experiments, so this is not an untouched test. "
  "Nothing here changes the production model, default version, dashboard, Lambda, checksums or the frozen winter 2026-27 protocol; nothing was committed, pushed or deployed. "
  "Machine-readable results: `results.json`. Declared protocol (written before any result): `config.json`. Rerun: `python experiments/weather_source_comparison/run_comparison.py` then `report.py`.\n")

# ---------------------------------------------------------------- answer
rv = R["reference_validity"]
rule = R["decision_rule"]
a6 = R["arms"]["A"]["P6_L1"]
d_ifs = a6["pooled"]["diff_vs_production"]["ecmwf_ifs025"]["hybrid"]
a4 = R["arms"]["A"]["P4_L1"]
d4_ifs = a4["pooled"]["diff_vs_production"]["ecmwf_ifs025"]["hybrid"]
d4_gfs = a4["pooled"]["diff_vs_production"]["gfs_global"]["hybrid"]
w("## 1. Answer in short\n")
w(f"* **No source reliably beats the current source. Retain it.** The live Lambda calls Open-Meteo without a `models` parameter (its default `best_match` blend). "
  f"Value-matching the archives shows that blend was **identical to DWD ICON in winter 2024-25 ({pct(rv['2024-25']['icon_global']['share_identical'])} of hours) and identical to ECMWF IFS HRES in winter 2025-26 "
  f"({pct(rv['2025-26']['ecmwf_ifs_hres']['share_identical'])} of hours)**. So 'the current production source' is IFS HRES, and only winter 2025-26 is a valid test against it.")
w(f"* **Winter 2025-26, matched dates (n = {a6['pooled']['n_days']}), current source vs the field (Hybrid MAE µg/m³, lower is better):** "
  + "; ".join(f"{NAMES[s]} {f2(a6['pooled']['models'][s + '__hybrid']['mae'])}" for s in a6["sources"])
  + f". Best point estimate: IFS 0.25° ({d_ifs['mae']:+.2f} µg/m³, {d_ifs['mae_pct']:+.1f}%), but its 95% interval {ci(d_ifs.get('mae_ci95_7d'))} (7-day blocks) and {ci(d_ifs.get('mae_ci95_14d'))} (14-day) includes zero and its exact-category accuracy is {100 * d_ifs['cat_acc']:+.1f} points lower. That is not evidence of an improvement.")
w(f"* **The large two-winter gaps are a reference artefact, not a source effect.** Pooling both winters shows IFS 0.25° {d4_ifs['mae_pct']:+.1f}% and GFS {d4_gfs['mae_pct']:+.1f}% versus 'production_default' with intervals excluding zero, but in 2024-25 'production_default' *was ICON*, "
  "which is the weakest source in that winter. These gaps say \"GFS and IFS 0.25° beat ICON in 2024-25\", not \"they beat what the Lambda uses today\". The pre-declared rule would have mechanically recommended IFS 0.25° on this pooled set; "
  "I did not apply it because its reference condition (one constant production source) is false, and I say so rather than hide it.")
w("* **Not manufactured:** the production-default row reproduces the earlier published V1/V2/Hybrid numbers exactly for 2025-26 and for the two-winter pooled set (Hybrid 37.21 µg/m³ pooled), so these are the same pipeline with only the weather input swapped.\n")

w("## 2. What the project actually does (Step 1, from the code)\n")
w("""| item | fact | where |
|---|---|---|
| live weather source | Open-Meteo Forecast API, **no `models` parameter** (default `best_match`) | `core/weather.py::fetch_forecast_weather`, `backend/forecast_handler.py` |
| coordinates | `LAT`/`LON` not set in `template.yaml` → defaults 28.6139, 77.2090 | handler defaults |
| variables | temperature_2m, relative_humidity_2m, wind_speed_10m, wind_direction_10m, precipitation, surface_pressure | `core/weather.py::HOURLY_VARS` |
| daily aggregation | tomorrow's 24 hourly values (Asia/Kolkata day): mean temp/RH/wind/pressure, summed precipitation, mean wind vector (u, v); an hour missing any variable is dropped; < 12 hours → no value | `hourly_to_daily_weather` (reused unchanged here) |
| issue time / horizon | `cron(30 17 * * ? *)` = 17:30 UTC = **23:00 IST**; target = next calendar day (1 day); PM2.5 inputs up to 22:59 IST | `template.yaml` |
| today's weather (wx0) | read from the same live response; no analysis archive exists → stays ERA5 for every source here | – |
| earlier historical evaluation | `weather_prevruns_2024_2026.json` (Previous Runs, lead 1 and 2, a fixed point 28.57, 77.18) used for the forecast-training experiment; training on ERA5 | `model/forecast_training_experiment.py` |
| archives available | Previous Runs API (`<var>_previous_dayN`); Historical Forecast API (analysis-like stitched first hours: **excluded** as input); Single Runs (non-HRES only from 2026-04-02: no winter coverage) | `fetch_sources.py`, `coverage.json` |
""")

w("## 3. Archive coverage (verified before downloading; `coverage.json`)\n")
w("| source (Open-Meteo id) | first complete lead-1 hour | complete lead-1 days | share of hours complete (2024-01-01 → 2026-02-16) | days in the 2024-25 test window | days in the 2025-26 test window |\n|---|---|---|---|---|---|")
for s in ["production_default", "gfs_global", "icon_global", "ecmwf_ifs025", "ecmwf_ifs_hres", "ecmwf_aifs025"]:
    a = R["archives"][s]
    oc = R["own_coverage_days"][s]["L1"]
    w(f"| {NAMES[s]} | {a['lead1']['first_complete_hour']} | {a['lead1']['complete_days']} | {pct(a['lead1']['share'])} | {oc['2024-25']} / 124 | {oc['2025-26']} / 104 |")
w("\nUnits (from the API responses): °C, %, km/h, degrees, mm, hPa. Timestamps are Asia/Kolkata hourly. Missing values are left missing (never filled); a day with fewer than 12 complete hours is dropped, exactly as in production.\n")
w("* **Only winter 2025-26 can test all six sources; 2024-25 can test four** (default, GFS, ICON, IFS 0.25°). IFS HRES lead-1 starts 2025-10-01; AIFS 2025-02-18. Because IFS HRES and AIFS cover one winter, 'consistent across winters' cannot be shown for them.")
w("* **Forecast lead**: `previous_day1` = value forecast about 24 h before the valid hour (primary, matches the live 23:00 IST → next-day setting) and `previous_day2` (48 h, strictly available). "
  "Open-Meteo does not publish the issue hour of each `previous_dayN` value, and Single Runs cannot verify it for these winters. The 12 UTC ECMWF run is published roughly 7–8 hours after initialisation, i.e. after the 17:30 UTC cutoff, so lead 1 may for some late hours of the target day lean on information a few hours newer than the Lambda would have. "
  "Lead 2 removes this doubt and is reported as a robustness arm; at lead 2 IFS 0.25° (−1.1 %) and AIFS (−1.7 %) are nominally ahead and nothing is separable from the current source (section 4).")
w("* **GFS data-quality flag:** GFS `relative_humidity_2m` averages 32 % in the archive against 64 % in the default blend and ERA5 (bias about −37 points on the test days), and its temperature is about +3.8 °C warm. Those weather errors are large (section 7) yet GFS still predicts PM2.5 well in 2024-25, a reminder that weather-forecast skill and PM2.5 skill are different things here.\n")

w("## 4. Primary result: winter 2025-26, current source (IFS HRES) as reference\n")
w("Arm A (**input swap**): the same frozen, ERA5-trained V1/V2/Hybrid models; only the next-day weather input changes with the source. Identical 104 dates, same PM2.5 truth, live-view features (hours ≤ 22 IST).\n")
w(table("A", "P6_L1"))
w("\nHybrid shown. V1 and V2 tables are in `results.json`. 'production default' and 'IFS HRES' predictions are bit-for-bit identical in this winter (the default blend resolves to HRES), which is itself a result: the live system is IFS HRES.\n")
w(sudden("P6_L1"))
w("\n### Lead 2 (strictly available forecasts), same dates\n")
w(table("A", "P6_L2"))
w("\n## 5. Four-source set over both winters (reference = whatever the default was)\n")
w("Valid as a comparison of GFS / ICON / IFS 0.25° with the **archived default**, but the default was ICON in 2024-25 and HRES in 2025-26, so this is **not** a test against the current source.\n")
w(table("A", "P4_L1"))
w("\nWinter by winter:\n")
w(per_winter("A", "P4_L1"))
w("\nIn 2024-25 the default (ICON) is the worst of the four, and GFS and IFS 0.25° are far better. In 2025-26 the picture reverses to 'no clear difference'. Winter consistency is therefore absent for GFS (+0.4 % in 2025-26) and weak for IFS 0.25° (−2.6 %, interval includes zero).\n")
w(sudden("P4_L1"))
w("\n## 6. Is it the source or the training coverage?\n")
w("Arm B retrains each source's models on that source's own archived forecasts where available (ERA5 before the archive starts: 837–1440 of ~1100–1450 origins are substituted, see `winters` in `results.json`), arm B_strict uses only origins with real forecasts (needs ≥ 365).\n")
for sn, lab in (("P6_L1", "winter 2025-26 (current source valid)"), ("P4_L1", "both winters")):
    w(f"\n**Hybrid MAE decomposition, {lab}** (µg/m³; total = source effect + training effect):\n\n| source | arm A (input swap) | arm B (source-trained) | source effect A_src − A_prod | training effect B_src − A_src | production's own B − A (control) | training effect net of control |\n|---|---|---|---|---|---|---|")
    A, B = R["arms"]["A"][sn], R["arms"]["B"][sn]
    for s, d in R["decomposition"][sn].items():
        w(f"| {NAMES[s]} | {f2(A['pooled']['models'][s + '__hybrid']['mae'])} | {f2(B['pooled']['models'][s + '__hybrid']['mae'])} | {d['source_effect_A_source_minus_A_production']:+.2f} | {d['training_effect_B_source_minus_A_source']:+.2f} | {d['control_production_own_B_minus_A']:+.2f} | {d['training_effect_net_of_control']:+.2f} |")
bs = R["arms"]["B_strict"]["P6_L1"]
w("\n**B_strict, winter 2025-26 (real forecasts only, ~576–617 training origins):** " + "; ".join(f"{NAMES[s]} {f2(bs['pooled']['models'][s + '__hybrid']['mae'])}" for s in bs["sources"]) +
  ". Every source loses a lot relative to arm A, including the production source (it has about 600 instead of 1450 training origins and only one winter of weather-forecast behaviour). "
  "IFS HRES and AIFS lack the ≥ 365 origins and are reported as insufficient coverage, not run. Differences between sources inside arm B_strict are therefore dominated by training size, not by the forecast source, and are not used for ranking.\n")
w("\n**Arm B (source-trained) vs the production source's own arm B, winter 2025-26, Hybrid:**\n")
w(table("B", "P6_L1"))
w("\nIn arm B, IFS 0.25° is the lowest in every seed and −5.5 % against the production source's own arm B; the 14-day-block interval just excludes zero while the 7-day one does not, so it fails the declared both-blocks criterion. It also mixes a source effect with the fact that the IFS 0.25° archive has 576 real-forecast training origins against 617 for the default and a different ERA5-substitution count, and it is one winter. It is the reason IFS 0.25° is named for forward logging, not evidence for a switch.\n")
w("\nSeed-to-seed noise for one fixed source and arm is about 0.5–0.9 µg/m³ in these tables (e.g. production arm A ranges 28.66–29.34 across the five seeds, see `seed_robustness`). ")
w("**Reading:** the input-swap arm is the clean source test, and in it nothing reliably beats the current source in 2025-26. Training on forecasts moves MAE by up to about 2 µg/m³, and moves it for the production source too; seed noise alone is 0.5–0.9 µg/m³. "
  "Arm B for the production source is not a clean control: the 617 real-forecast origins it trains on are ICON-era default forecasts (the default was ICON until Sep 2025), while it is tested on HRES, and its explicit-HRES counterpart has only 14 real origins before the winter. So arm B comparisons against production are tilted toward the other sources, which is why arm B is reported but not used for the decision. In 2025-26 the non-production sources gain from source-specific training by 0.0–1.9 µg/m³ (ICON gains most while its source effect is +1.46 worse), i.e. much of what arm B shows is a training effect. The two-winter 'improvement' of GFS and IFS 0.25° comes from the source effect against ICON in 2024-25 (the decomposition's source column), not from different training coverage.\n")
w("**Seed robustness (5 seeds, hybrid MAE, number of seeds in which the source beats production):**\n")
for key in ("A:P6_L1", "B:P6_L1", "A:P4_L1"):
    v = R["seed_robustness"][key]["seeds_where_source_beats_production"]
    w(f"* {key}: " + ", ".join(f"{NAMES[s]} {n}/5" for s, n in v.items()))
w("")
w("## 7. Weather-input error against ERA5 (independent of the PM2.5 model), winter 2025-26 dates\n")
w("| source | temp bias / RMSE (°C) | RH bias / RMSE (%) | wind bias / RMSE (km/h) | pressure bias / RMSE (hPa) | precip RMSE (mm) |\n|---|---|---|---|---|---|")
for s, d in R["weather_error_vs_era5"]["P6_L1"].items():
    w(f"| {NAMES[s]} | {d['temp']['bias']:+.2f} / {d['temp']['rmse']:.2f} | {d['rh']['bias']:+.1f} / {d['rh']['rmse']:.1f} | {d['wind']['bias']:+.2f} / {d['wind']['rmse']:.2f} | {d['pressure']['bias']:+.2f} / {d['pressure']['rmse']:.2f} | {d['precip']['rmse']:.2f} |")
w("\nERA5 is a reanalysis, not ground truth for Delhi, so these are mismatch figures, not forecast skill. IFS HRES has the smallest temperature, wind and RH errors; the PM2.5 model is trained on ERA5, so a source that looks more like ERA5 is not automatically a source that predicts PM2.5 better.\n")
w("## 8. Decision rule (declared in `config.json` before running)\n")
w("| set : source | Δ Hybrid MAE % | 95% CI 7-d | 95% CI 14-d | worst winter % | passes all five criteria | verdict |\n|---|---|---|---|---|---|---|")
for k, v in rule.items():
    w(f"| {k} | {v['hybrid_mae_pct_vs_production']:+.1f} | {ci(v['ci95_7d'])} | {ci(v['ci95_14d'])} | {v['worst_winter_pct']:+.1f} | {v['passes_all']} | {v['verdict']} |")
w("\n`passes all five criteria` is the mechanical result of the rule; the verdict column applies the reference-validity check. IFS 0.25° passes mechanically on the two-winter set only because the reference there is ICON in 2024-25.\n")

w("## 9. Answers to the five questions\n")
w(f"1. **Best-performing source on matched dates:** IFS 0.25° has the lowest Hybrid MAE at lead 1 in winter 2025-26 ({f2(a6['pooled']['models']['ecmwf_ifs025__hybrid']['mae'])} vs {f2(a6['pooled']['models'][PROD + '__hybrid']['mae'])} for the current source) and also on the four-source two-winter set, but only the latter gap is statistically separable, and that reference was ICON.")
w("2. **Meaningful and consistent across winters?** No. The advantage over the archived default is large in 2024-25 (ICON era) and small and indistinguishable from zero in 2025-26 (current source era). AIFS (−0.4 %), GFS (+0.4 %) and ICON (+5.1 %) show nothing against the current source. IFS HRES is the current source.")
w("3. **Usable coverage:** table in section 3. GFS, ICON and the default cover both winters; IFS 0.25° both winters (from 2024-02-29); IFS HRES lead-1 one winter (from 2025-10-01); AIFS one winter (from 2025-02-18). Single Runs and Historical Forecast cannot supply live-equivalent inputs for these winters.")
w("4. **Limitations:** two winters only (104 + 124 days, heavily autocorrelated; block bootstrap used); retrospective, both winters already seen; the default blend changed model between winters; lead-1 publication timing cannot be verified per hour (lead-2 arm included as a check); wx0 (today's weather) is ERA5 for every source, so the experiment only changes tomorrow's weather; GFS humidity looks mis-scaled in the archive; the forecast archive point (28.6139, 77.2090) is the live point, whereas the earlier experiments used 28.57, 77.18 (the default-source values matched the earlier archive's hybrid result, so the pipeline is comparable); training-set coverage differs by source in arm B; many comparisons were made, so a lone interval that excludes zero should be read cautiously.")
w("5. **Next source for the winter 2026-27 forward evaluation:** keep the current default as the scored source. **Log, do not score, three alternatives alongside it** on every issuance: the ECMWF IFS 0.25° forecast (best point estimate at lead 1 in 5/5 seeds; second to AIFS at lead 2), the explicit `ecmwf_ifs` (HRES) forecast, which pins today's de-facto source and detects a repeat of the ICON → HRES switch of Oct 2025, and AIFS (cheap, one winter of archive). "
  "Pinning the model explicitly instead of relying on `best_match` is worth considering, but is a production change and is not proposed until you ask. No production change follows from this experiment.\n")
w("## 10. Integrity\n")
w("* Training targets are always before 15 Oct of the tested winter; windows end 15 Feb; the untouched period is never read (asserted in code and tests).\n* No source-specific tuning; production hyper-parameters, seeds and eligibility rules reused.\n* Production code was imported read-only; protected-file hashes are checked in `integrity.json`.\n* No API key is used (public Open-Meteo endpoints); no `.env` file is read.\n")
(HERE / "report.md").write_text("\n".join(L))
print("report.md written", len("\n".join(L)))
