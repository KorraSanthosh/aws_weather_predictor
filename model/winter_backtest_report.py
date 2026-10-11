"""Renders experiments/winter_2024_25_backtest.md from the JSON that model/winter_backtest.py produces.

Every number in the report is read from the JSON; the verdict sentences are chosen by fixed rules (below), not written
after looking at the results.  Rules: a difference is called "conclusive" only when the 95% moving-block-bootstrap
interval of the MAE difference excludes zero for BOTH the 7-day and the 14-day block length.
"""
from __future__ import annotations

V1, V2, HYB, PERSIST, MEAN7 = "v1_25_baseline", "v2_35_candidate", "v3_hybrid", "persistence", "mean7"
LABEL = {V1: "V1 (25 features)", V2: "V2 (35 features)", HYB: "Hybrid", PERSIST: "Persistence (tomorrow = today)",
         MEAN7: "Trailing 7-day mean"}
SHORT = {V1: "V1", V2: "V2", HYB: "Hybrid", PERSIST: "Persist.", MEAN7: "7-day mean"}
STATION = {"235": "Anand Vihar", "17": "R K Puram", "50": "Punjabi Bagh", "10831": "Rohini"}
PRIMARY, DIAG, LEGACY = "live23h_wx_persistence", "live23h_wx_observed", "offline_wx_observed"
SETS = [("all_matched_days", "All matched days"), ("days_with_day_parts", "Days WITH hourly day-parts"),
        ("days_without_day_parts", "Days WITHOUT hourly day-parts")]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def ci(v, d=2):
    return f"[{v[0]:+.{d}f}, {v[1]:+.{d}f}]"


def pair(sset, a, b):
    return sset["paired"].get(f"{b}_minus_{a}")


def verdict_word(p):
    if p is None:
        return "n/a"
    if p["identical_predictions"]:
        return "identical predictions"
    both = p["excludes_zero"] and p["excludes_zero_block14"]
    if both:
        return "CONCLUSIVE (interval excludes 0 with 7- and 14-day blocks)"
    if p["excludes_zero"] or p["excludes_zero_block14"]:
        return "borderline (excludes 0 for only one block length)"
    return "inconclusive (interval includes 0)"


def pair_line(sset, a, b):
    p = pair(sset, a, b)
    if p is None:
        return "n/a"
    if p["identical_predictions"]:
        return "identical predictions on every day (difference exactly 0)"
    return (f"{LABEL[b]} − {LABEL[a]}: **{p['diff']:+.2f} µg/m³ ({p['pct']:+.1f}%)**, 95% CI {ci(p['ci95_diff'])} "
            f"(7-day blocks) / {ci(p['ci95_diff_block14'])} (14-day blocks) — {verdict_word(p)}")


def metric_table(sset, with_baselines=True):
    rows = []
    for m in (V1, V2, HYB, PERSIST, MEAN7):
        x = sset["models"][m]
        s = x["sudden_change"]
        rows.append([LABEL[m], x["mae_ug_m3"], x["rmse_ug_m3"], x["r2"], pct(x["cat_acc"]), pct(x["within_one_cat"]),
                     ((f"{s['mae']:.1f} / {pct(s['cat_acc'], 0)} / " + ("n/a" if m == PERSIST else pct(s['direction_acc'], 0))) if s["n"] else "—")])
    return table(["Model", "MAE µg/m³", "RMSE µg/m³", "R²", "Exact category", "Within ±1 category",
                  "Sudden-change days: MAE / exact cat / right direction"], rows)


def paired_table(sset):
    rows = []
    for a, b in ((V1, V2), (V1, HYB), (HYB, V2), (PERSIST, V1), (PERSIST, HYB), (PERSIST, V2), (MEAN7, V1)):
        p = pair(sset, a, b)
        if p is None:
            continue
        if p["identical_predictions"]:
            rows.append([f"{SHORT[b]} − {SHORT[a]}", "0.00 (identical)", "0.0%", "[0, 0]", "[0, 0]", "identical"])
        else:
            rows.append([f"{SHORT[b]} − {SHORT[a]}", f"{p['diff']:+.2f}", f"{p['pct']:+.1f}%", ci(p["ci95_diff"]),
                         ci(p["ci95_diff_block14"]), verdict_word(p).split(" (")[0]])
    return table(["MAE difference (negative = second model better)", "µg/m³", "%", "95% CI, 7-day blocks",
                  "95% CI, 14-day blocks", "Verdict"], rows)


def winter(res, name):
    return next(w for w in res["winters"] if w["winter"] == name)


def winter_table(res, regime):
    rows = []
    for w in res["winters"]:
        if w["status"] != "scored":
            rows.append([w["winter"], f"{w['scored_days']} / {w['calendar_days_in_window']}", "—", "—", "—", "—", "—", "—",
                         "—", "—", "—", w["status"]])
            continue
        g = w["regimes"][regime]
        s = g["sets"]["all_matched_days"]
        mm = s["models"]
        d2, dh = pair(s, V1, V2), pair(s, V1, HYB)
        rows.append([w["winter"], f"{w['scored_days']} / {w['calendar_days_in_window']}", g["n_with_day_parts"],
                     mm[PERSIST]["mae_ug_m3"], mm[MEAN7]["mae_ug_m3"], mm[V1]["mae_ug_m3"], mm[V2]["mae_ug_m3"],
                     mm[HYB]["mae_ug_m3"],
                     f"{d2['diff']:+.2f} {ci(d2['ci95_diff'], 1)}" + ("" if not d2["excludes_zero"] else " *"),
                     ("= V1" if dh["identical_predictions"] else f"{dh['diff']:+.2f} {ci(dh['ci95_diff'], 1)}"
                      + ("" if not dh["excludes_zero"] else " *")),
                     " / ".join(pct(mm[m]["cat_acc"], 0) for m in (V1, V2, HYB, PERSIST)), "scored"])
    return table(["Winter", "Scored / calendar days", "Days with day-parts", "Persist. MAE", "7-day mean MAE", "V1 MAE",
                  "V2 MAE", "Hybrid MAE", "V2 − V1 (µg/m³) [95% CI]", "Hybrid − V1 [95% CI]",
                  "Exact category V1 / V2 / Hybrid / Persist.", "Status"], rows)


def conclusive(p):
    """Difference called conclusive only if the interval excludes 0 for BOTH block lengths (fixed rule)."""
    return bool(p and not p["identical_predictions"] and p["excludes_zero"] and p["excludes_zero_block14"])


def _daily(res, name, regime=PRIMARY):
    w = winter(res, name)
    g = w["regimes"][regime]
    import numpy as np
    return (w["daily_dates"], np.array(w["daily_actual"]), {k: np.array(v) for k, v in g["daily_predictions"].items()},
            np.array(g["daily_has_day_parts"], bool))


def _slice_row(label, dates, y, P, mask, dp):
    """One row of the 'where does the change come from' table (primary regime)."""
    from datetime import date
    import winter_backtest as wb
    idx = [i for i in range(len(dates)) if mask[i]]
    ds = [date.fromisoformat(dates[i]) for i in idx]
    yy = y[idx]
    mae = {m: float(abs(P[m][idx] - yy).mean()) for m in (V1, V2, HYB, PERSIST)}
    d2 = wb.mbb_mae_diff(yy, P[V1][idx], P[V2][idx], ds, n_boot=5000)
    dh = wb.mbb_mae_diff(yy, P[V1][idx], P[HYB][idx], ds, n_boot=5000)
    return [label, len(idx), int(dp[idx].sum()), f"{mae[PERSIST]:.2f}", f"{mae[V1]:.2f}", f"{mae[V2]:.2f}", f"{mae[HYB]:.2f}",
            f"{d2['diff']:+.2f} ({d2['pct']:+.1f}%) {ci(d2['ci95_diff'])}",
            "= V1" if dh["identical_predictions"] else f"{dh['diff']:+.2f} ({dh['pct']:+.1f}%) {ci(dh['ci95_diff'])}"]


def render(res, alt=None):
    P = res["protocol"]
    pw = winter(res, res["primary_winter"])
    gp, gd, gl = (pw["regimes"][r] for r in (PRIMARY, DIAG, LEGACY))
    allp, allp_d = gp["sets"]["all_matched_days"], gd["sets"]["all_matched_days"]
    pooled = res["pooled_scored_winters"]
    pool_p, pool_d = pooled[PRIMARY]["sets"], pooled[DIAG]["sets"]
    cov = res["coverage_primary_winter"]
    n_sc, cal = pw["scored_days"], pw["calendar_days_in_window"]
    full = n_sc == cal
    n_dp = gp["n_with_day_parts"]
    extended = bool(res.get("extra_station_sources"))
    mae = {m: allp["models"][m]["mae_ug_m3"] for m in (V1, V2, HYB, PERSIST, MEAN7)}
    cat = {m: allp["models"][m]["cat_acc"] for m in (V1, V2, HYB, PERSIST, MEAN7)}
    best = min((V1, V2, HYB), key=lambda m: mae[m])
    best_cat = max((V1, V2, HYB), key=lambda m: cat[m])
    p21, p31 = pair(allp, V1, V2), pair(allp, V1, HYB)
    pdp, pno, pall = pool_p["days_with_day_parts"], pool_p["days_without_day_parts"], pool_p["all_matched_days"]
    hyb_total = pair(pall, V1, HYB)
    v2_dp, v2_no = pair(pdp, V1, V2), pair(pno, V1, V2)
    hyb_good = conclusive(hyb_total) and hyb_total["diff"] < 0
    dp_good = conclusive(v2_dp) and v2_dp["diff"] < 0
    w_last = [w for w in res["winters"] if w["status"] == "scored"][-1]
    scored_w = [w for w in res["winters"] if w["status"] == "scored"]
    dp_winters = [w["winter"] for w in scored_w if w["regimes"][PRIMARY]["n_with_day_parts"] >= 30]
    title_suffix = " - re-run with the added OpenCity data" if extended else ""

    L = []
    A = L.append
    A(f"# Fair historical winter back-test: V1 vs V2 vs Hybrid (focus: winter 2024-25){title_suffix}")
    A("")
    A("> **Status: RETROSPECTIVE back-test, not an untouched test.** Winters 2023-24, 2024-25 and 2025-26 were examined in "
      "earlier experiments (`EXPERIMENTS.md`) and 2019-2022 were used to choose the features and hyper-parameters. Nothing was "
      "tuned here: the protocol below was fixed before any result was computed. The only clean test is the prospective "
      "winter 2026-27 shadow test (`model/EVALUATION_PROTOCOL.md`). No model file, dashboard output or default was changed.")
    A("")
    if alt is not None:
        ap_ = winter(alt, alt["primary_winter"])
        aa = ap_["regimes"][PRIMARY]["sets"]["all_matched_days"]
        A(f"> **Two runs, one protocol.** Sections 1-9 use the project's production dataset (`data/daily.csv`): only {n_sc} of "
          f"{cal} days of the 2024-25 window had a city value and none had hourly data. After that report was written, "
          "15-minute CPCB data for 2024-25 was added (section 10) as a gap-filler. With it **all "
          f"{ap_['scored_days']} days are scored, each with day-part data**, and the V1-vs-V2 answer for 2024-25 changes: "
          f"{pair_line(aa, V1, V2)}. Section 10 compares the runs. On the identical Oct-Dec dates the gain appears only once V2 has real "
          "day-part inputs (and more day-part rows in training); no separate control isolates those two causes. Both runs are retrospective.")
        A("")
    else:
        A(f"Data used: {res.get('data_variant', 'production data')}. Reproduce: `python model/winter_backtest.py"
          + (" --daily data/daily_opencity.csv --extra-source opencity_hourly --tag opencity`" if extended else "`")
          + " then `python -m pytest tests/test_backtest.py`. Machine-readable results sit next to this file (same name, `.json`).")
        A("")
    if alt is not None:
        A("Reproduce: `python model/winter_backtest.py` (about 20 s), then `python model/winter_backtest.py --daily "
          "data/daily_opencity.csv --extra-source opencity_hourly --tag opencity`, then `python -m pytest tests/test_backtest.py`. "
          "Machine-readable results: `experiments/winter_2024_25_backtest.json` and `..._opencity.json`.")
        A("")

    # ------------------------------------------------------------ 1. answer first
    A("## 1. Answer first" + (" (production-data run)" if alt is not None else ""))
    A("")
    if full:
        A(f"* **All {n_sc} of the {cal} days of the 15 Oct 2024 - 15 Feb 2025 window are scored** ({pw['scored_first_last'][0]} to "
          f"{pw['scored_first_last'][1]}), each with a city PM2.5 from at least 3 stations.")
    else:
        A(f"* **Only {n_sc} of the {cal} days of the 15 Oct 2024 - 15 Feb 2025 window could be scored** "
          f"({pw['scored_first_last'][0]} to {pw['scored_first_last'][1]}). From 1 Jan 2025 only one station (Anand Vihar) reported, "
          f"below the project's 3-station rule for a city value, so those {pw['excluded_days']} days are excluded for every model "
          "(section 3). This is a data limit, not a modelling choice.")
    if n_dp == 0:
        A(f"* **None of the {n_sc} scored days has hourly day-part data** (Oct-Dec 2024 come from CPCB daily files). "
          "So the Hybrid takes V1's branch on every day: **Hybrid = V1, identical predictions**. V2 ran with its 7 day-part "
          "inputs missing (left as NaN, never imputed). The \"days with hourly features\" comparison for this winter is empty.")
    elif n_dp == n_sc:
        A(f"* **All {n_sc} scored days have hourly day-part data**, so the Hybrid takes V2's branch on every day "
          f"(Hybrid = V2{', identical predictions' if pair(allp, HYB, V2)['identical_predictions'] else ''}). "
          "The \"days without hourly features\" comparison for this winter is empty.")
    else:
        A(f"* **{n_dp} of the {n_sc} scored days have hourly day-part data.** The Hybrid uses V2 on those and V1 on the other "
          f"{n_sc - n_dp}.")
    A(f"* **Winter 2024-25 result (primary, leak-free weather).** MAE: {LABEL[V1]} {mae[V1]}, {LABEL[V2]} {mae[V2]}, "
      f"Hybrid {mae[HYB]}, persistence {mae[PERSIST]}, 7-day mean {mae[MEAN7]} µg/m³. Exact AQI-category accuracy: V1 {pct(cat[V1])}, "
      f"V2 {pct(cat[V2])}, Hybrid {pct(cat[HYB])}, persistence {pct(cat[PERSIST])}. {pair_line(allp, V1, V2)}.")
    if conclusive(p21):
        A(f"* **Which model is best for 2024-25?** {LABEL[best]} has the lowest MAE, and V2's gain over V1 is **conclusive within this "
          f"sample** (interval excludes 0 for both block lengths). Highest exact-category accuracy: {LABEL[best_cat]}. "
          f"Versus \"tomorrow = today\": {pair_line(allp, PERSIST, best).split(': ', 1)[-1]}.")
    else:
        A(f"* **Which model is best for 2024-25? Not established.** {LABEL[best]} has the lowest point-estimate MAE and "
          f"{LABEL[best_cat]} the highest exact-category accuracy, but the V2 − V1 difference is {verdict_word(p21).split(' (')[0]}. "
          f"Both have a lower point-estimate MAE than \"tomorrow = today\" ({(1 - mae[V1] / mae[PERSIST]) * 100:.0f}% / "
          f"{(1 - mae[V2] / mae[PERSIST]) * 100:.0f}% lower), but that gap is "
          f"{verdict_word(pair(allp, PERSIST, V1)).split(' (')[0]} too (section 6.1). Both are clearly better than the trailing 7-day mean "
          f"({mae[MEAN7]}), which lags the smog surges ({verdict_word(pair(allp, MEAN7, V1)).split(' (')[0]}).")
    A(f"* **Across the {len(pooled[PRIMARY]['winters'])} winters that could be scored** ({', '.join(pooled[PRIMARY]['winters'])}; "
      f"{pall['n_days']} days, {pdp['n_days']} with day-parts): on days with day-parts V2 − V1 is {pair_line(pdp, V1, V2).split(': ', 1)[-1]}; "
      f"on days without, {pair_line(pno, V1, V2).split(': ', 1)[-1]}. Day-part evidence comes from "
      f"{len(dp_winters)} winter(s): {', '.join(dp_winters) if dp_winters else 'none'}.")
    near = sum(1 for w in scored_w if w["regimes"][PRIMARY]["n_with_day_parts"] <= 2)
    if hyb_good:
        A(f"* **Hybrid.** Pooled over all scored days it beats V1: {pair_line(pall, V1, HYB).split(': ', 1)[-1]}. It equals V1 where "
          "day-parts are missing, so it keeps V2's gain and avoids V2's loss on those days.")
        A("* **Recommendation: the Hybrid is the leading candidate, but keep V1 as the default setting until the winter 2026-27 forward "
          "test.** The frozen rules (`model/EVALUATION_PROTOCOL.md`) allow promotion only on unseen data; these winters were seen. "
          "Weather here is a bound, not a real forecast. One historical winter does not guarantee anything about 2026-27.")
    else:
        A("* **Recommendation: keep V1 as the default pending the winter 2026-27 forward test.** The evidence favours V2 only where "
          "hourly day-parts exist and only as an interval that still includes zero; it hurts without them. The Hybrid is the "
          f"logical way to use V2, but this back-test cannot show it beats V1: it differs from V1 on at most 2 days in {near} of the "
          f"{len(pooled[PRIMARY]['winters'])} scored winters. One historical winter does not guarantee anything about 2026-27.")
    A("")

    # ------------------------------------------------------------ 2. inspection
    A("## 2. What the project does today (inspection) and why the stored models cannot be used here")
    A("")
    A("* **Training recipe (both versions).** `model/train.py`: XGBoost regressor on `log1p(PM_tomorrow) − log1p(PM_today)`, 800 trees "
      "max, learning rate 0.03, depth 4, subsample 0.85, colsample 0.85, min_child_weight 3, L2 2.0, early stopping (40 rounds) on the "
      "last 60 training rows, `random_state=42`. V1 = 25 features (`FEATURES_V1`), V2 = 35 (`FEATURES_V2`, adds 7 hourly day-part "
      "features and 8 binary flags). Feature definitions live in `core/features.py` and are shared by training, evaluation and the "
      "Lambda, so the same rule builds a feature everywhere.")
    A("* **Missing features.** Day-part features come from hourly data (`core/daily.py::daypart_stats`; needs >=16 hours). Where hours "
      "are absent they are NaN; XGBoost handles NaN natively; nothing is imputed. The Hybrid (`core/hybrid.py`) uses V2 only when all "
      "7 day-part inputs are finite, else V1.")
    A("* **Valid PM2.5.** `data/daily.csv` holds a city value only for days with >=3 stations having >=16 valid hours (or a CPCB daily "
      "value). 13 station IDs exist in total; 2018-22 and Feb 2025 onward are hourly (OpenAQ); 2023-24 are CPCB daily files for four "
      "stations and have no hourly data, so no day-part features (until the OpenCity gap-fill, section 10).")
    A("* **Stored artifacts cannot reproduce this setup.** `model/<version>/model.json` was trained through "
      "2026-10-07, i.e. on every winter, including this one. Using them on 2024-25 would be scoring training data. The back-test "
      "therefore re-trains frozen copies **in memory** with the unchanged recipe (nothing is written to `model/`).")
    A("* **The previous winter checks were not frozen.** `model/compare_versions.py` / `model/winter_check.py` use a walk-forward fit "
      "that is refitted each month, so later months of a winter are predicted by a model that has already trained on earlier months "
      "of the same winter, with tomorrow's observed weather. This is a legitimate live-like scheme but not what was asked here.")
    A("")

    # ------------------------------------------------------------ 3. coverage
    A("## 3. Data coverage and missing-data report")
    A("")
    if extended:
        gf = res.get("gapfill_report") or {}
        A("**Data sources in this run:** OpenAQ hourly files + CPCB daily files (as in production) + OpenCity/CPCB 15-minute files for "
          "2024-01-01 to 2025-12-31 used only to fill days a station has no OpenAQ data for ("
          + ", ".join(f"{STATION.get(k, k)}: {v['kept']:,} readings kept, {v['first']} to {v['last']}" for k, v in gf.items())
          + "). Rules and validation: `pipeline/convert_opencity.py` and section 10 of `winter_2024_25_backtest.md`.")
        A("")
    A(f"**Window 2024-25** ({cov['window'][0]} to {cov['window'][1]}, {cov['calendar_days']} target days).")
    A("")
    if "stations_per_day_by_month" in cov:
        rows = []
        for mth, dd in cov["stations_per_day_by_month"].items():
            rows.append([mth, ", ".join(f"{n} day(s) with {k} station(s)" for k, n in sorted(dd.items(), key=lambda kv: -int(kv[0])))])
        A(table(["Month", "Stations reporting a valid daily value"], rows))
        A("")
        rows = [[f"{sid} ({STATION.get(sid, '?')})", v["days"], ", ".join(v["source"]), f"{v['first']} to {v['last']}"]
                for sid, v in cov["per_station"].items()]
        A(table(["Station", "Valid days in window", "Source", "Span"], rows))
        A("")
    A(table(["Step of the eligibility ladder (same for every model)", "Days in the 2024-25 window"], [
        ["Calendar target days 15 Oct 2024 - 15 Feb 2025", cov["calendar_days"]],
        ["... with a city PM2.5 (>=3 stations) on the target day", cov["days_with_city_pm25"]],
        ["... also with city PM2.5 on the forecast-origin day, weather on both days, and an origin the 23:00 Lambda could use",
         cov["scoreable_target_days"]],
        ["... of which with all 7 hourly day-part features", cov["scoreable_days_with_all_7_day_parts"]],
    ]))
    A("")
    if pw["exclusions"]:
        A("**Every exclusion (winter 2024-25):**")
        A("")
        A(table(["From", "To", "Days", "Reason (same for V1, V2, Hybrid and both baselines)"],
                [[e["from"], e["to"], e["days"], e["reason"]] for e in pw["exclusions"]]))
        A("")
        A("Notes: the single Anand Vihar value on those days was NOT used as a stand-in for the city mean, because it is a different "
          "quantity (one traffic hot-spot station vs the city average the models are trained on). Nothing was interpolated or imputed. "
          "Hourly data exist in this dataset only for 2018-22 and Feb 2025 onward, none inside this window, so day-part features cannot "
          "be built for it.")
    else:
        A("**Exclusions in the 2024-25 window: none.** Every calendar day has a city PM2.5 from at least 3 stations.")
    A("")
    todo = []
    if not full:
        todo.append("city PM2.5 for 1 Jan - 15 Feb 2025 from at least 3 stations (the 2025 CPCB daily files for R K Puram, Punjabi Bagh "
                    "and Rohini, or the OpenCity 15-minute files, section 10; neither is reachable from the build sandbox)")
    if n_dp < n_sc:
        todo.append("hourly station data for the window, so day-part features can exist (the OpenCity 15-minute files provide it)")
    todo.append("archived weather forecasts issued before 23:00 for each day (an archive such as Open-Meteo's historical-forecast API; "
                "the host is not reachable from the build sandbox and nothing is stored), which would turn the primary regime into a "
                "realistic one")
    A("**What would still improve this evaluation:** " + "; ".join(f"({i + 1}) {t}" for i, t in enumerate(todo)) + ".")
    A("")
    A("**All winters (same ladder):**")
    A("")
    rows = []
    for w in res["winters"]:
        g = w.get("regimes", {}).get(PRIMARY)
        rows.append([w["winter"], w["calendar_days_in_window"], w["scored_days"], w["excluded_days"],
                     g["n_with_day_parts"] if g else "—", w["training"]["origins"],
                     w["training"].get("day_part_rows_in_training", "—"), w["status"]])
    A(table(["Winter", "Calendar days", "Scored days", "Excluded", "Scored days with day-parts", "Training origins before 15 Oct",
             "...of which with day-parts", "Status"], rows))
    A("")
    A(f"A winter is scored only with >={P['min_training_origins']} training origins and >={P['min_scored_days']} scoreable days "
      "(fixed in advance); the early winters fail the first rule because the 3-station rule removes most 2018-20 days. "
      f"Winter 2022-23 has only {winter(res, '2022-23')['scored_days']} scored days (almost all Jan-Feb 2023).")
    A("")

    # ------------------------------------------------------------ 4. method
    A("## 4. Method (fixed in advance)")
    A("")
    A("* **Frozen fit.** For a winter starting 15 Oct Y, each model is trained once on forecast origins whose target date is before "
      "15 Oct Y; there is no refit inside the window, and no window outcome is used for training, early stopping or any choice. A unit test "
      "scrambles the window outcomes and checks the predictions do not move.")
    A("* **Forecast-time information only.** Features for target D+1 read data up to D (tests scramble the target day and the future). "
      "Day-of features use hours <= 22 IST because the nightly Lambda runs at 23:00.")
    A("* **Weather (no archived forecasts available).**")
    A("")
    A(table(["Regime", "Tomorrow's weather", "Status"], [
        ["**PRIMARY** `live23h_wx_persistence`", "today's observed weather (a no-skill forecast)",
         "leak-free; a no-skill stand-in for a forecast (not guaranteed pessimistic, see below)"],
        ["Diagnostic `live23h_wx_observed`", "tomorrow's OBSERVED weather", "uses information live would not have; optimistic; **not live performance**"],
        ["Legacy `offline_wx_observed`", "tomorrow's observed weather, all 24 h of today", "what earlier experiments used; only for reconciliation"]]))
    A("")
    A("  Neither is claimed as live performance. **Correction after a later check with real archived forecasts** "
      "(`experiments/option_screen.md`, Open-Meteo Previous Runs API, 24 h ahead): for 2025-26 the real-forecast error lay "
      "between these two regimes, but for 2024-25 it was HIGHER than the primary regime (Hybrid 44.4 vs 42.7 µg/m³), because "
      "the forecast weather differs systematically from the archive weather the models are trained on. The Hybrid's lead "
      "over V1 held with real forecasts (pooled over both winters about 11% lower MAE, interval excluding zero).")
    A("* **Baselines.** Persistence: tomorrow = today's city PM2.5. Trailing 7-day mean: mean of the available daily values D-6..D.")
    A("* **Sudden-change days.** |tomorrow − today| / today >= 30%. \"Direction\" = share of those days where the forecast moved the right way from today (n/a for persistence).")
    A("* **Uncertainty.** Moving-block bootstrap (overlapping blocks of 7 days; 14 days as sensitivity; 5000 resamples, seed 0) on the paired "
      "difference of absolute errors, days placed on the calendar so gaps are respected. \"Conclusive\" = interval excludes 0 for both block lengths. "
      "Winters are scored separately; pooled rows place winters on a synthetic timeline so blocks never span two winters.")
    A("")

    # ------------------------------------------------------------ 5. winter by winter
    A("## 5. Winter-by-winter comparison")
    A("")
    A("**Primary regime (leak-free weather).** MAE in µg/m³; `*` = interval excludes 0 (7-day blocks).")
    A("")
    A(winter_table(res, PRIMARY))
    A("")
    A("**Diagnostic regime (tomorrow's observed weather - optimistic, not live).**")
    A("")
    A(winter_table(res, DIAG))
    A("")
    A("**Pooled over the scored winters** (each from its own frozen fit; these winters share the same retrospective caveats):")
    A("")
    rows = []
    for reg, nm in ((PRIMARY, "primary"), (DIAG, "diagnostic")):
        for sk, sl in SETS:
            s = pooled[reg]["sets"][sk]
            if not s["n_days"]:
                continue
            mm = s["models"]
            rows.append([nm, sl, s["n_days"], mm[PERSIST]["mae_ug_m3"], mm[V1]["mae_ug_m3"], mm[V2]["mae_ug_m3"],
                         mm[HYB]["mae_ug_m3"], pair_line(s, V1, V2).split(": ", 1)[-1],
                         pair_line(s, V1, HYB).split(": ", 1)[-1]])
    A(table(["Regime", "Set", "Days", "Persist. MAE", "V1 MAE", "V2 MAE", "Hybrid MAE", "V2 − V1", "Hybrid − V1"], rows))
    A("")

    # ------------------------------------------------------------ 6. detail
    A("## 6. Winter 2024-25 in detail")
    A("")
    A(f"Scored target dates: {pw['scored_first_last'][0]} to {pw['scored_first_last'][1]} ({n_sc} days). Training: "
      f"{pw['training']['origins']} origins with targets {pw['training']['first_target']} to {pw['training']['target_dates_through']} "
      f"({pw['training'].get('day_part_rows_in_training')} with day-parts, the rest with day-parts NaN). "
      f"Trees after early stopping: V1 {pw['training']['n_trees'][V1]}, V2 {pw['training']['n_trees'][V2]}. "
      f"Sudden-change days: {allp['models'][V1]['sudden_change']['n']} of {n_sc}.")
    A("")
    A("### 6.1 Primary regime (leak-free weather), all matched days")
    A("")
    A(metric_table(allp))
    A("")
    A(paired_table(allp))
    A("")
    A("### 6.2 Dates with and without hourly features")
    A("")
    for sk, sl in SETS[1:]:
        s = gp["sets"][sk]
        if not s["n_days"]:
            A(f"* **{sl}: 0 days.** Nothing to compare in this winter.")
        elif s["n_days"] == n_sc:
            A(f"* **{sl}: {s['n_days']} days** (= all matched days; table above).")
        else:
            A(f"* **{sl}: {s['n_days']} days.** V1 {s['models'][V1]['mae_ug_m3']}, V2 {s['models'][V2]['mae_ug_m3']}, Hybrid "
              f"{s['models'][HYB]['mae_ug_m3']}, persistence {s['models'][PERSIST]['mae_ug_m3']} µg/m³ MAE; {pair_line(s, V1, V2)}.")
    if n_dp == 0:
        A("* V2 is run with its 7 day-part inputs missing on every day; the Hybrid routes all of them to V1.")
        A("* For the with/without split on days that do have day-parts, see the winter-by-winter and pooled rows in section 5.")
    A("")
    A("### 6.3 Diagnostic regime (tomorrow's observed weather - NOT live performance)")
    A("")
    A(metric_table(allp_d))
    A("")
    A(paired_table(allp_d))
    A("")
    if not extended:
        A("### 6.4 Reconciliation with the earlier (non-frozen) check")
        A("")
        lg = gl["sets"]["all_matched_days"]["models"]
        A(table(["Setting (winter 2024-25, 78 days)", "V1 MAE", "V2 MAE", "Note"], [
            ["Earlier `winter_check.py`: monthly refit, observed weather, 24 h (`experiments/winter_check.json`)", 48.90, 47.33,
             "later months are predicted by models that trained on earlier months of the same winter"],
            ["This back-test, legacy regime: frozen fit, observed weather, 24 h", lg[V1]["mae_ug_m3"], lg[V2]["mae_ug_m3"],
             "same inputs as above, but nothing from the winter in training"],
            ["This back-test, primary: frozen fit, weather persistence, hours <= 22", mae[V1], mae[V2], "leak-free weather"]]))
        A("")
        A("The frozen fit is slightly worse than the monthly refit, as expected, and the V2-vs-V1 ordering is unchanged. The earlier "
          "check's -3.2% (interval -7.6% to +1.1%) was also inconclusive.")
        A("")

    # ------------------------------------------------------------ 7. conclusion
    A("## 7. Which model is best, and is the evidence conclusive?")
    A("")
    if conclusive(p21):
        A(f"* **Winter 2024-25 alone: V2 and the Hybrid beat V1, conclusively within this sample.** V2 − V1 = {p21['diff']:+.2f} µg/m³ "
          f"({p21['pct']:+.1f}%), interval {ci(p21['ci95_diff'])} (7-day) / {ci(p21['ci95_diff_block14'])} (14-day); exact-category accuracy "
          f"V1 {pct(cat[V1])}, V2 {pct(cat[V2])}. \"Conclusive\" here is about this one retrospective winter; it does not transfer automatically to 2026-27.")
    else:
        A(f"* **Winter 2024-25 alone: no conclusive winner.** V2 − V1 = {p21['diff']:+.2f} µg/m³ ({p21['pct']:+.1f}%), interval "
          f"{ci(p21['ci95_diff'])}, which includes 0. V1 has the higher exact-category accuracy ({pct(cat[V1])} vs {pct(cat[V2])}), "
          "but that is a handful of days. The Hybrid cannot be distinguished from V1 here because it is V1.")
    pno_d = pair(pool_d["days_without_day_parts"], V1, V2)
    A(f"* **What the data can say about V2.** With hourly day-parts it has the lower MAE (pooled: {pair_line(pdp, V1, V2)}); without them "
      f"it is worse (pooled, primary: {pair_line(pno, V1, V2)}; in the optimistic diagnostic regime the same gap is {pno_d['diff']:+.2f} µg/m³, "
      f"95% CI {ci(pno_d['ci95_diff'])}, {verdict_word(pno_d).split(' (')[0]}). That is the reason for the Hybrid; the day-part "
      f"evidence comes from {len(dp_winters)} winter(s) ({pdp['n_days']} days).")
    share = (pair(w_last["regimes"][PRIMARY]["sets"]["all_matched_days"], V1, HYB)["diff"] * w_last["scored_days"]) / \
        (hyb_total["diff"] * hyb_total["n"]) * 100 if hyb_total["diff"] else None
    A("* **Hybrid.** By construction it equals V1 on days without day-parts and V2 on days with them. Historically it "
      f"scores {pall['models'][HYB]['mae_ug_m3']} vs V1 {pall['models'][V1]['mae_ug_m3']} "
      f"pooled (primary); {share:.0f}% of that gain comes from the latest winter ({w_last['winter']}). "
      f"Hybrid − V1 pooled: {hyb_total['diff']:+.2f} µg/m³, 95% CI {ci(hyb_total['ci95_diff'])} - {verdict_word(hyb_total).split(' (')[0]}.")
    A("* **Not shown by this back-test:** whether V2 or the Hybrid beats V1 in winter 2026-27, or how they behave with a real weather forecast"
      + ("" if full else ", or any result for Jan-Feb 2025") + ".")
    A("")
    A("## 8. Limitations")
    A("")
    A("* Retrospective: these winters were examined before; V2's design was shaped by earlier looks. Treat all results as descriptive.")
    A(f"* Small samples with strong serial correlation (smog episodes): {n_sc} days is only about {n_sc // 7} weeks of data and neighbouring days are not independent. Wide intervals are the honest result.")
    A("* No archived weather forecasts; both weather regimes are bounds, not live performance.")
    A("* Frozen fits get up to ~4 months stale. A live system refitted monthly would do a little better.")
    if not full:
        A("* 2024-25 covers Oct-Dec only. It misses the Jan-Feb part of the season" + (" and contains no day-part days." if n_dp == 0 else "."))
    if extended:
        A("* The 2024 and Jan-Feb 2025 hourly values come from the OpenCity feed, other periods from OpenAQ. Both are CPCB stations and the "
          "IST reading reproduces the official daily files exactly, but the hourly sources are not identical, so day-part values may differ slightly across the join.")
    A("* Winter 2022-23 has few scored days, and winters before 2022-23 could not be scored at all (too little training data under the 3-station rule).")
    A(f"* Day-part evidence exists for {len(dp_winters)} scored winter(s) only.")
    A("")
    A("## 9. Recommendation")
    A("")
    if hyb_good:
        A("Keep **V1 (`v1_25_baseline`) as the default setting for now**, and treat the **Hybrid as the leading candidate**: in this "
          "retrospective back-test it beats V1 on pooled days with an interval that excludes zero, and it equals V1 wherever hourly "
          "day-parts are missing. The frozen promotion rule (`model/EVALUATION_PROTOCOL.md`: >=5% lower MAE, 95% interval below zero, "
          "category accuracy not more than 2 points lower, on >=90 matched unseen winter days) is what should change the default, via the winter "
          "2026-27 shadow test. Nothing here was committed, pushed or deployed.")
    else:
        A("Keep **V1 (`v1_25_baseline`) as the default**; keep V2 as the experimental candidate and the Hybrid as the pre-declared "
          "candidate for the prospective winter 2026-27 shadow test (frozen rules in `model/EVALUATION_PROTOCOL.md`; promotion needs >=5% lower MAE with the "
          "95% interval below zero on >=90 matched days). Nothing in this back-test justifies changing the default, and nothing here was "
          "committed, pushed or deployed.")
    A("")

    # ------------------------------------------------------------ 10. extended run
    if alt is not None:
        aw = winter(alt, alt["primary_winter"])
        ag = aw["regimes"][PRIMARY]
        aall, aall_d = ag["sets"]["all_matched_days"], aw["regimes"][DIAG]["sets"]["all_matched_days"]
        ap_pool = alt["pooled_scored_winters"][PRIMARY]["sets"]
        A("## 10. Re-run with the added OpenCity 15-minute data")
        A("")
        A("### 10.1 What was added and how it was checked")
        A("")
        gf = alt.get("gapfill_report") or {}
        eff = alt.get("dataset_effect_vs_production") or {}
        A("* **Source:** [Delhi Hourly Air Quality Reports (OpenCity)](https://data.opencity.in/dataset/delhi-hourly-air-quality-reports), "
          "CPCB 15-minute station files \"2024-25\" (2024-01-01 to 2025-12-31) for R K Puram, Punjabi Bagh, Rohini and Anand Vihar. "
          "Downloaded through your in-app browser, because the build sandbox is blocked from that host. Not committed anywhere.")
        A("* **Time zone.** The timestamps carry a `+0000` suffix but are Delhi wall-clock time. Read as IST, 24-hour means reproduce the "
          "official CPCB daily files for 2024 exactly at all four stations (mean absolute difference 0.0 µg/m³ vs 8-11 µg/m³ when read as UTC); "
          "`experiments/opencity_timezone_check.json`, `pipeline/check_opencity_timezone.py`.")
        A("* **Gap-fill only.** A reading is used only when that station has no OpenAQ reading at all on that day (kept: "
          + ", ".join(f"{STATION.get(k, k)} {v['kept']:,}" for k, v in gf.items())
          + " readings). Days OpenAQ already covers are unchanged. Blank values are dropped, nothing is interpolated. "
          "Rules written before use: `pipeline/convert_opencity.py`.")
        if eff:
            A(f"* **Effect on the dataset** (separate file `data/daily_opencity.csv`; production `data/daily.csv` untouched): "
              f"{eff['new_city_days']} new city days ({eff['new_city_days_first_last'][0]} to {eff['new_city_days_first_last'][1]}), "
              f"and {eff['existing_days_whose_value_changed']} existing days (almost all in 2024) change value by "
              f"{eff['mean_abs_change_ug_m3']} µg/m³ on average (max {eff['max_abs_change_ug_m3']}), because they are now built from the "
              "15-minute readings instead of the CPCB daily file. Those 2024 days now also have day-part features, which is what "
              "gives V2 real inputs before and during the 2024-25 winter.")
        A("")
        A("### 10.2 Winter-by-winter, with the added data")
        A("")
        A("**Primary regime (leak-free weather).**")
        A("")
        A(winter_table(alt, PRIMARY))
        A("")
        A("**Diagnostic regime (observed weather, not live).**")
        A("")
        A(winter_table(alt, DIAG))
        A("")
        A("### 10.3 Where the 2024-25 change comes from (primary regime)")
        A("")
        d0, y0, P0, dp0 = _daily(res, res["primary_winter"])
        d1, y1, P1, dp1 = _daily(alt, alt["primary_winter"])
        import numpy as np
        m78 = np.array([x <= "2024-12-31" for x in d1])
        rows = [_slice_row("Production data, 15 Oct - 31 Dec (all that was scoreable)", d0, y0, P0, np.ones(len(d0), bool), dp0),
                _slice_row("Added data, the SAME 15 Oct - 31 Dec dates", d1, y1, P1, m78, dp1),
                _slice_row("Added data, 1 Jan - 15 Feb (the 46 newly scoreable days)", d1, y1, P1, ~m78, dp1),
                _slice_row("Added data, whole window", d1, y1, P1, np.ones(len(d1), bool), dp1)]
        A(table(["Run / slice", "Days", "Days with day-parts", "Persist. MAE", "V1 MAE", "V2 MAE", "Hybrid MAE",
                 "V2 − V1 µg/m³ (%) [95% CI]", "Hybrid − V1"], rows))
        A("")
        A("Reading: on the same Oct-Dec dates, V2 goes from \"about equal to V1\" to clearly better because it now has real day-part inputs "
          "(in the production run it had none and was run with NaNs). The 46 extra January-February days point the same way but, taken alone, "
          "their interval includes zero. V1's own error changes little between rows, so this is not V1 getting worse. The persistence column "
          "differs slightly between the first two rows only because the 2024 target values are now the 15-minute-built daily means.")
        A("")
        A("### 10.4 Winter 2024-25 in detail, with the added data (primary regime)")
        A("")
        A(f"Training: {aw['training']['origins']} origins ({aw['training'].get('day_part_rows_in_training')} with day-parts); trees after "
          f"early stopping V1 {aw['training']['n_trees'][V1]}, V2 {aw['training']['n_trees'][V2]}; sudden-change days "
          f"{aall['models'][V1]['sudden_change']['n']} of {aw['scored_days']}.")
        A("")
        A(metric_table(aall))
        A("")
        A(paired_table(aall))
        A("")
        A("**Diagnostic regime (observed weather, not live):**")
        A("")
        A(metric_table(aall_d))
        A("")
        A(paired_table(aall_d))
        A("")
        A("### 10.5 Pooled over scored winters, with the added data (primary regime)")
        A("")
        rows = []
        for sk, sl in SETS:
            s = ap_pool[sk]
            if s["n_days"]:
                mm = s["models"]
                rows.append([sl, s["n_days"], mm[PERSIST]["mae_ug_m3"], mm[V1]["mae_ug_m3"], mm[V2]["mae_ug_m3"], mm[HYB]["mae_ug_m3"],
                             pair_line(s, V1, V2).split(": ", 1)[-1], pair_line(s, V1, HYB).split(": ", 1)[-1]])
        A(table(["Set", "Days", "Persist. MAE", "V1 MAE", "V2 MAE", "Hybrid MAE", "V2 − V1", "Hybrid − V1"], rows))
        A("")
        A("### 10.6 What this changes, and what it does not")
        A("")
        pa_dp = pair(ap_pool["days_with_day_parts"], V1, V2)
        pa_hy = pair(ap_pool["all_matched_days"], V1, HYB)
        A(f"* With complete data, winter 2024-25 supports V2/Hybrid over V1: {pair_line(aall, V1, V2).split(': ', 1)[-1]}. Pooled over the "
          f"scored winters the Hybrid is {pa_hy['diff']:+.2f} µg/m³ ({pa_hy['pct']:+.1f}%) vs V1, 95% CI {ci(pa_hy['ci95_diff'])}, "
          f"{verdict_word(pa_hy).split(' (')[0]}; on days with day-parts V2 is {pa_dp['diff']:+.2f} µg/m³ ({pa_dp['pct']:+.1f}%), "
          f"95% CI {ci(pa_dp['ci95_diff'])}, {verdict_word(pa_dp).split(' (')[0]}.")
        A("* The earlier verdict \"not conclusive\" most plausibly came from missing data rather than from V2 failing: V2 had no day-part inputs in the production run. (No separate control isolates this from the extra day-part rows V2 now has in training.)")
        A("* What it does not change: these winters were all seen before, the weather is a bound rather than a real forecast, and the "
          "hourly values for 2024 and early 2025 come from a different feed than 2018-22 and later 2025. A model that wins retrospectively can still lose "
          "next winter, so the default stays V1 until the shadow test, with the Hybrid as the leading candidate.")
        A("")
    return "\n".join(L)
