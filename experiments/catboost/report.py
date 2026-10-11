"""Render experiments/catboost/report.md from results.json + tuning.json (every number comes from those files)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "model"))
from winter_backtest_report import table  # noqa: E402

NICE = {"xgb": "XGBoost", "cb_base": "CatBoost (baseline)", "cb_tuned": "CatBoost (tuned)", "persistence": "Tomorrow = today",
        "mean7": "7-day mean"}
VNICE = {"v1": "V1 (25 feat.)", "v2": "V2 (35 feat.)", "hybrid": "Hybrid"}
ORDER = ["persistence", "mean7"] + [f"{f}_{v}" for v in ("v1", "v2", "hybrid") for f in ("xgb", "cb_base", "cb_tuned")]


def nm(k):
    if k in NICE:
        return NICE[k]
    for f in ("cb_base", "cb_tuned", "xgb"):
        if k.startswith(f + "_"):
            return f"{NICE[f]} {VNICE[k[len(f) + 1:]]}"
    return k


def ci(c, nd=2, mult=1.0):
    return "n/a" if not c else f"[{c[0] * mult:.{nd}f}, {c[1] * mult:.{nd}f}]"


def sg(x, nd=2, mult=1.0):
    return f"{x * mult:+.{nd}f}"


def verdict(d):
    if d.get("identical"):
        return "identical"
    return "conclusive" if d.get("excludes_zero_both") else "inconclusive"


def model_rows(b, keys=ORDER):
    rows = []
    for k in keys:
        m = b["models"].get(k)
        if not m:
            continue
        rows.append([nm(k), f"{m['mae']:.2f} {ci(m.get('mae_ci95'))}", f"{m['rmse']:.2f} {ci(m.get('rmse_ci95'))}", f"{m['r2']:.3f} {ci(m.get('r2_ci95'), 3)}",
                     f"{100 * m['cat_acc']:.1f} % {ci(m.get('cat_acc_ci95'), 1, 100)}", f"{100 * m['within_one_cat']:.1f} %",
                     (f"{m['vs_persistence_pct']:+.1f} % {ci(m.get('vs_persistence_pct_ci95'), 1)}" if "vs_persistence_pct" in m else "-")])
    return rows


MODEL_HDR = ["Model", "MAE µg/m³ [95 %]", "RMSE µg/m³ [95 %]", "R² [95 %]", "Exact category [95 %]", "Within one category", "Better than persistence [95 %]"]


def diff_rows(b, fams=("cb_base", "cb_tuned")):
    rows = []
    for f in fams:
        for v in ("v1", "v2", "hybrid"):
            d = b["comparisons"].get(f"{f}_vs_xgb_{v}")
            if not d:
                continue
            rows.append([nm(f"{f}_{v}") + " − XGBoost", f"{sg(d['mae'])} ({sg(d['mae_pct'], 1)} %)", ci(d.get("mae_ci95_7d")), ci(d.get("mae_ci95_14d")),
                         f"{sg(d['rmse'])} {ci(d.get('rmse_ci95_7d'))}", f"{sg(d['r2'], 3)} {ci(d.get('r2_ci95_7d'), 3)}",
                         f"{sg(d['cat_acc'], 1, 100)} pts {ci(d.get('cat_acc_ci95_7d'), 1, 100)}", f"{sg(d['within_one_cat'], 1, 100)} pts", verdict(d)])
    return rows


DIFF_HDR = ["Comparison", "ΔMAE", "95 % (7-d)", "95 % (14-d)", "ΔRMSE [95 %]", "ΔR² [95 %]", "Δ exact category [95 %]", "Δ within-one", "Verdict"]


def hyb_line(b, label, f="cb_tuned", v="hybrid"):
    d = b["comparisons"][f"{f}_vs_xgb_{v}"]
    x, c = b["models"][f"xgb_{v}"], b["models"][f"{f}_{v}"]
    return [label, b["n_days"], f"{x['mae']:.2f}", f"{c['mae']:.2f}", f"{sg(d['mae_pct'], 1)} %", ci(d.get("mae_ci95_7d")), ci(d.get("mae_ci95_14d")),
            f"{100 * x['cat_acc']:.1f} → {100 * c['cat_acc']:.1f} %", verdict(d)]


LINE_HDR = ["Set", "Days", "XGBoost MAE", "CatBoost MAE", "Δ", "95 % (7-d)", "95 % (14-d)", "Exact category", "Verdict"]


CHECK_LABELS = {"mae_at_least_2pct_lower": "MAE at least 2 % lower than XGBoost Hybrid", "ci_below_zero_7d_and_14d": "95 % interval below zero for BOTH 7- and 14-day blocks",
                "no_winter_more_than_2pct_worse": "no single winter more than 2 % worse", "exact_category_not_lower_by_more_than_2pts": "exact-category accuracy not lower by more than 2 points",
                "dev_cv_mae_not_worse_than_xgboost": "development-fold MAE not worse than XGBoost"}


def render(res, tun):
    T1, T2, T3 = (res.get(k) for k in ("T1_winters_persistence_weather", "T2_winters_forecast_weather", "T3_all_seasons_persistence_weather"))
    S1 = res.get("S1_production_data_persistence_weather")
    dec = res["decision"]
    integ_p = HERE / "integrity.json"
    integ = json.loads(integ_p.read_text()) if integ_p.exists() else None
    L = ["# CatBoost vs XGBoost for next-day Delhi PM2.5 - isolated experiment", "",
         "> **Retrospective and isolated.** Winters 2022-23 → 2025-26 were examined in earlier work, so none of this is an untouched hold-out; the clean test is still the "
         "winter 2026-27 forward test, which was not read. Everything lives in `experiments/catboost/`; no existing model, default, feature, checksum, Lambda, dashboard "
         "or Git state was changed. Rules were fixed in `config.json` before any result was looked at.", ""]

    # ------------------------------------------------------------ summary
    p = T1["pooled"]
    d = p["comparisons"]["cb_tuned_vs_xgb_hybrid"]
    db = p["comparisons"]["cb_base_vs_xgb_hybrid"]
    L += ["## Summary", "",
          f"**Verdict under the declared rule: {dec['verdict']}.** The evidence is mixed rather than negative: a small CatBoost edge appears in several looks but is not stable (see 'What the evidence says').", "",
          f"* Primary comparison (tuned CatBoost Hybrid vs XGBoost Hybrid, {p['n_days']} days over {len(T1['windows_scored'])} winters, leak-free weather): "
          f"MAE {p['models']['xgb_hybrid']['mae']:.2f} → {p['models']['cb_tuned_hybrid']['mae']:.2f} µg/m³ ({sg(d['mae_pct'], 1)} %), "
          f"95 % interval {ci(d.get('mae_ci95_7d'))} (7-d) / {ci(d.get('mae_ci95_14d'))} (14-d) - **{verdict(d)}**.",
          f"* The untuned CatBoost baseline: {sg(db['mae_pct'], 1)} % ({ci(db.get('mae_ci95_7d'))}) - {verdict(db)}.",
          f"* Development folds (before 15 Oct 2022, used for tuning): tuned CatBoost {tun['cb_tuned_oof']['mae']:.2f} vs XGBoost {tun['xgboost_reference']['mae']:.2f} µg/m³ "
          f"({sg(tun['tuned_minus_xgb_dev']['mae_pct'], 1)} %, {ci(tun['tuned_minus_xgb_dev'].get('mae_ci95_7d'))}).",
          "* Rule checks: " + "; ".join(f"{CHECK_LABELS[k]}: {'yes' if v else 'NO'}" for k, v in dec["checks"].items()) + ".", ""]


    # ------------------------------------------------------------ evidence overview
    seed_av = T1["pooled_seed_averaged"]["comparisons"]["cb_tuned_vs_xgb_hybrid"]
    rows = [["Development folds (before Oct 2022), tuned CatBoost vs XGBoost", f"{sg(tun['tuned_minus_xgb_dev']['mae_pct'], 1)} %", ci(tun["tuned_minus_xgb_dev"].get("mae_ci95_7d")), verdict(tun["tuned_minus_xgb_dev"])],
            ["T1 four winters, seed 42 (the declared primary comparison)", f"{sg(d['mae_pct'], 1)} %", ci(d.get("mae_ci95_7d")), verdict(d)],
            ["T1 four winters, 5 seeds averaged per library (robustness)", f"{sg(seed_av['mae_pct'], 1)} %", ci(seed_av.get("mae_ci95_7d")), verdict(seed_av)]]
    if T2 and "pooled" in T2:
        c = T2["pooled"]["comparisons"]["cb_tuned_vs_xgb_hybrid"]
        rows.append(["T2 real 24 h forecast weather, 2024-25 + 2025-26 only", f"{sg(c['mae_pct'], 1)} %", ci(c.get("mae_ci95_7d")), verdict(c)])
    if T3 and "pooled" in T3:
        c = T3["pooled"]["comparisons"]["cb_tuned_vs_xgb_hybrid"]
        rows.append(["T3 all seasons, 2025-01 → 2026-10", f"{sg(c['mae_pct'], 1)} %", ci(c.get("mae_ci95_7d")), verdict(c)])
    if S1 and "pooled" in S1:
        c = S1["pooled"]["comparisons"]["cb_tuned_vs_xgb_hybrid"]
        rows.append(["S1 production data only, four winters", f"{sg(c['mae_pct'], 1)} %", ci(c.get("mae_ci95_7d")), verdict(c)])
    wl = ", ".join(f"{n} {b['comparisons']['cb_tuned_vs_xgb_hybrid']['mae_pct']:+.1f} %" for n, b in T1["per_window"].items())
    L += ["## What the evidence says", "",
          table(["Look at the data", "CatBoost tuned Hybrid − XGBoost Hybrid, MAE", "95 % (7-d), µg/m³", "Verdict"], rows), "",
          f"Per winter (T1): {wl}. All look-ups use the same data, so they are not independent confirmations.", "",
          "**Reading.** CatBoost is better than XGBoost in most looks, by about 1-5 % (largest in 2024-25), but not everywhere, and the gain is "
          "small, not stable across windows (it is worse in 2022-23, the smallest window, and about zero over all seasons in T3), and the declared primary test misses the "
          "rule on two counts (the interval touches zero; 2022-23 is more than 2 % worse). The seed-averaged and real-forecast looks are supporting evidence added to "
          "describe the noise and were not the decision test. Fairness note: CatBoost had a 16-configuration tuning budget, while XGBoost is the project's frozen production "
          "recipe (an earlier 30-configuration random search found no meaningful gain for XGBoost either). Part of the dev-fold gap may be tuning budget rather than the algorithm. "
          "On sudden-change days - the days that matter for a school alert - the two are indistinguishable.", ""]

    # ------------------------------------------------------------ answers
    L += ["## Answers to the five questions", "",
          f"1. **Existing project unchanged?** " + (f"Yes - {integ['summary']}." if integ else "(integrity check not yet recorded)"),
          f"2. **Existing tests still pass?** " + (f"Yes - {integ['tests']}." if integ and integ.get("tests") else "(not yet recorded)"),
          f"3. **Results vs V1, V2, Hybrid, persistence:** section 3 (pooled), then sections 4-8 (by winter, season, hourly-data availability, sudden changes, real forecast weather, all seasons, robustness).",
          f"4. **Convincing?** " + ("No - it does not meet the pre-declared rule (section 9); the evidence is mixed and small (section 'What the evidence says')." if "no evidence" in dec["verdict"] else "It meets the rule on this retrospective test; this still needs your approval and a forward test (section 9)."),
          "5. **Commands to reproduce:** section 12.", ""]

    # ------------------------------------------------------------ design
    cfg = json.loads((HERE / "config.json").read_text())
    L += ["## 1. Design", "",
          f"* **Data.** Primary: `{cfg['data']['primary_daily_csv']}` (production data plus OpenCity gap-fill, which supplies hourly day-part inputs for 2024-25). "
          f"Secondary sensitivity: `{cfg['data']['secondary_daily_csv']}`, the data the deployed models were trained on.",
          "* **Same rows, target, features.** Every model is trained on the same forecast origins (target before the window start), the same target "
          "(log-change from today's PM2.5), the same 25 / 35 feature lists, and the same chronological early-stopping set (last 60 training rows). "
          "CatBoost gets no extra feature, no categorical declaration and no imputation; day-part values that do not exist stay missing and both libraries handle that natively.",
          "* **Frozen fit.** One fit per window, no refit, no early stopping on test days. Stored model files were not used or changed; frozen copies are retrained in memory.",
          "* **Weather at test time (no future observed weather).** T1: tomorrow's weather := today's observed weather. T2: the real archived 24 h-ahead forecast "
          "(2024-25 and 2025-26 only). Tomorrow's OBSERVED weather is never used as a test input.",
          "* **Same eligible days for every model** (eligibility ladder: target and origin PM2.5, weather, origin usable by the 23:00 Lambda).",
          f"* **Tuning on development data only.** Grid of {len(tun['grid'])} CatBoost configurations scored by blocked forward-chaining CV (6 folds, never shuffled) on "
          f"{tun['dev_rows']} rows with target {tun['dev_first_last'][0]} → {tun['dev_first_last'][1]} (all before the first test window, 15 Oct 2022); "
          "V1 features only because the development years have no hourly day-parts, and V2 reuses the same settings. The configuration was frozen before the test stage "
          "(`--stage test` refuses to run if `tuning.json` was made from a different `config.json`).",
          f"* **Seeds.** Primary seed 42 for every model; five seeds ({', '.join(str(s) for s in res['seeds'])}) were refitted to measure model noise.",
          f"* **Intervals.** Moving-block bootstrap (7-day blocks, 14-day sensitivity, 5000 resamples) on the same days for all models; pooled winters are placed 200 days apart "
          f"so blocks never straddle two winters. Sudden-change day = |tomorrow − today| / today ≥ {int(100 * res['sudden_change_threshold'])} %.",
          "* **Coverage.** Winters 2022-23 → 2025-26 are the only ones the project's frozen-fit protocol can score (the earlier ones lack hourly origins or have fewer than 365 training rows). "
          "Seasons other than the 15 Oct → 15 Feb window are covered only by T3 (section 7).", ""]
    rep = res.get("reproduces_existing_xgboost_numbers", {})
    if rep:
        L += ["**Sanity check - the XGBoost arm reproduces the project's existing back-test:**", "",
              table(["Winter", "This run (XGBoost Hybrid MAE)", "Existing `option_screen.json`", "Match"], [[n, r["this_run"], r["existing_option_screen"], "yes" if r["match"] else "NO"] for n, r in rep.items()]), ""]

    # ------------------------------------------------------------ tuning
    L += ["## 2. Development tuning (nothing from any test winter)", "",
          f"CatBoost configurations ranked by pooled out-of-fold MAE (µg/m³), V1 features, {tun['scored_oof_rows']} scored rows. The declared baseline is `{tun['baseline_key']}`.", ""]
    rows = [[r["key"], f"{r['oof_mae']:.3f}", f"{r['oof_rmse']:.2f}", "baseline" if r["key"] == tun["baseline_key"] else ("**selected**" if r["key"] == tun["selected"]["key"] else "")]
            for r in tun["grid"][:6]]
    base = next(r for r in tun["grid"] if r["key"] == tun["baseline_key"])
    if tun["baseline_key"] not in [r["key"] for r in tun["grid"][:6]]:
        rows.append([base["key"], f"{base['oof_mae']:.3f}", f"{base['oof_rmse']:.2f}", "baseline"])
    rows.append(["XGBoost (production recipe, not tuned)", f"{tun['xgboost_reference']['mae']:.3f}", f"{tun['xgboost_reference']['rmse']:.2f}", "reference"])
    rows.append(["Tomorrow = today", f"{tun['persistence_oof']['mae']:.3f}", f"{tun['persistence_oof']['rmse']:.2f}", "reference"])
    L += [table(["Configuration", "OOF MAE", "OOF RMSE", ""], rows), ""]
    L += [f"Tuned − XGBoost on the development folds: {sg(tun['tuned_minus_xgb_dev']['mae'])} µg/m³ ({sg(tun['tuned_minus_xgb_dev']['mae_pct'], 1)} %), "
          f"95 % {ci(tun['tuned_minus_xgb_dev'].get('mae_ci95_7d'))} (7-d) / {ci(tun['tuned_minus_xgb_dev'].get('mae_ci95_14d'))} (14-d). "
          f"Baseline − XGBoost: {sg(tun['base_minus_xgb_dev']['mae_pct'], 1)} %. Tuned − baseline: {sg(tun['tuned_minus_base_dev']['mae_pct'], 1)} % "
          f"({ci(tun['tuned_minus_base_dev'].get('mae_ci95_7d'))}), so tuning added little beyond the declared baseline. "
          "Picking the best of 16 configurations on the same folds is optimistic by construction, which is why the test windows decide, not these folds.", ""]

    # ------------------------------------------------------------ T1 pooled
    L += ["## 3. Primary test T1: four winters, frozen fit, leak-free weather", "",
          f"Windows scored: {', '.join(T1['windows_scored'])}; {p['n_days']} matched days, of which {p['days_with_day_parts']} have all 7 day-part inputs "
          "(Hybrid routes the others to V1). Persistence = tomorrow equals today.", ""]
    L += [table(MODEL_HDR, model_rows(p)), ""]
    L += ["### Difference from XGBoost (same version, same days)", "", "Negative ΔMAE / ΔRMSE and positive ΔR² / Δ accuracy favour CatBoost.", "",
          table(DIFF_HDR, diff_rows(p)), ""]
    L += ["### Within the same library (does the Hybrid / V2 story change?)", ""]
    rows = []
    for k, lab in (("xgb_hybrid_minus_xgb_v1", "XGBoost Hybrid − V1"), ("xgb_v2_minus_xgb_v1", "XGBoost V2 − V1"),
                   ("cb_tuned_hybrid_minus_cb_tuned_v1", "CatBoost Hybrid − V1"), ("cb_tuned_v2_minus_cb_tuned_v1", "CatBoost V2 − V1")):
        c = p["comparisons"][k]
        rows.append([lab, f"{sg(c['mae'])} ({sg(c['mae_pct'], 1)} %)", ci(c.get("mae_ci95_7d")), ci(c.get("mae_ci95_14d")), verdict(c)])
    L += [table(["Comparison", "ΔMAE", "95 % (7-d)", "95 % (14-d)", "Verdict"], rows), ""]

    # ------------------------------------------------------------ by winter
    L += ["## 4. By winter", "", "XGBoost vs tuned CatBoost, each model's own frozen fit.", ""]
    rows = []
    for n, b in T1["per_window"].items():
        for v in ("v1", "v2", "hybrid"):
            d = b["comparisons"][f"cb_tuned_vs_xgb_{v}"]
            x, c = b["models"][f"xgb_{v}"], b["models"][f"cb_tuned_{v}"]
            rows.append([n, b["n_days"], b["days_with_day_parts"], VNICE[v], f"{x['mae']:.2f}", f"{c['mae']:.2f}", f"{sg(d['mae_pct'], 1)} %", ci(d.get("mae_ci95_7d")),
                         f"{x['rmse']:.1f} → {c['rmse']:.1f}", f"{x['r2']:.3f} → {c['r2']:.3f}", f"{100 * x['cat_acc']:.1f} → {100 * c['cat_acc']:.1f} %",
                         f"{b['models']['persistence']['mae']:.2f}"])
    L += [table(["Winter", "Days", "With day-parts", "Model", "XGBoost MAE", "CatBoost MAE", "Δ", "95 % (7-d)", "RMSE", "R²", "Exact category", "Persistence MAE"], rows), ""]

    # ------------------------------------------------------------ subsets
    L += ["## 5. Season, hourly-data availability and sudden changes (T1, pooled)", ""]
    rows = [hyb_line(p, "all matched days")]
    for k, b in T1["subsets"].items():
        if "models" in b:
            rows.append(hyb_line(b, k.replace("_", " ")))
        else:
            rows.append([k, b["n_days"], "-", "-", "-", "-", "-", "-", b.get("note", "")])
    L += ["Hybrid, XGBoost vs tuned CatBoost.", "", table(LINE_HDR, rows), ""]
    rows = []
    for k, b in T1["subsets"].items():
        if "models" not in b:
            continue
        for f in ("xgb", "cb_tuned"):
            m = b["models"][f"{f}_hybrid"]
            rows.append([k.replace("_", " "), nm(f"{f}_hybrid"), b["n_days"], f"{m['mae']:.2f}", f"{m['rmse']:.2f}", f"{m['r2']:.3f}", f"{100 * m['cat_acc']:.1f} %",
                         f"{100 * m['within_one_cat']:.1f} %", f"{m.get('vs_persistence_pct', float('nan')):+.1f} %"])
    L += ["Hybrid in full detail:", "", table(["Subset", "Model", "Days", "MAE", "RMSE", "R²", "Exact", "Within one", "vs persistence"], rows), "",
          "Why the day counts differ: Hybrid uses V2 only when all 7 day-part inputs exist, so days without hourly data are predicted by V1 in both libraries; V2 on its own "
          "is still evaluated on those days with the day-part columns missing (section 3 tables).", ""]

    # ------------------------------------------------------------ T2
    if T2 and "pooled" in T2:
        q = T2["pooled"]
        L += ["## 6. T2: real archived 24 h weather forecast (2024-25 and 2025-26)", "",
              "This is the closest retrospective match to live inputs (the live Lambda receives a forecast). Same models as T1, only tomorrow's weather input differs.", "",
              table(MODEL_HDR, model_rows(q, [k for k in ORDER if k.endswith("hybrid") or k in ("persistence", "mean7") or k.endswith("_v1")])), "",
              table(DIFF_HDR, diff_rows(q)), ""]
        rows = [hyb_line(b, n) for n, b in T2["per_window"].items()]
        L += [table(LINE_HDR, rows), ""]

    # ------------------------------------------------------------ T3
    if T3 and "pooled" in T3:
        q = T3["pooled"]
        L += ["## 7. T3: all seasons, 2025-01-01 → 2026-10-07 (frozen fit before 2025)", "",
              f"Covers the four IMD seasons; {q['n_days']} days, {q['days_with_day_parts']} with day-parts. The fit gets up to 21 months old, which favours "
              "nothing in particular but is stricter than the winter-by-winter refit. Retrospective: this period was examined by the project's locked test.", "",
              table(MODEL_HDR, model_rows(q, [k for k in ORDER if k.endswith("hybrid") or k in ("persistence", "mean7") or k.endswith("_v1")])), "",
              table(DIFF_HDR, diff_rows(q)), ""]
        rows = [hyb_line(q, "all days")]
        for k, b in T3["subsets"].items():
            rows.append(hyb_line(b, k.replace("_", " ")) if "models" in b else [k, b["n_days"], "-", "-", "-", "-", "-", "-", b.get("note", "")])
        L += ["By season, hourly-data availability and sudden changes (Hybrid):", "", table(LINE_HDR, rows), ""]

    # ------------------------------------------------------------ robustness
    L += ["## 8. Robustness", ""]
    sa = T1["pooled_seed_averaged"]["comparisons"]["cb_tuned_vs_xgb_hybrid"]
    L += [f"* **Seed noise (T1 pooled, Hybrid MAE per seed):**", "",
          table(["Seed", "XGBoost", "CatBoost baseline", "CatBoost tuned", "tuned − XGBoost"],
                [[sd, f"{v['xgb_hybrid']:.2f}", f"{v['cb_base_hybrid']:.2f}", f"{v['cb_tuned_hybrid']:.2f}", f"{v['cb_tuned_hybrid'] - v['xgb_hybrid']:+.2f}"]
                 for sd, v in T1["per_seed_pooled_mae"].items()]), "",
          f"  Tuned CatBoost beats XGBoost in {sum(v['cb_tuned_hybrid'] < v['xgb_hybrid'] for v in T1['per_seed_pooled_mae'].values())} of {len(T1['per_seed_pooled_mae'])} seeds. "
          f"Averaging each library's five seeds: {sg(sa['mae_pct'], 1)} % ({ci(sa.get('mae_ci95_7d'))} / {ci(sa.get('mae_ci95_14d'))}) - {verdict(sa)}.", ""]
    if S1 and "pooled" in S1:
        s = S1["pooled"]
        L += [f"* **Production data only (`data/daily.csv`, T1):** {', '.join(S1['windows_scored'])}, {s['n_days']} days, {s['days_with_day_parts']} with day-parts "
              f"(without the OpenCity gap-fill the 2024-25 winter has no day-parts, so Hybrid = V1 there).", "",
              table(LINE_HDR, [hyb_line(s, "Hybrid"), hyb_line(s, "V1", v="v1"), hyb_line(s, "V2", v="v2")]), ""]
    L += ["* **Not covered:** different train/validation split lengths, other CatBoost loss functions, categorical encodings of month / weekday, and ensembling the two libraries. "
          "The project's earlier study found no model-family change beats XGBoost beyond noise; this experiment tests only CatBoost.", ""]

    # ------------------------------------------------------------ decision
    L += ["## 9. Decision (rule declared before running)", "",
          table(["Check", "Met"], [[CHECK_LABELS[k], "yes" if v else "no"] for k, v in dec["checks"].items()]), "",
          f"Primary comparison: {dec['primary_comparison']}: ΔMAE {sg(dec['mae_diff'])} µg/m³ ({sg(dec['mae_pct'], 1)} %), 95 % {ci(dec['ci95_7d'])} (7-d) / {ci(dec['ci95_14d'])} (14-d); "
          f"per winter {', '.join(f'{k} {v:+.1f} %' for k, v in dec['per_winter_pct'].items())}; development-fold ΔMAE {sg(dec['dev_cv_mae_diff'])} µg/m³.", "",
          f"**{dec['verdict'][0].upper() + dec['verdict'][1:]}.** Even if the rule were met, CatBoost would not be promoted: it stays a separate candidate until you review and approve further testing; "
          "the forward-test freeze, the default model and the dashboard are untouched.", ""]

    L += ["## 10. Limitations", "",
          "* Retrospective: every test winter was examined before; the dev-fold tuning avoids them, but the project's decisions about features and settings did not.",
          "* Few independent periods: four winters of strongly day-correlated data; a difference of a few percent needs more than this to be proven.",
          "* Hourly day-part data exists for part of 2022-23 and 2023-24 and for all of 2024-25 and 2025-26 only through the OpenCity gap-fill; the V2 and Hybrid comparisons rest on those winters.",
          "* Tuning used V1 features and observed weather (the development years have neither day-parts nor forecasts); settings may not be ideal for V2 or for forecast weather.",
          "* Test-time weather is either persistence (a no-skill bound) or a 24 h-old archived forecast (T2); live accuracy depends on the real nightly forecast.", ""]

    L += ["## 11. Files", "",
          "`config.json` (declared rules), `run_catboost.py` (tune and test stages), `stats.py`, `report.py`, `tuning.json`, `results.json` (machine-readable, daily predictions included), "
          "`report.md`, `tests/`, `requirements-catboost.txt`, `verify_unchanged.py`, `integrity.json`.", ""]
    L += ["## 12. Reproduce", "", "```bash",
          "cd kal-ki-hawa",
          "python -m venv --system-site-packages .venv-catboost            # separate environment, not committed",
          ".venv-catboost/bin/pip install -r experiments/catboost/requirements-catboost.txt",
          "python experiments/catboost/verify_unchanged.py --snapshot       # hash every protected file BEFORE running",
          "export PYTHONDONTWRITEBYTECODE=1",
          ".venv-catboost/bin/python experiments/catboost/run_catboost.py --stage tune",
          ".venv-catboost/bin/python experiments/catboost/run_catboost.py --stage test",
          "python experiments/catboost/verify_unchanged.py --check          # confirm nothing outside experiments/catboost changed",
          ".venv-catboost/bin/python experiments/catboost/report.py",
          ".venv-catboost/bin/python -m pytest experiments/catboost/tests -q   # experiment tests",
          "python -m pytest tests -q                                         # the project's own tests",
          "```", "",
          "Data needed: `data/daily_opencity.csv`, `data/daily.csv`, `data/raw/opencity_hourly/`, `data/raw/weather_prevruns_2024_2026.json` "
          "(`pipeline/fetch_prevruns.py`), and the hourly station files under `data/raw/` (git-ignored; see README)."]
    return "\n".join(L)


if __name__ == "__main__":  # pragma: no cover
    res = json.loads((HERE / "results.json").read_text())
    tun = json.loads((HERE / "tuning.json").read_text())
    (HERE / "report.md").write_text(render(res, tun))
    print("wrote", HERE / "report.md")
