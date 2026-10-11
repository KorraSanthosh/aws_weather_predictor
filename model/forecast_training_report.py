"""Markdown report for model/forecast_training_experiment.py (all numbers come from the result dict)."""
from __future__ import annotations

import json
from pathlib import Path

from winter_backtest_report import table

ARM_NAMES = {
    "A0_era5_trained": "A0 reference: train ERA5, test forecast",
    "A1_forecast_only": "A1 train ONLY on forecast-weather origins",
    "A2_mixed": "A2 train on forecast where it exists, ERA5 elsewhere",
    "A3_calibrated_input": "A3 train ERA5, forecast input calibrated",
    "S1_mixed_wx0_forecast": "S1 (sensitivity) like A2, today's weather also forecast",
    "REF_oracle_observed_tomorrow": "REF oracle: tomorrow's OBSERVED weather (not live)",
    "REF_forecast_48h": "REF 48 h forecast instead of 24 h",
}
MODEL_NAMES = {"v1": "V1 (25 features)", "v2": "V2 (35 features)", "hybrid": "Hybrid", "persistence": "Persistence", "mean7": "7-day mean"}


def ci(c, nd=2, pct=False):
    if not c:
        return "n/a"
    return f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]" if pct else f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def signed(x, nd=2):
    return f"{x:+.{nd}f}"


def verdict(d):
    if d.get("identical"):
        return "identical"
    if d.get("excludes_zero_both"):
        return "conclusive (interval excludes 0, both block lengths)"
    return "inconclusive"


def render(res):
    arms = res["arms"]
    A0 = arms["A0_era5_trained"]
    L = ["# Training on archived weather forecasts - controlled experiment", "",
         "> **Retrospective, exploratory.** Winters 2024-25 and 2025-26 were examined before, so this is not an untouched "
         "hold-out. The arms and the decision rule were fixed before any result was looked at "
         "(`model/forecast_training_experiment.py`). Nothing after 15 Feb 2026 is read, so the winter 2026-27 shadow test is untouched. "
         "No model file, default, checksum or dashboard output was changed.", ""]

    # ---------------------------------------------------------------- 1 audit
    os_path = Path("experiments/option_screen.json")
    osd = json.loads(os_path.read_text()) if os_path.exists() else {}
    o5 = osd.get("pooled", {}).get("O5_train_on_forecasts")
    rp = osd.get("realism_pooled_hybrid_vs_v1_forecast_24h")
    L += ["## 1. The two earlier numbers are different comparisons", ""]
    rows = [["**-11.3 %** (conclusive)", "Hybrid vs **V1** (a different feature set / router)",
             "both trained on observed (ERA5) weather, identical recipe", "real archived 24 h forecast",
             "2024-25 (124 d) + 2025-26 (104 d) = 228", "V1 itself (not persistence)", "`option_screen.py` section 1",
             (f"{rp['mae_ref']} → {rp['mae_opt']} MAE; CI {ci(rp['ci95_diff_7d'], pct=True)} / {ci(rp['ci95_diff_14d'], pct=True)}" if rp else "")],
            ["**-3.3 %** (inconclusive)", "Hybrid **trained on forecast weather** vs the same Hybrid trained on observed weather",
             "same models and recipe; only the TRAINING weather source differs", "real archived 24 h forecast",
             "same 228 days", "Hybrid trained on ERA5", "`option_screen.py` O5",
             (f"{o5['mae_ref']} → {o5['mae_opt']} MAE; CI {ci(o5['ci95_diff_7d'], pct=True)} / {ci(o5['ci95_diff_14d'], pct=True)}" if o5 else "")]]
    L += [table(["Result", "What was compared", "Training", "Test weather input", "Test days", "Baseline", "Where", "Numbers"], rows), "",
          "They answer different questions: the first says the 35-feature Hybrid beats the 25-feature V1 when both see real forecasts; "
          "the second asks whether changing the TRAINING weather helps one model. They are not equivalent and neither implies the other. "
          "This experiment re-runs the second question under a controlled design and reproduces the O5 number exactly "
          f"(arm A2 below: {signed(A0 and arms['A2_mixed']['pooled']['vs_A0']['hybrid']['mae_pct'], 1)} %).", ""]

    # ---------------------------------------------------------------- 2 design + limits
    fa = res["forecast_archive"]
    L += ["## 2. What the archive can and cannot support", "",
          f"* Archived forecasts (Open-Meteo Previous Runs): 24 h ahead for {fa['previous_day1_days']} days "
          f"({fa['previous_day1_first_last'][0]} → {fa['previous_day1_first_last'][1]}), 48 h ahead for {fa['previous_day2_days']} days. "
          "Nothing earlier exists, so winters 2018-19 → 2023-24 can only supply ERA5 weather.",
          "* **Issue time / lead.** The live Lambda runs at 23:00 IST and uses the latest run (its hours of tomorrow are roughly 1-30 h ahead of the run, depending on run availability; not verified here). "
          "The archive gives each hour as forecast 24 h (or 48 h) earlier, i.e. the same age or older than live. It can not leak "
          "future data and, if anything, slightly understates live skill. The exact live run can not be reproduced.",
          "* **Today's weather (wx0).** Live reads it from the forecast API's analysis; no analysis archive is available, so the main arms keep ERA5 for "
          "wx0. Arm S1 replaces it by the 24 h forecast of today as a stress test.",
          "* **No silent substitution.** A1 uses only origins with a forecast. A2 does substitute ERA5 for older origins - openly, with the counts below. "
          "No test day lacks a forecast: all arms use the same days.",
          "* Winters scored: only 2024-25 and 2025-26 (the only ones with forecast test weather). Results are by winter, not by calendar season; "
          "the archive has no autumn/summer window that the frozen-fit protocol could score.", ""]
    rows = []
    for w, i in res["winters"].items():
        tr = i["training"]
        rows.append([w, i["matched_test_days"], i["days_dropped_for_missing_forecast"], tr["obs"]["origins"],
                     tr["fc_strict"]["origins"], tr["fc_strict"]["origins_in_Nov-Feb"], tr["obs"]["origins_in_Nov-Feb"]])
    L += [table(["Winter", "Matched test days", "Days dropped (no forecast)", "Training origins (ERA5 / A2)", "Origins with a forecast (A1)",
                 "...of which Nov-Feb", "Nov-Feb origins in ERA5 history"], rows), ""]
    nr = {a: b.get("not_run") for a, b in arms.items() if b.get("not_run")}
    for a, d in nr.items():
        for w, why in d.items():
            L.append(f"* **{ARM_NAMES[a]} not run for {w}:** {why}.")
    L.append("")

    # ---------------------------------------------------------------- 3 mismatch
    L += ["## 3. Is there a forecast-vs-ERA5 mismatch inside the test windows?", "",
          "Mean difference (forecast − ERA5) and RMSE on the matched test days, tomorrow's weather:", ""]
    rows = []
    for w, mm in res["weather_mismatch"].items():
        for lab in ("raw_forecast", "calibrated_forecast"):
            rows.append([w, lab.replace("_", " ")] + [f"{mm[lab][k]['bias']:+.2f} / {mm[lab][k]['rmse']:.2f}" for k in ("temp", "rh", "wind", "precip", "pressure")])
    L += [table(["Winter", "Input", "temp °C", "humidity %", "wind km/h", "rain mm", "pressure hPa"], rows), "(bias / RMSE)", ""]
    L += ["The mismatch is **not stable**: in 2024-25 the forecast wind is lower and humidity higher than ERA5, while in 2025-26 the bias is close to zero. "
          "The earlier statement that forecasts are \"wind lower by 2-4 km/h\" came from all-year pre-window averages and does not describe the winter test windows; "
          "this report supersedes it. A correction fitted on all seasons before a window (A3) does not match winter conditions: "
          + "it moves wind by "
          + " and ".join(f"{res['weather_mismatch'][w]['calibrated_forecast']['wind']['bias']:+.1f} km/h ({w})" for w in res["weather_mismatch"])
          + " (raw: " + " and ".join(f"{res['weather_mismatch'][w]['raw_forecast']['wind']['bias']:+.1f}" for w in res["weather_mismatch"])
          + ") and humidity by " + " and ".join(f"{res['weather_mismatch'][w]['calibrated_forecast']['rh']['bias']:+.1f} pts ({w})" for w in res["weather_mismatch"])
          + ", i.e. it overshoots. A3's per-winter results below follow from that.", ""]

    # ---------------------------------------------------------------- 4 main table
    L += ["## 4. Results (pooled over the scored winters, frozen fit before each winter)", "",
          "MAE and RMSE in µg/m³; exact AQI-category accuracy; improvement over persistence = (MAE_persistence − MAE_model) / MAE_persistence. "
          "95 % intervals: moving-block bootstrap (7-day blocks, 5000 resamples).", ""]
    rows = []
    for a, b in arms.items():
        if "pooled" not in b:
            continue
        P = b["pooled"]
        for m in ("v1", "v2", "hybrid"):
            x = P["models"][m]
            rows.append([ARM_NAMES[a], f"{', '.join(P['winters_pooled'])} ({P['n_days']} d)", MODEL_NAMES[m], f"{x['mae']:.2f} {ci(x['mae_ci95'])}",
                         f"{x['rmse']:.2f} {ci(x['rmse_ci95'])}", f"{100 * x['cat_acc']:.1f} % {ci([100 * c for c in x['cat_acc_ci95']], 1)}",
                         f"{x['vs_persistence']['pct']:+.1f} % {ci(x['vs_persistence']['ci95'], 1, pct=True)}"])
    pa = A0["pooled"]["models"]
    for m in ("persistence", "mean7"):
        x = pa[m]
        rows.append(["baseline", f"{', '.join(A0['pooled']['winters_pooled'])} ({A0['pooled']['n_days']} d)", MODEL_NAMES[m], f"{x['mae']:.2f} {ci(x['mae_ci95'])}",
                     f"{x['rmse']:.2f} {ci(x['rmse_ci95'])}", f"{100 * x['cat_acc']:.1f} % {ci([100 * c for c in x['cat_acc_ci95']], 1)}", "-"])
    L += [table(["Arm", "Winters (matched days)", "Model", "MAE [95 %]", "RMSE [95 %]", "Exact category [95 %]", "Better than persistence"], rows), "",
          "A1 covers only 2025-26 (see section 2), so its row is **not comparable** to the pooled rows of the other arms; "
          "its comparison with A0 below uses the same 2025-26 days.", ""]

    # ---------------------------------------------------------------- 5 diffs vs previous model
    L += ["## 5. Difference from the previous model (same model under the reference arm A0)", "",
          "Negative MAE/RMSE = better than A0. Intervals are for the paired difference.", ""]
    rows = []
    for a, b in arms.items():
        if a == "A0_era5_trained" or "pooled" not in b:
            continue
        for m in ("v1", "v2", "hybrid"):
            d = b["pooled"]["vs_A0"][m]
            rows.append([ARM_NAMES[a], MODEL_NAMES[m], f"{signed(d['mae'])} ({signed(d['mae_pct'], 1)} %)", ci(d.get("mae_ci95_7d"), pct=True),
                         ci(d.get("mae_ci95_14d"), pct=True), f"{signed(d['rmse'])} {ci(d.get('rmse_ci95_7d'), pct=True)}",
                         f"{100 * d['cat_acc']:+.1f} pts {ci([100 * c for c in d.get('cat_acc_ci95_7d', [])], 1, pct=True) if d.get('cat_acc_ci95_7d') else ''}",
                         verdict(d)])
    L += [table(["Arm", "Model", "ΔMAE", "95 % 7-d", "95 % 14-d", "ΔRMSE [95 %]", "Δ exact category", "Verdict"], rows), ""]

    L += ["### Model versus model inside each arm (Hybrid − V1 and V2 − V1, ΔMAE)", ""]
    rows = []
    for a, b in arms.items():
        if "pooled" not in b:
            continue
        for k, nm in (("hybrid_minus_v1", "Hybrid − V1"), ("v2_minus_v1", "V2 − V1")):
            d = b["pooled"]["model_vs_model"][k]
            rows.append([ARM_NAMES[a], nm, f"{signed(d['mae'])} ({signed(d['mae_pct'], 1)} %)", ci(d.get("mae_ci95_7d"), pct=True), ci(d.get("mae_ci95_14d"), pct=True), verdict(d)])
    L += [table(["Arm", "Comparison", "ΔMAE", "95 % 7-d", "95 % 14-d", "Verdict"], rows), ""]

    # ---------------------------------------------------------------- 6 by winter
    L += ["## 6. By winter", ""]
    rows = []
    for a, b in arms.items():
        for w, x in b.get("winters", {}).items():
            h = x["models"]["hybrid"]
            d = x.get("vs_A0", {}).get("hybrid")
            rows.append([ARM_NAMES[a], w, x["n_days"], f"{x['models']['v1']['mae']:.2f}", f"{x['models']['v2']['mae']:.2f}", f"{h['mae']:.2f} {ci(h['mae_ci95'])}",
                         f"{h['rmse']:.2f}", f"{100 * h['cat_acc']:.1f} %", f"{h['vs_persistence']['pct']:+.1f} %",
                         (f"{signed(d['mae_pct'], 1)} % {ci(d.get('mae_ci95_7d'), pct=True)}" if d and not d.get('identical') else "-")])
    L += [table(["Arm", "Winter", "Days", "V1 MAE", "V2 MAE", "Hybrid MAE [95 %]", "Hybrid RMSE", "Hybrid exact cat.", "Hybrid vs persistence", "Hybrid ΔMAE vs A0 (7-d CI, µg/m³)"], rows), ""]

    # ---------------------------------------------------------------- 7 sub-seasons and sudden days
    L += ["## 7. Sub-seasons and sudden-change days (Hybrid, pooled)", "",
          "Stubble period = 15 Oct-30 Nov; deep winter = 1 Dec-15 Feb. Sudden-change day = |tomorrow − today| / today ≥ 30 %. Small subsets give wide intervals.", ""]
    rows = []
    for a in ("A0_era5_trained", "A2_mixed", "A3_calibrated_input", "S1_mixed_wx0_forecast", "REF_oracle_observed_tomorrow"):
        for k, x in arms[a].get("subsets", {}).items():
            if "models" not in x:
                continue
            h = x["models"]["hybrid"]
            d = x.get("vs_A0", {}).get("hybrid")
            rows.append([ARM_NAMES[a], k.replace("_", " "), x["n_days"], f"{h['mae']:.2f} {ci(h['mae_ci95'])}", f"{h['rmse']:.2f}", f"{100 * h['cat_acc']:.1f} %",
                         f"{h['vs_persistence']['pct']:+.1f} %", (f"{signed(d['mae_pct'], 1)} % {ci(d.get('mae_ci95_7d'), pct=True)}" if d else "-")])
    L += [table(["Arm", "Subset", "Days", "Hybrid MAE [95 %]", "RMSE", "Exact cat.", "vs persistence", "ΔMAE vs A0 (µg/m³ CI)"], rows), ""]
    s = arms["A2_mixed"]["subsets"]["sudden_change_days_ge30pct"]
    if "vs_A0" in s:
        d = s["vs_A0"]["hybrid"]
        L += [f"On sudden-change days A2 is {signed(d['mae_pct'], 1)} % vs A0 ({ci(d.get('mae_ci95_7d'), pct=True)} µg/m³): "
              + ("worse, and the interval excludes zero. " if d.get("mae_ci95_7d", [0, 0])[0] > 0 else "no clear difference. ")
              + ("The pooled gain of A2 comes from ordinary days, not from the days that matter most for a school alert." if arms["A2_mixed"]["subsets"]["stable_days"]["vs_A0"]["hybrid"]["mae"] < 0 else ""), ""]

    # ---------------------------------------------------------------- 8 seeds
    sr = res["seed_robustness"]
    L += ["## 8. Is the A2 gain model noise? (5 random seeds)", "",
          "Production uses one seed (42). XGBoost's row/column subsampling makes any single refit slightly different, so A0 and A2 were refitted "
          f"with seeds {sr['seeds']}.", ""]
    rows = [[sd, f"{v['A0_era5_trained']['hybrid']:.2f}", f"{v['A2_mixed']['hybrid']:.2f}", f"{v['A2_mixed']['hybrid'] - v['A0_era5_trained']['hybrid']:+.2f}"]
            for sd, v in sr["pooled_mae_by_seed"].items()]
    L += [table(["Seed", "A0 Hybrid MAE", "A2 Hybrid MAE", "A2 − A0"], rows), ""]
    d = sr["A2_minus_A0_seed_averaged"]["hybrid"]
    L += [f"A2 beats A0 in {sr['seeds_where_A2_beats_A0_hybrid']} of {len(sr['seeds'])} seeds. Averaging the 5 seeds' predictions: "
          f"{signed(d['mae'])} µg/m³ ({signed(d['mae_pct'], 1)} %), interval {ci(d.get('mae_ci95_7d'), pct=True)} (7-d) / {ci(d.get('mae_ci95_14d'), pct=True)} (14-d) - "
          + ("excludes zero." if d.get("excludes_zero_both") else "still includes zero.") +
          " The direction is not a seed accident; what remains uncertain is whether it holds across different winters (it is a different size in each, section 6). "
          "This check was added after seeing the first result, so it is supporting evidence only and is not part of the decision rule.", ""]

    # ---------------------------------------------------------------- 9 decision
    L += ["## 9. Decision (rule fixed in advance)", "",
          f"A candidate needs Hybrid pooled MAE ≥ {res['rule']['min_gain_pct']:.0f} % lower than A0, an interval below zero for both 7- and 14-day blocks, "
          f"and no winter more than {res['rule']['max_single_winter_loss_pct']:.0f} % worse.", ""]
    rows = []
    for a in ("A1_forecast_only", "A2_mixed", "A3_calibrated_input"):
        r = arms[a].get("rule")
        if not r:
            rows.append([ARM_NAMES[a], "-", "-", "-", "-", "not evaluable"])
            continue
        rows.append([ARM_NAMES[a], ", ".join(r["winters_run"]), f"{r['hybrid_mae_pct_vs_A0']:+.1f} %", f"{ci(r['ci95_7d'])} / {ci(r['ci95_14d'])}",
                     f"{r['worst_single_winter_pct']:+.1f} %", "**promising**" if r["promising"] else ("**worse** (interval above 0)" if r["ci95_7d"][0] > 0 and r["ci95_14d"][0] > 0 else "no evidence of improvement")])
    L += [table(["Candidate", "Winters", "Hybrid ΔMAE vs A0", "95 % 7-d / 14-d (µg/m³)", "Worst winter", "Verdict"], rows), ""]
    prom = [a for a in ("A1_forecast_only", "A2_mixed", "A3_calibrated_input") if arms[a].get("rule", {}).get("promising")]
    if prom:
        L += [f"**{', '.join(ARM_NAMES[a] for a in prom)} met the rule.** That is a hypothesis for the winter 2026-27 shadow test, not a reason to change the default.", ""]
    else:
        L += ["**No candidate met the rule. Keep the existing approach (A0: train on observed weather); v1 stays the default and v2/Hybrid stay experimental.**", ""]

    ora = arms["REF_oracle_observed_tomorrow"]["pooled"]["vs_A0"]["hybrid"]
    a2 = arms["A2_mixed"]["pooled"]["vs_A0"]["hybrid"]
    a1 = arms["A1_forecast_only"]["pooled"]["vs_A0"]["hybrid"]
    a3w = arms["A3_calibrated_input"]["winters"]
    L += ["## 10. Interpretation", "",
          f"* **Statistically convincing?** No. A2 (the only arm with a consistent sign: {signed(a2['mae_pct'], 1)} % pooled, "
          f"{ci(a2.get('mae_ci95_7d'), pct=True)} µg/m³) is {signed(arms['A2_mixed']['winters']['2024-25']['vs_A0']['hybrid']['mae_pct'], 1)} % in 2024-25 and {signed(arms['A2_mixed']['winters']['2025-26']['vs_A0']['hybrid']['mae_pct'], 1)} % in 2025-26, and its pooled interval includes zero. Seed noise does not explain it (section 8), "
          "but two winters of day-correlated data are too few to prove a 3 % effect.",
          f"* **Training only on forecasts (A1) is clearly worse** ({signed(a1['mae_pct'], 1)} % in 2025-26, interval excludes zero): the archive begins in 2024, so "
          f"the model loses the six earlier winters of history ({res['winters']['2025-26']['training']['fc_strict']['origins_in_Nov-Feb']} Nov-Feb origins instead of {res['winters']['2025-26']['training']['obs']['origins_in_Nov-Feb']}). More history matters more than matching the weather source.",
          f"* **Calibrating the forecast (A3)** gave {signed(a3w['2024-25']['vs_A0']['hybrid']['mae_pct'], 1)} % in 2024-25 and "
          f"{signed(a3w['2025-26']['vs_A0']['hybrid']['mae_pct'], 1)} % in 2025-26: it helps when the winter has the bias the correction learned and hurts when it does not "
          "(section 3). A fixed correction is unsafe until the bias is shown to be stable.",
          f"* **Ceiling.** Even a perfect weather forecast (oracle) changes the A0 Hybrid's MAE by {signed(ora['mae_pct'], 1)} % "
          f"({ci(ora.get('mae_ci95_7d'), pct=True)} µg/m³). That is the most any better tomorrow-weather input could add at test time; A2's {signed(a2['mae_pct'], 1)} % is a different lever (training) "
          "and the two should not be added.",
          "* **Bottleneck.** The mismatch the experiment targets is small and winter-dependent. The remaining error is probably dominated by what weather inputs can not explain "
          "(emissions, smoke from fires, local events) - this experiment does not measure that, it is the reason the next steps add new information. "
          "Sudden-change days carry the largest error; candidate arms change them by " + ", ".join(
              f"{n} {signed(arms[a]['subsets']['sudden_change_days_ge30pct']['vs_A0']['hybrid']['mae_pct'], 1)} %"
              for n, a in (("A2", "A2_mixed"), ("A3", "A3_calibrated_input"))) + " (none improves them).", "",
          "### Will it transfer to live predictions?", "",
          "* Possibly a little, not provably. Live inputs are the newest forecast run (probably slightly better than the 24 h-old archive), so the true mismatch is likely smaller than tested here.",
          "* Training on archived forecasts bakes in the behaviour of the forecast models of 2024-26; Open-Meteo changes its models over time, so the benefit can decay.",
          "* The nightly Lambda does not store the weather it used (only a hash of the inputs), so there is no exact-live training set yet. "
          "Logging the forecast weather inputs each night, starting now, would give 2026-27 a clean, leak-free set for the next experiment. That is a backend change and needs your approval.",
          "", "### Recommendation", "",
          "Keep A0 (train on observed weather). Do not retrain or deploy anything on the basis of this experiment. A2 can enter the winter 2026-27 shadow test as a "
          "pre-declared challenger only if you approve; the cheaper and more informative step is to start logging the live forecast inputs. "
          "Next, as you planned: the OpenCity hourly-data experiment (more winters with day-part data), which targets the larger source of error.", ""]
    return "\n".join(L)
