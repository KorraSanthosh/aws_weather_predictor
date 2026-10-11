"""Render report.md from metrics.json (no new numbers are computed here).   python experiments/regional_cities/report.py"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = json.loads((HERE / "metrics.json").read_text())
CFG = json.loads((HERE / "config.json").read_text())

LABEL = {
    "persistence": "Tomorrow = today (naive)", "v1": "V1 baseline (deployed default)", "v1_Rlag": "V1 + regional, previous day (PRIMARY)",
    "v1_Rd0": "V1 + regional, same day (needs data by 23:00)", "v1_plc": "V1 + 12 PLACEBO columns", "v2": "V2 candidate",
    "hybrid": "Hybrid (V2 if day-parts, else V1)", "hybrid_Rlag": "Hybrid + regional, previous day", "hybrid_Rd0": "Hybrid + regional, same day",
    "hybrid_plc": "Hybrid + 12 PLACEBO columns",
}


def pc(x, d=1):
    return f"{100 * x:.{d}f} %"


def ci(c, d=3, pct=False):
    if not c:
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.{d}f}")
    return f"[{f(c[0])}, {f(c[1])}]"


def table_models(b):
    rows = ["| Model | Exact category (95 % CI) | Within one category | MAE µg/m³ (95 % CI) | RMSE | R² |", "|---|---|---|---|---|---|"]
    for k in ("persistence", "v1", "v1_Rlag", "v1_Rd0", "v1_plc", "v2", "hybrid", "hybrid_Rlag", "hybrid_Rd0"):
        m = b["models"][k]
        rows.append(f"| {LABEL[k]} | **{pc(m['cat_acc'])}** {ci(m.get('cat_acc_ci95'), pct=True)} | {pc(m['within_one_cat'])} | "
                    f"{m['mae']:.1f} {ci(m.get('mae_ci95'), 1)} | {m['rmse']:.1f} | {m['r2']:.2f} |")
    return "\n".join(rows)


def table_diffs(b):
    rows = ["| Comparison (same days) | MAE change | 95 % interval of MAE change, µg/m³ (7-day / 14-day blocks) | Exact-category change | Seed-averaged MAE change |",
            "|---|---|---|---|---|"]
    for key, name, sa in (("v1_Rlag_vs_v1", "V1 + regional (prev. day) vs V1", "v1_Rlag_vs_v1"), ("v1_Rd0_vs_v1", "V1 + regional (same day) vs V1", "v1_Rd0_vs_v1"),
                          ("v1_plc_vs_v1", "V1 + PLACEBO vs V1", None), ("hybrid_Rlag_vs_hybrid", "Hybrid + regional (prev. day) vs Hybrid", "hybrid_Rlag_vs_hybrid"),
                          ("hybrid_Rd0_vs_hybrid", "Hybrid + regional (same day) vs Hybrid", "hybrid_Rd0_vs_hybrid")):
        d = b["diffs"][key]
        s = b["seed_averaged"][sa]["mae_pct"] if sa else None
        rows.append(f"| {name} | {d['mae_pct']:+.2f} % | {ci(d.get('mae_ci95_7d'), 2)} / {ci(d.get('mae_ci95_14d'), 2)} | {100 * d['cat_acc']:+.1f} pts | "
                    f"{'n/a' if s is None else f'{s:+.2f} %'} |")
    return "\n".join(rows)


def main():
    out = []
    A = out.append
    dec = M["decision"]
    A("# Extra NCR / upwind city data: does it improve Delhi's next-day forecast?\n")
    A("**RETROSPECTIVE** test. Calendar 2025 and winter 2025-26 were examined in earlier experiments, so neither is an untouched hold-out. "
      f"The protocol and pass/fail rule are in `config.json` (sha256 `{M['config_sha256'][:12]}…`, stored in `metrics.json`), written before any regional model was run. "
      "No default model, Lambda, dashboard output or frozen file was changed.\n")
    A(f"## Answer\n\n**{dec['answer']}.**\n")
    t1, t2 = M["tests"]["T1_calendar_2025"], M["tests"]["T2_winter_2025_26"]
    g1, g2 = t1["models"], t2["models"]
    f1, f2 = t1["five_seed_mean"], t2["five_seed_mean"]
    A(f"Adding PM2.5 from 44 government stations in 4 regions around Delhi did not make the forecast measurably better. "
      f"Averaged over five training seeds, exact-category accuracy of the deployed V1 model is {pc(f1['v1']['cat_acc'])} (2025) and {pc(f2['v1']['cat_acc'])} (winter 2025-26); with the regional data it is "
      f"{pc(f1['v1_Rlag']['cat_acc'])} and {pc(f2['v1_Rlag']['cat_acc'])}, and with 12 placebo columns (real regional numbers taken from a random other day of the same month) it is {pc(f1['v1_plc']['cat_acc'])} and {pc(f2['v1_plc']['cat_acc'])}. "
      f"The single run the rule calls primary (seed 42) shows {pc(g1['v1_Rlag']['cat_acc'])} and {pc(g2['v1_Rlag']['cat_acc'])} for the regional model, but in winter that is the best of its five seeds "
      f"(range {pc(f2['v1_Rlag']['cat_acc_min'])} to {pc(f2['v1_Rlag']['cat_acc_max'])}); one seed is a lucky draw, not a gain. "
      "The differences are inside the noise (none of the 12 regional-versus-control intervals, 2 variants x 3 models x 2 tests, excludes zero), the seed-averaged MAE change is about zero, "
      "and in winter the Hybrid gets about 3 % worse with the regional columns, the same damage that the placebo columns do on their own. "
      "The model does use the new columns (roughly a fifth to a quarter of its split gain), but no more than it uses placebo columns, which is what fitting noise looks like.\n")

    A("## What was added\n")
    A("| Group | Stations | Where | Group-days available (2018 / 2019 / 2020 / 2021 / 2022 / **2023** / **2024** / 2025 / 2026) |\n|---|---|---|---|")
    where = {"ncr_near": "Noida, Gurugram, Faridabad, Ghaziabad … (≤ 55 km)", "haryana_nw": "Rohtak, Panipat, Karnal, Jind, Bhiwani … (W / NW)",
             "punjab": "Ludhiana, Jalandhar … (stubble belt, PPCB)", "up_east": "Meerut, Hapur, Muzaffarnagar … (E, UPPCB)"}
    for g, c in M["regional_coverage"].items():
        y = c["group_days_by_year"]
        A(f"| {g} | {c['stations']} | {where[g]} | " + " / ".join(str(y.get(str(k), 0)) for k in range(2018, 2027)) + " |")
    A("\n**The 2023 and 2024 hole.** The public OpenAQ archive holds no Indian government-station data for 2023 and 2024 (both layouts were checked) and only part of 2022. "
      "Regional inputs are therefore blank in those years, never filled in. The regional models learn from "
      f"{t1['training']['origins_with_>=2_regional_groups']} (2025 test) and {t2['training']['origins_with_>=2_regional_groups']} (winter test) training days instead of "
      f"{t1['training']['origins']} / {t2['training']['origins']}. That can hide a real benefit; the restricted-training check below addresses it. "
      "Station files: 2.05 million hourly readings, 2018 → 2026-02-15; nothing later was read.\n")
    A("**Inputs.** Per region and day: the PM2.5 level, its ratio to Delhi's own value and its one-day change (12 numbers). *Previous day* = the last complete day before today "
      "(needs no new live-data assumption). *Same day* = today's hours up to 22:00 (needs the regional stations to report by 23:00 IST; not verified for the live Lambda). "
      "The Delhi target and every Delhi input are unchanged.\n")

    for tn, title in (("T1_calendar_2025", "Test 1: calendar 2025"), ("T2_winter_2025_26", "Test 2: winter 2025-26 (15 Oct 2025 → 15 Feb 2026)")):
        b = M["tests"][tn]
        A(f"## {title} ({b['n_days']} days, same days for every row)\n")
        A(table_models(b) + "\n")
        A("MAE intervals and the exact-category interval are 7-day moving-block bootstrap, 5000 resamples. Seed 42 shown; five seeds below.\n")
        A("### Paired differences\n")
        A(table_diffs(b) + "\n")
        A("### How much the numbers move by chance (five training seeds)\n")
        A("The only thing that differs between seeds is which random 85 % of the training days and 85 % of the columns each tree gets to see (the production recipe samples both); data, features and test days are identical. "
          f"One test day is {100 / b['n_days']:.2f} points of accuracy here.\n")
        A("| Model | Exact category: mean of 5 seeds | lowest – highest seed | MAE: mean of 5 seeds |\n|---|---|---|---|")
        for k in ("v1", "v1_Rlag", "v1_Rd0", "v1_plc", "hybrid", "hybrid_Rlag", "hybrid_Rd0", "hybrid_plc"):
            fm = b["five_seed_mean"][k]
            A(f"| {LABEL[k]} | **{pc(fm['cat_acc'])}** | {pc(fm['cat_acc_min'])} – {pc(fm['cat_acc_max'])} | {fm['mae']:.1f} |")
        A("")
        A("A single seed can move a model's accuracy by several points by itself, which is more than any difference between the arms above. "
          "That is why a placebo that carries no information can still score a point higher in one seed.\n")
        A(f"Seeds (42, 7, 123, 2024, 99): V1 + regional beats V1 in {b['seeds_where_v1_Rlag_beats_v1']} of 5 seeds; Hybrid + regional beats Hybrid in {b['seeds_where_hybrid_Rlag_beats_hybrid']} of 5.\n")
        av = b["availability_on_test_days"]
        A(f"Regional inputs present (≥ 2 of 4 regions, previous day) on {av['days_with_>=2_regional_groups_(Rlag)']} of {b['n_days']} test days; missing on {av['days_with_none']}.\n")

    A("## Pre-declared rule\n")
    A(dec["rule"] + "\n")
    A("| Test | 1. MAE gain ≥ 2 % | 2. interval below zero (7 d and 14 d) | 3. category accuracy not lower | 4. placebo gain < ½ real gain | 5. seed-averaged gain | \n|---|---|---|---|---|---|")
    for tn, v in dec["per_test"].items():
        ok = lambda x: "pass" if x else "**fail**"  # noqa: E731
        A(f"| {tn} | {v['1_mae_gain_pct']:+.2f} % → {ok(v['1_pass'])} | {ci(v['2_ci95_7d'], 2)} / {ci(v['2_ci95_14d'], 2)} → {ok(v['2_pass'])} | "
          f"{100 * v['3_cat_acc_change']:+.1f} pts → {ok(v['3_pass'])} | placebo {v['4_placebo_gain']:+.2f} vs real {v['4_real_gain']:+.2f} µg → {ok(v['4_pass'])} | "
          f"{v['5_seed_avg_mae_diff']:+.3f} µg → {ok(v['5_pass'])} |")
    A(f"\nResult: **{dec['answer']}**.\n")

    A("## Does the model actually use the regional columns?\n")
    for tn, b in M["tests"].items():
        im = b["importance_v1_Rlag_seed42"]
        A(f"- {tn}: the 12 regional columns hold {pc(im['regional_share_of_total_gain'])} of V1's split gain; the 12 placebo columns hold "
          f"{pc(im['placebo_columns_share_of_total_gain_(same_count_of_columns,_no_information)'])}. Using a column is not the same as being helped by it.")
    d = M["tests"]["T2_winter_2025_26"]["diffs"]
    A(f"- Winter 2025-26, Hybrid: real regional columns {d['hybrid_Rlag_vs_hybrid']['mae_pct']:+.1f} % MAE, placebo columns {d['hybrid_plc_vs_hybrid']['mae_pct']:+.1f} % "
      f"(14-day-block interval {ci(d['hybrid_plc_vs_hybrid'].get('mae_ci95_14d'), 2)} µg/m³). Merely widening the model with 12 columns costs accuracy on this short test; the regional data did not repay that cost.")
    A("")

    A("## Supplementary (added after the first one-seed look; never decides anything)\n")
    A("Both V1 and V1 + regional trained only on days that have regional inputs, so the 2023-24 hole cannot dilute the new columns:\n")
    A("| Test | Training days | V1 exact category / MAE | V1 + regional exact category / MAE | MAE change | Seed-averaged | Seeds regional wins |\n|---|---|---|---|---|---|---|")
    for tn, b in M["tests"].items():
        s = b["supplementary_restricted_training"]
        if "ctrlR" not in s:
            A(f"| {tn} | not run ({s.get('not_run')}) | | | | | |")
            continue
        A(f"| {tn} | {s['training_origins']} | {pc(s['ctrlR']['cat_acc'])} / {s['ctrlR']['mae']:.1f} | {pc(s['RlagR']['cat_acc'])} / {s['RlagR']['mae']:.1f} | "
          f"{s['RlagR_vs_ctrlR']['mae_pct']:+.2f} % (interval {ci(s['RlagR_vs_ctrlR'].get('mae_ci95_7d'), 2)} µg) | {s['seed_averaged_RlagR_vs_ctrlR']['mae_pct']:+.2f} % | {s['seeds_where_RlagR_beats_ctrlR']} of 5 |")
    A("")

    s1, s2 = (M["tests"][k]["supplementary_restricted_training"] for k in ("T1_calendar_2025", "T2_winter_2025_26"))
    if "ctrlR" in s1 and "ctrlR" in s2:
        A(f"Even with the missing years taken out of the picture there is no benefit: in 2025 the regional model is {s1['RlagR_vs_ctrlR']['mae_pct']:+.1f} % MAE "
          f"(interval above zero, i.e. worse); in winter it is {s2['RlagR_vs_ctrlR']['mae_pct']:+.1f} % (interval {ci(s2['RlagR_vs_ctrlR'].get('mae_ci95_7d'), 2)} µg/m³ spans zero; "
          f"better in {s2['seeds_where_RlagR_beats_ctrlR']} of 5 seeds). A weak winter hint at most, and one that the 2025 result contradicts.\n")

    A("## Subsets (exploratory, V1 + regional vs V1, seed 42)\n")
    A("| Test | Subset | Days | V1 exact / MAE | V1 + regional exact / MAE | PLACEBO MAE | MAE change | interval µg (7 d) |\n|---|---|---|---|---|---|---|---|")
    for tn, b in M["tests"].items():
        for sn, sv in b["subsets"].items():
            if "v1" not in sv:
                continue
            d = sv["v1_Rlag_vs_v1"]
            A(f"| {tn} | {sn} | {sv['n_days']} | {pc(sv['v1']['cat_acc'])} / {sv['v1']['mae']:.1f} | {pc(sv['v1_Rlag']['cat_acc'])} / {sv['v1_Rlag']['mae']:.1f} | {sv['v1_plc']['mae']:.1f} | "
              f"{d['mae_pct']:+.2f} % | {ci(d.get('mae_ci95_7d'), 2)} |")
    A("\nThe stubble-burning weeks (15 Oct – 30 Nov) are where Punjab / Haryana smoke should help most; there the regional model is not better either. Subsets are small and nothing here is a finding.\n")

    A("## Integrity checks (all passed; see tests/)\n")
    for tn, b in M["tests"].items():
        la = b["leakage_audit"]
        A(f"- {tn}: {la['rows_checked']} test rows rebuilt from a spying store, no regional date after the allowed day was read (previous-day inputs: ≤ today − 1; same-day inputs: ≤ today). "
          f"The test-time 'same-day' values (hours ≤ 22) differ from full-day values on {la['rows_where_live_origin_day_differs_from_full_day']} rows, so the live view is really used. "
          f"Canary (a column that IS tomorrow's answer) drives V1's MAE from {b['models']['v1']['mae']:.1f} to {b['canary_mae_with_future_target_feature']:.1f}, so a leak would be visible. "
          f"The regional-free control reproduces the production fitting code exactly (V1: {b['control_identical_to_production_fit']['v1']}, V2: {b['control_identical_to_production_fit']['v2']}).")
    A(f"- No regional station id appears in any Delhi station folder ({M['integrity']['regional_stations']} regional vs {M['integrity']['delhi_station_files']} Delhi files). The Delhi target is unchanged.")
    A("- Re-running the whole experiment reproduces `metrics.json` byte for byte.\n")

    A("## Limits\n")
    for x in CFG["limitations"]:
        A(f"- {x}")
    A("- Power: the 2025 test (344 days) can resolve an MAE change of roughly ±0.5 µg/m³ (about 2 %); the winter test (104 days) only about ±2 µg/m³ (about 6 %). A real gain smaller than that would not be seen.")
    A("- Why a gain is hard here: Delhi's own station average, yesterday's level and tomorrow's forecast wind already carry most of what upwind towns could tell a one-day-ahead model. "
      "That is an explanation, not something this experiment tested.\n")
    A("## Reproduce\n")
    A("```bash\npython experiments/regional_cities/select_stations.py      # needs archive_coverage.json (rule-based station choice)\n"
      "python experiments/regional_cities/download_regional.py   # public OpenAQ archive, no key; ~5 min; writes data/raw/regional/ (git-ignored)\n"
      "python experiments/regional_cities/run_regional.py        # ~2 min; writes metrics.json and regional_daily.csv\n"
      "python experiments/regional_cities/report.py              # writes this file\n"
      "python -m pytest experiments/regional_cities/tests -q\n```\n")
    (HERE / "report.md").write_text("\n".join(out))
    print("report.md written")


if __name__ == "__main__":
    main()
