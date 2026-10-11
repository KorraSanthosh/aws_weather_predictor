"""Build the calibration file that ships with the product range:  core/range_calibration.json

    python experiments/prediction_range/build_calibration.py          (from the project folder; needs data/raw + xgboost)

What it does (RETROSPECTIVE; nothing here changes a model, the default, the dashboard data or the Lambda):
  1. Rebuilds the three out-of-sample blocks used by the validated experiment (run_range.py), with the SAME recipe, seed 42 and
     archived 24 h forecast weather:
         A  models trained through 2023-12-31 -> forecast 2024-01-01 .. 2024-12-31
         B  models trained through 2024-12-31 -> forecast 2025-01-01 .. 2025-10-14
         C  models trained through 2025-10-14 -> forecast 2025-10-15 .. 2026-02-15
     Every residual therefore comes from a model that had not seen that day (and days with no archived forecast are skipped).
  2. Takes the residuals log1p(actual) - log1p(forecast) of the v1, v2 and hybrid point forecasts, and computes the M1 90 %
     offsets with core.prediction_range.conformal_offsets (the code that ships).
  3. Re-runs the validated experiment's two tests with the shipped code, calibrating only on days strictly BEFORE each test, and
     refuses to write anything if the shipped maths does not reproduce the experiment's numbers.
Outputs: core/range_calibration.json (shipped), experiments/prediction_range/calibration_residuals.json (audit trail),
         experiments/prediction_range/product_check.json (reproduction of the experiment with the shipped code).
"""
import hashlib
import json
import os
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "experiments" / "weather_source_comparison", HERE):
    sys.path.insert(0, str(p))
os.chdir(ROOT)

import forecast_training_experiment as fte  # noqa: E402
import option_screen as osn  # noqa: E402
import run_comparison as rc  # noqa: E402
import run_range as rr  # noqa: E402
from core import prediction_range as pr  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402

SEED = 42
VERSIONS = {"v1_25_baseline": "v1", "v2_35_candidate": "v2", "v3_hybrid": "hybrid"}
BLOCKS = [("A", date(2023, 12, 31), date(2024, 1, 1), date(2024, 12, 31)),
          ("B", date(2024, 12, 31), date(2025, 1, 1), date(2025, 10, 14)),
          ("C", date(2025, 10, 14), date(2025, 10, 15), date(2026, 2, 15))]
OUT_CAL = ROOT / "core" / "range_calibration.json"
OUT_RES = HERE / "calibration_residuals.json"
OUT_CHECK = HERE / "product_check.json"
ALPHA = 1 - pr.LEVEL


def oos_block(train, rows_by_day, days):
    """Forecasts for `days` from models trained on `train` only (v1, v2, hybrid), plus the actuals."""
    models = fte.fit_models(train, SEED)
    rows = [rows_by_day[t]["row"] for t in days]
    P = fte.predict_all(models, rows)
    return {"days": days, "y": np.array([rows_by_day[t]["y"] for t in days], float),
            "v1": P["v1"], "v2": P["v2"], "hybrid": P["hybrid"], "p": P["hybrid"]}


def cat_slice(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]) for k, v in d.items()}


def join(*blocks):
    return {"days": sum([b["days"] for b in blocks], []),
            **{k: np.concatenate([b[k] for b in blocks]) for k in ("y", "v1", "v2", "hybrid", "p")}}


def shipped_range(d, offsets, version="v3_hybrid"):
    """Range from the SHIPPED function (whole-number outward rounding) for every day of block d."""
    cal = {"level": pr.LEVEL, "method": pr.METHOD, "calibration_id": "check",
           "versions": {version: {"lo": offsets[0], "hi": offsets[1], "n": 0, "first": "", "last": ""}}}
    lo, hi = [], []
    for p in d["p"]:
        r = pr.prediction_range(p, version, calibration=cal)
        lo.append(r["pm25_low"]); hi.append(r["pm25_high"])
    return np.array(lo, float), np.array(hi, float)


def reproduce(name, cal_block, test_block, expected):
    """Experiment M1 90 % vs the shipped maths, calibrated on `cal_block` only (which must end before `test_block` starts)."""
    assert max(cal_block["days"]) < min(test_block["days"]), f"{name}: calibration is not strictly before the test"
    res = pr.residuals(cal_block["y"], cal_block["p"])
    offs = pr.conformal_offsets(res)
    exp_m = rr.m1({"y": cal_block["y"], "p": cal_block["p"]}, ALPHA)
    assert abs(offs[0] - exp_m["lo"]) < 1e-12 and abs(offs[1] - exp_m["hi"]) < 1e-12, f"{name}: offsets differ from the experiment"
    d = {**test_block, "pm0": np.zeros(len(test_block["y"]))}
    lo_e, hi_e = rr.apply(exp_m, d, ALPHA)
    lo_u = np.array([pr.interval(p, *offs)[0] for p in d["p"]]); hi_u = np.array([pr.interval(p, *offs)[1] for p in d["p"]])
    assert np.allclose(lo_u, lo_e, atol=1e-9) and np.allclose(hi_u, hi_e, atol=1e-9), f"{name}: unrounded range differs from the experiment"
    s_exp = rr.score(d, lo_e, hi_e, ALPHA)["scalars"]
    lo_s, hi_s = shipped_range(d, offs)
    s_ship = rr.score(d, lo_s, hi_s, ALPHA)["scalars"]
    keys = ("pm25_coverage", "range_accuracy", "mean_categories_in_range", "mean_width_ugm3")
    for k in keys:                                  # the experiment's stored numbers must be reproduced exactly
        assert abs(s_exp[k] - expected[k]) < 1e-9, f"{name}: {k} {s_exp[k]} != stored {expected[k]}"
    return {"n_calibration_days": len(cal_block["y"]), "n_test_days": len(test_block["y"]),
            "calibration_last_day": max(cal_block["days"]).isoformat(), "test_first_day": min(test_block["days"]).isoformat(),
            "offsets": {"lo": offs[0], "hi": offs[1]},
            "experiment_M1_90": {k: s_exp[k] for k in keys} | {"matches_stored_results_json": True},
            "shipped_function_whole_number_outward": {k: s_ship[k] for k in keys},
            "point_forecast_unchanged": True}


def main():
    hist = osn.load_hist("data/daily_opencity.csv")
    live = fte.build_live_view(Path("data/raw"), extra_subs=("opencity_hourly",))
    era = lambda t: {k: hist[t].get(k) for k in WEATHER_KEYS}                       # noqa: E731
    src = rc.load_source("production_default")
    all_train = fte.build(hist, live, era, view="offline")
    test_live = fte.build(hist, live, lambda t, f=src[1]: f.get(t))                 # archived 24 h forecast weather
    R = json.loads((HERE / "results.json").read_text())

    out = {}
    for name, train_end, a, b in BLOCKS:
        days = sorted(t for t in test_live if a <= t <= b)
        train = {t: v for t, v in all_train.items() if t <= train_end}
        assert max(train) <= train_end and days and days[0] > train_end
        out[name] = oos_block(train, test_live, days)
        print(f"block {name}: trained through {train_end}, {len(days)} days {days[0]}..{days[-1]}", flush=True)

    # ---- reproduce the validated experiment with the shipped maths (calibration strictly before the test) ----
    train24 = {t: v for t, v in all_train.items() if t <= date(2024, 12, 31)}
    days25 = sorted(t for t in test_live if date(2025, 1, 1) <= t <= date(2025, 12, 31))
    full25 = oos_block(train24, test_live, days25)
    check = {"label": "RETROSPECTIVE: the shipped range maths reproduces the validated experiment (M1, 90 %)",
             "calendar_2025": reproduce("calendar_2025", out["A"], full25, R["tests"]["calendar_2025"]["methods"]["M1"]["90%"]),
             "winter_2025_26": reproduce("winter_2025_26", join(out["A"], out["B"]), out["C"], R["tests"]["winter_2025_26"]["methods"]["M1"]["90%"])}
    for k in ("calendar_2025", "winter_2025_26"):
        c = check[k]
        print(f"{k}: experiment {c['experiment_M1_90']['pm25_coverage']:.3f} cov / {c['experiment_M1_90']['mean_categories_in_range']:.2f} cats"
              f" | shipped {c['shipped_function_whole_number_outward']['pm25_coverage']:.3f} cov / "
              f"{c['shipped_function_whole_number_outward']['mean_categories_in_range']:.2f} cats", flush=True)

    # ---- the shipped calibration: all out-of-sample residuals, 2024-01-19 .. 2026-02-15 ----
    allb = join(out["A"], out["B"], out["C"])
    order = np.argsort([t.toordinal() for t in allb["days"]], kind="stable")
    allb = {"days": [allb["days"][i] for i in order], **{k: allb[k][order] for k in ("y", "v1", "v2", "hybrid", "p")}}
    assert len(set(allb["days"])) == len(allb["days"]), "a day appears in two calibration blocks"
    res_file = {"label": "RETROSPECTIVE out-of-sample residual inputs for core/range_calibration.json",
                "seed": SEED, "blocks": [{"block": n, "trained_through": te.isoformat(), "days": [a.isoformat(), b.isoformat()]} for n, te, a, b in BLOCKS],
                "dates": [t.isoformat() for t in allb["days"]], "actual_pm25": [float(x) for x in allb["y"]],
                "forecast_pm25": {v: [float(x) for x in allb[tag]] for v, tag in VERSIONS.items()}}
    versions = {}
    for v, tag in VERSIONS.items():
        res = pr.residuals(allb["y"], allb[tag])
        lo, hi = pr.conformal_offsets(res)
        versions[v] = {"n": len(res), "first": res_file["dates"][0], "last": res_file["dates"][-1], "lo": lo, "hi": hi,
                       "in_sample_coverage": float(np.mean([(r >= lo) and (r <= hi) for r in res])),
                       "inputs_sha256": hashlib.sha256(json.dumps([res_file["dates"], res_file["actual_pm25"], res_file["forecast_pm25"][v]]).encode()).hexdigest()}
    body = {"schema": pr.SCHEMA, "method": pr.METHOD, "level": pr.LEVEL,
            "residual": "log1p(actual PM2.5) - log1p(point forecast)",
            "rounding": "low end rounded down, high end rounded up, to whole ug/m3 (never narrower than computed)",
            "label": "RETROSPECTIVE calibration from past out-of-sample forecasts; a statement about past errors, not a guarantee",
            "forecast_inputs": "archived 24 h-ahead forecast weather (Open-Meteo Previous Runs, production_default); no ERA5 substitution for the target day",
            "windows": [{"block": n, "models_trained_through": te.isoformat(), "forecast_days": [a.isoformat(), b.isoformat()],
                         "n_days": len(out[n]["days"])} for n, te, a, b in BLOCKS],
            "versions": versions}
    cid = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    cal = {**body, "calibration_id": cid,
           "built_with": {"python": platform.python_version(), "numpy": np.__version__, "seed": SEED,
                          "builder": "experiments/prediction_range/build_calibration.py"}}
    OUT_CAL.write_text(json.dumps(cal, indent=1) + "\n")
    OUT_RES.write_text(json.dumps(res_file) + "\n")
    OUT_CHECK.write_text(json.dumps(fte.jsonable(check), indent=1) + "\n")
    print("wrote", OUT_CAL.relative_to(ROOT), OUT_RES.relative_to(ROOT), OUT_CHECK.relative_to(ROOT), "id", cid)
    for v, e in versions.items():
        print(f"  {v:16} n={e['n']} offsets lo={e['lo']:+.4f} hi={e['hi']:+.4f}  in-sample coverage {e['in_sample_coverage']:.3f}")


if __name__ == "__main__":
    main()
