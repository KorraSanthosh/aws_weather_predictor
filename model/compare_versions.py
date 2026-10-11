"""Fair, paired comparison of the two model versions (v1_25_baseline vs v2_35_candidate).

    python model/compare_versions.py            # writes experiments/model_comparison_v1_v2.json

NOTHING here is tuned: both feature sets are the fixed lists in core/features.py (FEATURE_SETS), both use the same
model recipe (XGBoost on the log-change from today, early stopping on the last 60 training rows), and every metric is
computed on IDENTICAL dates. The only decisions are about how to measure, and they are made up front:

  Walk-forward, monthly re-fit: each month of the test period is predicted by a model trained only on earlier days.
  Test period: 2023-01-01 -> latest day. NOTE this period was examined in earlier experiments (EXPERIMENTS.md), so it is
  NOT a clean hold-out; the clean test is the winter 2026-27 forward evaluation (see model/MODEL_VERSIONS.md).

Three set-ups are compared, because the training data and the live Lambda do not see the same "today":

  OFFLINE   features for day D use all 24 hours of D (what every earlier experiment used).
  LIVE      the nightly Lambda runs at 23:00 IST, so for day D only hours 0-22 exist. Both models are trained on the
            full-day features (the stored artifacts) but are fed what the Lambda would actually see.
  CAUSAL    both models are re-trained AND tested on hours 0-22 only, so training and serving match.

Day-part features exist only where hourly data exist (2018-2022 and Feb 2025 onward; 2023-24 are CPCB daily files).
They are left as NaN (missing) elsewhere - never filled in. Results are therefore reported separately for
  * the full evaluation period,
  * HOURLY days (target date >= 2025-03-01 with day-part data) - the 523-day subset behind the 70.4% figure,
  * days WITHOUT day-parts (v2 runs with its 7 day-part features missing).
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pipeline"))

import build_dataset as bd  # noqa: E402
from core.daily import (DP_KEYS, city_daily_mean, city_hourly, daypart_stats, readings_to_daily,  # noqa: E402
                        readings_to_hourly, to_ist)
from core.features import FEATURE_SETS, feature_dict  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from experiments import OLD_PARAMS, metrics, months, season_of, SEASON_NAMES  # noqa: E402
from train import load_daily  # noqa: E402

TEST_START = date(2023, 1, 1)
HOURLY_START = date(2025, 3, 1)
LIVE_CUTOFF_HOUR = 22          # last hour that has completed when the Lambda runs at 23:00 IST
SUDDEN = 0.30                  # |tomorrow - today| / today >= 30%
V1, V2 = "v1_25_baseline", "v2_35_candidate"


# ------------------------------------------------------------------ what the Lambda sees at 23:00 IST
def build_live_view(raw=Path("data/raw"), cutoff=LIVE_CUTOFF_HOUR, min_stations=3, extra_subs=()):
    """{date: {"pm25", **dp_*}} computed exactly like pipeline/build_dataset.py but from readings up to `cutoff` hour."""
    station_files: dict[str, list[Path]] = {}
    for sub in ("openaq", "openaq_api", "openaq_recent", *extra_subs):
        for p in sorted((raw / sub).glob("*.csv")):
            station_files.setdefault(p.stem, []).append(p)
    cpcb: dict[str, list[Path]] = {}
    for p in sorted((raw / "cpcb").glob("*.csv")):
        cpcb.setdefault(bd.cpcb_station_key(p), []).append(p)
    for sid in cpcb:
        station_files.setdefault(sid, [])
    per_daily, per_hourly = [], []
    for sid, paths in station_files.items():
        readings = []
        for p in paths:
            readings.extend(bd.load_station(p))
        cut = [(ts, v) for ts, v in readings if to_ist(ts).hour <= cutoff]
        daily = readings_to_daily(cut)
        per_hourly.append(readings_to_hourly(cut))
        for p in cpcb.get(sid, []):                      # CPCB daily files have no hours: cannot be truncated
            for day, val in bd.load_cpcb_daily(p).items():
                daily.setdefault(day, val)
        per_daily.append(daily)
    city = city_daily_mean(per_daily, min_stations=min_stations)
    hourly = city_hourly(per_hourly)
    out = {}
    for d, (pm, n) in city.items():
        rec = {"pm25": pm, "n": n}
        rec.update(daypart_stats(hourly.get(d), max_hour=cutoff))
        out[d] = rec
    return out


# ------------------------------------------------------------------ feature matrices
def build_rows(hist, live, only_live_valid=True):
    """One row per forecast origin D (target D+1). Returns dicts for OFFLINE and LIVE views of "today"."""
    rows_off, rows_live, y, targets, d0s = [], [], [], [], []
    dropped = 0
    for d0 in sorted(hist):
        d1 = d0 + timedelta(days=1)
        nxt, cur = hist.get(d1), hist[d0]
        if not nxt or nxt["pm25"] is None or cur["pm25"] is None:
            continue
        if any(nxt[k] is None for k in WEATHER_KEYS):
            continue
        if only_live_valid and d0 not in live:
            dropped += 1                                  # Lambda could not forecast from this origin at 23:00
            continue
        wx1 = {k: nxt[k] for k in WEATHER_KEYS}
        rows_off.append(feature_dict(hist, d0, wx1))
        orig = hist[d0]
        lrec = dict(orig)
        lrec["pm25"] = live[d0]["pm25"]
        for k in DP_KEYS:
            lrec[k] = None if live[d0][k] != live[d0][k] else live[d0][k]
        hist[d0] = lrec
        try:
            rows_live.append(feature_dict(hist, d0, wx1))
        finally:
            hist[d0] = orig
        y.append(nxt["pm25"]); targets.append(d1); d0s.append(d0)
    return rows_off, rows_live, np.array(y), targets, d0s, dropped


def matrix(rows, names):
    return np.array([[r[n] for n in names] for r in rows], dtype=float)


# ------------------------------------------------------------------ walk-forward (train matrix and test matrix may differ)
def walk(Xtr, Xte, y, targets, names, start, end):
    import xgboost as xgb
    i0 = names.index("pm_d0")
    delta_tr = np.log1p(y) - np.log1p(Xtr[:, i0])
    da = np.array(targets)
    out = {}
    for m0 in months(start, end):
        m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
        te = np.where((da >= m0) & (da < m1) & (da >= start) & (da <= end))[0]
        tr = np.where(da < m0)[0]
        if len(te) == 0 or len(tr) < 150:
            continue
        model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42,
                                 early_stopping_rounds=40, n_jobs=4, **OLD_PARAMS)
        model.fit(Xtr[tr[:-60]], delta_tr[tr[:-60]], eval_set=[(Xtr[tr[-60:]], delta_tr[tr[-60:]])], verbose=False)
        pred = np.expm1(np.log1p(Xte[te, i0]) + model.predict(Xte[te]))
        for i, p in zip(te, pred):
            out[i] = float(p)
    return out


# ------------------------------------------------------------------ statistics
def paired_mae_change(y, pa, pb, dates, n_boot=2000, seed=0):
    """% change in MAE going from model A to model B (negative = B better) with a 95% interval from a
    7-day block bootstrap (neighbouring days are strongly correlated, so single days would be over-confident)."""
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    est = (eb.mean() - ea.mean()) / ea.mean() * 100
    blocks = {}
    for i, d in enumerate(dates):
        blocks.setdefault(d.toordinal() // 7, []).append(i)
    blocks = [np.array(v) for v in blocks.values()]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ix = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        vals.append((eb[ix].mean() - ea[ix].mean()) / ea[ix].mean() * 100)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"pct": round(float(est), 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def in_winter(d):
    return (d.month, d.day) >= (10, 15) or (d.month, d.day) <= (2, 15)


def main():
    hist, _ = load_daily("data/daily.csv")
    live = build_live_view()
    rows_off, rows_live, y, targets, d0s, dropped = build_rows(hist, live)
    end = max(targets)
    n_all = len(y)
    v1, v2 = FEATURE_SETS[V1], FEATURE_SETS[V2]
    has_dp = np.array([not np.isnan(r["pm_school"]) and not np.isnan(r["pm_late"]) for r in rows_off])
    has_dp_live = np.array([not np.isnan(r["pm_school"]) and not np.isnan(r["pm_late"]) for r in rows_live])
    print(f"forecast origins: {n_all} (dropped {dropped} origins the 23:00 Lambda could not use) | "
          f"with day-parts: offline {int(has_dp.sum())}, live-view {int(has_dp_live.sum())}")

    X = {(v, view): matrix(rows, names) for v, names in ((V1, v1), (V2, v2))
         for view, rows in (("off", rows_off), ("live", rows_live))}
    P = {}
    for label, (v, names, train_view, test_view) in {
        "v1_offline": (V1, v1, "off", "off"), "v2_offline": (V2, v2, "off", "off"),
        "v1_live":    (V1, v1, "off", "live"), "v2_live":    (V2, v2, "off", "live"),
        "v1_causal":  (V1, v1, "live", "live"), "v2_causal":  (V2, v2, "live", "live"),
    }.items():
        P[label] = walk(X[(v, train_view)], X[(v, test_view)], y, targets, names, TEST_START, end)
        print(f"  done {label}: {len(P[label])} predictions")

    ids_all = sorted(set.intersection(*[set(p) for p in P.values()]))
    pm0_off = np.array([r["pm_d0"] for r in rows_off])
    pm0_live = np.array([r["pm_d0"] for r in rows_live])
    change = np.abs(y - pm0_off) / np.maximum(pm0_off, 1)

    def block(ids, regime):
        ids = sorted(ids)
        if not ids:
            return None
        yy, dts = y[ids], [targets[i] for i in ids]
        if regime == "offline":
            models = {"persistence": pm0_off[ids], "v1_baseline": [P["v1_offline"][i] for i in ids],
                      "v2_candidate": [P["v2_offline"][i] for i in ids]}
            a, b = np.array(models["v1_baseline"]), np.array(models["v2_candidate"])
        elif regime == "live":
            models = {"persistence": pm0_live[ids], "v1_baseline": [P["v1_live"][i] for i in ids],
                      "v2_candidate": [P["v2_live"][i] for i in ids]}
            a, b = np.array(models["v1_baseline"]), np.array(models["v2_candidate"])
        else:
            models = {"persistence": pm0_live[ids], "v1_baseline": [P["v1_causal"][i] for i in ids],
                      "v2_candidate": [P["v2_causal"][i] for i in ids]}
            a, b = np.array(models["v1_baseline"]), np.array(models["v2_candidate"])
        res = {"n_days": len(ids), "n_with_daypart_data": int(sum(has_dp_live[i] for i in ids)),
               "first": min(dts).isoformat(), "last": max(dts).isoformat()}
        for k, v in models.items():
            res[k] = metrics(yy, np.asarray(v, float))
        res["v2_vs_v1_mae_change"] = paired_mae_change(yy, a, b, dts)
        res["v1_vs_persistence_mae_change"] = paired_mae_change(yy, np.asarray(models["persistence"], float), a, dts)
        return res

    hourly_ids = [i for i in ids_all if targets[i] >= HOURLY_START and has_dp_live[i]]
    nodp_ids = [i for i in ids_all if not has_dp_live[i]]
    subsets = {
        "full_period_2023+": ids_all,
        "hourly_days_2025-03+": hourly_ids,
        "days_without_daypart_data": nodp_ids,
        "winter_15Oct-15Feb_all": [i for i in ids_all if in_winter(targets[i])],
        "winter_hourly_days": [i for i in hourly_ids if in_winter(targets[i])],
        "sudden_change_days_full_period": [i for i in ids_all if change[i] >= SUDDEN],
        "sudden_change_days_hourly": [i for i in hourly_ids if change[i] >= SUDDEN],
    }
    for s in range(4):
        subsets[f"season_{SEASON_NAMES[s]}_full_period"] = [i for i in ids_all if season_of(targets[i]) == s]
        subsets[f"season_{SEASON_NAMES[s]}_hourly"] = [i for i in hourly_ids if season_of(targets[i]) == s]
    out = {
        "protocol": __doc__.strip().split("\n\n")[0],
        "live_cutoff_hour_IST": LIVE_CUTOFF_HOUR,
        "test_period": [TEST_START.isoformat(), end.isoformat()],
        "test_period_status": "ALREADY EXAMINED in earlier experiments - not a clean hold-out",
        "feature_sets": {V1: len(v1), V2: len(v2)},
        "origins_total": n_all, "origins_dropped_not_usable_at_23h": dropped,
        "test_days_common_to_all_models": len(ids_all),
        "coverage": {"days_with_daypart_data": len(hourly_ids) + sum(1 for i in ids_all if has_dp_live[i] and targets[i] < HOURLY_START),
                     "hourly_days_2025-03+": len(hourly_ids), "days_without_daypart_data": len(nodp_ids),
                     "v2_features_missing_on_no_daypart_days": [f for f in v2 if f in DP_KEYS or f.startswith("pm_") and f in
                                                                ("pm_night", "pm_commute", "pm_school", "pm_evening", "pm_late",
                                                                 "pm_buildup", "pm_last3h_ratio")]},
    }
    for regime in ("offline", "live", "causal"):
        out[regime] = {name: block(ids, regime) for name, ids in subsets.items()}

    # how much does the 23:00 cutoff move the CURRENT v2 artifact's own forecasts? (same trained models, two inputs)
    d = np.array([P["v2_live"][i] - P["v2_offline"][i] for i in hourly_ids])
    d1 = np.array([P["v1_live"][i] - P["v1_offline"][i] for i in ids_all])
    cat = lambda v: np.array([metrics(np.array([x]), np.array([x]))["n"] for x in v])  # noqa: E731
    from core.aqi import pm25_to_category
    flips = int(sum(pm25_to_category(P["v2_live"][i]) != pm25_to_category(P["v2_offline"][i]) for i in hourly_ids))
    flips1 = int(sum(pm25_to_category(P["v1_live"][i]) != pm25_to_category(P["v1_offline"][i]) for i in ids_all))
    out["cutoff_effect_on_forecasts"] = {
        "v2_hourly_days": {"n": len(hourly_ids), "mean_abs_change_ug_m3": round(float(np.abs(d).mean()), 2),
                           "mean_signed_change": round(float(d.mean()), 2), "category_flips": flips,
                           "category_flip_pct": round(100 * flips / len(hourly_ids), 1)},
        "v1_full_period": {"n": len(ids_all), "mean_abs_change_ug_m3": round(float(np.abs(d1).mean()), 2),
                           "mean_signed_change": round(float(d1.mean()), 2), "category_flips": flips1,
                           "category_flip_pct": round(100 * flips1 / len(ids_all), 1)}}
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/model_comparison_v1_v2.json").write_text(json.dumps(out, indent=1, default=float))

    def line(tag, r):
        if not r:
            return f"  {tag}: no days"
        a, b, p = r["v1_baseline"], r["v2_candidate"], r["persistence"]
        return (f"  {tag:34s} n={r['n_days']:4d} | MAE pers {p['mae']:6.2f} v1 {a['mae']:6.2f} v2 {b['mae']:6.2f} | "
                f"cat pers {p['cat_acc']:.3f} v1 {a['cat_acc']:.3f} v2 {b['cat_acc']:.3f} | "
                f"R2 v1 {a['r2']} v2 {b['r2']} | v2 vs v1 MAE {r['v2_vs_v1_mae_change']['pct']:+.1f}% {r['v2_vs_v1_mae_change']['ci95']}")

    for regime in ("offline", "live", "causal"):
        print(f"\n===== {regime.upper()} =====")
        for name in ("full_period_2023+", "hourly_days_2025-03+", "days_without_daypart_data", "winter_hourly_days",
                     "sudden_change_days_full_period", "sudden_change_days_hourly"):
            print(line(name, out[regime][name]))
    print("\nCutoff effect on forecasts:", json.dumps(out["cutoff_effect_on_forecasts"]))
    print("wrote experiments/model_comparison_v1_v2.json")


if __name__ == "__main__":
    main()
