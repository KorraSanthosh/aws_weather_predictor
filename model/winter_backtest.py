"""Fair historical winter back-test of v1_25_baseline, v2_35_candidate and the v3_hybrid router.

    python model/winter_backtest.py                 # writes experiments/winter_2024_25_backtest.{json,md}

This is a RETROSPECTIVE back-test.  Winters 2023-24, 2024-25 and 2025-26 were looked at in earlier experiments
(EXPERIMENTS.md) and 2019-2022 were used to CHOOSE the features / hyper-parameters, so none of this is an untouched
hold-out.  The only clean test is the prospective winter 2026-27 shadow test (model/EVALUATION_PROTOCOL.md).

Everything below is fixed in PROTOCOL BEFORE any result is looked at; nothing is tuned on the test winter:

  * Frozen fit.  For a winter Y (15 Oct Y -> 15 Feb Y+1) each model is trained ONCE on forecast origins whose TARGET
    date is before 15 Oct Y, using the unchanged production recipe (XGBoost on the log-change from today, 800 trees at
    lr 0.03, depth 4, early stopping on the last 60 training rows).  It is then frozen: no refit inside the window,
    nothing from the window enters training, early-stopping or any choice.  The stored artifacts (trained through
    2026-10-07) cannot be used for this - they have seen every winter - so frozen copies are re-trained here.
    They are kept in memory only; no model file is written, no existing model file is touched.
  * Same eligible dates for every model (the "eligibility ladder"): target-day city PM2.5, origin-day city PM2.5,
    target-day and origin-day weather, and an origin the 23:00 IST Lambda could actually use.
  * No imputation of hourly data.  Day-part features that cannot be computed stay NaN (XGBoost handles NaN natively);
    the hybrid routes those days to v1.
  * Weather.  The live Lambda receives a weather FORECAST for tomorrow.  Archived forecasts are not available (the
    Open-Meteo historical-forecast host is unreachable from the build sandbox and no archive is stored), so:
        PRIMARY      (leak-free)   tomorrow's weather := TODAY's observed weather ("weather persistence", a no-skill
                                   forecast; usually but not always pessimistic - see experiments/option_screen.md)
        DIAGNOSTIC   (NOT live)    tomorrow's weather := tomorrow's OBSERVED weather (optimistic - uses information
                                   the live system would not have)
        LEGACY       (NOT live)    observed weather and all 24 hours of "today" - what earlier experiments used,
                                   kept only so the numbers reconcile with EXPERIMENTS.md.
    Neither regime is presented as live performance (real archived forecasts: model/option_screen.py).
  * Time-aware uncertainty: moving-block bootstrap (overlapping blocks, 7 days; 14 days as a sensitivity) on the
    paired absolute-error difference.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE, ROOT / "pipeline"):
    sys.path.insert(0, str(p))

from compare_versions import build_live_view  # noqa: E402
from core.aqi import pm25_to_category  # noqa: E402
from core.daily import DP_KEYS  # noqa: E402
from core.features import FEATURE_SETS, feature_dict  # noqa: E402
from core.hybrid import daypart_valid, hybrid_pick  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from experiments import OLD_PARAMS, metrics  # noqa: E402
from train import load_daily  # noqa: E402

V1, V2, HYB = "v1_25_baseline", "v2_35_candidate", "v3_hybrid"
PERSIST, MEAN7 = "persistence", "mean7"
MODELS = [V1, V2, HYB, PERSIST, MEAN7]

PROTOCOL = {
    "status": "RETROSPECTIVE back-test - not an untouched hold-out",
    "window": "15 Oct Y -> 15 Feb Y+1 (target dates)",
    "primary_winter": 2024,
    "first_winter_considered": 2018,
    "min_training_origins": 365,        # a winter is not scored if the frozen fit would have fewer rows than this
    "min_scored_days": 30,              # ... or fewer than this many scored days
    "sudden_change_threshold": 0.30,    # |tomorrow - today| / today
    "bootstrap": {"kind": "moving-block (overlapping blocks) on the paired |error| difference",
                  "block_days": 7, "sensitivity_block_days": 14, "n_boot": 5000, "seed": 0, "level": 0.95},
    "refit_inside_window": False,
    "live_cutoff_hour_IST": 22,
    "model_recipe": OLD_PARAMS,
    "primary_regime": "live23h_wx_persistence",
    "regimes": {
        "live23h_wx_persistence": "PRIMARY: hours<=22 of today, tomorrow's weather := today's observed weather (leak-free)",
        "live23h_wx_observed": "DIAGNOSTIC (not live): hours<=22, tomorrow's OBSERVED weather",
        "offline_wx_observed": "LEGACY (not live): all 24 h, tomorrow's OBSERVED weather - as in earlier experiments",
    },
}
REGIMES = list(PROTOCOL["regimes"])
TRAIN_REGIME = "offline_wx_observed"      # production training recipe (stored artifacts are trained like this)


# ------------------------------------------------------------------ calendar helpers
def winter_window(year: int):
    return date(year, 10, 15), date(year + 1, 2, 15)


def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def compress_ranges(items):
    """[(date, key)] sorted -> [{"from", "to", "days", "reason"}] merging consecutive dates with the same key."""
    out = []
    for d, k in items:
        if out and out[-1]["_last"] + timedelta(days=1) == d and out[-1]["reason"] == k:
            out[-1]["_last"] = d
            out[-1]["days"] += 1
        else:
            out.append({"from": d, "_last": d, "days": 1, "reason": k})
    return [{"from": r["from"].isoformat(), "to": r["_last"].isoformat(), "days": r["days"], "reason": r["reason"]}
            for r in out]


# ------------------------------------------------------------------ raw station coverage (for the report only)
def station_day_counts(raw=Path("data/raw"), extra_subs=()):
    """{date: number of stations with a valid daily value} using the same per-station rules as the pipeline
    (>= 16 hourly readings, or a CPCB daily value).  Returns {} when the raw files are not present."""
    try:
        import build_dataset as bd
        from core.daily import readings_to_daily
    except Exception:  # pragma: no cover
        return {}, {}
    files: dict[str, list[Path]] = {}
    for sub in ("openaq", "openaq_api", "openaq_recent", *extra_subs):
        for p in sorted((raw / sub).glob("*.csv")):
            files.setdefault(p.stem, []).append(p)
    cpcb: dict[str, list[Path]] = {}
    for p in sorted((raw / "cpcb").glob("*.csv")):
        cpcb.setdefault(bd.cpcb_station_key(p), []).append(p)
    for sid in cpcb:
        files.setdefault(sid, [])
    counts: dict[date, int] = defaultdict(int)
    per_station: dict[str, dict[date, str]] = {}
    for sid, paths in files.items():
        days: dict[date, str] = {}
        readings = []
        for p in paths:
            readings.extend(bd.load_station(p))
        for d in readings_to_daily(readings):
            days[d] = "hourly"
        for p in cpcb.get(sid, []):
            for d in bd.load_cpcb_daily(p):
                days.setdefault(d, "daily-file")
        per_station[sid] = days
        for d in days:
            counts[d] += 1
    return dict(counts), per_station


# ------------------------------------------------------------------ eligibility ladder and feature rows
def _wx(rec):
    return {k: rec.get(k) for k in WEATHER_KEYS}


def classify_target(t, hist, live, counts=None):
    """None when target date t is scoreable, else (reason_key, detail).  Ladder order = order of the checks."""
    d0 = t - timedelta(days=1)
    nxt, cur = hist.get(t), hist.get(d0)
    n = (counts or {}).get(t)
    if not nxt or nxt.get("pm25") is None:
        detail = "no station reported" if not n else f"only {n} station(s) reported (< 3 needed for a city value)"
        return "1_no_city_pm25_on_target_day", detail
    if not cur or cur.get("pm25") is None:
        n0 = (counts or {}).get(d0)
        detail = "no station reported" if not n0 else f"only {n0} station(s) reported (< 3 needed for a city value)"
        return "2_no_city_pm25_on_origin_day", detail
    if any(nxt.get(k) is None for k in WEATHER_KEYS):
        return "3_target_day_weather_missing", ""
    if any(cur.get(k) is None for k in WEATHER_KEYS):
        return "4_origin_day_weather_missing", ""
    if d0 not in live:
        return "5_origin_not_usable_by_the_23h_lambda", ""
    return None


def build_rows(hist, live):
    """Feature rows for EVERY scoreable target date, in the three regimes.  Features for target t only read data
    <= t-1 (feature_dict contract) plus the weather chosen by the regime."""
    out = {}
    for t in sorted(hist):
        if classify_target(t, hist, live) is not None:
            continue
        d0 = t - timedelta(days=1)
        nxt, orig = hist[t], hist[d0]
        wx_obs, wx_per = _wx(nxt), _wx(orig)
        lrec = dict(orig)
        lrec["pm25"] = live[d0]["pm25"]
        for k in DP_KEYS:
            v = live[d0].get(k)
            lrec[k] = None if v is None or v != v else v
        rows = {"offline_wx_observed": feature_dict(hist, d0, wx_obs)}
        hist[d0] = lrec
        try:
            rows["live23h_wx_observed"] = feature_dict(hist, d0, wx_obs)
            rows["live23h_wx_persistence"] = feature_dict(hist, d0, wx_per)
        finally:
            hist[d0] = orig
        out[t] = {"y": float(nxt["pm25"]), "rows": rows}
    return out


def matrix(rows, names):
    return np.array([[r[n] for n in names] for r in rows], dtype=float)


# ------------------------------------------------------------------ frozen fit
def fit_frozen(X, y, version_names):
    """Train once.  X: matrix with column pm_d0; y: next-day PM2.5.  Early stopping uses only the LAST 60 training rows."""
    import xgboost as xgb
    i0 = version_names.index("pm_d0")
    delta = np.log1p(y) - np.log1p(X[:, i0])
    m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42,
                         early_stopping_rounds=40, n_jobs=4, **OLD_PARAMS)
    m.fit(X[:-60], delta[:-60], eval_set=[(X[-60:], delta[-60:])], verbose=False)
    return m


def predict_frozen(model, X, names):
    i0 = names.index("pm_d0")
    return np.expm1(np.log1p(X[:, i0]) + model.predict(X))


def split_before(targets, start):
    """Indices of training rows: target date strictly before the window start.  (Test rows never enter training.)"""
    return [i for i, t in enumerate(targets) if t < start]


# ------------------------------------------------------------------ moving-block bootstrap
def mbb_mae_diff(y, pa, pb, dates, block=7, n_boot=5000, seed=0, level=0.95):
    """MAE(B) - MAE(A) on the same days with a moving-block bootstrap interval.

    Days are laid on the calendar (missing days get weight 0) and overlapping blocks of `block` consecutive calendar
    days are resampled, so serial correlation and gaps are respected.  negative = B better than A."""
    y, pa, pb = (np.asarray(v, float) for v in (y, pa, pb))
    n = len(y)
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    res = {"n": int(n), "block_days": block, "mae_a": float(ea.mean()), "mae_b": float(eb.mean()),
           "diff": float((eb - ea).mean()), "pct": float((eb.mean() - ea.mean()) / ea.mean() * 100)}
    if np.array_equal(pa, pb):
        res.update(ci95_diff=[0.0, 0.0], ci95_pct=[0.0, 0.0], identical_predictions=True, excludes_zero=False)
        return res
    d0 = min(dates)
    span = (max(dates) - d0).days + 1
    pos = np.array([(d - d0).days for d in dates])
    A, B, W = np.zeros(span), np.zeros(span), np.zeros(span)
    A[pos], B[pos], W[pos] = ea, eb, 1.0
    L = max(1, min(block, span))
    nb = math.ceil(span / L)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, span - L + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :span]
    w = W[idx]
    sw = w.sum(1)
    ok = sw > 0
    sa, sb = (A[idx] * w).sum(1)[ok], (B[idx] * w).sum(1)[ok]
    diff = sb / sw[ok] - sa / sw[ok]
    pct = (sb - sa) / sa * 100
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    lo, hi = np.percentile(diff, q)
    plo, phi = np.percentile(pct, q)
    res.update(ci95_diff=[float(lo), float(hi)], ci95_pct=[float(plo), float(phi)], identical_predictions=False,
               excludes_zero=bool(lo > 0 or hi < 0))
    return res


# ------------------------------------------------------------------ scoring one winter
def sudden_mask(y, pm0, thr):
    return np.abs(y - pm0) / np.maximum(pm0, 1.0) >= thr


def _direction_acc(y, p, pm0):
    s_true, s_pred = np.sign(y - pm0), np.sign(p - pm0)
    return float(np.mean(s_true == s_pred)) if len(y) else None


def score_set(y, preds, pm0, dates, sudden_thr, boot):
    """Metrics for every model on THIS set of days + paired MAE differences with moving-block CIs."""
    out = {"n_days": int(len(y)), "models": {}, "paired": {}}
    if len(y) == 0:
        return out
    sd = sudden_mask(y, pm0, sudden_thr)
    for name, p in preds.items():
        m = metrics(y, p)
        m["mae_ug_m3"] = m.pop("mae")
        m["rmse_ug_m3"] = m.pop("rmse")
        if sd.any():
            ms = metrics(y[sd], p[sd])
            m["sudden_change"] = {"n": int(sd.sum()), "mae": ms["mae"], "rmse": ms["rmse"], "cat_acc": ms["cat_acc"],
                                  "within_one_cat": ms["within_one_cat"],
                                  "direction_acc": round(_direction_acc(y[sd], p[sd], pm0[sd]), 3)}
        else:
            m["sudden_change"] = {"n": 0}
        out["models"][name] = m
    pairs = [(V1, V2), (V1, HYB), (HYB, V2), (PERSIST, V1), (MEAN7, V1), (PERSIST, HYB), (MEAN7, HYB), (PERSIST, V2),
             (MEAN7, V2)]
    for a, b in pairs:
        key = f"{b}_minus_{a}"
        r = mbb_mae_diff(y, preds[a], preds[b], dates, block=boot["block_days"], n_boot=boot["n_boot"], seed=boot["seed"])
        r14 = mbb_mae_diff(y, preds[a], preds[b], dates, block=boot["sensitivity_block_days"], n_boot=boot["n_boot"],
                           seed=boot["seed"])
        r["ci95_diff_block14"], r["ci95_pct_block14"] = r14["ci95_diff"], r14["ci95_pct"]
        r["excludes_zero_block14"] = r14["excludes_zero"]
        out["paired"][key] = {k: (round(v, 3) if isinstance(v, float) else
                                  [round(x, 3) for x in v] if isinstance(v, list) else v) for k, v in r.items()}
    out["sudden_days_share"] = round(float(sd.mean()), 3)
    return out


def evaluate_winter(year, data, hist, live, counts=None, boot=None, proto=PROTOCOL, verbose=False):
    """Frozen-fit evaluation for the winter starting 15 Oct `year`.  `data` = build_rows output."""
    boot = boot or proto["bootstrap"]
    start, end = winter_window(year)
    res = {"winter": f"{year}-{str(year + 1)[2:]}", "window": [start.isoformat(), end.isoformat()],
           "calendar_days_in_window": (end - start).days + 1}
    # ---- eligibility ladder for this window
    excl, kept = [], []
    for t in daterange(start, end):
        c = classify_target(t, hist, live, counts)
        if c is None:
            kept.append(t)
        else:
            excl.append((t, f"{c[0]}: {c[1]}" if c[1] else c[0]))
    res["scored_days"] = len(kept)
    res["excluded_days"] = len(excl)
    res["exclusions"] = compress_ranges(excl)
    res["scored_first_last"] = [kept[0].isoformat(), kept[-1].isoformat()] if kept else None
    tr_t = sorted(t for t in data if t < start)
    res["training"] = {"origins": len(tr_t), "target_dates_through": tr_t[-1].isoformat() if tr_t else None,
                       "first_target": tr_t[0].isoformat() if tr_t else None, "refit_inside_window": False}
    if len(tr_t) < proto["min_training_origins"]:
        res["status"] = f"not run: only {len(tr_t)} training origins before {start} (< {proto['min_training_origins']})"
        return res
    if len(kept) < proto["min_scored_days"]:
        res["status"] = f"not scored: only {len(kept)} scoreable days in the window (< {proto['min_scored_days']})"
        return res
    res["status"] = "scored"

    # ---- freeze: ONE fit per model on rows with target < start; the window is never seen
    ytr = np.array([data[t]["y"] for t in tr_t])
    models = {}
    for v in (V1, V2):
        names = FEATURE_SETS[v]
        Xtr = matrix([data[t]["rows"][TRAIN_REGIME] for t in tr_t], names)
        models[v] = fit_frozen(Xtr, ytr, names)
    res["training"]["day_part_rows_in_training"] = int(sum(daypart_valid(data[t]["rows"][TRAIN_REGIME]) for t in tr_t))
    res["training"]["n_trees"] = {v: int(models[v].best_iteration + 1) for v in models}

    y = np.array([data[t]["y"] for t in kept])
    res["regimes"] = {}
    for reg in REGIMES:
        rows = [data[t]["rows"][reg] for t in kept]
        pm0 = np.array([r["pm_d0"] for r in rows])
        P = {v: predict_frozen(models[v], matrix(rows, FEATURE_SETS[v]), FEATURE_SETS[v]) for v in (V1, V2)}
        has_dp = np.array([daypart_valid(r) for r in rows])
        route = [hybrid_pick(r) for r in rows]
        P[HYB] = np.array([P[V2][i] if route[i] == V2 else P[V1][i] for i in range(len(rows))])
        P[PERSIST] = pm0.copy()
        P[MEAN7] = np.array([r["pm_mean7"] for r in rows])
        info = {"n_with_day_parts": int(has_dp.sum()), "n_without_day_parts": int((~has_dp).sum()),
                "hybrid_routed_to_v2": int(has_dp.sum()), "hybrid_routed_to_v1": int((~has_dp).sum())}
        dates = np.array(kept, dtype=object)
        sets = {"all_matched_days": np.ones(len(kept), bool), "days_with_day_parts": has_dp,
                "days_without_day_parts": ~has_dp}
        info["sets"] = {}
        for nm, mask in sets.items():
            idx = np.where(mask)[0]
            info["sets"][nm] = score_set(y[idx], {k: v[idx] for k, v in P.items()}, pm0[idx], list(dates[idx]),
                                         proto["sudden_change_threshold"], boot)
        info["daily_predictions"] = {k: [round(float(x), 2) for x in v] for k, v in P.items()}
        info["daily_has_day_parts"] = [bool(x) for x in has_dp]
        info["daily_pm_today"] = [round(float(x), 2) for x in pm0]
        res["regimes"][reg] = info
    res["daily_dates"] = [t.isoformat() for t in kept]
    res["daily_actual"] = [round(float(v), 2) for v in y]
    if verbose:
        print(f"  winter {res['winter']}: scored {len(kept)} d, train {len(tr_t)}")
    return res


# ------------------------------------------------------------------ whole experiment
def coverage_report(hist, live, counts, per_station, year, data=None):
    start, end = winter_window(year)
    days = list(daterange(start, end))
    has_pm = [d for d in days if hist.get(d) and hist[d].get("pm25") is not None]
    rep = {"window": [start.isoformat(), end.isoformat()], "calendar_days": len(days),
           "days_with_city_pm25": len(has_pm),
           "city_pm25_first_last": [has_pm[0].isoformat(), has_pm[-1].isoformat()] if has_pm else None,
           "scoreable_target_days": sum(1 for d in days if d in (data or {})),
           "scoreable_days_with_all_7_day_parts": sum(1 for d in days if d in (data or {}) and
                                                      daypart_valid(data[d]["rows"]["live23h_wx_observed"]))}
    if counts:
        by_month = defaultdict(lambda: defaultdict(int))
        for d in days:
            by_month[d.strftime("%Y-%m")][counts.get(d, 0)] += 1
        rep["stations_per_day_by_month"] = {m: dict(sorted(v.items())) for m, v in sorted(by_month.items())}
        rep["per_station"] = {}
        for sid, dd in sorted(per_station.items()):
            inw = {d: s for d, s in dd.items() if start <= d <= end}
            if inw:
                rep["per_station"][sid] = {"days": len(inw), "source": sorted(set(inw.values())),
                                           "first": min(inw).isoformat(), "last": max(inw).isoformat()}
    return rep


def pooled(winters, regime, proto=PROTOCOL):
    """All scored winters stacked (each from its own frozen fit).  Winters are placed 200 'days' apart on a synthetic
    timeline so a bootstrap block can never straddle two winters."""
    y, pm0, dp, dates, P = [], [], [], [], defaultdict(list)
    for k, w in enumerate((w for w in winters if w["status"] == "scored")):
        g = w["regimes"][regime]
        base = date.fromisoformat(w["daily_dates"][0]).toordinal()
        for i, ds in enumerate(w["daily_dates"]):
            dates.append(date.fromordinal(date.fromisoformat(ds).toordinal() - base + 1 + 200 * k + 1000))
        y += w["daily_actual"]
        pm0 += g["daily_pm_today"]
        dp += g["daily_has_day_parts"]
        for m, v in g["daily_predictions"].items():
            P[m] += v
    y, pm0, dp = np.array(y), np.array(pm0), np.array(dp, bool)
    P = {m: np.array(v) for m, v in P.items()}
    dates = np.array(dates, dtype=object)
    out = {"winters": [w["winter"] for w in winters if w["status"] == "scored"], "sets": {}}
    for nm, mask in (("all_matched_days", np.ones(len(y), bool)), ("days_with_day_parts", dp),
                     ("days_without_day_parts", ~dp)):
        idx = np.where(mask)[0]
        out["sets"][nm] = score_set(y[idx], {m: v[idx] for m, v in P.items()}, pm0[idx], list(dates[idx]),
                                    proto["sudden_change_threshold"], proto["bootstrap"])
    return out


def run(raw=Path("data/raw"), daily="data/daily.csv", proto=PROTOCOL, verbose=True, extra_subs=(), variant="production data"):
    hist, _ = load_daily(daily)
    live = build_live_view(raw, extra_subs=extra_subs) if raw.exists() else {d: {"pm25": r["pm25"], **{k: r.get(k) for k in DP_KEYS}}
                                                      for d, r in hist.items() if r["pm25"] is not None}
    counts, per_station = station_day_counts(raw, extra_subs) if raw.exists() else ({}, {})
    data = build_rows(hist, live)
    last = max(hist)
    winters = []
    for y in range(proto["first_winter_considered"], 2100):
        if winter_window(y)[0] > last:
            break
        winters.append(evaluate_winter(y, data, hist, live, counts, proto["bootstrap"], proto, verbose))
    primary = next(w for w in winters if w["winter"].startswith(str(proto["primary_winter"])))
    effect = None
    prod = Path("data/daily.csv")
    if Path(daily) != prod and prod.exists():
        h0, _ = load_daily(str(prod))
        both = [d for d in h0 if h0[d]["pm25"] is not None and hist.get(d) and hist[d]["pm25"] is not None]
        chg = [abs(hist[d]["pm25"] - h0[d]["pm25"]) for d in both if abs(hist[d]["pm25"] - h0[d]["pm25"]) > 1e-6]
        new = sorted(d for d in hist if hist[d]["pm25"] is not None and (d not in h0 or h0[d]["pm25"] is None))
        effect = {"production_daily_csv": str(prod), "days_with_pm25_production": sum(1 for r in h0.values() if r["pm25"] is not None),
                  "days_with_pm25_this_run": sum(1 for r in hist.values() if r["pm25"] is not None),
                  "new_city_days": len(new), "new_city_days_first_last": [new[0].isoformat(), new[-1].isoformat()] if new else None,
                  "existing_days_whose_value_changed": len(chg),
                  "mean_abs_change_ug_m3": round(float(np.mean(chg)), 2) if chg else 0.0,
                  "max_abs_change_ug_m3": round(float(np.max(chg)), 1) if chg else 0.0,
                  "why_values_change": "days now built from the 15-minute readings (>=16 h per station) instead of the CPCB daily file"}
    out = {"dataset_effect_vs_production": effect, "title": "Fair historical winter back-test: v1_25_baseline vs v2_35_candidate vs v3_hybrid",
           "data_variant": variant, "daily_csv": str(daily), "extra_station_sources": list(extra_subs),
           "protocol": {k: (v if k != "model_recipe" else dict(v)) for k, v in proto.items()},
           "status": proto["status"],
           "data": {"daily_csv_days_with_city_pm25": sum(1 for r in hist.values() if r["pm25"] is not None),
                    "last_date": last.isoformat(), "origins_eligible_total": len(data),
                    "origins_with_all_7_day_parts": int(sum(daypart_valid(v["rows"]["offline_wx_observed"])
                                                            for v in data.values()))},
           "coverage_primary_winter": coverage_report(hist, live, counts, per_station, proto["primary_winter"], data),
           "gapfill_report": (json.loads((raw / extra_subs[0] / "gapfill_report.json").read_text())
                              if extra_subs and (raw / extra_subs[0] / "gapfill_report.json").exists() else None),
           "winters": winters, "primary_winter": primary["winter"],
           "pooled_scored_winters": {reg: pooled(winters, reg, proto) for reg in REGIMES}}
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default="data/daily.csv")
    ap.add_argument("--extra-source", action="append", default=[], help="extra station folder under data/raw")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. opencity")
    a = ap.parse_args()
    os.chdir(ROOT)
    variant = "production data (data/daily.csv)" if not a.extra_source else \
        f"production data + gap-fill from {', '.join(a.extra_source)} ({a.daily})"
    res = run(daily=a.daily, extra_subs=tuple(a.extra_source), variant=variant)
    Path("experiments").mkdir(exist_ok=True)
    slim = json.loads(json.dumps(res, default=float))
    base = "experiments/winter_2024_25_backtest" + (f"_{a.tag}" if a.tag else "")
    Path(base + ".json").write_text(json.dumps(slim, indent=1))
    from winter_backtest_report import render
    alt_path = Path("experiments/winter_2024_25_backtest_opencity.json")
    alt = json.loads(alt_path.read_text()) if (not a.tag and alt_path.exists()) else None
    Path(base + ".md").write_text(render(slim, alt=alt))
    print(f"wrote {base}.json and .md")
    for w in res["winters"]:
        print(w["winter"], w["status"], "scored", w.get("scored_days"))
