"""Controlled experiment: does training on ARCHIVED WEATHER FORECASTS (instead of observed/ERA5 weather) help?

    python model/forecast_training_experiment.py
    -> experiments/forecast_training_experiment.json, experiments/forecast_training_experiment.md

Needs data/daily_opencity.csv, data/raw/opencity_hourly/ and data/raw/weather_prevruns_2024_2026.json
(Open-Meteo Previous Runs API, see pipeline/fetch_prevruns.py).

RETROSPECTIVE.  Winters 2024-25 and 2025-26 were examined in earlier experiments, so they are not an untouched hold-out.
The untouched final test is the prospective winter 2026-27 shadow test; this script refuses to touch any date after
15 Feb 2026 and never reads a model file.  Everything below was fixed BEFORE the results were looked at.

Weather inputs.  At 23:00 IST on day D the Lambda forecasts day D+1 from the latest weather run.  The archive holds the
value each hour of D+1 had in the forecast made 1 day earlier ("previous_day1", lead ~24 h) and 2 days earlier
("previous_day2", ~48 h).  previous_day1 therefore uses information that is the same age or OLDER than the live run
(live leads are ~7-30 h), so it can not leak and is, if anything, slightly pessimistic.  The exact live issue time
can not be reproduced from the archive; that is a stated limitation, not hidden.  Today's weather (wx0) is not
archived as an analysis, so the main arms keep ERA5 for wx0; arm S1 stress-tests that choice.

Arms (each changes the weather source in TRAINING and/or at TEST time; recipe, features and seed are identical):
  A0  reference      train: observed (ERA5) weather                  test: archived 24 h forecast
  A1  forecast-only  train: ONLY origins that have an archived forecast  test: archived 24 h forecast
                     (never substitutes observed weather; not run if < 365 training origins remain)
  A2  mixed          train: archived forecast where it exists, ERA5 elsewhere (counts reported)  test: forecast
  A3  calibrated     train: ERA5                                      test: forecast mapped to ERA5 by a per-variable
                     linear fit estimated on days BEFORE the window only
  S1  sensitivity    like A2 but today's weather (wx0) is also the archived forecast (stress test, not a candidate)
  REF oracle         train: ERA5  test: observed tomorrow (what the old numbers assumed - an upper bound, not live)
  REF 48 h           train: ERA5  test: archived 48 h forecast
Models: v1_25_baseline, v2_35_candidate, hybrid (V2 iff all 7 day-part inputs exist, else V1) - unchanged recipe,
frozen fit before each winter (target < 15 Oct Y), no refit inside the window, seed 42 (seed spread reported).
Baselines: persistence (tomorrow := today) and trailing 7-day mean.
Eligible days: the same for every arm - eligibility ladder AND archived 24 h forecasts for tomorrow and for today.

Decision rule for a candidate arm (A1-A3), written before running: Hybrid pooled MAE >= 2% lower than A0, moving-block
95% interval of the difference below zero for BOTH 7- and 14-day blocks, and no single winter more than 2% worse.
Otherwise: no evidence of improvement, keep A0.  Three candidates are tested; no multiplicity correction is applied,
so a pass would still only be a hypothesis for the winter 2026-27 shadow test.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE, ROOT / "pipeline"):
    sys.path.insert(0, str(p))

import option_screen as osn  # noqa: E402
import winter_backtest as wb  # noqa: E402
from compare_versions import build_live_view  # noqa: E402
from core.aqi import pm25_to_category  # noqa: E402
from core.features import FEATURE_SETS, feature_dict  # noqa: E402
from core.hybrid import daypart_valid  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from experiments import OLD_PARAMS  # noqa: E402

V1, V2 = "v1_25_baseline", "v2_35_candidate"
MODELS = ("v1", "v2", "hybrid")
BASELINES = ("persistence", "mean7")
WINTERS = (2024, 2025)                     # the only winters with archived forecasts (archive starts 2024-01-20)
HARD_STOP = date(2026, 2, 15)              # nothing after this is read; winter 2026-27 stays untouched
MIN_TRAIN = 365
SEED = 42
SEEDS = (42, 7, 123, 2024, 99)
SUDDEN = 0.30
BOOT = dict(n_boot=5000, seed=0)
RULE = {"min_gain_pct": 2.0, "max_single_winter_loss_pct": 2.0}
CANDIDATES = ("A1_forecast_only", "A2_mixed", "A3_calibrated_input")
ARMS = {
    "A0_era5_trained": ("obs", "fc"),
    "A1_forecast_only": ("fc_strict", "fc"),
    "A2_mixed": ("mixed", "fc"),
    "A3_calibrated_input": ("obs", "fc_cal"),
    "S1_mixed_wx0_forecast": ("mixed_both", "fc_both"),
    "REF_oracle_observed_tomorrow": ("obs", "obs"),
    "REF_forecast_48h": ("obs", "fc48"),
}
BOUNDS = {"precip": (0.0, None), "wind": (0.0, None), "rh": (0.0, 100.0)}


# ------------------------------------------------------------------ rows
def complete(wx):
    return wx is not None and all(wx.get(k) is not None for k in WEATHER_KEYS)


def build(hist, live, wx1_for, wx0_for=None, view="live"):
    """{t: {"y","row"}} for every scoreable t whose tomorrow weather (and, if given, today's weather) exists.
    Never substitutes: a day without the requested weather is simply absent."""
    out = {}
    for t in sorted(hist):
        if t >= date(2026, 10, 15):
            continue                                              # winter 2026-27 is never touched
        if wb.classify_target(t, hist, live) is not None:
            continue
        d0 = t - timedelta(days=1)
        wx = wx1_for(t)
        if not complete(wx):
            continue
        rec = osn._live_rec(hist, live, d0) if view == "live" else dict(hist[d0])
        if wx0_for is not None:
            w0 = wx0_for(d0)
            if not complete(w0):
                continue
            rec.update({k: w0[k] for k in WEATHER_KEYS})
        orig = hist[d0]
        hist[d0] = rec
        try:
            row = feature_dict(hist, d0, wx)
        finally:
            hist[d0] = orig
        out[t] = {"y": float(hist[t]["pm25"]), "row": row}
    return out


def fit_calibration(hist, fc1, before):
    """Per variable OLS  ERA5 = a + b * forecast  on days strictly before `before`.  Returns {key: (a, b, n)}."""
    cal = {}
    days = [t for t in fc1 if t < before and t in hist and complete(hist[t]) and complete(fc1[t])]
    for k in WEATHER_KEYS:
        x = np.array([fc1[t][k] for t in days])
        z = np.array([hist[t][k] for t in days])
        if len(days) < 30 or x.std() < 1e-9:
            cal[k] = (0.0, 1.0, len(days))
            continue
        b, a = np.polyfit(x, z, 1)
        cal[k] = (float(a), float(b), len(days))
    return cal


def apply_calibration(wx, cal):
    out = {}
    for k in WEATHER_KEYS:
        a, b, _ = cal[k]
        v = a + b * wx[k]
        lo, hi = BOUNDS.get(k, (None, None))
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        out[k] = v
    return out


# ------------------------------------------------------------------ models
def fit_models(train, seed=SEED):
    import xgboost as xgb
    ts = sorted(train)
    y = np.array([train[t]["y"] for t in ts])
    models = {}
    for tag, v in (("v1", V1), ("v2", V2)):
        names = FEATURE_SETS[v]
        X = osn.M([train[t]["row"] for t in ts], names)
        delta = np.log1p(y) - np.log1p(X[:, names.index("pm_d0")])
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed,
                             early_stopping_rounds=40, n_jobs=4, **OLD_PARAMS)
        m.fit(X[:-60], delta[:-60], eval_set=[(X[-60:], delta[-60:])], verbose=False)
        models[tag] = m
    return models


def predict_all(models, rows):
    out = {}
    for tag, v in (("v1", V1), ("v2", V2)):
        names = FEATURE_SETS[v]
        out[tag] = osn.predict(models[tag], osn.M(rows, names), names)
    out["hybrid"] = osn.hybrid(out["v1"], out["v2"], rows)
    out["persistence"] = np.array([r["pm_d0"] for r in rows], float)
    out["mean7"] = np.array([r["pm_mean7"] for r in rows], float)
    return out


# ------------------------------------------------------------------ statistics (moving-block bootstrap, shared draws)
_WCACHE: dict = {}


def boot_weights(dates, block, n_boot=BOOT["n_boot"], seed=BOOT["seed"]):
    key = (tuple(dates), block, n_boot, seed)
    if key not in _WCACHE:
        if len(_WCACHE) > 64:
            _WCACHE.clear()
        _WCACHE[key] = _boot_weights(dates, block, n_boot, seed)
    return _WCACHE[key]


def _boot_weights(dates, block, n_boot, seed):
    """Observation weights (n_boot x n) of overlapping `block`-day blocks on the calendar (same draws as
    winter_backtest.mbb_mae_diff, so every statistic and every comparison shares one set of resamples)."""
    d0 = min(dates)
    span = (max(dates) - d0).days + 1
    pos = np.array([(d - d0).days for d in dates])
    L = max(1, min(block, span))
    nb = int(np.ceil(span / L))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, span - L + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :span]
    cnt = np.zeros((n_boot, span))
    np.add.at(cnt, (np.repeat(np.arange(n_boot), idx.shape[1]), idx.ravel()), 1.0)
    return cnt[:, pos]


def _ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


class Stat:
    """Point estimates and block-bootstrap intervals for one set of days (y, predictions, dates)."""

    def __init__(self, y, preds, dates, pm0):
        self.y, self.P, self.dates, self.pm0 = np.asarray(y, float), preds, list(dates), np.asarray(pm0, float)
        self.n = len(self.y)
        self.W = {b: boot_weights(self.dates, b) for b in (7, 14)} if self.n >= 2 else {}
        self.cat_true = np.array([pm25_to_category(v) for v in self.y], dtype=object)

    def _hit(self, p):
        return np.array([pm25_to_category(v) == c for v, c in zip(p, self.cat_true)], float)

    def per_day(self, name, what):
        p = self.P[name]
        e = p - self.y
        return {"mae": np.abs(e), "mse": e ** 2, "cat": self._hit(p)}[what]

    def _boot_mean(self, v, block):
        W = self.W[block]
        sw = W.sum(1)
        ok = sw > 0
        return (W[ok] @ v) / sw[ok]

    def single(self, name):
        out = {}
        for what, tf in (("mae", lambda m: m), ("mse", np.sqrt), ("cat", lambda m: m)):
            v = self.per_day(name, what)
            key = {"mae": "mae", "mse": "rmse", "cat": "cat_acc"}[what]
            out[key] = float(tf(v.mean()))
            if self.n >= 2:
                out[key + "_ci95"] = _ci(tf(self._boot_mean(v, 7)))
        return out

    def vs_persistence(self, name):
        """(MAE_persistence - MAE_model) / MAE_persistence in %, positive = better than persistence."""
        em, ep = self.per_day(name, "mae"), self.per_day("persistence", "mae")
        pt = float((1 - em.mean() / ep.mean()) * 100)
        if self.n < 2:
            return {"pct": pt}
        bm, bp = self._boot_mean(em, 7), self._boot_mean(ep, 7)
        return {"pct": pt, "ci95": _ci((1 - bm / bp) * 100)}

    def diff(self, a, b):
        """metric(b) - metric(a); negative MAE/RMSE = b better, positive cat = b better."""
        out = {}
        ea, eb = self.per_day(a, "mae"), self.per_day(b, "mae")
        out["mae"] = float(eb.mean() - ea.mean())
        out["mae_pct"] = float((eb.mean() - ea.mean()) / ea.mean() * 100)
        out["rmse"] = float(np.sqrt(self.per_day(b, "mse").mean()) - np.sqrt(self.per_day(a, "mse").mean()))
        out["cat_acc"] = float(self.per_day(b, "cat").mean() - self.per_day(a, "cat").mean())
        out["identical"] = bool(np.array_equal(self.P[a], self.P[b]))
        if self.n >= 2 and not out["identical"]:
            for blk in (7, 14):
                d = self._boot_mean(eb, blk) - self._boot_mean(ea, blk)
                out[f"mae_ci95_{blk}d"] = _ci(d)
            out["mae_pct_ci95_7d"] = _ci((self._boot_mean(eb, 7) / self._boot_mean(ea, 7) - 1) * 100)
            out["rmse_ci95_7d"] = _ci(np.sqrt(self._boot_mean(self.per_day(b, "mse"), 7)) -
                                      np.sqrt(self._boot_mean(self.per_day(a, "mse"), 7)))
            out["cat_acc_ci95_7d"] = _ci(self._boot_mean(self.per_day(b, "cat"), 7) - self._boot_mean(self.per_day(a, "cat"), 7))
            out["excludes_zero_both"] = bool(all(out[f"mae_ci95_{k}d"][1] < 0 or out[f"mae_ci95_{k}d"][0] > 0 for k in (7, 14)))
        else:
            out["excludes_zero_both"] = False
        return out


def stat_block(y, preds, dates, pm0, ref_arm_preds=None):
    """Full table for one set of days: every model's metrics, vs persistence, and (optionally) vs the same model in the
    reference arm."""
    st = Stat(y, preds, dates, pm0)
    out = {"n_days": st.n, "models": {}}
    for m in MODELS + BASELINES:
        out["models"][m] = st.single(m)
        if m in MODELS:
            out["models"][m]["vs_persistence"] = st.vs_persistence(m)
    out["model_vs_model"] = {"v2_minus_v1": st.diff("v1", "v2"), "hybrid_minus_v1": st.diff("v1", "hybrid")}
    return out, st


def diff_vs_ref(y, preds, ref, dates, pm0):
    """Same-model difference (arm minus reference arm) on the same days, per model."""
    both = {f"{m}__arm": preds[m] for m in MODELS}
    both.update({f"{m}__ref": ref[m] for m in MODELS})
    st = Stat(y, both, dates, pm0)
    return {m: st.diff(f"{m}__ref", f"{m}__arm") for m in MODELS}


# ------------------------------------------------------------------ the experiment
def run(daily="data/daily_opencity.csv", raw=Path("data/raw"), extra=("opencity_hourly",),
        fc_path="data/raw/weather_prevruns_2024_2026.json", verbose=True):
    hist = osn.load_hist(daily)
    live = build_live_view(raw, extra_subs=extra)
    fc = osn.load_forecasts(fc_path)
    fc1, fc2 = fc[1], fc[2]
    era = lambda t: {k: hist[t].get(k) for k in WEATHER_KEYS}                       # noqa: E731
    res = {"protocol": __doc__.strip(), "rule": RULE, "forecast_archive": {
        "path": fc_path, "previous_day1_days": len(fc1), "previous_day1_first_last": [min(fc1).isoformat(), max(fc1).isoformat()],
        "previous_day2_days": len(fc2), "previous_day2_first_last": [min(fc2).isoformat(), max(fc2).isoformat()],
        "winters_with_archived_forecast_test_days": [f"{Y}-{str(Y + 1)[2:]}" for Y in WINTERS],
        "winters_without": "2018-19 .. 2023-24 (archive starts 2024-01-20): they can be used as ERA5 training history only"},
        "winters": {}, "arms": {}, "seed_control": {}, "weather_mismatch": {}}

    train_sets = {
        "obs": build(hist, live, era, view="offline"),
        "fc_strict": build(hist, live, lambda t: fc1.get(t), view="offline"),
        "mixed": build(hist, live, lambda t: fc1.get(t) if complete(fc1.get(t)) else era(t), view="offline"),
        "mixed_both": build(hist, live, lambda t: fc1.get(t) if complete(fc1.get(t)) and complete(fc1.get(t - timedelta(days=1))) else era(t),
                            wx0_for=lambda d0: fc1.get(d0) if complete(fc1.get(d0 + timedelta(days=1))) and complete(fc1.get(d0)) else era(d0),
                            view="offline"),
    }
    seed_pred = {}
    for Y in WINTERS:
        start, end = wb.winter_window(Y)
        assert end <= HARD_STOP, "evaluation window must end before the untouched period"
        wname = f"{Y}-{str(Y + 1)[2:]}"
        cal = fit_calibration(hist, fc1, start)
        test_sets = {
            "fc": build(hist, live, lambda t: fc1.get(t)),
            "fc48": build(hist, live, lambda t: fc2.get(t)),
            "obs": build(hist, live, era),
            "fc_cal": build(hist, live, lambda t: apply_calibration(fc1[t], cal) if complete(fc1.get(t)) else None),
            "fc_both": build(hist, live, lambda t: fc1.get(t), wx0_for=lambda d0: fc1.get(d0)),
        }
        days = [t for t in sorted(test_sets["fc"]) if start <= t <= end and all(t in s for s in test_sets.values())]
        info = {"window": [start.isoformat(), end.isoformat()], "matched_test_days": len(days),
                "scoreable_days_in_window": sum(1 for t in train_sets["obs"] if start <= t <= end),
                "days_dropped_for_missing_forecast": sum(1 for t in train_sets["obs"] if start <= t <= end) - len(days),
                "calibration": {k: {"intercept": round(a, 3), "slope": round(b, 3), "days": n} for k, (a, b, n) in cal.items()},
                "training": {}}
        y = np.array([test_sets["fc"][t]["y"] for t in days])
        assert all(t >= start for t in days) and all(t <= HARD_STOP for t in days)
        # weather mismatch diagnostic on the matched days
        mm = {}
        for lab, src in (("raw_forecast", test_sets["fc"]), ("calibrated_forecast", test_sets["fc_cal"])):
            mm[lab] = {k: {"bias": round(float(np.mean([src[t]["row"][f"wx1_{k}"] - hist[t][k] for t in days])), 2),
                           "rmse": round(float(np.sqrt(np.mean([(src[t]["row"][f"wx1_{k}"] - hist[t][k]) ** 2 for t in days]))), 2)}
                       for k in WEATHER_KEYS}
        res["weather_mismatch"][wname] = mm
        trained = {}
        for arm, (trk, tek) in ARMS.items():
            tr = {t: v for t, v in train_sets[trk].items() if t < start}
            assert all(t < start for t in tr), "leakage: training target inside/after the window"
            ts = sorted(tr)
            wins = sum(1 for t in ts if t.month in (11, 12, 1, 2))
            n_fc = sum(1 for t in ts if t in fc1) if trk != "obs" else 0
            if trk in ("fc_strict", "mixed", "mixed_both") or arm == "A0_era5_trained":
                info["training"][trk] = {"origins": len(ts), "origins_with_archived_forecast_weather": n_fc if trk != "obs" else 0,
                                         "origins_in_Nov-Feb": wins, "first_target": ts[0].isoformat() if ts else None,
                                         "last_target": ts[-1].isoformat() if ts else None}
            if len(ts) < MIN_TRAIN:
                res["arms"].setdefault(arm, {}).setdefault("not_run", {})[wname] = \
                    f"only {len(ts)} training origins with the required weather before {start} (< {MIN_TRAIN})"
                continue
            if trk not in trained:
                trained[trk] = fit_models(tr)
            rows = [test_sets[tek][t]["row"] for t in days]
            P = predict_all(trained[trk], rows)
            pm0 = np.array([r["pm_d0"] for r in rows])
            P_arm = res["arms"].setdefault(arm, {})
            P_arm.setdefault("_pred", {})[wname] = {"dates": days, "y": y, "pm0": pm0, "P": P,
                                                    "has_dp": np.array([daypart_valid(r) for r in rows])}
        res["winters"][wname] = info
        # seed control: A0 and A2 refitted with 5 seeds (model noise floor + seed-averaged robustness check)
        for arm in ("A0_era5_trained", "A2_mixed"):
            trk, tek = ARMS[arm]
            trs = {t: v for t, v in train_sets[trk].items() if t < start}
            rows = [test_sets[tek][t]["row"] for t in days]
            for sd in SEEDS:
                P = predict_all(fit_models(trs, seed=sd), rows)
                seed_pred.setdefault(arm, {}).setdefault(sd, {})[wname] = P
                res["seed_control"].setdefault(wname, {}).setdefault(arm, {})[str(sd)] = \
                    {m: round(float(np.abs(P[m] - y).mean()), 3) for m in MODELS}
        if verbose:
            print(wname, "matched days", len(days), {a: (a in res["arms"] and wname in res["arms"][a].get("_pred", {})) for a in ARMS})

    # ---- scoring ------------------------------------------------------------------------------------------
    ref = res["arms"]["A0_era5_trained"]["_pred"]
    for arm, body in res["arms"].items():
        pr = body.pop("_pred", {})
        if not pr:
            continue
        body["winters"], body["subsets"] = {}, {}
        # per winter
        for w, d in pr.items():
            sb, _ = stat_block(d["y"], d["P"], d["dates"], d["pm0"])
            if arm != "A0_era5_trained":
                sb["vs_A0"] = diff_vs_ref(d["y"], d["P"], ref[w]["P"], d["dates"], d["pm0"])
            body["winters"][w] = sb
        # pooled over the scored winters (placed 200 days apart on a synthetic timeline)
        order = [w for w in res["winters"] if w in pr]
        pdates = osn.pooled_dates([list(pr[w]["dates"]) for w in order])
        cat = lambda key: np.concatenate([pr[w][key] for w in order])               # noqa: E731
        Pp = {m: np.concatenate([pr[w]["P"][m] for w in order]) for m in MODELS + BASELINES}
        yy, pm0, hdp = cat("y"), cat("pm0"), cat("has_dp")
        real_dates = [d for w in order for d in pr[w]["dates"]]
        Rp = {m: np.concatenate([ref[w]["P"][m] for w in order]) for m in MODELS} if all(w in ref for w in order) else None
        sb, _ = stat_block(yy, Pp, pdates, pm0)
        if arm != "A0_era5_trained" and Rp is not None:
            sb["vs_A0"] = diff_vs_ref(yy, Pp, Rp, pdates, pm0)
        sb["winters_pooled"] = order
        body["pooled"] = sb
        # subsets of the pooled days
        def sub(mask):
            idx = np.where(mask)[0]
            if len(idx) < 10:
                return {"n_days": int(len(idx)), "note": "too few days for intervals"}
            s, _ = stat_block(yy[idx], {m: v[idx] for m, v in Pp.items()}, [pdates[i] for i in idx], pm0[idx])
            if arm != "A0_era5_trained" and Rp is not None:
                s["vs_A0"] = diff_vs_ref(yy[idx], {m: Pp[m][idx] for m in MODELS}, {m: Rp[m][idx] for m in MODELS},
                                         [pdates[i] for i in idx], pm0[idx])
            return s
        md = [(d.month, d.day) for d in real_dates]
        body["subsets"]["stubble_period_15Oct-30Nov"] = sub(np.array([(10, 15) <= x <= (11, 30) for x in md]))
        body["subsets"]["deep_winter_1Dec-15Feb"] = sub(np.array([x >= (12, 1) or x <= (2, 15) for x in md]))
        sd = np.abs(yy - pm0) / np.maximum(pm0, 1.0) >= SUDDEN
        body["subsets"]["sudden_change_days_ge30pct"] = sub(sd)
        body["subsets"]["stable_days"] = sub(~sd)
        body["subsets"]["days_with_all_7_day_parts"] = sub(hdp)
        body["daily"] = {w: {"dates": [d.isoformat() for d in pr[w]["dates"]], "actual": [round(float(v), 2) for v in pr[w]["y"]],
                             **{m: [round(float(v), 2) for v in pr[w]["P"][m]] for m in MODELS + BASELINES}} for w in order}
        # decision rule (candidates only)
        if arm in CANDIDATES and "vs_A0" in body["pooled"]:
            h = body["pooled"]["vs_A0"]["hybrid"]
            worst = max(body["winters"][w]["vs_A0"]["hybrid"]["mae_pct"] for w in order)
            body["rule"] = {"hybrid_mae_pct_vs_A0": round(h["mae_pct"], 2), "ci95_7d": [round(x, 2) for x in h.get("mae_ci95_7d", [0, 0])],
                            "ci95_14d": [round(x, 2) for x in h.get("mae_ci95_14d", [0, 0])], "worst_single_winter_pct": round(worst, 2),
                            "winters_run": order,
                            "promising": bool(h["mae_pct"] <= -RULE["min_gain_pct"] and h["excludes_zero_both"] and h["mae"] < 0
                                              and worst <= RULE["max_single_winter_loss_pct"] and len(order) == len(WINTERS))}
    # ---- seed robustness: seed-averaged predictions, A2 vs A0, same days ------------------------------------
    order = [w for w in res["winters"] if w in ref]
    pdates = osn.pooled_dates([list(ref[w]["dates"]) for w in order])
    yy = np.concatenate([ref[w]["y"] for w in order])
    pm0 = np.concatenate([ref[w]["pm0"] for w in order])
    avg = {a: {m: np.mean([np.concatenate([seed_pred[a][sd][w][m] for w in order]) for sd in SEEDS], axis=0)
               for m in MODELS} for a in seed_pred}
    both = {f"{m}__ref": avg["A0_era5_trained"][m] for m in MODELS}
    both.update({f"{m}__arm": avg["A2_mixed"][m] for m in MODELS})
    st = Stat(yy, both, pdates, pm0)
    per_seed = {str(sd): {a: {m: round(float(np.abs(np.concatenate([seed_pred[a][sd][w][m] for w in order]) - yy).mean()), 3)
                              for m in MODELS} for a in seed_pred} for sd in SEEDS}
    res["seed_robustness"] = {"seeds": list(SEEDS), "days": int(len(yy)),
                              "A2_minus_A0_seed_averaged": {m: st.diff(f"{m}__ref", f"{m}__arm") for m in MODELS},
                              "pooled_mae_by_seed": per_seed,
                              "seeds_where_A2_beats_A0_hybrid": int(sum(per_seed[str(sd)]["A2_mixed"]["hybrid"] <
                                                                         per_seed[str(sd)]["A0_era5_trained"]["hybrid"] for sd in SEEDS))}
    return res


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, date):
        return o.isoformat()
    return o


if __name__ == "__main__":  # pragma: no cover
    os.chdir(ROOT)
    out = jsonable(run())
    Path("experiments/forecast_training_experiment.json").write_text(json.dumps(out, indent=1))
    from forecast_training_report import render
    Path("experiments/forecast_training_experiment.md").write_text(render(out))
    print("wrote experiments/forecast_training_experiment.{json,md}")
