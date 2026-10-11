"""Does adding CAMS PM2.5 information to the existing Hybrid improve next-day prediction?  Isolated experiment.

FINDING THAT SHAPES EVERYTHING: Open-Meteo offers NO archive of past CAMS forecasts (see probe.json: every
`pm2_5_previous_dayN` request returns null for every period; the previous-runs endpoint has no air-quality route).
The only CAMS history available (data/raw/cams_daily.csv) is the API's stitched past series (analysis-like early hours of each run),
i.e. it is NOT what was knowable at 23:00 IST on the day before.  A REALISTIC forecast-time CAMS input therefore cannot be reconstructed.

So this script only runs clearly-labelled, NON-REALISTIC diagnostic arms, to learn whether CAMS carries any usable signal at all:
  O_target_day   cams of the TARGET day (+ origin-day, ratio)   -> information about the day being predicted: an optimistic upper bound
  O_lags_only    cams of the origin day and the day before       -> nearly realistic, but the origin-day mean contains hours after the 22:59 IST cutoff
Nothing here can justify a production change.  Production code is imported read-only.
"""
import csv, json, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "experiments" / "weather_source_comparison"):
    sys.path.insert(0, str(p))
import forecast_training_experiment as fte, option_screen as osn, winter_backtest as wb
import run_comparison as rc
from core.features import FEATURE_SETS
from core.weather import WEATHER_KEYS
from stats import Stats, sudden_mask

SEEDS = fte.SEEDS
V1, V2 = fte.V1, fte.V2
ARMS = {"O_target_day": ["cams_t", "cams_d0", "cams_chg"], "O_lags_only": ["cams_d0", "cams_d1", "cams_d0_minus_d1"]}
WINDOWS = {"calendar_2025 (train <= 2024-12-31)": (date(2024, 12, 31), date(2025, 1, 1), date(2025, 12, 31)),
           "winter_2025_26 (train < 2025-10-15)": (date(2025, 10, 14), date(2025, 10, 15), date(2026, 2, 15))}


def load_cams():
    return {date.fromisoformat(r["date"]): float(r["cams_pm25"]) for r in csv.DictReader(open(ROOT / "data/raw/cams_daily.csv")) if r["cams_pm25"] != ""}


def with_cams(rows_by_t, cams, extras):
    out = {}
    for t, v in rows_by_t.items():
        r = dict(v["row"]); d0 = t - timedelta(days=1); d1 = t - timedelta(days=2)
        c = lambda d: cams.get(d, float("nan"))                                     # noqa: E731
        r["cams_t"], r["cams_d0"], r["cams_d1"] = c(t), c(d0), c(d1)
        r["cams_chg"] = float(np.log1p(c(t)) - np.log1p(c(d0)))
        r["cams_d0_minus_d1"] = float(np.log1p(c(d0)) - np.log1p(c(d1)))
        out[t] = {"y": v["y"], "row": r}
    return out


def fit(train, v, extras, seed):
    import xgboost as xgb
    names = FEATURE_SETS[v] + extras
    ts = sorted(train)
    y = np.array([train[t]["y"] for t in ts]); X = osn.M([train[t]["row"] for t in ts], names)
    delta = np.log1p(y) - np.log1p(X[:, names.index("pm_d0")])
    m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed, early_stopping_rounds=40, n_jobs=4, **fte.OLD_PARAMS)
    m.fit(X[:-60], delta[:-60], eval_set=[(X[-60:], delta[-60:])], verbose=False)
    return m, names


def pred(model_names, rows):
    m, names = model_names
    return osn.predict(m, osn.M(rows, names), names)


def main():
    hist = osn.load_hist("data/daily_opencity.csv")
    live = fte.build_live_view(Path("data/raw"), extra_subs=("opencity_hourly",))
    era = lambda t: {k: hist[t].get(k) for k in WEATHER_KEYS}
    src = rc.load_source("production_default")
    cams = load_cams()
    res = {"label": "DIAGNOSTIC / NON-REALISTIC: no archived CAMS day-ahead forecast exists; see module docstring.",
           "cams_file": {"first": min(cams).isoformat(), "last": max(cams).isoformat(), "days": len(cams)}, "windows": {}}
    all_train = fte.build(hist, live, era, view="offline")
    test_all = fte.build(hist, live, lambda t, f=src[1]: f.get(t))
    for wname, (tr_end, a, b) in WINDOWS.items():
        train0 = {t: v for t, v in all_train.items() if t <= tr_end}
        days = sorted(t for t in test_all if a <= t <= b and t in cams and (t - timedelta(days=1)) in cams)
        res["windows"][wname] = {"train_origins": len(train0), "train_origins_with_cams_target": sum(1 for t in train0 if t in cams),
                                 "test_days_cams_available": len(days), "test_days_without_cams_dropped_for_all_models": sum(1 for t in test_all if a <= t <= b) - len(days)}
        y = np.array([test_all[t]["y"] for t in days]); rows0 = [test_all[t]["row"] for t in days]
        pm0 = np.array([r["pm_d0"] for r in rows0]); dp = np.array([fte.daypart_valid(r) for r in rows0])
        # baseline = existing Hybrid, identical fit
        base_models = {sd: fte.fit_models(train0, sd) for sd in SEEDS}
        Pb = {sd: fte.predict_all(base_models[sd], rows0) for sd in SEEDS}
        arms = {}
        for arm, extras in ARMS.items():
            tr = with_cams(train0, cams, extras)
            te = with_cams({t: test_all[t] for t in days}, cams, extras)
            rows = [te[t]["row"] for t in days]
            P = {}
            for sd in SEEDS:
                m1, m2 = fit(tr, V1, extras, sd), fit(tr, V2, extras, sd)
                p1, p2 = pred(m1, rows), pred(m2, rows)
                P[sd] = {"v1c": p1, "v2c": p2, "hybrid_c": osn.hybrid(p1, p2, rows0)}
            arms[arm] = P
        blk = {"n_days": len(days), "first": days[0].isoformat(), "last": days[-1].isoformat(), "hybrid_routing_V2_days": int(dp.sum()),
               "cams_benchmark_alone": None}
        yc = np.array([cams[t] for t in days])
        blk["cams_benchmark_alone"] = {"mae": float(np.abs(yc - y).mean()), "rmse": float(np.sqrt(((yc - y) ** 2).mean())), "mean_bias": float((yc - y).mean()),
                                       "corr_with_target_day_pm25": float(np.corrcoef(yc, y)[0, 1]),
                                       "corr_origin_day_cams_with_target_day_pm25": float(np.corrcoef([cams[t - timedelta(days=1)] for t in days], y)[0, 1])}
        for arm in ARMS:
            st = Stats(y, {"persistence": Pb[42]["persistence"], "v1": Pb[42]["v1"], "hybrid": Pb[42]["hybrid"], "v1c": arms[arm][42]["v1c"], "hybrid_c": arms[arm][42]["hybrid_c"]}, days)
            sdm = sudden_mask(y, pm0, 0.30)
            def sub(mask):
                idx = np.where(mask)[0]
                if len(idx) < 10: return {"n_days": int(len(idx))}
                s = Stats(y[idx], {"hybrid": Pb[42]["hybrid"][idx], "hybrid_c": arms[arm][42]["hybrid_c"][idx]}, [days[i] for i in idx])
                return {"n_days": int(len(idx)), "hybrid": s.point("hybrid"), "hybrid_c": s.point("hybrid_c"), "diff": s.diff("hybrid", "hybrid_c")}
            blk[arm] = {"models": {m: st.model(m) for m in ("persistence", "v1", "hybrid", "v1c", "hybrid_c")},
                        "hybrid_c_vs_hybrid": st.diff("hybrid", "hybrid_c"), "v1c_vs_v1": st.diff("v1", "v1c"),
                        "sudden": sub(sdm), "stable": sub(~sdm),
                        "seeds": {str(sd): {"hybrid": float(np.abs(Pb[sd]["hybrid"] - y).mean()), "hybrid_c": float(np.abs(arms[arm][sd]["hybrid_c"] - y).mean())} for sd in SEEDS}}
            blk[arm]["seeds_candidate_better"] = int(sum(v["hybrid_c"] < v["hybrid"] for v in blk[arm]["seeds"].values()))
        res["windows"][wname]["result"] = blk
        print(wname, {a: round(blk[a]["models"]["hybrid_c"]["mae"], 2) for a in ARMS}, round(blk["O_target_day"]["models"]["hybrid"]["mae"], 2), flush=True)
    return res


if __name__ == "__main__":
    r = main()
    (HERE / "results.json").write_text(json.dumps(fte.jsonable(r), indent=1))
    print("written")
