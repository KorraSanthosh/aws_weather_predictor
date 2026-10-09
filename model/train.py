"""Train the next-day PM2.5 model and evaluate it honestly.

    python model/train.py

Reads  data/daily.csv (+ data/daily.meta.json)
Writes model/model.json, model/model_meta.json, model/metrics.json,
       web/data/{metrics,history,forecast}.json  (static demo data for the website)

EVALUATION (rolling winter folds)
  Air quality in Delhi is easy in the monsoon and hard in winter, so we test on winters:
  each fold tests on one 15 Oct -> 15 Feb window, using a model trained ONLY on days before
  that window. The pooled errors are compared against:
    - persistence: "tomorrow = today"
    - 7-day mean
    - CAMS global model estimate (if data/daily.csv has cams_pm25)
  If no full winter can be tested (not enough earlier history), it falls back to the last
  120 days and says so loudly - those numbers are optimistic.

The uncertainty range shown in the app comes from the pooled winter test errors.
The final deployed model is refit on ALL available days.
"""
import csv
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.aqi import POOR_PM25_THRESHOLD, pm25_to_aqi, pm25_to_category, school_guidance  # noqa: E402
from core.features import FEATURE_NAMES, build_features  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402

MIN_TRAIN = 365        # a full-winter fold needs at least this many earlier days
MIN_TEST = 60          # ... and at least this many testable days in the window
MIN_TRAIN_LATE = 240   # late-winter holdout (used when only ONE winter exists): earlier days needed
MIN_TEST_LATE = 40     # ... and testable days in 1 Dec - 15 Feb
VAL_DAYS = 60          # last days of each training set, used for early stopping
FALLBACK_TEST_DAYS = 120
MIN_ROWS = 300

# Hyper-parameters chosen on the 2019-2022 development period only (model/experiments.py -> selection.json).
_SEL = json.loads((Path(__file__).resolve().parent / "selection.json").read_text()) \
    if (Path(__file__).resolve().parent / "selection.json").exists() else {}
PARAMS = dict(_SEL.get("params") or dict(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.85,
                                          colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0))
PARAMS.update(objective="reg:squarederror", tree_method="hist", random_state=42)
FIXED_TREES = bool(_SEL)      # tuned params use a fixed tree count (no early stopping), as validated


def load_daily(path="data/daily.csv"):
    hist, cams = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            rec = {"pm25": float(r["pm25"]) if r["pm25"] != "" else None}
            for k in WEATHER_KEYS:
                rec[k] = float(r[k]) if r.get(k, "") != "" else None
            hist[d] = rec
            if r.get("cams_pm25", "") != "":
                cams[d] = float(r["cams_pm25"])
    return hist, cams


def make_rows(hist):
    """One row per day D that has PM2.5 on D+1 and weather on D+1."""
    X, y, targets = [], [], []
    for d0 in sorted(hist):
        d1 = d0 + timedelta(days=1)
        nxt = hist.get(d1)
        if not nxt or nxt["pm25"] is None or any(nxt[k] is None for k in WEATHER_KEYS):
            continue
        feats = build_features(hist, d0, {k: nxt[k] for k in WEATHER_KEYS})
        if math.isnan(feats[FEATURE_NAMES.index("pm_d0")]):
            continue                                   # need today's reading
        X.append(feats)
        y.append(nxt["pm25"])
        targets.append(d1)
    return np.array(X, dtype=float), np.array(y, dtype=float), targets


def regression_metrics(y, p):
    err = p - y
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y - y.mean()) ** 2))
    cat_hit = np.mean([pm25_to_category(a) == pm25_to_category(b) for a, b in zip(y, p)])
    pos_true, pos_pred = y > POOR_PM25_THRESHOLD, p > POOR_PM25_THRESHOLD
    tp = int(np.sum(pos_true & pos_pred))
    return {
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(math.sqrt(np.mean(err ** 2))), 2),
        "r2": round(1 - ss_res / ss_tot, 3) if ss_tot else None,
        "category_accuracy": round(float(cat_hit), 3),
        "poor_or_worse_recall": round(tp / max(1, int(pos_true.sum())), 3),
        "poor_or_worse_precision": round(tp / max(1, int(pos_pred.sum())), 3),
        "poor_or_worse_share": round(float(pos_true.mean()), 3),   # base rate: how common these days are
        "n_days": int(len(y)),
    }


def winter_folds(targets, test_from=(10, 15), min_train=MIN_TRAIN, min_test=MIN_TEST):
    """Finished winters (test_from -> 15 Feb) that have enough earlier history to train on.

    Default test_from=(10, 15) tests the whole winter. test_from=(12, 1) tests only the peak
    (1 Dec - 15 Feb), which still lets the model see the start of that winter in training."""
    folds = []
    for yr in sorted({t.year for t in targets}):
        start, end = date(yr, *test_from), date(yr + 1, 2, 15)
        if targets[-1] < end:
            continue                                   # winter not finished yet
        test = [i for i, t in enumerate(targets) if start <= t <= end]
        train = [i for i, t in enumerate(targets) if t < start]
        if len(train) >= min_train and len(test) >= min_test:
            folds.append({"window": (start, end), "train": train, "test": test})
    return folds


def fit_with_early_stopping(X, log_y, train_idx):
    """Fit on a training set, holding out its last VAL_DAYS rows for early stopping."""
    if FIXED_TREES:                                   # tuned recipe: fixed n_estimators, as in the experiments
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X[train_idx], log_y[train_idx], verbose=False)
        return model, int(PARAMS["n_estimators"])
    tr, va = train_idx[:-VAL_DAYS], train_idx[-VAL_DAYS:]
    model = xgb.XGBRegressor(**PARAMS, early_stopping_rounds=40)
    model.fit(X[tr], log_y[tr], eval_set=[(X[va], log_y[va])], verbose=False)
    return model, int(model.best_iteration) + 1


def bootstrap_improvement(y, p, persist, block=7, n_boot=2000, seed=0):
    """95% interval for 'MAE improvement vs persistence', resampling WEEKS of days (not single days)
    because neighbouring days are strongly correlated. If the interval includes 0 the win is not proven."""
    gain = np.abs(persist - y) - np.abs(p - y)             # >0 on days the model was closer
    base_err = np.abs(persist - y)
    blocks = [np.arange(i, min(i + block, len(y))) for i in range(0, len(y), block)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        out.append(100 * gain[idx].sum() / base_err[idx].sum())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"low_pct": round(float(lo), 1), "high_pct": round(float(hi), 1),
            "prob_model_better": round(float(np.mean(np.array(out) > 0)), 3),
            "clearly_better": bool(lo > 0)}


def breakdowns(dates, y, p, persist, spike=0.30):
    """Where does the model help? Calm days vs days when PM2.5 jumped/dropped by 30%+, and by month."""
    change = np.abs(y - persist) / np.maximum(persist, 1.0)
    out = {}
    for name, mask in (("calm_days", change < spike), ("spike_days", change >= spike)):
        if mask.sum():
            out[name] = {"n_days": int(mask.sum()),
                         "model_mae": round(float(np.mean(np.abs(p - y)[mask])), 2),
                         "persistence_mae": round(float(np.mean(np.abs(persist - y)[mask])), 2)}
    months = {}
    for i, d in enumerate(dates):
        months.setdefault(d.strftime("%Y-%m"), []).append(i)
    out["by_month"] = {m: {"n_days": len(ix), "model_mae": round(float(np.mean(np.abs(p - y)[ix])), 2),
                           "persistence_mae": round(float(np.mean(np.abs(persist - y)[ix])), 2)}
                       for m, ix in sorted(months.items())}
    return out


def monthly_walk_forward(X, delta, base, y, targets, start=date(2025, 1, 1)):
    """Diagnostic for short histories: test each calendar month using a model trained on everything
    before it (expanding window). Gives a stability check beyond a single train/test split."""
    pm0_i = FEATURE_NAMES.index("pm_d0")
    rows, ym = [], sorted({(t.year, t.month) for t in targets})
    for yr, mo in ym:
        m0 = date(yr, mo, 1)
        tr = [i for i, t in enumerate(targets) if t < m0]
        te = [i for i, t in enumerate(targets) if t.year == yr and t.month == mo]
        if len(tr) < MIN_TRAIN_LATE or len(te) < 15:
            continue
        model, _ = fit_with_early_stopping(X, delta, tr)
        p = np.expm1(base[te] + model.predict(X[te]))
        rows.append({"month": f"{yr}-{mo:02d}", "n_days": len(te),
                     "model_mae": round(float(np.mean(np.abs(p - y[te]))), 2),
                     "persistence_mae": round(float(np.mean(np.abs(X[te][:, pm0_i] - y[te]))), 2)})
    return rows


def main():
    hist, cams = load_daily()
    meta_in = json.loads(Path("data/daily.meta.json").read_text()) if Path("data/daily.meta.json").exists() else {}
    source = meta_in.get("source", "unknown")
    X, y, targets = make_rows(hist)
    if len(y) < MIN_ROWS:
        sys.exit(f"Only {len(y)} usable days; need at least {MIN_ROWS}. Get more history or stations.")
    pm0_col = FEATURE_NAMES.index("pm_d0")
    base = np.log1p(X[:, pm0_col])      # today's level; the model learns the CHANGE from it
    log_y = np.log1p(y)
    delta = log_y - base
    n = len(y)

    folds = winter_folds(targets)
    mode = "rolling_winter_folds"
    if not folds:                                      # not enough history for a whole winter: test the peak
        folds = winter_folds(targets, test_from=(12, 1), min_train=MIN_TRAIN_LATE, min_test=MIN_TEST_LATE)
        mode = "late_winter_holdout"
        if folds:
            print("\nNOTE: not enough history to test a whole winter. Testing the peak of winter "
                  "(1 Dec - 15 Feb) with a model trained only on earlier days. With a single winter "
                  "this is indicative, not conclusive.\n")
    if not folds:
        te0 = n - FALLBACK_TEST_DAYS
        folds = [{"window": (targets[te0], targets[-1]),
                  "train": list(range(te0)), "test": list(range(te0, n))}]
        mode = "last_120_days_not_a_winter"
        print("\n!! WARNING: no full winter could be tested (need a 15 Oct-15 Feb window with "
              f"{MIN_TRAIN}+ earlier days). Falling back to the last {FALLBACK_TEST_DAYS} days - "
              "these numbers are OPTIMISTIC and say little about winter performance.\n")
    print(f"data source: {source} | usable days: {n} | evaluation: {mode} | folds: {len(folds)}")

    # ---------- evaluate each fold with a model that has only seen the past ----------
    pooled = {"y": [], "p": [], "persist": [], "mean7": [], "cams": [], "resid": []}
    pooled_dates = []
    fold_reports, best_iters, last_fold_rows = [], [], None
    pm0_i, mean7_i = FEATURE_NAMES.index("pm_d0"), FEATURE_NAMES.index("pm_mean7")
    for fold in folds:
        model, best = fit_with_early_stopping(X, delta, fold["train"])
        best_iters.append(best)
        te = fold["test"]
        pred_log = base[te] + model.predict(X[te])
        pred = np.expm1(pred_log)
        persist = X[te][:, pm0_i]
        mean7 = np.where(np.isnan(X[te][:, mean7_i]), persist, X[te][:, mean7_i])
        cams_v = np.array([cams.get(targets[i], np.nan) for i in te])
        pooled["y"].append(y[te]); pooled["p"].append(pred); pooled["persist"].append(persist)
        pooled["mean7"].append(mean7); pooled["cams"].append(cams_v)
        pooled["resid"].append(log_y[te] - pred_log)
        pooled_dates.extend(targets[i] for i in te)
        m_model, m_base = regression_metrics(y[te], pred), regression_metrics(y[te], persist)
        fold_reports.append({
            "window": [fold["window"][0].isoformat(), fold["window"][1].isoformat()],
            "n_train": len(fold["train"]), "n_test": len(te),
            "model_mae": m_model["mae"], "persistence_mae": m_base["mae"],
            "model_beats_persistence": bool(m_model["mae"] < m_base["mae"]),
        })
        last_fold_rows = [(targets[i], y[i], p_, b_) for i, p_, b_ in zip(te, pred, persist)]
        print(f"  fold {fold_reports[-1]['window'][0]} -> {fold_reports[-1]['window'][1]}: "
              f"train {len(fold['train'])}, test {len(te)} | MAE model {m_model['mae']} "
              f"vs persistence {m_base['mae']}")

    cat = lambda k: np.concatenate(pooled[k])  # noqa: E731
    y_all = cat("y")
    results = {
        "evaluation": mode,
        "folds": fold_reports,
        "model": regression_metrics(y_all, cat("p")),
        "baseline_persistence": regression_metrics(y_all, cat("persist")),
        "baseline_7day_mean": regression_metrics(y_all, cat("mean7")),
    }
    cams_all = cat("cams")
    ok = ~np.isnan(cams_all)
    if ok.sum() >= 30:
        results["cams_global_model"] = regression_metrics(y_all[ok], cams_all[ok])
    p_all, per_all = cat("p"), cat("persist")
    results["improvement_95pct_interval"] = bootstrap_improvement(y_all, p_all, per_all)
    results["breakdown"] = breakdowns(pooled_dates, y_all, p_all, per_all)
    if mode == "late_winter_holdout":
        results["walk_forward_monthly"] = monthly_walk_forward(X, delta, base, y, targets)
    ci = results["improvement_95pct_interval"]
    results["verdict"] = ("The model is clearly better than 'tomorrow = today'." if ci["clearly_better"] else
                          "The improvement over 'tomorrow = today' is NOT statistically clear yet "
                          "(95% interval includes 0) - more winters are needed to prove it.")
    results["improvement_vs_persistence_pct"] = round(
        100 * (1 - results["model"]["mae"] / results["baseline_persistence"]["mae"]), 1)
    results["folds_model_beats_persistence"] = f"{sum(f['model_beats_persistence'] for f in fold_reports)} of {len(fold_reports)}"
    if mode == "rolling_winter_folds":
        results["test_description"] = ("winters " + ", ".join(
            f"{f['window'][0][:4]}-{f['window'][1][2:4]}" for f in fold_reports)
            + " (15 Oct to 15 Feb), each predicted by a model trained only on earlier days")
    elif mode == "late_winter_holdout":
        results["test_description"] = ("the peak of winter " + ", ".join(
            f"{f['window'][0][:4]}-{f['window'][1][2:4]}" for f in fold_reports)
            + " (1 Dec to 15 Feb), predicted by a model trained only on earlier days")
        results["warning"] = (f"Only one winter of data: accuracy is estimated on {len(y_all)} days of a single "
                              "winter. Treat it as indicative, not conclusive.")
    else:
        results["test_description"] = (f"the last {FALLBACK_TEST_DAYS} days - NOT a winter, so this is optimistic")
        results["warning"] = "No full winter could be tested; do not quote these numbers as winter accuracy."
    test_dates = [targets[i] for f in folds for i in f["test"]]
    results["test_period"] = [min(test_dates).isoformat(), max(test_dates).isoformat()]
    results["data_source"] = source
    results["note_weather"] = ("Training uses observed weather for the target day; live use "
                               "relies on a weather forecast, so real-world error will be somewhat higher.")

    # ---------- uncertainty range from pooled (winter) residuals, log space ----------
    resid = cat("resid")
    q10, q90 = float(np.quantile(resid, 0.10)), float(np.quantile(resid, 0.90))

    # ---------- final model: all data, tree count = median of the folds' early-stopped counts ----------
    n_est = int(np.median(best_iters))
    final = xgb.XGBRegressor(**{**PARAMS, "n_estimators": n_est})
    final.fit(X, delta, verbose=False)
    importances = sorted(zip(FEATURE_NAMES, final.feature_importances_), key=lambda t: -t[1])
    results["top_features"] = [{"name": n_, "importance": round(float(v), 4)} for n_, v in importances[:8]]

    Path("model").mkdir(exist_ok=True)
    final.get_booster().save_model("model/model.json")
    Path("model/model_meta.json").write_text(json.dumps({
        "features": FEATURE_NAMES, "target_transform": "log1p_delta_vs_today",
        "log_resid_q10": round(q10, 4), "log_resid_q90": round(q90, 4),
        "trained_through": targets[-1].isoformat(), "data_source": source,
        "n_estimators": n_est, "evaluation": mode}, indent=2))
    Path("model/metrics.json").write_text(json.dumps(results, indent=2))

    # ---------- static demo data for the website ----------
    web = Path("web/data")
    web.mkdir(parents=True, exist_ok=True)
    (web / "history.json").write_text(json.dumps([
        {"date": t.isoformat(), "actual": round(float(a), 1), "predicted": round(float(p), 1),
         "baseline": round(float(b), 1)} for t, a, p, b in last_fold_rows]))
    (web / "metrics.json").write_text(json.dumps(results))

    d_last = max(d for d, r in hist.items() if r["pm25"] is not None)
    proxy_wx = {k: hist[d_last][k] for k in WEATHER_KEYS}      # offline demo: reuse today's weather
    feats = np.array([build_features(hist, d_last, proxy_wx)], dtype=float)
    pm = float(np.expm1(np.log1p(feats[0, pm0_col]) + final.predict(feats)[0]))
    lo, hi = pm * math.exp(q10), pm * math.exp(q90)
    category = pm25_to_category(pm)
    (web / "forecast.json").write_text(json.dumps({
        "target_date": (d_last + timedelta(days=1)).isoformat(), "generated_for": d_last.isoformat(),
        "pm25": round(pm, 1), "pm25_low": round(lo, 1), "pm25_high": round(hi, 1),
        "aqi": pm25_to_aqi(pm), "category": category, "guidance": school_guidance(category),
        "today_pm25": round(hist[d_last]["pm25"], 1), "data_source": source,
        "weather_source": "offline demo (today's weather reused)"}))

    print("\n=== POOLED RESULT ===")
    for k in ("model", "baseline_persistence", "baseline_7day_mean", "cams_global_model"):
        if k in results:
            r = results[k]
            print(f"{k:22} MAE {r['mae']:>6}  RMSE {r['rmse']:>6}  category-acc {r['category_accuracy']}  "
                  f"poor+ recall {r['poor_or_worse_recall']}  (n={r['n_days']})")
    print(f"MAE improvement vs persistence: {results['improvement_vs_persistence_pct']}% "
          f"(model wins in {results['folds_model_beats_persistence']} folds)")
    ci = results["improvement_95pct_interval"]
    print(f"95% interval for that improvement: {ci['low_pct']}% to {ci['high_pct']}% "
          f"(P(model better) = {ci['prob_model_better']})")
    print("->", results["verdict"])
    for name in ("calm_days", "spike_days"):
        b = results["breakdown"].get(name)
        if b:
            print(f"  {name:11} n={b['n_days']:>3}  model MAE {b['model_mae']}  persistence MAE {b['persistence_mae']}")
    for r in results.get("walk_forward_monthly", []):
        print(f"  walk-forward {r['month']}: n={r['n_days']} model {r['model_mae']} vs persistence {r['persistence_mae']}")
    print("top features:", [t['name'] for t in results['top_features'][:5]])


if __name__ == "__main__":
    main()
