"""Model-improvement experiments with a strict time split.

    python model/experiments.py dev      # all choices are made here (2019-2022 only)
    python model/experiments.py final    # evaluates the frozen choice ONCE on 2023 -> today

PROTOCOL
  * Walk-forward, monthly re-fit: every month is predicted by a model trained only on days whose
    target date is before that month (what a monthly-retrained live system would do).
  * Development period (DEV): 2019-01-01 .. 2022-12-31. Feature groups, hyper-parameters, model
    family, sample weighting and classification-vs-regression are ALL chosen on DEV.
    The choice is written to experiments/selection_v2_candidate.json.
  * Final test period (TEST): 2023-01-01 .. latest day. It is evaluated once, by `final`, with the
    frozen selection. It contains the winters 2023-24, 2024-25 (to ~31 Dec, 1-station data after)
    and 2025-26, plus all other seasons.
  * Nothing is shuffled. No value is interpolated. Days need >= 3 stations (pipeline rule).

Weather caveat: the history has OBSERVED weather for the target day; live use has a FORECAST.
`final` therefore also reports two realism checks: tomorrow's weather replaced by today's
(a crude 'no-skill' forecast, pessimistic) and a model trained with no tomorrow-weather at all.
"""
import csv
import json
import math
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.aqi import POOR_PM25_THRESHOLD, pm25_to_category  # noqa: E402
from core.features import ALL_FEATURES, BASE, GROUPS, feature_dict  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402

CATS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
DEV = (date(2019, 1, 1), date(2022, 12, 31))
TEST_START = date(2023, 1, 1)
MIN_TRAIN_ROWS = 150
OLD_PARAMS = dict(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.85,
                  colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0)
SEASON_NAMES = {0: "winter (Dec-Feb)", 1: "summer (Mar-May)", 2: "monsoon (Jun-Sep)", 3: "post-monsoon (Oct-Nov)"}


# ------------------------------------------------------------------ data
def load(path="data/daily.csv"):
    hist, cams = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            num = lambda k: float(r[k]) if r.get(k, "") not in ("", None) else None  # noqa: E731
            rec = {"pm25": num("pm25"), "n": num("n_stations") if r.get("pm25") else None,
                   "pm25_sd": num("pm25_sd"), "pm25_max": num("pm25_max"), "pm25_min": num("pm25_min")}
            for k in WEATHER_KEYS:
                rec[k] = num(k)
            hist[d] = rec
            if num("cams_pm25") is not None:
                cams[d] = num("cams_pm25")
    return hist, cams


def build_matrix(hist, wx1_mode="observed"):
    """Rows for every day D with PM2.5 on D and D+1 and weather on D+1.
    wx1_mode: 'observed' (target day's observed weather) or 'naive' (today's weather reused)."""
    rows, y, dates = [], [], []
    for d0 in sorted(hist):
        d1 = d0 + timedelta(days=1)
        nxt, cur = hist.get(d1), hist[d0]
        if not nxt or nxt["pm25"] is None or cur["pm25"] is None:
            continue
        if any(nxt[k] is None for k in WEATHER_KEYS):
            continue
        src = nxt if wx1_mode == "observed" else cur
        wx1 = {k: src[k] for k in WEATHER_KEYS}
        f = feature_dict(hist, d0, wx1)
        rows.append([f[n] for n in ALL_FEATURES])
        y.append(nxt["pm25"])
        dates.append(d1)
    return np.array(rows, dtype=float), np.array(y), dates


# ------------------------------------------------------------------ models
def make_model(kind, params, seed=0):
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed,
                                n_jobs=4, **params)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=4, **params)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=4, **params)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(kind)


def months(start, end):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        yield cur
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)


def walk_forward(X, y, dates, cols, start, end, kind="xgb", params=None, task="reg",
                 weight_fn=None, train_mask_fn=None, early_stop=False):
    """Monthly re-fit. Returns {row_index: prediction (pm25) or class probabilities}."""
    import xgboost as xgb
    idx_cols = [ALL_FEATURES.index(c) for c in cols]
    Xs = X[:, idx_cols]
    base = np.log1p(X[:, ALL_FEATURES.index("pm_d0")])
    delta = np.log1p(y) - base
    dates_arr = np.array(dates)
    out = {}
    for m0 in months(start, end):
        m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
        te = np.where((dates_arr >= m0) & (dates_arr < m1) & (dates_arr >= start) & (dates_arr <= end))[0]
        if len(te) == 0:
            continue
        tr = np.where(dates_arr < m0)[0]
        if train_mask_fn is not None:
            tr = tr[[train_mask_fn(dates[i]) for i in tr]]
        if len(tr) < MIN_TRAIN_ROWS:
            continue
        w = np.array([weight_fn(i) for i in tr]) if weight_fn else None
        if task == "clf":
            labels = np.array([CATS.index(pm25_to_category(v)) for v in y])
            dtr = xgb.DMatrix(Xs[tr], label=labels[tr], weight=w)
            p = dict(objective="multi:softprob", num_class=6, tree_method="hist", seed=0,
                     eta=params.get("learning_rate", 0.05), max_depth=params.get("max_depth", 4),
                     subsample=params.get("subsample", 0.85), colsample_bytree=params.get("colsample_bytree", 0.85),
                     min_child_weight=params.get("min_child_weight", 3), reg_lambda=params.get("reg_lambda", 2.0),
                     reg_alpha=params.get("reg_alpha", 0.0), nthread=4)
            bst = xgb.train(p, dtr, num_boost_round=params.get("n_estimators", 300))
            prob = bst.predict(xgb.DMatrix(Xs[te]))
            for i, pr in zip(te, prob):
                out[i] = pr
            continue
        if early_stop:                                   # the v1 recipe: last 60 training rows for early stopping
            model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42,
                                     early_stopping_rounds=40, n_jobs=4, **params)
            model.fit(Xs[tr[:-60]], delta[tr[:-60]], eval_set=[(Xs[tr[-60:]], delta[tr[-60:]])], verbose=False)
        else:
            model = make_model(kind, params)
            if w is not None:
                model.fit(Xs[tr], delta[tr], sample_weight=w)
            else:
                model.fit(Xs[tr], delta[tr])
        pred = np.expm1(base[te] + model.predict(Xs[te]))
        for i, p_ in zip(te, pred):
            out[i] = float(p_)
    return out


# ------------------------------------------------------------------ metrics
def metrics(y, p, persist=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {}
    err = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    tc = [pm25_to_category(v) for v in y]
    pc = [pm25_to_category(v) for v in p]
    pos_t, pos_p = y > POOR_PM25_THRESHOLD, p > POOR_PM25_THRESHOLD
    tp = int(np.sum(pos_t & pos_p))
    out = {"n": int(len(y)), "mae": round(float(np.mean(np.abs(err))), 2),
           "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
           "r2": round(1 - float(np.sum(err ** 2)) / ss_tot, 3) if ss_tot else None,
           "cat_acc": round(float(np.mean([a == b for a, b in zip(tc, pc)])), 3),
           "within_one_cat": round(float(np.mean([abs(CATS.index(a) - CATS.index(b)) <= 1 for a, b in zip(tc, pc)])), 3),
           "poor_recall": round(tp / max(1, int(pos_t.sum())), 3),
           "poor_precision": round(tp / max(1, int(pos_p.sum())), 3)}
    return out


def cat_metrics_from_labels(y, pred_cats):
    tc = [pm25_to_category(v) for v in y]
    return round(float(np.mean([a == b for a, b in zip(tc, pred_cats)])), 3)


def confusion(y, pred_cats):
    m = [[0] * 6 for _ in range(6)]
    for v, pc in zip(y, pred_cats):
        m[CATS.index(pm25_to_category(v))][CATS.index(pc)] += 1
    return m


def season_of(d):
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}.get(d.month, 3)


def in_winter_window(d):
    return (d.month, d.day) >= (10, 15) or (d.month, d.day) <= (2, 15)


def score(pred, X, y, dates, subset=None):
    """MAE etc. on the rows predicted (optionally a subset filter on dates)."""
    ids = sorted(i for i in pred if subset is None or subset(dates[i]))
    return metrics(y[ids], [pred[i] for i in ids]), ids


# ------------------------------------------------------------------ DEV: selection
def run_dev():
    hist, _ = load()
    X, y, dates = build_matrix(hist)
    a, b = DEV
    n_dev = sum(1 for d in dates if a <= d <= b)
    print(f"rows: {len(y)} | development rows: {n_dev} ({a} .. {b}) | test rows (untouched): "
          f"{sum(1 for d in dates if d >= TEST_START)}")
    log = {"protocol": "walk-forward monthly refit; selection on 2019-2022 only", "dev_rows": n_dev, "steps": []}
    t0 = time.time()

    def dev_eval(cols, kind="xgb", params=None, **kw):
        pred = walk_forward(X, y, dates, cols, a, b, kind=kind, params=params or OLD_PARAMS, **kw)
        m, ids = score(pred, X, y, dates)
        mw, _ = score(pred, X, y, dates, in_winter_window)
        return m, mw, pred

    # 0. baselines on DEV
    pm0 = X[:, ALL_FEATURES.index("pm_d0")]
    ref = walk_forward(X, y, dates, BASE, a, b, params=OLD_PARAMS, early_stop=True)
    ids = sorted(ref)
    log["dev_baselines"] = {"persistence": metrics(y[ids], pm0[ids]),
                            "v1_model (old features + old recipe)": metrics(y[ids], [ref[i] for i in ids])}
    print("DEV persistence:", log["dev_baselines"]["persistence"])
    print("DEV v1 model   :", log["dev_baselines"]["v1_model (old features + old recipe)"])

    # 1. cumulative feature-group ablation (fixed v1 params, no early stopping -> same for all)
    cols = list(BASE)
    m, mw, _ = dev_eval(cols)
    best = m
    ablation = [{"step": "1. current (v1) features", "kept": True, "n_features": len(cols), "dev": m, "dev_winter": mw}]
    print(f"[{time.time()-t0:5.0f}s] base: MAE {m['mae']} acc {m['cat_acc']}")
    names = {"lags": "2. + PM2.5 lag 7/14", "rolling": "3. + rolling/change/trend", "weather": "4. + weather-derived",
             "seasonal": "5. + seasonal/calendar", "spatial": "6. + station/spatial"}
    for g, feats in GROUPS.items():
        trial = cols + feats
        m, mw, _ = dev_eval(trial)
        keep = m["mae"] <= best["mae"] * 0.99            # rule fixed in advance: keep only if DEV MAE improves >= 1%
        ablation.append({"step": names[g], "kept": keep, "n_features": len(trial), "dev": m, "dev_winter": mw})
        print(f"[{time.time()-t0:5.0f}s] {names[g]}: MAE {m['mae']} acc {m['cat_acc']} -> {'KEEP' if keep else 'drop'}")
        if keep:
            cols, best = trial, m
    # also try each dropped group alone on top of the base (in case order hid a benefit) - informational
    log["ablation"] = ablation
    log["selected_features"] = cols

    # 2. hyper-parameter search (random, 40 configs, DEV MAE)
    rng = random.Random(7)
    space = dict(n_estimators=[150, 300, 500, 800, 1200], learning_rate=[0.01, 0.02, 0.03, 0.05, 0.08],
                 max_depth=[2, 3, 4, 5, 6], subsample=[0.6, 0.75, 0.85, 1.0], colsample_bytree=[0.5, 0.7, 0.85, 1.0],
                 min_child_weight=[1, 3, 5, 10, 20], reg_alpha=[0, 0.1, 1.0, 5.0], reg_lambda=[0.5, 2.0, 5.0, 10.0])
    trials = [dict(OLD_PARAMS, reg_alpha=0.0)]
    for _ in range(23):
        trials.append({k: rng.choice(v) for k, v in space.items()})
    tune = []
    for i, p in enumerate(trials):
        m, mw, _ = dev_eval(cols, params=p)
        tune.append({"params": p, "dev": m, "dev_winter": mw})
    tune.sort(key=lambda t: t["dev"]["mae"])
    best_params = tune[0]["params"]
    print(f"[{time.time()-t0:5.0f}s] tuned XGB: MAE {tune[0]['dev']['mae']} acc {tune[0]['dev']['cat_acc']} {best_params}")
    log["tuning_top5"] = tune[:5]
    log["tuning_default_v1_params"] = next(t for t in tune if t["params"] == trials[0])

    # 3. other model families (small grids, DEV MAE)
    fam = {"xgb (tuned)": ("xgb", [best_params]),
           "lightgbm": ("lgbm", [dict(n_estimators=n, learning_rate=lr, num_leaves=nl, min_child_samples=mc,
                                      subsample=0.85, subsample_freq=1, colsample_bytree=0.85)
                                 for n, lr, nl, mc in [(300, 0.03, 15, 20), (600, 0.02, 7, 20), (400, 0.05, 31, 30),
                                                       (800, 0.01, 15, 10), (300, 0.05, 7, 40)]]),
           "random_forest": ("rf", [dict(n_estimators=500, min_samples_leaf=l, max_features=mf)
                                    for l, mf in [(3, 0.5), (5, 0.33), (10, 0.5), (2, 1.0)]]),
           "hist_gradient_boosting": ("hgb", [dict(max_iter=n, learning_rate=lr, max_leaf_nodes=ml, min_samples_leaf=ms)
                                              for n, lr, ml, ms in [(300, 0.05, 15, 20), (500, 0.03, 31, 20),
                                                                    (300, 0.05, 7, 40), (800, 0.02, 15, 10)]])}
    families = {}
    for name, (kind, grid) in fam.items():
        res = []
        for p in grid:
            m, mw, _ = dev_eval(cols, kind=kind, params=p)
            res.append({"params": p, "dev": m, "dev_winter": mw})
        res.sort(key=lambda r: r["dev"]["mae"])
        families[name] = res[0]
        print(f"[{time.time()-t0:5.0f}s] {name}: MAE {res[0]['dev']['mae']} acc {res[0]['dev']['cat_acc']}")
    log["model_families"] = families
    best_family = min(families, key=lambda k: families[k]["dev"]["mae"])
    # keep a more complex / different family only if it beats tuned XGB by >= 1% (rule fixed in advance)
    chosen_kind, chosen_params = "xgb", best_params
    if best_family != "xgb (tuned)" and families[best_family]["dev"]["mae"] <= families["xgb (tuned)"]["dev"]["mae"] * 0.99:
        chosen_kind = fam[best_family][0]
        chosen_params = families[best_family]["params"]
    print("chosen family:", chosen_kind)

    # 4. sample weighting and season-aware training
    base_m, base_mw, _ = dev_eval(cols, chosen_kind, chosen_params)
    pm0_all = X[:, ALL_FEATURES.index("pm_d0")]
    chg = np.abs(np.log1p(y) - np.log1p(pm0_all))
    weights = {
        "none": None,
        "winter x2": lambda i: 2.0 if in_winter_window(dates[i]) else 1.0,
        "sudden-change (1 + 2|dlog|)": lambda i: 1.0 + 2.0 * chg[i],
        "high-pollution (1 + PM/150)": lambda i: 1.0 + y[i] / 150.0,
    }
    wres = {}
    for name, fn in weights.items():
        m, mw, _ = dev_eval(cols, chosen_kind, chosen_params, weight_fn=fn) if fn else (base_m, base_mw, None)
        wres[name] = {"dev": m, "dev_winter": mw}
        print(f"[{time.time()-t0:5.0f}s] weighting {name}: MAE {m['mae']} acc {m['cat_acc']}")
    log["sample_weighting"] = wres
    best_w = min(wres, key=lambda k: wres[k]["dev"]["mae"])
    chosen_weight = best_w if wres[best_w]["dev"]["mae"] <= base_m["mae"] * 0.99 else "none"

    # season-aware: winter-window rows predicted by a model trained on winter-window rows only
    m_all_w = base_mw
    pred_w = walk_forward(X, y, dates, cols, a, b, kind=chosen_kind, params=chosen_params,
                          train_mask_fn=in_winter_window)
    mw_only, _ = score(pred_w, X, y, dates, in_winter_window)
    log["season_aware"] = {"winter rows, model trained on all seasons": m_all_w,
                           "winter rows, model trained on winter only": mw_only}
    print(f"season-aware winter-only MAE {mw_only.get('mae')} vs all-season {m_all_w.get('mae')}")
    season_aware = bool(mw_only and mw_only["mae"] <= m_all_w["mae"] * 0.99)

    # 5. direct AQI classification vs regression -> category
    reg_pred = walk_forward(X, y, dates, cols, a, b, kind=chosen_kind, params=chosen_params)
    ids = sorted(reg_pred)
    reg_acc = metrics(y[ids], [reg_pred[i] for i in ids])["cat_acc"]
    clf_res = []
    for p in [dict(best_params), dict(n_estimators=300, learning_rate=0.05, max_depth=3),
              dict(n_estimators=150, learning_rate=0.1, max_depth=4, min_child_weight=5)]:
        prob = walk_forward(X, y, dates, cols, a, b, task="clf", params=p)
        cid = sorted(prob)
        pcat = [CATS[int(np.argmax(prob[i]))] for i in cid]
        clf_res.append({"params": p, "cat_acc": cat_metrics_from_labels(y[cid], pcat), "n": len(cid)})
    clf_best = max(clf_res, key=lambda r: r["cat_acc"])
    log["classification"] = {"A: regression -> category (dev acc)": reg_acc, "B: direct classifier": clf_best,
                             "all_classifier_trials": clf_res}
    print(f"category accuracy: A regression {reg_acc} | B classifier {clf_best['cat_acc']}")
    approach = "B" if clf_best["cat_acc"] >= reg_acc + 0.01 else "A"

    selection = {"features": cols, "kind": chosen_kind, "params": chosen_params,
                 "sample_weighting": chosen_weight, "season_aware": season_aware,
                 "category_approach": approach, "classifier_params": clf_best["params"],
                 "selected_on": f"{a} .. {b} (walk-forward monthly)", "frozen_at": date.today().isoformat()}
    log["selection"] = selection
    Path("experiments/selection_v2_candidate.json").write_text(json.dumps(selection, indent=2))
    Path("experiments/experiments_dev.json").write_text(json.dumps(log, indent=2, default=str))
    print("\nFROZEN SELECTION -> experiments/selection_v2_candidate.json")
    print(json.dumps(selection, indent=2))


# ------------------------------------------------------------------ FINAL: evaluate once
def run_final():
    sel = json.loads(Path("experiments/selection_v2_candidate.json").read_text())
    hist, cams = load()
    X, y, dates = build_matrix(hist)
    Xn, yn, dn = build_matrix(hist, wx1_mode="naive")
    assert dn == dates
    end = max(dates)
    a = TEST_START
    wfn = None
    if sel["sample_weighting"] != "none":
        pm0_all = X[:, ALL_FEATURES.index("pm_d0")]
        chg = np.abs(np.log1p(y) - np.log1p(pm0_all))
        wfn = {"winter x2": lambda i: 2.0 if in_winter_window(dates[i]) else 1.0,
               "sudden-change (1 + 2|dlog|)": lambda i: 1.0 + 2.0 * chg[i],
               "high-pollution (1 + PM/150)": lambda i: 1.0 + y[i] / 150.0}[sel["sample_weighting"]]

    print(f"FINAL TEST {a} .. {end} (evaluated once with the frozen selection)")
    new = walk_forward(X, y, dates, sel["features"], a, end, kind=sel["kind"], params=sel["params"], weight_fn=wfn)
    old = walk_forward(X, y, dates, BASE, a, end, params=OLD_PARAMS, early_stop=True)
    ids = sorted(set(new) & set(old))
    pm0 = X[:, ALL_FEATURES.index("pm_d0")]
    m7 = X[:, ALL_FEATURES.index("pm_mean7")]
    m7 = np.where(np.isnan(m7), pm0, m7)
    preds = {"new model": np.array([new[i] for i in ids]), "old (v1) model": np.array([old[i] for i in ids]),
             "tomorrow = today": pm0[ids], "7-day average": m7[ids]}
    if sel["category_approach"] == "B":
        prob = walk_forward(X, y, dates, sel["features"], a, end, task="clf", params=sel["classifier_params"])
        clf_cats = {i: CATS[int(np.argmax(prob[i]))] for i in prob}
    yy = y[ids]
    dd = [dates[i] for i in ids]
    rep = {"test_period": [a.isoformat(), end.isoformat()], "selection": sel, "n_days": len(ids)}

    def block(filter_fn=None):
        sub = [k for k, d in enumerate(dd) if filter_fn is None or filter_fn(d)]
        res = {name: metrics(yy[sub], p[sub]) for name, p in preds.items()}
        cid = [k for k in sub if dd[k] in cams]
        if cid:
            res["CAMS global model"] = metrics(yy[cid], [cams[dd[k]] for k in cid])
        return res

    rep["overall"] = block()
    rep["winter_windows_15Oct_15Feb"] = block(in_winter_window)
    rep["by_winter"] = {}
    for yr in (2022, 2023, 2024, 2025):
        f = lambda d, yr=yr: date(yr, 10, 15) <= d <= date(yr + 1, 2, 15)  # noqa: E731
        if any(f(d) for d in dd):
            rep["by_winter"][f"{yr}-{str(yr + 1)[2:]}"] = block(f)
    rep["by_season"] = {SEASON_NAMES[s]: block(lambda d, s=s: season_of(d) == s) for s in range(4)}
    change = np.abs(yy - pm0[ids]) / np.maximum(pm0[ids], 1)
    spike = {dd[k] for k in range(len(dd)) if change[k] >= 0.3}
    rep["sudden_change_days (>=30% day-to-day)"] = block(lambda d: d in spike)
    poor = {dd[k] for k in range(len(dd)) if yy[k] > POOR_PM25_THRESHOLD}
    rep["poor_or_worse_days"] = block(lambda d: d in poor)
    new_cats = [pm25_to_category(v) for v in preds["new model"]]
    rep["confusion_new_model (rows=true, cols=pred)"] = {"labels": CATS, "matrix": confusion(yy, new_cats)}
    rep["confusion_persistence"] = {"labels": CATS, "matrix": confusion(yy, [pm25_to_category(v) for v in pm0[ids]])}
    by_cat = {}
    for c in CATS:
        k = [j for j in range(len(yy)) if pm25_to_category(yy[j]) == c]
        if k:
            by_cat[c] = {"n": len(k), "new_model_recall": round(np.mean([new_cats[j] == c for j in k]), 3),
                         "persistence_recall": round(np.mean([pm25_to_category(pm0[ids][j]) == c for j in k]), 3)}
    rep["accuracy_by_true_category"] = by_cat
    if sel["category_approach"] == "B":
        cc = [clf_cats.get(i) for i in ids]
        rep["direct_classifier_cat_acc"] = cat_metrics_from_labels(yy, cc)

    # direct-classifier comparison on TEST is reported for transparency even if A was chosen
    prob = walk_forward(X, y, dates, sel["features"], a, end, task="clf", params=sel["classifier_params"])
    cid = [i for i in ids if i in prob]
    rep["approach_B_direct_classifier_on_test (not selected unless chosen on dev)"] = {
        "cat_acc": cat_metrics_from_labels(y[cid], [CATS[int(np.argmax(prob[i]))] for i in cid]), "n": len(cid)}

    # weather-forecast realism checks
    naive = walk_forward(Xn, yn, dn, sel["features"], a, end, kind=sel["kind"], params=sel["params"], weight_fn=wfn)
    no_wx1 = [c for c in sel["features"] if not c.startswith("wx1_") and not c.startswith("wx_d")]
    nowx = walk_forward(X, y, dates, no_wx1, a, end, kind=sel["kind"], params=sel["params"], weight_fn=wfn)
    rep["weather_realism"] = {
        "observed tomorrow weather (upper bound)": metrics(yy, preds["new model"]),
        "tomorrow weather := today's weather (no-skill forecast)": metrics(yy, [naive[i] for i in ids]),
        "model without tomorrow-weather features (lower bound)": metrics(yy, [nowx[i] for i in ids]),
        "tomorrow = today": metrics(yy, pm0[ids]),
    }
    # paired bootstrap (weekly blocks): new vs old, new vs persistence
    def boot(p_a, p_b, n_boot=2000):
        gain = np.abs(p_b - yy) - np.abs(p_a - yy)
        base_err = np.abs(p_b - yy)
        blocks = [np.arange(i, min(i + 7, len(yy))) for i in range(0, len(yy), 7)]
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(n_boot):
            ix = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
            vals.append(100 * gain[ix].sum() / base_err[ix].sum())
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"mae_reduction_pct": round(100 * gain.sum() / base_err.sum(), 1),
                "ci95": [round(float(lo), 1), round(float(hi), 1)]}
    rep["new_vs_old"] = boot(preds["new model"], preds["old (v1) model"])
    rep["new_vs_persistence"] = boot(preds["new model"], preds["tomorrow = today"])
    rep["old_vs_persistence"] = boot(preds["old (v1) model"], preds["tomorrow = today"])

    # ablation re-scored on TEST (transparency only; nothing is chosen from these numbers)
    abl = []
    cols = list(BASE)
    abl.append(("1. current (v1) features", cols))
    for g, feats in GROUPS.items():
        cols = cols + feats
        abl.append((f"+ {g}", cols))
    rep["ablation_on_test_for_transparency"] = []
    for name, c in abl:
        p = walk_forward(X, y, dates, c, a, end, params=OLD_PARAMS)
        k = sorted(p)
        rep["ablation_on_test_for_transparency"].append({"step": name, **metrics(y[k], [p[i] for i in k])})

    Path("experiments/experiments_final.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("overall", "winter_windows_15Oct_15Feb", "new_vs_old",
                                          "new_vs_persistence", "weather_realism")}, indent=1))


if __name__ == "__main__":
    {"dev": run_dev, "final": run_final}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
