"""Isolated experiment: can CatBoost beat the project's XGBoost models for next-day Delhi PM2.5?

    PYTHONDONTWRITEBYTECODE=1 .venv-catboost/bin/python experiments/catboost/run_catboost.py --stage tune
    PYTHONDONTWRITEBYTECODE=1 .venv-catboost/bin/python experiments/catboost/run_catboost.py --stage test

Everything this script writes is inside experiments/catboost/.  It only READS the project (data, features, hybrid router,
back-test helpers); it never writes a model file, never retrains a stored artifact, never touches the default.
Rules are in config.json (declared before any result was looked at); `--stage test` refuses to run if the tuning file
was made from a different config.  RETROSPECTIVE: winters 2022-23 .. 2025-26 were examined before, so nothing here is an
untouched hold-out.  Winter 2026-27 is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT, ROOT / "model", ROOT / "pipeline"):
    sys.path.insert(0, str(p))

import forecast_training_experiment as fte  # noqa: E402
import option_screen as osn  # noqa: E402
import winter_backtest as wb  # noqa: E402
from compare_versions import build_live_view  # noqa: E402
from core.features import FEATURE_SETS  # noqa: E402
from core.hybrid import daypart_valid  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from stats import Stats, sudden_mask  # noqa: E402

CONFIG_PATH = HERE / "config.json"
TUNING_PATH = HERE / "tuning.json"
RESULTS_PATH = HERE / "results.json"
V1N, V2N = FEATURE_SETS["v1_25_baseline"], FEATURE_SETS["v2_35_candidate"]
NAMES = {"v1": V1N, "v2": V2N}
FAMS = ("xgb", "cb_base", "cb_tuned")
VERS = ("v1", "v2", "hybrid")
MIN_TRAIN, MIN_DAYS = 365, 30
HARD_END = date(2026, 10, 7)               # last day of data; winter 2026-27 (from 2026-10-15) is never read


def config():
    return json.loads(CONFIG_PATH.read_text())


def config_hash():
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


# ------------------------------------------------------------------ CatBoost model (same recipe shape as the XGBoost one)
def cb_params(cfg, fixed, seed):
    kw = dict(iterations=fixed["iterations"], learning_rate=cfg["learning_rate"], depth=cfg["depth"], l2_leaf_reg=cfg["l2_leaf_reg"],
              loss_function=fixed["loss_function"], random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False,
              od_type="Iter", od_wait=fixed["od_wait"], use_best_model=True)
    if cfg["bootstrap"] == "Bernoulli":
        kw.update(bootstrap_type="Bernoulli", subsample=fixed["bernoulli_subsample"])
    else:
        kw.update(bootstrap_type="Bayesian", bagging_temperature=1.0)
    return kw


def cb_fit(X, y, names, cfg, fixed, seed):
    """Target = log-change from today.  Last 60 rows = chronological early-stopping set (as the XGBoost recipe)."""
    from catboost import CatBoostRegressor
    delta = np.log1p(y) - np.log1p(X[:, names.index("pm_d0")])
    m = CatBoostRegressor(**cb_params(cfg, fixed, seed))
    m.fit(X[:-60], delta[:-60], eval_set=(X[-60:], delta[-60:]))
    return m


def pm_from(model, X, names):
    return np.expm1(np.log1p(X[:, names.index("pm_d0")]) + model.predict(X))


def fit_family(fam, train, seed, cfgs, fixed):
    if fam == "xgb":
        return fte.fit_models(train, seed)
    ts = sorted(train)
    y = np.array([train[t]["y"] for t in ts])
    return {tag: cb_fit(osn.M([train[t]["row"] for t in ts], NAMES[tag]), y, NAMES[tag], cfgs[fam], fixed, seed) for tag in ("v1", "v2")}


def predict_family(models, rows):
    out = {}
    for tag in ("v1", "v2"):
        out[tag] = pm_from(models[tag], osn.M(rows, NAMES[tag]), NAMES[tag])
    out["hybrid"] = osn.hybrid(out["v1"], out["v2"], rows)
    return out


def best_iters(models, fam):
    if fam == "xgb":
        return {t: int(models[t].best_iteration + 1) for t in models}
    return {t: int(models[t].get_best_iteration() + 1) for t in models}


# ------------------------------------------------------------------ data
def load(daily, extra, fc_path="data/raw/weather_prevruns_2024_2026.json"):
    hist = osn.load_hist(daily)
    live = build_live_view(Path("data/raw"), extra_subs=tuple(extra))
    fc = osn.load_forecasts(fc_path)
    obs = lambda t, d0: {k: hist[t].get(k) for k in WEATHER_KEYS}       # noqa: E731
    pers = lambda t, d0: {k: hist[d0].get(k) for k in WEATHER_KEYS}     # noqa: E731
    train_obs = osn.build(hist, live, obs, view="offline")
    test_pers = osn.build(hist, live, pers)
    test_fc = osn.build(hist, live, lambda t, d0: fc[1].get(t))
    for d in (train_obs, test_pers, test_fc):                            # nothing after the last data day is ever read
        assert all(t <= HARD_END for t in d)
    return {"hist": hist, "live": live, "train_obs": train_obs, "test_pers": test_pers, "test_fc": test_fc}


# ------------------------------------------------------------------ stage 1: tuning on development data only
def grid_configs(cfg):
    g = cfg["tuning"]["grid"]
    return [dict(depth=d, learning_rate=lr, l2_leaf_reg=l2, bootstrap=b)
            for d, lr, l2, b in product(g["depth"], g["learning_rate"], g["l2_leaf_reg"], g["bootstrap"])]


def cfg_key(c):
    return f"d{c['depth']}_lr{c['learning_rate']}_l2{c['l2_leaf_reg']}_{c['bootstrap']}"


def pick_tuned(rows, baseline_cfg):
    """Lowest pooled out-of-fold MAE; configs within 0.1 percent of the best are tied and go to the baseline-like one,
    then the shallower, then the slower learning rate (the simplest)."""
    best = min(r["oof_mae"] for r in rows)
    tied = [r for r in rows if r["oof_mae"] <= best * 1.001]
    tied.sort(key=lambda r: (cfg_key(r["config"]) != cfg_key(baseline_cfg), r["config"]["depth"], r["config"]["learning_rate"],
                             r["config"]["bootstrap"] != "Bayesian", r["config"]["l2_leaf_reg"]))
    return tied[0]


def stage_tune():
    from sklearn.model_selection import TimeSeriesSplit
    cfg = config()
    tcfg = cfg["tuning"]
    cutoff = date.fromisoformat(tcfg["cutoff_target_date_exclusive"])
    D = load(cfg["data"]["primary_daily_csv"], cfg["data"]["primary_extra_sources"])
    dev_t = sorted(t for t in D["train_obs"] if t < cutoff)
    first_test = min(wb.winter_window(Y)[0] for Y in (2022, 2023, 2024, 2025))
    assert cutoff <= first_test and all(t < first_test for t in dev_t), "tuning would touch a test window"
    X = osn.M([D["train_obs"][t]["row"] for t in dev_t], V1N)
    y = np.array([D["train_obs"][t]["y"] for t in dev_t])
    folds = list(TimeSeriesSplit(n_splits=6).split(np.arange(len(dev_t))))
    fold_info = [{"test_from": dev_t[te[0]].isoformat(), "test_to": dev_t[te[-1]].isoformat(), "n_test": int(len(te)), "n_train": int(len(tr))}
                 for tr, te in folds]
    test_idx = np.concatenate([te for _, te in folds])

    def oof(fit_pred):
        pr = np.full(len(y), np.nan)
        for tr, te in folds:
            pr[te] = fit_pred(tr, te)
        return pr

    def xgb_fp(tr, te):
        m = osn.fit(X[tr], y[tr], V1N)
        return osn.predict(m, X[te], V1N)

    t0 = time.time()
    pr_x = oof(xgb_fp)
    rows = []
    preds = {"xgb": pr_x}
    for c in grid_configs(cfg):
        t1 = time.time()

        def cb_fp(tr, te, c=c):
            m = cb_fit(X[tr], y[tr], V1N, c, tcfg["fixed"], 42)
            return pm_from(m, X[te], V1N)

        pr = oof(cb_fp)
        preds[cfg_key(c)] = pr
        fm = [float(np.mean(np.abs(pr[te] - y[te]))) for _, te in folds]
        rows.append({"config": c, "key": cfg_key(c), "oof_mae": float(np.mean(np.abs(pr[test_idx] - y[test_idx]))),
                     "oof_rmse": float(np.sqrt(np.mean((pr[test_idx] - y[test_idx]) ** 2))), "fold_mae": fm, "seconds": round(time.time() - t1, 1)})
        print(f"  {cfg_key(c):32s} MAE {rows[-1]['oof_mae']:.3f}  ({rows[-1]['seconds']}s)")
    base_key = cfg_key(tcfg["baseline_config"])
    chosen = pick_tuned(rows, tcfg["baseline_config"])
    ti = [dev_t[i] for i in test_idx]
    P = {"xgb": pr_x[test_idx], "cb_base": preds[base_key][test_idx], "cb_tuned": preds[chosen["key"]][test_idx], "persistence": np.array([X[i, V1N.index("pm_d0")] for i in test_idx])}
    st = Stats(y[test_idx], P, ti)
    out = {"config_sha256": config_hash(), "dev_rows": len(dev_t), "dev_first_last": [dev_t[0].isoformat(), dev_t[-1].isoformat()],
           "cutoff_exclusive": cutoff.isoformat(), "folds": fold_info, "scored_oof_rows": int(len(test_idx)),
           "xgboost_reference": {**st.point("xgb"), "mae_ci95": st.model("xgb")["mae_ci95"]},
           "grid": sorted(rows, key=lambda r: r["oof_mae"]), "baseline_key": base_key, "selected": chosen,
           "cb_base_oof": st.point("cb_base"), "cb_tuned_oof": st.point("cb_tuned"), "persistence_oof": st.point("persistence"),
           "tuned_minus_xgb_dev": st.diff("xgb", "cb_tuned"), "base_minus_xgb_dev": st.diff("xgb", "cb_base"),
           "tuned_minus_base_dev": st.diff("cb_base", "cb_tuned"), "seconds_total": round(time.time() - t0, 1),
           "note": "features = v1 (25); weather = observed (development years have no forecast archive); V2 reuses the same hyper-parameters"}
    TUNING_PATH.write_text(json.dumps(fte.jsonable(out), indent=1))
    print("selected", chosen["key"], f"oof MAE {chosen['oof_mae']:.3f}  xgboost {out['xgboost_reference']['mae']:.3f}")


# ------------------------------------------------------------------ stage 2: frozen-fit head-to-head
def fit_window(D, start, test_sets, seeds, cfgs, fixed, min_days=MIN_DAYS):
    """One frozen fit per (family, seed) on rows with target < start; predicts every supplied test set."""
    tr = {t: v for t, v in D["train_obs"].items() if t < start}
    assert all(t < start for t in tr)
    if len(tr) < MIN_TRAIN:
        return {"status": f"not run: {len(tr)} training origins (< {MIN_TRAIN})"}
    out = {"status": "scored", "train_origins": len(tr), "train_last_target": max(tr).isoformat(), "sets": {}, "best_iterations": {}}
    for name, (days, rows) in test_sets.items():
        out["sets"][name] = {"days": days, "rows": rows, "pred": {f: {} for f in FAMS}}
    for fam in FAMS:
        for seed in seeds:
            models = fit_family(fam, tr, seed, cfgs, fixed)
            out["best_iterations"].setdefault(fam, {})[seed] = best_iters(models, fam)
            for name, (days, rows) in test_sets.items():
                out["sets"][name]["pred"][fam][seed] = predict_family(models, rows)
    return out


def assemble(window_sets, key, seed=None, avg=False, seeds=()):
    """Concatenate windows -> y, pm0, has_dp, months, dates, {model name: predictions}.  avg=True averages the seeds' predictions."""
    y, pm0, dp, mo, dates, P = [], [], [], [], [], {}
    per_dates = []
    for w in window_sets:
        s = w["sets"][key]
        rows = s["rows"]
        y += list(s["y"])
        pm0 += [r["pm_d0"] for r in rows]
        dp += [bool(daypart_valid(r)) for r in rows]
        mo += [d.month for d in s["days"]]
        per_dates.append(list(s["days"]))
        dates += list(s["days"])
        for fam in FAMS:
            for v in VERS:
                arr = (np.mean([s["pred"][fam][sd][v] for sd in seeds], axis=0) if avg else s["pred"][fam][seed][v])
                P.setdefault(f"{fam}_{v}", []).extend(list(arr))
        P.setdefault("persistence", []).extend([r["pm_d0"] for r in rows])
        P.setdefault("mean7", []).extend([r["pm_mean7"] for r in rows])
    return (np.array(y), np.array(pm0), np.array(dp, bool), np.array(mo), per_dates, {k: np.array(v) for k, v in P.items()})


def block(y, P, dates):
    st = Stats(y, P, dates)
    out = {"n_days": st.n, "models": {k: st.model(k) for k in P} if st.ok else {k: st.point(k) for k in P}}
    cmp = {}
    if st.ok:
        for fam in ("cb_base", "cb_tuned"):
            for v in VERS:
                cmp[f"{fam}_vs_xgb_{v}"] = st.diff(f"xgb_{v}", f"{fam}_{v}")
        cmp["xgb_hybrid_minus_xgb_v1"] = st.diff("xgb_v1", "xgb_hybrid")
        cmp["cb_tuned_hybrid_minus_cb_tuned_v1"] = st.diff("cb_tuned_v1", "cb_tuned_hybrid")
        cmp["xgb_v2_minus_xgb_v1"] = st.diff("xgb_v1", "xgb_v2")
        cmp["cb_tuned_v2_minus_cb_tuned_v1"] = st.diff("cb_tuned_v1", "cb_tuned_v2")
    out["comparisons"] = cmp
    return out


def subsets(y, pm0, dp, mo, P, pdates, kind, thr):
    """Blocks for hourly-data availability, sudden changes and seasons.  pdates = pooled synthetic dates (array of date)."""
    masks = {"days_with_all_7_day_parts": dp, "days_without_day_parts": ~dp}
    sd = sudden_mask(y, pm0, thr)
    masks["sudden_change_days_ge30pct"], masks["stable_days_lt30pct"] = sd, ~sd
    if kind == "winter":
        masks["IMD post-monsoon (15 Oct-30 Nov)"] = np.isin(mo, (10, 11))
        masks["IMD winter (1 Dec-15 Feb)"] = np.isin(mo, (12, 1, 2))
    else:
        masks["IMD winter (Dec-Feb)"] = np.isin(mo, (12, 1, 2))
        masks["IMD pre-monsoon (Mar-May)"] = np.isin(mo, (3, 4, 5))
        masks["IMD monsoon (Jun-Sep)"] = np.isin(mo, (6, 7, 8, 9))
        masks["IMD post-monsoon (Oct-Nov)"] = np.isin(mo, (10, 11))
    out = {}
    for name, m in masks.items():
        idx = np.where(m)[0]
        if len(idx) < 10:
            out[name] = {"n_days": int(len(idx)), "note": "too few days for intervals"}
            continue
        out[name] = block(y[idx], {k: v[idx] for k, v in P.items()}, [pdates[i] for i in idx])
    return out


def analyse_windows(windows, key, kind, seeds, thr, label):
    """windows: {name: fit_window result}.  Returns per-window blocks + pooled block + subsets + seed-averaged pooled block."""
    scored = {n: w for n, w in windows.items() if w.get("status") == "scored" and key in w["sets"] and len(w["sets"][key]["days"]) >= MIN_DAYS}
    res = {"label": label, "windows_scored": list(scored), "per_window": {},
           "windows_skipped": {n: (w.get("status") if w.get("status") != "scored" else
                                   ("no archived forecast for this winter" if key not in w["sets"] else f"fewer than {MIN_DAYS} scoreable days"))
                               for n, w in windows.items() if n not in scored}}
    ws = list(scored.values())
    for n, w in scored.items():
        y, pm0, dp, mo, per, P = assemble([w], key, seed=42)
        b = block(y, P, per[0])
        b.update({"train_origins": w["train_origins"], "train_last_target": w["train_last_target"], "best_iterations": w["best_iterations"],
                  "days_with_day_parts": int(dp.sum())})
        res["per_window"][n] = b
    if not ws:
        return res
    y, pm0, dp, mo, per, P = assemble(ws, key, seed=42)
    pd_ = np.array(osn.pooled_dates(per), dtype=object) if kind == "winter" else np.array(per[0], dtype=object)
    res["pooled"] = block(y, P, list(pd_))
    res["pooled"]["days_with_day_parts"] = int(dp.sum())
    res["subsets"] = subsets(y, pm0, dp, mo, P, list(pd_), kind, thr)
    ya, pma, dpa, moa, pera, Pa = assemble(ws, key, avg=True, seeds=seeds)
    res["pooled_seed_averaged"] = block(ya, Pa, list(pd_))
    res["per_seed_pooled_mae"] = {str(sd): {f"{fam}_{v}": float(np.abs(np.concatenate([w["sets"][key]["pred"][fam][sd][v] for w in ws]) - y).mean())
                                           for fam in FAMS for v in VERS} for sd in seeds}
    res["daily"] = {n: {"dates": [d.isoformat() for d in w["sets"][key]["days"]], "actual": [round(float(v), 2) for v in w["sets"][key]["y"]],
                        **{f"{fam}_{v}": [round(float(x), 2) for x in w["sets"][key]["pred"][fam][42][v]] for fam in FAMS for v in VERS}}
                    for n, w in scored.items()}
    return res


def make_test_sets(D, start, end, with_fc):
    out = {}
    pers_days = [t for t in sorted(D["test_pers"]) if start <= t <= end]
    out["pers"] = (pers_days, [D["test_pers"][t]["row"] for t in pers_days])
    out["pers_y"] = [D["test_pers"][t]["y"] for t in pers_days]
    if with_fc:
        fc_days = [t for t in pers_days if t in D["test_fc"]]
        out["fc"] = (fc_days, [D["test_fc"][t]["row"] for t in fc_days])
        out["fc_y"] = [D["test_fc"][t]["y"] for t in fc_days]
    return out


def run_windows(D, spec, seeds, cfgs, fixed):
    """spec: {name: (train_before, test_start, test_end, with_fc)}."""
    windows = {}
    for name, (start, t0, t1, with_fc) in spec.items():
        ts = make_test_sets(D, t0, t1, with_fc)
        sets = {"pers": ts["pers"]}
        if with_fc:
            sets["fc"] = ts["fc"]
        w = fit_window(D, start, sets, seeds, cfgs, fixed)
        if w.get("status") == "scored":
            w["sets"]["pers"]["y"] = ts["pers_y"]
            if with_fc:
                w["sets"]["fc"]["y"] = ts["fc_y"]
        windows[name] = w
        print(f"  {name}: {w.get('status')}  train {w.get('train_origins')}  test days {len(sets['pers'][0])}")
    return windows


def stage_test(only_primary_data=False):
    cfg = config()
    tun = json.loads(TUNING_PATH.read_text()) if TUNING_PATH.exists() else None
    if not tun or tun["config_sha256"] != config_hash():
        sys.exit("tuning.json is missing or was made from a different config.json - run --stage tune first (the config is frozen).")
    tcfg, T = cfg["tuning"], cfg["test"]
    cfgs = {"cb_base": tcfg["baseline_config"], "cb_tuned": tun["selected"]["config"]}
    fixed, seeds, thr = tcfg["fixed"], T["seeds"], T["sudden_change_threshold"]
    import catboost
    import xgboost
    res = {"title": cfg["title"], "status": "RETROSPECTIVE - winters examined before; not an untouched hold-out", "config_sha256": config_hash(),
           "tuning_sha256": hashlib.sha256(TUNING_PATH.read_bytes()).hexdigest(),
           "environment": {"python": platform.python_version(), "catboost": catboost.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__},
           "configs": {"xgb": "production recipe (model/experiments.py OLD_PARAMS), unchanged", "cb_base": cfgs["cb_base"], "cb_tuned": cfgs["cb_tuned"]},
           "seeds": seeds, "primary_seed": T["primary_seed"], "sudden_change_threshold": thr}
    t0 = time.time()

    print("primary data:", cfg["data"]["primary_daily_csv"])
    D = load(cfg["data"]["primary_daily_csv"], cfg["data"]["primary_extra_sources"])
    win_spec = {f"{Y}-{str(Y + 1)[2:]}": (wb.winter_window(Y)[0], wb.winter_window(Y)[0], wb.winter_window(Y)[1], Y in (2024, 2025)) for Y in (2022, 2023, 2024, 2025)}
    W = run_windows(D, win_spec, seeds, cfgs, fixed)
    res["T1_winters_persistence_weather"] = analyse_windows(W, "pers", "winter", seeds, thr, "T1 primary: four winters, tomorrow's weather := today's observed (leak-free)")
    res["T2_winters_forecast_weather"] = analyse_windows(W, "fc", "winter", seeds, thr, "T2 live-like: 2024-25 and 2025-26, archived 24 h-ahead weather forecast")
    all_spec = {"2025-01-01 -> 2026-10-07": (date(2025, 1, 1), date(2025, 1, 1), HARD_END, False)}
    A = run_windows(D, all_spec, seeds, cfgs, fixed)
    res["T3_all_seasons_persistence_weather"] = analyse_windows(A, "pers", "all", seeds, thr, "T3: frozen fit before 2025-01-01, scored through 2026-10-07 (all IMD seasons)")

    # reproduction check: does the XGBoost arm reproduce the project's existing back-test numbers?
    rep = {}
    osj = ROOT / "experiments" / "option_screen.json"
    if osj.exists():
        ref = json.loads(osj.read_text())["winters"]
        for n, b in res["T1_winters_persistence_weather"]["per_window"].items():
            if n in ref:
                mine, theirs = b["models"]["xgb_hybrid"]["mae"], ref[n]["reference_hybrid_mae"]
                rep[n] = {"this_run": round(mine, 2), "existing_option_screen": theirs, "match": bool(abs(mine - theirs) < 0.01)}
    res["reproduces_existing_xgboost_numbers"] = rep

    # secondary data: exactly what the deployed models were trained on (first test only)
    if not only_primary_data:
        print("secondary data:", cfg["data"]["secondary_daily_csv"])
        D2 = load(cfg["data"]["secondary_daily_csv"], cfg["data"]["secondary_extra_sources"])
        W2 = run_windows(D2, win_spec, seeds, cfgs, fixed)
        res["S1_production_data_persistence_weather"] = analyse_windows(W2, "pers", "winter", seeds, thr, "S1: production data/daily.csv, four winters, persistence weather")

    res["decision"] = decide(res, tun, cfg)
    res["seconds_total"] = round(time.time() - t0, 1)
    RESULTS_PATH.write_text(json.dumps(fte.jsonable(res), indent=1))
    print("wrote", RESULTS_PATH, res["decision"]["verdict"])


def decide(res, tun, cfg):
    rule = cfg["decision_rule"]
    T1 = res["T1_winters_persistence_weather"]
    d = T1["pooled"]["comparisons"]["cb_tuned_vs_xgb_hybrid"]
    wins = {n: b["comparisons"]["cb_tuned_vs_xgb_hybrid"]["mae_pct"] for n, b in T1["per_window"].items()}
    tdev = tun["tuned_minus_xgb_dev"]
    checks = {
        "mae_at_least_2pct_lower": bool(d["mae_pct"] <= -2.0),
        "ci_below_zero_7d_and_14d": bool(d.get("mae_ci95_7d", [0, 1])[1] < 0 and d.get("mae_ci95_14d", [0, 1])[1] < 0),
        "no_winter_more_than_2pct_worse": bool(max(wins.values()) <= 2.0),
        "exact_category_not_lower_by_more_than_2pts": bool(d["cat_acc"] >= -0.02),
        "dev_cv_mae_not_worse_than_xgboost": bool(tdev["mae"] <= 0),
    }
    ok = all(checks.values())
    return {"primary_comparison": rule["primary_comparison"], "mae_diff": d["mae"], "mae_pct": d["mae_pct"], "ci95_7d": d.get("mae_ci95_7d"),
            "ci95_14d": d.get("mae_ci95_14d"), "per_winter_pct": wins, "cat_acc_diff": d["cat_acc"], "dev_cv_mae_diff": tdev["mae"],
            "dev_cv_mae_pct": tdev["mae_pct"], "checks": checks,
            "verdict": "meets the pre-declared rule: eligible for further testing ONLY with your approval" if ok else
            "no evidence of improvement: keep XGBoost; CatBoost stays a documented reference"}


if __name__ == "__main__":  # pragma: no cover
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "test"], required=True)
    ap.add_argument("--primary-data-only", action="store_true")
    a = ap.parse_args()
    stage_tune() if a.stage == "tune" else stage_test(a.primary_data_only)
