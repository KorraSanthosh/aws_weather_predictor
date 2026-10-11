"""Fresh-training study: many model families, selected ONLY by blocked time-series cross-validation on 2018-2024, then
scored ONCE on 2025-01-01 -> 2026-10-07 (the "predict 2025 and 2026" test).

    python model/dev_cv_study.py --stage dev       # CV on targets < 2025-01-01; writes experiments/dev_cv_selection.json
    python model/dev_cv_study.py --stage locked    # reads that selection, trains on <= 2024, scores 2025-26 once

Rules (set before running):
  * Cross-validation is BLOCKED forward-chaining (6 expanding-window folds, sklearn TimeSeriesSplit), never shuffled
    K-fold: shuffled folds let neighbouring days leak across the split. `--stage dev` also measures how optimistic
    shuffled K-fold would have been.
  * Everything (hyper-parameters, family, ensemble members) is chosen on 2018-2024 only. Nothing is looked at on 2025-26
    until `--stage locked`, which uses the selection file exactly as written.
  * Target is the log-change from today's PM2.5 (as v1/v2). Features are the two stored sets (v1 25, v2 35). Models that
    cannot take missing values get median imputation + a missing-indicator (fit on training rows only).
  * 2025-26 has been examined before for v1/v2 (it is inside the earlier 2023+ comparison) but nothing here was designed
    on it. It is not a clean hold-out; winter 2026-27 (model/EVALUATION_PROTOCOL.md) remains the only clean evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pipeline"))
warnings.filterwarnings("ignore")
import compare_versions as cv  # noqa: E402
from core.aqi import POOR_PM25_THRESHOLD, pm25_to_category  # noqa: E402
from core.daily import DP_KEYS  # noqa: E402
from core.features import FEATURE_SETS  # noqa: E402
from train import load_daily  # noqa: E402

LOCK_START = date(2025, 1, 1)
CATS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
SEL = Path("experiments/dev_cv_selection.json")
OUT_DEV, OUT_LOCKED = Path("experiments/dev_cv_study.json"), Path("experiments/locked_2025_26_result.json")
V1, V2 = FEATURE_SETS["v1_25_baseline"], FEATURE_SETS["v2_35_candidate"]


# ------------------------------------------------------------------ data
def load():
    hist, _ = load_daily("data/daily.csv")
    live = cv.build_live_view()
    for d in hist:                       # offline view: origins the 23:00 Lambda could not rebuild are still usable here
        if d not in live and hist[d]["pm25"] is not None:
            live[d] = {"pm25": hist[d]["pm25"], **{k: float("nan") for k in DP_KEYS}}
    rows, _, y, targets, _, _ = cv.build_rows(hist, live, only_live_valid=False)
    X = {"v1": cv.matrix(rows, V1), "v2": cv.matrix(rows, V2)}
    return X, np.asarray(y, float), np.array(targets)


# ------------------------------------------------------------------ models (each: factory(params) -> object with fit/predict on log-delta)
def _imputed(estimator):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), estimator)


class XGB:
    def __init__(self, **p):
        self.p = p

    def fit(self, X, d):
        import xgboost as xgb
        self.m = xgb.XGBRegressor(tree_method="hist", random_state=42, n_jobs=2, **self.p).fit(X, d, verbose=False)
        return self

    def predict(self, X):
        return self.m.predict(X)


class HGB:
    def __init__(self, **p):
        self.p = p

    def fit(self, X, d):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(random_state=42, early_stopping=False, **self.p).fit(X, d)
        return self

    def predict(self, X):
        return self.m.predict(X)


class ET:
    def __init__(self, **p):
        self.p = p

    def fit(self, X, d):
        from sklearn.ensemble import ExtraTreesRegressor
        self.m = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=2, **self.p).fit(X, d)
        return self

    def predict(self, X):
        return self.m.predict(X)


class Ridge:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, d):
        from sklearn.linear_model import Ridge as R
        self.m = _imputed(R(alpha=self.alpha)).fit(X, d)
        return self

    def predict(self, X):
        return self.m.predict(X)


class MLP:
    """Small neural net (2 hidden layers), averaged over 5 seeds; fixed epochs + L2, no shuffled validation split."""
    def __init__(self, alpha, hidden=(32, 16)):
        self.alpha, self.hidden = alpha, hidden

    def fit(self, X, d):
        from sklearn.neural_network import MLPRegressor
        self.ms = [_imputed(MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=0.003, max_iter=300,
                                         random_state=s)).fit(X, d) for s in range(5)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)


def xgb_space(rng):
    return dict(n_estimators=int(rng.choice([150, 250, 400, 600])), learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08])),
                max_depth=int(rng.integers(2, 7)), subsample=float(rng.uniform(0.6, 1.0)),
                colsample_bytree=float(rng.uniform(0.5, 1.0)), min_child_weight=int(rng.integers(1, 11)),
                reg_lambda=float(rng.choice([0.5, 1, 2, 5, 10])), reg_alpha=float(rng.choice([0, 0.1, 0.5, 1, 2])),
                objective=str(rng.choice(["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"])))


def candidate_grid(n_xgb=30):
    rng = np.random.default_rng(0)
    c = {}
    base = dict(n_estimators=243, learning_rate=0.03, max_depth=4, subsample=0.85, colsample_bytree=0.85,
                min_child_weight=3, reg_lambda=2.0)
    for fs in ("v1", "v2"):
        c[f"xgb_default_{fs}"] = (fs, lambda b=base: XGB(**b))
        for i in range(n_xgb):
            p = xgb_space(rng)
            c[f"xgb_rs{i:02d}_{fs}"] = (fs, lambda p=p: XGB(**p))
        for lr, depth, leaf, l2 in [(0.03, 3, 20, 1.0), (0.05, 4, 10, 1.0), (0.03, 5, 20, 5.0)]:
            c[f"hgb_lr{lr}_d{depth}_l{leaf}_{fs}"] = (fs, lambda a=(lr, depth, leaf, l2): HGB(
                learning_rate=a[0], max_depth=a[1], min_samples_leaf=a[2], l2_regularization=a[3], max_iter=250,
                loss="absolute_error" if a[3] == 5.0 else "squared_error"))
        for leaf in (3, 8, 20):
            for mf in (0.5, 1.0):
                c[f"et_leaf{leaf}_mf{mf}_{fs}"] = (fs, lambda a=(leaf, mf): ET(min_samples_leaf=a[0], max_features=a[1]))
        for a in (1, 10, 100, 1000):
            c[f"ridge_a{a}_{fs}"] = (fs, lambda a=a: Ridge(a))
        for a in (0.01, 0.1, 1.0, 3.0):
            c[f"mlp_a{a}_{fs}"] = (fs, lambda a=a: MLP(a))
    return c


# ------------------------------------------------------------------ evaluation helpers
def delta_target(y, pm0):
    return np.log1p(y) - np.log1p(pm0)


def to_pm(pm0, delta):
    return np.expm1(np.log1p(pm0) + delta)


def stats(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    tc, pc = [pm25_to_category(v) for v in y], [pm25_to_category(v) for v in p]
    pos_t, pos_p = y > POOR_PM25_THRESHOLD, p > POOR_PM25_THRESHOLD
    return {"n": int(len(y)), "mae": round(float(np.abs(e).mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
            "r2": round(1 - float((e ** 2).sum()) / float(((y - y.mean()) ** 2).sum()), 3),
            "cat_acc": round(float(np.mean([a == b for a, b in zip(tc, pc)])), 3),
            "within_one": round(float(np.mean([abs(CATS.index(a) - CATS.index(b)) <= 1 for a, b in zip(tc, pc)])), 3),
            "poor_recall": round(float((pos_t & pos_p).sum() / max(1, pos_t.sum())), 3),
            "poor_precision": round(float((pos_t & pos_p).sum() / max(1, pos_p.sum())), 3)}


def blocked_folds(n, k=6):
    from sklearn.model_selection import TimeSeriesSplit
    return list(TimeSeriesSplit(n_splits=k).split(np.arange(n)))


def cv_predict(make, X, y, dev_idx, folds):
    """Out-of-fold PM2.5 predictions {row index: value} over blocked folds (train strictly before each test block)."""
    pm0 = X[:, FEATURE_SETS_IDX]
    out = {}
    for tr, te in folds:
        a, b = dev_idx[tr], dev_idx[te]
        m = make().fit(X[a], delta_target(y[a], pm0[a]))
        out.update(zip(b.tolist(), to_pm(pm0[b], m.predict(X[b])).tolist()))
    return out


FEATURE_SETS_IDX = V1.index("pm_d0")           # pm_d0 is column 0 in both sets
assert V2.index("pm_d0") == FEATURE_SETS_IDX


def classifier_cv(X, y, dev_idx, folds):
    import xgboost as xgb
    cls = np.array([CATS.index(pm25_to_category(v)) for v in y])
    out = {}
    for tr, te in folds:
        a, b = dev_idx[tr], dev_idx[te]
        present = np.unique(cls[a]); remap = {c: i for i, c in enumerate(present)}
        m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.85,
                              tree_method="hist", random_state=42, n_jobs=2, objective="multi:softprob")
        m.fit(X[a], np.array([remap[c] for c in cls[a]]))
        pred = present[m.predict(X[b])]
        out.update(zip(b.tolist(), pred.tolist()))
    return out


# ------------------------------------------------------------------ stage 1: development CV
def stage_dev():
    X, y, t = load()
    dev = np.where(t < LOCK_START)[0]
    folds = blocked_folds(len(dev), 6)
    pm0 = X["v1"][:, FEATURE_SETS_IDX]
    print(f"dev rows {len(dev)} ({t[dev].min()} -> {t[dev].max()}); locked rows untouched: {int((t >= LOCK_START).sum())}")
    print("blocked folds (test windows):", [(str(t[dev[te[0]]]), str(t[dev[te[-1]]]), len(te)) for _, te in folds])
    test_idx = np.concatenate([dev[te] for _, te in folds])
    results, oof = {}, {}
    grid = candidate_grid()
    for i, (name, (fs, make)) in enumerate(grid.items()):
        pr = cv_predict(make, X[fs], y, dev, folds)
        oof[name] = pr
        idx = np.array(sorted(pr))
        results[name] = {"features": fs, **stats(y[idx], np.array([pr[j] for j in idx]))}
        fold_mae = [float(np.mean(np.abs(np.array([pr[j] for j in dev[te]]) - y[dev[te]]))) for _, te in folds]
        results[name]["fold_mae"] = [round(v, 1) for v in fold_mae]
        if (i + 1) % 10 == 0 or i == len(grid) - 1:
            print(f"  {i + 1}/{len(grid)} done; latest {name}: MAE {results[name]['mae']}", flush=True)
    persist = stats(y[test_idx], pm0[test_idx])
    # how optimistic would shuffled K-fold have been? (same model, same rows)
    from sklearn.model_selection import KFold
    shuf = {}
    for nm in ("xgb_default_v1", "xgb_default_v2"):
        fs, make = grid[nm]
        pr = {}
        for tr, te in KFold(6, shuffle=True, random_state=0).split(dev):
            a, b = dev[tr], dev[te]
            m = make().fit(X[fs][a], delta_target(y[a], pm0[a]))
            pr.update(zip(b.tolist(), to_pm(pm0[b], m.predict(X[fs][b])).tolist()))
        idx = np.array(sorted(pr))
        shuf[nm] = stats(y[idx], np.array([pr[j] for j in idx]))
    # classifier route (category accuracy only)
    clf = {}
    for fs in ("v1", "v2"):
        pr = classifier_cv(X[fs], y, dev, folds)
        idx = np.array(sorted(pr))
        truth = np.array([CATS.index(pm25_to_category(v)) for v in y[idx]])
        clf[f"xgb_classifier_{fs}"] = {"cat_acc": round(float(np.mean(np.array([pr[j] for j in idx]) == truth)), 3),
                                       "within_one": round(float(np.mean(np.abs(np.array([pr[j] for j in idx]) - truth) <= 1)), 3)}
    # ---- selection rule (fixed in advance): best config per family by pooled blocked-CV MAE
    fam = lambda n: n.split("_")[0]                                              # noqa: E731
    best = {}
    for n, r in results.items():
        f = fam(n)
        if f not in best or r["mae"] < results[best[f]]["mae"]:
            best[f] = n
    ranked = sorted(best.values(), key=lambda n: results[n]["mae"])
    members = ranked[:3]
    ens = np.mean([[oof[m][j] for j in sorted(oof[members[0]])] for m in members], axis=0)
    idx = np.array(sorted(oof[members[0]]))
    results["ENSEMBLE_top3_families"] = {"members": members, **stats(y[idx], ens)}
    best_single = ranked[0]
    best_class = max(clf, key=lambda k: clf[k]["cat_acc"])
    selection = {"finalists": {"best_single": best_single, "ensemble_top3": members, "best_mlp": best["mlp"],
                               "best_classifier": best_class},
                 "per_family_best": best, "rule": "best per family by pooled blocked-CV MAE; ensemble = unweighted mean of the 3 best families",
                 "dev_rows": int(len(dev))}
    SEL.write_text(json.dumps(selection, indent=1))
    OUT_DEV.write_text(json.dumps({"folds": [(str(t[dev[te[0]]]), str(t[dev[te[-1]]]), len(te)) for _, te in folds],
                                   "persistence_on_cv_rows": persist, "shuffled_kfold_optimism": shuf,
                                   "candidates": results, "classifiers": clf, "selection": selection}, indent=1))
    print("\nPersistence on CV rows:", persist["mae"], persist["cat_acc"])
    print("Shuffled K-fold (leaky) vs blocked:", {k: (v["mae"], v["cat_acc"]) for k, v in shuf.items()},
          {k: (results[k]["mae"], results[k]["cat_acc"]) for k in shuf})
    print("Best per family:", {f: (n, results[n]["mae"], results[n]["cat_acc"]) for f, n in best.items()})
    print("Ensemble:", results["ENSEMBLE_top3_families"]["mae"], results["ENSEMBLE_top3_families"]["cat_acc"], "| classifiers:", clf)
    print("wrote", SEL, OUT_DEV)


# ------------------------------------------------------------------ stage 2: locked 2025-26 test, once
def fit_pred(make, Xs, y, pm0, tr, te):
    m = make().fit(Xs[tr], delta_target(y[tr], pm0[tr]))
    return to_pm(pm0[te], m.predict(Xs[te]))


def stage_locked():
    sel = json.loads(SEL.read_text())
    X, y, t = load()
    grid = candidate_grid()
    pm0 = X["v1"][:, FEATURE_SETS_IDX]
    dev = np.where(t < LOCK_START)[0]
    lock = np.where(t >= LOCK_START)[0]
    members = sel["finalists"]["ensemble_top3"]
    named = {"v1_25_baseline(default params)": "xgb_default_v1", "v2_35_candidate(default params)": "xgb_default_v2",
             "best_single:" + sel["finalists"]["best_single"]: sel["finalists"]["best_single"],
             "best_mlp:" + sel["finalists"]["best_mlp"]: sel["finalists"]["best_mlp"]}
    models = {k: [v] for k, v in named.items()}
    models["ensemble_top3:" + "+".join(members)] = members
    months = list(cv.months(LOCK_START, max(t)))
    pred_static, pred_wf = {}, {}
    for label, mem in models.items():
        ps, pw = [], []
        for name in mem:
            fs, make = grid[name]
            ps.append(fit_pred(make, X[fs], y, pm0, dev, lock))                      # trained once on <= 2024
            wf = np.full(len(lock), np.nan)
            for m0 in months:                                                       # monthly refit on everything earlier
                m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
                te = np.where((t[lock] >= m0) & (t[lock] < m1))[0]
                if len(te):
                    tr = np.where(t < m0)[0]
                    wf[te] = fit_pred(make, X[fs], y, pm0, tr, lock[te])
            pw.append(wf)
        pred_static[label], pred_wf[label] = np.mean(ps, axis=0), np.mean(pw, axis=0)
        print("done", label, flush=True)
    yy, dts = y[lock], list(t[lock])
    winter = np.array([(d.month, d.day) >= (10, 15) or (d.month, d.day) <= (2, 15) for d in dts])
    out = {"what": "locked test 2025-01-01 -> 2026-10-07; selection fixed in dev stage; scored once", "n_days": int(len(lock)),
           "note": "all of these days have hourly day-part data; 2025-26 was examined before for v1/v2 (not for these models)",
           "persistence": stats(yy, pm0[lock]), "selection": sel["finalists"]}
    for proto, preds in (("static_trained_through_2024", pred_static), ("walk_forward_monthly_refit", pred_wf)):
        blk = {}
        base = preds["v1_25_baseline(default params)"]
        for label, p in preds.items():
            blk[label] = {"all": stats(yy, p), "winter_15Oct_15Feb": stats(yy[winter], p[winter]),
                          "vs_v1_mae": cv.paired_mae_change(yy, base, p, dts)}
        out[proto] = blk
    OUT_LOCKED.write_text(json.dumps(out, indent=1))
    print("persistence", out["persistence"])
    for proto in ("static_trained_through_2024", "walk_forward_monthly_refit"):
        print("\n==", proto)
        for label, r in out[proto].items():
            a, w, c = r["all"], r["winter_15Oct_15Feb"], r["vs_v1_mae"]
            print(f"  {label[:60]:60} MAE {a['mae']:>6} RMSE {a['rmse']:>6} R2 {a['r2']:>6} cat {a['cat_acc']} +-1 {a['within_one']} | "
                  f"winter MAE {w['mae']} cat {w['cat_acc']} | vs v1 {c['pct']:+.1f}% {c['ci95']}")
    print("wrote", OUT_LOCKED)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "locked"], required=True)
    args = ap.parse_args()
    stage_dev() if args.stage == "dev" else stage_locked()
