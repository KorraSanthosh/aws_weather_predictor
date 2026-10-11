"""Domain-knowledge feature study (35 features) - fixed in advance, evaluated once.

    python model/feature_study.py

The deployed set (core/features.py FEATURE_SETS['v2_35_candidate']) was written down BEFORE any result was looked at; nothing is
tuned on the test. It adds (a) day-part PM2.5 from hourly data (Delhi's daily cycle: night build-up under the
inversion, morning rush, school/mixing hours, evening rush) and (b) binary calendar/weather flags with Delhi domain
meaning (stubble-burning season, Diwali window, winter inversion, weekend, rain, stagnant air, fog, NW wind).

Compared PAIRED on identical days against the previous 25-feature model (same model recipe), using the production
feature code and data/daily.csv. Day-part features exist only where hourly data exist (2018-2022, 2025-02 onward;
2023-24 is CPCB daily files), so the day-part effect is judged on days that HAVE them (HOURLY test) and the
flag-only effect on all days. Output: experiments/feature_study_35.json
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.features import BASE, DAYPART_FEATURES, FEATURE_SETS, FEATURES_V1, FLAG_FEATURES, feature_dict  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from experiments import OLD_PARAMS, metrics, months  # noqa: E402
from train import load_daily  # noqa: E402

TEST_START = date(2023, 1, 1)
HOURLY_TEST_START = date(2025, 3, 1)
FEATURES_35 = FEATURE_SETS["v2_35_candidate"]
FEATURES_FLAGS_ONLY = [f for f in FEATURES_35 if f not in DAYPART_FEATURES]     # 28: no hourly data needed
FEATURES_V1_PLUS_DAYPARTS = list(FEATURES_V1) + DAYPART_FEATURES                   # 32 (post-hoc ablation)
assert len(FEATURES_35) == 35


def build(hist):
    rows, y, dates = [], [], []
    for d0 in sorted(hist):
        d1 = d0 + timedelta(days=1)
        nxt, cur = hist.get(d1), hist[d0]
        if not nxt or nxt["pm25"] is None or cur["pm25"] is None:
            continue
        if any(nxt[k] is None for k in WEATHER_KEYS):
            continue
        rows.append(feature_dict(hist, d0, {k: nxt[k] for k in WEATHER_KEYS}))
        y.append(nxt["pm25"]); dates.append(d1)
    hasdp = np.array([r["pm_school"] == r["pm_school"] and r["pm_late"] == r["pm_late"] for r in rows])
    return rows, np.array(y), dates, hasdp


def walk(X, y, dates, base_idx, start, end):
    import xgboost as xgb
    delta = np.log1p(y) - np.log1p(X[:, base_idx])
    da = np.array(dates)
    out = {}
    for m0 in months(start, end):
        m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
        te = np.where((da >= m0) & (da < m1) & (da >= start) & (da <= end))[0]
        tr = np.where(da < m0)[0]
        if len(te) == 0 or len(tr) < 150:
            continue
        model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42,
                                 early_stopping_rounds=40, n_jobs=4, **OLD_PARAMS)
        model.fit(X[tr[:-60]], delta[tr[:-60]], eval_set=[(X[tr[-60:]], delta[tr[-60:]])], verbose=False)
        pred = np.expm1(np.log1p(X[te, base_idx]) + model.predict(X[te]))
        for i, p in zip(te, pred):
            out[i] = float(p)
    return out


def boot_gain(y, pa, pb, n=2000, seed=0):
    """% change in MAE of B vs A (negative = B better), paired bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    base = (eb.mean() - ea.mean()) / ea.mean() * 100
    g = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        g.append((eb[i].mean() - ea[i].mean()) / ea[i].mean() * 100)
    return round(base, 2), [round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)]


def main():
    hist, _ = load_daily()
    rows, y, dates, hasdp = build(hist)
    end = max(dates)
    mat = lambda cols: np.array([[r[n] for n in cols] for r in rows], float)  # noqa: E731
    sets = {"v1_25": list(FEATURES_V1), "flags_28": FEATURES_FLAGS_ONLY, "v1+dayparts_32": FEATURES_V1_PLUS_DAYPARTS,
            "full_35": FEATURES_35}
    preds = {k: walk(mat(c), y, dates, c.index("pm_d0"), TEST_START, end) for k, c in sets.items()}
    pm0 = np.array([r["pm_d0"] for r in rows])
    print(f"rows {len(y)}; with day-part data {int(hasdp.sum())}; test start {TEST_START}")

    def block(name, ids):
        ids = sorted(ids)
        yy = y[ids]
        res = {"n": len(ids), "persistence": metrics(yy, pm0[ids])}
        for k in sets:
            res[k] = metrics(yy, [preds[k][i] for i in ids])
        pa = np.array([preds["v1_25"][i] for i in ids])
        for k in ("flags_28", "v1+dayparts_32", "full_35"):
            res[f"mae_change_pct_{k}_vs_v1"] = boot_gain(yy, pa, np.array([preds[k][i] for i in ids]))
        print(f"\n== {name}: n={len(ids)}")
        for k in ["persistence"] + list(sets):
            m_ = res[k]
            print(f"  {k:16s} MAE {m_['mae']:6.2f}  cat_acc {m_['cat_acc']:.3f}  R2 {m_['r2']}  within1 {m_['within_one_cat']}  poor_recall {m_['poor_recall']}")
        for k in ("flags_28", "v1+dayparts_32", "full_35"):
            print(f"  MAE change vs v1 for {k}: {res[f'mae_change_pct_{k}_vs_v1']}  (negative = better)")
        return res

    common = [i for i in preds["v1_25"] if all(i in p for p in preds.values())]
    out = {"features_35": FEATURES_35, "protocol": "walk-forward monthly refit; 35-feature set fixed in advance; compared once; "
           "the 32-feature row is a post-hoc ablation", "ALL_2023+": block("ALL 2023+ (day-parts missing in 2023-24)", common)}
    hrs = [i for i in common if dates[i] >= HOURLY_TEST_START and hasdp[i]]
    out["HOURLY_2025-03+"] = block("HOURLY days 2025-03+ (day-parts present)", hrs)
    win = [i for i in hrs if (dates[i].month, dates[i].day) >= (10, 15) or (dates[i].month, dates[i].day) <= (2, 15)]
    out["HOURLY_winter"] = block("HOURLY winter days (15 Oct-15 Feb)", win)
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/feature_study_35.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote experiments/feature_study_35.json")


if __name__ == "__main__":
    main()
