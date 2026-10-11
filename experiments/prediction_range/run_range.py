"""Prediction ranges around the Hybrid forecast.  Protocol: config.json (declared before running).  RETROSPECTIVE.
Production code is imported read-only; the point forecast (Hybrid, production recipe, seed 42) is unchanged."""
import json, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "experiments" / "weather_source_comparison"):
    sys.path.insert(0, str(p))
import forecast_training_experiment as fte, option_screen as osn
import run_comparison as rc
from core.features import FEATURE_SETS
from core.aqi import CATEGORIES
from core.weather import WEATHER_KEYS
from stats import cat_index

CFG = json.loads((HERE / "config.json").read_text())
NOM = CFG["nominal_level"]; SENS = CFG["sensitivity_levels"]
DEC = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2}            # Good/Satisfactory -> GO, Moderate -> CAUTION, Poor and worse -> INDOOR
DEC_NAMES = ["GO", "CAUTION", "INDOOR"]
LG = lambda x: np.log1p(np.maximum(x, 0.0))            # noqa: E731


# ------------------------------------------------------------------ fits
def fit_quant(train, v, seed, alphas):
    import xgboost as xgb
    names = FEATURE_SETS[v]; ts = sorted(train)
    y = np.array([train[t]["y"] for t in ts]); X = osn.M([train[t]["row"] for t in ts], names)
    delta = np.log1p(y) - np.log1p(X[:, names.index("pm_d0")])
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(alphas), tree_method="hist", random_state=seed,
                         early_stopping_rounds=40, n_jobs=4, **fte.OLD_PARAMS)
    m.fit(X[:-60], delta[:-60], eval_set=[(X[-60:], delta[-60:])], verbose=False)
    return m, names


def predict_quant(mn, rows):
    m, names = mn
    X = osn.M(rows, names)
    q = m.predict(X)
    base = np.log1p(X[:, names.index("pm_d0")])
    return base[:, None] + q                        # log-space quantiles, shape (n, 2)


def oos(train, rows_by_day, days, seed=42):
    """Point + 5/95 quantile predictions for `days` from models trained on `train` only."""
    pt = fte.fit_models(train, seed)
    a = [(1 - NOM) / 2, 1 - (1 - NOM) / 2]
    q1, q2 = fit_quant(train, fte.V1, seed, a), fit_quant(train, fte.V2, seed, a)
    rows = [rows_by_day[t]["row"] for t in days]
    P = fte.predict_all(pt, rows)
    dp = np.array([fte.daypart_valid(r) for r in rows])
    Q1, Q2 = predict_quant(q1, rows), predict_quant(q2, rows)
    Q = np.where(dp[:, None], Q2, Q1)
    return {"days": days, "y": np.array([rows_by_day[t]["y"] for t in days]), "pm0": np.array([r["pm_d0"] for r in rows]),
            "p": P["hybrid"], "p_v1": P["v1"], "persistence": P["persistence"], "qlog": np.sort(Q, axis=1), "dp": dp}


# ------------------------------------------------------------------ interval methods (log space)
def cq(res, level):
    n = len(res)
    return float(np.quantile(res, min(1.0, level * (1 + 1 / n))))


def m1(cal, alpha):
    r = LG(cal["y"]) - LG(cal["p"])
    return {"kind": "M1", "lo": float(np.quantile(r, alpha / 2)), "hi": cq(r, 1 - alpha / 2)}


BUCKETS = [(0, 60), (60, 120), (120, 1e9)]


def m2(cal, alpha):
    glob = m1(cal, alpha); r = LG(cal["y"]) - LG(cal["p"]); out = {"kind": "M2", "global": glob, "buckets": []}
    for lo, hi in BUCKETS:
        idx = (cal["p"] >= lo) & (cal["p"] < hi)
        out["buckets"].append((lo, hi, ({"lo": float(np.quantile(r[idx], alpha / 2)), "hi": cq(r[idx], 1 - alpha / 2)} if idx.sum() >= 40 else glob)))
    return out


def m3(cal, alpha):
    lo, hi = cal["qlog"][:, 0], cal["qlog"][:, 1]; y = LG(cal["y"])
    s = np.maximum(lo - y, y - hi)
    return {"kind": "M3", "q": cq(s, 1 - alpha)}


def apply(method, d, alpha):
    p = d["p"]
    if method["kind"] == "M1":
        lo, hi = LG(p) + method["lo"], LG(p) + method["hi"]
    elif method["kind"] == "M2":
        lo, hi = np.zeros(len(p)), np.zeros(len(p))
        for a, b, mm in method["buckets"]:
            idx = (p >= a) & (p < b)
            lo[idx], hi[idx] = LG(p[idx]) + mm["lo"], LG(p[idx]) + mm["hi"]
    else:
        lo, hi = d["qlog"][:, 0] - method["q"], d["qlog"][:, 1] + method["q"]
    lo, hi = np.expm1(lo), np.expm1(hi)
    lo = np.minimum(np.maximum(lo, 0.0), p); hi = np.maximum(hi, p)               # the point forecast is always inside its range
    return lo, hi


def sub(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(d["days"]) else v) for k, v in d.items() if k != "days"} | {"days": [d["days"][i] for i in idx]}


def cat_idx(v): return np.array([cat_index(x) for x in v])


def score(d, lo, hi, alpha):
    y, p = d["y"], d["p"]
    cy, cl, ch, cp = cat_idx(y), cat_idx(lo), cat_idx(hi), cat_idx(p)
    inside = (y >= lo) & (y <= hi)
    wink = (hi - lo) + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    cont = (cy >= cl) & (cy <= ch)
    span = ch - cl + 1
    dy = np.array([DEC[c] for c in cy]); dp_ = np.array([DEC[c] for c in cp]); dl = np.array([DEC[c] for c in cl]); dh = np.array([DEC[c] for c in ch])
    naive_lo, naive_hi = np.maximum(cp - 1, 0), np.minimum(cp + 1, 5)
    return {"arrays": {"inside": inside, "cont": cont, "span": span, "single_hit": (span == 1) & cont, "dec_hit_point": dy == dp_, "dec_hit_cautious": dy == dh,
                       "dec_in_range": (dy >= dl) & (dy <= dh), "under_point": dp_ < dy, "under_cautious": dh < dy, "over_point": dp_ > dy, "over_cautious": dh > dy,
                       "naive_cont": (cy >= naive_lo) & (cy <= naive_hi), "single": span == 1},
            "scalars": {"pm25_coverage": float(inside.mean()), "mean_width_ugm3": float((hi - lo).mean()), "median_width_ugm3": float(np.median(hi - lo)), "winkler": float(wink.mean()),
                        "range_accuracy": float(cont.mean()), "mean_categories_in_range": float(span.mean()), "share_single_category_range": float((span == 1).mean()),
                        "accuracy_when_single_category": float(((span == 1) & cont).sum() / max((span == 1).sum(), 1)),
                        "point_exact_category": float((cy == cp).mean()),
                        "naive_point_pm1_category_accuracy": float(((cy >= naive_lo) & (cy <= naive_hi)).mean()), "naive_mean_categories": float((naive_hi - naive_lo + 1).mean()),
                        "decision_accuracy_point": float((dy == dp_).mean()), "decision_accuracy_cautious_from_range_top": float((dy == dh).mean()),
                        "decision_range_contains_actual": float(((dy >= dl) & (dy <= dh)).mean()),
                        "under_warning_rate_point(GO/CAUTION when worse)": float((dp_ < dy).mean()), "under_warning_rate_cautious": float((dh < dy).mean()),
                        "over_warning_rate_point": float((dp_ > dy).mean()), "over_warning_rate_cautious": float((dh > dy).mean())}}


def with_ci(d, sc, keys):
    dates = d["days"]
    out = {}
    for blk in (7, 14):
        W = fte.boot_weights(dates, blk); sw = W.sum(1); ok = sw > 0
        for k in keys:
            b = (W[ok] @ sc["arrays"][k].astype(float)) / sw[ok]
            out[f"{k}_ci95_{blk}d"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out


def summarize(d, lo, hi, alpha):
    sc = score(d, lo, hi, alpha)
    res = dict(sc["scalars"]); res["n_days"] = len(d["y"])
    res.update(with_ci(d, sc, ["inside", "cont", "span", "dec_hit_point", "dec_hit_cautious"]))
    sd = np.abs(d["y"] - d["pm0"]) / np.maximum(d["pm0"], 1.0) >= 0.30
    res["sudden_change_days"] = {"n": int(sd.sum()), "pm25_coverage": float(sc["arrays"]["inside"][sd].mean()) if sd.sum() else None,
                                 "range_accuracy": float(sc["arrays"]["cont"][sd].mean()) if sd.sum() else None,
                                 "mean_categories_in_range": float(sc["arrays"]["span"][sd].mean()) if sd.sum() else None}
    res["stable_days"] = {"n": int((~sd).sum()), "pm25_coverage": float(sc["arrays"]["inside"][~sd].mean()), "range_accuracy": float(sc["arrays"]["cont"][~sd].mean()),
                          "mean_categories_in_range": float(sc["arrays"]["span"][~sd].mean())}
    mo = {}
    for m in range(1, 13):
        idx = np.array([i for i, t in enumerate(d["days"]) if t.month == m])
        if len(idx):
            mo[str(m)] = {"n": int(len(idx)), "range_accuracy": float(sc["arrays"]["cont"][idx].mean()), "mean_categories": float(sc["arrays"]["span"][idx].mean())}
    res["by_month"] = mo
    return res


def make(kind, cal, a):
    return {"M1": m1, "M2": m2, "M3": m3}[kind](cal, a)


def main():
    hist = osn.load_hist("data/daily_opencity.csv")
    live = fte.build_live_view(Path("data/raw"), extra_subs=("opencity_hourly",))
    era = lambda t: {k: hist[t].get(k) for k in WEATHER_KEYS}
    src = rc.load_source("production_default")
    all_train = fte.build(hist, live, era, view="offline")
    test_live = fte.build(hist, live, lambda t, f=src[1]: f.get(t))              # archived 24 h forecast weather, live-view features
    res = {"label": "RETROSPECTIVE; point forecast unchanged; see config.json", "config": CFG}
    tr23 = {t: v for t, v in all_train.items() if t <= date(2023, 12, 31)}
    tr24 = {t: v for t, v in all_train.items() if t <= date(2024, 12, 31)}
    tr25w = {t: v for t, v in all_train.items() if t <= date(2025, 10, 14)}
    d24 = sorted(t for t in test_live if date(2024, 1, 1) <= t <= date(2024, 12, 31))
    d25a = sorted(t for t in test_live if date(2025, 1, 1) <= t <= date(2025, 10, 14))
    d25 = sorted(t for t in test_live if date(2025, 1, 1) <= t <= date(2025, 12, 31))
    dw = sorted(t for t in test_live if date(2025, 10, 15) <= t <= date(2026, 2, 15))
    C24 = oos(tr23, test_live, d24)
    T25 = oos(tr24, test_live, d25)
    Tw = oos(tr25w, test_live, dw)
    idx = [i for i, t in enumerate(T25["days"]) if t <= date(2025, 10, 14)]
    C25a = sub(T25, np.array(idx))
    res["sizes"] = {"calibration_2024_days": len(d24), "calibration_2025_to_oct14_days": len(idx), "test_2025_days": len(d25), "test_winter_2025_26_days": len(dw),
                    "first_calibration_day": d24[0].isoformat()}
    cat = lambda *ds: {k: (np.concatenate([x[k] for x in ds]) if isinstance(ds[0][k], np.ndarray) and k != "days" else None) for k in ds[0] if k != "days"} | {"days": sum([x["days"] for x in ds], [])}
    cal_w = cat(C24, C25a)
    # ---- method choice on the calibration period only
    h1 = [i for i, t in enumerate(C24["days"]) if t <= date(2024, 6, 30)]; h2 = [i for i, t in enumerate(C24["days"]) if t > date(2024, 6, 30)]
    A, B = sub(C24, np.array(h1)), sub(C24, np.array(h2))
    alpha = 1 - NOM; choice = {}
    for k in ("M1", "M2", "M3"):
        mm = make(k, A, alpha); lo, hi = apply(mm, B, alpha); s = score(B, lo, hi, alpha)["scalars"]
        choice[k] = {"coverage": s["pm25_coverage"], "winkler": s["winkler"], "gap_to_nominal": abs(s["pm25_coverage"] - NOM), "mean_width": s["mean_width_ugm3"]}
    best = sorted(choice, key=lambda k: (round(choice[k]["gap_to_nominal"], 2), choice[k]["winkler"]))[0]
    res["method_choice_on_2024_calibration_split"] = {"scores": choice, "chosen": best, "rule": CFG["method_choice_rule"]}
    # ---- evaluation (all methods reported; chosen one flagged)
    res["tests"] = {}
    for tname, Td, Cd in (("calendar_2025", T25, C24), ("winter_2025_26", Tw, cal_w)):
        out = {"cal_days": len(Cd["y"]), "test_days": len(Td["y"]), "first": Td["days"][0].isoformat(), "last": Td["days"][-1].isoformat(),
               "calibration_last_day": Cd["days"][-1].isoformat(), "calibration_before_test": bool(max(Cd["days"]) < min(Td["days"])), "methods": {}}
        for k in ("M1", "M2", "M3"):
            out["methods"][k] = {}
            for lev in [NOM] + (SENS if k in ("M1", "M2") else []):
                a = 1 - lev
                if k == "M3" and lev != NOM: continue
                mm = make(k, Cd, a); lo, hi = apply(mm, Td, a)
                out["methods"][k][f"{int(lev * 100)}%"] = summarize(Td, lo, hi, a)
        out["point_forecast"] = {"hybrid_mae": float(np.abs(Td["p"] - Td["y"]).mean()), "hybrid_exact_category": float((cat_idx(Td["p"]) == cat_idx(Td["y"])).mean()),
                                 "hybrid_within_one": float((np.abs(cat_idx(Td["p"]) - cat_idx(Td["y"])) <= 1).mean())}
        # chosen method daily output
        mm = make(best, Cd, 1 - NOM); lo, hi = apply(mm, Td, 1 - NOM)
        out["daily_chosen"] = {"dates": [t.isoformat() for t in Td["days"]], "actual": Td["y"].round(1).tolist(), "point": Td["p"].round(1).tolist(), "lo": lo.round(1).tolist(), "hi": hi.round(1).tolist()}
        res["tests"][tname] = out
    # ---- gate
    g = CFG["gate"]; chosen = {t: res["tests"][t]["methods"][best][f"{int(NOM * 100)}%"] for t in res["tests"]}
    res["gate"] = {"chosen_method": best, "per_test": {t: {"range_accuracy": c["range_accuracy"], "range_accuracy_ci95_7d": c["cont_ci95_7d"], "mean_categories_in_range": c["mean_categories_in_range"],
                                                            "passes_range_accuracy": bool(c["range_accuracy"] >= g["range_accuracy_min"]),
                                                            "passes_narrower_than_naive": bool(c["mean_categories_in_range"] < g["mean_categories_in_range_max"])} for t, c in chosen.items()}}
    res["gate"]["passes_all"] = bool(all(v["passes_range_accuracy"] and v["passes_narrower_than_naive"] for v in res["gate"]["per_test"].values()))
    return res


if __name__ == "__main__":
    r = main()
    (HERE / "results.json").write_text(json.dumps(fte.jsonable(r), indent=1))
    print(json.dumps(r["method_choice_on_2024_calibration_split"], indent=0)[:600]); print(json.dumps(r["gate"], indent=0)[:900])
