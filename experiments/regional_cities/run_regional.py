"""Do PM2.5 readings from NCR / upwind cities improve the next-day Delhi forecast?   (protocol: config.json)

    python experiments/regional_cities/run_regional.py [--seeds 42 7 123 2024 99] [--out metrics.json]

RETROSPECTIVE.  2025 and winter 2025-26 were examined in earlier experiments, so this is NOT an untouched hold-out.
The protocol and the pass / fail rule are in config.json, written before any regional model was run (its sha256 is stored in
the output).  Production code is imported read-only; nothing outside experiments/regional_cities/ is written, no model
file is saved, the default model and the Lambda are untouched.

What is compared (same days, same recipe, same seeds):
    ctrl  the unchanged V1 / V2 feature lists                               (what is deployed / experimental today)
    Rlag  + 12 regional features from the previous COMPLETE day (origin - 1)  PRIMARY: needs no live-latency assumption
    Rd0   + 12 regional features from the origin day (hours <= 22 at test)    SECONDARY: assumes regional data by 23:00
    plc   + the Rlag columns permuted among days of the same month            PLACEBO: effect of merely adding 12 columns
The Delhi target and every Delhi input are unchanged; regional stations are kept in their own folder and never enter the Delhi mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "experiments" / "weather_source_comparison"):
    sys.path.insert(0, str(p))

import build_dataset as bd                                   # noqa: E402  (read-only: load_station)
import forecast_training_experiment as fte                   # noqa: E402
import option_screen as osn                                  # noqa: E402
import run_comparison as rc                                  # noqa: E402
from core.daily import readings_to_daily, to_ist             # noqa: E402
from core.features import FEATURE_SETS                       # noqa: E402
from core.hybrid import daypart_valid                        # noqa: E402
from core.weather import WEATHER_KEYS                        # noqa: E402
from stats import Stats, cat_index, sudden_mask                         # noqa: E402

CFG_PATH = HERE / "config.json"
CFG = json.loads(CFG_PATH.read_text())
NAN = float("nan")
GROUPS = ("ncr_near", "haryana_nw", "punjab", "up_east")
FEATS = ("level", "rel", "chg")
MIN_GROUP_STATIONS = 2
LIVE_CUTOFF_HOUR = 22
V1, V2 = fte.V1, fte.V2
RAW = ROOT / "data" / "raw"
REGIONAL = RAW / "regional"
DELHI_DIRS = ("openaq", "openaq_api", "openaq_recent", "opencity_hourly", "opencity_2024_25", "cpcb")
TESTS = {
    "T1_calendar_2025": {"train_end": date(2024, 12, 31), "start": date(2025, 1, 1), "end": date(2025, 12, 31)},
    "T2_winter_2025_26": {"train_end": date(2025, 10, 14), "start": date(2025, 10, 15), "end": date(2026, 2, 15)},
}
assert all(t["end"] <= fte.HARD_STOP for t in TESTS.values())
PREFIX = {"Rlag": "r1", "Rd0": "r0", "plc": "p1"}
EXTRA = {"ctrl": [], **{arm: [f"{pre}_{g}_{f}" for g in GROUPS for f in FEATS] for arm, pre in PREFIX.items()}}
ARMS = tuple(EXTRA)
MODELS = ("v1", "v2")
SERIES = [m if a == "ctrl" else f"{m}_{a}" for a in ARMS for m in MODELS + ("hybrid",)]
BASELINES = ("persistence", "mean7")


# ------------------------------------------------------------------ regional daily values
def load_regional(cutoff=None):
    """({group: {day: group mean}}, {group: {day: n stations}}).  A station-day follows the Delhi rule (>= 16 valid
    hours, India Standard Time); a group-day needs >= MIN_GROUP_STATIONS station-days, otherwise it is absent (NaN later)."""
    stations = json.loads((HERE / "stations.json").read_text())["stations"]
    acc = {g: defaultdict(list) for g in GROUPS}
    for st in stations:
        readings = bd.load_station(REGIONAL / f"{st['id']}.csv")
        if cutoff is not None:
            readings = [(ts, v) for ts, v in readings if to_ist(ts).hour <= cutoff]
        for day, (mean, _n) in readings_to_daily(readings).items():
            acc[st["group"]][day].append(mean)
    val = {g: {d: float(np.mean(v)) for d, v in a.items() if len(v) >= MIN_GROUP_STATIONS} for g, a in acc.items()}
    cnt = {g: {d: len(v) for d, v in a.items() if len(v) >= MIN_GROUP_STATIONS} for g, a in acc.items()}
    return val, cnt


class Spy(dict):
    """dict that remembers every key it was asked for (leakage audit)."""

    def __init__(self, d):
        super().__init__(d)
        self.seen = set()

    def get(self, k, default=None):
        self.seen.add(k)
        return super().get(k, default)

    def __getitem__(self, k):
        self.seen.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.seen.add(k)
        return super().__contains__(k)


def reg_feats(store_a, store_b, a: date, b: date, delhi_pm, prefix: str) -> dict:
    """12 regional numbers.  Reads group values of day `a` (from store_a) and day `b` (from store_b) only.
    level = log1p(group_a);  rel = log1p(group_a) - log1p(Delhi city value of the same day);  chg = log1p(group_a) - log1p(group_b)."""
    out = {}
    for g in GROUPS:
        va, vb = store_a[g].get(a), store_b[g].get(b)
        la = math.log1p(va) if va is not None else NAN
        out[f"{prefix}_{g}_level"] = la
        out[f"{prefix}_{g}_rel"] = la - math.log1p(delhi_pm) if (va is not None and delhi_pm == delhi_pm) else NAN
        out[f"{prefix}_{g}_chg"] = la - math.log1p(vb) if (va is not None and vb is not None) else NAN
    return out


def augment(base: dict, S: dict, view: str) -> dict:
    """Copy of {t: {"y","row"}} whose rows also hold the Rlag and Rd0 columns.  Nothing else in a row is changed.
    Rlag: previous complete day (origin-1) vs the day before it.  Rd0: origin day (train = full day, test = hours <= 22)."""
    one = timedelta(days=1)
    store0 = S["full"] if view == "train" else S["live"]
    out = {}
    for t, v in base.items():
        d0 = t - one
        row = dict(v["row"])
        row.update(reg_feats(S["full"], S["full"], d0 - one, d0 - 2 * one, row["pm_d1"], "r1"))
        row.update(reg_feats(store0, S["full"], d0, d0 - one, row["pm_d0"], "r0"))
        out[t] = {"y": v["y"], "row": row}
    return out


def placebo_map(keys):
    """Fixed permutation of target dates WITHIN each calendar month (seed 0)."""
    rng = np.random.default_rng(0)
    by_month = defaultdict(list)
    for t in sorted(keys):
        by_month[t.month].append(t)
    pi = {}
    for _m, ts in sorted(by_month.items()):
        perm = rng.permutation(len(ts))
        for i, t in enumerate(ts):
            pi[t] = ts[int(perm[i])]
    return pi


def add_placebo(sets: list, pi: dict, vec: dict):
    for d in sets:
        for t, v in d.items():
            src = vec[pi[t]]
            v["row"].update({k.replace("r1_", "p1_", 1): x for k, x in src.items()})


def n_groups(row, prefix="r1"):
    return sum(1 for g in GROUPS if row[f"{prefix}_{g}_level"] == row[f"{prefix}_{g}_level"])


# ------------------------------------------------------------------ models
def fit_arm(train, names, seed):
    import xgboost as xgb
    ts = sorted(train)
    X = osn.M([train[t]["row"] for t in ts], names)
    y = np.array([train[t]["y"] for t in ts])
    delta = np.log1p(y) - np.log1p(X[:, names.index("pm_d0")])
    m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed,
                         early_stopping_rounds=40, n_jobs=4, **fte.OLD_PARAMS)
    m.fit(X[:-60], delta[:-60], eval_set=[(X[-60:], delta[-60:])], verbose=False)
    return m


def names_for(model, arm):
    return FEATURE_SETS[V1 if model == "v1" else V2] + EXTRA[arm]


def run_seed(train, rows, seed, arms=ARMS):
    """{series: predictions} for one seed.  Hybrid = V2 if all 7 day-part inputs exist else V1, per arm."""
    P, fitted = {}, {}
    for a in arms:
        pa = {}
        for m in MODELS:
            nm = names_for(m, a)
            mod = fit_arm(train, nm, seed)
            fitted[(m, a)] = mod
            pa[m] = osn.predict(mod, osn.M(rows, nm), nm)
        pa["hybrid"] = osn.hybrid(pa["v1"], pa["v2"], rows)
        for m in MODELS + ("hybrid",):
            P[m if a == "ctrl" else f"{m}_{a}"] = pa[m]
    return P, fitted


# ------------------------------------------------------------------ leakage audit and canary
def same(a, b):
    return all((x == y) or (x != x and y != y) for x, y in zip(a, b))


def leakage_audit(test: dict, S: dict):
    """Rebuild the regional columns of every test day from spying stores and check which dates were read."""
    one = timedelta(days=1)
    bad, n_diff_live = [], 0
    for t, v in test.items():
        d0 = t - one
        sf = {g: Spy(S["full"][g]) for g in GROUPS}
        r1 = reg_feats(sf, sf, d0 - one, d0 - 2 * one, v["row"]["pm_d1"], "r1")
        seen1 = {k for g in GROUPS for k in sf[g].seen}
        sl = {g: Spy(S["live"][g]) for g in GROUPS}
        sf0 = {g: Spy(S["full"][g]) for g in GROUPS}
        r0 = reg_feats(sl, sf0, d0, d0 - one, v["row"]["pm_d0"], "r0")
        seen0 = {k for g in GROUPS for k in sl[g].seen} | {k for g in GROUPS for k in sf0[g].seen}
        ok_vals = same([v["row"][k] for k in r1], list(r1.values())) and same([v["row"][k] for k in r0], list(r0.values()))
        if max(seen1) > d0 - one or max(seen0) > d0 or not ok_vals:
            bad.append({"target": t.isoformat(), "max_read_lag": max(seen1).isoformat(), "max_read_d0": max(seen0).isoformat(), "values_match": ok_vals})
        n_diff_live += int(any(S["live"][g].get(d0) != S["full"][g].get(d0) for g in GROUPS if g in S["live"]))
    return {"rows_checked": len(test), "rows_reading_too_late": bad, "no_date_after_allowed_day_read": not bad,
            "rows_where_live_origin_day_differs_from_full_day": n_diff_live}


def canary(train, test_rows, y, seed=42):
    """A feature that IS the answer (log1p of the target day's Delhi value) must collapse the error: proves a leak would show."""
    tr = {t: {"y": v["y"], "row": {**v["row"], "canary": math.log1p(v["y"])}} for t, v in train.items()}
    rows = [{**r, "canary": math.log1p(yy)} for r, yy in zip(test_rows, y)]
    nm = FEATURE_SETS[V1] + ["canary"]
    mod = fit_arm(tr, nm, seed)
    p = osn.predict(mod, osn.M(rows, nm), nm)
    return float(np.abs(p - y).mean())


# ------------------------------------------------------------------ evaluation
def verdict(res):
    """Pre-declared rule (config.json: decision_rule), v1+Rlag vs v1, both tests."""
    comp, per = {}, {}
    for tn, b in res["tests"].items():
        d = b["diffs"]["v1_Rlag_vs_v1"]
        gain = -d["mae_pct"]
        mc, mr, mp = b["models"]["v1"]["mae"], b["models"]["v1_Rlag"]["mae"], b["models"]["v1_plc"]["mae"]
        sa = b["seed_averaged"]["v1_Rlag_vs_v1"]
        per[tn] = {
            "1_mae_gain_pct": round(gain, 2), "1_pass": gain >= CFG["decision_rule"]["min_mae_gain_pct"],
            "2_ci95_7d": d.get("mae_ci95_7d"), "2_ci95_14d": d.get("mae_ci95_14d"),
            "2_pass": bool(d.get("mae_ci95_7d") and d.get("mae_ci95_14d") and d["mae_ci95_7d"][1] < 0 and d["mae_ci95_14d"][1] < 0),
            "3_cat_acc_change": round(d["cat_acc"], 4), "3_pass": d["cat_acc"] >= 0,
            "4_placebo_gain": round(mc - mp, 3), "4_real_gain": round(mc - mr, 3), "4_pass": bool((mc - mp) < 0.5 * (mc - mr)),
            "5_seed_avg_mae_diff": round(sa["mae"], 3), "5_pass": bool(sa["mae"] < 0 and d["mae"] < 0),
        }
    helpful = all(v[f"{i}_pass"] for v in per.values() for i in range(1, 6))
    return {"rule": CFG["decision_rule"]["text"], "per_test": per, "regional_data_helpful": bool(helpful),
            "answer": "HELPFUL under the pre-declared rule (still only a hypothesis for the winter 2026-27 shadow test)" if helpful
            else "NO EVIDENCE OF IMPROVEMENT under the pre-declared rule"}


def main(seeds=(42, 7, 123, 2024, 99), out=HERE / "metrics.json", verbose=True):
    os.chdir(ROOT)
    seeds = tuple(seeds)
    assert seeds[0] == CFG["seeds"]["primary"]
    hist = osn.load_hist("data/daily_opencity.csv")
    live = fte.build_live_view(RAW, extra_subs=("opencity_hourly",))
    src = rc.load_source("production_default")
    era = lambda t: {k: hist[t].get(k) for k in WEATHER_KEYS}                        # noqa: E731

    # ---- regional data (two views) and integrity checks on the data
    full_v, full_n = load_regional(None)
    live_v, _ = load_regional(LIVE_CUTOFF_HOUR)
    S = {"full": full_v, "live": live_v}
    reg_ids = {str(s["id"]) for s in json.loads((HERE / "stations.json").read_text())["stations"]}
    delhi_ids = {p.stem for d in DELHI_DIRS if (RAW / d).exists() for p in (RAW / d).glob("*.csv")}
    res = {"label": "RETROSPECTIVE; 2025 and winter 2025-26 had been examined before; not an untouched test",
           "config_sha256": hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(), "seeds": list(seeds),
           "protocol": __doc__.strip(), "tests": {}, "integrity": {
               "regional_station_ids_in_delhi_folders": sorted(reg_ids & delhi_ids), "regional_stations": len(reg_ids),
               "delhi_station_files": len(delhi_ids)}}
    assert not (reg_ids & delhi_ids), "a regional station is also a Delhi station"
    cov = {}
    for g in GROUPS:
        by_year = defaultdict(int)
        for d in full_v[g]:
            by_year[d.year] += 1
        cov[g] = {"stations": sum(1 for s in json.loads((HERE / "stations.json").read_text())["stations"] if s["group"] == g),
                  "group_days_by_year": {str(k): v for k, v in sorted(by_year.items())},
                  "median_stations_per_group_day": float(np.median(list(full_n[g].values()))) if full_n[g] else None}
    res["regional_coverage"] = cov

    # ---- rows
    train_base = fte.build(hist, live, era, view="offline")                       # ERA5 weather, full-day "today" (production training rows)
    test_base = fte.build(hist, live, lambda t: src[1].get(t))                    # archived 24 h forecast, 23:00 view of "today"
    all_train = augment(train_base, S, "train")
    test_aug = augment(test_base, S, "test")
    vec = {}
    for d in (all_train, test_aug):
        for t, v in d.items():
            vec.setdefault(t, {k: v["row"][k] for k in EXTRA["Rlag"]})
    pi = placebo_map(vec)
    add_placebo([all_train, test_aug], pi, vec)
    res["placebo"] = {"days": len(pi), "fixed_points": int(sum(1 for t, u in pi.items() if t == u)), "seed": 0, "within": "calendar month"}

    for tn, T in TESTS.items():
        train = {t: v for t, v in all_train.items() if t <= T["train_end"]}
        assert train and max(train) <= T["train_end"] and len(train) >= fte.MIN_TRAIN
        test = {t: v for t, v in test_aug.items() if T["start"] <= t <= T["end"]}
        days = sorted(test)
        y = np.array([test[t]["y"] for t in days])
        rows = [test[t]["row"] for t in days]
        pm0 = np.array([r["pm_d0"] for r in rows])
        by_year = defaultdict(lambda: [0, 0])
        for t, v in train.items():
            by_year[t.year][0] += 1
            by_year[t.year][1] += int(n_groups(v["row"]) >= 2)
        blk = {"window": [days[0].isoformat(), days[-1].isoformat()], "n_days": len(days), "train_end": T["train_end"].isoformat(),
               "training": {"origins": len(train), "first_target": min(train).isoformat(), "last_target": max(train).isoformat(),
                            "origins_by_year_with_>=2_regional_groups": {str(k): {"origins": a, "with_regional": b} for k, (a, b) in sorted(by_year.items())},
                            "origins_with_>=2_regional_groups": int(sum(b for _a, b in by_year.values()))}}
        avail = np.array([n_groups(r) for r in rows])
        blk["availability_on_test_days"] = {"days_with_>=2_regional_groups_(Rlag)": int((avail >= 2).sum()), "days_with_all_4": int((avail == 4).sum()),
                                            "days_with_none": int((avail == 0).sum()),
                                            "per_group_days": {g: int(sum(1 for r in rows if r[f'r1_{g}_level'] == r[f'r1_{g}_level'])) for g in GROUPS},
                                            "days_with_>=2_groups_(Rd0)": int(sum(n_groups(r, 'r0') >= 2 for r in rows))}
        blk["leakage_audit"] = leakage_audit(test, S)
        blk["canary_mae_with_future_target_feature"] = canary(train, rows, y)
        # ---- fits
        P_all, fits42 = {}, None
        for sd in seeds:
            P, fitted = run_seed(train, rows, sd)
            P_all[sd] = P
            if sd == seeds[0]:
                fits42 = fitted
            if verbose:
                print(tn, "seed", sd, {k: round(float(np.abs(v - y).mean()), 2) for k, v in P.items() if k.startswith(("v1", "hybrid"))}, flush=True)
        P0 = {**P_all[seeds[0]], "persistence": pm0, "mean7": np.array([r["pm_mean7"] for r in rows])}
        # control identity with the production fitting code
        ref = fte.fit_models({t: {"y": v["y"], "row": v["row"]} for t, v in train.items()}, seeds[0])
        ident = {m: bool(np.array_equal(osn.predict(ref[m], osn.M(rows, FEATURE_SETS[V1 if m == "v1" else V2]), FEATURE_SETS[V1 if m == "v1" else V2]), P0[m]))
                 for m in MODELS}
        blk["control_identical_to_production_fit"] = ident
        st = Stats(y, P0, days)
        blk["models"] = {k: st.model(k) for k in list(P0) if k not in ("v2_Rd0", "v2_plc", "hybrid_plc")}
        blk["models"] = {**blk["models"], **{k: st.point(k) for k in ("v2_Rd0", "v2_plc", "hybrid_plc")}}
        dd = {}
        for m in MODELS + ("hybrid",):
            dd[f"{m}_Rlag_vs_{m}"] = st.diff(m, f"{m}_Rlag")
            dd[f"{m}_Rd0_vs_{m}"] = st.diff(m, f"{m}_Rd0")
            dd[f"{m}_plc_vs_{m}"] = st.diff(m, f"{m}_plc")
            dd[f"{m}_Rlag_vs_plc"] = st.diff(f"{m}_plc", f"{m}_Rlag")
        dd["v1_vs_persistence"] = st.diff("persistence", "v1")
        blk["diffs"] = dd
        # ---- seeds
        blk["seeds"] = {str(sd): {k: float(np.abs(P_all[sd][k] - y).mean()) for k in SERIES} for sd in seeds}
        cy = np.array([cat_index(v) for v in y])
        blk["seeds_cat_acc"] = {str(sd): {k: float(np.mean(np.array([cat_index(v) for v in P_all[sd][k]]) == cy)) for k in SERIES} for sd in seeds}
        blk["five_seed_mean"] = {k: {"cat_acc": float(np.mean([blk["seeds_cat_acc"][str(sd)][k] for sd in seeds])),
                                     "cat_acc_min": float(min(blk["seeds_cat_acc"][str(sd)][k] for sd in seeds)),
                                     "cat_acc_max": float(max(blk["seeds_cat_acc"][str(sd)][k] for sd in seeds)),
                                     "mae": float(np.mean([blk["seeds"][str(sd)][k] for sd in seeds]))} for k in SERIES}
        avg = {k: np.mean([P_all[sd][k] for sd in seeds], axis=0) for k in SERIES}
        sa = Stats(y, avg, days)
        blk["seed_averaged"] = {}
        for m in MODELS + ("hybrid",):
            for a in ("Rlag", "Rd0"):
                blk["seed_averaged"][f"{m}_{a}_vs_{m}"] = sa.diff(m, f"{m}_{a}")
        blk["seed_averaged"]["points"] = {k: sa.point(k) for k in ("v1", "v1_Rlag", "v1_Rd0", "v1_plc", "hybrid", "hybrid_Rlag")}
        blk["seeds_where_v1_Rlag_beats_v1"] = int(sum(blk["seeds"][str(sd)]["v1_Rlag"] < blk["seeds"][str(sd)]["v1"] for sd in seeds))
        blk["seeds_where_hybrid_Rlag_beats_hybrid"] = int(sum(blk["seeds"][str(sd)]["hybrid_Rlag"] < blk["seeds"][str(sd)]["hybrid"] for sd in seeds))
        # ---- supplementary (added after a one-seed look; never decides): train ONLY on origins that have regional inputs,
        #      so the 2023-24 hole cannot dilute the regional columns.  Compares like with like on the same rows.
        train_r = {t: v for t, v in train.items() if n_groups(v["row"]) >= 2}
        if len(train_r) >= fte.MIN_TRAIN:
            PR = {}
            for sd in seeds:
                PR[sd] = {}
                for a_, lab in (("ctrl", "ctrlR"), ("Rlag", "RlagR")):
                    nm_ = names_for("v1", a_)
                    PR[sd][lab] = osn.predict(fit_arm(train_r, nm_, sd), osn.M(rows, nm_), nm_)
            sr = Stats(y, {"persistence": pm0, "ctrlR": PR[seeds[0]]["ctrlR"], "RlagR": PR[seeds[0]]["RlagR"]}, days)
            avgR = Stats(y, {k: np.mean([PR[sd][k] for sd in seeds], axis=0) for k in ("ctrlR", "RlagR")}, days)
            blk["supplementary_restricted_training"] = {
                "note": "v1 trained only on origins with >= 2 regional groups; added after the first one-seed run; does not enter the decision",
                "training_origins": len(train_r), "ctrlR": sr.model("ctrlR"), "RlagR": sr.model("RlagR"),
                "RlagR_vs_ctrlR": sr.diff("ctrlR", "RlagR"), "seed_averaged_RlagR_vs_ctrlR": avgR.diff("ctrlR", "RlagR"),
                "seeds_mae": {str(sd): {k: float(np.abs(PR[sd][k] - y).mean()) for k in ("ctrlR", "RlagR")} for sd in seeds},
                "seeds_where_RlagR_beats_ctrlR": int(sum(np.abs(PR[sd]["RlagR"] - y).mean() < np.abs(PR[sd]["ctrlR"] - y).mean() for sd in seeds))}
        else:
            blk["supplementary_restricted_training"] = {"not_run": f"only {len(train_r)} origins with regional inputs"}
        # ---- subsets (exploratory)
        sudden = sudden_mask(y, pm0, fte.SUDDEN)
        md = [(d.month, d.day) for d in days]
        masks = {"regional_available_ge2_groups": avail >= 2, "no_regional_data": avail == 0,
                 "stubble_period_15Oct-30Nov": np.array([(10, 15) <= x <= (11, 30) for x in md]),
                 "sudden_change_ge30pct": sudden, "stable": ~sudden}
        sub = {}
        for nm_, mk in masks.items():
            idx = np.where(mk)[0]
            if len(idx) < 10:
                sub[nm_] = {"n_days": int(len(idx)), "note": "too few days"}
                continue
            s2 = Stats(y[idx], {k: P0[k][idx] for k in ("persistence", "v1", "v1_Rlag", "v1_Rd0", "v1_plc", "hybrid", "hybrid_Rlag")}, [days[i] for i in idx])
            sub[nm_] = {"n_days": int(len(idx)), **{k: s2.point(k) for k in ("persistence", "v1", "v1_Rlag", "v1_Rd0", "v1_plc", "hybrid", "hybrid_Rlag")},
                        "v1_Rlag_vs_v1": s2.diff("v1", "v1_Rlag"), "hybrid_Rlag_vs_hybrid": s2.diff("hybrid", "hybrid_Rlag")}
        blk["subsets"] = sub
        # ---- what the model used
        gain = fits42[("v1", "Rlag")].get_booster().get_score(importance_type="gain")
        nm = names_for("v1", "Rlag")
        imp = {nm[int(k[1:])]: float(v) for k, v in gain.items()}
        tot = sum(imp.values()) or 1.0
        by_g = {g: sum(v for k, v in imp.items() if k.startswith(f"r1_{g}_")) / tot for g in GROUPS}
        gp = fits42[("v1", "plc")].get_booster().get_score(importance_type="gain")
        nmp = names_for("v1", "plc")
        impp = {nmp[int(k[1:])]: float(v) for k, v in gp.items()}
        totp = sum(impp.values()) or 1.0
        blk["importance_v1_Rlag_seed42"] = {"regional_share_of_total_gain": sum(by_g.values()), "by_group_share": by_g,
                                            "placebo_columns_share_of_total_gain_(same_count_of_columns,_no_information)":
                                                sum(v for k, v in impp.items() if k.startswith("p1_")) / totp,
                                            "top10": sorted(((k, v / tot) for k, v in imp.items()), key=lambda kv: -kv[1])[:10]}
        blk["daily"] = {"dates": [d.isoformat() for d in days], "actual": y.round(2).tolist(),
                        **{k: P0[k].round(2).tolist() for k in ("persistence", "v1", "v1_Rlag", "v1_Rd0", "hybrid", "hybrid_Rlag")}}
        res["tests"][tn] = blk
    res["decision"] = verdict(res)
    # compact derived data (the raw station files are large and git-ignored): one row per group-day
    with open(HERE / "regional_daily.csv", "w") as f:
        f.write("date,group,full_day_mean,stations,hours_le22_mean\n")
        for g in GROUPS:
            for d in sorted(full_v[g]):
                lv = live_v[g].get(d)
                f.write(f"{d.isoformat()},{g},{full_v[g][d]:.3f},{full_n[g][d]},{'' if lv is None else f'{lv:.3f}'}\n")
    out.write_text(json.dumps(fte.jsonable(res), indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CFG["seeds"]["all"]))
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    a = ap.parse_args()
    r = main(a.seeds, Path(a.out))
    print(r["decision"]["answer"])
