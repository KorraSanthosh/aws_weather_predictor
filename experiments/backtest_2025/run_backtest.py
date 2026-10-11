"""Retrospective chronological backtest: train on targets <= 2024-12-31, freeze, evaluate calendar year 2025.

RETROSPECTIVE: 2025 data had been examined in earlier experiments, so this is NOT an untouched independent test.
No tuning on 2025: production recipe (fte.OLD_PARAMS, V1/V2 feature sets, hybrid router, seed 42) is reused unchanged.
Production code is imported read-only; nothing outside experiments/backtest_2025/ is written.
"""
import json, sys, hashlib
from datetime import date, timedelta
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "experiments" / "weather_source_comparison"):
    sys.path.insert(0, str(p))
import forecast_training_experiment as fte, option_screen as osn, winter_backtest as wb
import run_comparison as rc
from core.weather import WEATHER_KEYS
from stats import Stats, sudden_mask, cat_index

TRAIN_END = date(2024, 12, 31)
TEST_START, TEST_END = date(2025, 1, 1), date(2025, 12, 31)
SEEDS = fte.SEEDS
SEASONS = {"winter (Dec-Feb)": (12, 1, 2), "pre-monsoon (Mar-May)": (3, 4, 5), "monsoon (Jun-Sep)": (6, 7, 8, 9), "post-monsoon (Oct-Nov)": (10, 11)}
MODELS = ("persistence", "mean7", "v1", "v2", "hybrid")


class Spy(dict):
    def __init__(self, d): super().__init__(d); self.seen = set()
    def __getitem__(self, k): self.seen.add(k); return super().__getitem__(k)
    def get(self, k, default=None): self.seen.add(k); return super().get(k, default)
    def __contains__(self, k): self.seen.add(k); return super().__contains__(k)


def leakage_audit(hist, live, src, days, rows_by_day):
    """For every test day: the feature row is rebuilt from a spying history; no date after the origin day may be read."""
    bad = []
    for t in days:
        d0 = t - timedelta(days=1)
        spy = Spy(hist); spy[d0] = osn._live_rec(hist, live, d0); spy.seen.clear()
        row = fte.feature_dict(spy, d0, src[t])
        assert row == rows_by_day[t], f"rebuild differs on {t}"
        fut = sorted(k.isoformat() for k in spy.seen if isinstance(k, date) and k > d0)
        if fut: bad.append((t.isoformat(), fut[:3]))
    return bad


def month_season(dates, y, P, pm0):
    out = {"by_month": {}, "by_season": {}}
    def blk(idx):
        s = Stats(y[idx], {m: P[m][idx] for m in ("persistence", "v1", "hybrid")}, [dates[i] for i in idx])
        pts = {m: s.point(m) for m in ("persistence", "v1", "hybrid")}
        d = s.diff("v1", "hybrid") if len(idx) >= 10 else None
        return {"n_days": int(len(idx)), **{f"{m}_mae": pts[m]["mae"] for m in pts}, "v1_exact": pts["v1"]["cat_acc"], "hybrid_exact": pts["hybrid"]["cat_acc"],
                "hybrid_rmse": pts["hybrid"]["rmse"], "v1_rmse": pts["v1"]["rmse"],
                "hybrid_minus_v1_mae": (d["mae"] if d else None), "hybrid_minus_v1_mae_ci95_7d": (d.get("mae_ci95_7d") if d else None)}
    for m in range(1, 13):
        idx = np.array([i for i, d in enumerate(dates) if d.month == m])
        if len(idx): out["by_month"][str(m)] = blk(idx)
    for k, ms in SEASONS.items():
        idx = np.array([i for i, d in enumerate(dates) if d.month in ms])
        if len(idx): out["by_season"][k] = blk(idx)
    return out


def main(save_models=True):
    hist = osn.load_hist("data/daily_opencity.csv")
    live = fte.build_live_view(Path("data/raw"), extra_subs=("opencity_hourly",))
    era = lambda t: {k: hist[t].get(k) for k in WEATHER_KEYS}
    src = rc.load_source("production_default")
    res = {"label": "RETROSPECTIVE chronological backtest; 2025 had been examined in earlier experiments; not an untouched independent test",
           "protocol": __doc__.strip(), "train_cutoff_target_date": TRAIN_END.isoformat(), "test_window": [TEST_START.isoformat(), TEST_END.isoformat()]}
    # ---- training (targets <= 2024-12-31, ERA5 weather as in production training)
    all_train = fte.build(hist, live, era, view="offline")
    train = {t: v for t, v in all_train.items() if t <= TRAIN_END}
    assert train and max(train) <= TRAIN_END
    by_year = {}
    for t in train: by_year[t.year] = by_year.get(t.year, 0) + 1
    res["training"] = {"origins": len(train), "first_target": min(train).isoformat(), "last_target": max(train).isoformat(),
                       "origins_by_year": {str(k): v for k, v in sorted(by_year.items())}, "weather": "ERA5 daily (production training recipe; no forecast archive before 2024-01-19)",
                       "hyper_parameters": "fte.OLD_PARAMS unchanged; early stopping on the last 60 training rows only", "min_required_origins": fte.MIN_TRAIN}
    assert len(train) >= fte.MIN_TRAIN
    dpy = {}
    for t, v in train.items():
        dpy[t.year] = dpy.get(t.year, 0) + int(bool(fte.daypart_valid(v["row"])))
    res["training"]["origins_with_all_7_dayparts_by_year"] = {str(k): v for k, v in sorted(dpy.items())}
    res["training"]["origins_with_all_7_dayparts"] = sum(dpy.values())
    fits = {sd: fte.fit_models(train, sd) for sd in SEEDS}
    # ---- preserve the frozen models (separate directory)
    mdir = HERE / "models"
    if save_models:
        mdir.mkdir(exist_ok=True)
        meta = {}
        for tag in ("v1", "v2"):
            pth = mdir / f"{tag}_seed42.json"
            fits[42][tag].save_model(str(pth))
            meta[tag] = hashlib.sha256(pth.read_bytes()).hexdigest()
        (mdir / "hybrid_router.json").write_text(json.dumps({"rule": "use V2 if all 7 day-part inputs are finite, else V1 (option_screen.hybrid)", "uses": ["v1_seed42.json", "v2_seed42.json"],
                                                           "trained_on_targets_through": TRAIN_END.isoformat()}, indent=1))
        meta["note"] = "experiment copies; production artifacts untouched"
        (mdir / "sha256.json").write_text(json.dumps(meta, indent=1))
    # ---- test sets
    res["test"] = {}
    ref_obs = fte.build(hist, live, era)                                         # LABELLED REFERENCE ONLY: observed weather of the target day
    scoreable = [t for t in ref_obs if TEST_START <= t <= TEST_END]
    res["test"]["scoreable_days_2025_with_observed_weather"] = len(scoreable)
    for L in (1, 2):
        test = fte.build(hist, live, lambda t, f=src[L]: f.get(t))
        days = sorted(t for t in test if TEST_START <= t <= TEST_END)
        no_fc = sorted(t for t in scoreable if t not in test)
        y = np.array([test[t]["y"] for t in days]); rows = [test[t]["row"] for t in days]
        rows_by_day = {t: test[t]["row"] for t in days}
        pm0 = np.array([r["pm_d0"] for r in rows])
        dp = np.array([fte.daypart_valid(r) for r in rows])
        bad = leakage_audit(hist, live, src[L], days, rows_by_day)
        # ineligibility ladder for every 2025 calendar day
        ladder = {}
        for i in range((TEST_END - TEST_START).days + 1):
            t = TEST_START + timedelta(i)
            r = wb.classify_target(t, hist, live)
            key = r[0] if r else ("scoreable" if t in test else "9_forecast_weather_missing")
            ladder[key] = ladder.get(key, 0) + 1
        P = {sd: fte.predict_all(fits[sd], rows) for sd in SEEDS}
        st = Stats(y, {m: P[42][m] for m in MODELS}, days)
        blk = {"n_matched_days": len(days), "first": days[0].isoformat(), "last": days[-1].isoformat(),
               "days_by_month": {str(m): sum(1 for d in days if d.month == m) for m in range(1, 13)},
               "eligibility_ladder_all_365_days": ladder, "scoreable_days_lost_for_missing_forecast": [d.isoformat() for d in no_fc],
               "leakage_audit": {"rows_checked": len(days), "rows_reading_dates_after_origin": bad, "max_history_date_read_is_origin_day": not bad,
                                 "training_targets_max": max(train).isoformat(), "all_training_targets_le_cutoff": True,
                                 "forecast_days_equal_to_observed_ERA5": sum(1 for t in days if all(abs(src[L][t][k] - hist[t][k]) < 1e-9 for k in WEATHER_KEYS)),
                                 "live_23h_pm25_used_as_today": bool(all(abs(r['pm_d0'] - live[t - timedelta(days=1)]['pm25']) < 1e-9 for t, r in zip(days, rows)))},
               "hybrid_routing": {"V2_days": int(dp.sum()), "V1_fallback_days": int((~dp).sum())},
               "origin_day_stations": {"median": float(np.median([live[t - timedelta(days=1)]["n"] for t in days])), "min": int(min(live[t - timedelta(days=1)]["n"] for t in days)), "max": int(max(live[t - timedelta(days=1)]["n"] for t in days))},
               "models": {m: st.model(m) for m in MODELS},
               "diffs": {"hybrid_vs_v1": st.diff("v1", "hybrid"), "v2_vs_v1": st.diff("v1", "v2"), "hybrid_vs_v2": st.diff("v2", "hybrid"), "v1_vs_persistence": st.diff("persistence", "v1"),
                         "hybrid_vs_persistence": st.diff("persistence", "hybrid"), "hybrid_vs_mean7": st.diff("mean7", "hybrid"), "v1_vs_mean7": st.diff("mean7", "v1"), "mean7_vs_persistence": st.diff("persistence", "mean7")}}
        sd_ = sudden_mask(y, pm0, 0.30)
        def sub(mask):
            idx = np.where(mask)[0]
            if len(idx) < 10: return {"n_days": int(len(idx)), "note": "too few days"}
            s = Stats(y[idx], {m: P[42][m][idx] for m in MODELS}, [days[i] for i in idx])
            return {"n_days": int(len(idx)), **{m: s.point(m) for m in MODELS}, "hybrid_vs_v1": s.diff("v1", "hybrid"), "hybrid_vs_persistence": s.diff("persistence", "hybrid")}
        blk["subsets"] = {"sudden_change_ge30pct": sub(sd_), "stable": sub(~sd_), "hybrid_used_V2": sub(dp)}
        blk.update(month_season(days, y, P[42], pm0))
        blk["seeds"] = {str(sd): {m: float(np.abs(P[sd][m] - y).mean()) for m in ("v1", "v2", "hybrid")} for sd in SEEDS}
        blk["seeds_hybrid_beats_v1"] = int(sum(blk["seeds"][str(sd)]["hybrid"] < blk["seeds"][str(sd)]["v1"] for sd in SEEDS))
        blk["daily"] = {"dates": [d.isoformat() for d in days], "actual": y.round(2).tolist(), **{m: P[42][m].round(2).tolist() for m in MODELS}}
        res["test"][f"lead{L}"] = blk
    # ---- reference arm (NOT realistic): observed weather of the target day, labelled separately
    days = sorted(scoreable)
    y = np.array([ref_obs[t]["y"] for t in days]); rows = [ref_obs[t]["row"] for t in days]
    P = fte.predict_all(fits[42], rows)
    st = Stats(y, {m: P[m] for m in MODELS}, days)
    res["reference_observed_weather_NOT_REALISTIC"] = {"note": "uses ERA5 weather of the target day (not available at 23:00 IST); upper-bound reference and coverage check only",
                                                      "n_days": len(days), "models": {m: st.point(m) for m in MODELS}, "hybrid_vs_v1": st.diff("v1", "hybrid")}
    return res


if __name__ == "__main__":
    r = main()
    (HERE / "metrics.json").write_text(json.dumps(fte.jsonable(r), indent=1))
    print("metrics.json written")
