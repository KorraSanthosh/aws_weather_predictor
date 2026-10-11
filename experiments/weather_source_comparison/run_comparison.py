"""Retrospective comparison of weather-forecast sources for next-day Delhi PM2.5 (isolated experiment).

Protocol is declared in config.json before any result was produced.  Production code is imported read-only;
nothing outside experiments/weather_source_comparison/ and data/raw/weather_sources/ is written.

  arm A  (input swap)      one set of models per winter, trained on ERA5 weather exactly like production; only the
                           weather INPUT at prediction time changes with the source -> isolates the forecast source.
  arm B  (source trained)  models re-trained on the source's own archived forecasts (ERA5 where the archive has not
                           started yet; substitutions counted).  A SEPARATE experiment, never pooled with A.
  arm B_strict             training origins that have the source's forecast only; run if >= 365 origins.

Date sets (all chronological, frozen fit per winter, targets strictly before 15 Oct of the winter in training):
  P4  identical dates in 2024-25 and 2025-26 where production_default, gfs_global, icon_global, ecmwf_ifs025 all exist
  P6  2025-26 only, additionally ecmwf_ifs_hres and ecmwf_aifs025
Leads: previous_day1 (primary) and previous_day2 (strictly-available robustness arm).
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "model", ROOT / "pipeline", HERE):
    sys.path.insert(0, str(p))

import forecast_training_experiment as fte  # noqa: E402
import option_screen as osn  # noqa: E402
import winter_backtest as wb  # noqa: E402
from core.weather import HOURLY_VARS, WEATHER_KEYS, hourly_to_daily_weather  # noqa: E402
from stats import Stats, sudden_mask  # noqa: E402

CFG = json.loads((HERE / "config.json").read_text())
RAW = ROOT / "data" / "raw" / "weather_sources"
SOURCES = ["production_default", "gfs_global", "icon_global", "ecmwf_ifs025", "ecmwf_ifs_hres", "ecmwf_aifs025"]
P4_SOURCES = SOURCES[:4]
WINTERS = fte.WINTERS
SEEDS = fte.SEEDS
MODELS = fte.MODELS
SUDDEN = CFG["sudden_change_threshold"]
MIN_TRAIN = fte.MIN_TRAIN
PROD = "production_default"


# ------------------------------------------------------------------ archives
def load_source(label):
    """{lead: {date: daily weather}} built with the production aggregation (core.weather.hourly_to_daily_weather)."""
    files = sorted(glob.glob(str(RAW / label / "*.json")))
    if not files:
        return None
    times, cols = [], {f"{v}_previous_day{n}": [] for n in (1, 2) for v in HOURLY_VARS}
    for f in files:
        h = json.loads(Path(f).read_text())["hourly"]
        times += h["time"]
        for k in cols:
            cols[k] += h.get(k, [None] * len(h["time"]))
    out = {}
    for lead in (1, 2):
        hh = {"time": times, **{v: cols[f"{v}_previous_day{lead}"] for v in HOURLY_VARS}}
        out[lead] = hourly_to_daily_weather(hh)
    out["hourly"] = {"time": times, **cols}
    return out


def hourly_summary(src):
    """Missing-hour accounting per lead (documented in the report)."""
    h = src["hourly"]
    n = len(h["time"])
    rec = {"hours": n, "first": h["time"][0], "last": h["time"][-1]}
    for lead in (1, 2):
        ok = [all(h[f"{v}_previous_day{lead}"][i] is not None for v in HOURLY_VARS) for i in range(n)]
        first = next((h["time"][i] for i, o in enumerate(ok) if o), None)
        rec[f"lead{lead}"] = {"complete_hours": int(sum(ok)), "share": round(sum(ok) / n, 4), "first_complete_hour": first,
                              "complete_days": len(src[lead])}
    return rec


# ------------------------------------------------------------------ helpers
def collect(sets, dates, key):
    return [sets[t][key] for t in dates]


def pooled(per_winter):
    """Concatenate per-winter dicts (dates, y, pm0, P) placed 200 days apart for the block bootstrap."""
    order = [w for w in per_winter]
    pdates = osn.pooled_dates([list(per_winter[w]["dates"]) for w in order])
    y = np.concatenate([per_winter[w]["y"] for w in order])
    pm0 = np.concatenate([per_winter[w]["pm0"] for w in order])
    keys = per_winter[order[0]]["P"].keys()
    P = {k: np.concatenate([per_winter[w]["P"][k] for w in order]) for k in keys}
    return pdates, y, pm0, P


def metrics_block(y, pm0, dates, preds, ref_src=PROD, srcs=()):
    """preds: {"<src>__<model>": array}.  Point metrics + CIs for every key; hybrid/v1 differences vs ref_src."""
    st = Stats(y, preds, dates)
    out = {"n_days": int(len(y)), "models": {k: st.model(k, baseline=f"{k.split('__')[0]}__persistence") if f"{k.split('__')[0]}__persistence" in preds else st.point(k)
                                            for k in preds}}
    out["diff_vs_production"] = {}
    for s in srcs:
        if s == ref_src:
            continue
        out["diff_vs_production"][s] = {m: st.diff(f"{ref_src}__{m}", f"{s}__{m}") for m in ("v1", "hybrid")}
    return out


def subset(y, pm0, dates, preds, mask, srcs):
    idx = np.where(mask)[0]
    if len(idx) < 10:
        return {"n_days": int(len(idx)), "note": "too few days for intervals"}
    return metrics_block(y[idx], pm0[idx], [dates[i] for i in idx], {k: v[idx] for k, v in preds.items()}, srcs=srcs)


# ------------------------------------------------------------------ main
def run(verbose=True):
    hist = osn.load_hist("data/daily_opencity.csv")
    live = fte.build_live_view(Path("data/raw"), extra_subs=("opencity_hourly",))
    era = lambda t: {k: hist[t].get(k) for k in WEATHER_KEYS}                       # noqa: E731
    src = {s: load_source(s) for s in SOURCES}
    present = [s for s in SOURCES if src[s] is not None]
    res = {"config": CFG, "label": "RETROSPECTIVE: winters 2024-25 and 2025-26 were examined in earlier experiments; not an untouched test",
           "archives": {s: hourly_summary(src[s]) for s in present}, "arms": {}, "sets": {}, "winters": {}}

    # ---- test rows (live view) per source, lead
    test = {s: {L: fte.build(hist, live, lambda t, f=src[s][L]: f.get(t)) for L in (1, 2)} for s in present}
    train_off = {"obs": fte.build(hist, live, era, view="offline")}
    for s in present:
        f1 = src[s][1]
        train_off[f"mixed::{s}"] = fte.build(hist, live, lambda t, f=f1: f.get(t) if fte.complete(f.get(t)) else era(t), view="offline")
        train_off[f"strict::{s}"] = fte.build(hist, live, lambda t, f=f1: f.get(t), view="offline")

    windows = {Y: wb.winter_window(Y) for Y in WINTERS}
    wname = lambda Y: f"{Y}-{str(Y + 1)[2:]}"                                      # noqa: E731
    in_win = lambda Y, t: windows[Y][0] <= t <= windows[Y][1]                      # noqa: E731

    # ---- date sets
    def own_days(s, L, Y):
        return {t for t in test[s][L] if in_win(Y, t)}

    sets = {}
    for L in (1, 2):
        sets[f"P4_L{L}"] = {wname(Y): sorted(set.intersection(*[own_days(s, L, Y) for s in P4_SOURCES])) for Y in WINTERS}
        if "ecmwf_ifs_hres" in present and "ecmwf_aifs025" in present:
            Y = 2025
            sets[f"P6_L{L}"] = {wname(Y): sorted(set.intersection(*[own_days(s, L, Y) for s in SOURCES if s in present]))}
    res["scoreable_days_in_window"] = {wname(Y): sum(1 for t in train_off["obs"] if in_win(Y, t)) for Y in WINTERS}
    res["own_coverage_days"] = {s: {f"L{L}": {wname(Y): len(own_days(s, L, Y)) for Y in WINTERS} for L in (1, 2)} for s in present}
    res["sets"] = {k: {w: {"days": len(d), "first": d[0].isoformat() if d else None, "last": d[-1].isoformat() if d else None}
                       for w, d in v.items()} for k, v in sets.items()}

    # ---- weather-mismatch (forecast vs ERA5) on the P4_L1 / P6_L1 dates, independent of the PM model
    res["weather_error_vs_era5"] = {}
    for sname in ("P4_L1", "P6_L1"):
        if sname not in sets:
            continue
        res["weather_error_vs_era5"][sname] = {}
        for s in (P4_SOURCES if sname == "P4_L1" else present):
            days = [t for w in sets[sname].values() for t in w]
            res["weather_error_vs_era5"][sname][s] = {k: {"bias": round(float(np.mean([src[s][1][t][k] - hist[t][k] for t in days])), 3),
                                                         "rmse": round(float(np.sqrt(np.mean([(src[s][1][t][k] - hist[t][k]) ** 2 for t in days]))), 3)}
                                                     for k in WEATHER_KEYS}

    # ---- identify what the default blend resolves to (value matching on identical hours)
    if PROD in present:
        comp = {}
        hp = src[PROD]["hourly"]
        for s in present:
            if s == PROD:
                continue
            hs = src[s]["hourly"]
            n_eq = n_tot = 0
            for v in ("temperature_2m", "wind_speed_10m", "surface_pressure"):
                a, b = hp[f"{v}_previous_day1"], hs[f"{v}_previous_day1"]
                for x, y_ in zip(a, b):
                    if x is None or y_ is None:
                        continue
                    n_tot += 1
                    n_eq += abs(x - y_) < 1e-6
            comp[s] = {"hours_compared": n_tot, "share_identical_to_production_default": round(n_eq / n_tot, 4) if n_tot else None}
        res["best_match_composition"] = comp

    # ---- is production_default the SAME model in every winter?  (value matching, winter window hours only) -------
    res["reference_validity"] = {}
    if PROD in present:
        hp = src[PROD]["hourly"]
        for Y in WINTERS:
            a0, b0 = windows[Y]
            idx = [i for i, t in enumerate(hp["time"]) if a0.isoformat() <= t[:10] <= b0.isoformat()]
            rec = {}
            for cand in ("icon_global", "ecmwf_ifs_hres"):
                if cand not in present:
                    continue
                hc = src[cand]["hourly"]
                n = eq = 0
                for v in HOURLY_VARS:
                    for i in idx:
                        x, z = hp[f"{v}_previous_day1"][i], hc[f"{v}_previous_day1"][i]
                        if x is None or z is None:
                            continue
                        n += 1
                        eq += abs(x - z) < 1e-6
                rec[cand] = {"hours_compared": n, "share_identical": round(eq / n, 4) if n else None}
            hres = rec.get("ecmwf_ifs_hres", {}).get("share_identical")
            icon = rec.get("icon_global", {}).get("share_identical")
            hres_n = rec.get("ecmwf_ifs_hres", {}).get("hours_compared")
            rec["production_default_equals"] = ("ecmwf_ifs_hres" if hres and hres >= 0.95 and (hres_n or 0) >= 1000 else "icon_global" if icon and icon >= 0.95 else "unresolved")
            res["reference_validity"][wname(Y)] = rec
    current_ref_winters = {w for w, v in res["reference_validity"].items() if v["production_default_equals"] == "ecmwf_ifs_hres"}

    # ---- fits ------------------------------------------------------------------------------------------------
    models_A, models_B, info = {}, {}, {}
    for Y in WINTERS:
        start, end = windows[Y]
        w = wname(Y)
        assert end <= fte.HARD_STOP
        tr = {t: v for t, v in train_off["obs"].items() if t < start}
        assert all(t < start for t in tr)
        models_A[(w, "obs")] = {sd: fte.fit_models(tr, sd) for sd in SEEDS}
        info[w] = {"A_train_origins": len(tr)}
        for s in present:
            tm = {t: v for t, v in train_off[f"mixed::{s}"].items() if t < start}
            assert all(t < start for t in tm)
            n_fc = sum(1 for t in tm if fte.complete(src[s][1].get(t)))
            info[w][s] = {"B_origins": len(tm), "B_origins_with_source_forecast": n_fc, "B_substituted_with_era5": len(tm) - n_fc}
            if len(tm) >= MIN_TRAIN and n_fc > 0:
                models_B[(w, s)] = {sd: fte.fit_models(tm, sd) for sd in SEEDS}
            ts = {t: v for t, v in train_off[f"strict::{s}"].items() if t < start}
            info[w][s]["B_strict_origins"] = len(ts)
            if len(ts) >= MIN_TRAIN:
                models_B[(w, s, "strict")] = {sd: fte.fit_models(ts, sd) for sd in SEEDS}
            else:
                info[w][s]["B_strict_note"] = f"insufficient coverage: {len(ts)} origins before {start} (< {MIN_TRAIN})"
        if verbose:
            print("fitted", w, flush=True)
    res["winters"] = info

    # ---- predictions and scoring ----------------------------------------------------------------------------
    def predict_for(arm, s, w, dates, L, seed):
        rows = [test[s][L][t]["row"] for t in dates]
        if arm == "A":
            m = models_A[(w, "obs")][seed]
        elif arm == "B":
            m = models_B.get((w, s))
            m = m[seed] if m else None
        else:
            m = models_B.get((w, s, "strict"))
            m = m[seed] if m else None
        if m is None:
            return None
        return fte.predict_all(m, rows), rows

    def score_set(sname, arm, seed=42, detail=True):
        L = int(sname[-1])
        dsets = sets[sname]
        srcs = [s for s in (P4_SOURCES if sname.startswith("P4") else present) if s in present]
        if arm != "A":
            # B can only be scored for sources that have models in every winter of the set
            srcs = [s for s in srcs if all((w, s) in models_B if arm == "B" else (w, s, "strict") in models_B for w in dsets)]
        if PROD not in srcs or not srcs:
            return None
        per_w = {}
        for w, dates in dsets.items():
            if not dates:
                continue
            P, y, pm0 = {}, None, None
            for s in srcs:
                out = predict_for(arm, s, w, dates, L, seed)
                P_s, rows = out
                for m in ("v1", "v2", "hybrid", "persistence"):
                    P[f"{s}__{m}"] = P_s[m]
                y = np.array([test[s][L][t]["y"] for t in dates])
                pm0 = np.array([r["pm_d0"] for r in rows])
            per_w[w] = {"dates": dates, "y": y, "pm0": pm0, "P": P}
        if not per_w:
            return None
        out = {"sources": srcs, "winters": {}}
        for w, d in per_w.items():
            out["winters"][w] = metrics_block(d["y"], d["pm0"], d["dates"], d["P"], srcs=srcs)
        pdates, y, pm0, P = pooled(per_w)
        out["pooled"] = metrics_block(y, pm0, pdates, P, srcs=srcs)
        if detail:
            sd = sudden_mask(y, pm0, SUDDEN)
            out["subsets"] = {"sudden_change_days_ge30pct": subset(y, pm0, pdates, P, sd, srcs),
                              "stable_days": subset(y, pm0, pdates, P, ~sd, srcs)}
        out["_hybrid_mae"] = {s: float(np.abs(P[f"{s}__hybrid"] - y).mean()) for s in srcs}
        return out

    for arm in ("A", "B", "B_strict"):
        res["arms"][arm] = {}
        for sname in sets:
            if arm != "A" and not sname.endswith("L1"):
                continue
            r = score_set(sname, arm)
            if r is None:
                res["arms"][arm][sname] = {"not_run": "no source has models in every winter of this set (insufficient training coverage)"}
                continue
            res["arms"][arm][sname] = r
            if verbose:
                print(arm, sname, {s: round(v, 2) for s, v in r["_hybrid_mae"].items()}, flush=True)

    # ---- seed robustness (hybrid MAE per seed, pooled, arm A and B, lead 1) -----------------------------------
    res["seed_robustness"] = {}
    for arm in ("A", "B"):
        for sname in [k for k in sets if k.endswith("L1")]:
            by_seed = {}
            for sd in SEEDS:
                r = score_set(sname, arm, seed=sd, detail=False)
                if r is not None:
                    by_seed[str(sd)] = {s: round(v, 3) for s, v in r["_hybrid_mae"].items()}
            if by_seed:
                srcs = list(next(iter(by_seed.values())))
                res["seed_robustness"][f"{arm}:{sname}"] = {
                    "hybrid_mae_by_seed": by_seed,
                    "seeds_where_source_beats_production": {s: int(sum(by_seed[sd][s] < by_seed[sd][PROD] for sd in by_seed)) for s in srcs if s != PROD},
                    "mean_hybrid_mae": {s: round(float(np.mean([by_seed[sd][s] for sd in by_seed])), 3) for s in srcs}}

    # ---- decomposition: source effect vs training-coverage effect (hybrid MAE, seed 42) ----------------------
    res["decomposition"] = {}
    for sname in [k for k in sets if k.endswith("L1")]:
        a, b = res["arms"]["A"].get(sname), res["arms"]["B"].get(sname)
        if not a or "pooled" not in a or not b or "pooled" not in b:
            continue
        dec = {}
        for s in a["sources"]:
            if s not in b["sources"] or PROD not in b["sources"]:
                continue
            A_s, A_p = a["_hybrid_mae"][s], a["_hybrid_mae"][PROD]
            B_s, B_p = b["_hybrid_mae"][s], b["_hybrid_mae"][PROD]
            dec[s] = {"total_B_source_minus_A_production": round(B_s - A_p, 3), "source_effect_A_source_minus_A_production": round(A_s - A_p, 3),
                      "training_effect_B_source_minus_A_source": round(B_s - A_s, 3),
                      "control_production_own_B_minus_A": round(B_p - A_p, 3),
                      "training_effect_net_of_control": round((B_s - A_s) - (B_p - A_p), 3)}
        res["decomposition"][sname] = dec

    # ---- decision rule ------------------------------------------------------------------------------------------
    rule = {}
    A1 = res["arms"]["A"]
    for sname in [k for k in sets if k.endswith("L1")]:
        r = A1.get(sname)
        if not r or "pooled" not in r:
            continue
        l2 = A1.get(sname[:-1] + "2")
        for s in r["sources"]:
            if s == PROD:
                continue
            h = r["pooled"]["diff_vs_production"][s]["hybrid"]
            per_w = {w: v["diff_vs_production"][s]["hybrid"] for w, v in r["winters"].items()}
            worst = max(d["mae_pct"] for d in per_w.values())
            l2ok = None
            if l2 and "pooled" in l2 and s in l2["pooled"]["diff_vs_production"]:
                l2ok = l2["pooled"]["diff_vs_production"][s]["hybrid"]["mae"] < 0
            single = len(per_w) < 2
            crit = {"mae_at_least_2pct_lower": bool(h["mae_pct"] <= -2.0),
                    "ci_excludes_zero_both_blocks_below": bool(h.get("mae_ci95_7d", [0, 0])[1] < 0 and h.get("mae_ci95_14d", [0, 0])[1] < 0),
                    "negative_in_every_covered_winter": bool(all(d["mae"] < 0 for d in per_w.values())),
                    "no_winter_more_than_2pct_worse": bool(worst <= 2.0),
                    "exact_category_not_lower_by_gt_2pts": bool(h["cat_acc"] >= -0.02),
                    "lead2_point_estimate_negative": l2ok}
            rule[f"{sname}:{s}"] = {**crit, "hybrid_mae_pct_vs_production": round(h["mae_pct"], 2),
                                    "hybrid_mae_diff": round(h["mae"], 3), "ci95_7d": [round(x, 3) for x in h.get("mae_ci95_7d", [0, 0])],
                                    "ci95_14d": [round(x, 3) for x in h.get("mae_ci95_14d", [0, 0])],
                                    "worst_winter_pct": round(worst, 2), "winters": list(per_w), "single_winter_source": single,
                                    "passes_all": bool(all(v is True for v in crit.values()))}
            ref_ok = set(per_w) <= current_ref_winters
            rule[f"{sname}:{s}"]["reference_is_current_source_in_all_winters"] = ref_ok
            rule[f"{sname}:{s}"]["verdict"] = (
                "NOT VALID EVIDENCE vs the current source: production_default was a different model in "
                + ", ".join(w for w in per_w if w not in current_ref_winters) if not ref_ok else
                "logging candidate only (one winter)" if single and all(v is True for v in crit.values()) else
                "recommend for forward test" if all(v is True for v in crit.values()) else "does not reliably beat the current source")
    res["decision_rule"] = rule
    for arm in res["arms"].values():
        for r in arm.values():
            r.pop("_hybrid_mae", None)
    return res


if __name__ == "__main__":
    out = run()
    (HERE / "results.json").write_text(json.dumps(fte.jsonable(out), indent=1))
    print("wrote results.json")
