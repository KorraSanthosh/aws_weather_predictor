"""Screening of UNTESTED options for improving accuracy, on the same frozen historical winters as winter_backtest.py.

    python model/option_screen.py      # needs data/daily_opencity.csv, data/raw/opencity_hourly/, and
                                       # data/raw/weather_prevruns_2024_2026.json (Open-Meteo Previous Runs API)
    -> experiments/option_screen.json, experiments/option_screen.md

EXPLORATORY.  These winters were all examined before, so a "promising" option here is a HYPOTHESIS for the prospective
winter 2026-27 test, never a reason to change the default.  Everything below was fixed before running:

Reference = the Hybrid (V2 when all 7 day-part inputs exist, else V1), frozen fit before each winter, unchanged recipe.
Options (each changes ONE thing):
  O1 error correction   add alpha x (yesterday's log error) to tomorrow's log forecast.  alpha is fitted ONLY on the year
                        before the window with an inner model trained on data before that year; clipped to [0, 1].
                        Yesterday's error uses the 23:00 value of today's PM2.5 (what the Lambda knows), not the full day.
  O2 recency weight     training rows from the last 365 days before the window count double.
  O3 monthly refit      refit at the start of every month inside the window on all earlier days (what a live system
                        retrained monthly would do; it is NOT frozen, so it is reported as an operational option).
  O4 spatial features   V1 and V2 each get the 4 station-spread features (n_stations, sd, spread, max ratio).
  O5 train on forecasts training rows use the archived 24-h-ahead forecast as tomorrow's weather where it exists
                        (from 2024-01-20), observed weather before that.  Tested with forecast weather.
  O6 forecast bias fix  forecast weather is shifted by its mean difference from the training (archive) weather,
                        estimated only on days before the window.  Tested with forecast weather.
Weather regimes: O1-O4 use the PRIMARY regime of winter_backtest.py (tomorrow's weather := today's, leak-free).
O5-O6 and the realism table use the REAL archived forecasts (Previous Runs API, value predicted 24 h / 48 h ahead),
which exist for winters 2024-25 and 2025-26 only.

Fixed rule for "promising": pooled over the winters the option was run on, the option's Hybrid MAE is >= 2% lower than
the reference with a moving-block 95% interval of the difference below zero for both 7- and 14-day blocks, AND no
single winter is more than 2% worse.  Anything else is reported as "no evidence of improvement".
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE, ROOT / "pipeline"):
    sys.path.insert(0, str(p))

import winter_backtest as wb  # noqa: E402
from compare_versions import build_live_view  # noqa: E402
from core.daily import DP_KEYS  # noqa: E402
from core.features import FEATURE_SETS, GROUPS, feature_dict  # noqa: E402
from core.hybrid import daypart_valid  # noqa: E402
from core.weather import HOURLY_VARS, WEATHER_KEYS, hourly_to_daily_weather  # noqa: E402
from experiments import OLD_PARAMS, metrics  # noqa: E402
from train import load_daily  # noqa: E402

V1, V2 = "v1_25_baseline", "v2_35_candidate"
SPATIAL = GROUPS["spatial"]
WINTERS = (2022, 2023, 2024, 2025)
FC_WINTERS = (2024, 2025)
RULE = {"min_gain_pct": 2.0, "max_single_winter_loss_pct": 2.0}
BOOT = dict(n_boot=5000, seed=0)


# ------------------------------------------------------------------ data
def load_hist(path):
    hist, _ = load_daily(path)
    with open(path, newline="") as f:                      # station-spread columns for the spatial features
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            num = lambda k: float(r[k]) if r.get(k, "") not in ("", None) else None  # noqa: E731
            hist[d].update({"n": num("n_stations") if r.get("pm25") else None, "pm25_sd": num("pm25_sd"),
                            "pm25_max": num("pm25_max"), "pm25_min": num("pm25_min")})
    return hist


def load_forecasts(path):
    h = json.loads(Path(path).read_text())["hourly"]
    out = {}
    for lead in (1, 2):
        hh = {"time": h["time"], **{v: h[f"{v}_previous_day{lead}"] for v in HOURLY_VARS}}
        out[lead] = hourly_to_daily_weather(hh)
    return out


def _live_rec(hist, live, d0):
    lrec = dict(hist[d0])
    lrec["pm25"] = live[d0]["pm25"]
    for k in DP_KEYS:
        v = live[d0].get(k)
        lrec[k] = None if v is None or v != v else v
    return lrec


def build(hist, live, wx_for, view="live"):
    """{t: {"y", "row", "pm_live_t"}} for every scoreable target t whose weather wx_for(t, d0) exists.
    view "live" = today's PM2.5 and day-parts from hours <= 22 (the Lambda); "offline" = full day (training)."""
    out = {}
    for t in sorted(hist):
        if wb.classify_target(t, hist, live) is not None:
            continue
        d0 = t - timedelta(days=1)
        wx = wx_for(t, d0)
        if wx is None or any(wx.get(k) is None for k in WEATHER_KEYS):
            continue
        if view == "live":
            orig = hist[d0]
            hist[d0] = _live_rec(hist, live, d0)
            try:
                row = feature_dict(hist, d0, wx)
            finally:
                hist[d0] = orig
        else:
            row = feature_dict(hist, d0, wx)
        out[t] = {"y": float(hist[t]["pm25"]), "row": row}
    return out


# ------------------------------------------------------------------ models
def fit(X, y, names, w=None):
    import xgboost as xgb
    i0 = names.index("pm_d0")
    delta = np.log1p(y) - np.log1p(X[:, i0])
    m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42,
                         early_stopping_rounds=40, n_jobs=4, **OLD_PARAMS)
    kw = {}
    if w is not None:
        kw = {"sample_weight": w[:-60], "sample_weight_eval_set": [w[-60:]]}
    m.fit(X[:-60], delta[:-60], eval_set=[(X[-60:], delta[-60:])], verbose=False, **kw)
    return m


def predict(m, X, names):
    return np.expm1(np.log1p(X[:, names.index("pm_d0")]) + m.predict(X))


def M(rows, names):
    return np.array([[r[n] for n in names] for r in rows], dtype=float)


def fit_predict(train, test_rows, names, w=None):
    ts = sorted(train)
    m = fit(M([train[t]["row"] for t in ts], names), np.array([train[t]["y"] for t in ts]), names, w)
    return predict(m, M(test_rows, names), names)


def hybrid(p1, p2, rows):
    return np.array([p2[i] if daypart_valid(r) else p1[i] for i, r in enumerate(rows)])


# ------------------------------------------------------------------ statistics
def pooled_dates(per_winter):
    """Winters placed 200 days apart on a synthetic timeline (blocks never straddle two winters)."""
    out = []
    for k, ds in enumerate(per_winter):
        base = ds[0].toordinal()
        out += [date.fromordinal(d.toordinal() - base + 1000 + 200 * k) for d in ds]
    return out


def compare(y, p_ref, p_opt, dates):
    r7 = wb.mbb_mae_diff(y, p_ref, p_opt, dates, block=7, **BOOT)
    r14 = wb.mbb_mae_diff(y, p_ref, p_opt, dates, block=14, **BOOT)
    return {"n": r7["n"], "mae_ref": round(r7["mae_a"], 2), "mae_opt": round(r7["mae_b"], 2),
            "diff": round(r7["diff"], 2), "pct": round(r7["pct"], 2),
            "ci95_diff_7d": [round(x, 2) for x in r7["ci95_diff"]], "ci95_diff_14d": [round(x, 2) for x in r14["ci95_diff"]],
            "excludes_zero_both": bool(r7["excludes_zero"] and r14["excludes_zero"])}


def cat_acc(y, p):
    return metrics(y, p)["cat_acc"]


# ------------------------------------------------------------------ main experiment
def run(daily="data/daily_opencity.csv", raw=Path("data/raw"), extra=("opencity_hourly",),
        fc_path="data/raw/weather_prevruns_2024_2026.json"):
    hist = load_hist(daily)
    live = build_live_view(raw, extra_subs=extra)
    fc = load_forecasts(fc_path)
    obs = lambda t, d0: {k: hist[t].get(k) for k in WEATHER_KEYS}  # noqa: E731
    pers = lambda t, d0: {k: hist[d0].get(k) for k in WEATHER_KEYS}  # noqa: E731
    train_obs = build(hist, live, obs, view="offline")                        # production training rows
    test_pers = build(hist, live, pers)                                       # PRIMARY regime
    test_obs = build(hist, live, obs)                                         # diagnostic (optimistic)
    test_fc = {k: build(hist, live, (lambda t, d0, k=k: fc[k].get(t))) for k in (1, 2)}
    train_mixed = build(hist, live, lambda t, d0: fc[1].get(t) or obs(t, d0), view="offline")   # O5

    res = {"protocol": __doc__.strip(), "rule": RULE, "winters": {}, "realism": {}, "pooled": {}}
    pool = {}                                                                # name -> lists for pooling

    def add(name, wname, y, pref, popt, ds):
        pool.setdefault(name, {"y": [], "ref": [], "opt": [], "dates": [], "per_winter": {}})
        P = pool[name]
        P["y"] += list(y); P["ref"] += list(pref); P["opt"] += list(popt); P["dates"].append(list(ds))
        P["per_winter"][wname] = compare(y, pref, popt, list(ds))
        P["per_winter"][wname]["cat_ref"], P["per_winter"][wname]["cat_opt"] = cat_acc(y, pref), cat_acc(y, popt)

    for Y in WINTERS:
        start, end = wb.winter_window(Y)
        wname = f"{Y}-{str(Y + 1)[2:]}"
        tr = {t: v for t, v in train_obs.items() if t < start}
        te_t = [t for t in sorted(test_pers) if start <= t <= end]
        rows = [test_pers[t]["row"] for t in te_t]
        y = np.array([test_pers[t]["y"] for t in te_t])
        names = {V1: FEATURE_SETS[V1], V2: FEATURE_SETS[V2]}
        base = {v: fit_predict(tr, rows, names[v]) for v in names}
        ref = hybrid(base[V1], base[V2], rows)
        info = {"scored_days": len(te_t), "days_with_day_parts": int(sum(daypart_valid(r) for r in rows)),
                "reference_hybrid_mae": round(float(np.abs(ref - y).mean()), 2)}

        # O1 error correction ----------------------------------------------------------------------------
        a0 = start - timedelta(days=365)
        inner_tr = {t: v for t, v in train_obs.items() if t < a0}
        cal_t = [t for t in sorted(test_pers) if a0 <= t < start]
        alpha = {}
        opt1 = {}
        for v in names:
            pc = fit_predict(inner_tr, [test_pers[t]["row"] for t in cal_t], names[v])
            r = {t: math.log1p(live[t]["pm25"]) - math.log1p(p) for t, p in zip(cal_t, pc) if t in live}
            pairs = [(r[t], math.log1p(test_pers[t + timedelta(days=1)]["y"]) - math.log1p(
                pc[cal_t.index(t + timedelta(days=1))])) for t in r if t + timedelta(days=1) in cal_t]
            num = sum(a * b for a, b in pairs); den = sum(a * a for a, _ in pairs)
            alpha[v] = float(min(1.0, max(0.0, num / den))) if den > 0 else 0.0
            # in the window: yesterday's error of the FROZEN model, using the 23:00 PM2.5 value
            pw = dict(zip(te_t, base[v]))
            corr = []
            for t, p in zip(te_t, base[v]):
                d0 = t - timedelta(days=1)
                if d0 in pw and d0 in live:
                    e = math.log1p(live[d0]["pm25"]) - math.log1p(pw[d0])
                    corr.append(math.expm1(math.log1p(p) + alpha[v] * e))
                else:
                    corr.append(p)
            opt1[v] = np.array(corr)
        add("O1_error_correction", wname, y, ref, hybrid(opt1[V1], opt1[V2], rows), te_t)
        info["O1_alpha"] = {k: round(a, 3) for k, a in alpha.items()}

        # O2 recency weighting ---------------------------------------------------------------------------
        ts = sorted(tr)
        w = np.array([2.0 if t >= start - timedelta(days=365) else 1.0 for t in ts])
        opt2 = {v: fit_predict(tr, rows, names[v], w) for v in names}
        add("O2_recency_weight", wname, y, ref, hybrid(opt2[V1], opt2[V2], rows), te_t)

        # O3 monthly refit (operational, not frozen) --------------------------------------------------------
        opt3 = {v: np.zeros(len(te_t)) for v in names}
        for m0 in wb.daterange(start, end):
            if m0 != start and m0.day != 1:
                continue
            m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
            idx = [i for i, t in enumerate(te_t) if m0 <= t < m1]
            if not idx:
                continue
            trm = {t: v for t, v in train_obs.items() if t < m0}
            for v in names:
                opt3[v][idx] = fit_predict(trm, [rows[i] for i in idx], names[v])
        add("O3_monthly_refit", wname, y, ref, hybrid(opt3[V1], opt3[V2], rows), te_t)

        # O4 spatial features ------------------------------------------------------------------------------
        opt4 = {v: fit_predict(tr, rows, names[v] + SPATIAL) for v in names}
        add("O4_spatial_features", wname, y, ref, hybrid(opt4[V1], opt4[V2], rows), te_t)

        res["winters"][wname] = info

        # realism + O5/O6 with REAL forecasts ---------------------------------------------------------------
        if Y in FC_WINTERS:
            common = [t for t in te_t if t in test_fc[1] and t in test_fc[2] and t in test_obs]
            yy = np.array([test_pers[t]["y"] for t in common])
            real = {"days": len(common)}
            for lab, src in (("weather_persistence", test_pers), ("forecast_48h", test_fc[2]),
                             ("forecast_24h", test_fc[1]), ("observed_weather", test_obs)):
                rr = [src[t]["row"] for t in common]
                p = {v: fit_predict(tr, rr, names[v]) for v in names}
                hb = hybrid(p[V1], p[V2], rr)
                real[lab] = {k: {"mae": round(float(np.abs(q - yy).mean()), 2), "cat_acc": cat_acc(yy, q)}
                             for k, q in (("v1", p[V1]), ("v2", p[V2]), ("hybrid", hb))}
                real[lab]["hybrid_minus_v1"] = compare(yy, p[V1], hb, common)
                if lab == "forecast_24h":
                    pool.setdefault("_realfc_hybrid_vs_v1", {"y": [], "ref": [], "opt": [], "dates": [], "per_winter": {}})
                    Q = pool["_realfc_hybrid_vs_v1"]
                    Q["y"] += list(yy); Q["ref"] += list(p[V1]); Q["opt"] += list(hb); Q["dates"].append(list(common))
                    Q["per_winter"][wname] = real[lab]["hybrid_minus_v1"]
            res["realism"][wname] = real
            rr = [test_fc[1][t]["row"] for t in common]
            pref = {v: fit_predict(tr, rr, names[v]) for v in names}
            fref = hybrid(pref[V1], pref[V2], rr)
            # O5: train with forecast weather where it exists
            trm = {t: v for t, v in train_mixed.items() if t < start}
            p5 = {v: fit_predict(trm, rr, names[v]) for v in names}
            add("O5_train_on_forecasts", wname, yy, fref, hybrid(p5[V1], p5[V2], rr), common)
            # O6: remove the forecast-vs-archive mean bias, estimated before the window only
            bias = {}
            pre = [t for t in fc[1] if t < start and t in hist and all(hist[t].get(k) is not None for k in WEATHER_KEYS)]
            for k in WEATHER_KEYS:
                bias[k] = float(np.mean([fc[1][t][k] - hist[t][k] for t in pre]))
            adj = {t: {k: fc[1][t][k] - bias[k] for k in WEATHER_KEYS} for t in fc[1]}
            test_adj = build(hist, live, lambda t, d0: adj.get(t))
            rr6 = [test_adj[t]["row"] for t in common]
            p6 = {v: fit_predict(tr, rr6, names[v]) for v in names}
            add("O6_forecast_bias_fix", wname, yy, fref, hybrid(p6[V1], p6[V2], rr6), common)
            info["O6_bias_removed"] = {k: round(b, 2) for k, b in bias.items()}
            info["O6_bias_days"] = len(pre)

    Q = pool.pop("_realfc_hybrid_vs_v1", None)
    if Q:
        res["realism_pooled_hybrid_vs_v1_forecast_24h"] = compare(
            np.array(Q["y"]), np.array(Q["ref"]), np.array(Q["opt"]), pooled_dates([list(d) for d in Q["dates"]]))
    for name, P in pool.items():
        ds = pooled_dates([list(d) for d in P["dates"]])
        c = compare(np.array(P["y"]), np.array(P["ref"]), np.array(P["opt"]), ds)
        c["cat_ref"], c["cat_opt"] = cat_acc(np.array(P["y"]), np.array(P["ref"])), cat_acc(np.array(P["y"]), np.array(P["opt"]))
        worst = max(v["pct"] for v in P["per_winter"].values())
        c["worst_single_winter_pct"] = round(worst, 2)
        c["promising"] = bool(c["pct"] <= -RULE["min_gain_pct"] and c["excludes_zero_both"] and c["diff"] < 0
                              and worst <= RULE["max_single_winter_loss_pct"])
        c["per_winter"] = P["per_winter"]
        res["pooled"][name] = c
    return res


def render(res):
    L = ["# Screening of untested accuracy options (exploratory)", "",
         "> **Exploratory, retrospective.** These winters were examined before. A \"promising\" result is a hypothesis for the "
         "winter 2026-27 shadow test, not a reason to change the default. Rules were fixed before running (see "
         "`model/option_screen.py`). Data: `data/daily_opencity.csv` (production data + OpenCity gap-fill).", "",
         "## 1. How accurate is it with a REAL weather forecast?", "",
         "Models trained exactly as before (observed weather); only tomorrow's weather input changes. MAE µg/m³ / exact AQI category.", ""]
    hdr = ["Winter", "Days", "Tomorrow's weather", "V1", "V2", "Hybrid"]
    rows = []
    for w, r in res["realism"].items():
        for lab, nice in (("weather_persistence", "today's weather reused (primary bound)"),
                          ("forecast_48h", "real forecast, 48 h ahead"), ("forecast_24h", "real forecast, 24 h ahead"),
                          ("observed_weather", "observed (optimistic bound)")):
            x = r[lab]
            rows.append([w, r["days"], nice] + [f"{x[k]['mae']} / {100 * x[k]['cat_acc']:.1f}%" for k in ("v1", "v2", "hybrid")])
    L += [wb_table(hdr, rows), ""]
    rows = []
    for w, r in res["realism"].items():
        for lab in ("weather_persistence", "forecast_24h", "observed_weather"):
            c = r[lab]["hybrid_minus_v1"]
            rows.append([w, lab.replace("_", " "), f"{c['diff']:+.2f} ({c['pct']:+.1f}%)", f"[{c['ci95_diff_7d'][0]:+.2f}, {c['ci95_diff_7d'][1]:+.2f}]",
                         f"[{c['ci95_diff_14d'][0]:+.2f}, {c['ci95_diff_14d'][1]:+.2f}]", "yes" if c["excludes_zero_both"] else "no"])
    c = res.get("realism_pooled_hybrid_vs_v1_forecast_24h")
    if c:
        rows.append(["both winters", "forecast 24h", f"{c['diff']:+.2f} ({c['pct']:+.1f}%)", f"[{c['ci95_diff_7d'][0]:+.2f}, {c['ci95_diff_7d'][1]:+.2f}]",
                     f"[{c['ci95_diff_14d'][0]:+.2f}, {c['ci95_diff_14d'][1]:+.2f}]", "yes" if c["excludes_zero_both"] else "no"])
    L += ["**Hybrid − V1 (MAE µg/m³)** under each weather input:", "",
          wb_table(["Winter", "Weather", "Hybrid − V1", "95% CI 7-d", "95% CI 14-d", "Excludes 0 (both)"], rows), "",
          "Note: \"today's weather reused\" was described as a pessimistic bound in the backtest report. With real forecasts that "
          "holds for 2025-26 but NOT for 2024-25, where real forecasts gave a higher error. The archived forecasts differ "
          "from the archive (ERA5) weather the models are trained on when averaged over ALL seasons (wind lower by 2-4 km/h, humidity "
          "higher by 2-5 points); inside the winter test windows the difference is smaller and differs between winters "
          "(see experiments/forecast_training_experiment.md, section 3, which supersedes this sentence).", ""]
    L += ["## 2. Options screened (Hybrid, frozen per winter)", "",
          "Negative = the option has lower error than the reference Hybrid.", ""]
    rows = []
    for name, c in res["pooled"].items():
        rows.append([name.replace("_", " "), c["n"], c["mae_ref"], c["mae_opt"], f"{c['diff']:+.2f} ({c['pct']:+.1f}%)",
                     f"[{c['ci95_diff_7d'][0]:+.2f}, {c['ci95_diff_7d'][1]:+.2f}] / [{c['ci95_diff_14d'][0]:+.2f}, {c['ci95_diff_14d'][1]:+.2f}]",
                     f"{100 * c['cat_ref']:.1f}% → {100 * c['cat_opt']:.1f}%", f"{c['worst_single_winter_pct']:+.1f}%",
                     "**promising**" if c["promising"] else "no evidence"])
    L += [wb_table(["Option", "Days", "Ref MAE", "Option MAE", "Difference", "95% CI 7-d / 14-d blocks",
                    "Exact category", "Worst winter", "Verdict"], rows), ""]
    L += ["### Per winter", ""]
    rows = []
    for name, c in res["pooled"].items():
        for w, v in c["per_winter"].items():
            rows.append([name.replace("_", " "), w, v["n"], v["mae_ref"], v["mae_opt"], f"{v['diff']:+.2f} ({v['pct']:+.1f}%)",
                         f"[{v['ci95_diff_7d'][0]:+.2f}, {v['ci95_diff_7d'][1]:+.2f}]"])
    L += [wb_table(["Option", "Winter", "Days", "Ref MAE", "Option MAE", "Difference", "95% CI (7-d)"], rows), ""]
    L += ["### Fitted settings (from data before each window only)", ""]
    for w, i in res["winters"].items():
        extra = f"; forecast bias removed {i['O6_bias_removed']} (from {i['O6_bias_days']} days)" if "O6_bias_removed" in i else ""
        L.append(f"* {w}: {i['scored_days']} days ({i['days_with_day_parts']} with day-parts); error-correction alpha {i['O1_alpha']}{extra}")
    L.append("")
    return "\n".join(L)


def wb_table(h, rows):
    from winter_backtest_report import table
    return table(h, rows)


if __name__ == "__main__":  # pragma: no cover
    os.chdir(ROOT)
    out = run()
    Path("experiments/option_screen.json").write_text(json.dumps(out, indent=1, default=float))
    Path("experiments/option_screen.md").write_text(render(out))
    print(render(out))
