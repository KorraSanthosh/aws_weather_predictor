"""Winter-by-winter check of the old (v1, 25 features) and new (v2, 35 features) models, plus the hybrid.

    python model/winter_check.py        # writes experiments/winter_check.json

Same walk-forward monthly refit, same recipe and same days as model/compare_versions.py (offline view). Winters are the
protocol's 15 Oct -> 15 Feb target-date windows. Descriptive: all of these winters were examined before; the clean winter test is
2026-27 (model/EVALUATION_PROTOCOL.md).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pipeline"))
import compare_versions as cv  # noqa: E402
from core.daily import DP_KEYS  # noqa: E402
from core.features import FEATURE_SETS  # noqa: E402
from core.hybrid import daypart_valid  # noqa: E402
from dev_cv_study import stats  # noqa: E402
from train import load_daily  # noqa: E402

WINTERS = {"2022-23 (Jan 1 - Feb 15 only; test starts 2023-01-01)": (date(2023, 1, 1), date(2023, 2, 15)),
           "2023-24": (date(2023, 10, 15), date(2024, 2, 15)),
           "2024-25": (date(2024, 10, 15), date(2025, 2, 15)),
           "2025-26": (date(2025, 10, 15), date(2026, 2, 15))}


def main():
    hist, _ = load_daily("data/daily.csv")
    live = cv.build_live_view()
    for d in hist:
        if d not in live and hist[d]["pm25"] is not None:
            live[d] = {"pm25": hist[d]["pm25"], **{k: float("nan") for k in DP_KEYS}}
    rows, _, y, targets, _, _ = cv.build_rows(hist, live, only_live_valid=False)
    y, t = np.asarray(y, float), np.array(targets)
    end = max(targets)
    P = {}
    for v in ("v1_25_baseline", "v2_35_candidate"):
        names = FEATURE_SETS[v]
        X = cv.matrix(rows, names)
        P[v] = cv.walk(X, X, y, targets, names, cv.TEST_START, end)
        print("done", v, flush=True)
    pm0 = np.array([r["pm_d0"] for r in rows])
    valid = np.array([daypart_valid(r) for r in rows])
    ids = sorted(set(P["v1_25_baseline"]) & set(P["v2_35_candidate"]))
    out = {"note": "descriptive; every winter here was examined before. Clean test = winter 2026-27.", "winters": {}}
    for name, (a, b) in list(WINTERS.items()) + [("ALL four winters pooled", (None, None))]:
        sel = [i for i in ids if ((a <= t[i] <= b) if a else any(lo <= t[i] <= hi for lo, hi in WINTERS.values()))]
        if len(sel) < 5:
            continue
        sub = {"all_days": sel, "days_with_dayparts": [i for i in sel if valid[i]],
               "days_without_dayparts": [i for i in sel if not valid[i]],
               "sudden_change_days(>=30%)": [i for i in sel if abs(y[i] - pm0[i]) / max(pm0[i], 1) >= 0.30]}
        rec = {}
        for seg, ix in sub.items():
            if len(ix) < 5:
                rec[seg] = {"n_days": len(ix), "too_few": True}
                continue
            yy, dts = y[ix], [t[i] for i in ix]
            p1 = np.array([P["v1_25_baseline"][i] for i in ix]); p2 = np.array([P["v2_35_candidate"][i] for i in ix])
            ph = np.array([P["v2_35_candidate"][i] if valid[i] else P["v1_25_baseline"][i] for i in ix])
            rec[seg] = {"n_days": len(ix), "persistence": stats(yy, pm0[ix]), "v1": stats(yy, p1), "v2": stats(yy, p2),
                        "hybrid": stats(yy, ph), "v2_vs_v1_mae": cv.paired_mae_change(yy, p1, p2, dts),
                        "hybrid_vs_v1_mae": cv.paired_mae_change(yy, p1, ph, dts)}
        out["winters"][name] = rec
    Path("experiments/winter_check.json").write_text(json.dumps(out, indent=1))
    for name, rec in out["winters"].items():
        print(f"\n=== {name}")
        for seg, r in rec.items():
            if r.get("too_few"):
                print(f"  {seg:28} n={r['n_days']} (too few)"); continue
            f = lambda k: f"{r[k]['mae']:>5} / {r[k]['cat_acc']:.3f}"      # noqa: E731
            print(f"  {seg:28} n={r['n_days']:>3} | MAE/cat  persist {f('persistence')} | v1 {f('v1')} | v2 {f('v2')} | "
                  f"hybrid {f('hybrid')} | v2 vs v1 {r['v2_vs_v1_mae']['pct']:+.1f}% {r['v2_vs_v1_mae']['ci95']}")


if __name__ == "__main__":
    main()
