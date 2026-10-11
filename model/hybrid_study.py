"""Descriptive history of the predeclared hybrid ("v3_hybrid": v2 when all 7 day-part inputs exist, else v1).

    python model/hybrid_study.py        # writes experiments/hybrid_study_historical.json

ONE rule, written down before this was run (core/hybrid.py); no thresholds or variants were tried. It reuses the stored
walk-forward predictions logic of model/compare_versions.py (same days, same recipe). The 2023-2026 period was already
examined, so this is a DESCRIPTION of how the router behaves, NOT evidence for promotion - the winter 2026-27 forward test
(model/EVALUATION_PROTOCOL.md) is the only evidence that counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import compare_versions as cv  # noqa: E402
from core.aqi import pm25_to_category  # noqa: E402
from core.features import FEATURE_SETS  # noqa: E402
from core.hybrid import daypart_valid  # noqa: E402
from train import load_daily  # noqa: E402

V1, V2 = cv.V1, cv.V2


def stats(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"mae": round(float(np.abs(e).mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
            "r2": round(1 - float((e ** 2).sum()) / float(((y - y.mean()) ** 2).sum()), 3),
            "cat_acc": round(float(np.mean([pm25_to_category(a) == pm25_to_category(b) for a, b in zip(p, y)])), 3)}


def main():
    hist, _ = load_daily("data/daily.csv")
    live = cv.build_live_view()
    rows_off, rows_live, y, targets, d0s, dropped = cv.build_rows(hist, live)
    end = max(targets)
    v1, v2 = FEATURE_SETS[V1], FEATURE_SETS[V2]
    X = {(v, view): cv.matrix(rows, names) for v, names in ((V1, v1), (V2, v2))
         for view, rows in (("off", rows_off), ("live", rows_live))}
    P = {}
    for label, (v, names, tr, te) in {"v1_offline": (V1, v1, "off", "off"), "v2_offline": (V2, v2, "off", "off"),
                                      "v1_live": (V1, v1, "off", "live"), "v2_live": (V2, v2, "off", "live")}.items():
        P[label] = cv.walk(X[(v, tr)], X[(v, te)], y, targets, names, cv.TEST_START, end)
        print("done", label)
    ids = sorted(set.intersection(*[set(p) for p in P.values()]))
    out = {"what": __doc__.split("\n\n")[0], "rule": "v2_35_candidate if all 7 day-part inputs finite else v1_25_baseline",
           "status": "DESCRIPTIVE on an already-examined period; not evidence for promotion",
           "test_days": len(ids), "no_variants_tried": True}
    for regime, rows in (("offline", rows_off), ("live", rows_live)):
        valid = {i: daypart_valid(rows[i]) for i in ids}
        pred = {"v1": {i: P[f"v1_{regime}"][i] for i in ids}, "v2": {i: P[f"v2_{regime}"][i] for i in ids}}
        pred["hybrid"] = {i: (pred["v2"][i] if valid[i] else pred["v1"][i]) for i in ids}
        subsets = {"full_period": ids, "days_with_valid_dayparts": [i for i in ids if valid[i]],
                   "days_without_dayparts_hybrid_falls_back_to_v1": [i for i in ids if not valid[i]],
                   "winter_15Oct-15Feb": [i for i in ids if cv.in_winter(targets[i])],
                   "winter_with_dayparts": [i for i in ids if cv.in_winter(targets[i]) and valid[i]]}
        reg = {}
        for name, sel in subsets.items():
            if not sel:
                continue
            yy, dts = y[sel], [targets[i] for i in sel]
            arr = {k: np.array([pred[k][i] for i in sel]) for k in pred}
            reg[name] = {"n_days": len(sel), **{k: stats(yy, arr[k]) for k in arr},
                         "hybrid_vs_v1_mae": cv.paired_mae_change(yy, arr["v1"], arr["hybrid"], dts),
                         "hybrid_vs_v2_mae": cv.paired_mae_change(yy, arr["v2"], arr["hybrid"], dts),
                         "v2_vs_v1_mae": cv.paired_mae_change(yy, arr["v1"], arr["v2"], dts)}
            print(regime, name, reg[name]["n_days"], "MAE v1/v2/hybrid", reg[name]["v1"]["mae"], reg[name]["v2"]["mae"],
                  reg[name]["hybrid"]["mae"], "| hybrid vs v1", reg[name]["hybrid_vs_v1_mae"]["pct"], reg[name]["hybrid_vs_v1_mae"]["ci95"])
        out[regime] = reg
    Path("experiments/hybrid_study_historical.json").write_text(json.dumps(out, indent=1))
    print("wrote experiments/hybrid_study_historical.json")


if __name__ == "__main__":
    main()
