"""Score the winter 2026-27 shadow test with the FROZEN rules in model/forward_protocol.py.

    python model/forward_eval.py --table <DynamoDB table> --out experiments/forward_test_result.json     # read-only
    python model/forward_eval.py --input shadow_rows.json --out ...                                     # offline file

Input rows are the items logged under pk="shadow" (sk = "<target date>#<version>") with their actual_pm25 filled in.

What it does (model/EVALUATION_PROTOCOL.md has the reasoning):
  * keeps only forecasts whose target date is inside the window and drops, with counted reasons, any day on which the
    three versions were not all logged, all on identical inputs, error-free, with an actual;
  * scores v1, v2, hybrid and "tomorrow = today" on those SAME matched days: MAE (primary), RMSE, R2, AQI-category
    accuracy (secondary), for the complete eligible set, the days with valid day-parts and the days without;
  * puts a 95% paired 7-day block-bootstrap interval on the MAE change and on the category-accuracy change;
  * applies the promotion rule exactly as frozen. It never reads the data before the analysis date unless --force-early,
    which stamps the output "NOT A VALID ANALYSIS".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.aqi import pm25_to_category  # noqa: E402
from forward_protocol import PROTOCOL as P  # noqa: E402

VERSIONS = [P["baseline"], *P["candidates"]]


# ------------------------------------------------------------------ loading
def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def export_shadow(table) -> list[dict]:
    """Read every pk="shadow" item (read-only query, paginated)."""
    from boto3.dynamodb.conditions import Key
    items, kw = [], {"KeyConditionExpression": Key("pk").eq("shadow")}
    while True:
        r = table.query(**kw)
        items += r["Items"]
        if not r.get("LastEvaluatedKey"):
            return items
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]


def match_days(rows: list[dict]) -> tuple[list[dict], dict]:
    """Group rows by target date; keep days where every version is usable. Returns (days, exclusion counts)."""
    lo, hi = (date.fromisoformat(x) for x in P["window_target_dates"])
    by_day: dict[date, dict[str, dict]] = {}
    for r in rows:
        sk = str(r.get("sk", ""))
        day_s, _, ver = sk.partition("#")
        ver = r.get("model_version", ver)
        try:
            d = date.fromisoformat(day_s)
        except ValueError:
            continue
        if lo <= d <= hi and ver in VERSIONS:
            by_day.setdefault(d, {})[ver] = r
    excl = {"no_forecast_logged": 0, "a_version_missing": 0, "a_version_errored": 0, "no_actual": 0,
            "inputs_differ_between_versions": 0}
    days = []
    for d in sorted(by_day):
        recs = by_day[d]
        if any(v not in recs for v in VERSIONS):
            excl["a_version_missing"] += 1
        elif any("error" in recs[v] or _f(recs[v].get("pm25")) is None for v in VERSIONS):
            excl["a_version_errored"] += 1
        elif _f(recs[P["baseline"]].get("actual_pm25")) is None:
            excl["no_actual"] += 1
        elif len({recs[v].get("inputs_sha256") for v in VERSIONS}) != 1:
            excl["inputs_differ_between_versions"] += 1
        else:
            base = recs[P["baseline"]]
            days.append({"date": d, "actual": _f(base["actual_pm25"]), "today": _f(base.get("today_pm25")),
                         "daypart_valid": bool(recs[P["candidates"][0]].get("daypart_valid")),
                         "pred": {v: _f(recs[v]["pm25"]) for v in VERSIONS},
                         "served": {v: recs[v].get("served_model", v) for v in VERSIONS}})
    n_window = (hi - lo).days + 1
    excl["no_forecast_logged"] = max(0, n_window - len(by_day))
    return days, excl


# ------------------------------------------------------------------ statistics
def _blocks(dates):
    b: dict[int, list[int]] = {}
    for i, d in enumerate(dates):
        b.setdefault(d.toordinal() // P["bootstrap"]["block_days"], []).append(i)
    return [np.array(v) for v in b.values()]


def _boot(dates, stat):
    """Percentile interval of stat(index array) over a paired block bootstrap."""
    blocks = _blocks(dates)
    rng = np.random.default_rng(P["bootstrap"]["seed"])
    vals = [stat(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]))
            for _ in range(P["bootstrap"]["n_boot"])]
    a = (1 - P["ci_level"]) / 2 * 100
    lo, hi = np.percentile(vals, [a, 100 - a])
    return [round(float(lo), 2), round(float(hi), 2)]


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "mae": round(float(np.abs(e).mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
            "r2": round(1 - float(np.sum(e ** 2)) / ss_tot, 3) if ss_tot > 0 else None,
            "category_accuracy": round(float(np.mean([pm25_to_category(a) == pm25_to_category(b) for a, b in zip(p, y)])), 3)}


def compare(days: list[dict], cand: str) -> dict:
    y = np.array([d["actual"] for d in days])
    a = np.array([d["pred"][P["baseline"]] for d in days])
    b = np.array([d["pred"][cand] for d in days])
    dates = [d["date"] for d in days]
    ea, eb = np.abs(a - y), np.abs(b - y)
    ca = np.array([pm25_to_category(x) == pm25_to_category(t) for x, t in zip(a, y)], float)
    cb = np.array([pm25_to_category(x) == pm25_to_category(t) for x, t in zip(b, y)], float)
    mae_pct = lambda ix: (eb[ix].mean() - ea[ix].mean()) / ea[ix].mean() * 100          # noqa: E731
    acc_pp = lambda ix: (cb[ix].mean() - ca[ix].mean()) * 100                           # noqa: E731
    allix = np.arange(len(y))
    return {"mae_change_pct": round(float(mae_pct(allix)), 2), "mae_change_ci95": _boot(dates, mae_pct),
            "category_accuracy_change_pp": round(float(acc_pp(allix)), 2), "category_accuracy_change_ci95": _boot(dates, acc_pp)}


def segment_report(days: list[dict]) -> dict:
    if not days:
        return {"n": 0}
    y = [d["actual"] for d in days]
    out = {"n": len(days), "models": {v: metrics(y, [d["pred"][v] for d in days]) for v in VERSIONS}}
    pers = [d["today"] for d in days]
    if all(x is not None for x in pers):
        out["models"]["persistence_tomorrow_equals_today"] = metrics(y, pers)
    if len(days) >= 2:
        out["vs_baseline"] = {c: compare(days, c) for c in P["candidates"]}
    return out


# ------------------------------------------------------------------ decision (frozen rule)
def decide(report: dict) -> dict:
    gate = report["segments"][P["gate_segment"]]
    n, n_dp = gate["n"], report["segments"]["daypart_days"]["n"]
    out = {"n_matched_days": n, "n_matched_daypart_days": n_dp, "candidates": {}}
    if n < P["min_matched_days"] or n_dp < P["min_matched_daypart_days"]:
        out.update(decision="INCONCLUSIVE_KEEP_V1",
                   reason=f"need >= {P['min_matched_days']} matched winter days and >= {P['min_matched_daypart_days']} "
                          f"with day-parts; have {n} and {n_dp}. Nothing may be claimed; v1 stays the default.")
        return out
    passed = []
    for c in P["candidates"]:
        v = gate["vs_baseline"][c]
        checks = {"mae_at_least_required_lower": v["mae_change_pct"] <= -P["mae_reduction_required_pct"],
                  "mae_ci_upper_below_zero": v["mae_change_ci95"][1] < P["mae_ci_upper_must_be_below_pct"],
                  "category_accuracy_not_materially_worse": v["category_accuracy_change_pp"] >= -P["category_accuracy_max_drop_pp"]}
        out["candidates"][c] = {**checks, "passes": all(checks.values())}
        if all(checks.values()):
            passed.append(c)
    n_no_dp = report["segments"]["no_daypart_days"]["n"]
    if not passed:
        out.update(decision="KEEP_V1", reason="no candidate met every condition on the complete eligible set")
    elif len(passed) == 2:
        pick = "v3_hybrid" if n_no_dp > 0 else "v2_35_candidate"      # frozen tie rule: hybrid whenever any gap day exists
        out.update(decision=f"PROMOTE_{pick.upper()}", reason="both candidates passed; frozen tie rule applied")
    else:
        out.update(decision=f"PROMOTE_{passed[0].upper()}", reason="only this candidate met every condition")
    return out


def build_report(rows: list[dict]) -> dict:
    days, excl = match_days(rows)
    segs = {"complete_eligible": days,
            "daypart_days": [d for d in days if d["daypart_valid"]],
            "no_daypart_days": [d for d in days if not d["daypart_valid"]]}
    rep = {"protocol": P, "exclusions": excl,
           "segments": {k: segment_report(v) for k, v in segs.items()},
           "note": "Only matched days (all versions logged, same inputs, no errors, actual known) are scored."}
    rep["decision"] = decide(rep)
    return rep


def print_report(rep: dict) -> None:
    print(f"matched winter days: {rep['decision']['n_matched_days']} "
          f"(with day-parts {rep['decision']['n_matched_daypart_days']}) | excluded: {rep['exclusions']}")
    for name, seg in rep["segments"].items():
        print(f"\n== {name}  (n={seg['n']})")
        for m, r in seg.get("models", {}).items():
            print(f"  {m:36} MAE {r['mae']:>6}  RMSE {r['rmse']:>6}  R2 {r['r2']}  category-acc {r['category_accuracy']}")
        for c, v in seg.get("vs_baseline", {}).items():
            print(f"  {c:36} vs v1: MAE {v['mae_change_pct']:+.1f}% (95% CI {v['mae_change_ci95']}) | "
                  f"category-acc {v['category_accuracy_change_pp']:+.1f} pp (95% CI {v['category_accuracy_change_ci95']})")
    d = rep["decision"]
    print(f"\nDECISION: {d['decision']} - {d['reason']}")
    for c, v in d["candidates"].items():
        print(f"  {c}: {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="JSON file: list of shadow items")
    src.add_argument("--table", help="DynamoDB table to read (read-only query of pk='shadow')")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--today", help="override today's date (testing)")
    ap.add_argument("--force-early", action="store_true", help="run before the analysis date; output is stamped invalid")
    args = ap.parse_args(argv)
    today = date.fromisoformat(args.today) if args.today else date.today()
    early = today < date.fromisoformat(P["analysis_not_before"])
    if early and not args.force_early:
        sys.exit(f"The frozen protocol allows one analysis, not before {P['analysis_not_before']} (today {today}). "
                 "Use operational checks only (model/EVALUATION_PROTOCOL.md section 7).")
    if args.input:
        rows = json.loads(Path(args.input).read_text())
    else:
        import boto3
        rows = export_shadow(boto3.resource("dynamodb").Table(args.table))
    rep = build_report(rows)
    if early:
        rep["VALID"] = False
        rep["WARNING"] = "NOT A VALID ANALYSIS: run before the analysis date; do not use for any decision"
        rep["decision"] = {**rep["decision"], "decision": "NOT_VALID_EARLY_LOOK"}
        print("*** " + rep["WARNING"] + " ***")
    print_report(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return rep


if __name__ == "__main__":
    main()
