#!/usr/bin/env python3
"""Operate the winter 2026-27 forward (shadow) test WITHOUT looking at accuracy.

    python scripts/forward_ops.py ops       --table <DynamoDB table> [--region ap-south-1]     # is logging working?
    python scripts/forward_ops.py reproduce --table <DynamoDB table> [--date 2026-10-15]       # can every night be re-created?
    ... either command also takes  --input items.json  (a JSON list of items) instead of --table, for offline use.

`ops` reports only operational facts: which nights were logged, whether the Hybrid had its day-part inputs, how late the
hourly feed was, whether the cutoff held, whether V1 kept serving. It never computes an error, an MAE or a category hit
rate - the protocol (model/EVALUATION_PROTOCOL.md) allows ONE accuracy analysis, not before 2027-02-16, by
model/forward_eval.py. It only COUNTS how many nights already have an actual attached.

`reproduce` rebuilds each night's feature row from the stored history, checks the fingerprint, re-runs the models from the
repository's model files and compares with the logged numbers; it also compares the code/model hashes recorded that night
with model/forward_test_manifest.json. Reading from DynamoDB needs your own AWS credentials (this script never prints them).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "backend", ROOT / "model"):
    sys.path.insert(0, str(p))

from core.daily import IST                                    # noqa: E402
from core.forward_provenance import PARTITION                 # noqa: E402
from core.hybrid import HYBRID_VERSION, hybrid_pick           # noqa: E402
from forward_protocol import PROTOCOL                         # noqa: E402

V1, V2 = "v1_25_baseline", "v2_35_candidate"
ALL_VERSIONS = (V1, V2, HYBRID_VERSION)
WINDOW = tuple(PROTOCOL["window_target_dates"])
ANALYSIS_NOT_BEFORE = PROTOCOL["analysis_not_before"]


# ----------------------------------------------------------------------------- loading
def plain(obj):
    """DynamoDB Decimals -> int/float, recursively."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [plain(v) for v in obj]
    return obj


def load_items(table_name: str, region: str | None = None) -> list[dict]:
    import boto3
    from boto3.dynamodb.conditions import Key
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    items = []
    for pk in ("delhi", "shadow", PARTITION):
        kw = {"KeyConditionExpression": Key("pk").eq(pk)}
        while True:
            r = table.query(**kw)
            items += r["Items"]
            if "LastEvaluatedKey" not in r:
                break
            kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    return [plain(i) for i in items]


def split(items: list[dict]):
    primary = {i["sk"]: i for i in items if i.get("pk") == "delhi"}
    shadow: dict[str, dict[str, dict]] = {}
    for i in items:
        if i.get("pk") == "shadow":
            target, _, version = i["sk"].partition("#")
            shadow.setdefault(target, {})[version] = i
    prov = {i["sk"]: i for i in items if i.get("pk") == PARTITION}
    return primary, shadow, prov


def _med(xs):
    return round(statistics.median(xs), 1) if xs else None


# ----------------------------------------------------------------------------- ops
def expected_targets(today: date) -> list[str]:
    """Window targets whose forecast should already exist: the forecast for day T is made at 23:00 IST on T-1."""
    a, b = (date.fromisoformat(x) for x in WINDOW)
    end = min(b, today)
    return [(a + timedelta(days=i)).isoformat() for i in range(max(0, (end - a).days + 1))]


def ops_report(items: list[dict], today: date) -> dict:
    primary, shadow, prov = split(items)
    window_lo, window_hi = WINDOW
    in_window = lambda t: window_lo <= t <= window_hi                       # noqa: E731
    expected = expected_targets(today)
    complete = sorted(t for t, rows in shadow.items() if all(v in rows and "error" not in rows[v] for v in ALL_VERSIONS))
    logged = [p for sk, p in prov.items() if p.get("status") == "logged"]
    late = [p for p in prov.values() if p.get("status") == "late_run_not_logged"]
    failed = [p for p in prov.values() if p.get("status") == "failed"]
    win_logged = [p for p in logged if p.get("in_protocol_window")]
    pre_logged = [p for p in logged if not p.get("in_protocol_window")]

    def summarise(plist):
        valid = [p for p in plist if p["hourly_feed"]["daypart"]["valid"]]
        reasons = Counter()
        for p in plist:
            for r in p["hourly_feed"]["daypart"]["fallback_reasons"]:
                reasons[r.split(":")[0] if r.startswith("day-part") else r] += 1
        lags = [s["feed_lag_minutes"] for p in plist for s in p["hourly_feed"]["sensors"] if s.get("feed_lag_minutes") is not None]
        cut = [p["hourly_feed"]["cutoff"] for p in plist]
        return {
            "nights": len(plist),
            "hybrid_had_all_7_daypart_inputs": len(valid),
            "hybrid_fell_back_to_v1": len(plist) - len(valid),
            "fallback_reasons": dict(reasons),
            "nights_with_hour_21": sum(p["hourly_feed"]["origin_day"]["has_hour_21"] for p in plist),
            "nights_with_hour_22": sum(p["hourly_feed"]["origin_day"]["has_hour_22"] for p in plist),
            "feed_lag_minutes_median_over_sensor_nights": _med(lags),
            "feed_lag_minutes_max": round(max(lags), 1) if lags else None,
            "cutoff": {
                "readings_dropped_as_unfinished": sum(c["readings_dropped_as_unfinished"] for c in cut),
                "readings_used_from_unfinished_hours": sum(c["readings_used_from_unfinished_hours"] for c in cut),
                "runs_not_before_target_day": sum(1 for c in cut if not c["run_before_target_day_starts"]),
                "minutes_after_2300_ist_min_max": ([min(c["minutes_after_scheduled_cutoff"] for c in cut),
                                                    max(c["minutes_after_scheduled_cutoff"] for c in cut)] if cut else None),
            },
            "nights_with_different_input_fingerprints": sum(1 for p in plist if not p["all_versions_saw_identical_inputs"]),
            "nights_with_capture_errors": sum(1 for p in plist if p.get("capture_errors")),
        }

    not_v1 = sorted(sk for sk, r in primary.items() if in_window(sk)
                    and (r.get("served_model") != V1 or r.get("model_version") != V1))
    routing = Counter(rows[HYBRID_VERSION].get("served_model") for rows in shadow.values()
                      if HYBRID_VERSION in rows and "error" not in rows[HYBRID_VERSION] and in_window(next(iter(rows.values()))["target_date"]))
    matched = [t for t in complete if in_window(t) and all("actual_pm25" in shadow[t][v] for v in ALL_VERSIONS)]
    matched_dp = [t for t in matched if shadow[t][HYBRID_VERSION].get("daypart_valid")]
    return {
        "today_ist": today.isoformat(),
        "window_target_dates": list(WINDOW),
        "analysis_not_before": ANALYSIS_NOT_BEFORE,
        "logging_has_started": bool(complete or logged),
        "expected_forecast_nights_so_far": len(expected),
        "nights_with_all_three_shadow_rows": len([t for t in complete if in_window(t)]),
        "missing_nights": [t for t in expected if t not in complete],
        "nights_with_shadow_rows_but_no_provenance": sorted(t for t in complete if t not in prov),
        "nights_with_provenance_but_no_complete_shadow_rows": sorted(p["target_date"] for p in logged if p["target_date"] not in complete),
        "shadow_rows_with_errors": sorted(f"{t}#{v}" for t, rows in shadow.items() for v, r in rows.items() if "error" in r),
        "late_runs_refused": sorted(p["sk"] for p in late),
        "failed_runs": sorted(p["sk"] for p in failed),
        "primary_rows_in_window_not_served_by_v1": not_v1,
        "hybrid_routing_in_window": dict(routing),
        "in_window": summarise(win_logged),
        "pre_window_rehearsal": summarise(pre_logged),
        "progress_toward_minimums": {"matched_days_with_actual": len(matched), "needed": PROTOCOL["min_matched_days"],
                                     "matched_daypart_days": len(matched_dp), "needed_daypart": PROTOCOL["min_matched_daypart_days"]},
        "accuracy": f"NOT COMPUTED here by design. One analysis, not before {ANALYSIS_NOT_BEFORE}: python model/forward_eval.py",
    }


def format_ops(rep: dict) -> str:
    L = [f"Forward-test operations as of {rep['today_ist']} (IST)   window of target dates {rep['window_target_dates'][0]} .. {rep['window_target_dates'][1]}",
         f"  logging has started          : {'YES' if rep['logging_has_started'] else 'NO - no shadow/provenance rows found'}",
         f"  nights with all 3 shadow rows: {rep['nights_with_all_three_shadow_rows']} of {rep['expected_forecast_nights_so_far']} expected so far"]
    for key, label in (("missing_nights", "missing nights"), ("nights_with_shadow_rows_but_no_provenance", "shadow rows without provenance"),
                       ("nights_with_provenance_but_no_complete_shadow_rows", "provenance without 3 shadow rows"),
                       ("shadow_rows_with_errors", "shadow rows with an error"), ("late_runs_refused", "late runs refused"),
                       ("failed_runs", "failed runs"), ("primary_rows_in_window_not_served_by_v1", "primary rows NOT served by V1")):
        v = rep[key]
        L.append(f"  {label:29}: {len(v) if v else 0}" + (f"  {v[:8]}{' ...' if len(v) > 8 else ''}" if v else ""))
    L.append(f"  hybrid routing (window)      : {rep['hybrid_routing_in_window']}")
    for name in ("in_window", "pre_window_rehearsal"):
        s = rep[name]
        L.append(f"  [{name}] provenance nights: {s['nights']}")
        if s["nights"]:
            L += [f"      hybrid had all 7 day-part inputs: {s['hybrid_had_all_7_daypart_inputs']}   fell back to V1: {s['hybrid_fell_back_to_v1']}",
                  f"      reasons: {s['fallback_reasons'] or '-'}",
                  f"      hour 21 present: {s['nights_with_hour_21']}   hour 22 present: {s['nights_with_hour_22']}   "
                  f"feed lag median/max (min): {s['feed_lag_minutes_median_over_sensor_nights']}/{s['feed_lag_minutes_max']}",
                  f"      cutoff: {s['cutoff']}",
                  f"      input fingerprints differ: {s['nights_with_different_input_fingerprints']}   capture errors: {s['nights_with_capture_errors']}"]
    p = rep["progress_toward_minimums"]
    L += [f"  matched days with an actual  : {p['matched_days_with_actual']} (need {p['needed']})   with day-parts: {p['matched_daypart_days']} (need {p['needed_daypart']})",
          f"  accuracy                     : {rep['accuracy']}"]
    return "\n".join(L)


# ----------------------------------------------------------------------------- reproduce
def _same(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and float(a) == float(b))


def reproduce_item(prov: dict, shadow_rows: dict, manifest: dict | None, tol: float = 0.011) -> list[str]:
    """Return the list of problems for one night (empty = fully reproduced). Needs xgboost and the repo model files."""
    import forecast_handler as fh
    from core.features import feature_dict
    problems = []
    inp = prov["inputs"]
    d0 = date.fromisoformat(prov["origin_date"])
    history = {date.fromisoformat(d): rec for d, rec in inp["history"].items()}
    row = feature_dict(history, d0, inp["target_weather"])
    stored = inp["feature_row"]
    bad = [k for k in stored if not _same(None if row.get(k) != row.get(k) else row.get(k), stored[k])]
    if bad or set(row) != set(stored):
        problems.append(f"feature row rebuilt from the stored history differs in {bad[:5] or 'its columns'}")
    fp = fh._inputs_fingerprint(row)
    if fp != inp.get("inputs_sha256"):
        problems.append("fingerprint of the rebuilt row does not match the stored inputs_sha256")
    for v in prov["shadow_versions_requested"]:
        r = shadow_rows.get(v)
        if r is None:
            problems.append(f"{v}: no shadow row for this night")
            continue
        if "error" in r:
            problems.append(f"{v}: logged with an error ({r['error']})")
            continue
        if r.get("inputs_sha256") != fp:
            problems.append(f"{v}: logged inputs_sha256 differs from the rebuilt row")
        served = hybrid_pick(row) if v == HYBRID_VERSION else v
        if served != r.get("served_model"):
            problems.append(f"{v}: logged served_model {r.get('served_model')} but the rule gives {served}")
        booster, meta = fh._load_model(served)
        feats = [row[n] for n in meta["features"]]
        pm, _lo, _hi = fh._predict(booster, meta, feats)
        if not math.isfinite(pm) or abs(round(pm, 2) - float(r["pm25"])) > tol:
            problems.append(f"{v}: re-run gives {pm:.2f} but {r['pm25']} was logged")
    if manifest:
        sw = prov.get("software", {}).get("files_sha256", {})
        diff = sorted(k for k, h in manifest["files_sha256"].items() if k in sw and sw[k] != h)
        if diff:
            problems.append(f"files that produced this night differ from the frozen manifest: {diff}")
    return problems


def load_manifest():
    p = ROOT / "model" / "forward_test_manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ops", "reproduce"])
    ap.add_argument("--table")
    ap.add_argument("--region")
    ap.add_argument("--input", help="JSON file with a list of items instead of reading DynamoDB")
    ap.add_argument("--date", help="reproduce only this target date")
    ap.add_argument("--today", help="override today's IST date (YYYY-MM-DD), ops only")
    ap.add_argument("--json", action="store_true", help="ops: print the report as JSON")
    a = ap.parse_args(argv)
    if bool(a.table) == bool(a.input):
        ap.error("give exactly one of --table or --input")
    items = plain(json.load(open(a.input))) if a.input else load_items(a.table, a.region)

    if a.command == "ops":
        today = date.fromisoformat(a.today) if a.today else datetime.now(timezone.utc).astimezone(IST).date()
        rep = ops_report(items, today)
        print(json.dumps(rep, indent=2) if a.json else format_ops(rep))
        return 0

    primary, shadow, prov = split(items)
    manifest = load_manifest()
    nights = sorted(sk for sk, p in prov.items() if p.get("status") == "logged" and (not a.date or sk == a.date))
    if not nights:
        print("no provenance records to reproduce" + (f" for {a.date}" if a.date else ""))
        return 1
    failures = 0
    for sk in nights:
        problems = reproduce_item(prov[sk], shadow.get(sk, {}), manifest)
        print(f"{'OK  ' if not problems else 'FAIL'} {sk}")
        for pr in problems:
            print(f"     - {pr}")
        failures += bool(problems)
    print(f"{len(nights) - failures} of {len(nights)} night(s) reproduced")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
