"""Provenance for the winter 2026-27 forward (shadow) test.

WHY THIS EXISTS. The frozen handler (backend/forecast_handler.py) already logs every shadow forecast under
pk="shadow" (first write wins, with an `inputs_sha256` fingerprint). What it does not keep is the evidence behind a
night's numbers: which hourly readings were really available at the 23:00 IST run, which weather call supplied
tomorrow's weather, which exact files produced the prediction. This module builds that evidence as ONE record per
forecast night (pk="provenance", sk=<target date>), written by backend/forward_logger.py just before the shadow rows.

Everything here is a PURE function of data handed in: no AWS, no network, no environment reads, no secrets. Nothing
in this module changes a forecast. It must not import anything that is not already part of the Lambda package.

What a record lets a reviewer do, without trusting anyone's memory:
  * see that no reading from the future was used  (`hourly_feed.cutoff`)
  * see which of the seven day-part inputs existed and, when not, why the Hybrid fell back to V1  (`hourly_feed.daypart`)
  * re-create tomorrow's feature row from stored data and re-run both models to the same numbers  (`inputs`)
  * tie the numbers to exact model and code files  (`software`), which scripts/forward_ops.py compares with the frozen
    manifest.
Only the information needed for that is stored; the OpenAQ key and the AWS credentials never enter a record.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

from . import weather as wx
from .daily import (DAYPARTS, IST, LAST3_HOURS, MAX_VALID_PM25, MIN_HOURS_PER_DAY, city_hourly, daypart_stats,
                    readings_to_hourly, to_ist)
from .features import DAYPART_FEATURES, feature_dict
from .hybrid import HYBRID_RULE, daypart_valid

SCHEMA = "kkh.forward_provenance.v1"
PARTITION = "provenance"
CUTOFF_HOUR_IST = 23                       # the nightly run is scheduled for 23:00 IST (EventBridge cron(30 17 * * ? *))
HOUR = timedelta(hours=1)
# Copy of model/forward_protocol.PROTOCOL["window_target_dates"] (model/ is not in the Lambda package);
# tests/test_forward_logging.py fails if the two ever differ.
WINDOW_TARGET_DATES = ("2026-10-15", "2027-02-15")
DP_PARTS = {**DAYPARTS, "last3": LAST3_HOURS}     # the six windows behind the seven day-part features (+ the day mean)


# ----------------------------------------------------------------------------- small helpers
def aware_utc(dt: datetime) -> datetime:
    """Same convention as core.daily.to_ist: a naive timestamp is taken to be UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def valid_value(v) -> bool:
    """Same validity rule as core.daily (not None / NaN, 0 <= value <= 1000)."""
    return v is not None and v == v and 0 <= v <= MAX_VALID_PM25


def jsonable(obj):
    """Plain JSON types only: NaN/inf -> None, dates -> ISO text, tuples/sets -> lists, numpy scalars -> Python."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, (float, Decimal)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {(k.isoformat() if isinstance(k, (date, datetime)) else str(k)): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [jsonable(v) for v in sorted(obj)]
    if hasattr(obj, "item"):                                   # numpy scalar
        return jsonable(obj.item())
    return str(obj)


def to_dynamo(obj) -> dict:
    """DynamoDB takes Decimal, not float. allow_nan=False makes a stray NaN fail loudly here instead of in AWS."""
    return json.loads(json.dumps(jsonable(obj), allow_nan=False), parse_float=Decimal)


def file_sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ----------------------------------------------------------------------------- the 23:00 IST cutoff
def split_complete(readings, run_utc: datetime):
    """(complete, incomplete). An hourly reading stamped `ts` covers [ts, ts + 1 h). It may be used only if that hour had
    ENDED when the run started - otherwise it is a partial hour or a future observation."""
    complete, incomplete = [], []
    for ts, val in readings:
        (complete if aware_utc(ts) + HOUR <= run_utc else incomplete).append((ts, val))
    return complete, incomplete


def target_day_start(target_iso: str) -> datetime:
    """00:00 IST of the target day, as an aware datetime."""
    return datetime.combine(date.fromisoformat(target_iso), time(0, 0), tzinfo=IST)


def is_late(generated_at_iso: str, target_iso: str) -> bool:
    """True when a forecast was generated at or after the start of the day it predicts. Such a row may have seen part of
    the answer (the target day's own measurements), so it must not enter the forward test."""
    return aware_utc(datetime.fromisoformat(generated_at_iso)) >= target_day_start(target_iso).astimezone(timezone.utc)


def split_late_rows(shadow: dict) -> tuple[dict, list[str]]:
    """Separate shadow rows that were generated too late. A row whose timestamps cannot be parsed is KEPT (the guard must
    never make logging stricter because of its own bug); forward_ops reports it instead."""
    keep, dropped = {}, []
    for v, rec in shadow.items():
        try:
            late = is_late(rec["generated_at"], rec["target_date"])
        except Exception:
            late = False
        if late:
            dropped.append(v)
        else:
            keep[v] = rec
    return keep, dropped


# ----------------------------------------------------------------------------- what the capture holds
class Capture:
    """Filled by backend/forward_logger.py while the frozen handler runs. Plain data; nothing here calls anything."""

    def __init__(self):
        self.fetches: list[dict] = []          # one per sensor: received + kept readings
        self.weather_calls: list[dict] = []
        self.result: dict | None = None        # the PRIMARY forecast result as run_forecast returned it
        self.shadow_args: tuple | None = None  # (history, d0, wx1, versions) exactly as shadow_forecasts got them
        self.shadow_out: dict | None = None
        self.late_dropped: list[str] = []
        self.errors: list[str] = []

    def record_fetch(self, sensor_id, dt_from, dt_to, readings) -> list:
        """Remember what the API returned and hand back only the readings whose hour had ended by the run time."""
        received = list(readings)
        kept, _ = split_complete(received, aware_utc(dt_to))
        self.fetches.append({"sensor_id": sensor_id, "dt_from": aware_utc(dt_from), "dt_to": aware_utc(dt_to),
                             "received": received, "kept": kept})
        return kept


# ----------------------------------------------------------------------------- hourly feed audit
def sensor_audit(fetch: dict, run_utc: datetime, d0: date) -> dict:
    received, kept = fetch["received"], fetch["kept"]
    rec_valid = [(aware_utc(t), v) for t, v in received if valid_value(v)]
    kept_valid = [(aware_utc(t), v) for t, v in kept if valid_value(v)]
    out = {
        "sensor_id": fetch["sensor_id"],
        "n_received": len(received), "n_valid_received": len(rec_valid),
        # the cutoff: what the API returned that was NOT a finished hour, and (must be 0) what was actually used
        "received_started_after_run": sum(1 for t, _ in rec_valid if t > run_utc),
        "received_hour_not_finished": sum(1 for t, _ in rec_valid if t + HOUR > run_utc),
        "used_hour_not_finished": sum(1 for t, _ in kept_valid if t + HOUR > run_utc),
        "used_started_after_run": sum(1 for t, _ in kept_valid if t > run_utc),
        "stamp_minutes_utc": sorted({t.minute for t, _ in rec_valid}),          # [30] = hours start on the IST hour
    }
    if kept_valid:
        latest = max(t for t, _ in kept_valid)
        out["latest_finished_hour_start_ist"] = to_ist(latest).isoformat()
        out["latest_finished_hour_end_ist"] = to_ist(latest + HOUR).isoformat()
        out["feed_lag_minutes"] = round((run_utc - (latest + HOUR)).total_seconds() / 60, 1)
    else:
        out["latest_finished_hour_start_ist"] = out["latest_finished_hour_end_ist"] = out["feed_lag_minutes"] = None
    hours = readings_to_hourly(kept).get(d0, {})
    out["origin_day_hours"] = sorted(hours)
    out["n_origin_day_hours"] = len(hours)
    return out


def part_coverage(city_hours: dict[int, float]) -> dict:
    """For each day-part window: which hours the city series has, how many are needed (core.daily: max(2, len//2))."""
    out = {}
    for name, rng in DP_PARTS.items():
        present = [h for h in rng if h in city_hours]
        need = max(2, len(rng) // 2)
        out[name] = {"hours_present": present, "of": len(rng), "needed": need, "ok": len(present) >= need}
    return out


def fallback_reasons(row: dict, city_hours: dict[int, float]) -> list[str]:
    """Why the Hybrid fell back to V1 - empty when all seven day-part inputs are finite."""
    if daypart_valid(row):
        return []
    reasons, n = [], len(city_hours)
    if n < MIN_HOURS_PER_DAY:
        reasons.append(f"origin day has {n} hourly city values; at least {MIN_HOURS_PER_DAY} are required")
    else:
        for name, c in part_coverage(city_hours).items():
            if not c["ok"]:
                reasons.append(f"day-part {name}: {len(c['hours_present'])} of {c['of']} hours present, {c['needed']} needed")
    if not reasons:
        reasons.append("non-finite day-part feature(s): " + ", ".join(f for f in DAYPART_FEATURES if not finite(row.get(f))))
    return reasons


def hourly_feed_block(cap: Capture, run_utc: datetime, d0: date, history: dict, row: dict) -> dict:
    audits = [sensor_audit(f, run_utc, d0) for f in cap.fetches]
    by_sensor = {f["sensor_id"]: readings_to_hourly(f["kept"]).get(d0, {}) for f in cap.fetches}
    city = city_hourly([{d0: h} for h in by_sensor.values() if h]).get(d0, {})
    per_hour = {h: sum(1 for hs in by_sensor.values() if h in hs) for h in range(24)}
    recomputed = daypart_stats(city)
    stored = history.get(d0, {})
    same = all((a != a and (b is None or b != b)) or (a == b) for a, b in
               ((recomputed[k], stored.get(k)) for k in recomputed))
    now_ist = to_ist(run_utc)
    cutoff = datetime.combine(d0, time(CUTOFF_HOUR_IST, 0), tzinfo=IST)
    return {
        "cutoff": {
            "scheduled_cutoff_ist": cutoff.isoformat(),
            "run_ist": now_ist.isoformat(),
            "minutes_after_scheduled_cutoff": round((run_utc - cutoff.astimezone(timezone.utc)).total_seconds() / 60, 1),
            "run_before_target_day_starts": not is_late(run_utc.isoformat(), (d0 + timedelta(days=1)).isoformat()),
            "rule": "an hourly reading is used only if its hour had ended at the run time",
            "readings_dropped_as_unfinished": sum(a["received_hour_not_finished"] for a in audits),
            "readings_used_from_unfinished_hours": sum(a["used_hour_not_finished"] for a in audits),
            "readings_used_started_after_run": sum(a["used_started_after_run"] for a in audits),
        },
        "n_sensors_queried": len(audits),
        "n_sensors_with_origin_day_hours": sum(1 for a in audits if a["n_origin_day_hours"]),
        "sensors": audits,
        "origin_day": {
            "date": d0.isoformat(),
            "city_hours_present": sorted(city),
            "n_city_hours": len(city),
            "hours_missing_0_to_22": [h for h in range(23) if h not in city],
            "stations_per_hour": {h: n for h, n in per_hour.items() if n},
            "last_hour_present": max(city) if city else None,
            "has_hour_21": 21 in city, "has_hour_22": 22 in city, "has_hour_23": 23 in city,
            "city_hourly_pm25": {h: city[h] for h in sorted(city)},
            "hourly_pm25_by_sensor": {sid: {h: hs[h] for h in sorted(hs)} for sid, hs in by_sensor.items() if hs},
        },
        "daypart": {
            "rule": HYBRID_RULE,
            "valid": daypart_valid(row),
            "features": {f: row.get(f) for f in DAYPART_FEATURES},
            "coverage": part_coverage(city),
            "fallback_reasons": fallback_reasons(row, city),
            "recomputed_from_hourly_equals_history": same,
            "note": "at the 23:00 IST run hour 23 does not exist yet, so dp_late uses hours 20-22 and dp_last3 hours 21-22 "
                    "(the models were trained on full days: the known train/serve gap, model/EVALUATION_PROTOCOL.md)",
        },
    }


# ----------------------------------------------------------------------------- weather block
def weather_block(cap: Capture, d0: date, history: dict, wx1: dict, run_utc: datetime) -> dict:
    return {
        "source": "Open-Meteo forecast API (live), hourly values reduced to daily means",
        "url": wx.FORECAST_URL,
        "hourly_variables": list(wx.HOURLY_VARS),
        "timezone": "Asia/Kolkata",
        "model_selection": "API default (best match) - not pinned by this project",
        "daily_reduction": f"core.weather.hourly_to_daily_weather; a day needs >= {wx.MIN_HOURS} hourly values",
        "calls": list(cap.weather_calls),          # lat, lon, past_days, forecast_days, retrieved_at_utc, days_returned
        "origin_day_weather": {k: history.get(d0, {}).get(k) for k in wx.WEATHER_KEYS},
        "target_day_weather": {k: wx1.get(k) for k in wx.WEATHER_KEYS},
        "target_day_value_is": "a forecast issued before the target day began" if not is_late(
            run_utc.isoformat(), (d0 + timedelta(days=1)).isoformat()) else "NOT a pure forecast: the run was late",
        "note": "archived training weather (ERA5) is not what is used live; see model/MODEL_VERSIONS.md for the caveat",
    }


# ----------------------------------------------------------------------------- the record
def primary_block(result: dict) -> dict:
    keys = ("model_version", "served_model", "pm25", "pm25_low", "pm25_high", "aqi", "category", "n_features",
            "missing_features", "today_pm25", "n_stations")
    out = {k: result.get(k) for k in keys}
    out["school_decision"] = (result.get("guidance") or {}).get("decision")
    return out


def build_provenance(cap: Capture, *, env: dict, software: dict, request: dict, written_late: bool = False) -> dict:
    """The full record for one forecast night. Raises if the capture is incomplete (the caller logs it and goes on)."""
    if cap.result is None or cap.shadow_args is None or cap.shadow_out is None or not cap.fetches:
        raise ValueError("capture incomplete: need the primary result, the shadow inputs/outputs and the sensor fetches")
    history, d0, wx1, versions = cap.shadow_args
    run_utc = aware_utc(datetime.fromisoformat(cap.result["generated_at"]))
    target = d0 + timedelta(days=1)
    used = {d: rec for d, rec in history.items() if d <= d0}          # feature_dict reads nothing later than d0
    row = feature_dict(used, d0, wx1)
    shadow = {v: {k: r.get(k) for k in ("model_version", "served_model", "pm25", "pm25_low", "pm25_high", "aqi", "category",
                                        "n_features", "daypart_valid", "inputs_sha256", "missing_features", "error")
                  if k in r} for v, r in cap.shadow_out.items()}
    fingerprints = {r.get("inputs_sha256") for r in cap.shadow_out.values() if r.get("inputs_sha256")}
    window = WINDOW_TARGET_DATES
    rec = {
        "pk": PARTITION,
        "sk": target.isoformat() if not written_late else f"{target.isoformat()}#late#{run_utc.isoformat()}",
        "schema": SCHEMA,
        "status": "late_run_not_logged" if written_late else "logged",
        "target_date": target.isoformat(),
        "origin_date": d0.isoformat(),
        "forecast_generated_at_utc": run_utc.isoformat(),
        "forecast_generated_at_ist": to_ist(run_utc).isoformat(),
        "in_protocol_window": window[0] <= target.isoformat() <= window[1],
        "protocol_window_target_dates": list(window),
        "request": request,
        "environment": env,
        "primary": primary_block(cap.result),
        "shadow": shadow,
        "shadow_versions_requested": list(versions),
        "shadow_rows_dropped_as_late": list(cap.late_dropped),
        "all_versions_saw_identical_inputs": len(fingerprints) == 1,
        "hourly_feed": hourly_feed_block(cap, run_utc, d0, used, row),
        "weather": weather_block(cap, d0, used, wx1, run_utc),
        "inputs": {
            "how_to_reproduce": "feature_dict(history, origin_date, target_weather) -> feature row -> each model; "
                                "python scripts/forward_ops.py reproduce does exactly this",
            "history": used,                     # one record per day <= origin_date: city PM2.5, weather, day-part means
            "target_weather": wx1,
            "feature_row": row,                  # every candidate feature (NaN stored as null)
            "inputs_sha256": next(iter(fingerprints)) if len(fingerprints) == 1 else None,
        },
        "software": software,
        "capture_errors": list(cap.errors),
    }
    return jsonable(rec)


def failure_record(*, error: str, now_utc: datetime, request: dict, env: dict) -> dict:
    ist_day = to_ist(now_utc).date().isoformat()
    return jsonable({"pk": PARTITION, "sk": f"{ist_day}#failed#{now_utc.isoformat()}", "schema": SCHEMA, "status": "failed",
                     "failed_at_utc": now_utc.isoformat(), "failed_at_ist": to_ist(now_utc).isoformat(),
                     "error": error[:500], "request": request, "environment": env})
