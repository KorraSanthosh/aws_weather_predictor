"""Lambda entry point for the forward (shadow) test:  Handler: forward_logger.handler

It runs the FROZEN handler (backend/forecast_handler.py, unchanged) and only watches it:

    * the primary forecast, the dashboard row (pk="delhi"), the alert and the school decision are produced by the frozen
      code exactly as before - V1 serves unless MODEL_VERSION says otherwise (it is not set);
    * the shadow forecasts (SHADOW_VERSIONS) are produced and stored by the frozen code, first write wins;
    * in addition this wrapper writes ONE provenance record per night (pk="provenance", sk=<target date>, first write
      wins) holding the evidence described in core/forward_provenance.py.

What the wrapper may change - nothing else:
    1. Hourly readings whose hour had not ended at the run time are not passed on (they would be partial or future
       observations). With a well-behaved API this removes nothing; the count is recorded either way.
    2. Shadow rows generated at or after 00:00 IST of the day they predict are not stored (they may have seen part of the
       answer). Nothing is lost for the forward test: such a night is simply not eligible. The run itself still finishes.
Everything the wrapper adds is guarded: if recording fails the nightly forecast still completes, the error is logged and
recorded in the item. If SHADOW_VERSIONS is empty the wrapper calls the frozen handler untouched.

To go back to the plain frozen behaviour at any time, set the function handler back to forecast_handler.handler.
"""
from __future__ import annotations

import copy
import logging
import os
import platform
from datetime import datetime, timezone

import forecast_handler as fh                 # frozen, unchanged
from core import forward_provenance as fp
from core import openaq
from core import weather as wx

log = logging.getLogger(__name__)
WRAPPER_VERSION = "1"
_SOFTWARE: dict = {}


# ----------------------------------------------------------------------------- what produced the numbers
def _software() -> dict:
    """Hashes of the code and model files in this package (computed once per container). Keys are repo-relative paths so
    scripts/forward_ops.py can compare them with model/forward_test_manifest.json."""
    if _SOFTWARE:
        return _SOFTWARE
    core_dir = os.path.dirname(os.path.abspath(fp.__file__))
    files = {"backend/forecast_handler.py": fh.__file__, "backend/forward_logger.py": __file__}
    for name in ("features", "daily", "aqi", "hybrid", "openaq", "weather", "forward_provenance"):
        files[f"core/{name}.py"] = os.path.join(core_dir, f"{name}.py")
    try:
        base = fh._model_dir(None)
        for v in ("v1_25_baseline", "v2_35_candidate"):
            for f in ("model.json", "model_meta.json"):
                files[f"model/{v}/{f}"] = os.path.join(base, v, f)
        files["model/model.json"] = os.path.join(base, "model.json")          # the packaged default (V1)
        files["model/model_meta.json"] = os.path.join(base, "model_meta.json")
    except Exception as e:                                                    # pragma: no cover - defensive
        log.warning("model hashes unavailable: %s", e)
    build_info = None
    try:
        import json
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_info.json")) as f:
            build_info = json.load(f)
    except Exception:
        pass
    try:
        import xgboost
        xgb_v = xgboost.__version__
    except Exception:                                                         # pragma: no cover
        xgb_v = None
    _SOFTWARE.update({"files_sha256": {k: fp.file_sha256(p) for k, p in files.items()},
                      "build_info": build_info, "python": platform.python_version(), "xgboost": xgb_v,
                      "wrapper_version": WRAPPER_VERSION})
    return _SOFTWARE


def _env() -> dict:
    """The few non-secret settings that decide what is served and logged. No key, no ARN, no table name."""
    return {k: (os.environ.get(k) or None) for k in ("MODEL_VERSION", "SHADOW_VERSIONS", "SENSOR_IDS", "LAT", "LON")}


def _request(event, context) -> dict:
    ev = event if isinstance(event, dict) else {}
    return {"aws_request_id": getattr(context, "aws_request_id", None),
            "function_version": getattr(context, "function_version", None),
            "event_source": ev.get("source"), "event_detail_type": ev.get("detail-type"), "event_time": ev.get("time")}


def _scrub(text: str) -> str:
    key = os.environ.get("OPENAQ_API_KEY")
    return text.replace(key, "***") if key else text


def _table():
    import boto3
    return boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])


def _put_first_write_wins(table, item: dict) -> bool:
    try:
        table.put_item(Item=fp.to_dynamo(item), ConditionExpression="attribute_not_exists(sk)")
        return True
    except Exception as e:                       # already recorded for this night, or a DynamoDB hiccup
        log.warning("provenance item %s not written: %s", item.get("sk"), _scrub(str(e)))
        return False


# ----------------------------------------------------------------------------- watching the frozen handler
def _install(cap: fp.Capture, event, context, saved: list) -> None:
    orig_fetch, orig_weather = openaq.fetch_sensor_hours, wx.fetch_forecast_weather
    orig_run, orig_shadow, orig_store = fh.run_forecast, fh.shadow_forecasts, fh.store_shadow

    def fetch(sensor_id, dt_from, dt_to, api_key, *a, **k):
        readings = orig_fetch(sensor_id, dt_from, dt_to, api_key, *a, **k)
        try:
            return cap.record_fetch(sensor_id, dt_from, dt_to, readings)      # only finished hours are passed on
        except Exception as e:                                                # never break the forecast
            cap.errors.append(f"fetch capture: {type(e).__name__}: {e}")
            return readings

    def weather(lat, lon, past_days=10, forecast_days=3):
        out = orig_weather(lat, lon, past_days=past_days, forecast_days=forecast_days)
        try:
            cap.weather_calls.append({"lat": lat, "lon": lon, "past_days": past_days, "forecast_days": forecast_days,
                                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "days_returned": sorted(d.isoformat() for d in out)})
        except Exception as e:
            cap.errors.append(f"weather capture: {type(e).__name__}: {e}")
        return out

    def run_forecast(*a, **k):
        result = orig_run(*a, **k)
        try:
            cap.result = copy.deepcopy(result)
        except Exception as e:
            cap.errors.append(f"result capture: {type(e).__name__}: {e}")
        return result

    def shadow_forecasts(history, d0, wx1, versions):
        try:
            cap.shadow_args = (copy.deepcopy(history), d0, copy.deepcopy(wx1), list(versions))
        except Exception as e:
            cap.errors.append(f"shadow input capture: {type(e).__name__}: {e}")
        out = orig_shadow(history, d0, wx1, versions)
        try:
            cap.shadow_out = copy.deepcopy(out)
        except Exception as e:
            cap.errors.append(f"shadow output capture: {type(e).__name__}: {e}")
        return out

    def store_shadow(table, shadow, actuals):
        keep, cap.late_dropped = fp.split_late_rows(shadow)
        if cap.late_dropped:
            log.warning("shadow rows %s were generated after the target day began - not stored", cap.late_dropped)
        _record_provenance(table, cap, event, context)                         # before the shadow rows, never raises
        return orig_store(table, keep, actuals)

    for owner, name, new in ((openaq, "fetch_sensor_hours", fetch), (wx, "fetch_forecast_weather", weather),
                             (fh, "run_forecast", run_forecast), (fh, "shadow_forecasts", shadow_forecasts),
                             (fh, "store_shadow", store_shadow)):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, new)


def _uninstall(saved: list) -> None:
    while saved:
        owner, name, original = saved.pop()
        setattr(owner, name, original)


def _record_provenance(table, cap: fp.Capture, event, context) -> None:
    try:
        rec = fp.build_provenance(cap, env=_env(), software=_software(), request=_request(event, context),
                                  written_late=bool(cap.late_dropped))
        if _put_first_write_wins(table, rec):
            log.info("provenance recorded for target %s", rec["target_date"])
    except Exception as e:
        log.error("provenance not recorded: %s", _scrub(f"{type(e).__name__}: {e}"))


def _record_failure(exc: Exception, event, context) -> None:
    try:
        rec = fp.failure_record(error=_scrub(f"{type(exc).__name__}: {exc}"), now_utc=datetime.now(timezone.utc),
                                request=_request(event, context), env=_env())
        _put_first_write_wins(_table(), rec)
    except Exception as e:
        log.error("failure record not written: %s", _scrub(str(e)))


def handler(event, context):
    if not fh.parse_shadow_versions(os.environ.get("SHADOW_VERSIONS", "")):
        return fh.handler(event, context)                    # shadow logging is off: exactly the frozen behaviour
    cap, saved = fp.Capture(), []
    try:
        _install(cap, event, context, saved)
    except Exception:                                        # could not set up the recorder: run without it
        log.exception("forward recorder not installed; running the frozen handler alone")
        _uninstall(saved)
        return fh.handler(event, context)
    try:
        return fh.handler(event, context)
    except Exception as exc:
        _uninstall(saved)
        _record_failure(exc, event, context)                 # so a missing night is explained, then re-raise unchanged
        raise
    finally:
        _uninstall(saved)
