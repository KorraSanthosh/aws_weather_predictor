"""Tests for the shadow-evaluation recorder: backend/forward_logger.py, core/forward_provenance.py,
scripts/forward_ops.py and scripts/check_hourly_feed.py. Every network and AWS call is faked.

The things that must stay true (see FORWARD_TEST_OPERATIONS.md):
  * V1 keeps serving; the recorder cannot change a primary forecast, the alert or the dashboard row
  * the Hybrid and V2 only watch (pk="shadow"), first write wins
  * every night leaves one provenance record that is enough to re-create the numbers
  * future / unfinished hours are never used and a missing hour is never invented
  * nothing secret is ever written
"""
import json
import math
import subprocess
import sys
import textwrap
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

pytestmark = pytest.mark.skipif(
    not all((ROOT / "model" / p / "model.json").exists() for p in ("", "v1_25_baseline", "v2_35_candidate")),
    reason="needs the stored v1/v2 model files")

import check_hourly_feed as chf                                      # noqa: E402
import forecast_handler as fh                                        # noqa: E402
import forward_logger                                                # noqa: E402
import forward_ops as ops                                            # noqa: E402
import test_backend as tb                                            # noqa: E402
from core import forward_provenance as fp                            # noqa: E402
from core import openaq                                              # noqa: E402
from core import weather as wx                                       # noqa: E402
from core.daily import IST                                           # noqa: E402
from core.features import ALL_FEATURES                               # noqa: E402

HOUR = timedelta(hours=1)
ALL3 = "v1_25_baseline,v2_35_candidate,v3_hybrid"
TARGET = "2026-10-09"                          # tb.NOW is 23:00 IST on 8 Oct 2026
SECRET = "SECRET-KEY-0123456789abcdef"


class Ctx:
    aws_request_id = "req-123"
    function_version = "$LATEST"


def make_feed(drop=(), unfinished=False, boom=None):
    """Hour-aligned like the real API (an IST hour starts at :30Z). `drop` removes those IST hours from the run day,
    `unfinished` adds the hour in progress and a future hour with a wild value, `boom` makes every call fail."""
    def value(sid, t):
        return 70 + 6 * (t.date().toordinal() % 5) + sid + 2 * abs(12 - t.hour)

    def feed(sensor_id, dt_from, dt_to, api_key):
        if boom:
            raise RuntimeError(boom)
        run_day = dt_to.astimezone(IST).date()
        t, out = dt_from.astimezone(IST).replace(minute=0, second=0, microsecond=0), []
        while t + HOUR <= dt_to:
            if not (t.date() == run_day and t.hour in drop):
                out.append((t.astimezone(timezone.utc), value(sensor_id, t)))
            t += HOUR
        if unfinished:
            now_hour = dt_to.astimezone(IST).replace(minute=0, second=0, microsecond=0)
            out += [((now_hour + k * HOUR).astimezone(timezone.utc), 900.0) for k in (0, 1)]
        return out
    return feed


@pytest.fixture
def run(monkeypatch):
    """run(**feed_options) -> (table, handler output). Pass table=... to run again against the same table."""
    def _run(shadow=ALL3, table=None, wrapper=True, **feed_options):
        table = table if table is not None else tb._Table()
        monkeypatch.setattr(openaq, "fetch_sensor_hours", make_feed(**feed_options))
        monkeypatch.setattr(wx, "fetch_forecast_weather", tb.fake_weather)
        tb._handler_env(monkeypatch, table, shadow)
        handler = forward_logger.handler if wrapper else fh.handler
        return table, handler({"source": "aws.events", "detail-type": "Scheduled Event", "time": "2026-10-08T17:30:00Z"}, Ctx())
    return _run


def prov_of(table, sk=TARGET):
    return ops.plain(table.items[("provenance", sk)])


def shadow_keys(table):
    return sorted(k for k in table.items if k[0] == "shadow")


# ----------------------------------------------------------------------------- V1 keeps serving, nothing else moves
def test_primary_output_is_identical_with_and_without_the_recorder(run):
    plain, _ = run(wrapper=False)
    wrapped, out = run()
    assert out == {"ok": True, "target_date": TARGET, "category": plain.items[("delhi", TARGET)]["category"]}
    assert wrapped.items[("delhi", TARGET)] == plain.items[("delhi", TARGET)]          # every field, incl. pm25 and alerts
    primary = wrapped.items[("delhi", TARGET)]
    assert primary["model_version"] == primary["served_model"] == "v1_25_baseline" and primary["n_features"] == 25
    assert shadow_keys(wrapped) == shadow_keys(plain) and len(shadow_keys(wrapped)) == 3
    for k in shadow_keys(plain):
        assert wrapped.items[k] == plain.items[k]                                       # shadow rows untouched too
    assert {pk for pk, _ in wrapped.items} == {"delhi", "shadow", "provenance"}
    assert {pk for pk, _ in plain.items} == {"delhi", "shadow"}


def test_recorder_does_nothing_when_shadow_logging_is_off(run):
    table, out = run(shadow=None)
    assert out["ok"] and {pk for pk, _ in table.items} == {"delhi"}


def test_frozen_functions_are_restored_after_success_and_after_failure(monkeypatch):
    def snapshot():
        return (openaq.fetch_sensor_hours, wx.fetch_forecast_weather, fh.run_forecast, fh.shadow_forecasts, fh.store_shadow)

    monkeypatch.setattr(openaq, "fetch_sensor_hours", make_feed())
    monkeypatch.setattr(wx, "fetch_forecast_weather", tb.fake_weather)
    tb._handler_env(monkeypatch, tb._Table(), ALL3)
    before = snapshot()
    forward_logger.handler({}, Ctx())
    assert all(a is b for a, b in zip(snapshot(), before)), "a patched function was left behind after a normal run"

    monkeypatch.setattr(openaq, "fetch_sensor_hours", make_feed(boom="OpenAQ down"))
    before = snapshot()
    with pytest.raises(RuntimeError, match="OpenAQ down"):
        forward_logger.handler({}, Ctx())
    assert all(a is b for a, b in zip(snapshot(), before)), "a patched function was left behind after a failed run"


# ----------------------------------------------------------------------------- what a night records
def test_provenance_record_has_everything_asked_for(run):
    table, _ = run()
    rec = prov_of(table)
    assert rec["schema"] == fp.SCHEMA and rec["status"] == "logged"
    assert (rec["target_date"], rec["origin_date"]) == (TARGET, "2026-10-08")
    assert rec["forecast_generated_at_utc"].startswith("2026-10-08T17:30:00")
    assert rec["forecast_generated_at_ist"].startswith("2026-10-08T23:00:00") and rec["forecast_generated_at_ist"].endswith("+05:30")
    assert rec["in_protocol_window"] is False                   # 9 Oct is before the window opens on 15 Oct: a rehearsal night
    assert rec["request"]["aws_request_id"] == "req-123" and rec["request"]["event_time"] == "2026-10-08T17:30:00Z"

    prim = rec["primary"]                                       # which model served
    assert prim["model_version"] == prim["served_model"] == "v1_25_baseline" and prim["school_decision"] in ("GO", "CAUTION", "INDOOR")
    assert rec["environment"]["MODEL_VERSION"] is None
    assert set(rec["shadow"]) == {"v1_25_baseline", "v2_35_candidate", "v3_hybrid"}
    for v, r in rec["shadow"].items():                          # predicted PM2.5 and category per version
        assert r["pm25"] > 0 and r["category"] and "error" not in r, v
    assert abs(rec["shadow"]["v1_25_baseline"]["pm25"] - prim["pm25"]) <= 0.05          # shadow V1 == what was served
    assert rec["shadow"]["v3_hybrid"]["served_model"] == "v2_35_candidate" and rec["shadow"]["v3_hybrid"]["daypart_valid"] is True
    assert rec["all_versions_saw_identical_inputs"] is True

    feed = rec["hourly_feed"]                                   # hourly-feature availability
    assert feed["origin_day"]["city_hours_present"] == list(range(23))
    assert feed["origin_day"]["has_hour_21"] and feed["origin_day"]["has_hour_22"] and not feed["origin_day"]["has_hour_23"]
    assert feed["origin_day"]["stations_per_hour"]["22"] == 2 and feed["n_sensors_with_origin_day_hours"] == 2
    assert feed["daypart"]["valid"] is True and feed["daypart"]["fallback_reasons"] == []
    assert feed["daypart"]["recomputed_from_hourly_equals_history"] is True
    assert all(isinstance(feed["daypart"]["features"][f], float) for f in feed["daypart"]["features"])

    w = rec["weather"]                                          # weather forecast source
    assert w["url"] == wx.FORECAST_URL and w["timezone"] == "Asia/Kolkata" and w["hourly_variables"] == wx.HOURLY_VARS
    assert w["calls"][0]["forecast_days"] == 3 and TARGET in w["calls"][0]["days_returned"]
    assert set(w["target_day_weather"]) == set(wx.WEATHER_KEYS) and w["target_day_value_is"].startswith("a forecast")

    inp = rec["inputs"]                                         # everything needed to reproduce
    assert set(inp["feature_row"]) == set(ALL_FEATURES) and max(inp["history"]) == "2026-10-08"
    assert inp["inputs_sha256"] == rec["shadow"]["v1_25_baseline"]["inputs_sha256"]


def test_software_hashes_tie_the_night_to_the_frozen_manifest(run):
    table, _ = run()
    sw = prov_of(table)["software"]["files_sha256"]
    manifest = json.loads((ROOT / "model" / "forward_test_manifest.json").read_text())["files_sha256"]
    shared = set(sw) & set(manifest)
    assert {"backend/forecast_handler.py", "core/features.py", "core/hybrid.py", "model/v1_25_baseline/model.json",
            "model/v2_35_candidate/model.json"} <= shared
    assert {k: sw[k] for k in shared} == {k: manifest[k] for k in shared}
    assert all(sw[k] for k in sw if k != "model/model.json") and sw["model/model.json"] == sw["model/v1_25_baseline/model.json"]


def test_every_logged_prediction_can_be_reproduced_from_the_record(run):
    table, _ = run()
    rows = {sk.split("#")[1]: ops.plain(table.items[("shadow", sk)]) for _, sk in shadow_keys(table)}
    manifest = json.loads((ROOT / "model" / "forward_test_manifest.json").read_text())
    assert ops.reproduce_item(prov_of(table), rows, manifest) == []


def test_reproduction_catches_a_tampered_record(run):
    table, _ = run()
    rows = {sk.split("#")[1]: ops.plain(table.items[("shadow", sk)]) for _, sk in shadow_keys(table)}
    manifest = json.loads((ROOT / "model" / "forward_test_manifest.json").read_text())

    tampered = prov_of(table)
    tampered["inputs"]["history"]["2026-10-08"]["pm25"] += 10
    assert any("fingerprint" in p or "feature row" in p for p in ops.reproduce_item(tampered, rows, manifest))

    other_model = prov_of(table)
    other_model["software"]["files_sha256"]["model/v2_35_candidate/model.json"] = "0" * 64
    assert any("differ from the frozen manifest" in p for p in ops.reproduce_item(other_model, rows, manifest))

    changed_row = {**rows, "v1_25_baseline": {**rows["v1_25_baseline"], "pm25": rows["v1_25_baseline"]["pm25"] + 5}}
    assert any("re-run gives" in p for p in ops.reproduce_item(prov_of(table), changed_row, manifest))


# ----------------------------------------------------------------------------- the 23:00 IST cutoff
def test_unfinished_and_future_hours_never_reach_the_models(run, monkeypatch):
    clean_table, _ = run()
    dirty_table, _ = run(unfinished=True)
    assert dirty_table.items[("delhi", TARGET)] == clean_table.items[("delhi", TARGET)]
    assert [dirty_table.items[k] for k in shadow_keys(dirty_table)] == [clean_table.items[k] for k in shadow_keys(clean_table)]
    cut = prov_of(dirty_table)["hourly_feed"]["cutoff"]
    assert cut["readings_dropped_as_unfinished"] == 4                       # 2 sensors x (the hour in progress + a future hour)
    assert cut["readings_used_from_unfinished_hours"] == 0 and cut["readings_used_started_after_run"] == 0
    assert prov_of(clean_table)["hourly_feed"]["cutoff"]["readings_dropped_as_unfinished"] == 0
    # the guard has teeth: the frozen code alone WOULD have used the 900 ug/m3 hour in progress
    monkeypatch.setattr(openaq, "fetch_sensor_hours", make_feed(unfinished=True))
    monkeypatch.setattr(wx, "fetch_forecast_weather", tb.fake_weather)
    unguarded = fh.run_forecast(tb.NOW, "k", [1, 2], 28.6, 77.2)
    assert unguarded["pm25"] != clean_table.items[("delhi", TARGET)]["pm25"]


def test_readings_are_used_only_if_their_hour_has_ended():
    run_at = datetime(2026, 10, 8, 17, 30, tzinfo=timezone.utc)
    ends_exactly = (run_at - HOUR, 50.0)
    ends_a_second_later = (run_at - HOUR + timedelta(seconds=1), 60.0)
    naive_utc = (datetime(2026, 10, 8, 16, 30), 70.0)                   # naive = UTC, same convention as core.daily
    done, not_done = fp.split_complete([ends_exactly, ends_a_second_later, naive_utc], run_at)
    assert done == [ends_exactly, naive_utc] and not_done == [ends_a_second_later]


def test_a_run_after_the_target_day_began_is_not_logged_as_a_forecast(run):
    # the origin day has only 10 hours of data, so the handler falls back to yesterday and "forecasts" today at 23:00
    table, out = run(drop=tuple(range(10, 23)))
    assert out["target_date"] == "2026-10-08"
    assert ("delhi", "2026-10-08") in table.items                          # the frozen primary behaviour is unchanged
    assert shadow_keys(table) == []                                        # but no shadow row may count for the test
    late = [k for k in table.items if k[0] == "provenance"]
    assert len(late) == 1 and "#late#" in late[0][1]
    rec = ops.plain(table.items[late[0]])
    assert rec["status"] == "late_run_not_logged" and sorted(rec["shadow_rows_dropped_as_late"]) == sorted(ALL3.split(","))
    assert rec["hourly_feed"]["cutoff"]["run_before_target_day_starts"] is False


def test_late_row_rule_edges():
    gen = lambda s: {"generated_at": s, "target_date": "2026-10-09"}    # noqa: E731
    rows = {"before": gen("2026-10-08T18:29:59+00:00"),                 # 23:59:59 IST on 8 Oct
            "exactly": gen("2026-10-08T18:30:00+00:00"),                # 00:00:00 IST on 9 Oct
            "after": gen("2026-10-09T02:00:00+00:00"),
            "garbled": {"generated_at": "not a time", "target_date": "2026-10-09"}}
    keep, dropped = fp.split_late_rows(rows)
    assert sorted(keep) == ["before", "garbled"] and sorted(dropped) == ["after", "exactly"]


# ----------------------------------------------------------------------------- missing hourly data: say so, fall back, invent nothing
def test_missing_late_hours_make_the_hybrid_fall_back_to_v1_and_say_why(run):
    table, _ = run(drop=(21, 22))                                       # the feed has not delivered the last two hours yet
    rec = prov_of(table)
    feed = rec["hourly_feed"]
    assert feed["origin_day"]["city_hours_present"] == list(range(21))
    assert not feed["origin_day"]["has_hour_21"] and not feed["origin_day"]["has_hour_22"] and feed["origin_day"]["last_hour_present"] == 20
    assert feed["daypart"]["valid"] is False
    reasons = " | ".join(feed["daypart"]["fallback_reasons"])
    assert "day-part late: 1 of 4" in reasons and "day-part last3: 0 of 3" in reasons
    row = rec["inputs"]["feature_row"]
    assert row["pm_late"] is None and row["pm_last3h_ratio"] is None and row["pm_buildup"] is None   # never filled in
    assert row["pm_night"] is not None and row["pm_school"] is not None
    hyb, v1, v2 = (rec["shadow"][v] for v in ("v3_hybrid", "v1_25_baseline", "v2_35_candidate"))
    assert hyb["served_model"] == "v1_25_baseline" and hyb["daypart_valid"] is False and hyb["pm25"] == v1["pm25"]
    assert {"pm_late", "pm_buildup", "pm_last3h_ratio"} <= set(v2["missing_features"])
    assert rec["primary"]["served_model"] == "v1_25_baseline"


def test_a_day_with_too_few_hours_is_explained(run):
    row = {f: float("nan") for f in ("pm_night", "pm_commute", "pm_school", "pm_evening", "pm_late", "pm_buildup", "pm_last3h_ratio")}
    assert "origin day has 10 hourly city values" in fp.fallback_reasons(row, {h: 1.0 for h in range(10)})[0]


# ----------------------------------------------------------------------------- failure handling and storage safety
def test_recorder_failure_never_stops_the_forecast(run, monkeypatch):
    def broken(*a, **k):
        raise ValueError("recorder bug")
    monkeypatch.setattr(fp, "build_provenance", broken)
    table, out = run()
    assert out["ok"] and ("delhi", TARGET) in table.items and len(shadow_keys(table)) == 3
    assert [k for k in table.items if k[0] == "provenance"] == []


def test_a_dynamodb_error_on_the_provenance_item_does_not_stop_the_forecast(run):
    class Flaky(tb._Table):
        def put_item(self, Item, ConditionExpression=None):
            if Item["pk"] == "provenance":
                raise RuntimeError("ProvisionedThroughputExceeded")
            return super().put_item(Item, ConditionExpression)
    table, out = run(table=Flaky())
    assert out["ok"] and len(shadow_keys(table)) == 3 and ("delhi", TARGET) in table.items


def test_provenance_is_first_write_wins(run):
    table, _ = run()
    first = dict(table.items[("provenance", TARGET)])
    run(table=table, unfinished=True)
    assert table.items[("provenance", TARGET)] == first


def test_a_failed_night_is_recorded_and_the_error_is_raised_unchanged(monkeypatch):
    table = tb._Table()
    monkeypatch.setattr(openaq, "fetch_sensor_hours", make_feed(boom=f"GET failed, key {SECRET} rejected"))
    tb._handler_env(monkeypatch, table, ALL3)
    monkeypatch.setenv("OPENAQ_API_KEY", SECRET)                       # after _handler_env, which sets a dummy key
    with pytest.raises(RuntimeError):
        forward_logger.handler({}, Ctx())
    failed = [ops.plain(v) for k, v in table.items.items() if k[0] == "provenance"]
    assert len(failed) == 1 and failed[0]["status"] == "failed" and "#failed#" in failed[0]["sk"]
    assert SECRET not in json.dumps(failed) and "***" in failed[0]["error"]
    assert not [k for k in table.items if k[0] in ("delhi", "shadow")]       # nothing half-written


def test_nothing_secret_is_written(monkeypatch):
    import boto3
    table = tb._Table()
    monkeypatch.setattr(openaq, "fetch_sensor_hours", make_feed())
    monkeypatch.setattr(wx, "fetch_forecast_weather", tb.fake_weather)
    tb._handler_env(monkeypatch, table, ALL3)
    for k, v in (("OPENAQ_API_KEY", SECRET), ("TABLE_NAME", "very-private-table-name"),
                 ("TOPIC_ARN", "arn:aws:sns:ap-south-1:123456789012:private-topic"),
                 ("AWS_SECRET_ACCESS_KEY", "aws-secret-xyz"), ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLEEXAMPLE1")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: type("C", (), {"publish": lambda self, **kw: None})())
    forward_logger.handler({}, Ctx())
    blob = json.dumps({str(k): v for k, v in table.items.items()}, default=str)
    for needle in (SECRET, "very-private-table-name", "123456789012", "private-topic", "aws-secret-xyz", "AKIA"):
        assert needle not in blob, f"{needle!r} was written to DynamoDB"


def test_stored_items_are_dynamodb_safe_and_small(run):
    table, _ = run()

    def walk(x):
        if isinstance(x, float):
            raise AssertionError(f"float {x!r} would be rejected by DynamoDB")
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for item in table.items.values():
        walk(item)
    size = len(json.dumps(ops.plain(table.items[("provenance", TARGET)])))
    assert 5_000 < size < 120_000, size                                 # DynamoDB's item limit is 400 KB


def test_jsonable_and_to_dynamo():
    nan = float("nan")
    out = fp.jsonable({date(2026, 10, 8): [nan, math.inf, 1.5, (1, 2), {3, 1}], "n": None, 5: True})
    assert out == {"2026-10-08": [None, None, 1.5, [1, 2], [1, 3]], "n": None, "5": True}
    import numpy as np
    assert fp.jsonable(np.float64(2.5)) == 2.5 and fp.jsonable(np.float64("nan")) is None
    item = fp.to_dynamo({"a": 1.25, "b": [nan]})
    from decimal import Decimal
    assert item == {"a": Decimal("1.25"), "b": [None]}


# ----------------------------------------------------------------------------- configuration
def test_frozen_constants_are_mirrored_correctly():
    sys.path.insert(0, str(ROOT / "model"))
    from forward_protocol import PROTOCOL
    assert fp.WINDOW_TARGET_DATES == tuple(PROTOCOL["window_target_dates"])
    assert fp.CUTOFF_HOUR_IST == 23
    assert "cron(30 17 * * ? *)" in (ROOT / "template.yaml").read_text()          # 17:30 UTC = 23:00 IST


def test_template_runs_the_wrapper_keeps_v1_and_keeps_shadow_opt_in():
    text = (ROOT / "template.yaml").read_text()
    assert "Handler: forward_logger.handler" in text and "Handler: api_handler.handler" in text
    assert "MODEL_VERSION:" not in text                                # no environment entry: V1 is the default model
    assert 'ShadowVersions:\n    Type: String\n    Default: ""' in text   # shadow logging is switched on only by a parameter
    assert "NoEcho: true" in text


def test_the_package_built_by_the_script_records_a_night_end_to_end(tmp_path):
    pkg = tmp_path / "pkg"
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "SHADOW": "1", "SKIP_DEPS": "1", "BUILD_DIR": str(pkg), "HOME": str(tmp_path)}
    done = subprocess.run(["bash", str(ROOT / "scripts" / "build_lambda.sh")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    info = json.loads((pkg / "build_info.json").read_text())
    assert info["shadow_package"] is True and info["default_model_source"] == "model"
    for rel in ("forecast_handler.py", "forward_logger.py", "api_handler.py", "core/forward_provenance.py",
                "model/model.json", "model/v1_25_baseline/model.json", "model/v2_35_candidate/model.json"):
        assert (pkg / rel).exists(), rel
    assert (pkg / "forecast_handler.py").read_bytes() == (ROOT / "backend" / "forecast_handler.py").read_bytes()

    script = tmp_path / "run_packaged.py"
    script.write_text(textwrap.dedent('''
        import json, sys
        from datetime import datetime, timedelta, timezone
        sys.path.insert(0, sys.argv[1])
        import forward_logger, forecast_handler as fh
        from core import openaq, weather as wx
        assert forward_logger.__file__.startswith(sys.argv[1]) and fh.__file__.startswith(sys.argv[1])
        IST = timezone(timedelta(hours=5, minutes=30)); H = timedelta(hours=1)
        NOW = datetime(2026, 10, 8, 17, 30, tzinfo=timezone.utc)
        def feed(sid, a, b, k):
            t, out = a.astimezone(IST).replace(minute=0, second=0, microsecond=0), []
            while t + H <= b:
                out.append((t.astimezone(timezone.utc), 70 + sid + 6 * (t.date().toordinal() % 5) + 2 * abs(12 - t.hour))); t += H
            return out
        def weather(lat, lon, past_days=10, forecast_days=3):
            d0 = datetime(2026, 10, 8).date()
            return {d0 + timedelta(days=i): {k: 10.0 + i for k in wx.WEATHER_KEYS} for i in range(-past_days, forecast_days)}
        openaq.fetch_sensor_hours, wx.fetch_forecast_weather = feed, weather
        real = fh.run_forecast
        fh.run_forecast = lambda now, *a, **k: real(NOW, *a, **k)
        items = {}
        class T:
            def put_item(self, Item, ConditionExpression=None): items[(Item["pk"], Item["sk"])] = Item
            def update_item(self, **kw): pass
        import boto3
        boto3.resource = lambda *a, **k: type("R", (), {"Table": lambda s, n: T()})()
        import os
        os.environ.update(SENSOR_IDS="1,2", OPENAQ_API_KEY="x", TABLE_NAME="t", SHADOW_VERSIONS="v1_25_baseline,v2_35_candidate,v3_hybrid")
        os.environ.pop("MODEL_VERSION", None); os.environ.pop("TOPIC_ARN", None)
        forward_logger.handler({}, None)
        prov = items[("provenance", "2026-10-09")]
        print(json.dumps({"keys": sorted("%s|%s" % k for k in items),
                          "served": str(items[("delhi", "2026-10-09")]["served_model"]),
                          "hybrid_served": str(prov["shadow"]["v3_hybrid"]["served_model"]),
                          "build_info_branch_recorded": prov["software"]["build_info"] is not None,
                          "handler_hash": prov["software"]["files_sha256"]["backend/forecast_handler.py"]}))
    '''))
    ran = subprocess.run([sys.executable, "-I", str(script), str(pkg)], capture_output=True, text=True, cwd=tmp_path)
    assert ran.returncode == 0, ran.stderr[-2000:]
    got = json.loads(ran.stdout.strip().splitlines()[-1])
    assert got["served"] == "v1_25_baseline" and got["hybrid_served"] == "v2_35_candidate"
    assert got["build_info_branch_recorded"] is True
    assert got["handler_hash"] == fp.file_sha256(ROOT / "backend" / "forecast_handler.py")
    assert sorted(got["keys"]) == sorted(["delhi|2026-10-09", "provenance|2026-10-09", "shadow|2026-10-09#v1_25_baseline",
                                          "shadow|2026-10-09#v2_35_candidate", "shadow|2026-10-09#v3_hybrid"])


# ----------------------------------------------------------------------------- operations report: counts, never accuracy
def clone_night(items, new_target, *, in_window=True):
    """Copy one logged night to another date (to test the report on a window-length log; reproduce is not used on it)."""
    out = []
    for it in items:
        it = json.loads(json.dumps(it))
        if it["pk"] == "delhi":
            it["sk"] = new_target
        elif it["pk"] == "shadow":
            it["sk"] = f"{new_target}#{it['sk'].split('#')[1]}"
            it["target_date"] = new_target
        elif it["pk"] == "provenance":
            it["sk"] = it["target_date"] = new_target
            it["in_protocol_window"] = in_window
        out.append(it)
    return out


def night_items(table):
    return [ops.plain(v) for v in table.items.values()]


def test_ops_report_counts_nights_and_fallbacks_but_computes_no_accuracy(run):
    ok_items = night_items(run()[0])
    lagged_items = night_items(run(drop=(21, 22))[0])
    log = clone_night(ok_items, "2026-10-15") + clone_night(lagged_items, "2026-10-16") + clone_night(ok_items, "2026-10-17")
    log = [i for i in log if not (i["pk"] == "shadow" and i["sk"].startswith("2026-10-17#v2"))]       # one row missing -> incomplete night
    for i in log:                                   # 15 Oct: all three rows have the actual; 16 Oct: only V1's row does
        if i["pk"] == "shadow" and (i["sk"].startswith("2026-10-15") or i["sk"] == "2026-10-16#v1_25_baseline"):
            i["actual_pm25"] = 123.0
    rep = ops.ops_report(log, date(2026, 10, 18))
    assert rep["logging_has_started"] and rep["expected_forecast_nights_so_far"] == 4
    assert rep["nights_with_all_three_shadow_rows"] == 2 and rep["missing_nights"] == ["2026-10-17", "2026-10-18"]
    assert rep["nights_with_provenance_but_no_complete_shadow_rows"] == ["2026-10-17"]
    s = rep["in_window"]
    assert s["nights"] == 3 and s["hybrid_had_all_7_daypart_inputs"] == 2 and s["hybrid_fell_back_to_v1"] == 1
    assert s["nights_with_hour_21"] == 2 and s["nights_with_hour_22"] == 2
    assert s["cutoff"]["readings_used_from_unfinished_hours"] == 0 and s["nights_with_different_input_fingerprints"] == 0
    assert rep["primary_rows_in_window_not_served_by_v1"] == []
    prog = rep["progress_toward_minimums"]                                         # counted, never scored
    assert prog["matched_days_with_actual"] == 1 and prog["matched_daypart_days"] == 1
    assert rep["accuracy"].startswith("NOT COMPUTED")
    text = ops.format_ops(rep)
    assert "mae" not in text.lower().replace("matched", "") and "accuracy" in text
    assert not any(k in json.dumps(rep).lower() for k in ('"mae"', '"rmse"', 'category_accuracy'))


def test_ops_report_flags_a_primary_row_that_was_not_served_by_v1(run):
    items = night_items(run()[0])
    log = clone_night(items, "2026-10-15")
    next(i for i in log if i["pk"] == "delhi")["served_model"] = "v2_35_candidate"
    assert ops.ops_report(log, date(2026, 10, 16))["primary_rows_in_window_not_served_by_v1"] == ["2026-10-15"]


def test_ops_before_anything_is_logged_says_so():
    rep = ops.ops_report([], date(2026, 10, 11))
    assert rep["logging_has_started"] is False and rep["expected_forecast_nights_so_far"] == 0
    assert "NO" in ops.format_ops(rep)


def test_ops_and_reproduce_command_line(run, tmp_path, capsys):
    table, _ = run()
    path = tmp_path / "items.json"
    path.write_text(json.dumps(night_items(table)))
    assert ops.main(["reproduce", "--input", str(path)]) == 0
    assert "1 of 1 night(s) reproduced" in capsys.readouterr().out
    assert ops.main(["ops", "--input", str(path), "--today", "2026-10-09"]) == 0
    assert "pre_window_rehearsal" in capsys.readouterr().out
    items = json.loads(path.read_text())
    next(i for i in items if i["pk"] == "provenance")["inputs"]["history"]["2026-10-08"]["pm25"] += 3
    path.write_text(json.dumps(items))
    assert ops.main(["reproduce", "--input", str(path)]) == 1


# ----------------------------------------------------------------------------- the live feed check (analysis part, offline)
def test_feed_check_needs_the_newest_hour_to_be_in_the_feed_at_2300(capsys):
    now = datetime(2026, 10, 14, 17, 30, 5, tzinfo=timezone.utc)
    by_lag = {lag: chf.analyse(chf.synthetic_feed(now, lag_hours=lag), now) for lag in (0, 1, 2)}
    assert by_lag[0]["hybrid_inputs_expected_at_2300_ist_with_this_lag"] is True
    assert by_lag[1]["hybrid_inputs_expected_at_2300_ist_with_this_lag"] is False       # one hour behind is already too late
    assert by_lag[1]["expected_newest_hour_at_2300_ist_with_this_lag"] == 21
    assert by_lag[2]["expected_newest_hour_at_2300_ist_with_this_lag"] == 20
    assert by_lag[0]["days_with_all_7_inputs_by_newest_hour"]["22"] == by_lag[0]["days_checked"]
    assert by_lag[0]["days_with_all_7_inputs_by_newest_hour"]["21"] == 0
    assert chf.selftest() == 0


def test_feed_check_counts_rows_from_unfinished_hours():
    now = datetime(2026, 10, 14, 17, 30, 5, tzinfo=timezone.utc)
    feed = chf.synthetic_feed(now, lag_hours=0)
    feed[1] = feed[1] + [(datetime(2026, 10, 14, 17, 30, tzinfo=timezone.utc), 5.0), (datetime(2026, 10, 14, 18, 30, tzinfo=timezone.utc), 5.0)]
    fut = chf.analyse(feed, now)["future_observations"]
    assert fut == {**fut, "rows_hour_not_finished": 2, "rows_started_after_now": 1}


def test_feed_check_refuses_to_run_without_a_key_and_never_prints_one(monkeypatch, capsys):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    assert chf.main([]) == 2
    err = capsys.readouterr().err
    assert "OPENAQ_API_KEY is not set" in err
