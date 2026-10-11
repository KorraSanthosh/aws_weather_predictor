"""Guards for the calibrated 90 % prediction range (core/prediction_range.py, core/range_calibration.json).

The range is derived data: it must never change the point forecast, the AQI category, the school decision or the alert, must
never be invented when it cannot be calibrated, and the shipped calibration must be exactly what its audit-trail inputs give.
"""
import hashlib
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from core import prediction_range as pr  # noqa: E402
from core.aqi import CATEGORIES, pm25_to_category  # noqa: E402

CAL = json.loads(Path(pr.CALIBRATION_FILE).read_text())
VERSIONS = ("v1_25_baseline", "v2_35_candidate", "v3_hybrid")
RESIDUALS = ROOT / "experiments" / "prediction_range" / "calibration_residuals.json"
GRID = [0, 0.4, 5, 12.5, 30, 30.5, 45, 60, 60.9, 75, 90, 90.04, 120, 150.5, 220, 300, 380, 420]


# ------------------------------------------------------------------ the shipped calibration file
def test_calibration_file_is_the_declared_method_and_level():
    assert CAL["schema"] == pr.SCHEMA and CAL["method"] == pr.METHOD == "M1_split_conformal_log1p"
    assert CAL["level"] == pr.LEVEL == 0.90
    assert set(CAL["versions"]) == set(VERSIONS)
    assert "RETROSPECTIVE" in CAL["label"]
    assert pr.load_calibration()["calibration_id"] == CAL["calibration_id"]


def test_calibration_uses_only_out_of_sample_days_before_the_forward_test():
    """Every calibration window was forecast by models trained BEFORE it, and nothing reaches the 2026-10-15 forward test."""
    prev_end = ""
    for w in CAL["windows"]:
        assert w["models_trained_through"] < w["forecast_days"][0]          # the model never saw these days
        assert w["forecast_days"][0] > prev_end                              # windows are disjoint and in order
        prev_end = w["forecast_days"][1]
    for v, e in CAL["versions"].items():
        assert e["first"] >= "2024-01-19", v                                 # first day with an archived 24 h forecast
        assert e["last"] < "2026-10-15", v
        assert e["n"] == sum(w["n_days"] for w in CAL["windows"]) >= 600, v
        assert e["lo"] < 0 < e["hi"], v
        assert abs(e["in_sample_coverage"] - pr.LEVEL) < 0.02, v


@pytest.mark.skipif(not RESIDUALS.exists(), reason="residual audit trail not present")
def test_shipped_offsets_are_exactly_what_the_stored_residual_inputs_give():
    R = json.loads(RESIDUALS.read_text())
    assert R["dates"] == sorted(R["dates"]) and len(set(R["dates"])) == len(R["dates"])
    for v in VERSIONS:
        fc = R["forecast_pm25"][v]
        assert len(fc) == len(R["actual_pm25"]) == len(R["dates"]) == CAL["versions"][v]["n"]
        digest = hashlib.sha256(json.dumps([R["dates"], R["actual_pm25"], fc]).encode()).hexdigest()
        assert digest == CAL["versions"][v]["inputs_sha256"], f"{v}: residual inputs changed since the calibration was built"
        lo, hi = pr.conformal_offsets(pr.residuals(R["actual_pm25"], fc))
        assert lo == pytest.approx(CAL["versions"][v]["lo"], abs=1e-12) and hi == pytest.approx(CAL["versions"][v]["hi"], abs=1e-12)


@pytest.mark.skipif(not RESIDUALS.exists(), reason="residual audit trail not present")
def test_range_as_shown_covers_at_least_the_nominal_share_of_calibration_days():
    """The shown range is rounded OUTWARD, so on the calibration days it can only cover more than the computed 90 %."""
    R = json.loads(RESIDUALS.read_text())
    for v in VERSIONS:
        hits = 0
        for y, f in zip(R["actual_pm25"], R["forecast_pm25"][v]):
            r = pr.prediction_range(f, v)
            hits += r["pm25_low"] <= y <= r["pm25_high"]
        assert hits / len(R["dates"]) >= 0.90, v


# ------------------------------------------------------------------ the maths
def test_quantile_and_conformal_offsets_match_numpy():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    for n in (2, 3, 17, 250, 721):
        x = rng.normal(0, 0.4, n)
        for q in (0.0, 0.05, 0.5, 0.95, 1.0):
            assert pr.quantile(x, q) == pytest.approx(float(np.quantile(x, q)), abs=1e-12)
        alpha = 1 - pr.LEVEL
        lo, hi = pr.conformal_offsets(list(x))
        assert lo == pytest.approx(float(np.quantile(x, alpha / 2)), abs=1e-12)
        assert hi == pytest.approx(float(np.quantile(x, min(1.0, (1 - alpha / 2) * (1 + 1 / n)))), abs=1e-12)


def test_residual_is_log1p_difference_and_clips_negatives_like_training():
    assert pr.residuals([100.0], [50.0])[0] == pytest.approx(math.log1p(100) - math.log1p(50))
    assert pr.residuals([-5.0], [0.0])[0] == 0.0


@pytest.mark.parametrize("v", VERSIONS)
def test_interval_contains_point_is_non_negative_and_monotone(v):
    e = CAL["versions"][v]
    prev = (-1.0, -1.0)
    for p in [0, 0.4, 5, 12.5, 30, 45, 60, 75, 90, 120, 150.5, 220, 300, 380, 500]:
        lo, hi = pr.interval(p, e["lo"], e["hi"])
        assert 0 <= lo <= p <= hi
        assert lo >= prev[0] and hi >= prev[1]           # a higher forecast never gives a lower range
        prev = (lo, hi)


# ------------------------------------------------------------------ what the product returns
@pytest.mark.parametrize("v", VERSIONS)
def test_prediction_range_is_consistent_with_the_forecast_and_the_cpcb_categories(v):
    for p in GRID:
        r = pr.prediction_range(p, v)
        assert r["level"] == 0.90 and r["method"] == pr.METHOD and r["basis"] == "retrospective"
        assert isinstance(r["pm25_low"], int) and isinstance(r["pm25_high"], int)
        assert 0 <= r["pm25_low"] <= p <= r["pm25_high"]                      # outward whole numbers: the point is always inside
        assert r["category_low"] == pm25_to_category(r["pm25_low"]) and r["category_high"] == pm25_to_category(r["pm25_high"])
        i, j = CATEGORIES.index(r["category_low"]), CATEGORIES.index(r["category_high"])
        assert r["categories"] == CATEGORIES[i:j + 1]
        assert CATEGORIES.index(pm25_to_category(p)) in range(i, j + 1)      # the forecast's own category is inside its range
        assert r["calibration_id"] == CAL["calibration_id"] and r["calibration_days"] == CAL["versions"][v]["n"]


def test_range_is_never_invented():
    assert pr.prediction_range(56.5, "v9_unknown") is None
    assert pr.prediction_range(56.5, None) is None
    assert pr.prediction_range(56.5, "") is None
    assert pr.prediction_range(56.5, "v1_25_baseline", calibration={**CAL, "versions": {}}) is None
    for bad in (None, "abc", float("nan"), float("inf"), -1.0):
        assert pr.prediction_range(bad, "v1_25_baseline") is None
    assert pr.prediction_range(Decimal("56.5"), "v1_25_baseline") == pr.prediction_range(56.5, "v1_25_baseline")


def test_attach_adds_one_key_and_changes_nothing_else():
    rec = {"pm25": Decimal("56.5"), "category": "Satisfactory", "aqi": 94, "pm25_low": Decimal("42.6"), "pm25_high": Decimal("74.9"),
           "guidance": {"decision": "GO"}, "model_version": "v1_25_baseline"}
    before = json.dumps(rec, default=str, sort_keys=True)
    out = pr.attach_prediction_range(rec)
    assert json.dumps(rec, default=str, sort_keys=True) == before                      # input not mutated
    assert {k: v for k, v in out.items() if k != "prediction_range"} == rec           # point forecast, category, decision, old band: identical
    assert out["prediction_range"]["pm25_low"] <= 56 and out["prediction_range"]["pm25_high"] >= 57


def test_attach_omits_the_key_when_no_range_can_be_given_and_never_raises(monkeypatch):
    assert "prediction_range" not in pr.attach_prediction_range({"pm25": 50.0})                       # no model_version (old rows)
    assert "prediction_range" not in pr.attach_prediction_range({"pm25": 50.0, "model_version": "unversioned"})
    monkeypatch.setattr(pr, "prediction_range", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert pr.attach_prediction_range({"pm25": 50.0, "model_version": "v1_25_baseline"}) == {"pm25": 50.0, "model_version": "v1_25_baseline"}


def test_a_broken_calibration_file_is_rejected(tmp_path):
    bad = tmp_path / "c.json"
    for body in ({**CAL, "level": 0.8}, {**CAL, "method": "other"}, {**CAL, "schema": 99}, {"x": 1}):
        bad.write_text(json.dumps(body))
        pr._CAL.pop(str(bad), None)
        with pytest.raises(ValueError):
            pr.load_calibration(str(bad))


# ------------------------------------------------------------------ API: additive only
def _fake_boto(monkeypatch, items):
    class FakeTable:
        def query(self, **kw):
            return {"Items": items}

    class FakeResource:
        def Table(self, name):
            return FakeTable()

    import boto3
    monkeypatch.setattr(boto3, "resource", lambda *a, **k: FakeResource())
    monkeypatch.setenv("TABLE_NAME", "t")


ITEMS = [{"pk": "delhi", "sk": "2026-10-09", "pm25": Decimal("56.5"), "pm25_low": Decimal("42.6"), "pm25_high": Decimal("74.9"),
          "category": "Satisfactory", "model_version": "v1_25_baseline", "guidance": {"decision": "GO"}},
         {"pk": "delhi", "sk": "2026-10-08", "pm25": Decimal("140"), "actual_pm25": Decimal("138")}]


def test_api_adds_the_range_and_keeps_everything_else(monkeypatch):
    import api_handler
    _fake_boto(monkeypatch, ITEMS)
    body = json.loads(api_handler.handler({}, None)["body"])
    f = body["forecast"]
    assert f["prediction_range"] == pr.prediction_range(56.5, "v1_25_baseline")
    assert (f["pm25"], f["pm25_low"], f["pm25_high"], f["category"], f["guidance"]) == (56.5, 42.6, 74.9, "Satisfactory", {"decision": "GO"})
    assert body["history"] == [{"date": "2026-10-08", "predicted": 140.0, "actual": 138.0}]      # history untouched


def test_api_still_works_when_the_range_module_fails(monkeypatch):
    import api_handler
    _fake_boto(monkeypatch, ITEMS)
    monkeypatch.setattr(pr, "attach_prediction_range", lambda f: (_ for _ in ()).throw(RuntimeError("calibration file missing")))
    resp = api_handler.handler({}, None)
    f = json.loads(resp["body"])["forecast"]
    assert resp["statusCode"] == 200 and f["pm25"] == 56.5 and "prediction_range" not in f


# ------------------------------------------------------------------ the frozen forward-test pieces are not touched
def test_nightly_forecast_and_shadow_log_do_not_carry_the_range():
    """The range is computed when the forecast is READ (api_handler), so the frozen nightly Lambda, its stored rows and the
    shadow log used by the forward test are unchanged."""
    text = (ROOT / "backend" / "forecast_handler.py").read_text()
    assert "prediction_range" not in text and "range_calibration" not in text


def test_calibration_ships_inside_core_which_the_lambda_build_copies_whole():
    assert Path(pr.CALIBRATION_FILE).parent.name == "core" and Path(pr.CALIBRATION_FILE).exists()
    # the build script copies the whole core/ folder (the output folder is $OUT, default build/lambda, since BUILD_DIR was added)
    assert 'cp -r core "$OUT/core"' in (ROOT / "scripts" / "build_lambda.sh").read_text()


# ------------------------------------------------------------------ static dashboard data
def test_static_range_file_matches_the_static_forecast_if_present():
    """web/data/range.json (and its frontend copy) must have been made for exactly the forecast.json next to it."""
    f = ROOT / "web" / "data" / "forecast.json"
    rj = ROOT / "web" / "data" / "range.json"
    if not rj.exists():
        pytest.skip("no static range file")
    fc, rg = json.loads(f.read_text()), json.loads(rj.read_text())
    assert (rg["target_date"], rg["pm25"]) == (fc["target_date"], fc["pm25"]), "re-run: python scripts/make_range_json.py"
    assert rg["prediction_range"] == pr.prediction_range(fc["pm25"], rg["model_version"])
    twin = ROOT / "frontend" / "public" / "data" / "range.json"
    if twin.exists():
        assert twin.read_text() == rj.read_text()


def test_make_range_json_builds_from_a_forecast_without_touching_it():
    import make_range_json as m
    fc = {"target_date": "2026-10-08", "pm25": 56.5}
    snapshot = dict(fc)
    out = m.build(fc, "v1_25_baseline")
    assert fc == snapshot and out["target_date"] == "2026-10-08" and out["pm25"] == 56.5
    assert out["prediction_range"] == pr.prediction_range(56.5, "v1_25_baseline")
    with pytest.raises(SystemExit):
        m.build(fc, "v9_unknown")
