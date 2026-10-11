"""Known-answer guard: the stored models must give the numbers they were frozen with.

Why this exists: the model files were written by xgboost 3.4.1.  xgboost < 3.1 reads their base_score as 0.5 instead of
0.0039, which silently multiplies every prediction by about 1.65 (V1 134.98 instead of 81.8 on the fixed fake inputs).
Nothing else in the suite checks an exact number, so an old xgboost in a local venv or in the Lambda package passed
every test.  These tests fail loudly instead.

The inputs are the fixed fake data in tests/test_backend.py (no network, no randomness).
"""
import json
import re
import sys
from pathlib import Path

import pytest
import xgboost

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))

import forecast_handler as fh  # noqa: E402
import test_backend as t  # noqa: E402

# Values produced by xgboost 3.4.1 / 3.2.0 / 3.1.3 (identical to the 0.1 shown).  Tolerance 1.0 is far smaller than the
# 53-point error an old xgboost produces, and far larger than any float noise between >=3.1 releases.
KNOWN = {"v1_25_baseline": 81.8, "v2_35_candidate": 83.9, "v3_hybrid": 83.9}
TOLERANCE = 1.0


def _version_tuple(text):
    return tuple(int(p) for p in re.findall(r"\d+", text)[:2])


def test_xgboost_is_new_enough_for_the_stored_models():
    assert _version_tuple(xgboost.__version__) >= (3, 1), (
        f"xgboost {xgboost.__version__} misreads the stored models (base_score) and makes forecasts ~1.65x too high. "
        "Use xgboost >= 3.1 (needs Python >= 3.10).")


@pytest.mark.parametrize("version", ["v1_25_baseline", "v2_35_candidate"])
def test_stored_models_carry_the_tiny_base_score_that_old_xgboost_misreads(version):
    """Documents the cause: if a model is ever re-saved so that this is no longer the case, update KNOWN too."""
    raw = json.loads((ROOT / "model" / version / "model.json").read_text())
    base = str(raw["learner"]["learner_model_param"]["base_score"])
    assert base.startswith("["), f"{version}: base_score is {base!r}; the version note in this file is out of date"


def test_fixed_inputs_give_the_known_forecasts(monkeypatch):
    monkeypatch.setattr(fh.openaq, "fetch_sensor_hours", t.fake_hours)
    monkeypatch.setattr(fh.wx, "fetch_forecast_weather", t.fake_weather)
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    res = fh.run_forecast(t.NOW, "k", [1, 2], 28.6, 77.2, shadow_versions=list(KNOWN))
    got = {v: r["pm25"] for v, r in res["shadow"].items()}
    for version, expected in KNOWN.items():
        assert got[version] == pytest.approx(expected, abs=TOLERANCE), (
            f"{version} gave {got[version]}, expected {expected}.  xgboost {xgboost.__version__} may be too old (<3.1).")
    assert res["pm25"] == pytest.approx(KNOWN["v1_25_baseline"], abs=TOLERANCE)   # V1 is what is served
    assert res["model_version"] == "v1_25_baseline"
