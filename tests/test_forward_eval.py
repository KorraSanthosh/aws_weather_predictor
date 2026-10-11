"""Tests of the frozen forward-test scoring (model/forward_eval.py) on synthetic shadow logs."""
import copy
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import forward_eval as fe  # noqa: E402
from forward_protocol import PROTOCOL as P  # noqa: E402


def make_rows(n=124, start=date(2026, 10, 15), sd_v1=25.0, sd_v2=10.0, sd_v2_nodp=None, nodp_every=0, seed=1):
    """Shadow items for n consecutive target days. Every `nodp_every`-th day has no day-parts (hybrid -> v1 there)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = start + timedelta(days=i)
        actual = float(rng.uniform(80, 320))
        dp = not (nodp_every and i % nodp_every == 0)
        p1 = actual + rng.normal(0, sd_v1)
        p2 = actual + rng.normal(0, sd_v2 if dp or sd_v2_nodp is None else sd_v2_nodp)
        preds = {"v1_25_baseline": p1, "v2_35_candidate": p2, "v3_hybrid": p2 if dp else p1}
        for v, p in preds.items():
            rows.append({"pk": "shadow", "sk": f"{d.isoformat()}#{v}", "model_version": v, "pm25": round(float(p), 2),
                         "actual_pm25": round(actual, 2), "today_pm25": round(actual * 1.05, 2), "daypart_valid": dp,
                         "inputs_sha256": f"h{i}", "served_model": ("v2_35_candidate" if (v == "v3_hybrid" and dp) else
                                                                    "v1_25_baseline" if v == "v3_hybrid" else v)})
    return rows


def test_frozen_constants_match_the_agreed_rules():
    assert P["baseline"] == "v1_25_baseline" and P["candidates"] == ["v2_35_candidate", "v3_hybrid"]
    assert P["window_target_dates"] == ["2026-10-15", "2027-02-15"]
    assert P["min_matched_days"] == 90 and P["mae_reduction_required_pct"] == 5.0
    assert P["primary_metric"].startswith("MAE") and "category" in P["secondary_metric"].lower()
    assert P["ci_level"] == 0.95 and P["gate_segment"] == "complete_eligible"


def test_clear_improvement_is_promoted():
    rep = fe.build_report(make_rows())
    d = rep["decision"]
    assert d["n_matched_days"] == 124 and d["decision"] == "PROMOTE_V2_35_CANDIDATE"      # no gap days -> hybrid == v2
    v = rep["segments"]["complete_eligible"]["vs_baseline"]["v2_35_candidate"]
    assert v["mae_change_pct"] < -5 and v["mae_change_ci95"][1] < 0


def test_no_difference_keeps_v1():
    rep = fe.build_report(make_rows(sd_v2=25.0, seed=3))
    assert rep["decision"]["decision"] == "KEEP_V1"


def test_fewer_than_90_matched_days_is_inconclusive_even_if_v2_looks_great():
    rep = fe.build_report(make_rows(n=60))
    assert rep["decision"]["decision"] == "INCONCLUSIVE_KEEP_V1"


def test_needs_90_days_with_daypart_inputs_not_just_90_days():
    rep = fe.build_report(make_rows(n=124, nodp_every=2))          # 62 day-part days, 62 without
    assert rep["decision"]["n_matched_days"] == 124 and rep["decision"]["n_matched_daypart_days"] == 62
    assert rep["decision"]["decision"] == "INCONCLUSIVE_KEEP_V1"


def test_hybrid_wins_when_v2_is_bad_on_days_without_dayparts():
    """v2 is much worse without day-parts, so v2 alone fails the gate but the hybrid (v1 on those days) passes."""
    rows = make_rows(n=124, sd_v1=25, sd_v2=8, sd_v2_nodp=100, nodp_every=4, seed=5)
    rep = fe.build_report(rows)
    c = rep["decision"]["candidates"]
    assert not c["v2_35_candidate"]["passes"] and c["v3_hybrid"]["passes"]
    assert rep["decision"]["decision"] == "PROMOTE_V3_HYBRID"


def test_tie_rule_prefers_hybrid_when_gap_days_exist():
    rep = fe.build_report(make_rows(n=124, sd_v1=30, sd_v2=8, nodp_every=10, seed=7))   # v2 fine without day-parts too
    c = rep["decision"]["candidates"]
    assert c["v2_35_candidate"]["passes"] and c["v3_hybrid"]["passes"]
    assert rep["decision"]["decision"] == "PROMOTE_V3_HYBRID"


def test_category_accuracy_guard_blocks_promotion():
    base = {"segments": {"complete_eligible": {"n": 100, "vs_baseline": {
        "v2_35_candidate": {"mae_change_pct": -12.0, "mae_change_ci95": [-18.0, -6.0],
                            "category_accuracy_change_pp": -3.5, "category_accuracy_change_ci95": [-8, 1]},
        "v3_hybrid": {"mae_change_pct": -4.0, "mae_change_ci95": [-9.0, 1.0],
                      "category_accuracy_change_pp": 0.5, "category_accuracy_change_ci95": [-3, 4]}}},
            "daypart_days": {"n": 100}, "no_daypart_days": {"n": 0}}}
    d = fe.decide(base)
    assert d["decision"] == "KEEP_V1"
    assert d["candidates"]["v2_35_candidate"]["category_accuracy_not_materially_worse"] is False
    assert d["candidates"]["v3_hybrid"]["mae_at_least_required_lower"] is False


def test_improvement_must_be_statistically_distinguishable():
    seg = {"n": 100, "vs_baseline": {"v2_35_candidate": {"mae_change_pct": -8.0, "mae_change_ci95": [-17.0, 2.0],
                                                         "category_accuracy_change_pp": 0.0, "category_accuracy_change_ci95": [-2, 2]},
                                     "v3_hybrid": {"mae_change_pct": -8.0, "mae_change_ci95": [-17.0, 2.0],
                                                   "category_accuracy_change_pp": 0.0, "category_accuracy_change_ci95": [-2, 2]}}}
    d = fe.decide({"segments": {"complete_eligible": seg, "daypart_days": {"n": 100}, "no_daypart_days": {"n": 0}}})
    assert d["decision"] == "KEEP_V1"


def test_unmatched_days_are_excluded_and_counted_never_scored():
    rows = make_rows(n=100)
    drop = {"2026-11-03#v3_hybrid"}                                      # a version missing
    rows = [r for r in rows if r["sk"] not in drop]
    for r in rows:
        if r["sk"] == "2026-11-04#v2_35_candidate":
            r["error"] = "boom"; r.pop("pm25")                           # a version errored
        if r["sk"].startswith("2026-11-05#"):
            r["actual_pm25"] = None                                      # actual unknown
        if r["sk"] == "2026-11-06#v2_35_candidate":
            r["inputs_sha256"] = "different"                             # not the same inputs
    days, ex = fe.match_days(rows)
    assert len(days) == 96
    assert (ex["a_version_missing"], ex["a_version_errored"], ex["no_actual"], ex["inputs_differ_between_versions"]) == (1, 1, 1, 1)


def test_forecasts_outside_the_window_are_ignored():
    rows = make_rows(n=10, start=date(2026, 10, 1)) + make_rows(n=10, start=date(2027, 2, 10)) + make_rows(n=5, start=date(2027, 3, 1))
    days, _ = fe.match_days(rows)
    assert all(date(2026, 10, 15) <= d["date"] <= date(2027, 2, 15) for d in days)
    assert len(days) == 6 and days[0]["date"] == date(2027, 2, 10)      # only 10-15 Feb 2027 is inside the window


def test_scoring_is_the_same_for_every_model_on_the_same_days():
    rep = fe.build_report(make_rows())
    seg = rep["segments"]["complete_eligible"]
    assert {m["n"] for m in seg["models"].values()} == {124}
    assert "persistence_tomorrow_equals_today" in seg["models"]
    assert set(rep["segments"]) == {"complete_eligible", "daypart_days", "no_daypart_days"}


def test_early_analysis_is_refused_or_stamped_invalid(tmp_path, capsys):
    f = tmp_path / "rows.json"
    import json
    f.write_text(json.dumps(make_rows()))
    with pytest.raises(SystemExit):
        fe.main(["--input", str(f), "--today", "2026-12-20"])
    rep = fe.main(["--input", str(f), "--today", "2026-12-20", "--force-early"])
    assert rep["VALID"] is False and rep["decision"]["decision"] == "NOT_VALID_EARLY_LOOK"
    ok = fe.main(["--input", str(f), "--today", "2027-02-16", "--out", str(tmp_path / "o.json")])
    assert "VALID" not in ok and (tmp_path / "o.json").exists()


def test_analysis_does_not_change_its_input_and_is_deterministic():
    rows = make_rows()
    before = copy.deepcopy(rows)
    a, b = fe.build_report(rows), fe.build_report(rows)
    assert rows == before and a == b


def test_export_shadow_pages_through_all_items():
    pages = [{"Items": [{"sk": "a"}], "LastEvaluatedKey": {"k": 1}}, {"Items": [{"sk": "b"}]}]

    class T:
        def __init__(self):
            self.calls = []

        def query(self, **kw):
            self.calls.append(kw)
            return pages[len(self.calls) - 1]

    t = T()
    assert [i["sk"] for i in fe.export_shadow(t)] == ["a", "b"] and "ExclusiveStartKey" in t.calls[1]
