"""Guards for the regional-cities experiment (use the committed metrics.json / config.json; the raw station files are not needed
except where a test says it is skipped without them)."""
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_regional as rr  # noqa: E402

M = json.loads((HERE / "metrics.json").read_text())
CFG = json.loads((HERE / "config.json").read_text())
HARD_STOP = date(2026, 2, 15)


# ------------------------------------------------------------------ protocol
def test_protocol_was_not_changed_after_the_run():
    assert hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest() == M["config_sha256"]


def test_windows_and_cutoffs():
    assert rr.TESTS["T1_calendar_2025"] == {"train_end": date(2024, 12, 31), "start": date(2025, 1, 1), "end": date(2025, 12, 31)}
    assert rr.TESTS["T2_winter_2025_26"] == {"train_end": date(2025, 10, 14), "start": date(2025, 10, 15), "end": date(2026, 2, 15)}
    assert all(t["end"] <= HARD_STOP and t["train_end"] < t["start"] for t in rr.TESTS.values())


def test_hard_stop_is_respected_everywhere():
    for b in M["tests"].values():
        ds = [date.fromisoformat(d) for d in b["daily"]["dates"]]
        assert max(ds) <= HARD_STOP and ds == sorted(set(ds))
        assert date.fromisoformat(b["training"]["last_target"]) <= date.fromisoformat(b["train_end"]) < ds[0]
    with open(HERE / "regional_daily.csv") as f:
        days = [line.split(",")[0] for line in f.read().splitlines()[1:]]
    assert max(days) <= HARD_STOP.isoformat()
    assert max(days) < "2026-10-15"                                      # winter 2026-27 untouched


def test_every_series_is_scored_on_the_same_days():
    for b in M["tests"].values():
        n = b["n_days"]
        assert len(b["daily"]["actual"]) == n == len(b["daily"]["dates"])
        for k in ("persistence", "v1", "v1_Rlag", "v1_Rd0", "hybrid", "hybrid_Rlag"):
            assert len(b["daily"][k]) == n and np.isfinite(b["daily"][k]).all()


# ------------------------------------------------------------------ leakage
def test_no_regional_date_after_the_allowed_day_is_read():
    for b in M["tests"].values():
        a = b["leakage_audit"]
        assert a["rows_reading_too_late"] == [] and a["no_date_after_allowed_day_read"] and a["rows_checked"] == b["n_days"]
        assert a["rows_where_live_origin_day_differs_from_full_day"] > 0           # the 23:00 view is really used


def test_canary_would_expose_a_leak():
    for b in M["tests"].values():
        assert b["canary_mae_with_future_target_feature"] < 0.5 * b["models"]["v1"]["mae"]


def test_control_equals_production_fitting_code():
    for b in M["tests"].values():
        assert b["control_identical_to_production_fit"] == {"v1": True, "v2": True}


def test_regional_stations_never_enter_the_delhi_mean():
    assert M["integrity"]["regional_station_ids_in_delhi_folders"] == []
    stations = json.loads((HERE / "stations.json").read_text())["stations"]
    assert len(stations) == M["integrity"]["regional_stations"] == 44
    assert {s["group"] for s in stations} == set(rr.GROUPS)
    assert {s["board"] for s in stations} <= {"HSPCB", "UPPCB", "PPCB", "CPCC"}   # state pollution-board networks, none from Delhi (DPCC / CPCB-Delhi / IMD)
    assert not any("delhi" in s["name"].lower() for s in stations)
    assert min(s["dist_km"] for s in stations) > 10 and all(s["group"] in rr.GROUPS for s in stations)


def test_regional_features_read_only_the_requested_days():
    d0 = date(2025, 11, 10)
    one = timedelta(days=1)
    store = {g: rr.Spy({d0 - 2 * one: 100.0, d0 - one: 200.0, d0: 300.0, d0 + one: 999.0}) for g in rr.GROUPS}
    f = rr.reg_feats(store, store, d0 - one, d0 - 2 * one, 150.0, "r1")
    seen = {k for g in rr.GROUPS for k in store[g].seen}
    assert seen == {d0 - one, d0 - 2 * one} and max(seen) < d0
    assert f["r1_punjab_level"] == pytest.approx(math.log1p(200.0))
    assert f["r1_punjab_rel"] == pytest.approx(math.log1p(200.0) - math.log1p(150.0))
    assert f["r1_punjab_chg"] == pytest.approx(math.log1p(200.0) - math.log1p(100.0))
    assert len(f) == 12


def test_missing_regional_values_stay_missing():
    one = timedelta(days=1)
    d = date(2025, 3, 3)
    store = {g: {} for g in rr.GROUPS}
    store["punjab"] = {d: 80.0}
    f = rr.reg_feats(store, store, d, d - one, 50.0, "r1")
    assert math.isnan(f["r1_ncr_near_level"]) and math.isnan(f["r1_punjab_chg"]) and f["r1_punjab_level"] == pytest.approx(math.log1p(80.0))
    assert math.isnan(rr.reg_feats(store, store, d, d - one, float("nan"), "r1")["r1_punjab_rel"])


# ------------------------------------------------------------------ placebo and data rules
def test_placebo_permutes_within_each_calendar_month_only():
    keys = [date(2019, 1, 1) + timedelta(days=i) for i in range(0, 1500, 3)]
    pi = rr.placebo_map(keys)
    assert sorted(pi) == sorted(pi.values()) == sorted(keys)                       # a bijection
    assert all(t.month == u.month for t, u in pi.items())
    assert pi == rr.placebo_map(keys)                                              # fixed seed -> reproducible
    assert sum(t != u for t, u in pi.items()) > 0.9 * len(keys)


def test_group_day_needs_two_stations_and_16_valid_hours(tmp_path, monkeypatch):
    stations = [{"id": 1, "group": "punjab"}, {"id": 2, "group": "punjab"}, {"id": 3, "group": "up_east"}]
    (tmp_path / "stations.json").write_text(json.dumps({"stations": stations}))
    ist = timezone(timedelta(hours=5, minutes=30))

    def write(sid, spans):
        lines = ["datetime,pm25"]
        for day, hours, v in spans:
            for h in range(hours):
                lines.append(datetime(day.year, day.month, day.day, h, 0, tzinfo=ist).isoformat() + f",{v}")
        (tmp_path / f"{sid}.csv").write_text("\n".join(lines))

    d1, d2, d3 = date(2025, 1, 10), date(2025, 1, 11), date(2025, 1, 12)
    write(1, [(d1, 24, 100), (d2, 24, 100), (d3, 10, 100)])
    write(2, [(d1, 24, 200), (d3, 24, 200)])
    write(3, [(d1, 24, 50)])
    monkeypatch.setattr(rr, "HERE", tmp_path)
    monkeypatch.setattr(rr, "REGIONAL", tmp_path)
    val, cnt = rr.load_regional()
    assert val["punjab"] == {d1: 150.0} and cnt["punjab"] == {d1: 2}              # d2: one station; d3: station 1 has only 10 hours
    assert val["up_east"] == {}                                                    # a single station never makes a group value
    live, _ = rr.load_regional(cutoff=22)
    assert d1 in live["punjab"]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "regional" / "manifest.json").exists(), reason="raw station files are git-ignored")
def test_raw_download_stops_at_the_hard_stop():
    man = json.loads((ROOT / "data" / "raw" / "regional" / "manifest.json").read_text())
    assert man["end"] == HARD_STOP.isoformat() and len(man["stations"]) == 44
    for p in (ROOT / "data" / "raw" / "regional").glob("[0-9]*.csv"):
        last = p.read_text().strip().splitlines()[-1].split(",")[0]
        assert rr.to_ist(datetime.fromisoformat(last.replace("Z", "+00:00"))).date() <= HARD_STOP


# ------------------------------------------------------------------ the decision
def test_decision_matches_the_declared_rule_and_is_recomputable():
    again = rr.verdict(M)
    assert again["regional_data_helpful"] == M["decision"]["regional_data_helpful"]
    assert again["per_test"] == json.loads(json.dumps(M["decision"]["per_test"]))
    helpful = all(v[f"{i}_pass"] for v in M["decision"]["per_test"].values() for i in range(1, 6))
    assert helpful == M["decision"]["regional_data_helpful"]
    assert M["decision"]["answer"].startswith("HELPFUL" if helpful else "NO EVIDENCE")


def test_verdict_logic_on_a_synthetic_pass_and_fail():
    def blk(mae_pct, lo7, hi7, lo14, hi14, cat, mc, mr, mp, sa):
        return {"diffs": {"v1_Rlag_vs_v1": {"mae_pct": mae_pct, "mae": mr - mc, "cat_acc": cat, "mae_ci95_7d": [lo7, hi7], "mae_ci95_14d": [lo14, hi14]}},
                "models": {"v1": {"mae": mc}, "v1_Rlag": {"mae": mr}, "v1_plc": {"mae": mp}}, "seed_averaged": {"v1_Rlag_vs_v1": {"mae": sa}}}
    good = blk(-3.0, -1.0, -0.2, -1.1, -0.1, 0.01, 30.0, 29.1, 30.1, -0.8)
    bad_ci = blk(-3.0, -1.0, 0.4, -1.1, 0.3, 0.01, 30.0, 29.1, 30.1, -0.8)
    bad_pl = blk(-3.0, -1.0, -0.2, -1.1, -0.1, 0.01, 30.0, 29.1, 29.2, -0.8)
    assert rr.verdict({"tests": {"a": good, "b": good}})["regional_data_helpful"] is True
    assert rr.verdict({"tests": {"a": good, "b": bad_ci}})["regional_data_helpful"] is False
    assert rr.verdict({"tests": {"a": good, "b": bad_pl}})["regional_data_helpful"] is False
    assert rr.verdict({"tests": {"a": bad_ci, "b": bad_ci}})["regional_data_helpful"] is False


def test_experiment_writes_only_inside_its_own_folder_and_saves_no_model():
    src = (HERE / "run_regional.py").read_text()
    assert "save_model" not in src and "save_raw" not in src
    writes = [ln for ln in src.splitlines() if "write_text(" in ln or 'open(HERE' in ln or ", \"w\")" in ln]
    assert writes and all(("HERE" in ln) or ("out.write_text" in ln) for ln in writes)


# ------------------------------------------------------------------ the audit itself must be able to fail
def _toy():
    rng = np.random.default_rng(1)
    days = [date(2025, 11, 1) + timedelta(days=i) for i in range(-3, 20)]
    S = {v: {g: {d: float(rng.uniform(50, 300)) for d in days} for g in rr.GROUPS} for v in ("full", "live")}
    base = {date(2025, 11, 5) + timedelta(days=i): {"y": 100.0, "row": {"pm_d0": 120.0, "pm_d1": 110.0}} for i in range(8)}
    return S, base


def test_audit_passes_the_real_construction_and_flags_leaking_mutants():
    one = timedelta(days=1)
    S, base = _toy()
    assert rr.leakage_audit(rr.augment(base, S, "test"), S)["rows_reading_too_late"] == []

    def mutant(lag_day, d0_day):
        out = {}
        for t, v in base.items():
            d0, row = t - one, dict(v["row"])
            row.update(rr.reg_feats(S["full"], S["full"], lag_day(d0, t), lag_day(d0, t) - one, row["pm_d1"], "r1"))
            row.update(rr.reg_feats(S["live"], S["full"], d0_day(d0, t), d0, row["pm_d0"], "r0"))
            out[t] = {"y": v["y"], "row": row}
        return out

    leak_lag = mutant(lambda d0, t: d0, lambda d0, t: d0)             # lag columns read the origin day
    leak_d0 = mutant(lambda d0, t: d0 - one, lambda d0, t: t)         # same-day columns read the target day
    assert len(rr.leakage_audit(leak_lag, S)["rows_reading_too_late"]) == len(base)
    assert len(rr.leakage_audit(leak_d0, S)["rows_reading_too_late"]) == len(base)


# ------------------------------------------------------------------ headline numbers must be the stable ones
def test_five_seed_means_are_consistent_with_the_stored_runs():
    for b in M["tests"].values():
        assert set(b["seeds_cat_acc"]) == {"42", "7", "123", "2024", "99"}
        for k, fm in b["five_seed_mean"].items():
            vals = [b["seeds_cat_acc"][s][k] for s in b["seeds_cat_acc"]]
            assert fm["cat_acc"] == pytest.approx(sum(vals) / 5) and fm["cat_acc_min"] == min(vals) and fm["cat_acc_max"] == max(vals)
        assert b["seeds_cat_acc"]["42"]["v1"] == pytest.approx(b["models"]["v1"]["cat_acc"])      # seed 42 equals the single-run table
        assert b["seeds_cat_acc"]["42"]["v1_Rlag"] == pytest.approx(b["models"]["v1_Rlag"]["cat_acc"])


def test_report_headline_quotes_the_five_seed_means():
    txt = (HERE / "report.md").read_text()
    for b in M["tests"].values():
        for k in ("v1", "v1_Rlag", "v1_plc"):
            assert f"{100 * b['five_seed_mean'][k]['cat_acc']:.1f} %" in txt.split("## What was added")[0]
