"""Tests for the isolated CatBoost experiment. Run with the CatBoost environment:
    PYTHONDONTWRITEBYTECODE=1 .venv-catboost/bin/python -m pytest experiments/catboost/tests -q
They are skipped (not failed) when CatBoost is not installed, and they are not part of the project's own tests/ suite."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("catboost")
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
for p in (HERE, ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "tests"):
    sys.path.insert(0, str(p))

import run_catboost as rc  # noqa: E402
import stats as st  # noqa: E402
import winter_backtest as wb  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from test_backtest import synth  # noqa: E402
import option_screen as osn  # noqa: E402


def _train_rows(nan_dayparts=False):
    hist, live = synth(last=date(2023, 6, 1))
    obs = lambda t, d0: {k: hist[t][k] for k in WEATHER_KEYS}  # noqa: E731
    rows = osn.build(hist, live, obs, view="offline")
    if nan_dayparts:
        for v in rows.values():
            for k in ("pm_night", "pm_commute", "pm_school", "pm_evening", "pm_late", "pm_buildup", "pm_last3h_ratio"):
                v["row"][k] = float("nan")
    return rows


FIXED = {"iterations": 60, "od_wait": 10, "bernoulli_subsample": 0.85, "loss_function": "RMSE"}
CFG = {"depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 3, "bootstrap": "Bayesian"}


def test_config_was_declared_and_tuning_ends_before_every_test_window():
    cfg = rc.config()
    cut = date.fromisoformat(cfg["tuning"]["cutoff_target_date_exclusive"])
    assert cut <= min(wb.winter_window(Y)[0] for Y in (2022, 2023, 2024, 2025))
    assert cfg["declared_before_running"] is True and len(rc.grid_configs(cfg)) == 16
    assert rc.cfg_key(cfg["tuning"]["baseline_config"]) in {rc.cfg_key(c) for c in rc.grid_configs(cfg)}
    assert rc.HARD_END < date(2026, 10, 15)                    # winter 2026-27 is never read


def test_selection_rule_is_lowest_mae_with_ties_to_the_baseline():
    base = {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3, "bootstrap": "Bayesian"}
    rows = [{"config": {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 10, "bootstrap": "Bayesian"}, "oof_mae": 24.99},
            {"config": base, "oof_mae": 25.00},
            {"config": {"depth": 4, "learning_rate": 0.06, "l2_leaf_reg": 3, "bootstrap": "Bernoulli"}, "oof_mae": 27.0}]
    assert rc.pick_tuned(rows, base)["config"] == base        # within 0.1 % -> tie -> baseline
    rows[1]["oof_mae"] = 25.5
    assert rc.pick_tuned(rows, base)["config"]["depth"] == 6   # clearly better -> chosen


def test_catboost_keeps_nan_dayparts_and_predicts_finite_values():
    tr = _train_rows(nan_dayparts=True)
    ts = sorted(tr)
    X = osn.M([tr[t]["row"] for t in ts], rc.V2N)
    assert np.isnan(X).any()                                   # nothing was imputed
    y = np.array([tr[t]["y"] for t in ts])
    m = rc.cb_fit(X, y, rc.V2N, CFG, FIXED, 42)
    p = rc.pm_from(m, X[-20:], rc.V2N)
    assert np.isfinite(p).all() and (p > 0).all()


def test_same_seed_is_reproducible_and_no_files_are_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = _train_rows()
    ts = sorted(tr)
    X = osn.M([tr[t]["row"] for t in ts], rc.V1N)
    y = np.array([tr[t]["y"] for t in ts])
    a = rc.pm_from(rc.cb_fit(X, y, rc.V1N, CFG, FIXED, 7), X[-15:], rc.V1N)
    b = rc.pm_from(rc.cb_fit(X, y, rc.V1N, CFG, FIXED, 7), X[-15:], rc.V1N)
    assert np.allclose(a, b)
    assert list(tmp_path.iterdir()) == []                      # allow_writing_files=False: no catboost_info


def test_frozen_fit_ignores_rows_at_or_after_the_window(monkeypatch):
    monkeypatch.setattr(rc, "MIN_TRAIN", 30)
    tr = _train_rows()
    ts = sorted(tr)
    start = ts[len(ts) // 2]
    D = {"train_obs": tr}
    cfgs = {"cb_base": CFG, "cb_tuned": CFG}
    rows = [tr[ts[-1]]["row"]]
    a = rc.fit_window(D, start, {"s": ([ts[-1]], rows)}, [1], cfgs, FIXED)
    changed = {t: dict(v) for t, v in tr.items()}
    for t in ts:
        if t >= start:
            changed[t] = {"y": v_y * 5 if (v_y := tr[t]["y"]) else 1.0, "row": tr[t]["row"]}
    b = rc.fit_window({"train_obs": changed}, start, {"s": ([ts[-1]], rows)}, [1], cfgs, FIXED)
    assert a["status"] == "scored" and b["status"] == "scored"
    assert np.allclose(a["sets"]["s"]["pred"]["cb_base"][1]["v1"], b["sets"]["s"]["pred"]["cb_base"][1]["v1"])


def test_r2_and_within_one_are_consistent():
    rng = np.random.default_rng(0)
    y = rng.uniform(40, 300, 60)
    p = y + rng.normal(0, 25, 60)
    dates = [date(2024, 11, 1).fromordinal(date(2024, 11, 1).toordinal() + i) for i in range(60)]
    s = st.Stats(y, {"a": p, "persistence": y * 0.9}, dates)
    pt = s.point("a")
    assert abs(pt["r2"] - (1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())) < 1e-12
    assert pt["within_one_cat"] >= pt["cat_acc"]
    assert s.diff("a", "a")["identical"] and not s.diff("a", "a")["excludes_zero_both"]
    assert s.model("a")["r2_ci95"][0] < pt["r2"] < s.model("a")["r2_ci95"][1] + 0.2


def test_interval_matches_the_projects_own_bootstrap():
    rng = np.random.default_rng(3)
    y = rng.uniform(40, 300, 80)
    pa, pb = y + rng.normal(0, 30, 80), y + rng.normal(0, 24, 80)
    dates = [date.fromordinal(date(2024, 11, 1).toordinal() + i) for i in range(80)]
    s = st.Stats(y, {"a": pa, "b": pb}, dates)
    ref = wb.mbb_mae_diff(y, pa, pb, dates, block=7, n_boot=5000, seed=0)
    assert np.allclose(s.diff("a", "b")["mae_ci95_7d"], ref["ci95_diff"], atol=1e-9)


def test_everything_is_written_inside_the_catboost_folder_only():
    for p in (rc.TUNING_PATH, rc.RESULTS_PATH, rc.CONFIG_PATH):
        assert p.parent == HERE
    assert HERE == ROOT / "experiments" / "catboost"


def test_test_stage_refuses_a_tuning_file_from_another_config(tmp_path, monkeypatch):
    fake = tmp_path / "tuning.json"
    fake.write_text(json.dumps({"config_sha256": "0" * 64}))
    monkeypatch.setattr(rc, "TUNING_PATH", fake)
    with pytest.raises(SystemExit):
        rc.stage_test()
