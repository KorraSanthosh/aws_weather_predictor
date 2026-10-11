"""Metrics and moving-block-bootstrap intervals (copy of experiments/catboost/stats.py so this experiment is self-contained)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "model", ROOT / "pipeline"):
    sys.path.insert(0, str(p))

from core.aqi import pm25_to_category  # noqa: E402
from experiments import CATS  # noqa: E402
import forecast_training_experiment as fte  # noqa: E402  (only boot_weights is used)

LEVEL = (2.5, 97.5)


def _ci(x):
    lo, hi = np.percentile(x, LEVEL)
    return [float(lo), float(hi)]


def cat_index(v):
    return CATS.index(pm25_to_category(float(v)))


class Stats:
    """All metrics for one set of days.  Resamples are shared by every statistic and every comparison on the set."""

    def __init__(self, y, preds, dates):
        self.y = np.asarray(y, float)
        self.P = {k: np.asarray(v, float) for k, v in preds.items()}
        self.dates = list(dates)
        self.n = len(self.y)
        self.ok = self.n >= 10
        self.W = {b: fte.boot_weights(self.dates, b) for b in (7, 14)} if self.ok else {}
        self.cy = np.array([cat_index(v) for v in self.y])
        self._cache = {}

    # per-day building blocks ------------------------------------------------------------------
    def day(self, name, what):
        key = (name, what)
        if key not in self._cache:
            p = self.P[name]
            e = p - self.y
            if what == "ae":
                v = np.abs(e)
            elif what == "se":
                v = e ** 2
            else:
                cp = np.array([cat_index(x) for x in p])
                v = (cp == self.cy).astype(float) if what == "hit" else (np.abs(cp - self.cy) <= 1).astype(float)
            self._cache[key] = v
        return self._cache[key]

    def sst_terms(self):
        return self.y, self.y ** 2

    # point estimates --------------------------------------------------------------------------
    def r2(self, name):
        sst = float(((self.y - self.y.mean()) ** 2).sum())
        return float(1 - self.day(name, "se").sum() / sst) if sst > 0 else float("nan")

    def point(self, name):
        return {"mae": float(self.day(name, "ae").mean()), "rmse": float(np.sqrt(self.day(name, "se").mean())), "r2": self.r2(name),
                "cat_acc": float(self.day(name, "hit").mean()), "within_one_cat": float(self.day(name, "hit1").mean())}

    # bootstrap distributions ------------------------------------------------------------------
    def boot(self, name, metric, block=7):
        W = self.W[block]
        sw = W.sum(1)
        ok = sw > 0
        W, sw = W[ok], sw[ok]
        if metric == "mae":
            return (W @ self.day(name, "ae")) / sw
        if metric == "rmse":
            return np.sqrt((W @ self.day(name, "se")) / sw)
        if metric == "r2":
            ybar = (W @ self.y) / sw
            sst = (W @ self.y ** 2) - sw * ybar ** 2
            return 1 - (W @ self.day(name, "se")) / np.where(sst > 0, sst, np.nan)
        return (W @ self.day(name, {"cat_acc": "hit", "within_one_cat": "hit1"}[metric])) / sw

    METRICS = ("mae", "rmse", "r2", "cat_acc", "within_one_cat")

    def model(self, name, baseline="persistence"):
        out = self.point(name)
        if self.ok:
            for m in self.METRICS:
                b = self.boot(name, m)
                out[m + "_ci95"] = _ci(b[np.isfinite(b)])
            if baseline in self.P and name != baseline:
                bm, bp = self.boot(name, "mae"), self.boot(baseline, "mae")
                out["vs_persistence_pct"] = float((1 - out["mae"] / self.point(baseline)["mae"]) * 100)
                out["vs_persistence_pct_ci95"] = _ci((1 - bm / bp) * 100)
        return out

    def diff(self, a, b):
        """metric(b) - metric(a) on the same days.  MAE/RMSE: negative = b better.  R2/accuracy: positive = b better."""
        pa, pb = self.point(a), self.point(b)
        out = {m: pb[m] - pa[m] for m in self.METRICS}
        out["mae_pct"] = (pb["mae"] / pa["mae"] - 1) * 100
        out["identical"] = bool(np.array_equal(self.P[a], self.P[b]))
        out["n_days"] = self.n
        if self.ok and not out["identical"]:
            for blk in (7, 14):
                out[f"mae_ci95_{blk}d"] = _ci(self.boot(b, "mae", blk) - self.boot(a, "mae", blk))
            out["mae_pct_ci95_7d"] = _ci((self.boot(b, "mae") / self.boot(a, "mae") - 1) * 100)
            for m in ("rmse", "r2", "cat_acc", "within_one_cat"):
                d = self.boot(b, m) - self.boot(a, m)
                out[m + "_ci95_7d"] = _ci(d[np.isfinite(d)])
            out["excludes_zero_both"] = bool(all(out[f"mae_ci95_{k}d"][1] < 0 or out[f"mae_ci95_{k}d"][0] > 0 for k in (7, 14)))
        else:
            out["excludes_zero_both"] = False
        return out


def sudden_mask(y, pm0, thr):
    return np.abs(np.asarray(y) - np.asarray(pm0)) / np.maximum(np.asarray(pm0), 1.0) >= thr
