"""Prove the CatBoost experiment changed nothing outside experiments/catboost/.

    python experiments/catboost/verify_unchanged.py --snapshot [FILE]   # before the experiment: hash every protected file
    python experiments/catboost/verify_unchanged.py --check    [FILE] [--tests "123 passed"]   # after: compare, write integrity.json

Protected = every file under the project folder except experiments/catboost/, caches (__pycache__, .pytest_cache),
.venv-catboost and .git.  (Git itself is not touched: this script never runs git commands that write.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv-catboost", ".git", "node_modules"}
DEFAULT = HERE / "protected_snapshot.sha256"


def files():
    out = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS and not (Path(dp) == ROOT / "experiments" and d == "catboost")]
        out += [Path(dp) / f for f in fn if not f.endswith(".pyc")]
    return sorted(out)


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    return {str(p.relative_to(ROOT)): digest(p) for p in files()}


def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        h, _, name = line.partition("  ")
        out[name[2:] if name.startswith("./") else name] = h
    return out


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", nargs="?", const=str(DEFAULT))
    g.add_argument("--check", nargs="?", const=str(DEFAULT))
    ap.add_argument("--tests", default="")
    a = ap.parse_args()
    if a.snapshot:
        s = snapshot()
        Path(a.snapshot).write_text("".join(f"{h}  {n}\n" for n, h in sorted(s.items())))
        print(f"hashed {len(s)} protected files -> {a.snapshot}")
    else:
        before, now = load(a.check), snapshot()
        changed = sorted(n for n in before if n in now and before[n] != now[n])
        removed = sorted(n for n in before if n not in now)
        added = sorted(n for n in now if n not in before)
        ok = not changed and not removed and not added
        res = {"files_checked": len(before), "changed": changed, "removed": removed, "added_outside_catboost_folder": added, "unchanged": ok,
               "summary": (f"all {len(before)} protected files (models, checksums, data, code, docs, tests) are byte-identical; nothing added or removed outside experiments/catboost/"
                           if ok else f"PROBLEM: changed={changed} removed={removed} added={added}"),
               "tests": a.tests}
        (HERE / "integrity.json").write_text(json.dumps(res, indent=1))
        print(res["summary"])
        raise SystemExit(0 if ok else 1)
