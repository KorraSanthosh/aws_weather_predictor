"""Record (and later verify) exactly what the winter 2026-27 forward test is made of.

    python model/freeze_manifest.py --write     # (re)generate model/forward_test_manifest.json - read-only otherwise
    python model/freeze_manifest.py --check     # exit 1 if any frozen file, constant or model artifact differs

The manifest lists the SHA-256 of every model artifact, the feature definitions, the day-part and AQI-category code, the
hybrid router, the scoring code, the constants and this protocol. Until you commit it and tag the commit the status is
"PREPARED" and --write may be re-run while you review. After the freeze, any difference is an amendment (see
model/EVALUATION_PROTOCOL.md section 9) and results from before the amendment may not be used to justify it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from forward_protocol import PROTOCOL  # noqa: E402

MANIFEST = ROOT / "model" / "forward_test_manifest.json"
FROZEN_FILES = [
    "model/v1_25_baseline/model.json", "model/v1_25_baseline/model_meta.json",
    "model/v2_35_candidate/model.json", "model/v2_35_candidate/model_meta.json",
    "core/features.py",          # feature definitions, FEATURE_SETS
    "core/daily.py",             # how day-part PM2.5 is computed from hourly data
    "core/aqi.py",               # AQI category breakpoints used for category accuracy
    "core/hybrid.py",            # the hybrid routing rule
    "backend/forecast_handler.py",   # builds the identical inputs and logs the shadow rows
    "model/forward_protocol.py", "model/forward_eval.py", "model/EVALUATION_PROTOCOL.md",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path = ROOT, files=FROZEN_FILES) -> dict:
    try:
        import xgboost
        xgb_v = xgboost.__version__
    except Exception:                     # pragma: no cover
        xgb_v = "unknown"
    return {"status": "PREPARED - becomes FROZEN when this file is committed and the commit is tagged",
            "protocol": PROTOCOL,
            "files_sha256": {f: sha256(root / f) for f in files},
            "environment_when_prepared": {"python": platform.python_version(), "xgboost": xgb_v}}


def verify(manifest: dict, root: Path = ROOT) -> list[str]:
    """Return a list of differences (empty = identical to the manifest)."""
    problems = []
    for f, digest in manifest["files_sha256"].items():
        p = root / f
        if not p.exists():
            problems.append(f"missing: {f}")
        elif sha256(p) != digest:
            problems.append(f"changed: {f}")
    if manifest["protocol"] != PROTOCOL:
        problems.append("changed: protocol constants (model/forward_protocol.py) differ from the manifest copy")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.write:
        MANIFEST.write_text(json.dumps(build(), indent=2) + "\n")
        print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(FROZEN_FILES)} files)")
        return 0
    problems = verify(json.loads(MANIFEST.read_text()))
    for p in problems:
        print("MISMATCH", p)
    print("forward-test manifest: " + ("OK - everything matches" if not problems else f"{len(problems)} difference(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
