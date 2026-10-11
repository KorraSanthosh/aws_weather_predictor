#!/usr/bin/env bash
# READ-ONLY validation of the working copy. It changes no model file, makes no git commit/stage/push and touches no AWS.
# (pytest writes only to temporary folders.)   Usage, from the project folder:   bash scripts/validate_working_copy.sh
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }
pass=0; fail=0
ok()  { echo "PASS  $1"; pass=$((pass+1)); }
bad() { echo "FAIL  $1"; fail=$((fail+1)); }
note(){ echo "INFO  $1"; }
sha() { if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$@"; else sha256sum "$@"; fi; }

echo "== 1. Python environment ($PY)"
if "$PY" -c "import numpy, xgboost, pytest, boto3" 2>/dev/null; then ok "numpy, xgboost, pytest, boto3 import"; else
  bad "missing packages - run:  $PY -m pip install -r requirements.txt pytest"; fi

echo "== 2. Default model is still the v1 baseline (unchanged)"
cmp -s model/model.json model/v1_25_baseline/model.json && ok "model/model.json == v1_25_baseline/model.json" || bad "default model.json differs from v1"
cmp -s model/model_meta.json model/v1_25_baseline/model_meta.json && ok "model_meta.json == v1" || bad "default model_meta.json differs from v1"
cmp -s web/data/forecast.json model/v1_25_baseline/web_forecast.json && ok "dashboard data (web/data/forecast.json) == v1" || bad "dashboard data differs from v1"
"$PY" - <<'PY' && ok "default model has 25 features" || bad "default model is not the 25-feature model"
import json,sys
sys.exit(0 if len(json.load(open("model/model_meta.json"))["features"]) == 25 else 1)
PY

echo "== 3. Both model versions are byte-for-byte what was stored"
( cd model && sha -c CHECKSUMS.sha256 >/dev/null 2>&1 ) && ok "model/CHECKSUMS.sha256 verifies (v1 and v2 files)" || bad "a stored model file changed"
"$PY" model/freeze_manifest.py --check && ok "forward-test manifest matches" || bad "forward-test manifest differs (re-run: python model/freeze_manifest.py --write, if the change was deliberate)"

echo "== 4. Full test suite"
if "$PY" -m pytest tests -q -p no:cacheprovider; then ok "pytest: all tests pass"; else bad "pytest reported failures"; fi

echo "== 5. Both models and the hybrid forecast from identical inputs (fake data, no network)"
"$PY" - <<'PY' && ok "v1, v2 and v3_hybrid ran side by side with identical input fingerprints" || bad "side-by-side shadow run failed"
import sys
sys.path.insert(0, "."); sys.path.insert(0, "backend"); sys.path.insert(0, "tests")
import test_backend as t, forecast_handler as fh
import pytest
mp = pytest.MonkeyPatch()
mp.setattr(fh.openaq, "fetch_sensor_hours", t.fake_hours); mp.setattr(fh.wx, "fetch_forecast_weather", t.fake_weather)
mp.delenv("MODEL_VERSION", raising=False)
res = fh.run_forecast(t.NOW, "k", [1, 2], 28.6, 77.2, shadow_versions=["v1_25_baseline", "v2_35_candidate", "v3_hybrid"])
sh = res["shadow"]
for v, r in sh.items():
    print(f"   {v:16} pm25={r.get('pm25')} served={r.get('served_model')} missing={r.get('missing_features')}")
assert len({r["inputs_sha256"] for r in sh.values()}) == 1 and res["model_version"] == "v1_25_baseline"
PY

echo "== 6. Scoring script dry run on SYNTHETIC logs (plumbing only, not real results)"
"$PY" - <<'PY' && ok "forward_eval produces a decision on synthetic rows" || bad "forward_eval failed"
import sys
sys.path.insert(0, "model"); sys.path.insert(0, "tests")
import forward_eval as fe, test_forward_eval as tf
rep = fe.build_report(tf.make_rows())
print("   synthetic decision:", rep["decision"]["decision"], "| matched days:", rep["decision"]["n_matched_days"])
PY

echo "== 7. Git safety (nothing committed, staged or pushed by this work)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  note "HEAD: $(git log -1 --oneline)"
  [ "$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)" = "0" ] && ok "no local commits ahead of origin/main" || bad "there are local commits not on origin/main"
  git diff --cached --quiet && ok "nothing staged" || bad "something is staged"
  note "uncommitted changes: $(git status --short | wc -l | tr -d ' ') path(s) (review with: git status; git diff --stat)"
  FILES=$(git ls-files -co --exclude-standard)
else
  note "not a git repository here - git checks skipped (run this on the Mac project folder)"
  FILES=$(find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './node_modules/*' -not -path './frontend/node_modules/*' | sed 's|^\./||')
fi

echo "== 8. No secrets in files git could pick up"
BAD_ENV=$(echo "$FILES" | grep -E '(^|/)\.env($|\.)' | grep -v '\.env\.example' || true)
[ -z "$BAD_ENV" ] && ok ".env files are ignored / not tracked" || bad "env file(s) not ignored: $BAD_ENV"
LEAK=$(echo "$FILES" | grep -v -E '^(scripts/validate_working_copy.sh|data/|model/CHECKSUMS|model/forward_test_manifest|frontend/)' \
  | xargs grep -I -l -E "(OPENAQ_API_KEY|X-API-Key|api[_-]?key)[\"' ]*[:=][\"' ]*[0-9a-fA-F]{32,}" 2>/dev/null || true)
[ -z "$LEAK" ] && ok "no API-key-looking values in tracked/untracked files" || bad "possible key in: $LEAK"

echo "== 9. Nothing deployed from here"
[ -d .aws-sam ] && note ".aws-sam/ exists (a previous 'sam build'; ignored by git)" || ok "no .aws-sam build folder"
note "this script never calls AWS; deployment is a separate step you run yourself"

echo; echo "RESULT: $pass passed, $fail failed"
[ "$fail" = "0" ]
