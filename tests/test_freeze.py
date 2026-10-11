import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
import freeze_manifest as fm  # noqa: E402


def test_manifest_detects_a_changed_or_missing_frozen_file(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "b.txt").write_text("two")
    m = fm.build(tmp_path, ["a.txt", "b.txt"])
    assert fm.verify(m, tmp_path) == []
    (tmp_path / "a.txt").write_text("ONE")
    (tmp_path / "b.txt").unlink()
    assert sorted(fm.verify(m, tmp_path)) == ["changed: a.txt", "missing: b.txt"]


def test_manifest_detects_changed_protocol_constants(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    m = fm.build(tmp_path, ["a.txt"])
    m["protocol"] = {**m["protocol"], "min_matched_days": 30}
    assert any("protocol constants" in p for p in fm.verify(m, tmp_path))


def test_working_copy_matches_its_forward_test_manifest():
    """Fails the moment a frozen file (models, features, hybrid rule, scoring, protocol) is edited without re-preparing
    the manifest. Before the freeze: run `python model/freeze_manifest.py --write` after a deliberate change."""
    if not fm.MANIFEST.exists():
        pytest.skip("manifest not prepared")
    assert fm.verify(json.loads(fm.MANIFEST.read_text())) == []


def test_protocol_document_states_the_frozen_numbers():
    doc = (ROOT / "model" / "EVALUATION_PROTOCOL.md")
    if not doc.exists():
        pytest.skip("protocol not written")
    text = doc.read_text().lower()
    for needle in ("2026-10-15", "2027-02-15", "90", "5%", "95%", "7-day", "v3_hybrid", "first write wins"):
        assert needle in text, needle
