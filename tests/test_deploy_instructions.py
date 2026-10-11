"""Guard against re-introducing `sam build` into the deploy path.

Why: build/lambda is a finished package (Linux wheels included).  `sam build` copies a source folder that has no
requirements.txt with an exclusion list that contains "*.so" (aws-lambda-builders, python_pip workflow), so numpy, scipy and
xgboost lose their compiled files and the Lambda fails on `import numpy`.  This happened on the first real deploy (upload
15.5 MB instead of 60.5 MB).  `sam deploy --template-file template.yaml` zips the folder as it is.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["scripts/build_lambda.sh", "scripts/deploy_shadow.sh"]


def _commands(path):
    """Lines that would actually be executed: not comments, not echo/printf text, not inside a here-document."""
    out, heredoc_end = [], None
    for raw in (ROOT / path).read_text().splitlines():
        line = raw.strip()
        if heredoc_end:
            if line == heredoc_end:
                heredoc_end = None
            continue
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            heredoc_end = m.group(1)
        if line and not line.startswith("#") and not line.startswith(("echo", "printf", "die ", "step ")):
            out.append(line)
    return out


def test_scripts_never_run_sam_build():
    for path in SCRIPTS:
        bad = [c for c in _commands(path) if re.match(r"(\w+=\S+\s+)*sam\s+build\b", c)]
        assert not bad, f"{path} runs `sam build`, which strips the *.so files from the package: {bad}"


def test_deploy_script_deploys_the_prebuilt_folder_directly():
    text = (ROOT / "scripts/deploy_shadow.sh").read_text()
    assert "sam deploy --guided --template-file template.yaml" in text
    assert ".aws-sam/build/template.yaml" not in text      # that file would point at the stripped copy


def test_build_script_tells_people_not_to_run_sam_build():
    text = (ROOT / "scripts/build_lambda.sh").read_text()
    assert "Next:  sam deploy --guided --template-file template.yaml" in text
    assert "Do NOT run 'sam build'" in text


def test_docs_do_not_tell_people_to_run_sam_build():
    for doc in ("README.md", "FORWARD_TEST_OPERATIONS.md"):
        text = (ROOT / doc).read_text()
        assert "sam deploy --guided --template-file template.yaml" in text, doc
        assert re.search(r"[Dd]o not run `sam build`|Do \*\*not\*\* run `sam build`|\*\*Do not run `sam build`", text), doc
        # no instruction line that starts with `sam build` (a bare command, or inside a numbered step)
        assert not re.search(r"(?m)^\s*(\d+\.\s+)?`?sam build\b", text), f"{doc} still has a `sam build` instruction"


def test_template_points_at_the_prebuilt_folder():
    text = (ROOT / "template.yaml").read_text()
    assert text.count("CodeUri: build/lambda/") == 2
