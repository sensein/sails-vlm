"""pyproject invariants from the 2026-07 follow-up phase (issue #2)."""
import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def project():
    return tomllib.loads(PYPROJECT_PATH.read_text())["project"]


def _names(requirements):
    """Bare package names from PEP 508 requirement strings."""
    return {re.split(r"[<>=!\[ ;]", r, 1)[0].strip() for r in requirements}


def test_torchaudio_dropped(project):
    """Never imported anywhere in sails_vlm/ or tests/ - pure install weight."""
    assert "torchaudio" not in _names(project["dependencies"])


def test_accelerate_in_family_extras_not_core(project):
    """accelerate is only needed by transformers' device_map='auto'. Adapters
    using device_map (verified 2026-07-27): qwen2_5/qwen3/qwen3_video (qwen
    extra), cosmos/cosmos_video (cosmos), internvl. ovis2.py uses none."""
    extras = project["optional-dependencies"]
    assert "accelerate" not in _names(project["dependencies"])
    for family in ("qwen", "cosmos", "internvl"):
        assert "accelerate" in _names(extras[family]), family
    assert "accelerate" not in _names(extras["ovis2"])


def test_bitsandbytes_where_quantization_paths_exist(project):
    """qwen3/qwen3_video expose load_in_4bit/8bit (qwen extra); internvl was
    fixed in PR #1 and must keep it."""
    extras = project["optional-dependencies"]
    for family in ("qwen", "internvl"):
        assert "bitsandbytes" in _names(extras[family]), family


def test_license_and_readme_declared(project):
    """Apache-2.0, matching senselab (user-approved 2026-07-27). authors is
    deliberately absent: deferred to a follow-up issue filed at PR time."""
    assert project.get("readme") == "README.md"
    assert project.get("license") == {"file": "LICENSE"}
    assert "authors" not in project
    license_text = (PYPROJECT_PATH.parent / "LICENSE").read_text()
    assert "Apache License" in license_text
    assert "Copyright 2026 Sensein Lab, MIT" in license_text
