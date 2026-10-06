"""pyproject invariants from the 2026-07 follow-up phase (issue #2)."""
import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads(PYPROJECT_PATH.read_text())


@pytest.fixture(scope="module")
def project(pyproject):
    return pyproject["project"]


@pytest.fixture(scope="module")
def sdist_include(pyproject):
    targets = pyproject["tool"]["hatch"]["build"]["targets"]
    return targets["sdist"]["include"]


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


# --- sdist file selection -------------------------------------------------
# Without an explicit sdist target, hatchling falls back to "everything not
# VCS-ignored", so any untracked directory left in the working tree ships in
# the sdist. A 5.9G .venv-audit/ (not covered by the .venv/ gitignore rule)
# made `uv build` hang for >5min on 2026-08-10; .superpowers/ and logs/ were
# shipping silently for the same reason. An allowlist makes the contents a
# function of the config rather than of whatever is lying around.
#
# These assert the declared config, which is what can regress in review. That
# hatchling honours it is proved empirically by `uv build` (Task 11 Step 3),
# not here - building an sdist inside the suite would reintroduce the hang.


def test_sdist_declares_explicit_allowlist(sdist_include):
    assert isinstance(sdist_include, list) and sdist_include


@pytest.mark.parametrize("pattern", ["/sails_vlm", "/tests", "/configs", "/pyproject.toml", "/README.md", "/LICENSE"])
def test_sdist_ships_what_the_suite_needs(sdist_include, pattern):
    """Every repo path the tests resolve via parent.parent must be shipped, or
    the sdist is not testable: configs/ (test_configs, test_adapter_contract),
    sails_vlm/models/ (test_adapter_contract), LICENSE (test_license...)."""
    assert pattern in sdist_include


def test_sdist_patterns_are_root_anchored(sdist_include):
    """The bug this file exists for. hatchling include patterns are
    gitignore-style: unanchored "tests" matches **/tests, so an unanchored
    allowlist re-admits site-packages/scipy/stats/tests and dist-info/LICENSE
    from any venv in the tree. Config-level assertions alone missed this - it
    was caught by inspecting the built artifact - hence this explicit check."""
    unanchored = [p for p in sdist_include if not p.startswith("/")]
    assert not unanchored, f"unanchored patterns match at any depth: {unanchored}"


SDIST_GLOB = "sails_vlm-*.tar.gz"
# Mirrors the exclusions the sdist allowlist implies: workspace scratch
# (.venv*, .superpowers, .pytest_cache, dist), SLURM output (logs), and the
# dirs the pyproject comment names as deliberately out (docs, notebooks,
# scripts). site-packages catches a venv shipped under any other name.
JUNK_RE = re.compile(
    r"(^|/)(\.venv[^/]*|\.superpowers|\.pytest_cache|logs|notebooks|docs|scripts|dist)(/|$)|site-packages"
)


def test_sdist_patterns_match_existing_paths(sdist_include):
    """hatchling drops a non-matching include pattern silently - verified
    2026-08-14: a bogus "/THIS-PATH-DOES-NOT-EXIST.txt" entry still built,
    exit 0, no warning. So a rename would quietly empty the sdist of that
    path with every other assertion here still green."""
    root = PYPROJECT_PATH.parent
    missing = [p for p in sdist_include if not (root / p.lstrip("/")).exists()]
    assert not missing, f"include patterns match nothing on disk: {missing}"


def test_built_sdist_contains_no_workspace_junk():
    """Verifies the real artifact, not just the config. Skips when nothing has
    been built - `uv build` is the gate that populates dist/ (Task 11 Step 3).
    Deliberately does not shell out to `uv build`: a regressed config makes the
    build walk the whole tree, which hangs rather than fails."""
    import tarfile

    # Newest by mtime, not lexicographic: `uv build` does not clean dist/, and
    # sorted() puts 0.9.0 after 0.10.0, so [-1] would inspect a stale tarball.
    sdists = sorted((PYPROJECT_PATH.parent / "dist").glob(SDIST_GLOB), key=lambda p: p.stat().st_mtime)
    if not sdists:
        pytest.skip("no built sdist in dist/; run `uv build` first")
    with tarfile.open(sdists[-1]) as tf:
        names = tf.getnames()
    leaked = [n for n in names if JUNK_RE.search(n)]
    assert not leaked, f"{len(leaked)} junk paths in sdist, e.g. {leaked[:5]}"
    assert any(n.endswith("/configs") or "/configs/" in n for n in names), "configs/ missing"
