"""Registry behavior: lazy loading, internvl wired, helpful errors."""
import pytest

from sails_vlm.models import REGISTRY, load_model

EXPECTED_NAMES = {
    "ovis2", "qwen2_5", "cosmos", "cosmos_video", "qwen3", "qwen3_video", "internvl",
}


def test_registry_names():
    assert set(REGISTRY) == EXPECTED_NAMES


def test_unknown_model_error_lists_available():
    with pytest.raises(ValueError) as exc:
        load_model({"name": "nonexistent_model"})
    msg = str(exc.value)
    assert "nonexistent_model" in msg
    for name in EXPECTED_NAMES:
        assert name in msg


def test_registry_is_lazy():
    """Importing sails_vlm.models must not import any adapter module.

    Runs in a fresh interpreter so other tests' legitimate adapter imports
    (e.g. the contract tests) can't pollute the check.
    """
    import subprocess
    import sys

    code = (
        "import sys; import sails_vlm.models; "
        "leaked = [m for m in sys.modules "
        "if m.startswith('sails_vlm.models.') and m != 'sails_vlm.models.base_vlm']; "
        "assert not leaked, f'eagerly imported: {leaked}'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
