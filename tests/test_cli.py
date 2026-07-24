"""CLI wiring: --help works without GPU, data root, or model deps."""
import subprocess
import sys


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "sails_vlm.runners.run_prediction", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "config" in result.stdout.lower()
