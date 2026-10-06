"""Shipped classification configs must headline balanced_accuracy (issue #3).

Lab convention 2026-07-24: SAILS classes are imbalanced, so balanced accuracy
(= macro recall), not raw accuracy, is the headline classification metric.
"""
from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _classification_configs():
    paths = sorted(
        list(CONFIGS_DIR.rglob("*.yaml")) + list(CONFIGS_DIR.rglob("*.yml"))
    )
    out = []
    for p in paths:
        cfg = yaml.safe_load(p.read_text())
        if (
            isinstance(cfg, dict)
            and cfg.get("task", {}).get("type") == "classification"
        ):
            out.append(p)
    return out


def test_expected_classification_config_count():
    """Guards the discovery itself (glob/extension bugs would silently shrink
    coverage). Update the number consciously when configs are added/removed."""
    assert len(_classification_configs()) == 53


@pytest.mark.parametrize(
    "path",
    _classification_configs(),
    ids=lambda p: str(p.relative_to(CONFIGS_DIR)),
)
def test_balanced_accuracy_headlines(path):
    cfg = yaml.safe_load(path.read_text())
    metrics = cfg.get("evaluation", {}).get("metrics") or []
    assert metrics and metrics[0] == "balanced_accuracy", (
        f"{path.name}: metrics starts with {metrics[:1]!r}, "
        "expected balanced_accuracy"
    )
