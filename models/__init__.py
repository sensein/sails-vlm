"""VLM model implementations and factory functions."""

from __future__ import annotations

from typing import Any, Dict

from .ovis2 import Ovis2VLM


def load_model(model_config: Dict[str, Any]) -> Ovis2VLM:
    """Factory used by runners.

    Expects cfg["model"] dict with at least: {"name": "..."}.
    """
    name = str(model_config.get("name", "")).lower().strip()

    if name == "ovis2":
        return Ovis2VLM(model_config)

    raise ValueError(f"Unknown model name: {name!r}. Available: ['ovis2']")
