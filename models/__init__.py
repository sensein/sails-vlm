"""VLM model implementations and factory functions."""

from __future__ import annotations

from typing import Any, Dict

from .base_vlm import BaseVLM
from .ovis2 import Ovis2VLM
from .qwen2_5 import Qwen25VLM
from .internvl import InternVL
from .qwen3 import Qwen3
from .cosmos_reason import CosmosReason


def load_model(model_config: Dict[str, Any]) -> BaseVLM:
    """Factory used by runners.

    Expects cfg["model"] dict with at least: {"name": "..."}.
    """
    name = str(model_config.get("name", "")).lower().strip()

    if name == "ovis2":
        return Ovis2VLM(model_config)
    if name == "qwen2_5":
        return Qwen25VLM(model_config)
    if name == "internvl":
        return InternVL(model_config)
    if name == "qwen3":
        return Qwen3(model_config)
    if name in ("cosmos", "cosmos_reason", "cosmos-reason"):
        return CosmosReason(model_config)
    raise ValueError(f"Unknown model name: {name!r}. Available: ['ovis2', 'qwen2_5', 'internvl', 'qwen3', 'cosmos']")
