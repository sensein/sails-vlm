"""VLM model implementations and factory functions."""

from __future__ import annotations

from typing import Any, Dict

from .base_vlm import BaseVLM, VLMRawOutput
from .cosmos import CosmosReason2VLM
from .cosmos_video import CosmosVideo
from .ovis2 import Ovis2VLM
from .qwen2_5 import Qwen25VLM
from .qwen3 import Qwen3
from .qwen3_video import Qwen3Video


def load_model(model_config: Dict[str, Any]) -> BaseVLM:
    """Factory used by runners.

    Expects cfg["model"] dict with at least: {"name": "..."}.
    """
    name = str(model_config.get("name", "")).lower().strip()

    if name == "ovis2":
        return Ovis2VLM(model_config)
    if name == "qwen2_5":
        return Qwen25VLM(model_config)
    if name == "cosmos":
        return CosmosReason2VLM(model_config)
    if name == "qwen3":
        return Qwen3(model_config)
    if name == "qwen3_video":
        return Qwen3Video(model_config)
    if name == "cosmos_video":
        return CosmosVideo(model_config)
    raise ValueError(
        f"Unknown model name: {name!r}. Available: ['ovis2', 'qwen2_5', 'cosmos', 'cosmos_video', 'qwen3', 'qwen3_video']"
    )
