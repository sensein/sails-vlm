"""VLM model implementations and factory functions."""

from __future__ import annotations

import importlib
from typing import Any, Dict

from .base_vlm import BaseVLM, VLMRawOutput

# name -> (submodule, class). Adapters import lazily so that installing only
# one family's extra is enough to use that family.
REGISTRY: Dict[str, tuple[str, str]] = {
    "ovis2": ("ovis2", "Ovis2VLM"),
    "qwen2_5": ("qwen2_5", "Qwen25VLM"),
    "cosmos": ("cosmos", "CosmosReason2VLM"),
    "cosmos_video": ("cosmos_video", "CosmosVideo"),
    "qwen3": ("qwen3", "Qwen3"),
    "qwen3_video": ("qwen3_video", "Qwen3Video"),
    "internvl": ("internvl", "InternVL"),
}


def load_model(model_config: Dict[str, Any]) -> BaseVLM:
    """Factory used by runners.

    Expects cfg["model"] dict with at least: {"name": "..."}.
    """
    name = str(model_config.get("name", "")).lower().strip()
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model name: {name!r}. Available: {sorted(REGISTRY)}"
        )
    submodule, class_name = REGISTRY[name]
    module = importlib.import_module(f".{submodule}", __package__)
    return getattr(module, class_name)(model_config)


__all__ = ["BaseVLM", "VLMRawOutput", "REGISTRY", "load_model"]
