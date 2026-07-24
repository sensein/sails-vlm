"""Data-root resolution for SAILS configs.

Resolution order (schema rule: env var -> local config file -> error):
1. $SAILS_DATA_ROOT
2. key `data_root` in ./sails-vlm.yaml (git-ignored local file;
   see sails-vlm.example.yaml)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

LOCAL_CONFIG = "sails-vlm.yaml"
PLACEHOLDER = "${SAILS_DATA_ROOT}"


def data_root() -> Path:
    env = os.environ.get("SAILS_DATA_ROOT")
    if env:
        return Path(env)
    local = Path(LOCAL_CONFIG)
    if local.is_file():
        loaded = yaml.safe_load(local.read_text()) or {}
        if loaded.get("data_root"):
            return Path(loaded["data_root"])
    raise RuntimeError(
        "No data root configured. Either export SAILS_DATA_ROOT=/path/to/data "
        f"or create ./{LOCAL_CONFIG} with a `data_root:` key "
        "(see sails-vlm.example.yaml)."
    )


def interpolate(obj: Any) -> Any:
    """Recursively substitute the ${SAILS_DATA_ROOT} placeholder in strings."""
    if isinstance(obj, str):
        if PLACEHOLDER in obj:
            return obj.replace(PLACEHOLDER, str(data_root()))
        return obj
    if isinstance(obj, dict):
        return {k: interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate(v) for v in obj]
    return obj
