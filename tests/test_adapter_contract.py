"""Every registered adapter must satisfy the BaseVLM contract.

Family extras may be absent in a dev env: importorskip skips what isn't
installed, but registry integrity (module file exists, class name resolves in
source) is checked for ALL adapters regardless.
"""
import ast
import importlib
from pathlib import Path

import pytest

from sails_vlm.models import REGISTRY
from sails_vlm.models.base_vlm import BaseVLM

MODELS_DIR = Path(__file__).resolve().parent.parent / "sails_vlm" / "models"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_adapter_module_file_exists(name):
    submodule, _ = REGISTRY[name]
    assert (MODELS_DIR / f"{submodule}.py").is_file()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_adapter_class_defined_in_source(name):
    """Class exists in the module source — checked via AST so it works even
    when the family's deps aren't installed."""
    submodule, class_name = REGISTRY[name]
    tree = ast.parse((MODELS_DIR / f"{submodule}.py").read_text())
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert class_name in classes


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_adapter_subclasses_basevlm_when_importable(name):
    submodule, class_name = REGISTRY[name]
    try:
        module = importlib.import_module(f"sails_vlm.models.{submodule}")
    except ImportError as e:
        pytest.skip(f"family deps not installed: {e}")
    cls = getattr(module, class_name)
    assert issubclass(cls, BaseVLM)
    assert not getattr(cls, "__abstractmethods__", None), (
        f"{class_name} leaves abstract methods unimplemented"
    )


def test_every_config_family_is_registered():
    """No orphaned config dirs: each configs/<family>/ maps to a registry name
    (prompt_ablation subdirs and non-model dirs excluded by the known set)."""
    configs = Path(__file__).resolve().parent.parent / "configs"
    family_dirs = {p.name for p in configs.iterdir() if p.is_dir()}
    known_non_model = set()  # add here if a non-model config dir is ever created
    assert family_dirs - known_non_model <= set(REGISTRY), (
        f"config dirs without a registered model: {family_dirs - set(REGISTRY)}"
    )
