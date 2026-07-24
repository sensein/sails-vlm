# Adding a new VLM (onboarding checklist)

Adding a model touches exactly four places. Nothing in runners/,
postprocessing/, or evaluation/ changes, and no existing family's pins move.

1. **Adapter** — `sails_vlm/models/<family>.py`: subclass
   `BaseVLM` (`sails_vlm/models/base_vlm.py`), implement `load()` and
   `generate() -> VLMRawOutput`. `predict()` is inherited.
2. **Registry** — one line in `REGISTRY` in `sails_vlm/models/__init__.py`:
   `"<name>": ("<family>", "<ClassName>"),`
3. **Dependencies** — a new extra in `pyproject.toml` with EXACT pins for the
   family's needs (its own transformers pin if it needs a newer one). If its
   pins cannot coexist with an existing extra, declare it under
   `[tool.uv] conflicts`. Run `uv lock`.
4. **Config** — `configs/<name>/<task>.yaml` using `${SAILS_DATA_ROOT}` for
   all data/output paths, plus `experiment.seed` if you need repeatability.

Verify before committing:

```bash
uv run pytest tests/test_registry.py tests/test_adapter_contract.py -v
```

The contract tests will fail if: the registry name has no module file, the
class name doesn't exist in the module source, the adapter doesn't subclass
BaseVLM (when its deps are installed), or a `configs/<dir>` has no registry
entry (the historical internvl bug).
