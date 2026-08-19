# VLM Baseline Evaluation — Documentation

## Overview

The `sails_vlm` package provides a baseline framework for automatic annotation of videos using Video-Language Models (VLMs). The primary goal is to automate the manual annotation process currently performed on SAILS videos.

### Key Concepts

- **Automatic Annotation**: Videos that are currently manually annotated will be processed automatically using VLMs.
- **Annotation Types**:
  - **Classifications**: Categorical labels (e.g., gesture types)
  - **Descriptions**: Free-text descriptions of video content (e.g. activity)
- **Evaluation**: Different metrics are used for each annotation type to evaluate the VLM performances.
- **Inference Process**: Run VLM inference on all available videos and compare predictions against ground truth annotations.
- **Output Format**: Videos processed are from the BIDS folder, and evaluation results are saved to locations specified in the configuration file.

**Key Architecture Principle:**

- `sails_vlm/models/` handles model interaction - if you want to try performances of a new VLM, you'll need to implement it here (see `docs/adding-a-model.md`)
- `sails_vlm/postprocessing/` converts raw VLM output into task-specific prediction format
- `sails_vlm/evaluation/` computes metrics comparing predictions vs. ground truth
- `sails_vlm/runners/` orchestrates the entire pipeline (config loading, data iteration, output saving, evaluation)

## Setup

```bash
git clone <repo-url> && cd sails-vlm
uv sync                          # core (no model families)
uv sync --extra qwen             # + the family you plan to run
export SAILS_DATA_ROOT=/orcd/data/satra/002/projects/SAILS   # or copy sails-vlm.example.yaml -> sails-vlm.yaml
```

| extra | models | pins |
|---|---|---|
| `qwen` | qwen2_5, qwen3, qwen3_video | transformers (exact pin), qwen-vl-utils, accelerate (`device_map`), bitsandbytes (4-bit quantization — shipped qwen3 configs set `load_in_4bit: true`) |
| `cosmos` | cosmos, cosmos_video | transformers (exact pin), accelerate (`device_map`) |
| `ovis2` | ovis2 | transformers (exact pin) — this adapter uses neither `device_map` nor quantization |
| `internvl` | internvl | transformers (exact pin), bitsandbytes (4-bit quantization — shipped configs set `load_in_4bit: true`), accelerate (`device_map`) |
| `text-metrics` | free-text eval (BLEU/ROUGE/semantic) | rouge, nltk, sentence-transformers, scipy |

`accelerate` is a per-family extra, not a core dependency: `uv sync` alone
installs neither it nor `bitsandbytes`, so pick the extra for the family you
plan to run.

Model weights must be pre-cached in the HuggingFace cache before running
(`local_files_only: true`). Set `HF_HOME` if you want to use a shared cache
location, then download the model you need (requires internet, do this
outside the compute node):

```bash
export HF_HOME=/path/to/shared/huggingface/cache
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
huggingface-cli download nvidia/Cosmos-Reason2-8B
```

Description-task semantic metrics also need the embedding model cached, since
evaluation scripts export `HF_HUB_OFFLINE=1` and will fail to fetch it on
demand:

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

Config YAMLs use `${SAILS_DATA_ROOT}` for data/output paths — set the
environment variable (or `sails-vlm.yaml`, see above) before running. The
`./sails-vlm.yaml` fallback is resolved relative to the current working
directory, so run from the repo root, or use the environment variable instead.

## Run

```bash
uv run sails-vlm-predict configs/qwen3/rmm.yaml
```

Runs are **not deterministic by default** (`do_sample: true`); set
`experiment.seed` in the config for typically repeatable sampling on identical
hardware (GPU kernels are not guaranteed deterministic). Model weights must be
pre-cached (`local_files_only: true`).

Build a srun session with a GPU, then from the repo root, or use the provided
SLURM scripts. **Submit from the repo root** — log paths (`logs/<job>_%j.out`)
are relative to the submission directory:

```bash
CONFIG=configs/qwen3/rmm.yaml sbatch scripts/qwen3_rmm.sh
```

`CONFIG` (which config to run) is read from the shell environment at runtime,
so it can be set on the command line as shown. The scripts invoke
`uv run sails-vlm-predict` internally, so no environment activation step is
needed.

The partition and log paths live in `#SBATCH` directives, which sbatch does
**not** shell-expand. To change them, edit the script or override on the
`sbatch` command line (these take precedence over the directives):

```bash
sbatch -p gpu -o ~/logs/qwen3_%j.out scripts/qwen3_rmm.sh
```


## Configuration File (YAML)

A config defines one complete experiment (one model + one task + one dataset + one prompt + one output directory). If you want to try a vlm on a particular annotation prediction, feel free to create a new configuration file with the same structure as the ones already present.

### Handling YAML booleans

YAML treats the following as boolean values:
 - no
 - yes
 - on
 - off

If these are being used as ground-truth labels, ensure they are enclosed by single or double quotes. Otherwise they will be converted to their boolean counterparts and produce unexpected behavior.

### Handling missing / NaN labels in the ground-truth CSV

Some label columns in the annotation CSV contain missing values (pandas `NaN`).

- For **classification** tasks, the runner normally converts missing values to the
  literal string `"NaN"` internally.
- For **description** tasks, the runner converts missing values to the empty string `""`.

If you want to **exclude unlabeled rows entirely**,
set the following in your config:

```yaml
data:
  drop_missing_labels: true
```

When enabled, rows with missing ground-truth labels are removed **before any VLM
inference** (those videos are not processed and do not appear in predictions/eval).

## Models (sails_vlm/models/)

This folder contains thin wrappers around VLM backends (Ovis2, Qwen2.5, …).
It loads the model, runs inference on a video + prompt, returns raw generated text

## Postprocessing (sails_vlm/postprocessing/)

Postprocessing converts raw model output into the prediction type expected by the task. It then validates the postprocessed output

## Evaluation (sails_vlm/evaluation/)

Evaluation metrics depend on `task.type`. For free text tasks, see the `text-metrics` extra (BLEU/ROUGE/semantic similarity).

### Classification Evaluation

Common metrics include:

#### Headline metric: balanced accuracy

SAILS annotation classes are imbalanced, so raw accuracy is inflated by the
majority class. The headline metric for any classification result is
`balanced_accuracy` (mathematically identical to macro-averaged recall,
`recall_macro`) or macro-F1, always alongside per-class recall/F1 (or a
confusion matrix) and class prevalence. Note: `balanced_accuracy` is averaged
over the *configured* label set with `zero_division=0`, so a label that never
appears in the ground truth of a given split (e.g., a CV fold missing a class)
counts as recall 0 — deflating the value relative to sklearn's
`balanced_accuracy_score`, which averages only over classes present in
`y_true`. `accuracy` may appear only as a
secondary number — never alone, and never as the basis of a model
comparison. All shipped classification configs list `balanced_accuracy`
first, and `evaluate_classification` emits it as the first metric field in
`results.json`.

- **Accuracy** (though not always most relevant for unbalanced datasets)
- Macro-F1 / Weighted-F1
- Per-class precision/recall/F1
- Confusion matrix

**Inputs**: Ground truth labels from CSV vs. postprocessed predictions

### How to add a new model

See `docs/adding-a-model.md` for the full onboarding checklist (adapter,
registry entry, dependency extra, config — and the contract tests that
verify all four are wired up consistently).
