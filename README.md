# VLM Baseline Evaluation — Documentation

## Overview

This `vlm_baseline` folder provides a baseline framework for automatic annotation of videos using Video-Language Models (VLMs). The primary goal is to automate the manual annotation process currently performed on SAILS videos.

### Key Concepts

- **Automatic Annotation**: Videos that are currently manually annotated will be processed automatically using VLMs.
- **Annotation Types**:
  - **Classifications**: Categorical labels (e.g., gesture types)
  - **Descriptions**: Free-text descriptions of video content (e.g. activity)
- **Evaluation**: Different metrics are used for each annotation type to evaluate the VLM performances.
- **Inference Process**: Run VLM inference on all available videos and compare predictions against ground truth annotations.
- **Output Format**: Videos processed are from the BIDS folder, and evaluation results are saved to locations specified in the configuration file.

**Key Architecture Principle:**

- `models/` handles model interaction - if you want to try performances of a new VLM, you'll need to implement it here
- `postprocessing/` converts raw VLM output into task-specific prediction format
- `evaluation/` computes metrics comparing predictions vs. ground truth
- `runners/` orchestrates the entire pipeline (config loading, data iteration, output saving, evaluation)

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate qwen
```

### 2. Set up module imports

The runner uses `from vlm_baseline.models import ...` which requires a self-referential symlink at the repo root:

```bash
ln -s . vlm_baseline  # only needed once; already present if cloned from main
```

### 3. Download model weights

Models are loaded with `local_files_only: true`, so weights must be in the HuggingFace cache before running. Set `HF_HOME` if you want to use a shared cache location:

```bash
export HF_HOME=/path/to/shared/huggingface/cache
```

Then download the model you need (requires internet, do this outside the compute node):

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
huggingface-cli download nvidia/Cosmos-Reason2-8B
```

### 4. Update config paths

Config YAMLs contain paths specific to the original cluster environment (`/orcd/...`). Before running, update these fields in your config:

- `data.ground_truth_csv` — path to your validation CSV
- `data.video_dir` — path to the directory containing video clips
- `output.save_dir` — where to write predictions and metrics

## How to Run

Build a srun session with a GPU, then from the repo root:

```bash
conda activate qwen
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python -m runners.run_prediction configs/qwen3/rmm.yaml
```

Or use the provided SLURM scripts. **Submit from the repo root** — log paths
(`logs/<job>_%j.out`) are relative to the submission directory:

```bash
CONFIG=configs/qwen3/rmm.yaml sbatch scripts/qwen3_rmm.sh
```

`CONFIG` (which config to run) and `CONDA_ENV` (which conda env to activate,
default `qwen`) are read from the shell environment at runtime, so they can be
set on the command line as shown.

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

## Models (models/)

This folder contains thin wrappers around VLM backends (Ovis2, Qwen2.5, …).
It loads the model, runs inference on a video + prompt, returns raw generated text

## Postprocessing (postprocessing/)

Postprocessing converts raw model output into the prediction type expected by the task. It then validates the postprocessed output

## Evaluation (evaluation/)

Evaluation metrics depend on `task.type`. For free text tasks, we haven't any metrics implemented yet.

### Classification Evaluation

Common metrics include:
- **Accuracy** (though not always most relevant for unbalanced datasets)
- Macro-F1 / Weighted-F1
- Per-class precision/recall/F1
- Confusion matrix

**Inputs**: Ground truth labels from CSV vs. postprocessed predictions

### How to add a new model

## How to Add a New Model

To integrate a new VLM into the baseline framework, follow these steps:

### 1. Create Model Wrapper

Create a new file `models/<new_model>.py` with a class that inherits from `BaseVLM`:

```python
class NewModelVLM(BaseVLM):
    def load(self):
        # Load weights/processor, set device, eval mode
        pass

    def generate(self, video_path, prompt, video_cfg=None, gen_cfg=None):
        # Implement inference logic
        # Return VLMRawOutput
        pass

    # Usually no need to override predict()
```

### 2. Register the Model

Update `models/__init__.py`:

- Import your new class
- Add a case in the `load_model()` function for your model's `config["name"]`

### 3. Create Configuration

Add a config YAML file under `configs/<new_model>/...yaml` with at least the annotation description, prompt etc,... and for the model configuration:

```yaml
model:
  name: "your_model_name"
  model_path: "HF_repo_id"  # or local path
  device: "cuda"
  precision: "bf16"

```

### 4. Test the Integration

Run your existing runner with the new config:

```bash
python -m runners.run_prediction configs/<new_model>/your_config.yaml
```

**Note**: Downstream postprocessing automatically determines whether it's a classification or free-text task based on the configuration.
