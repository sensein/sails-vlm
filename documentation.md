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

## How to Run

Build a srun session with a gpu, then from the repo root, run:

```bash
poetry run python vlm_baseline/runners/run_prediction.py vlm_baseline/configs/ovis2/response_to_name.yaml
```


## Configuration File (YAML)

A config defines one complete experiment (one model + one task + one dataset + one prompt + one output directory). If you want to try a vlm on a particular annotation prediction, feel free to create a new configuration file with the same structure as the ones already present.

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
poetry run python vlm_baseline/runners/run_prediction.py vlm_baseline/configs/<new_model>/your_config.yaml
```

**Note**: Downstream postprocessing automatically determines whether it's a classification or free-text task based on the configuration.
