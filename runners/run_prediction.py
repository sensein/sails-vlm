"""Main runner script for VLM baseline evaluation."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from tqdm.auto import tqdm
from vlm_baseline.evaluation.metrics import (
    evaluate_classification,
    evaluate_counting,
    evaluate_description,
)
from vlm_baseline.models import load_model
from vlm_baseline.postprocessing.validation import (
    validate_classification_output,
    validate_counting_output,
)

INVALID_LABEL = "INVALID"


def now_tag() -> str:
    """Generate a timestamp tag for the current run."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def normalize_space(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", str(s).strip())


def main(config_path: str) -> None:
    """Run the VLM baseline evaluation pipeline."""
    # ---------------------------
    # Load config
    # ---------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    task_type = str(cfg["task"]["type"]).lower().strip()

    # Force unbuffered output for immediate visibility in SLURM logs
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f"Starting experiment: {exp_name}", flush=True)
    print(f"Task type: {task_type}", flush=True)

    # ---------------------------
    # Output dir (add run tag)
    # ---------------------------
    run_id = f"{now_tag()}"
    out_root = Path(cfg["output"]["save_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}", flush=True)

    # ---------------------------
    # Load annotation CSV
    # ---------------------------
    print("Loading ground truth CSV...", flush=True)
    gt_file = cfg["data"]["ground_truth_csv"]
    df = pd.read_csv(gt_file)
    print(f"   Loaded {len(df)} rows from {gt_file}", flush=True)

    video_col = cfg["data"]["video_path_column"]
    label_col = cfg["data"].get("label_column")

    if video_col not in df.columns:
        raise ValueError(
            f"CSV missing video_path_column '{video_col}'. Columns: {list(df.columns)}"
        )

    # ---------------------------
    # Prompt
    # ---------------------------
    prompt = cfg["prompt"]["message"]

    # ---------------------------
    # Load model
    # ---------------------------
    print(f"Loading model: {cfg['model'].get('name', 'unknown')}...", flush=True)
    model = load_model(cfg["model"])
    print("loading model :", model)
    if hasattr(model, "load"):
        model.load()
    print("Model loaded successfully", flush=True)

    # ---------------------------
    # Task setup
    # ---------------------------
    metrics_cfg = cfg.get("evaluation", {}).get("metrics", [])

    if video_col is None or video_col not in df.columns:
        raise ValueError("data.video_path_column must be present in the CSV.")

    if label_col is None or label_col not in df.columns:
        raise ValueError(
            "data.label_column must be present in the CSV for both "
            "classification and description tasks."
        )

    # ---------------------------
    # Optionally drop rows with missing ground-truth labels
    # ---------------------------
    drop_missing_labels = bool(cfg.get("data", {}).get("drop_missing_labels", False))
    dropped_missing_labels = 0
    if drop_missing_labels:
        before = len(df)
        df = df[df[label_col].notna()].copy()
        dropped_missing_labels = before - len(df)

    # ---------------------------
    # Optionally limit number of samples for testing
    # ---------------------------
    max_samples = cfg.get("data", {}).get("max_samples", None)
    if max_samples is not None and max_samples > 0:
        df = df.head(int(max_samples)).copy()
        print(f"Limiting to first {max_samples} samples for testing")

    allowed_labels: List[str] = []
    if task_type == "classification":
        allowed_labels = list(cfg["task"]["labels"])
        df[label_col] = df[label_col].astype(object).where(df[label_col].notna(), "NaN")
    elif task_type == "counting":
        # For counting tasks, convert ground truth to integers
        # Handle values like "10+", "5+" by extracting the numeric part
        def parse_count_label(val):
            if pd.isna(val):
                return None
            val_str = str(val).strip()
            match = re.match(r"(\d+)", val_str)
            if match:
                return int(match.group(1))
            return None
        df[label_col] = df[label_col].apply(parse_count_label)
    else:
        df[label_col] = df[label_col].astype(object).where(df[label_col].notna(), "")

    preds_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    # For classification/description tasks
    y_true: List[str] = []
    y_pred_top1: List[str] = []
    y_pred_top2raw: List[str] = []

    # For counting tasks
    y_true_counts: List[int] = []
    y_pred_counts: List[Optional[int]] = []

    print(f"Experiment: {exp_name}")
    print(f"Task: {task_type}")
    print(f"Labels: {allowed_labels}")
    print(
        "Number of Samples with Ground-Truth Labels (non-NaN):"
        f" {len(df[df[label_col].notna()])}"
    )
    n_missing = (
        dropped_missing_labels if drop_missing_labels else df[label_col].isna().sum()
    )

    print(f"Number of Samples with Missing Ground-Truth Labels (NaN): {n_missing}")
    print(f"Missing Ground-Truth Labels Dropped?: {drop_missing_labels}")

    # ---------------------------
    # Progress bar counters
    # ---------------------------
    skipped_not_found = 0
    predict_errors = 0
    invalid_preds = 0

    # ---------------------------
    # Run inference (with progress bar)
    # ---------------------------
    iterator = df.iterrows()

    pbar = tqdm(
        iterator,
        total=len(df),
        desc="Processing videos",
        unit="video",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for i, row in pbar:
        video_path = row[video_col]
        gt = row[label_col]

        if not isinstance(video_path, str) or not Path(video_path).exists():
            skipped_not_found += 1
            debug_rows.append(
                {
                    "index": int(i),
                    "video_path": str(video_path),
                    "error": "video_not_found",
                }
            )
            pbar.set_postfix(
                skipped=skipped_not_found, errors=predict_errors, invalid=invalid_preds
            )
            continue

        try:
            raw = model.predict(str(video_path), prompt, allowed_labels)
            raw_top1 = str(raw).split("|", 1)[0]
        except Exception as e:
            import traceback
            print(f"Error during prediction: {e}")
            print(traceback.format_exc())
            raw = ""
            predict_errors += 1
            debug_rows.append(
                {
                    "index": int(i),
                    "video_path": str(video_path),
                    "error": f"predict_exception: {repr(e)}",
                }
            )
            pbar.set_postfix(
                skipped=skipped_not_found, errors=predict_errors, invalid=invalid_preds
            )
            continue

        if task_type == "classification":
            y_pred_top2raw.append(str(raw))
            pred_label, dbg = validate_classification_output(
                raw_output=str(raw_top1),
                allowed_labels=allowed_labels,
                invalid_label=INVALID_LABEL,
            )
            if pred_label is None:
                pred_label = INVALID_LABEL

            if str(pred_label) == INVALID_LABEL:
                invalid_preds += 1

            dbg.update({"index": int(i), "video_path": str(video_path)})
            debug_rows.append(dbg)

            preds_rows.append(
                {
                    "index": int(i),
                    "video_path": str(video_path),
                    "ground_truth": str(gt),
                    "raw_prediction": raw,
                    "prediction": str(pred_label),
                }
            )

            y_true.append(str(gt))
            y_pred_top1.append(str(pred_label))

        elif task_type == "counting":
            pred_count, dbg = validate_counting_output(raw_output=str(raw))

            if pred_count is None:
                invalid_preds += 1

            dbg.update({"index": int(i), "video_path": str(video_path)})
            debug_rows.append(dbg)

            # Convert ground truth to int (may be float from pandas)
            gt_int = int(gt) if pd.notna(gt) else None

            preds_rows.append(
                {
                    "index": int(i),
                    "video_path": str(video_path),
                    "ground_truth": gt_int,
                    "raw_prediction": raw,
                    "prediction": pred_count,
                }
            )

            if gt_int is not None:
                y_true_counts.append(gt_int)
                y_pred_counts.append(pred_count)

        elif task_type == "description":
            pred_text = normalize_space(str(raw))
            preds_rows.append(
                {
                    "index": int(i),
                    "video_path": str(video_path),
                    "ground_truth": str(gt),
                    "raw_prediction": raw,
                    "prediction": pred_text,
                }
            )
        else:
            raise ValueError(
                f"Unknown task.type '{task_type}'. Expected 'classification', "
                "'counting', or 'description'."
            )

        pbar.set_postfix(
            skipped=skipped_not_found, errors=predict_errors, invalid=invalid_preds
        )

    # ---------------------------
    # Make a single predictions DF for saving + for description evaluation
    # ---------------------------
    pred_df = pd.DataFrame(preds_rows)

    # ---------------------------
    # Evaluation
    # ---------------------------
    if task_type == "classification":
        metrics = evaluate_classification(
            y_true=y_true,
            y_pred=y_pred_top1,
            y_pred_top2=y_pred_top2raw,
            labels=allowed_labels,
            metrics=metrics_cfg,
            invalid_label=INVALID_LABEL,
            binary=False,
        )
    elif task_type == "counting":
        metrics = evaluate_counting(
            y_true=y_true_counts,
            y_pred=y_pred_counts,
            metrics=metrics_cfg,
        )
    elif task_type == "description":
        metrics = evaluate_description(
            predictions_df=pred_df,
            metrics=metrics_cfg,
            cfg=cfg,
        )

    # ---------------------------
    # Save artifacts
    # ---------------------------
    debug_df = pd.DataFrame(debug_rows) if debug_rows else pd.DataFrame()

    results = {
        "experiment": exp_name,
        "run_id": run_id,
        "model": cfg["model"]["name"],
        "task": task_type,
        "num_samples": int(len(pred_df)),
        "dropped_missing_labels": int(dropped_missing_labels),
        "drop_missing_labels_enabled": bool(drop_missing_labels),
        "metrics": metrics,
        "files": {
            "predictions_csv": str(out_dir / "predictions.csv"),
            "debug_csv": str(out_dir / "debug.csv"),
            "results_json": str(out_dir / "results.json"),
            "config_used": str(out_dir / "config_used.yaml"),
        },
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    if cfg["output"].get("save_predictions", True):
        pred_df.to_csv(out_dir / "predictions.csv", index=False)

    debug_df.to_csv(out_dir / "debug.csv", index=False)

    print(f"\nExperiment completed successfully. Saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m runners.run_experiment path/to/config.yaml")
    main(sys.argv[1])
