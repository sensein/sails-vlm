"""Main runner script for VLM baseline evaluation."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from tqdm.auto import tqdm
from vlm_baseline.evaluation.metrics import (
    evaluate_classification,
    evaluate_description,
)
from vlm_baseline.models import load_model
from vlm_baseline.postprocessing.validation import validate_classification_output

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

    # ---------------------------
    # Output dir (add run tag)
    # ---------------------------
    run_id = f"{now_tag()}"
    out_root = Path(cfg["output"]["save_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load annotation CSV
    # ---------------------------
    gt_file = cfg["data"]["ground_truth_csv"]
    df = pd.read_csv(gt_file)

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
    model = load_model(cfg["model"])
    if hasattr(model, "load"):
        model.load()

    # ---------------------------
    # Task setup
    # ---------------------------
    task_type = cfg["task"]["type"]
    metrics_cfg = cfg.get("evaluation", {}).get("metrics", [])

    if video_col is None or video_col not in df.columns:
        raise ValueError("data.video_path_column must be present in the CSV.")

    if label_col is None or label_col not in df.columns:
        raise ValueError(
            "data.label_column must be present in the CSV for both "
            "classification and description tasks."
        )

    allowed_labels: List[str] = []
    if task_type == "classification":
        allowed_labels = list(cfg["task"]["labels"])
        df[label_col] = df[label_col].astype(object).where(df[label_col].notna(), "NaN")
    else:
        df[label_col] = df[label_col].astype(object).where(df[label_col].notna(), "")

    preds_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    y_true: List[str] = []
    y_pred: List[str] = []

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
            raw = model.predict(str(video_path), prompt)
        except Exception as e:
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
            pred_label, dbg = validate_classification_output(
                raw_output=str(raw),
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
            y_pred.append(str(pred_label))

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
                f"Unknown task.type '{task_type}'. Expected 'classification' "
                "or 'description'."
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
            y_pred=y_pred,
            labels=allowed_labels,
            metrics=metrics_cfg,
            invalid_label=INVALID_LABEL,
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

    print(f"✅ Experiment completed successfully. Saved to: {out_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m runners.run_experiment path/to/config.yaml")
    main(sys.argv[1])
