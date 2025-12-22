"""Evaluation metrics for annotation tasks.

This module contains evaluation functions for classification and free-text
annotation tasks in the VLM baseline framework.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate_classification(
    y_true: List[str],
    y_pred: List[str],
    metrics: List[str],
    labels: List[str],
    invalid_label: str = "INVALID",
) -> Dict[str, Any]:
    """Evaluate classification predictions against ground truth.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metrics: List of metrics to compute
        labels: List of valid label classes
        invalid_label: Label used for invalid predictions

    Returns:
        Dictionary containing computed metrics
    """
    out: Dict[str, Any] = {}

    out["n"] = len(y_true)
    out["invalid_rate"] = sum(1 for p in y_pred if p == invalid_label) / max(
        1, len(y_pred)
    )

    if "accuracy" in metrics:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))

    if "f1_macro" in metrics:
        out["f1_macro"] = float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        )

    if "f1_weighted" in metrics:
        out["f1_weighted"] = float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        )

    if "report" in metrics:
        out["report"] = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

    return out


def evaluate_description(
    predictions_df: pd.DataFrame,
    metrics: List[str],
    cfg: dict | None = None,
) -> Dict[str, Any]:
    """Evaluate free-text description predictions.

    Placeholder: your free-text eval likely uses BLEU/ROUGE/BERTScore,
    or LLM-as-judge, or keyword-based scoring, etc.
    Implement your own logic here.

    Args:
        predictions_df: DataFrame containing predictions
        metrics: List of metrics to compute
        cfg: Optional configuration dictionary

    Returns:
        Dictionary containing computed metrics
    """
    out: Dict[str, Any] = {"n": int(len(predictions_df))}

    # Example: just track average length
    if "avg_len" in metrics:
        out["avg_len"] = float(predictions_df["prediction"].fillna("").map(len).mean())

    return out
