"""Evaluation metrics for annotation tasks.

This module contains evaluation functions for classification and free-text
annotation tasks in the VLM baseline framework.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate_classification(
    y_true: List[str],
    y_pred: List[str],  # top1 only
    metrics: List[str],
    labels: List[str],
    binary: bool,
    y_pred_top2: Optional[List[str]] = None,  # <-- NEW
    invalid_label: str = "INVALID",
) -> Dict[str, Any]:
    """Evaluate classification predictions."""
    out: Dict[str, Any] = {}
    out["n"] = len(y_true)
    out["invalid_rate"] = sum(1 for p in y_pred if p == invalid_label) / max(
        1, len(y_pred)
    )

    if "accuracy" in metrics:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))

    if "top2_accuracy" in metrics and not binary:
        if not y_pred_top2:
            out["top2_accuracy"] = 0.0
        else:
            correct_top2 = 0
            denom = 0
            for true, raw in zip(y_true, y_pred_top2):
                if raw == invalid_label:
                    continue
                pred_labels = [p for p in str(raw).split("|") if p]
                if not pred_labels:
                    continue
                denom += 1
                if true in pred_labels[:2]:
                    correct_top2 += 1
            out["top2_accuracy"] = float(correct_top2 / max(1, denom))

    if "f1_macro" in metrics:
        out["f1_macro"] = float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        )
    ...
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


def evaluate_counting(
    y_true: List[int],
    y_pred: List[Optional[int]],
    metrics: List[str],
) -> Dict[str, Any]:
    """Evaluate counting predictions against ground truth.

    Computes regression-style metrics for integer counting tasks.

    Args:
        y_true: Ground truth counts (integers).
        y_pred: Predicted counts (integers or None for invalid predictions).
        metrics: List of metrics to compute. Options:
            - "mae": Mean Absolute Error
            - "rmse": Root Mean Squared Error
            - "exact_match": Percentage of exact matches
            - "off_by_one": Percentage within +/- 1 of ground truth

    Returns:
        Dictionary containing computed metrics.
    """
    out: Dict[str, Any] = {}

    out["n"] = len(y_true)

    # Count invalid predictions (None values)
    invalid_count = sum(1 for p in y_pred if p is None)
    out["invalid_count"] = invalid_count
    out["invalid_rate"] = invalid_count / max(1, len(y_pred))

    # Filter to valid predictions only for metric computation
    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred) if p is not None
    ]

    if not valid_pairs:
        # No valid predictions to evaluate
        out["valid_n"] = 0
        if "mae" in metrics:
            out["mae"] = None
        if "rmse" in metrics:
            out["rmse"] = None
        if "exact_match" in metrics:
            out["exact_match"] = None
        if "off_by_one" in metrics:
            out["off_by_one"] = None
        return out

    valid_true = [t for t, p in valid_pairs]
    valid_pred = [p for t, p in valid_pairs]
    out["valid_n"] = len(valid_pairs)

    if "mae" in metrics:
        # Mean Absolute Error
        mae = sum(abs(t - p) for t, p in valid_pairs) / len(valid_pairs)
        out["mae"] = float(mae)

    if "rmse" in metrics:
        # Root Mean Squared Error
        mse = sum((t - p) ** 2 for t, p in valid_pairs) / len(valid_pairs)
        out["rmse"] = float(math.sqrt(mse))

    if "exact_match" in metrics:
        # Percentage of exact matches
        exact = sum(1 for t, p in valid_pairs if t == p)
        out["exact_match"] = float(exact / len(valid_pairs))

    if "off_by_one" in metrics:
        # Percentage within +/- 1 of ground truth
        within_one = sum(1 for t, p in valid_pairs if abs(t - p) <= 1)
        out["off_by_one"] = float(within_one / len(valid_pairs))

    return out
