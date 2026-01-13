"""Evaluation metrics for annotation tasks.

This module contains evaluation functions for classification and free-text
annotation tasks in the VLM baseline framework.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
