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
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.spatial.distance import cosine


def evaluate_classification(
    y_true: List[str],
    y_pred: List[str],  # top1 only
    metrics: List[str],
    labels: List[str],
    binary: bool,
    y_pred_top2: Optional[List[str]] = None,
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
    
    return out


def evaluate_description(
    predictions: List[str],
    ground_truths: List[str],
    metrics: List[str],
    embedding_model_name: str = 'all-MiniLM-L6-v2',
) -> Dict[str, Any]:
    """Evaluate free-text description predictions.

    Computes lexical metrics (BLEU, ROUGE, word overlap) and semantic
    similarity metrics (cosine similarity, euclidean distance) for 
    free-text descriptions.

    Args:
        predictions: List of predicted descriptions
        ground_truths: List of ground truth descriptions
        metrics: List of metrics to compute. Options:
            - "bleu": BLEU score
            - "rouge1", "rouge2", "rougeL": ROUGE scores
            - "word_overlap": Word-level overlap ratio
            - "cosine_similarity": Semantic similarity using embeddings
            - "euclidean_distance": Embedding distance (lower is better)
            - "dot_product": Embedding dot product
            - "avg_len": Average prediction length
        embedding_model_name: Name of sentence-transformers model

    Returns:
        Dictionary containing computed metrics
    """
    out: Dict[str, Any] = {}
    
    # Filter valid pairs - handle pandas NA properly
    valid_pairs = []
    for pred, truth in zip(predictions, ground_truths):
        # Skip if either is actually NA/NaN
        if pd.isna(pred) or pd.isna(truth):
            continue
            
        # Convert to string
        pred_str = str(pred).strip()
        truth_str = str(truth).strip()
        
        # Skip only if empty string or the string literal "nan"
        # Keep "none" as it's a valid category value
        if (pred_str == '' or truth_str == '' or 
            pred_str.lower() == 'nan' or truth_str.lower() == 'nan'):
            continue
            
        valid_pairs.append((pred_str, truth_str))
    
    out["n"] = len(predictions)
    out["valid_n"] = len(valid_pairs)
    
    if not valid_pairs:
        # Return zeros for all metrics if no valid pairs
        return {
            'n': len(predictions),
            'valid_n': 0,
            'bleu': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'word_overlap': 0.0,
            'cosine_similarity': 0.0,
            'euclidean_distance': 0.0,
            'dot_product': 0.0,
            'avg_len': 0.0,
        }
    
    pred_texts, truth_texts = zip(*valid_pairs)
    
    # Average length
    if "avg_len" in metrics:
        out["avg_len"] = float(np.mean([len(p) for p in pred_texts]))
    
    # Lexical metrics
    if any(m in metrics for m in ["bleu", "rouge1", "rouge2", "rougeL", "word_overlap"]):
        rouge = Rouge()
        bleu_scores = []
        rouge_scores = []
        word_overlaps = []
        
        for pred, truth in valid_pairs:
            # BLEU
            if "bleu" in metrics:
                bleu_scores.append(_compute_bleu(pred, truth))
            
            # ROUGE
            if any(m in metrics for m in ["rouge1", "rouge2", "rougeL"]):
                rouge_scores.append(_compute_rouge(pred, truth, rouge))
            
            # Word overlap
            if "word_overlap" in metrics:
                word_overlaps.append(_compute_word_overlap(pred, truth))
        
        if "bleu" in metrics and bleu_scores:
            out["bleu"] = float(np.mean(bleu_scores))
        
        if rouge_scores:
            if "rouge1" in metrics:
                out["rouge1"] = float(np.mean([s['rouge1'] for s in rouge_scores]))
            if "rouge2" in metrics:
                out["rouge2"] = float(np.mean([s['rouge2'] for s in rouge_scores]))
            if "rougeL" in metrics:
                out["rougeL"] = float(np.mean([s['rougeL'] for s in rouge_scores]))
        
        if "word_overlap" in metrics and word_overlaps:
            out["word_overlap"] = float(np.mean(word_overlaps))
    
    # Semantic similarity metrics
    if any(m in metrics for m in ["cosine_similarity", "euclidean_distance", "dot_product"]):
        semantic_metrics = _compute_semantic_similarity(
            list(pred_texts),
            list(truth_texts),
            embedding_model_name
        )
        
        if "cosine_similarity" in metrics:
            out["cosine_similarity"] = semantic_metrics["cosine_similarity"]
        if "euclidean_distance" in metrics:
            out["euclidean_distance"] = semantic_metrics["euclidean_distance"]
        if "dot_product" in metrics:
            out["dot_product"] = semantic_metrics["dot_product"]
    
    return out


def _compute_word_overlap(pred: str, truth: str) -> float:
    """Computes word overlap ratio between two texts."""
    pred_words = set(pred.lower().split())
    truth_words = set(truth.lower().split())
    
    if not truth_words:
        return 0.0
    
    overlap = len(pred_words & truth_words)
    return overlap / len(truth_words)


def _compute_bleu(pred: str, truth: str) -> float:
    """Computes BLEU score with smoothing."""
    reference = [truth.split()]
    candidate = pred.split()
    smoothing = SmoothingFunction().method1
    
    try:
        return sentence_bleu(
            reference, 
            candidate, 
            smoothing_function=smoothing
        )
    except:
        return 0.0


def _compute_rouge(pred: str, truth: str, rouge: Rouge) -> Dict[str, float]:
    """Computes ROUGE scores."""
    try:
        scores = rouge.get_scores(pred, truth)[0]
        return {
            'rouge1': scores['rouge-1']['f'],
            'rouge2': scores['rouge-2']['f'],
            'rougeL': scores['rouge-l']['f']
        }
    except:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def _compute_semantic_similarity(
    predictions: List[str],
    ground_truths: List[str],
    embedding_model_name: str,
) -> Dict[str, float]:
    """Computes semantic similarity metrics using embeddings."""
    # Load model
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Generate embeddings
    pred_embeddings = embedding_model.encode(
        predictions,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    truth_embeddings = embedding_model.encode(
        ground_truths,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    # Compute metrics
    cosine_sims = [
        1 - cosine(pred_emb, truth_emb)
        for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings)
    ]
    
    euclidean_dists = [
        np.linalg.norm(pred_emb - truth_emb)
        for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings)
    ]
    
    dot_products = [
        np.dot(pred_emb, truth_emb)
        for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings)
    ]
    
    return {
        'cosine_similarity': float(np.mean(cosine_sims)),
        'euclidean_distance': float(np.mean(euclidean_dists)),
        'dot_product': float(np.mean(dot_products)),
    }


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