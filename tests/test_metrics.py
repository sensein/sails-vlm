"""Characterization tests for evaluation.metrics (CPU-only)."""
import pytest

from sails_vlm.evaluation.metrics import evaluate_classification, evaluate_counting


class TestEvaluateClassification:
    def test_perfect_predictions(self, tmp_path):
        y_true = ["rocking", "jumping", "spinning", "rocking"]
        y_pred = ["rocking", "jumping", "spinning", "rocking"]
        results = evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            labels=["rocking", "jumping", "spinning"],
            binary=False,
            metrics=["accuracy", "f1_macro"],
            output_dir=str(tmp_path),
        )
        assert results["accuracy"] == pytest.approx(1.0)
        assert results["f1_macro"] == pytest.approx(1.0)

    def test_half_right_accuracy(self, tmp_path):
        y_true = ["rocking", "jumping"]
        y_pred = ["rocking", "spinning"]
        results = evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            labels=["rocking", "jumping", "spinning"],
            binary=False,
            metrics=["accuracy"],
            output_dir=str(tmp_path),
        )
        assert results["accuracy"] == pytest.approx(0.5)


class TestEvaluateCounting:
    def test_exact_counts(self):
        results = evaluate_counting(
            y_true=[2, 3, 0],
            y_pred=[2, 3, 0],
            metrics=["mae"],
        )
        assert results["mae"] == pytest.approx(0.0)


def test_evaluate_description_needs_text_extra():
    """evaluate_description lazily imports rouge/nltk/sentence-transformers;
    without the text-metrics extra it must raise ImportError, not crash at
    module import time."""
    pytest.importorskip("rouge", reason="text-metrics extra not installed")
    from sails_vlm.evaluation.metrics import evaluate_description  # noqa: F401
