"""Tests for evaluation metrics."""

import numpy as np

from tac_personalization.evaluation.metrics import compute_binary_metrics


class TestComputeBinaryMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_binary_metrics(y_true, y_pred)
        assert m["acc"] == 1.0
        assert m["brier"] < 0.1
        assert m["auc_roc"] == 1.0
        assert m["ll"] < 0  # log-likelihood is negative

    def test_worst_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_binary_metrics(y_true, y_pred)
        assert m["acc"] == 0.0
        assert m["auc_roc"] == 0.0

    def test_empty_input(self):
        m = compute_binary_metrics(np.array([]), np.array([]))
        assert np.isnan(m["acc"])
        assert np.isnan(m["ll"])

    def test_single_class_auc_is_nan(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0.7, 0.8, 0.9])
        m = compute_binary_metrics(y_true, y_pred)
        assert np.isnan(m["auc_roc"])
        assert m["acc"] == 1.0

    def test_random_predictions(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.uniform(0, 1, 100)
        m = compute_binary_metrics(y_true, y_pred)
        # Random should be near 0.5 accuracy
        assert 0.3 < m["acc"] < 0.7
        assert 0.3 < m["auc_roc"] < 0.7
