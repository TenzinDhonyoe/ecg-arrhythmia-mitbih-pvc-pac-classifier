"""Unit tests for the rich metrics report and bootstrap CI."""

from __future__ import annotations

import numpy as np

from ecg_arrhythmia.training import _bootstrap_ci, _full_report


def _toy_predictions():
    rng = np.random.default_rng(0)
    n = 90
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 30)
    # 80% accuracy with some confusion across classes
    y_pred = y_true.copy()
    flip = rng.choice(n, size=int(0.2 * n), replace=False)
    y_pred[flip] = (y_pred[flip] + 1) % 3
    proba = np.zeros((n, 3), dtype=float)
    for i in range(n):
        proba[i, y_pred[i]] = 0.7
        for j in range(3):
            if j != y_pred[i]:
                proba[i, j] = 0.15
    return y_true, y_pred, proba


def test_full_report_shape():
    y_true, y_pred, proba = _toy_predictions()
    report = _full_report(y_true, y_pred, proba)
    assert set(report["per_class"].keys()) == {"N", "V", "a"}
    for cls in ("N", "V", "a"):
        assert 0.0 <= report["per_class"][cls]["precision"] <= 1.0
        assert 0.0 <= report["per_class"][cls]["recall"] <= 1.0
        assert 0.0 <= report["per_class"][cls]["f1"] <= 1.0
        assert report["per_class"][cls]["support"] == 30
    cm = np.array(report["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == 90
    assert 0.0 <= report["balanced_accuracy"] <= 1.0
    assert len(report["balanced_accuracy_ci95"]) == 2
    lo, hi = report["balanced_accuracy_ci95"]
    assert lo <= report["balanced_accuracy"] <= hi
    assert report["roc_auc_ovr_macro"] is None or 0.0 <= report["roc_auc_ovr_macro"] <= 1.0


def test_bootstrap_ci_brackets_point_estimate():
    y_true, y_pred, _ = _toy_predictions()
    lo, hi = _bootstrap_ci(y_true, y_pred, "balanced_accuracy", n_iter=200, seed=1)
    assert lo <= hi
    # 95% CI should contain the empirical balanced-accuracy with very high prob
    from sklearn.metrics import balanced_accuracy_score

    point = balanced_accuracy_score(y_true, y_pred)
    assert lo - 0.1 <= point <= hi + 0.1


def test_full_report_handles_missing_proba():
    y_true, y_pred, _ = _toy_predictions()
    report = _full_report(y_true, y_pred, None)
    assert report["roc_auc_ovr_macro"] is None
