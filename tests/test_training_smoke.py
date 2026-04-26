"""End-to-end smoke test for the synthetic training path."""

from __future__ import annotations

import json

from ecg_arrhythmia.training import train_baseline_quick_smoke


def test_quick_smoke_writes_all_artifacts(tmp_path):
    artifacts = train_baseline_quick_smoke(tmp_path, seed=42)
    assert artifacts.model_path.exists()
    assert artifacts.scaler_path.exists()
    assert artifacts.metrics_path.exists()
    assert artifacts.split_path.exists()
    # The smoke metrics file must be named distinctly so it is never confused
    # with a real-data metrics file.
    assert artifacts.metrics_path.name == "metrics_smoke.json"

    metrics = json.loads(artifacts.metrics_path.read_text())
    assert metrics["model"] == "baseline_lr_smoke"
    assert "warning" in metrics
    # Synthetic Gaussians are well separated, so val balanced_accuracy should be
    # comfortably above chance, but we don't pin to 1.0 so the harness can
    # change later.
    assert metrics["val"]["balanced_accuracy"] >= 0.5
    assert metrics["test"]["balanced_accuracy"] >= 0.5


def test_quick_smoke_is_deterministic(tmp_path):
    a = train_baseline_quick_smoke(tmp_path / "a", seed=7)
    b = train_baseline_quick_smoke(tmp_path / "b", seed=7)
    ma = json.loads(a.metrics_path.read_text())
    mb = json.loads(b.metrics_path.read_text())
    assert ma["val"] == mb["val"]
    assert ma["test"] == mb["test"]
