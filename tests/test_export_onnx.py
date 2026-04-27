"""Tests for ONNX export. Skipped automatically if onnx/skl2onnx aren't installed."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ecg_arrhythmia.preprocessing import WINDOW_SIZE
from ecg_arrhythmia.training import train_baseline_quick_smoke

skl2onnx = importlib.util.find_spec("skl2onnx")
onnxruntime = importlib.util.find_spec("onnxruntime")
pytestmark = pytest.mark.skipif(
    skl2onnx is None or onnxruntime is None,
    reason="onnx export deps not installed; install with [onnx] extra",
)


def test_export_baseline_to_onnx_roundtrip(tmp_path):
    import joblib
    import onnxruntime as ort

    from ecg_arrhythmia.export import export_baseline_to_onnx

    artifacts = train_baseline_quick_smoke(tmp_path, seed=0)
    out = tmp_path / "baseline_lr.onnx"
    report = export_baseline_to_onnx(
        artifacts.model_path,
        artifacts.scaler_path,
        out,
    )
    assert out.is_file()
    assert report.max_abs_diff < 1e-3

    # Independent round-trip on a fresh sample.
    rng = np.random.default_rng(42)
    sample = rng.standard_normal((8, WINDOW_SIZE + 2)).astype(np.float32)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_proba = session.run(None, {session.get_inputs()[0].name: sample})[1]

    model = joblib.load(artifacts.model_path)
    scaler = joblib.load(artifacts.scaler_path)
    sk_proba = model.predict_proba(scaler.transform(sample))
    np.testing.assert_allclose(onnx_proba, sk_proba, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed; install with [deep] extra",
)
def test_export_resnet_to_onnx_roundtrip(tmp_path):
    import json

    import onnxruntime as ort
    import torch

    from ecg_arrhythmia.export import export_resnet_to_onnx
    from ecg_arrhythmia.models.resnet1d import ResNet1D

    config = {
        "in_channels": 1,
        "samples_per_lead": 180,
        "rr_features": 2,
        "n_classes": 3,
        "channels": [16, 32],
        "blocks_per_stage": 1,
    }
    model = ResNet1D(**config)
    model.eval()
    weights = tmp_path / "tiny_resnet.pt"
    config_path = tmp_path / "config.json"
    torch.save(model.state_dict(), weights)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    out = tmp_path / "resnet.onnx"
    report = export_resnet_to_onnx(weights, config_path, out)
    assert out.is_file()
    assert report.max_abs_diff < 1e-3

    # Dynamic batch axis should let us run a different batch size.
    morph = np.random.randn(7, 1, 180).astype(np.float32)
    rr = np.random.randn(7, 2).astype(np.float32)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"morph": morph, "rr": rr})[0]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(morph), torch.from_numpy(rr)).numpy()
    np.testing.assert_allclose(onnx_logits, torch_logits, rtol=1e-3, atol=1e-4)
