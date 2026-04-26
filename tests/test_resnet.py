"""ResNet-1D architecture and save/load tests.

Both tests are skipped automatically if PyTorch is not installed (the default
``[dev]`` extra does not pull torch).
"""

from __future__ import annotations

import json

import pytest


def _torch_or_skip():
    return pytest.importorskip("torch")


def test_resnet_forward_shape():
    torch = _torch_or_skip()
    from ecg_arrhythmia.models.resnet1d import ResNet1D

    model = ResNet1D(in_channels=1, samples_per_lead=180, rr_features=2, n_classes=3)
    morph = torch.randn(4, 1, 180)
    rr = torch.randn(4, 2)
    out = model(morph, rr)
    assert out.shape == (4, 3)


def test_resnet_param_count_in_budget():
    torch = _torch_or_skip()  # noqa: F841
    from ecg_arrhythmia.models.resnet1d import ResNet1D

    model = ResNet1D()
    n = sum(p.numel() for p in model.parameters())
    # We want this small enough to run on CPU; loose upper bound.
    assert n < 1_000_000, f"ResNet1D has {n} params (>1M)"


def test_resnet_save_load_roundtrip(tmp_path):
    torch = _torch_or_skip()
    from ecg_arrhythmia.models.resnet1d import ResNet1D

    model_a = ResNet1D()
    model_a.eval()
    morph = torch.randn(2, 1, 180)
    rr = torch.randn(2, 2)
    with torch.no_grad():
        out_a = model_a(morph, rr)
    weights = tmp_path / "w.pt"
    cfg = tmp_path / "c.json"
    torch.save(model_a.state_dict(), weights)
    cfg.write_text(json.dumps({
        "in_channels": 1, "samples_per_lead": 180, "rr_features": 2,
        "n_classes": 3, "channels": [32, 64, 128], "blocks_per_stage": 2,
    }))

    cfg_dict = json.loads(cfg.read_text())
    model_b = ResNet1D(**cfg_dict)
    model_b.load_state_dict(torch.load(weights, weights_only=True))
    model_b.eval()
    with torch.no_grad():
        out_b = model_b(morph, rr)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_resnet_quick_smoke_training(tmp_path):
    torch = _torch_or_skip()  # noqa: F841
    from ecg_arrhythmia.training import train_resnet_1d

    artifacts = train_resnet_1d(
        data_dir=tmp_path,
        out_dir=tmp_path / "resnet_smoke",
        seed=0,
        epochs=2,
        batch_size=32,
        device="cpu",
        quick_smoke=True,
    )
    assert artifacts.model_path.exists()
    assert artifacts.metrics_path.exists()
    metrics = json.loads(artifacts.metrics_path.read_text())
    assert metrics["model"] == "resnet1d"
    assert metrics["val"]["balanced_accuracy"] >= 0.0
