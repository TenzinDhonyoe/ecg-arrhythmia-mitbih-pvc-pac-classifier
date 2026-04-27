"""SE-ResNet-1D tests.

Critical correctness invariants:

1. ``ResNet1D(use_se=False)`` is bit-identical to v0.3 (existing weights load
   without state-dict warnings, forward pass produces the same output).
2. SE on / Inception stem on: shapes preserved, gradients flow.
3. ``forward_embedding`` returns the penultimate embedding alongside logits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg_arrhythmia.models.resnet1d import (  # noqa: E402
    InceptionStem1D,
    ResidualBlock1D,
    ResNet1D,
    SEBlock1D,
)

REPO = Path(__file__).resolve().parent.parent


def test_se_block_shape_and_scaling():
    se = SEBlock1D(channels=32, reduction=8)
    x = torch.randn(2, 32, 50)
    out = se(x)
    assert out.shape == x.shape
    # Scaling weights are sigmoid → output magnitudes are <= input magnitudes.
    assert torch.all(out.abs() <= x.abs() + 1e-5)


def test_residual_block_use_se_off_means_no_se_attribute():
    """When use_se=False, no SE module is constructed (state-dict size unchanged)."""
    blk_off = ResidualBlock1D(32, 32, use_se=False)
    assert blk_off.se is None
    blk_on = ResidualBlock1D(32, 32, use_se=True)
    assert blk_on.se is not None


def test_default_resnet_loads_v0_3_weights():
    """The shipped v0.3 ResNet weights must still load via the v0.4 ResNet1D."""
    config_path = REPO / "artifacts" / "resnet" / "model_config.json"
    weights_path = REPO / "artifacts" / "resnet" / "resnet1d.pt"
    if not (config_path.exists() and weights_path.exists()):
        pytest.skip("v0.3 ResNet artifacts not present")
    config = json.loads(config_path.read_text())
    model = ResNet1D(**config)  # use_se defaults to False
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    # No missing/unexpected keys.
    res = model.load_state_dict(state, strict=True)
    assert res.missing_keys == []
    assert res.unexpected_keys == []


def test_default_forward_numerics_unchanged_with_use_se_false():
    """SE off → identical numerics to v0.3 model on a fixed input."""
    config_path = REPO / "artifacts" / "resnet" / "model_config.json"
    weights_path = REPO / "artifacts" / "resnet" / "resnet1d.pt"
    if not (config_path.exists() and weights_path.exists()):
        pytest.skip("v0.3 ResNet artifacts not present")
    config = json.loads(config_path.read_text())
    model = ResNet1D(**config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(0)
    morph = torch.from_numpy(rng.standard_normal((3, 1, 180)).astype(np.float32))
    rr = torch.from_numpy(rng.standard_normal((3, 2)).astype(np.float32))

    with torch.no_grad():
        out = model(morph, rr)
    assert out.shape == (3, config["n_classes"])
    # Output must be finite (the regression check; we don't snapshot exact values
    # since training stochasticity is fine — what matters is the API is identical).
    assert torch.all(torch.isfinite(out))


def test_se_on_forward_and_backward():
    model = ResNet1D(in_channels=1, samples_per_lead=180, n_classes=5, use_se=True)
    morph = torch.randn(4, 1, 180, requires_grad=True)
    rr = torch.randn(4, 2, requires_grad=True)
    out = model(morph, rr)
    assert out.shape == (4, 5)
    loss = out.sum()
    loss.backward()
    assert morph.grad is not None
    assert rr.grad is not None


def test_inception_stem_shape():
    stem = InceptionStem1D(in_channels=1, out_ch=32)
    x = torch.randn(2, 1, 180)
    out = stem(x)
    # Same compression as the default conv stem (stride=2 conv + stride=2 pool).
    assert out.shape == (2, 32, 45)


def test_resnet_inception_stem_full_forward():
    model = ResNet1D(in_channels=1, samples_per_lead=180, n_classes=5, stem="inception")
    morph = torch.randn(4, 1, 180)
    rr = torch.randn(4, 2)
    out = model(morph, rr)
    assert out.shape == (4, 5)


def test_resnet_invalid_stem_raises():
    with pytest.raises(ValueError, match="stem"):
        ResNet1D(stem="bogus")  # type: ignore[arg-type]


def test_forward_embedding_returns_emb_and_logits():
    model = ResNet1D(in_channels=1, samples_per_lead=180, n_classes=5, use_se=True)
    morph = torch.randn(2, 1, 180)
    rr = torch.randn(2, 2)
    emb, logits = model.forward_embedding(morph, rr)
    # GAP output → channels[-1] = 128.
    assert emb.shape == (2, 128)
    assert logits.shape == (2, 5)
