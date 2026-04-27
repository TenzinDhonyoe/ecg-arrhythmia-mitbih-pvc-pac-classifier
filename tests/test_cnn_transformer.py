"""CNN-Transformer (experimental) tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ecg_arrhythmia.models.cnn_transformer import CNNTransformer1D  # noqa: E402


def test_forward_shapes_5class():
    model = CNNTransformer1D(in_channels=1, samples_per_lead=180, n_classes=5)
    morph = torch.randn(4, 1, 180)
    rr = torch.randn(4, 2)
    out = model(morph, rr)
    assert out.shape == (4, 5)


def test_forward_shapes_3class_back_compat():
    """Same default config but with n_classes=3 (legacy MITBIH3)."""
    model = CNNTransformer1D(in_channels=1, samples_per_lead=180, n_classes=3)
    out = model(torch.randn(2, 1, 180), torch.randn(2, 2))
    assert out.shape == (2, 3)


def test_backward_pass():
    model = CNNTransformer1D(n_classes=5)
    morph = torch.randn(4, 1, 180, requires_grad=True)
    rr = torch.randn(4, 2, requires_grad=True)
    out = model(morph, rr)
    loss = out.sum()
    loss.backward()
    assert morph.grad is not None
    assert rr.grad is not None


def test_mean_pool_alternative_path():
    """Setting use_cls_token=False switches to mean pooling."""
    model = CNNTransformer1D(n_classes=5, use_cls_token=False)
    morph = torch.randn(2, 1, 180)
    rr = torch.randn(2, 2)
    out = model(morph, rr)
    assert out.shape == (2, 5)


def test_param_count_within_budget():
    """Sanity-check: experimental model stays well under the SE-ResNet budget."""
    model = CNNTransformer1D(n_classes=5)
    n_params = sum(p.numel() for p in model.parameters())
    # Should be << 1M params; if it grows past that, the architecture is no
    # longer "scoped down" — revisit before training on MIT-BIH.
    assert n_params < 250_000, f"unexpectedly large model: {n_params} params"


def test_dropout_is_active_under_train_mode():
    """Two forward passes in train mode should differ (dropout fires)."""
    model = CNNTransformer1D(n_classes=5, dropout=0.5).train()
    morph = torch.randn(4, 1, 180)
    rr = torch.randn(4, 2)
    a = model(morph, rr)
    b = model(morph, rr)
    assert not torch.allclose(a, b)
