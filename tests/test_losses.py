"""FocalLoss + mixup + balanced-sampler tests.

Pin two correctness invariants that are easy to drift on:

1. ``FocalLoss(gamma=0)`` reduces to weighted CE *exactly*.
2. ``WeightedRandomSampler`` produces a roughly class-balanced epoch.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg_arrhythmia.losses import FocalLoss, make_balanced_sampler, mixup_batch  # noqa: E402


def test_focal_gamma_zero_equals_ce():
    rng = np.random.default_rng(0)
    n, c = 32, 5
    logits = torch.from_numpy(rng.standard_normal((n, c)).astype(np.float32))
    targets = torch.from_numpy(rng.integers(0, c, size=n).astype(np.int64))

    focal = FocalLoss(gamma=0.0)
    fl = focal(logits, targets).item()
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="mean").item()
    assert fl == pytest.approx(ce, rel=1e-5, abs=1e-6)


def test_focal_gamma_zero_with_class_weights_equals_weighted_ce():
    rng = np.random.default_rng(0)
    n, c = 32, 5
    logits = torch.from_numpy(rng.standard_normal((n, c)).astype(np.float32))
    targets = torch.from_numpy(rng.integers(0, c, size=n).astype(np.int64))
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8])

    focal = FocalLoss(gamma=0.0, weight=weights)
    fl = focal(logits, targets).item()
    # Reference: PyTorch CE with weight, mean reduction.
    ce = torch.nn.functional.cross_entropy(logits, targets, weight=weights, reduction="mean").item()
    # Our FocalLoss takes plain mean over batch (not weighted mean), so we
    # match the unweighted-mean variant of CE that has already absorbed the
    # per-class weight. This is intentional — see FocalLoss docstring.
    # Compare with reduction='none' weighted CE, then mean unweighted.
    ce_none = torch.nn.functional.cross_entropy(logits, targets, weight=weights, reduction="none")
    # PyTorch's CE-with-weight returns weighted-per-sample loss; our impl
    # multiplies the same per-class weight then takes a plain mean. So:
    expected = ce_none.mean().item()
    assert fl == pytest.approx(expected, rel=1e-4, abs=1e-5)
    # Sanity: focal should be on the same order of magnitude as CE-mean.
    assert fl == pytest.approx(ce, rel=2.0)


def test_focal_gamma_positive_downweights_easy_examples():
    """A confident-correct example should contribute less under focal than under CE."""
    # Highly confident, correct: logit on class 0 dominates.
    logits = torch.tensor([[10.0, -5.0, -5.0]])
    target = torch.tensor([0])
    ce = torch.nn.functional.cross_entropy(logits, target).item()
    fl_g0 = FocalLoss(gamma=0.0)(logits, target).item()
    fl_g2 = FocalLoss(gamma=2.0)(logits, target).item()
    assert fl_g0 == pytest.approx(ce, rel=1e-5, abs=1e-6)
    # γ=2 on a confident-correct prediction must shrink the loss.
    assert fl_g2 < fl_g0


def test_focal_validates_gamma_and_smoothing():
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError):
        FocalLoss(label_smoothing=1.5)
    with pytest.raises(ValueError):
        FocalLoss(label_smoothing=-0.1)


def test_focal_label_smoothing_increases_loss_on_perfect_pred():
    """Label smoothing should raise the loss floor even on a perfect prediction."""
    logits = torch.tensor([[20.0, -20.0, -20.0]])  # near-perfect for class 0
    target = torch.tensor([0])
    no_smooth = FocalLoss(gamma=0.0, label_smoothing=0.0)(logits, target).item()
    smoothed = FocalLoss(gamma=0.0, label_smoothing=0.1)(logits, target).item()
    assert smoothed > no_smooth


def test_mixup_batch_shapes_and_lam():
    rng = np.random.default_rng(0)
    n, ch, samples = 8, 1, 180
    morph = torch.from_numpy(rng.standard_normal((n, ch, samples)).astype(np.float32))
    rr = torch.from_numpy(rng.standard_normal((n, 2)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 5, size=n).astype(np.int64))
    morph_m, rr_m, y_a, y_b, lam = mixup_batch(morph, rr, y, alpha=0.2, rng=rng)
    assert morph_m.shape == morph.shape
    assert rr_m.shape == rr.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= float(lam.item()) <= 1.0
    assert torch.equal(y_a, y)


def test_mixup_zero_alpha_is_identity():
    rng = np.random.default_rng(0)
    morph = torch.from_numpy(rng.standard_normal((4, 1, 180)).astype(np.float32))
    rr = torch.from_numpy(rng.standard_normal((4, 2)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 5, size=4).astype(np.int64))
    morph_m, rr_m, y_a, y_b, lam = mixup_batch(morph, rr, y, alpha=0.0)
    assert float(lam.item()) == 1.0
    assert torch.equal(morph_m, morph)
    assert torch.equal(rr_m, rr)
    assert torch.equal(y_a, y_b)


def test_balanced_sampler_distribution():
    """Class frequency in the sampler stream should be far flatter than in y."""
    # 90% class 0, 10% class 1, 0.1% class 2 — roughly MIT-BIH-shape.
    y = np.concatenate([np.zeros(900), np.ones(99), np.full(1, 2)]).astype(int)
    sampler = make_balanced_sampler(y, n_classes=3, seed=0)
    drawn = list(sampler)
    classes = y[drawn]
    counts = np.bincount(classes, minlength=3)
    # In the original distribution class 2 has 1/1000 = 0.1%; under the sampler
    # we expect roughly 1/3 each (sampler draws are weighted equally per class).
    # Demand: class 2's share is at least 10% — far above its natural 0.1%.
    assert counts[2] / len(drawn) > 0.1
    # And class 0 must drop below 60% (was 90%).
    assert counts[0] / len(drawn) < 0.6


def test_balanced_sampler_handles_zero_count_classes():
    """If a class is entirely absent, the sampler shouldn't blow up."""
    y = np.array([0, 0, 0, 1, 1])
    sampler = make_balanced_sampler(y, n_classes=3, seed=0)
    drawn = list(sampler)
    classes = y[drawn]
    # No drawn index should resolve to class 2 (which has zero samples).
    assert 2 not in classes
