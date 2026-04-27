"""Tests for BeatAugmenter and AugmentedECGDataset.

Pin the correctness invariants documented in :mod:`ecg_arrhythmia.augment`:
determinism, shape preservation, RR-features-untouched, off-by-default safety.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg_arrhythmia.augment import (
    AugmentedECGDataset,
    AugmentSpec,
    BeatAugmenter,
)
from ecg_arrhythmia.preprocessing import WINDOW_SIZE


def _fake_window(rng):
    return rng.standard_normal(WINDOW_SIZE).astype(np.float32)


def test_apply_preserves_shape_and_dtype():
    aug = BeatAugmenter(seed=0)
    rng = np.random.default_rng(0)
    batch = np.stack([_fake_window(rng) for _ in range(4)])
    out = aug.apply(batch)
    assert out.shape == batch.shape
    assert np.all(np.isfinite(out))


def test_apply_is_deterministic_given_seed():
    rng_data = np.random.default_rng(0)
    batch = np.stack([_fake_window(rng_data) for _ in range(8)])

    a = BeatAugmenter(seed=42).apply(batch.copy())
    b = BeatAugmenter(seed=42).apply(batch.copy())
    np.testing.assert_array_equal(a, b)


def test_apply_actually_changes_values():
    """With all probabilities at 1.0, the output must differ from input."""
    spec = AugmentSpec(
        p_amplitude=1.0, p_noise=1.0, p_baseline_wander=1.0,
        p_time_warp=1.0, p_shift=1.0,
    )
    aug = BeatAugmenter(spec=spec, seed=1)
    rng_data = np.random.default_rng(0)
    batch = np.stack([_fake_window(rng_data) for _ in range(4)])
    out = aug.apply(batch.copy())
    # At least one window must have moved — and meaningfully.
    diffs = np.linalg.norm(out - batch, axis=1)
    assert np.any(diffs > 0.01)


def test_off_by_default_when_all_probs_zero():
    spec = AugmentSpec(
        p_amplitude=0.0, p_noise=0.0, p_baseline_wander=0.0,
        p_time_warp=0.0, p_shift=0.0,
    )
    aug = BeatAugmenter(spec=spec, seed=0)
    rng_data = np.random.default_rng(0)
    batch = np.stack([_fake_window(rng_data) for _ in range(4)])
    out = aug.apply(batch.copy())
    np.testing.assert_array_equal(out, batch.astype(float))


def test_apply_rejects_wrong_window_size():
    aug = BeatAugmenter(seed=0)
    bad = np.zeros((2, WINDOW_SIZE - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="WINDOW_SIZE|180"):
        aug.apply(bad)


def test_apply_handles_1d_input():
    aug = BeatAugmenter(seed=0)
    rng = np.random.default_rng(0)
    w = _fake_window(rng)
    out = aug.apply(w)
    assert out.shape == (1, WINDOW_SIZE)


def test_does_not_mutate_input():
    aug = BeatAugmenter(seed=0)
    rng = np.random.default_rng(0)
    batch = np.stack([_fake_window(rng) for _ in range(4)])
    snapshot = batch.copy()
    _ = aug.apply(batch)
    np.testing.assert_array_equal(batch, snapshot)


def test_temporal_shift_preserves_length():
    aug = BeatAugmenter(spec=AugmentSpec(p_shift=1.0), seed=0)
    rng = np.random.default_rng(0)
    w = _fake_window(rng)
    out = aug.temporal_shift(w, rng)
    assert out.shape == w.shape


def test_time_warp_preserves_length():
    aug = BeatAugmenter(spec=AugmentSpec(p_time_warp=1.0), seed=0)
    rng = np.random.default_rng(0)
    w = _fake_window(rng)
    out = aug.time_warp(w, rng)
    assert out.shape == w.shape


# ---------------------------------------------------------------------------
# AugmentedECGDataset
# ---------------------------------------------------------------------------
def _make_dataset(n=8, augmenter=None, in_channels=1):
    rng = np.random.default_rng(0)
    morph = rng.standard_normal((n, WINDOW_SIZE * in_channels)).astype(np.float32)
    rr = rng.standard_normal((n, 2)).astype(np.float32)
    y = rng.integers(0, 3, size=n)
    return AugmentedECGDataset(
        morph, rr, y,
        in_channels=in_channels,
        samples_per_lead=WINDOW_SIZE,
        augmenter=augmenter,
    )


def test_dataset_length_and_no_augmentation():
    pytest.importorskip("torch")
    ds = _make_dataset(n=5, augmenter=None)
    assert len(ds) == 5
    morph_t, rr_t, y_t = ds[0]
    assert morph_t.shape == (1, WINDOW_SIZE)
    assert rr_t.shape == (2,)
    assert y_t.dtype.is_signed


def test_dataset_returns_torch_tensors():
    torch = pytest.importorskip("torch")
    ds = _make_dataset(n=3, augmenter=BeatAugmenter(seed=0))
    morph_t, rr_t, y_t = ds[0]
    assert isinstance(morph_t, torch.Tensor)
    assert isinstance(rr_t, torch.Tensor)
    assert isinstance(y_t, torch.Tensor)


def test_dataset_rr_features_untouched_under_augmentation():
    """Critical correctness invariant: augmentation only touches morphology."""
    pytest.importorskip("torch")
    spec = AugmentSpec(
        p_amplitude=1.0, p_noise=1.0, p_baseline_wander=1.0,
        p_time_warp=1.0, p_shift=1.0,
    )
    ds = _make_dataset(n=6, augmenter=BeatAugmenter(spec=spec, seed=42))
    for idx in range(len(ds)):
        _, rr_t, _ = ds[idx]
        np.testing.assert_array_equal(rr_t.numpy(), ds.rr[idx])


def test_dataset_multi_lead_split_and_recombine():
    pytest.importorskip("torch")
    ds = _make_dataset(n=4, augmenter=BeatAugmenter(seed=0), in_channels=2)
    morph_t, _, _ = ds[0]
    assert morph_t.shape == (2, WINDOW_SIZE)


def test_dataset_length_mismatch_raises():
    rng = np.random.default_rng(0)
    morph = rng.standard_normal((4, WINDOW_SIZE)).astype(np.float32)
    rr = rng.standard_normal((3, 2)).astype(np.float32)
    y = rng.integers(0, 3, size=4)
    with pytest.raises(ValueError, match="length mismatch"):
        AugmentedECGDataset(
            morph, rr, y, in_channels=1, samples_per_lead=WINDOW_SIZE
        )
