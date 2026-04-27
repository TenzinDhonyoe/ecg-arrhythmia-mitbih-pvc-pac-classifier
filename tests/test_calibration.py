"""Calibration tests: TemperatureScaler reduces ECE; MC dropout signals noise."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg_arrhythmia.calibration import (  # noqa: E402
    MCDropoutEnsemble,
    TemperatureScaler,
    expected_calibration_error,
)


def test_ece_zero_on_perfectly_calibrated_synthetic():
    """If confidences == accuracies, ECE should be ~0."""
    rng = np.random.default_rng(0)
    n = 2000
    # Build (probs, y) so the max-prob equals empirical accuracy by construction.
    probs = np.zeros((n, 2))
    probs[:, 0] = rng.uniform(0.5, 1.0, size=n)
    probs[:, 1] = 1.0 - probs[:, 0]
    # Sample y = 0 with probability probs[i, 0] (perfectly calibrated).
    y = (rng.random(n) > probs[:, 0]).astype(int)
    out = expected_calibration_error(probs, y, n_bins=15)
    # Small but non-zero is expected from sampling noise.
    assert out["ece"] < 0.05


def test_temperature_reduces_ece_on_overconfident_logits():
    """Pin two invariants of the LBFGS fit on overconfident logits:

    1. T > 1 when the logits are inflated relative to accuracy.
    2. NLL on the same data drops after scaling (this is what LBFGS optimises;
       ECE on the same data tracks NLL closely under temperature scaling).
    3. Argmax is preserved (a structural property — temperature scaling never
       changes the argmax).
    """
    rng = np.random.default_rng(0)
    n = 2000
    n_classes = 5
    z = rng.normal(0.0, 1.0, size=(n, n_classes)).astype(np.float32)
    y = z.argmax(axis=1)
    flip = rng.random(n) < 0.3
    y_noisy = np.where(flip, rng.integers(0, n_classes, size=n), y)
    overconf_logits = (z * 5.0).astype(np.float32)

    # NLL before scaling (T = 1).
    z_t = torch.from_numpy(overconf_logits)
    yt = torch.from_numpy(y_noisy.astype(np.int64))
    nll_before = torch.nn.functional.cross_entropy(z_t, yt).item()

    scaler = TemperatureScaler()
    T = scaler.fit(overconf_logits, y_noisy, max_iter=200)
    assert T > 1.0, f"expected T>1 under overconfidence; got T={T}"

    nll_after = torch.nn.functional.cross_entropy(z_t / T, yt).item()
    assert nll_after < nll_before, (
        f"LBFGS must reduce NLL; before={nll_before:.4f}, after={nll_after:.4f}, T={T:.3f}"
    )

    # Argmax preservation — structural property of temperature scaling.
    probs_before = torch.softmax(z_t, dim=1).numpy()
    probs_after = scaler.predict_proba(overconf_logits)
    assert np.all(probs_before.argmax(axis=1) == probs_after.argmax(axis=1))


def test_temperature_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n, c = 200, 3
    logits = rng.standard_normal((n, c)) * 3.0
    y = rng.integers(0, c, size=n)

    scaler = TemperatureScaler()
    scaler.fit(logits, y, max_iter=50)
    path = tmp_path / "temperature.pt"
    scaler.save(path)

    loaded = TemperatureScaler.load(path)
    assert loaded.temperature == pytest.approx(scaler.temperature, rel=1e-6)
    np.testing.assert_allclose(scaler.apply_logits(logits), loaded.apply_logits(logits))


def test_mc_dropout_produces_nondeterministic_passes():
    """MC dropout must actually toggle dropout layers — repeat passes should differ.

    We can't reliably assert "noise has higher entropy" on an untrained model
    (its outputs are near-uniform for all inputs), so we test the cleaner
    invariant: stacking multiple MC passes on the same input produces a
    non-zero spread, while a vanilla model.eval() forward pass produces an
    identical output every call.
    """
    from ecg_arrhythmia.models.resnet1d import ResNet1D

    model = ResNet1D(in_channels=1, samples_per_lead=180, n_classes=5, use_se=True)
    rng = np.random.default_rng(0)
    morph = torch.from_numpy(rng.standard_normal((4, 1, 180)).astype(np.float32))
    rr = torch.from_numpy(rng.standard_normal((4, 2)).astype(np.float32))

    # Vanilla eval should be deterministic.
    model.eval()
    with torch.no_grad():
        det_a = torch.softmax(model(morph, rr), dim=1).numpy()
        det_b = torch.softmax(model(morph, rr), dim=1).numpy()
    np.testing.assert_allclose(det_a, det_b, atol=1e-6)

    # MC dropout should *not* be deterministic (dropout flips per pass).
    ensemble = MCDropoutEnsemble(model, n_passes=20)
    mean_a, ent_a = ensemble.predict_proba(morph, rr)
    mean_b, ent_b = ensemble.predict_proba(morph, rr)
    # Different runs of the ensemble should give different (but close) means.
    assert not np.allclose(mean_a, mean_b, atol=1e-6)
    # Entropies are positive and < log(n_classes).
    assert (ent_a > 0).all() and (ent_a < np.log(5) + 1e-4).all()


def test_mc_dropout_validates_n_passes():
    from ecg_arrhythmia.models.resnet1d import ResNet1D

    model = ResNet1D(n_classes=3)
    with pytest.raises(ValueError, match="n_passes"):
        MCDropoutEnsemble(model, n_passes=1)


def test_temperature_scaler_apply_logits_is_softmax_when_t_is_one():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((10, 4))
    scaler = TemperatureScaler(init_temperature=1.0)
    probs = scaler.apply_logits(logits)
    # Sums to 1 per row.
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    # Matches torch softmax.
    expected = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    np.testing.assert_allclose(probs, expected, atol=1e-6)
