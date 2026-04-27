"""Tests for the calibrated/uncertainty-aware ECGClassifier path.

Pin the v0.4 invariants:

1. ``ECGClassifier.from_artifacts`` auto-loads ``temperature.pt`` if present
   and applies it inside ``predict()``.
2. ``predict_with_uncertainty`` adds a non-None ``uncertainty`` field per
   beat for ResNet-backed classifiers.
3. The LR baseline returns ``uncertainty=None`` (no MC dropout path).
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg_arrhythmia import ECGClassifier  # noqa: E402
from ecg_arrhythmia.calibration import TemperatureScaler  # noqa: E402
from ecg_arrhythmia.training import train_baseline_quick_smoke, train_resnet_1d  # noqa: E402


def _synthetic_ecg(fs=360.0, duration=10.0, hr_bpm=72.0):
    t = np.arange(int(duration * fs)) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros_like(t)
    for tk in np.arange(rr, duration, rr):
        sig += np.exp(-((t - tk) / 0.02) ** 2)
    for tk in np.arange(rr + 0.25, duration, rr):
        sig += 0.3 * np.exp(-((t - tk) / 0.05) ** 2)
    return sig


def _write_csv(path, sig):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal"])
        for v in sig:
            w.writerow([f"{v:.6f}"])


def test_baseline_uncertainty_is_none(tmp_path):
    artifacts = train_baseline_quick_smoke(tmp_path, seed=0)
    clf = ECGClassifier.load_baseline(artifacts.model_path, artifacts.scaler_path)

    sig = _synthetic_ecg()
    result = clf.predict_with_uncertainty(sig, input_fs=360.0, lead="MLII", n_mc=5)
    assert len(result) >= 1
    for beat in result:
        assert beat.uncertainty is None


def test_resnet_uncertainty_is_finite_and_positive(tmp_path):
    """A ResNet smoke model returns positive epistemic entropy per beat."""
    artifacts = train_resnet_1d(
        tmp_path,
        tmp_path / "out",
        seed=0,
        epochs=2,
        batch_size=32,
        device="cpu",
        quick_smoke=True,
    )
    clf = ECGClassifier.load_resnet(artifacts.model_path, artifacts.scaler_path)
    sig = _synthetic_ecg()
    result = clf.predict_with_uncertainty(sig, input_fs=360.0, lead="MLII", n_mc=5)
    assert len(result) >= 1
    for beat in result:
        assert beat.uncertainty is not None
        assert np.isfinite(beat.uncertainty)
        assert beat.uncertainty >= 0.0


def test_temperature_auto_loaded_from_artifacts(tmp_path):
    """If ``temperature.pt`` is in the artifact dir, ``from_artifacts`` auto-loads it."""
    # Train a tiny ResNet so we can exercise from_artifacts.
    artifacts = train_resnet_1d(
        tmp_path,
        tmp_path / "out",
        seed=0,
        epochs=2,
        batch_size=32,
        device="cpu",
        quick_smoke=True,
    )
    out = artifacts.model_path.parent
    # Drop in a fake temperature.pt with a known value, then re-load.
    TemperatureScaler(init_temperature=2.5).save(out / "temperature.pt")
    clf = ECGClassifier.from_artifacts(out, prefer="resnet")
    assert clf.temperature == pytest.approx(2.5)


def test_temperature_changes_probabilities_but_not_argmax(tmp_path):
    """Setting a temperature reshapes probs but preserves the predicted label."""
    artifacts = train_resnet_1d(
        tmp_path,
        tmp_path / "out",
        seed=0,
        epochs=2,
        batch_size=32,
        device="cpu",
        quick_smoke=True,
    )
    clf = ECGClassifier.load_resnet(artifacts.model_path, artifacts.scaler_path)
    sig = _synthetic_ecg()

    clf.temperature = None
    res_t1 = clf.predict(sig, input_fs=360.0, lead="MLII")
    clf.temperature = 5.0
    res_t5 = clf.predict(sig, input_fs=360.0, lead="MLII")

    if len(res_t1) == 0:
        pytest.skip("synthetic signal produced no beats")
    # Argmax labels unchanged.
    assert [b.label for b in res_t1] == [b.label for b in res_t5]
    # Probabilities should differ (T = 5 widens the distribution).
    p1 = res_t1.probabilities
    p5 = res_t5.probabilities
    assert not np.allclose(p1, p5, atol=1e-3)


def test_predict_with_uncertainty_empty_signal(tmp_path):
    artifacts = train_resnet_1d(
        tmp_path,
        tmp_path / "out",
        seed=0,
        epochs=2,
        batch_size=32,
        device="cpu",
        quick_smoke=True,
    )
    clf = ECGClassifier.load_resnet(artifacts.model_path, artifacts.scaler_path)
    result = clf.predict_with_uncertainty(np.array([]), input_fs=360.0, n_mc=5)
    assert len(result) == 0
