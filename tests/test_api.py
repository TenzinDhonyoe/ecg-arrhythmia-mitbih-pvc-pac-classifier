"""Tests for the high-level ECGClassifier API."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from ecg_arrhythmia import BeatPrediction, ECGClassifier, PredictionResult
from ecg_arrhythmia.training import train_baseline_quick_smoke


def _synthetic_ecg(fs=360.0, duration=10.0, hr_bpm=72.0):
    t = np.arange(int(duration * fs)) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros_like(t)
    for tk in np.arange(rr, duration, rr):
        sig += np.exp(-((t - tk) / 0.02) ** 2)
    for tk in np.arange(rr + 0.25, duration, rr):
        sig += 0.3 * np.exp(-((t - tk) / 0.05) ** 2)
    return sig


@pytest.fixture(scope="module")
def baseline_artifacts(tmp_path_factory):
    return train_baseline_quick_smoke(tmp_path_factory.mktemp("smoke_api"), seed=0)


def test_from_artifacts_loads_baseline(baseline_artifacts):
    clf = ECGClassifier.from_artifacts(baseline_artifacts.model_path.parent, prefer="baseline")
    assert clf.backend == "baseline"


def test_predict_returns_prediction_result(baseline_artifacts):
    clf = ECGClassifier.load_baseline(baseline_artifacts.model_path, baseline_artifacts.scaler_path)
    sig = _synthetic_ecg()
    result = clf.predict(sig, input_fs=360.0, lead="MLII")
    assert isinstance(result, PredictionResult)
    assert len(result) >= 1
    for beat in result:
        assert isinstance(beat, BeatPrediction)
        assert beat.label in {"N", "V", "a"}
        assert 0.0 <= beat.confidence <= 1.0
        assert set(beat.probabilities.keys()) == {"N", "V", "a"}
        # confidence is the max prob
        assert beat.confidence == pytest.approx(max(beat.probabilities.values()))


def test_prediction_result_helpers(baseline_artifacts):
    clf = ECGClassifier.load_baseline(baseline_artifacts.model_path, baseline_artifacts.scaler_path)
    sig = _synthetic_ecg(hr_bpm=72.0)
    result = clf.predict(sig, input_fs=360.0, lead="MLII")

    counts = result.class_counts()
    assert sum(counts.values()) == len(result)

    hr = result.heart_rate_bpm()
    if len(result) >= 2:
        assert hr is not None
        # synthetic ECG was 72 bpm; allow generous slack for peak detection edge effects
        assert 50.0 <= hr <= 100.0

    summary = result.summary()
    assert summary["n_beats"] == len(result)
    assert summary["class_counts"] == counts
    assert "backend" in summary

    probs = result.probabilities
    assert probs.shape == (len(result), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_predict_csv(baseline_artifacts, tmp_path):
    clf = ECGClassifier.load_baseline(baseline_artifacts.model_path, baseline_artifacts.scaler_path)
    sig = _synthetic_ecg()
    csv_path = tmp_path / "ecg.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal"])
        for v in sig:
            w.writerow([f"{v:.6f}"])
    result = clf.predict_csv(csv_path, input_fs=360.0, lead="MLII")
    assert len(result) >= 1


def test_predict_empty_signal(baseline_artifacts):
    clf = ECGClassifier.load_baseline(baseline_artifacts.model_path, baseline_artifacts.scaler_path)
    result = clf.predict(np.array([]), input_fs=360.0)
    assert len(result) == 0
    assert result.summary()["n_beats"] == 0


def test_topk_returns_sorted(baseline_artifacts):
    clf = ECGClassifier.load_baseline(baseline_artifacts.model_path, baseline_artifacts.scaler_path)
    sig = _synthetic_ecg()
    result = clf.predict(sig, input_fs=360.0)
    assert len(result) >= 1
    beat = result[0]
    top2 = beat.topk(2)
    assert len(top2) == 2
    assert top2[0][1] >= top2[1][1]


def test_from_artifacts_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ECGClassifier.from_artifacts(tmp_path / "does_not_exist")


def test_from_artifacts_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No recognised model artifacts"):
        ECGClassifier.from_artifacts(tmp_path)
