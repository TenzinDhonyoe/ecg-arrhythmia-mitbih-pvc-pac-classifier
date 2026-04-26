"""Tests for the wearable inference helpers (polarity flip, lead warning)."""

from __future__ import annotations

import csv

import numpy as np

from ecg_arrhythmia.inference import (
    WearableInferenceConfig,
    auto_polarity_check,
    detect_r_peaks,
)


def _synthetic_ecg(fs=250.0, duration=10.0, hr_bpm=72.0):
    """Make a coarse, periodic ECG-like signal we can find R-peaks in."""
    t = np.arange(int(duration * fs)) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros_like(t)
    # Place narrow Gaussian "R-peaks" at every rr seconds
    for tk in np.arange(rr, duration, rr):
        sig += np.exp(-((t - tk) / 0.02) ** 2)
    # Add small T waves
    for tk in np.arange(rr + 0.25, duration, rr):
        sig += 0.3 * np.exp(-((t - tk) / 0.05) ** 2)
    return sig


def test_auto_polarity_flips_inverted_signal():
    fs = 250.0
    sig = _synthetic_ecg(fs=fs)
    inverted = -sig
    assert auto_polarity_check(inverted, fs=fs) is True
    assert auto_polarity_check(sig, fs=fs) is False


def test_detect_r_peaks_finds_expected_count():
    fs = 250.0
    sig = _synthetic_ecg(fs=fs, duration=10.0, hr_bpm=72.0)
    peaks = detect_r_peaks(sig, fs=fs)
    # ~12 beats in 10 seconds at 72 bpm; allow a wide window for end-effects
    assert 8 <= len(peaks) <= 16


def test_wearable_config_validates_lead_name():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        WearableInferenceConfig(input_fs=250.0, lead_name="bogus")
        assert any("Unknown lead" in str(item.message) for item in w)


def test_wearable_inference_end_to_end_on_synthetic_csv(tmp_path):
    """Train a smoke model and run wearable inference on a synthetic Lead-I CSV."""
    from ecg_arrhythmia.inference import run_baseline_inference
    from ecg_arrhythmia.training import train_baseline_quick_smoke

    artifacts = train_baseline_quick_smoke(tmp_path, seed=0)
    sig = _synthetic_ecg(fs=250.0, duration=10.0)
    csv_path = tmp_path / "wearable.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal"])
        for v in sig:
            w.writerow([f"{v:.6f}"])
    rows = run_baseline_inference(
        csv_path,
        artifacts.model_path,
        artifacts.scaler_path,
        input_fs=250.0,
        lead_name="I",
        auto_polarity=True,
    )
    # Synthetic ECG has detectable R-peaks; we don't assert label correctness
    # because the model was trained on synthetic Gaussians, just that the
    # pipeline produces well-formed rows.
    assert len(rows) >= 1
    for row in rows:
        assert row["label"] in {"N", "V", "a"}
        assert 0.0 <= row["prob_N"] <= 1.0
        assert 0.0 <= row["prob_V"] <= 1.0
        assert 0.0 <= row["prob_a"] <= 1.0
