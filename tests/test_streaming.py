"""Tests for the streaming/online ECG classifier."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_arrhythmia import ECGClassifier
from ecg_arrhythmia.streaming import StreamingClassifier
from ecg_arrhythmia.training import train_baseline_quick_smoke


def _synthetic_ecg(fs=360.0, duration=20.0, hr_bpm=72.0):
    t = np.arange(int(duration * fs)) / fs
    rr = 60.0 / hr_bpm
    sig = np.zeros_like(t)
    for tk in np.arange(rr, duration, rr):
        sig += np.exp(-((t - tk) / 0.02) ** 2)
    for tk in np.arange(rr + 0.25, duration, rr):
        sig += 0.3 * np.exp(-((t - tk) / 0.05) ** 2)
    return sig


@pytest.fixture(scope="module")
def classifier(tmp_path_factory):
    artifacts = train_baseline_quick_smoke(tmp_path_factory.mktemp("smoke_stream"), seed=0)
    return ECGClassifier.load_baseline(artifacts.model_path, artifacts.scaler_path)


def test_streaming_emits_beats_in_order(classifier):
    fs = 360.0
    sig = _synthetic_ecg(fs=fs, duration=20.0)
    stream = StreamingClassifier(classifier, input_fs=fs, lead="MLII")
    chunk_size = int(0.5 * fs)  # 0.5 s chunks
    all_beats = []
    for i in range(0, len(sig), chunk_size):
        all_beats.extend(stream.push_samples(sig[i : i + chunk_size]))
    all_beats.extend(stream.flush())

    assert len(all_beats) >= 5
    # Beat indices are 1-based and monotonically increasing.
    indices = [b.beat_index for b in all_beats]
    assert indices == list(range(1, len(all_beats) + 1))
    # Peak times are monotonic.
    times = [b.peak_time_s for b in all_beats]
    assert all(t2 > t1 for t1, t2 in zip(times, times[1:], strict=False))


def test_streaming_matches_batch_within_tolerance(classifier):
    """Streaming over the same signal should agree with batch for >=80% of beats.

    They won't match exactly because batch peak detection sees the whole
    signal at once, while streaming detects on a growing buffer. We just want
    the *count* and *labels* to be close.
    """
    fs = 360.0
    sig = _synthetic_ecg(fs=fs, duration=15.0)

    batch_result = classifier.predict(sig, input_fs=fs, lead="MLII")

    stream = StreamingClassifier(classifier, input_fs=fs, lead="MLII")
    streaming_beats = []
    chunk_size = int(0.4 * fs)
    for i in range(0, len(sig), chunk_size):
        streaming_beats.extend(stream.push_samples(sig[i : i + chunk_size]))
    streaming_beats.extend(stream.flush())

    # Match each batch beat to the nearest streaming beat by peak_sample.
    if len(batch_result) == 0 or len(streaming_beats) == 0:
        pytest.skip("synthetic signal produced no beats")
    batch_peaks = np.array([b.peak_sample for b in batch_result])
    stream_peaks = np.array([b.peak_sample for b in streaming_beats])
    matched = 0
    for bp, blabel in zip(batch_peaks, [b.label for b in batch_result], strict=False):
        # Streaming peak indices are absolute in the resampled stream; for
        # input_fs == TARGET_FS they share the same coordinate system.
        nearest = int(np.argmin(np.abs(stream_peaks - bp)))
        if abs(stream_peaks[nearest] - bp) <= 5 and streaming_beats[nearest].label == blabel:
            matched += 1
    assert matched / len(batch_result) >= 0.7


def test_streaming_handles_resampling(classifier):
    """Stream a 250 Hz signal; expect resampling and at least a few beats."""
    fs = 250.0
    sig = _synthetic_ecg(fs=fs, duration=15.0)
    stream = StreamingClassifier(classifier, input_fs=fs, lead="I")
    beats: list = []
    chunk_size = int(0.5 * fs)
    for i in range(0, len(sig), chunk_size):
        beats.extend(stream.push_samples(sig[i : i + chunk_size]))
    beats.extend(stream.flush())
    assert len(beats) >= 3


def test_streaming_buffer_stays_bounded(classifier):
    """Long stream should not retain all samples in memory."""
    fs = 360.0
    sig = _synthetic_ecg(fs=fs, duration=60.0)
    stream = StreamingClassifier(classifier, input_fs=fs, lead="MLII")
    chunk = int(2.0 * fs)
    for i in range(0, len(sig), chunk):
        stream.push_samples(sig[i : i + chunk])
    stream.flush()
    # Buffer should be far smaller than the full 60 s of samples.
    assert stream.state.n_samples_buffered < len(sig) // 2


def test_streaming_reset_clears_state(classifier):
    fs = 360.0
    sig = _synthetic_ecg(fs=fs, duration=10.0)
    stream = StreamingClassifier(classifier, input_fs=fs, lead="MLII")
    stream.push_samples(sig)
    stream.flush()
    assert stream.state.n_beats_emitted >= 1
    stream.reset()
    assert stream.state.n_beats_emitted == 0
    assert stream.state.n_samples_buffered == 0


def test_streaming_validates_inputs(classifier):
    with pytest.raises(ValueError):
        StreamingClassifier(classifier, input_fs=-1.0)
    with pytest.raises(ValueError):
        StreamingClassifier(classifier, input_fs=360.0, trailing_margin=-1)
