"""Streaming / online beat-by-beat ECG classification.

The CLI and :class:`ecg_arrhythmia.ECGClassifier` operate on whole signals.
Real-time wearable applications need to process samples as they arrive and emit
predictions per beat without re-running the full pipeline on the entire history.

:class:`StreamingClassifier` maintains an internal sample buffer at the model's
target rate (360 Hz). Each call to :meth:`push_samples` returns
:class:`BeatPrediction` objects for any beats whose 180-sample window is now
fully present and trailed by enough signal to be stable. Already-emitted beats
are not re-emitted, and the buffer is trimmed periodically to keep memory
bounded.

This module is intentionally lightweight — it composes :class:`ECGClassifier`
rather than replacing it, so model improvements automatically propagate.

Example
-------
>>> import numpy as np
>>> from ecg_arrhythmia import ECGClassifier
>>> from ecg_arrhythmia.streaming import StreamingClassifier
>>> clf = ECGClassifier.from_artifacts("artifacts/baseline")  # doctest: +SKIP
>>> stream = StreamingClassifier(clf, input_fs=250.0, lead="I")  # doctest: +SKIP
>>> for chunk in chunks_of_signal:  # doctest: +SKIP
...     for beat in stream.push_samples(chunk):
...         print(beat.beat_index, beat.label, beat.confidence)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .api import BeatPrediction, ECGClassifier
from .inference import (
    KNOWN_LEADS,
    TARGET_FS,
    auto_polarity_check,
    detect_r_peaks,
    resample_signal,
)
from .preprocessing import HALF_WINDOW, WINDOW_SIZE, compute_rr_features, preprocess_windows

# Wait until at least this many samples have arrived past an R-peak before
# emitting a prediction for it. The window itself needs HALF_WINDOW; we add a
# small safety margin so a peak detected near the buffer edge is not classified
# until the trailing samples are stable.
DEFAULT_TRAILING_MARGIN = 30  # ~83 ms at 360 Hz

# Minimum number of samples to keep in the buffer when trimming, so peak
# detection has enough context for the next pass.
DEFAULT_BUFFER_FLOOR_S = 4.0  # 4 s of context


@dataclass
class StreamingState:
    """Snapshot of the classifier's internal state, useful for debugging."""

    n_samples_buffered: int
    n_samples_consumed: int
    n_beats_emitted: int


class StreamingClassifier:
    """Beat-by-beat streaming classifier on top of :class:`ECGClassifier`.

    Parameters
    ----------
    classifier : ECGClassifier
        Pre-loaded classifier (any backend).
    input_fs : float
        Sampling rate of incoming chunks in Hz. Internally resampled to 360 Hz.
    lead : str
        Lead label for domain-shift warning bookkeeping.
    auto_polarity : bool
        Run a one-shot polarity check on the first ``polarity_window_s`` of
        signal and lock the polarity for the rest of the stream.
    invert_polarity : bool
        Force-invert (overrides auto-polarity).
    trailing_margin : int
        Extra samples (at 360 Hz) required past an R-peak before that beat is
        emitted. Trades latency for stability.
    polarity_window_s : float
        Seconds of buffered signal to use for the one-shot polarity decision.
    """

    def __init__(
        self,
        classifier: ECGClassifier,
        *,
        input_fs: float = TARGET_FS,
        lead: str = "unknown",
        auto_polarity: bool = True,
        invert_polarity: bool = False,
        trailing_margin: int = DEFAULT_TRAILING_MARGIN,
        polarity_window_s: float = 5.0,
    ) -> None:
        if input_fs <= 0:
            raise ValueError(f"input_fs must be positive, got {input_fs}")
        if trailing_margin < 0:
            raise ValueError(f"trailing_margin must be non-negative, got {trailing_margin}")
        if lead not in KNOWN_LEADS:
            import warnings

            warnings.warn(
                f"Unknown lead name {lead!r}; expected one of {sorted(KNOWN_LEADS)}",
                stacklevel=2,
            )
        self.classifier = classifier
        self.input_fs = float(input_fs)
        self.lead = lead
        self.auto_polarity = auto_polarity
        self._polarity_locked: bool | None = None if auto_polarity and not invert_polarity else invert_polarity
        self.trailing_margin = int(trailing_margin)
        self.polarity_window_samples = int(polarity_window_s * TARGET_FS)
        self._buffer = np.empty(0, dtype=float)
        self._buffer_origin: int = 0  # absolute sample index (360 Hz) of buffer[0]
        self._emitted_peak_indices: set[int] = set()
        self._emitted_peaks_ordered: deque[int] = deque(maxlen=2048)
        self._n_beats_emitted: int = 0
        self._domain_warned = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all state. Call when starting a new recording."""
        self._buffer = np.empty(0, dtype=float)
        self._buffer_origin = 0
        self._emitted_peak_indices.clear()
        self._emitted_peaks_ordered.clear()
        self._n_beats_emitted = 0
        if self.auto_polarity:
            self._polarity_locked = None

    @property
    def state(self) -> StreamingState:
        return StreamingState(
            n_samples_buffered=int(self._buffer.size),
            n_samples_consumed=int(self._buffer_origin),
            n_beats_emitted=self._n_beats_emitted,
        )

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------
    def push_samples(self, chunk: Iterable[float]) -> list[BeatPrediction]:
        """Append samples to the buffer and return any newly-emittable beats."""
        arr = np.asarray(list(chunk) if not isinstance(chunk, np.ndarray) else chunk, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return []

        if self.input_fs != TARGET_FS:
            arr = resample_signal(arr, self.input_fs, TARGET_FS)

        # Lazy domain-shift warning, once per stream.
        if not self._domain_warned and self.lead not in {"MLII", "II", "unknown"}:
            import warnings

            warnings.warn(
                f"Model was trained on MIT-BIH MLII; expect a noticeable accuracy drop on lead "
                f"{self.lead!r}. Streaming output should not be used for clinical decisions.",
                stacklevel=2,
            )
            self._domain_warned = True

        self._buffer = np.concatenate([self._buffer, arr]) if self._buffer.size else arr.astype(float, copy=True)

        # Lock polarity once we've seen enough samples for a confident decision.
        if self._polarity_locked is None and self._buffer.size >= max(self.polarity_window_samples, int(2 * TARGET_FS)):
            self._polarity_locked = bool(auto_polarity_check(self._buffer[: self.polarity_window_samples], TARGET_FS))

        if self._polarity_locked is True:
            # Apply polarity flip in-place to the just-appended portion. We flip
            # the whole buffer the first time the lock kicks in, then only the
            # new tail thereafter.
            new_chunk_len = int(arr.size)
            if not getattr(self, "_polarity_applied", False):
                self._buffer = -self._buffer
                self._polarity_applied = True
            else:
                self._buffer[-new_chunk_len:] = -self._buffer[-new_chunk_len:]

        # If we still haven't decided polarity, hold off on emitting beats.
        if self._polarity_locked is None:
            return []

        return self._scan_and_emit()

    def flush(self) -> list[BeatPrediction]:
        """Emit any remaining beats, ignoring the trailing-margin requirement.

        Use at end-of-stream when no more samples will arrive.
        """
        old_margin = self.trailing_margin
        self.trailing_margin = 0
        try:
            out = self._scan_and_emit()
        finally:
            self.trailing_margin = old_margin
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _scan_and_emit(self) -> list[BeatPrediction]:
        if self._buffer.size < WINDOW_SIZE + self.trailing_margin:
            return []

        peaks_local = detect_r_peaks(self._buffer, TARGET_FS)
        if peaks_local.size == 0:
            self._maybe_trim_buffer(latest_peak_local=None)
            return []

        # Convert local (within-buffer) peak indices to absolute (since stream start).
        peaks_abs = peaks_local + self._buffer_origin

        emittable_local: list[int] = []
        emittable_abs: list[int] = []
        for p_local, p_abs in zip(peaks_local, peaks_abs, strict=False):
            if int(p_abs) in self._emitted_peak_indices:
                continue
            window_end = p_local + HALF_WINDOW
            window_start = p_local - HALF_WINDOW
            if window_start < 0:
                continue
            if window_end + self.trailing_margin > self._buffer.size:
                # Not enough trailing signal yet.
                break
            emittable_local.append(int(p_local))
            emittable_abs.append(int(p_abs))

        if not emittable_local:
            self._maybe_trim_buffer(latest_peak_local=int(peaks_local[-1]))
            return []

        # Build feature matrix from the emittable peaks only.
        windows = np.stack(
            [self._buffer[p - HALF_WINDOW : p + HALF_WINDOW] for p in emittable_local],
            axis=0,
        )
        morph = preprocess_windows(windows)
        # RR features need neighbouring peaks; use the absolute peak indices for
        # this batch but include their immediate neighbours where available.
        rr_peaks = np.asarray(emittable_abs, dtype=int)
        rr = compute_rr_features(rr_peaks)

        proba = self.classifier._predict_proba(morph, rr)

        scheme = self.classifier.scheme
        id_to_label = scheme.id_to_label
        n_classes = scheme.n_classes
        new_beats: list[BeatPrediction] = []
        for p_abs, row in zip(emittable_abs, proba, strict=False):
            self._n_beats_emitted += 1
            yhat = int(np.argmax(row))
            probs = {id_to_label[k]: float(row[k]) for k in range(n_classes)}
            new_beats.append(
                BeatPrediction(
                    beat_index=self._n_beats_emitted,
                    peak_sample=int(p_abs),
                    peak_time_s=float(p_abs) / TARGET_FS,
                    label=id_to_label[yhat],
                    confidence=float(row[yhat]),
                    probabilities=probs,
                )
            )
            self._emitted_peak_indices.add(int(p_abs))
            self._emitted_peaks_ordered.append(int(p_abs))

        self._maybe_trim_buffer(latest_peak_local=int(peaks_local[-1]))
        return new_beats

    def _maybe_trim_buffer(self, latest_peak_local: int | None) -> None:
        """Drop already-emitted prefix of the buffer to keep memory bounded."""
        floor = int(DEFAULT_BUFFER_FLOOR_S * TARGET_FS)
        if self._buffer.size <= floor * 2:
            return
        # Keep at least the last `floor` samples plus one window's worth of
        # context behind the most recent peak (so RR features remain stable).
        keep_from_end = max(floor, WINDOW_SIZE * 4)
        if latest_peak_local is not None:
            keep_from_end = max(keep_from_end, self._buffer.size - latest_peak_local + WINDOW_SIZE)
        if keep_from_end >= self._buffer.size:
            return
        drop = self._buffer.size - keep_from_end
        self._buffer = self._buffer[drop:].copy()
        self._buffer_origin += drop


__all__ = ["StreamingClassifier", "StreamingState"]
