"""Signal-level augmentation for ECG morphology windows.

The augmenter operates on already-segmented, baseline-corrected, normalised
180-sample beat windows (the same shape that comes out of
:func:`preprocessing.preprocess_windows`). It is **stateless after construction**
and **deterministic given a seed**, so results are reproducible across runs.

Critical correctness invariants:

- **RR features are not touched.** Augmentation only modifies the morphology
  window; pre/post RR ratios are computed once on the original peak indices
  and are passed through unchanged. If you ever want to augment by shifting
  the peak itself, recompute RR from the shifted peak set or RR will desync.
- **Original windows are never mutated in place.** Every transform allocates
  a fresh array.
- **Augmentation only runs in train mode.** The companion
  :class:`AugmentedECGDataset` enforces this; ad-hoc users should call
  :meth:`BeatAugmenter.apply` only on training batches.

The transform set is deliberately small and physiology-aware:

- *Amplitude scaling* — uniform multiplicative gain in ``[0.8, 1.2]``.
- *Gaussian noise* — additive Gaussian at controllable SNR (default 20 dB).
- *Baseline wander* — low-frequency sinusoid (0.1–0.5 Hz) at 5–10% peak
  amplitude. Mimics electrode drift and respiratory motion.
- *Time-warp* — local linear interpolation (±5%). Mimics small heart-rate
  variations.
- *Temporal shift* — uniform integer shift ±5 samples, then re-pad. Helps the
  model tolerate sub-window QRS-position jitter from imperfect peak detection.

Notably absent: SMOTE on signals (non-physiological waveforms) and harsh
masking (would wipe out QRS features).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .preprocessing import WINDOW_SIZE


@dataclass(frozen=True)
class AugmentSpec:
    """Probabilities + hyperparameters for the supported transforms."""

    p_amplitude: float = 0.5
    amplitude_range: tuple[float, float] = (0.8, 1.2)
    p_noise: float = 0.5
    noise_snr_db_range: tuple[float, float] = (15.0, 25.0)
    p_baseline_wander: float = 0.5
    baseline_amp_range: tuple[float, float] = (0.05, 0.10)
    baseline_freq_range: tuple[float, float] = (0.1, 0.5)  # Hz
    p_time_warp: float = 0.3
    time_warp_range: tuple[float, float] = (-0.05, 0.05)
    p_shift: float = 0.4
    shift_range: tuple[int, int] = (-5, 5)


class BeatAugmenter:
    """Apply a sequence of stochastic beat-level augmentations.

    Parameters
    ----------
    spec : AugmentSpec
        Per-transform probabilities and hyperparameters.
    fs : float
        Sampling rate of the morphology windows in Hz. Used to convert
        baseline-wander frequencies to per-sample radians.
    seed : int
        Master seed; per-call RNGs are spawned from this.
    """

    def __init__(
        self,
        spec: AugmentSpec | None = None,
        *,
        fs: float = 360.0,
        seed: int = 0,
    ) -> None:
        self.spec = spec or AugmentSpec()
        self.fs = float(fs)
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the RNG. Useful for replaying an augmentation sequence in tests."""
        self._rng = np.random.default_rng(self._seed if seed is None else seed)

    # ------------------------------------------------------------------
    # Per-window primitives (pure: take + return arrays, no internal state)
    # ------------------------------------------------------------------
    def amplitude_scale(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.spec.amplitude_range
        gain = float(rng.uniform(lo, hi))
        return window * gain

    def add_gaussian_noise(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.spec.noise_snr_db_range
        snr_db = float(rng.uniform(lo, hi))
        signal_power = float(np.mean(window**2))
        if signal_power <= 0:
            return window.copy()
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=window.shape)
        return window + noise

    def baseline_wander(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        amp_lo, amp_hi = self.spec.baseline_amp_range
        f_lo, f_hi = self.spec.baseline_freq_range
        amp = float(rng.uniform(amp_lo, amp_hi))
        freq = float(rng.uniform(f_lo, f_hi))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        n = len(window)
        t = np.arange(n) / self.fs
        # Scale wander by signal amplitude so it stays a relative perturbation.
        s_amp = float(np.std(window)) or 1.0
        wander = amp * s_amp * np.sin(2.0 * np.pi * freq * t + phase)
        return window + wander

    def time_warp(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Resample window length by ``1 + r`` then re-stretch back to ``WINDOW_SIZE``.

        Equivalent to a tiny global time-stretch; preserves morphology but
        slightly compresses or expands the QRS complex.
        """
        lo, hi = self.spec.time_warp_range
        r = float(rng.uniform(lo, hi))
        n = len(window)
        n_new = max(2, int(round(n * (1.0 + r))))
        # Linear interpolation onto a ``n_new``-length grid, then back to ``n``.
        x_orig = np.linspace(0.0, 1.0, n)
        x_warp = np.linspace(0.0, 1.0, n_new)
        warped = np.interp(x_warp, x_orig, window)
        return np.interp(x_orig, np.linspace(0.0, 1.0, n_new), warped).astype(window.dtype)

    def temporal_shift(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.spec.shift_range
        s = int(rng.integers(lo, hi + 1))
        if s == 0:
            return window.copy()
        out = np.empty_like(window)
        if s > 0:
            out[:s] = window[0]
            out[s:] = window[:-s]
        else:
            out[s:] = window[-1]
            out[:s] = window[-s:]
        return out

    # ------------------------------------------------------------------
    # Batch entry point
    # ------------------------------------------------------------------
    def apply(self, windows: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Apply all configured transforms to a batch of windows.

        Each transform fires independently per-window with its configured
        probability. The output is a fresh array of the same shape and dtype.
        """
        if rng is None:
            rng = self._rng
        if windows.ndim == 1:
            windows = windows[None, :]
        if windows.shape[-1] != WINDOW_SIZE:
            raise ValueError(
                f"BeatAugmenter expects windows of length {WINDOW_SIZE}; "
                f"got {windows.shape[-1]}."
            )
        out = np.empty_like(windows, dtype=float)
        spec = self.spec
        for i in range(len(windows)):
            w = windows[i].astype(float, copy=True)
            if rng.random() < spec.p_amplitude:
                w = self.amplitude_scale(w, rng)
            if rng.random() < spec.p_noise:
                w = self.add_gaussian_noise(w, rng)
            if rng.random() < spec.p_baseline_wander:
                w = self.baseline_wander(w, rng)
            if rng.random() < spec.p_time_warp:
                w = self.time_warp(w, rng)
            if rng.random() < spec.p_shift:
                w = self.temporal_shift(w, rng)
            out[i] = w
        return out


# ---------------------------------------------------------------------------
# Torch-side wrapper
# ---------------------------------------------------------------------------
class AugmentedECGDataset:
    """Lightweight torch ``Dataset`` over pre-loaded numpy arrays.

    The arrays are loaded once into memory by the training pipeline (this is
    fine for MIT-BIH at ~70k beats). Augmentation runs on-the-fly inside
    ``__getitem__``, so DataLoader workers can run it in parallel without ever
    touching WFDB.

    Notes
    -----
    - Augmentation is applied **only when** ``augmenter is not None``. Set to
      ``None`` for val/test loaders.
    - RR features are passed through verbatim. See :mod:`ecg_arrhythmia.augment`
      module docstring for the rationale.
    """

    def __init__(
        self,
        morph: np.ndarray,
        rr: np.ndarray,
        y: np.ndarray,
        *,
        in_channels: int,
        samples_per_lead: int,
        augmenter: BeatAugmenter | None = None,
    ) -> None:
        if len(morph) != len(rr) or len(morph) != len(y):
            raise ValueError(
                f"length mismatch: morph={len(morph)}, rr={len(rr)}, y={len(y)}"
            )
        self.morph = np.asarray(morph, dtype=np.float32)
        self.rr = np.asarray(rr, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.in_channels = int(in_channels)
        self.samples_per_lead = int(samples_per_lead)
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.y)

    def _augment_one(self, morph_row: np.ndarray, idx: int) -> np.ndarray:
        if self.augmenter is None:
            return morph_row
        # Per-sample RNG so worker shards stay independent and reproducible.
        rng = np.random.default_rng(self.augmenter._seed + idx)
        aug = self.augmenter.apply(morph_row.reshape(1, -1), rng=rng)
        return aug[0]

    def __getitem__(self, idx: int):  # returns (morph_tensor, rr_tensor, y_tensor)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("AugmentedECGDataset requires PyTorch") from exc
        morph_row = self.morph[idx]
        # Multi-lead concat: split → augment per-lead → re-concat.
        if self.in_channels > 1 and self.augmenter is not None:
            n = self.samples_per_lead
            pieces = [
                self._augment_one(morph_row[ci * n : (ci + 1) * n], idx + ci)
                for ci in range(self.in_channels)
            ]
            morph_aug = np.concatenate(pieces, axis=0)
        else:
            morph_aug = self._augment_one(morph_row, idx)
        morph_t = torch.from_numpy(
            morph_aug.astype(np.float32).reshape(self.in_channels, self.samples_per_lead)
        )
        return morph_t, torch.from_numpy(self.rr[idx]), torch.tensor(int(self.y[idx]), dtype=torch.long)


__all__ = ["AugmentSpec", "BeatAugmenter", "AugmentedECGDataset"]


# Quiet the unused-import linter for ``Sequence`` if a future edit drops it.
_ = Sequence
