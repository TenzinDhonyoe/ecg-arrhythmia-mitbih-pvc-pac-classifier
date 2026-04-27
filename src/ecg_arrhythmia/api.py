"""High-level Python API for ECG beat classification.

The CLI in :mod:`ecg_arrhythmia.infer` is convenient for one-off scripts, but
people building applications want an importable, stateful classifier they can
hand a numpy array and get predictions back from. This module is that surface.

Example
-------
>>> import numpy as np
>>> from ecg_arrhythmia import ECGClassifier
>>> clf = ECGClassifier.from_artifacts("artifacts/baseline")
>>> signal = np.loadtxt("examples/sample_ecg.csv", delimiter=",", skiprows=1)
>>> result = clf.predict(signal, input_fs=360, lead="MLII")
>>> result.summary()  # doctest: +SKIP
{'n_beats': 23, 'mean_hr_bpm': 71.5, 'class_counts': {'N': 22, 'V': 1, 'a': 0}}

The same ``ECGClassifier`` works for both the LogisticRegression baseline and
the ResNet-1D model — :meth:`from_artifacts` looks at what files exist in the
directory and picks the right backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .inference import (
    KNOWN_LEADS,
    TARGET_FS,
    WearableInferenceConfig,
    auto_polarity_check,
    detect_r_peaks,
    resample_signal,
    windows_from_peaks,
)
from .labels import MITBIH3, LabelScheme, get_scheme
from .preprocessing import compute_rr_features, preprocess_windows

Backend = Literal["baseline", "resnet"]


@dataclass(frozen=True)
class BeatPrediction:
    """A single classified beat.

    Attributes
    ----------
    beat_index : int
        1-based position of this beat in the prediction batch.
    peak_sample : int
        Sample index of the R-peak in the resampled (360 Hz) signal.
    peak_time_s : float
        R-peak time in seconds, measured against the resampled 360 Hz timeline.
    label : str
        Predicted class — one of ``"N"`` (Normal), ``"V"`` (PVC), ``"a"`` (PAC).
    confidence : float
        Softmax probability of the predicted class.
    probabilities : dict[str, float]
        Full per-class probability dict, e.g. ``{"N": 0.95, "V": 0.04, "a": 0.01}``.
    """

    beat_index: int
    peak_sample: int
    peak_time_s: float
    label: str
    confidence: float
    probabilities: dict[str, float]
    uncertainty: float | None = None  # epistemic entropy from MC dropout, if available

    def topk(self, k: int = 2) -> list[tuple[str, float]]:
        """Top-k labels with their probabilities, sorted descending."""
        items = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return items[:k]

    def to_dict(self) -> dict:
        """Flat JSON-friendly dict.

        Emits ``prob_<label>`` for each class label in :attr:`probabilities`.
        For the legacy 3-class scheme this naturally produces
        ``prob_N``/``prob_V``/``prob_a``, identical to v0.3 output.
        """
        out = {
            "beat_index": self.beat_index,
            "peak_sample_360hz": self.peak_sample,
            "peak_time_s": self.peak_time_s,
            "label": self.label,
            "confidence": self.confidence,
        }
        for label, p in self.probabilities.items():
            out[f"prob_{label}"] = float(p)
        if self.uncertainty is not None:
            out["uncertainty"] = float(self.uncertainty)
        return out


@dataclass
class PredictionResult:
    """A batch of beat predictions plus per-run metadata."""

    beats: list[BeatPrediction]
    info: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.beats)

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, idx) -> BeatPrediction:
        return self.beats[idx]

    @property
    def labels(self) -> list[str]:
        return [b.label for b in self.beats]

    @property
    def confidences(self) -> np.ndarray:
        return np.asarray([b.confidence for b in self.beats], dtype=float)

    @property
    def probabilities(self) -> np.ndarray:
        """Shape ``(n_beats, n_classes)`` array of per-class probabilities.

        Column order matches the order of keys in the first beat's
        ``probabilities`` dict, which itself follows the loaded scheme's
        ``id_to_label`` order (e.g. ``[N, V, a]`` for MITBIH3 or
        ``[N, S, V, F, Q]`` for AAMI5).
        """
        if not self.beats:
            return np.empty((0, 0), dtype=float)
        labels = list(self.beats[0].probabilities.keys())
        return np.asarray(
            [[b.probabilities[lab] for lab in labels] for b in self.beats],
            dtype=float,
        )

    def class_counts(self) -> dict[str, int]:
        """Per-class beat counts. Includes zero entries for classes that didn't fire."""
        counts: dict[str, int] = {}
        if self.beats:
            for label in self.beats[0].probabilities:
                counts[label] = 0
        for b in self.beats:
            counts[b.label] = counts.get(b.label, 0) + 1
        return counts

    def heart_rate_bpm(self) -> float | None:
        """Mean heart rate from R-R intervals, or None if <2 beats."""
        if len(self.beats) < 2:
            return None
        peak_times = np.asarray([b.peak_time_s for b in self.beats], dtype=float)
        rr = np.diff(peak_times)
        rr = rr[rr > 0]
        if len(rr) == 0:
            return None
        return float(60.0 / np.mean(rr))

    def summary(self) -> dict:
        """Compact run summary suitable for logging / JSON output."""
        return {
            "n_beats": len(self.beats),
            "mean_hr_bpm": self.heart_rate_bpm(),
            "class_counts": self.class_counts(),
            "mean_confidence": float(np.mean(self.confidences)) if self.beats else None,
            **self.info,
        }

    def to_records(self) -> list[dict]:
        return [b.to_dict() for b in self.beats]


class ECGClassifier:
    """One classifier object, two possible backends, one ``predict()``.

    Construct via :meth:`from_artifacts` (recommended) or pass loaded objects
    directly. The class is intentionally stateless except for the loaded model
    weights, so a single instance is safe to call repeatedly across signals.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        sklearn_model=None,
        sklearn_scaler=None,
        torch_model=None,
        torch_config: dict | None = None,
        scheme: LabelScheme = MITBIH3,
        temperature: float | None = None,
    ) -> None:
        if backend == "baseline":
            if sklearn_model is None or sklearn_scaler is None:
                raise ValueError("baseline backend requires sklearn_model and sklearn_scaler")
        elif backend == "resnet":
            if torch_model is None or torch_config is None:
                raise ValueError("resnet backend requires torch_model and torch_config")
        else:
            raise ValueError(f"Unknown backend {backend!r}")
        self.backend: Backend = backend
        self.scheme: LabelScheme = scheme
        self.temperature: float | None = float(temperature) if temperature is not None else None
        self._sklearn_model = sklearn_model
        self._sklearn_scaler = sklearn_scaler
        self._torch_model = torch_model
        self._torch_config = torch_config

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str | Path,
        *,
        prefer: Backend | None = None,
    ) -> ECGClassifier:
        """Load a classifier by sniffing the artifacts directory.

        Looks for the standard filenames produced by ``ecg-train``:

        - baseline: ``baseline_lr_mitdb.joblib`` + ``baseline_lr_scaler.joblib``
        - resnet:   ``resnet1d.pt`` + ``model_config.json``

        If both exist, ``prefer`` resolves the tie (default: ``"resnet"`` if
        torch is importable, else ``"baseline"``).
        """
        d = Path(artifacts_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"artifacts dir not found: {d}")

        baseline_model = d / "baseline_lr_mitdb.joblib"
        baseline_scaler = d / "baseline_lr_scaler.joblib"
        resnet_weights = d / "resnet1d.pt"
        resnet_config = d / "model_config.json"

        has_baseline = baseline_model.is_file() and baseline_scaler.is_file()
        has_resnet = resnet_weights.is_file() and resnet_config.is_file()

        if not has_baseline and not has_resnet:
            raise FileNotFoundError(
                f"No recognised model artifacts under {d}. Expected either "
                "(baseline_lr_mitdb.joblib + baseline_lr_scaler.joblib) or "
                "(resnet1d.pt + model_config.json)."
            )

        if prefer is None:
            try:
                import torch  # noqa: F401

                prefer = "resnet" if has_resnet else "baseline"
            except ImportError:
                prefer = "baseline"

        # Auto-load temperature.pt if present so the returned classifier is
        # already calibrated.
        temperature = _sniff_temperature(d)

        if prefer == "resnet" and has_resnet:
            clf = cls.load_resnet(resnet_weights, resnet_config)
        elif prefer == "baseline" and has_baseline:
            clf = cls.load_baseline(baseline_model, baseline_scaler)
        elif has_resnet:
            clf = cls.load_resnet(resnet_weights, resnet_config)
        else:
            clf = cls.load_baseline(baseline_model, baseline_scaler)
        if temperature is not None and clf.backend == "resnet":
            clf.temperature = float(temperature)
        return clf

    @classmethod
    def load_baseline(
        cls,
        model_path: str | Path,
        scaler_path: str | Path,
        *,
        scheme: LabelScheme | None = None,
    ) -> ECGClassifier:
        import joblib

        model = joblib.load(Path(model_path))
        scaler = joblib.load(Path(scaler_path))
        # Sniff scheme from a sibling metrics_*.json if not supplied.
        if scheme is None:
            scheme = _sniff_scheme(Path(model_path).parent) or MITBIH3
        return cls(
            backend="baseline",
            sklearn_model=model,
            sklearn_scaler=scaler,
            scheme=scheme,
        )

    @classmethod
    def load_resnet(
        cls,
        weights_path: str | Path,
        config_path: str | Path,
        *,
        scheme: LabelScheme | None = None,
    ) -> ECGClassifier:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ResNet backend requires PyTorch. Install with: pip install -e '.[deep]'"
            ) from exc
        from .models.resnet1d import ResNet1D

        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        # Strip our bookkeeping fields. The remaining keys go straight into
        # the relevant model's constructor.
        scheme_name = config.pop("scheme", None) if isinstance(config, dict) else None
        architecture = config.pop("architecture", "resnet") if isinstance(config, dict) else "resnet"

        if architecture == "cnn_transformer":
            from .models.cnn_transformer import CNNTransformer1D

            cnn_kwargs = {
                k: v for k, v in config.items()
                if k in {"in_channels", "samples_per_lead", "rr_features", "n_classes",
                         "cnn_channels", "cnn_blocks_per_stage", "d_model", "nhead",
                         "num_layers", "dim_feedforward", "dropout",
                         "use_se", "use_cls_token"}
            }
            model = CNNTransformer1D(**cnn_kwargs)
            model_kwargs = cnn_kwargs
        else:
            model_kwargs = {
                k: v for k, v in config.items()
                if k in {"in_channels", "samples_per_lead", "rr_features", "n_classes",
                         "channels", "blocks_per_stage", "use_se", "stem"}
            }
            model = ResNet1D(**model_kwargs)
        state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        if scheme is None:
            if scheme_name is not None:
                scheme = get_scheme(scheme_name)
            else:
                # Fall back to sniffing metrics_*.json or, ultimately, MITBIH3.
                scheme = _sniff_scheme(Path(weights_path).parent) or MITBIH3
        return cls(
            backend="resnet",
            torch_model=model,
            torch_config=model_kwargs,
            scheme=scheme,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        signal: np.ndarray,
        input_fs: float = TARGET_FS,
        *,
        lead: str = "unknown",
        auto_polarity: bool = True,
        invert_polarity: bool = False,
    ) -> PredictionResult:
        """Classify every detected beat in a single-channel ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            1-D float array of ECG samples.
        input_fs : float
            Sampling rate of ``signal`` in Hz. Resampled to 360 Hz internally.
        lead : str
            Lead label (``"MLII"``, ``"II"``, ``"I"``, ...). Affects domain-shift warnings.
        auto_polarity : bool
            If True, flip the signal when the negative tail dominates the positive tail.
        invert_polarity : bool
            If True, force-invert before peak detection (overrides auto-polarity).
        """
        signal = np.asarray(signal, dtype=float).ravel()
        signal = signal[np.isfinite(signal)]

        if lead not in KNOWN_LEADS:
            import warnings

            warnings.warn(
                f"Unknown lead name {lead!r}; expected one of {sorted(KNOWN_LEADS)}",
                stacklevel=2,
            )

        info: dict = {
            "n_samples": int(signal.size),
            "input_fs": float(input_fs),
            "lead": lead,
            "polarity_flipped": False,
            "domain_shift_warning": False,
            "backend": self.backend,
        }

        if signal.size == 0:
            return PredictionResult(beats=[], info=info)

        if input_fs != TARGET_FS:
            signal = resample_signal(signal, input_fs, TARGET_FS)

        flip = invert_polarity
        if auto_polarity and not flip:
            flip = auto_polarity_check(signal, TARGET_FS)
        if flip:
            signal = -signal
            info["polarity_flipped"] = True

        if lead not in {"MLII", "II", "unknown"}:
            import warnings

            warnings.warn(
                f"Model was trained on MIT-BIH MLII; expect a noticeable accuracy drop on lead "
                f"{lead!r}. PVC/PAC recall in particular depends on QRS morphology, "
                "which is lead-specific.",
                stacklevel=2,
            )
            info["domain_shift_warning"] = True

        peaks = detect_r_peaks(signal, TARGET_FS)
        windows, peak_idx = windows_from_peaks(signal, peaks)
        info["n_beats"] = int(len(windows))
        if len(windows) == 0:
            return PredictionResult(beats=[], info=info)

        morph = preprocess_windows(windows)
        rr = compute_rr_features(peak_idx)
        proba = self._predict_proba(morph, rr)
        beats = self._beats_from_proba(peak_idx, proba)
        return PredictionResult(beats=beats, info=info)

    def predict_csv(
        self,
        csv_path: str | Path,
        input_fs: float = TARGET_FS,
        *,
        lead: str = "unknown",
        auto_polarity: bool = True,
        invert_polarity: bool = False,
    ) -> PredictionResult:
        """Convenience wrapper for the common CSV-loading case."""
        from .inference import load_signal_csv

        signal = load_signal_csv(Path(csv_path))
        return self.predict(
            signal,
            input_fs=input_fs,
            lead=lead,
            auto_polarity=auto_polarity,
            invert_polarity=invert_polarity,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _predict_proba(self, morph: np.ndarray, rr: np.ndarray) -> np.ndarray:
        if self.backend == "baseline":
            X = np.hstack([morph, rr])
            Xs = self._sklearn_scaler.transform(X)
            return np.asarray(self._sklearn_model.predict_proba(Xs), dtype=float)

        # resnet — optionally apply temperature scaling.
        import torch

        cfg = self._torch_config
        n_samples = cfg["samples_per_lead"]
        in_channels = cfg["in_channels"]
        morph_t = torch.from_numpy(morph.astype(np.float32)).reshape(-1, in_channels, n_samples)
        rr_t = torch.from_numpy(rr.astype(np.float32))
        with torch.no_grad():
            logits = self._torch_model(morph_t, rr_t)
            if self.temperature is not None and self.temperature > 0:
                logits = logits / self.temperature
            proba = torch.softmax(logits, dim=1).numpy()
        return np.asarray(proba, dtype=float)

    def predict_with_uncertainty(
        self,
        signal: np.ndarray,
        input_fs: float = TARGET_FS,
        *,
        lead: str = "unknown",
        auto_polarity: bool = True,
        invert_polarity: bool = False,
        n_mc: int = 20,
    ) -> PredictionResult:
        """Like :meth:`predict`, but returns MC-dropout-based uncertainty per beat.

        For the LR baseline the dropout path is a no-op; uncertainty is
        reported as ``None`` on every beat in that case.
        """
        result = self.predict(
            signal,
            input_fs=input_fs,
            lead=lead,
            auto_polarity=auto_polarity,
            invert_polarity=invert_polarity,
        )
        if self.backend != "resnet" or len(result) == 0:
            return result

        # Re-run the (already-prepared) inputs through MC-dropout. The cleanest
        # way is to recover morph/rr from the original peak indices using the
        # same pipeline `predict` used. We keep the predictions but overwrite
        # the per-beat uncertainty.
        from .calibration import MCDropoutEnsemble

        cfg = self._torch_config
        n_samples = cfg["samples_per_lead"]
        in_channels = cfg["in_channels"]

        # Reconstruct features from `predict` was a clean read. The result
        # objects already carry peak_sample (in 360 Hz coordinates), but the
        # actual morph/rr arrays were transient. Easiest: we re-prepare from
        # the same signal exactly as `predict` did (deterministic).
        from .inference import (
            KNOWN_LEADS,
            auto_polarity_check,
            detect_r_peaks,
            resample_signal,
            windows_from_peaks,
        )
        from .preprocessing import compute_rr_features, preprocess_windows

        signal = np.asarray(signal, dtype=float).ravel()
        signal = signal[np.isfinite(signal)]
        if lead not in KNOWN_LEADS:
            pass  # already warned by predict()
        if input_fs != TARGET_FS:
            signal = resample_signal(signal, input_fs, TARGET_FS)
        flip = invert_polarity
        if auto_polarity and not flip:
            flip = auto_polarity_check(signal, TARGET_FS)
        if flip:
            signal = -signal
        peaks = detect_r_peaks(signal, TARGET_FS)
        windows, peak_idx = windows_from_peaks(signal, peaks)
        if len(windows) == 0:
            return result
        morph = preprocess_windows(windows)
        rr = compute_rr_features(peak_idx)

        import torch

        morph_t = torch.from_numpy(morph.astype(np.float32)).reshape(-1, in_channels, n_samples)
        rr_t = torch.from_numpy(rr.astype(np.float32))
        ensemble = MCDropoutEnsemble(self._torch_model, n_passes=n_mc)
        _, entropies = ensemble.predict_proba(morph_t, rr_t)

        # Rebuild beats with the new uncertainty field. Predict() already gave
        # us the deterministic-pass labels and probs; we keep those and just
        # decorate.
        new_beats = []
        for beat, ent in zip(result.beats, entropies, strict=False):
            new_beats.append(
                BeatPrediction(
                    beat_index=beat.beat_index,
                    peak_sample=beat.peak_sample,
                    peak_time_s=beat.peak_time_s,
                    label=beat.label,
                    confidence=beat.confidence,
                    probabilities=beat.probabilities,
                    uncertainty=float(ent),
                )
            )
        return PredictionResult(beats=new_beats, info={**result.info, "n_mc": int(n_mc)})

    def _beats_from_proba(self, peak_idx: np.ndarray, proba: np.ndarray) -> list[BeatPrediction]:
        id_to_label = self.scheme.id_to_label
        n_classes = self.scheme.n_classes
        beats: list[BeatPrediction] = []
        for i, (p, row) in enumerate(zip(peak_idx, proba, strict=False), start=1):
            yhat = int(np.argmax(row))
            probs = {id_to_label[k]: float(row[k]) for k in range(n_classes)}
            beats.append(
                BeatPrediction(
                    beat_index=i,
                    peak_sample=int(p),
                    peak_time_s=float(p) / TARGET_FS,
                    label=id_to_label[yhat],
                    confidence=float(row[yhat]),
                    probabilities=probs,
                )
            )
        return beats


def _sniff_temperature(artifacts_dir: Path) -> float | None:
    """Read a fitted temperature scalar from ``artifacts_dir/temperature.pt``.

    Returns ``None`` if the file is missing or unreadable. We don't blow up on
    a broken file because ``from_artifacts`` should still load the model.
    """
    path = artifacts_dir / "temperature.pt"
    if not path.is_file():
        return None
    try:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=True)
        return float(state["temperature"])
    except (ImportError, OSError, KeyError, RuntimeError):
        return None


def _sniff_scheme(artifacts_dir: Path) -> LabelScheme | None:
    """Read a scheme name out of a sibling metrics_*.json, if any.

    Each `train_*` writer in :mod:`ecg_arrhythmia.training` records the active
    scheme into the ``"scheme"`` field of its metrics file. We look for any
    ``metrics_*.json`` and return the first scheme we recognise.
    """
    if not artifacts_dir.is_dir():
        return None
    for path in sorted(artifacts_dir.glob("metrics_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = data.get("scheme")
        if isinstance(name, str):
            try:
                return get_scheme(name)
            except ValueError:
                continue
    return None


__all__ = [
    "ECGClassifier",
    "BeatPrediction",
    "PredictionResult",
    "WearableInferenceConfig",
]
