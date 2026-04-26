"""Inference helpers for CSV ECG signals.

Two backends are supported:

- ``run_baseline_inference``: scikit-learn logistic regression (no extra deps).
- ``run_resnet_inference``: ResNet-1D + RR head (requires the ``[deep]`` extra).

The "wearable" path (``WearableInferenceConfig``, ``auto_polarity_check``) is a
thin pre-processing layer on top of the standard pipeline. It exists because
consumer single-lead devices (Apple Watch, KardiaMobile, Withings) often
deliver Lead-I morphology, lower sampling rates, and sometimes inverted
polarity — none of which match the MIT-BIH MLII training distribution. The
wearable helpers do their best to align the input before the model sees it
and emit a domain-shift warning, but they cannot eliminate the lead mismatch.
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample

from .labels import ID_TO_LABEL
from .preprocessing import HALF_WINDOW, WINDOW_SIZE, compute_rr_features, preprocess_windows

TARGET_FS = 360.0

KNOWN_LEADS = {"I", "II", "III", "MLII", "V1", "V2", "V5", "unknown"}


@dataclass
class WearableInferenceConfig:
    input_fs: float = TARGET_FS
    lead_name: str = "unknown"
    invert_polarity: bool = False
    auto_polarity: bool = True

    def __post_init__(self) -> None:
        if self.lead_name not in KNOWN_LEADS:
            warnings.warn(
                f"Unknown lead name {self.lead_name!r}; expected one of {sorted(KNOWN_LEADS)}",
                stacklevel=2,
            )


def load_signal_csv(path: Path) -> np.ndarray:
    """Read a single-channel ECG signal from CSV.

    Recognises a ``signal`` column (case-insensitive) or falls back to the first
    numeric column. Empty rows and non-numeric values are skipped.
    """
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return np.array([], dtype=float)
        header_l = [h.strip().lower() for h in header]
        sig_col = header_l.index("signal") if "signal" in header_l else 0
        vals = []
        for row in reader:
            if not row:
                continue
            try:
                vals.append(float(row[sig_col]))
            except (ValueError, IndexError):
                continue
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def resample_signal(signal: np.ndarray, input_fs: float, output_fs: float = TARGET_FS) -> np.ndarray:
    if input_fs == output_fs:
        return signal.astype(float)
    n_out = max(2, int(len(signal) * output_fs / input_fs))
    return resample(signal.astype(float), n_out)


def auto_polarity_check(signal: np.ndarray, fs: float = TARGET_FS) -> bool:
    """Return True iff the signal is likely upside down.

    Heuristic: in a clean ECG the QRS complex creates strong positive R-peaks
    that dominate the upper tail of the bandpassed amplitude distribution. If
    the negative tail is consistently larger than the positive tail (after
    centering and bandpassing), the signal is probably inverted. A 5-second
    window is enough for this judgement in practice.
    """
    if len(signal) < int(2 * fs):
        return False
    n = min(len(signal), int(5 * fs))
    s = signal[:n] - np.mean(signal[:n])
    nyq = 0.5 * fs
    b, a = butter(2, [5.0 / nyq, min(40.0 / nyq, 0.99)], btype="band")
    filt = filtfilt(b, a, s)
    pos = float(np.quantile(filt, 0.99))
    neg = float(-np.quantile(filt, 0.01))
    return neg > pos * 1.3


def detect_r_peaks(signal: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    centered = signal - np.mean(signal)
    nyq = fs * 0.5
    b, a = butter(2, [5.0 / nyq, min(40.0 / nyq, 0.99)], btype="band")
    filt = filtfilt(b, a, centered)
    env = filt**2
    peaks, _ = find_peaks(
        env,
        distance=max(int(0.25 * fs), 1),
        prominence=max(np.std(env) * 0.35, 1e-12),
    )
    return peaks


def windows_from_peaks(signal: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    valid_peaks = []
    for p in peaks:
        s = p - HALF_WINDOW
        e = p + HALF_WINDOW
        if s < 0 or e > len(signal):
            continue
        w = signal[s:e]
        if len(w) == WINDOW_SIZE:
            windows.append(w)
            valid_peaks.append(p)
    return np.array(windows), np.array(valid_peaks, dtype=int)


def _prepare_features(
    csv_path: Path,
    config: WearableInferenceConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shared front-end: CSV → resample → polarity → R-peaks → preprocessed features.

    Returns ``(X, peak_idx, info)`` where ``X`` has 182 columns (180 morphology +
    2 RR ratios) and ``info`` is a small diagnostic dict.
    """
    signal = load_signal_csv(csv_path)
    info: dict = {
        "n_samples": int(len(signal)),
        "input_fs": float(config.input_fs),
        "lead": config.lead_name,
        "polarity_flipped": False,
        "domain_shift_warning": False,
    }
    if len(signal) == 0:
        return np.empty((0, WINDOW_SIZE + 2)), np.empty(0, dtype=int), info

    if config.input_fs != TARGET_FS:
        signal = resample_signal(signal, config.input_fs, TARGET_FS)

    flip = config.invert_polarity
    if config.auto_polarity and not flip:
        flip = auto_polarity_check(signal, TARGET_FS)
    if flip:
        signal = -signal
        info["polarity_flipped"] = True

    if config.lead_name not in {"MLII", "II", "unknown"}:
        warnings.warn(
            f"Model was trained on MIT-BIH MLII; expect a noticeable accuracy drop on lead "
            f"{config.lead_name!r}. PVC/PAC recall in particular depends on QRS morphology, "
            "which is lead-specific.",
            stacklevel=2,
        )
        info["domain_shift_warning"] = True

    peaks = detect_r_peaks(signal, TARGET_FS)
    windows, peak_idx = windows_from_peaks(signal, peaks)
    if len(windows) == 0:
        return np.empty((0, WINDOW_SIZE + 2)), peak_idx, info

    X_morph = preprocess_windows(windows)
    rr = compute_rr_features(peak_idx)
    X = np.hstack([X_morph, rr])
    info["n_beats"] = int(len(X))
    return X, peak_idx, info


def run_baseline_inference(
    csv_path: Path,
    model_path: Path,
    scaler_path: Path,
    input_fs: float = TARGET_FS,
    *,
    lead_name: str = "unknown",
    auto_polarity: bool = True,
    invert_polarity: bool = False,
) -> list[dict]:
    config = WearableInferenceConfig(
        input_fs=input_fs,
        lead_name=lead_name,
        invert_polarity=invert_polarity,
        auto_polarity=auto_polarity,
    )
    X, peak_idx, _ = _prepare_features(csv_path, config)
    if len(X) == 0:
        return []

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)
    pred = np.argmax(proba, axis=1)
    rows = []
    for i, (p, yhat) in enumerate(zip(peak_idx, pred, strict=False), start=1):
        rows.append(
            {
                "beat_index": i,
                "peak_sample_360hz": int(p),
                "label": ID_TO_LABEL[int(yhat)],
                "prob_N": float(proba[i - 1, 0]),
                "prob_V": float(proba[i - 1, 1]),
                "prob_a": float(proba[i - 1, 2]),
            }
        )
    return rows


def run_resnet_inference(
    csv_path: Path,
    weights_path: Path,
    config_path: Path,
    input_fs: float = TARGET_FS,
    *,
    lead_name: str = "unknown",
    auto_polarity: bool = True,
    invert_polarity: bool = False,
) -> list[dict]:
    """ResNet-1D inference on a CSV ECG. Requires the ``[deep]`` extra (PyTorch)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ResNet inference requires PyTorch. Install with: pip install -e '.[deep]'"
        ) from exc
    from .models.resnet1d import ResNet1D

    config = WearableInferenceConfig(
        input_fs=input_fs,
        lead_name=lead_name,
        invert_polarity=invert_polarity,
        auto_polarity=auto_polarity,
    )
    X, peak_idx, _ = _prepare_features(csv_path, config)
    if len(X) == 0:
        return []

    model_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = ResNet1D(**model_config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_samples = model_config["samples_per_lead"]
    in_channels = model_config["in_channels"]
    morph = X[:, :-2]
    rr = X[:, -2:]
    morph_t = (
        torch.from_numpy(morph.astype(np.float32))
        .reshape(-1, in_channels, n_samples)
    )
    rr_t = torch.from_numpy(rr.astype(np.float32))

    with torch.no_grad():
        logits = model(morph_t, rr_t)
        proba = torch.softmax(logits, dim=1).numpy()
    pred = proba.argmax(axis=1)

    rows = []
    for i, (p, yhat) in enumerate(zip(peak_idx, pred, strict=False), start=1):
        rows.append(
            {
                "beat_index": i,
                "peak_sample_360hz": int(p),
                "label": ID_TO_LABEL[int(yhat)],
                "prob_N": float(proba[i - 1, 0]),
                "prob_V": float(proba[i - 1, 1]),
                "prob_a": float(proba[i - 1, 2]),
            }
        )
    return rows
