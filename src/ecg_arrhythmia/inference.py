"""Inference helpers for CSV ECG signals using baseline LR artifacts."""

from __future__ import annotations

from pathlib import Path
import csv

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample

from .labels import ID_TO_LABEL
from .preprocessing import HALF_WINDOW, WINDOW_SIZE, compute_rr_features, preprocess_windows

TARGET_FS = 360.0


def load_signal_csv(path: Path) -> np.ndarray:
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


def detect_r_peaks(signal: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    centered = signal - np.mean(signal)
    nyq = fs * 0.5
    b, a = butter(2, [5.0 / nyq, min(40.0 / nyq, 0.99)], btype="band")
    filt = filtfilt(b, a, centered)
    env = filt**2
    peaks, _ = find_peaks(env, distance=max(int(0.25 * fs), 1), prominence=max(np.std(env) * 0.35, 1e-12))
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


def run_baseline_inference(csv_path: Path, model_path: Path, scaler_path: Path, input_fs: float = TARGET_FS) -> list[dict]:
    signal = load_signal_csv(csv_path)
    if input_fs != TARGET_FS:
        signal = resample_signal(signal, input_fs, TARGET_FS)
    peaks = detect_r_peaks(signal, TARGET_FS)
    windows, peak_idx = windows_from_peaks(signal, peaks)
    if len(windows) == 0:
        return []
    X_morph = preprocess_windows(windows)
    rr = compute_rr_features(peak_idx)
    X = np.hstack([X_morph, rr])
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)
    pred = np.argmax(proba, axis=1)
    rows = []
    for i, (p, yhat) in enumerate(zip(peak_idx, pred), start=1):
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
