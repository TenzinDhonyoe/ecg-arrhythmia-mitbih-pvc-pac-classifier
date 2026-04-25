"""Preprocessing and dataset utilities for MIT-BIH ECG windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import wfdb

from .labels import LABEL_TO_ID, TARGET_SYMBOLS

WINDOW_SIZE = 180
HALF_WINDOW = 90


@dataclass(frozen=True)
class DatasetSplit:
    train_records: list[str]
    val_records: list[str]
    test_records: list[str]


def load_record_names(data_dir: Path) -> list[str]:
    records = (data_dir / "RECORDS").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in records if line.strip()]


def correct_baseline_wander(window: np.ndarray) -> np.ndarray:
    n = len(window)
    trend = np.linspace(window[0], window[-1], n)
    return window - trend


def normalize_beat(window: np.ndarray) -> np.ndarray:
    mean = float(np.mean(window))
    std = float(np.std(window))
    if std == 0:
        return window - mean
    return (window - mean) / std


def compute_rr_features(peak_indices: np.ndarray) -> np.ndarray:
    peak_indices = np.asarray(peak_indices, dtype=int)
    n = len(peak_indices)
    if n < 2:
        return np.ones((n, 2), dtype=float)
    rr = np.diff(peak_indices).astype(float)
    med = np.median(rr)
    med = med if med > 0 else 1.0
    out = np.ones((n, 2), dtype=float)
    for i in range(n):
        out[i, 0] = rr[i - 1] / med if i > 0 else rr[0] / med
        out[i, 1] = rr[i] / med if i < n - 1 else rr[-1] / med
    return out


def segment_record(record_name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original_cwd = Path.cwd()
    try:
        # wfdb compatibility for local relative reads
        import os

        os.chdir(data_dir)
        record = wfdb.rdrecord(record_name)
        ann = wfdb.rdann(record_name, "atr")
    finally:
        import os

        os.chdir(original_cwd)

    ecg = record.p_signal[:, 0]
    windows: list[np.ndarray] = []
    labels: list[int] = []
    peaks: list[int] = []

    for pos, symbol in zip(ann.sample, ann.symbol):
        if symbol not in TARGET_SYMBOLS:
            continue
        start = pos - HALF_WINDOW
        end = pos + HALF_WINDOW
        if start < 0 or end > len(ecg):
            continue
        w = ecg[start:end]
        if len(w) != WINDOW_SIZE:
            continue
        windows.append(w)
        labels.append(LABEL_TO_ID[symbol])
        peaks.append(int(pos))

    n = len(labels)
    record_ids = np.array([record_name] * n, dtype="<U8")
    return np.array(windows), np.array(labels), record_ids, np.array(peaks, dtype=int)


def preprocess_windows(windows: np.ndarray) -> np.ndarray:
    out = np.zeros_like(windows, dtype=float)
    for i, w in enumerate(windows):
        out[i] = normalize_beat(correct_baseline_wander(w))
    return out


def split_by_record(record_ids: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> DatasetSplit:
    unique = np.array(sorted(np.unique(record_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_total = len(unique)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train = unique[:n_train].tolist()
    val = unique[n_train : n_train + n_val].tolist()
    test = unique[n_train + n_val :].tolist()
    return DatasetSplit(train_records=train, val_records=val, test_records=test)


def mask_for_records(record_ids: np.ndarray, records: Iterable[str]) -> np.ndarray:
    allowed = set(records)
    return np.array([rid in allowed for rid in record_ids], dtype=bool)


def class_weights(y_train: np.ndarray) -> dict[int, float]:
    n_samples = len(y_train)
    n_classes = 3
    out: dict[int, float] = {}
    for k in [0, 1, 2]:
        count = int(np.sum(y_train == k))
        out[k] = (n_samples / (n_classes * count)) if count else 0.0
    return out
