"""Preprocessing and dataset utilities for MIT-BIH ECG windows."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from .labels import MITBIH3, LabelScheme

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


def _read_record(record_name: str, data_dir: Path):
    """Read a WFDB record + annotations relative to ``data_dir``.

    ``wfdb.rdrecord`` resolves paths relative to the current working directory,
    so we cd in and out for the duration of the read.
    """
    original_cwd = Path.cwd()
    try:
        os.chdir(data_dir)
        record = wfdb.rdrecord(record_name)
        ann = wfdb.rdann(record_name, "atr")
    finally:
        os.chdir(original_cwd)
    return record, ann


def available_leads(record_name: str, data_dir: Path) -> list[str]:
    """Return signal-channel names for an MIT-BIH record (e.g. ``["MLII", "V5"]``)."""
    record, _ = _read_record(record_name, data_dir)
    return list(record.sig_name)


def _resolve_lead(record, lead: str | int) -> int:
    if isinstance(lead, int):
        if not 0 <= lead < record.p_signal.shape[1]:
            raise ValueError(f"Lead index {lead} out of range for record with {record.p_signal.shape[1]} channels")
        return lead
    sig_names = list(record.sig_name)
    if lead in sig_names:
        return sig_names.index(lead)
    raise KeyError(f"Lead {lead!r} not in record. Available: {sig_names}")


def segment_record(
    record_name: str,
    data_dir: Path,
    lead: str | int = 0,
    scheme: LabelScheme = MITBIH3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract beat windows + labels + record IDs + peak indices from a WFDB record.

    Parameters
    ----------
    record_name : str
        WFDB record stem (e.g. ``"100"``).
    data_dir : Path
        Directory containing the WFDB files plus ``RECORDS``.
    lead : str | int, default 0
        Either a signal-name string from ``record.sig_name`` (e.g. ``"MLII"``,
        ``"V1"``, ``"V5"``) or a 0-based channel index. MIT-BIH records mostly
        contain MLII as channel 0; some records use channels other than V1 for
        their second channel, so name-based lookup is preferred for portability.
    scheme : LabelScheme, default MITBIH3
        Mapping from WFDB beat symbols to class ids. Pass :data:`AAMI5` to use
        the AAMI EC57 5-class scheme.
    """
    record, ann = _read_record(record_name, data_dir)
    channel = _resolve_lead(record, lead)
    ecg = record.p_signal[:, channel]

    windows: list[np.ndarray] = []
    labels: list[int] = []
    peaks: list[int] = []

    label_to_id = scheme.label_to_id
    target_symbols = scheme.target_symbols
    for pos, symbol in zip(ann.sample, ann.symbol, strict=False):
        if symbol not in target_symbols:
            continue
        start = pos - HALF_WINDOW
        end = pos + HALF_WINDOW
        if start < 0 or end > len(ecg):
            continue
        w = ecg[start:end]
        if len(w) != WINDOW_SIZE:
            continue
        windows.append(w)
        labels.append(label_to_id[symbol])
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


# ---------------------------------------------------------------------------
# AAMI EC57 / de Chazal 2004 inter-patient split
# ---------------------------------------------------------------------------
# These two record sets are the canonical inter-patient split for MIT-BIH used
# by virtually every published AAMI EC57 benchmark. DS1 trains, DS2 tests; we
# carve a small validation slice off DS1 so the user still gets early-stopping.
#
# Reference: de Chazal, O'Dwyer & Reilly, "Automatic classification of
# heartbeats using ECG morphology and heartbeat interval features," IEEE
# Transactions on Biomedical Engineering, 2004.
DE_CHAZAL_DS1: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)
DE_CHAZAL_DS2: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)
# Paced records contain virtually all MIT-BIH Q-class beats. Reporting "without
# paced" subset metrics is standard for AAMI evaluation; users can opt in via
# ``exclude_paced=True`` on the split helper or via ``--exclude-paced-records``
# at the CLI.
PACED_RECORDS: tuple[str, ...] = ("102", "104", "107", "217")


def split_de_chazal(
    record_ids: np.ndarray,
    *,
    val_records: int = 4,
    seed: int = 42,
    exclude_paced: bool = False,
) -> DatasetSplit:
    """Return the canonical de Chazal DS1/DS2 inter-patient split.

    A small validation slice is carved off DS1 (deterministic, seeded) so that
    early-stopping has somewhere to look. DS2 stays intact as the held-out
    test set so reported numbers are directly comparable to published AAMI
    benchmarks.
    """
    present = set(np.unique(record_ids).tolist())
    ds1 = [r for r in DE_CHAZAL_DS1 if r in present]
    ds2 = [r for r in DE_CHAZAL_DS2 if r in present]

    if exclude_paced:
        paced = set(PACED_RECORDS)
        ds1 = [r for r in ds1 if r not in paced]
        ds2 = [r for r in ds2 if r not in paced]

    if not ds1 or not ds2:
        raise ValueError(
            "de Chazal split requires records from both DS1 and DS2 to be "
            f"present in the dataset. Found {len(ds1)} DS1 / {len(ds2)} DS2."
        )

    rng = np.random.default_rng(seed)
    ds1_arr = np.array(ds1)
    rng.shuffle(ds1_arr)
    val_records = max(1, min(val_records, len(ds1_arr) - 1))
    val = ds1_arr[:val_records].tolist()
    train = ds1_arr[val_records:].tolist()
    return DatasetSplit(train_records=train, val_records=val, test_records=ds2)


def mask_for_records(record_ids: np.ndarray, records: Iterable[str]) -> np.ndarray:
    allowed = set(records)
    return np.array([rid in allowed for rid in record_ids], dtype=bool)


def class_weights(
    y_train: np.ndarray,
    n_classes: int = 3,
    max_weight: float | None = 50.0,
) -> dict[int, float]:
    """Inverse-frequency class weights, ``n_samples / (n_classes * count_k)``.

    Returns 0.0 for any class with zero samples in ``y_train`` so the caller
    can decide whether to mask, error, or carry on. Matches sklearn's
    ``class_weight="balanced"`` formula, with one safety addition: when
    ``max_weight`` is set, weights are clipped above that cap. Without the cap,
    AAMI 5-class on MIT-BIH produces weights >1000 for the Q class (which has
    ~6 train beats), and the model collapses to predicting that class always.
    A cap of 50 keeps minority classes meaningfully upweighted without making
    a few examples dominate the loss.
    """
    n_samples = len(y_train)
    out: dict[int, float] = {}
    for k in range(n_classes):
        count = int(np.sum(y_train == k))
        if count == 0:
            out[k] = 0.0
            continue
        w = n_samples / (n_classes * count)
        if max_weight is not None and w > max_weight:
            w = float(max_weight)
        out[k] = w
    return out
