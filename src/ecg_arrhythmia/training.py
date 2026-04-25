"""Training entry points for baseline LR and ResNet-1D+RR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .preprocessing import (
    class_weights,
    compute_rr_features,
    load_record_names,
    mask_for_records,
    preprocess_windows,
    segment_record,
    split_by_record,
)


@dataclass
class TrainArtifacts:
    model_path: Path
    scaler_path: Path
    metrics_path: Path
    split_path: Path


def _collect_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    rid_all: list[np.ndarray] = []
    peak_all: list[np.ndarray] = []
    for name in load_record_names(data_dir):
        x, y, rid, peaks = segment_record(name, data_dir)
        if len(x) == 0:
            continue
        x_all.append(x)
        y_all.append(y)
        rid_all.append(rid)
        peak_all.append(peaks)
    return (
        np.concatenate(x_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.concatenate(rid_all, axis=0),
        np.concatenate(peak_all, axis=0),
    )


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_baseline_lr(data_dir: Path, out_dir: Path, seed: int = 42) -> TrainArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_raw, y, record_ids, peaks = _collect_dataset(data_dir)
    X_morph = preprocess_windows(X_raw)

    # RR features are computed per record to avoid cross-record timing leakage.
    rr = np.zeros((len(peaks), 2), dtype=float)
    for rid in np.unique(record_ids):
        idx = np.flatnonzero(record_ids == rid)
        rr[idx] = compute_rr_features(peaks[idx])
    X = np.hstack([X_morph, rr])

    split = split_by_record(record_ids, seed=seed)
    train_mask = mask_for_records(record_ids, split.train_records)
    val_mask = mask_for_records(record_ids, split.val_records)
    test_mask = mask_for_records(record_ids, split.test_records)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_val = scaler.transform(X[val_mask])
    X_test = scaler.transform(X[test_mask])
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    model = LogisticRegression(
        class_weight=class_weights(y_train),
        max_iter=8000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    metrics = {
        "val": _scores(y_val, model.predict(X_val)),
        "test": _scores(y_test, model.predict(X_test)),
    }

    model_path = out_dir / "baseline_lr_mitdb.joblib"
    scaler_path = out_dir / "baseline_lr_scaler.joblib"
    split_path = out_dir / "record_split.json"
    metrics_path = out_dir / "metrics_baseline.json"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    split_path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainArtifacts(model_path, scaler_path, metrics_path, split_path)


def train_baseline_quick_smoke(out_dir: Path, seed: int = 42) -> TrainArtifacts:
    """Fast synthetic training path for CI/smoke verification."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_per_class = 120
    n_features = 182
    x0 = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, n_features))
    x1 = rng.normal(loc=0.8, scale=1.0, size=(n_per_class, n_features))
    x2 = rng.normal(loc=-0.8, scale=1.0, size=(n_per_class, n_features))
    X = np.vstack([x0, x1, x2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    n_train = int(0.7 * len(y))
    n_val = int(0.15 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    model.fit(X_train, y_train)
    metrics = {"val": _scores(y_val, model.predict(X_val)), "test": _scores(y_test, model.predict(X_test))}
    split = {
        "mode": "quick_smoke_synthetic",
        "n_train_samples": int(len(y_train)),
        "n_val_samples": int(len(y_val)),
        "n_test_samples": int(len(y_test)),
    }

    model_path = out_dir / "baseline_lr_mitdb.joblib"
    scaler_path = out_dir / "baseline_lr_scaler.joblib"
    split_path = out_dir / "record_split.json"
    metrics_path = out_dir / "metrics_baseline.json"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainArtifacts(model_path, scaler_path, metrics_path, split_path)
