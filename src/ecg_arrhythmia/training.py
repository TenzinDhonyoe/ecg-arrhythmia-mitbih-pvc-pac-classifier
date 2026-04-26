"""Training entry points for ECG beat classification models.

Currently provides:
- ``train_baseline_lr``: real MIT-BIH logistic regression with morphology + RR
  features. Writes a full metrics report (per-class precision/recall/F1,
  confusion matrix, ROC-AUC, bootstrap 95% CIs) plus an inference latency
  benchmark.
- ``train_baseline_quick_smoke``: synthetic-data smoke training for fast CI
  validation; does not require MIT-BIH and writes to a separate ``smoke/``
  output directory so its outputs are never confused with real metrics.
- ``train_resnet_1d``: optional ResNet-1D + RR head. Requires the ``[deep]``
  extra (PyTorch). Imported lazily so the default install does not pull
  PyTorch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .labels import ID_TO_LABEL
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


def _collect_dataset(
    data_dir: Path,
    leads: Sequence[str | int] = (0,),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all MIT-BIH records and return (X, y, record_ids, peaks).

    For multi-lead requests, morphology windows from each lead are concatenated
    along the feature axis so a single sample carries beat morphology from all
    requested leads. RR-interval features are computed once per beat and
    appended later, in ``train_baseline_lr``/``train_resnet_1d``.
    """
    per_lead_x: list[list[np.ndarray]] = [[] for _ in leads]
    y_all: list[np.ndarray] = []
    rid_all: list[np.ndarray] = []
    peak_all: list[np.ndarray] = []
    for name in load_record_names(data_dir):
        first_lead_segments = None
        for li, lead in enumerate(leads):
            try:
                x, y, rid, peaks = segment_record(name, data_dir, lead=lead)
            except (ValueError, KeyError):
                # Lead missing on this record; pad with zeros to keep alignment
                if first_lead_segments is None:
                    # No anchor lead yet: skip the whole record
                    break
                x = np.zeros_like(first_lead_segments[0])
                y = first_lead_segments[1]
                rid = first_lead_segments[2]
                peaks = first_lead_segments[3]
            if len(x) == 0:
                if first_lead_segments is None:
                    break
                x = np.zeros_like(first_lead_segments[0])
                y = first_lead_segments[1]
                rid = first_lead_segments[2]
                peaks = first_lead_segments[3]
            if first_lead_segments is None:
                first_lead_segments = (x, y, rid, peaks)
            per_lead_x[li].append(x)
        if first_lead_segments is None:
            continue
        _, y, rid, peaks = first_lead_segments
        y_all.append(y)
        rid_all.append(rid)
        peak_all.append(peaks)

    if not y_all:
        empty = np.array([])
        return empty, empty, empty, empty

    morph_per_lead = [np.concatenate(buf, axis=0) for buf in per_lead_x]
    morph = np.concatenate(morph_per_lead, axis=1)  # concat windows across leads
    return (
        morph,
        np.concatenate(y_all, axis=0),
        np.concatenate(rid_all, axis=0),
        np.concatenate(peak_all, axis=0),
    )


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (low, high) 95% bootstrap CI for ``metric`` in {balanced_accuracy, macro_f1}."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    vals = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if metric == "balanced_accuracy":
            vals[i] = balanced_accuracy_score(yt, yp)
        else:
            vals[i] = f1_score(yt, yp, average="macro", zero_division=0)
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return (lo, hi)


def _full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    *,
    label_ids: Sequence[int] = (0, 1, 2),
    bootstrap_seed: int = 42,
) -> dict:
    """Per-class precision/recall/F1, confusion matrix, ROC-AUC, bootstrap CIs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    target_names = [ID_TO_LABEL[i] for i in label_ids]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(label_ids),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(label_ids)).tolist()

    auc_macro: float | None = None
    if y_proba is not None and len(np.unique(y_true)) == len(label_ids):
        try:
            auc_macro = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=list(label_ids),
                )
            )
        except ValueError:
            auc_macro = None

    bal_lo, bal_hi = _bootstrap_ci(y_true, y_pred, "balanced_accuracy", seed=bootstrap_seed)
    f1_lo, f1_hi = _bootstrap_ci(y_true, y_pred, "macro_f1", seed=bootstrap_seed + 1)

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "balanced_accuracy_ci95": [bal_lo, bal_hi],
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1_ci95": [f1_lo, f1_hi],
        "roc_auc_ovr_macro": auc_macro,
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
        "confusion_matrix": cm,
        "labels": target_names,
        "n_samples": int(len(y_true)),
    }


def _benchmark_inference(
    model,
    scaler,
    X_sample: np.ndarray,
    *,
    n_iter: int = 1000,
) -> dict:
    """Median + p95 single-beat inference latency in milliseconds."""
    if len(X_sample) == 0:
        return {"median_ms": 0.0, "p95_ms": 0.0, "n_iter": 0}
    rng = np.random.default_rng(0)
    times = np.empty(n_iter, dtype=float)
    n = len(X_sample)
    for i in range(n_iter):
        x = X_sample[rng.integers(0, n)].reshape(1, -1)
        t0 = time.perf_counter()
        xs = scaler.transform(x) if scaler is not None else x
        model.predict_proba(xs)
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "n_iter": int(n_iter),
    }


def _model_size_kb(path: Path) -> float:
    return float(path.stat().st_size) / 1024.0


def train_baseline_lr(
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
    leads: Sequence[str | int] = ("MLII",),
) -> TrainArtifacts:
    """Train the LR baseline on real MIT-BIH and write full metrics + benchmarks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_morph, y, record_ids, peaks = _collect_dataset(data_dir, leads=leads)
    X_morph = preprocess_windows(X_morph)

    # RR features per record to avoid cross-record timing leakage.
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

    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    metrics = {
        "model": "baseline_lr",
        "leads": list(leads),
        "n_features": int(X.shape[1]),
        "class_distribution_train": {ID_TO_LABEL[k]: int((y_train == k).sum()) for k in (0, 1, 2)},
        "class_distribution_val": {ID_TO_LABEL[k]: int((y_val == k).sum()) for k in (0, 1, 2)},
        "class_distribution_test": {ID_TO_LABEL[k]: int((y_test == k).sum()) for k in (0, 1, 2)},
        "val": _full_report(y_val, val_pred, val_proba, bootstrap_seed=seed),
        "test": _full_report(y_test, test_pred, test_proba, bootstrap_seed=seed),
    }

    model_path = out_dir / "baseline_lr_mitdb.joblib"
    scaler_path = out_dir / "baseline_lr_scaler.joblib"
    split_path = out_dir / "record_split.json"
    metrics_path = out_dir / "metrics_baseline.json"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    split_path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bench = {
        "model": "baseline_lr",
        "model_size_kb": _model_size_kb(model_path),
        "scaler_size_kb": _model_size_kb(scaler_path),
        **_benchmark_inference(model, scaler, X_test if len(X_test) else X_train),
    }
    (out_dir / "benchmarks.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")

    return TrainArtifacts(model_path, scaler_path, metrics_path, split_path)


def train_baseline_quick_smoke(out_dir: Path, seed: int = 42) -> TrainArtifacts:
    """Synthetic smoke training (no MIT-BIH required). Outputs go under ``out_dir``.

    Callers should pass a dedicated smoke directory (e.g. ``artifacts/smoke``) so
    smoke metrics can never be confused with real-data metrics.
    """
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
    metrics = {
        "model": "baseline_lr_smoke",
        "warning": "These metrics come from synthetic Gaussian classes, NOT real MIT-BIH ECG data. "
        "Do not quote them as model performance. See docs/MODEL_CARD.md for real numbers.",
        "val": _scores(y_val, model.predict(X_val)),
        "test": _scores(y_test, model.predict(X_test)),
    }
    split = {
        "mode": "quick_smoke_synthetic",
        "n_train_samples": int(len(y_train)),
        "n_val_samples": int(len(y_val)),
        "n_test_samples": int(len(y_test)),
    }

    model_path = out_dir / "baseline_lr_mitdb.joblib"
    scaler_path = out_dir / "baseline_lr_scaler.joblib"
    split_path = out_dir / "record_split.json"
    metrics_path = out_dir / "metrics_smoke.json"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainArtifacts(model_path, scaler_path, metrics_path, split_path)


def train_resnet_1d(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    leads: Sequence[str | int] = ("MLII",),
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "auto",
    quick_smoke: bool = False,
) -> TrainArtifacts:
    """Train a ResNet-1D + RR-feature head. Requires the ``[deep]`` extra (PyTorch)."""
    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "ResNet training requires PyTorch. Install with: pip install -e '.[deep]'"
        ) from exc

    from .models.resnet1d import ResNet1D

    out_dir.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    torch_device = torch.device(device)

    if quick_smoke:
        # Synthetic data path identical in shape to the real pipeline.
        rng = np.random.default_rng(seed)
        n_per_class = 64
        n_morph = 180
        morph = np.vstack(
            [
                rng.normal(0.0, 1.0, size=(n_per_class, n_morph)),
                rng.normal(0.8, 1.0, size=(n_per_class, n_morph)),
                rng.normal(-0.8, 1.0, size=(n_per_class, n_morph)),
            ]
        )
        rr = rng.normal(1.0, 0.1, size=(len(morph), 2))
        y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
        n_train = int(0.7 * len(y))
        n_val = int(0.15 * len(y))
        record_ids = np.array([f"rec_{i // 8}" for i in range(len(y))])
        perm = rng.permutation(len(y))
        morph, rr, y, record_ids = morph[perm], rr[perm], y[perm], record_ids[perm]
        m_train, m_val, m_test = (
            slice(0, n_train),
            slice(n_train, n_train + n_val),
            slice(n_train + n_val, None),
        )
        morph_train, morph_val, morph_test = morph[m_train], morph[m_val], morph[m_test]
        rr_train, rr_val, rr_test = rr[m_train], rr[m_val], rr[m_test]
        y_train, y_val, y_test = y[m_train], y[m_val], y[m_test]
        split_meta = {"mode": "resnet_quick_smoke_synthetic", "n_total": int(len(y))}
        epochs = min(epochs, 3)
    else:
        X_morph, y, record_ids, peaks = _collect_dataset(data_dir, leads=leads)
        X_morph = preprocess_windows(X_morph)
        rr_all = np.zeros((len(peaks), 2), dtype=float)
        for rid in np.unique(record_ids):
            idx = np.flatnonzero(record_ids == rid)
            rr_all[idx] = compute_rr_features(peaks[idx])
        split = split_by_record(record_ids, seed=seed)
        train_mask = mask_for_records(record_ids, split.train_records)
        val_mask = mask_for_records(record_ids, split.val_records)
        test_mask = mask_for_records(record_ids, split.test_records)
        morph_train, morph_val, morph_test = X_morph[train_mask], X_morph[val_mask], X_morph[test_mask]
        rr_train, rr_val, rr_test = rr_all[train_mask], rr_all[val_mask], rr_all[test_mask]
        y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
        split_meta = asdict(split)

    n_per_lead_samples = morph_train.shape[1] // max(len(leads), 1)
    in_channels = max(len(leads), 1)
    model_config = {
        "in_channels": in_channels,
        "samples_per_lead": int(n_per_lead_samples),
        "rr_features": 2,
        "n_classes": 3,
        "channels": [32, 64, 128],
        "blocks_per_stage": 2,
    }
    model = ResNet1D(**model_config).to(torch_device)

    def to_tensors(morph_arr, rr_arr, y_arr):
        morph_t = (
            torch.from_numpy(morph_arr.astype(np.float32))
            .reshape(-1, in_channels, n_per_lead_samples)
        )
        return (
            morph_t,
            torch.from_numpy(rr_arr.astype(np.float32)),
            torch.from_numpy(y_arr.astype(np.int64)),
        )

    morph_train_t, rr_train_t, y_train_t = to_tensors(morph_train, rr_train, y_train)
    morph_val_t, rr_val_t, y_val_t = to_tensors(morph_val, rr_val, y_val)
    morph_test_t, rr_test_t, y_test_t = to_tensors(morph_test, rr_test, y_test)

    train_loader = DataLoader(
        TensorDataset(morph_train_t, rr_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    cw = class_weights(y_train)
    weights = torch.tensor([cw[0], cw[1], cw[2]], dtype=torch.float32, device=torch_device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_f1 = -1.0
    patience = 5
    bad_epochs = 0
    best_state: dict | None = None
    last_epoch = 0
    for epoch in range(epochs):
        last_epoch = epoch
        model.train()
        for morph_b, rr_b, y_b in train_loader:
            morph_b = morph_b.to(torch_device)
            rr_b = rr_b.to(torch_device)
            y_b = y_b.to(torch_device)
            optimizer.zero_grad()
            logits = model(morph_b, rr_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(morph_val_t.to(torch_device), rr_val_t.to(torch_device))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        cur_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if cur_f1 > best_val_f1 + 1e-4:
            best_val_f1 = cur_f1
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(morph_val_t.to(torch_device), rr_val_t.to(torch_device))
        val_proba = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_pred = val_proba.argmax(axis=1)
        test_logits = model(morph_test_t.to(torch_device), rr_test_t.to(torch_device))
        test_proba = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_pred = test_proba.argmax(axis=1)

    metrics = {
        "model": "resnet1d",
        "leads": list(leads),
        "config": model_config,
        "epochs_run": last_epoch + 1,
        "best_val_macro_f1": float(best_val_f1),
        "device": device,
        "class_distribution_train": {ID_TO_LABEL[k]: int((y_train == k).sum()) for k in (0, 1, 2)},
        "class_distribution_val": {ID_TO_LABEL[k]: int((y_val == k).sum()) for k in (0, 1, 2)},
        "class_distribution_test": {ID_TO_LABEL[k]: int((y_test == k).sum()) for k in (0, 1, 2)},
        "val": _full_report(y_val, val_pred, val_proba, bootstrap_seed=seed),
        "test": _full_report(y_test, test_pred, test_proba, bootstrap_seed=seed),
    }

    weights_path = out_dir / "resnet1d.pt"
    config_path = out_dir / "model_config.json"
    metrics_path = out_dir / "metrics_resnet.json"
    split_path = out_dir / "record_split.json"
    torch.save(model.state_dict(), weights_path)
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    split_path.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Latency benchmark — single-beat CPU forward pass.
    cpu_model = model.to("cpu").eval()
    times = []
    morph_cpu = morph_test_t.to("cpu") if len(morph_test_t) else morph_train_t.to("cpu")
    rr_cpu = rr_test_t.to("cpu") if len(rr_test_t) else rr_train_t.to("cpu")
    n_bench = min(1000, max(len(morph_cpu), 1))
    with torch.no_grad():
        for i in range(n_bench):
            mi = morph_cpu[i % len(morph_cpu)].unsqueeze(0)
            ri = rr_cpu[i % len(rr_cpu)].unsqueeze(0)
            t0 = time.perf_counter()
            cpu_model(mi, ri)
            times.append((time.perf_counter() - t0) * 1000.0)
    n_params = int(sum(p.numel() for p in cpu_model.parameters()))
    bench = {
        "model": "resnet1d",
        "n_params": n_params,
        "model_size_kb": _model_size_kb(weights_path),
        "median_ms": float(np.median(times)) if times else 0.0,
        "p95_ms": float(np.quantile(times, 0.95)) if times else 0.0,
        "n_iter": len(times),
        "device": "cpu",
    }
    (out_dir / "benchmarks.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")

    return TrainArtifacts(weights_path, config_path, metrics_path, split_path)
