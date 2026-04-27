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

from .labels import MITBIH3, LabelScheme
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
    scheme: LabelScheme = MITBIH3,
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
                x, y, rid, peaks = segment_record(name, data_dir, lead=lead, scheme=scheme)
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
    label_ids: Sequence[int] | None = None,
    id_to_label: dict[int, str] | None = None,
    bootstrap_seed: int = 42,
) -> dict:
    """Per-class precision/recall/F1, confusion matrix, ROC-AUC, bootstrap CIs.

    ``label_ids`` and ``id_to_label`` default to :data:`MITBIH3` for back-compat
    so older callers with no label-scheme awareness still get the v0.3 behaviour.
    """
    if id_to_label is None:
        id_to_label = MITBIH3.id_to_label
    if label_ids is None:
        label_ids = list(range(len(id_to_label)))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    target_names = [id_to_label[i] for i in label_ids]
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

    # ECE + Brier reporting if probabilities are available (ResNet path always
    # supplies them; LR baseline does too via predict_proba).
    ece_block: dict | None = None
    brier: float | None = None
    if y_proba is not None:
        try:
            from .calibration import expected_calibration_error

            ece_block = expected_calibration_error(np.asarray(y_proba), y_true, n_bins=15)
            n_classes = y_proba.shape[1]
            one_hot = np.zeros_like(y_proba, dtype=float)
            one_hot[np.arange(len(y_true)), y_true] = 1.0
            brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1) / n_classes))
        except (ImportError, ValueError):
            ece_block = None
            brier = None

    out = {
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
    if ece_block is not None:
        out["ece"] = float(ece_block["ece"])
        out["calibration_bins"] = {
            "centers": ece_block["bin_centers"],
            "confidences": ece_block["bin_confidences"],
            "accuracies": ece_block["bin_accuracies"],
            "counts": ece_block["bin_counts"],
        }
    if brier is not None:
        out["brier_score"] = brier
    return out


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
    scheme: LabelScheme = MITBIH3,
    split_strategy: str = "random",
    exclude_paced: bool = False,
) -> TrainArtifacts:
    """Train the LR baseline on real MIT-BIH and write full metrics + benchmarks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_morph, y, record_ids, peaks = _collect_dataset(data_dir, leads=leads, scheme=scheme)
    X_morph = preprocess_windows(X_morph)

    # RR features per record to avoid cross-record timing leakage.
    rr = np.zeros((len(peaks), 2), dtype=float)
    for rid in np.unique(record_ids):
        idx = np.flatnonzero(record_ids == rid)
        rr[idx] = compute_rr_features(peaks[idx])
    X = np.hstack([X_morph, rr])

    if split_strategy == "ds1ds2":
        from .preprocessing import split_de_chazal

        split = split_de_chazal(record_ids, seed=seed, exclude_paced=exclude_paced)
    elif split_strategy == "random":
        split = split_by_record(record_ids, seed=seed)
    else:
        raise ValueError(
            f"Unknown split_strategy {split_strategy!r}; expected 'random' or 'ds1ds2'."
        )
    train_mask = mask_for_records(record_ids, split.train_records)
    val_mask = mask_for_records(record_ids, split.val_records)
    test_mask = mask_for_records(record_ids, split.test_records)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_val = scaler.transform(X[val_mask])
    X_test = scaler.transform(X[test_mask])
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    model = LogisticRegression(
        class_weight=class_weights(y_train, n_classes=scheme.n_classes),
        max_iter=8000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    cls_range = range(scheme.n_classes)
    id_to_label = scheme.id_to_label
    metrics = {
        "model": "baseline_lr",
        "scheme": scheme.name,
        "leads": list(leads),
        "n_features": int(X.shape[1]),
        "class_distribution_train": {id_to_label[k]: int((y_train == k).sum()) for k in cls_range},
        "class_distribution_val": {id_to_label[k]: int((y_val == k).sum()) for k in cls_range},
        "class_distribution_test": {id_to_label[k]: int((y_test == k).sum()) for k in cls_range},
        "val": _full_report(y_val, val_pred, val_proba, id_to_label=id_to_label, bootstrap_seed=seed),
        "test": _full_report(y_test, test_pred, test_proba, id_to_label=id_to_label, bootstrap_seed=seed),
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

    try:
        from .plots import confusion_matrix_plot

        confusion_matrix_plot(
            metrics["test"]["confusion_matrix"],
            scheme.labels,
            out_dir / "confusion_matrix_test.png",
            title=f"Confusion matrix (test, {scheme.name})",
        )
        confusion_matrix_plot(
            metrics["val"]["confusion_matrix"],
            scheme.labels,
            out_dir / "confusion_matrix_val.png",
            title=f"Confusion matrix (val, {scheme.name})",
        )
    except ImportError:
        pass

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
    scheme: LabelScheme = MITBIH3,
    # ----- v0.4 training upgrades (all opt-in; defaults preserve v0.3 behaviour) -----
    architecture: str = "resnet",
    use_se: bool = False,
    stem_kind: str = "conv",
    augment: bool = False,
    focal_gamma: float = 0.0,
    patience: int = 5,
    label_smoothing: float = 0.0,
    balanced_sampler: bool = False,
    mixup_alpha: float = 0.0,
    ema_decay: float = 0.0,
    grad_clip: float = 0.0,
    warmup_epochs: int = 0,
    use_amp: bool | None = None,
    num_workers: int = 0,
    split_strategy: str = "random",
    exclude_paced: bool = False,
    two_stage: bool = False,
) -> TrainArtifacts:
    """Train a ResNet-1D + RR-feature head. Requires the ``[deep]`` extra (PyTorch).

    The default arguments reproduce v0.3 behaviour exactly (vanilla weighted
    CE, no augmentation, no warmup, no AMP). Set the v0.4 flags to opt in.
    """
    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "ResNet training requires PyTorch. Install with: pip install -e '.[deep]'"
        ) from exc

    from .augment import AugmentedECGDataset, BeatAugmenter
    from .losses import FocalLoss, make_balanced_sampler, mixup_batch
    from .models.cnn_transformer import CNNTransformer1D
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
    # AMP is only safe on CUDA — MPS support is partial / silently incorrect.
    if use_amp is None:
        use_amp = device == "cuda"
    elif use_amp and device != "cuda":
        use_amp = False  # ignore the request rather than crash

    if quick_smoke:
        # Synthetic data path identical in shape to the real pipeline. We
        # synthesize one Gaussian cluster per class in ``scheme``, so smoke
        # works for any label cardinality (not only 3-class).
        rng = np.random.default_rng(seed)
        n_per_class = 64
        n_morph = 180
        n_classes = scheme.n_classes
        cluster_means = np.linspace(-0.8, 0.8, n_classes)
        morph = np.vstack(
            [rng.normal(mu, 1.0, size=(n_per_class, n_morph)) for mu in cluster_means]
        )
        rr = rng.normal(1.0, 0.1, size=(len(morph), 2))
        y = np.concatenate([np.full(n_per_class, k, dtype=int) for k in range(n_classes)])
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
        X_morph, y, record_ids, peaks = _collect_dataset(data_dir, leads=leads, scheme=scheme)
        X_morph = preprocess_windows(X_morph)
        rr_all = np.zeros((len(peaks), 2), dtype=float)
        for rid in np.unique(record_ids):
            idx = np.flatnonzero(record_ids == rid)
            rr_all[idx] = compute_rr_features(peaks[idx])
        if split_strategy == "ds1ds2":
            from .preprocessing import split_de_chazal

            split = split_de_chazal(record_ids, seed=seed, exclude_paced=exclude_paced)
        elif split_strategy == "random":
            split = split_by_record(record_ids, seed=seed)
        else:
            raise ValueError(
                f"Unknown split_strategy {split_strategy!r}; expected 'random' or 'ds1ds2'."
            )
        train_mask = mask_for_records(record_ids, split.train_records)
        val_mask = mask_for_records(record_ids, split.val_records)
        test_mask = mask_for_records(record_ids, split.test_records)
        morph_train, morph_val, morph_test = X_morph[train_mask], X_morph[val_mask], X_morph[test_mask]
        rr_train, rr_val, rr_test = rr_all[train_mask], rr_all[val_mask], rr_all[test_mask]
        y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
        split_meta = asdict(split)
        split_meta["strategy"] = split_strategy
        split_meta["exclude_paced"] = bool(exclude_paced)

    n_per_lead_samples = morph_train.shape[1] // max(len(leads), 1)
    in_channels = max(len(leads), 1)
    if architecture == "resnet":
        model_config = {
            "in_channels": in_channels,
            "samples_per_lead": int(n_per_lead_samples),
            "rr_features": 2,
            "n_classes": scheme.n_classes,
            "channels": [32, 64, 128],
            "blocks_per_stage": 2,
        }
        model = ResNet1D(**model_config, use_se=use_se, stem=stem_kind).to(torch_device)
    elif architecture == "cnn_transformer":
        model_config = {
            "in_channels": in_channels,
            "samples_per_lead": int(n_per_lead_samples),
            "rr_features": 2,
            "n_classes": scheme.n_classes,
            "cnn_channels": [32, 64],
            "cnn_blocks_per_stage": 2,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.2,
            "use_se": use_se,
        }
        model = CNNTransformer1D(**model_config).to(torch_device)
    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}; expected 'resnet' or 'cnn_transformer'."
        )

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

    # Build the train dataset/loader. AugmentedECGDataset operates on the
    # pre-loaded numpy arrays so DataLoader workers stay WFDB-free.
    if augment:
        augmenter = BeatAugmenter(fs=360.0, seed=seed)
        train_ds: torch.utils.data.Dataset = AugmentedECGDataset(
            morph_train, rr_train, y_train,
            in_channels=in_channels,
            samples_per_lead=n_per_lead_samples,
            augmenter=augmenter,
        )
    else:
        train_ds = TensorDataset(morph_train_t, rr_train_t, y_train_t)

    def make_loader(*, use_balanced: bool, alt_dataset: torch.utils.data.Dataset | None = None):
        ds = alt_dataset if alt_dataset is not None else train_ds
        # ``drop_last=True`` because the model has BatchNorm layers that
        # cannot handle a trailing size-1 batch in train mode. Dropping at
        # most ``batch_size - 1`` beats per epoch is harmless.
        if use_balanced:
            sampler = make_balanced_sampler(y_train, n_classes=scheme.n_classes, seed=seed)
            return DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                persistent_workers=bool(num_workers),
                drop_last=True,
            )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=bool(num_workers),
            drop_last=True,
        )

    cw = class_weights(y_train, n_classes=scheme.n_classes)
    weights = torch.tensor(
        [cw[k] for k in range(scheme.n_classes)],
        dtype=torch.float32,
        device=torch_device,
    )
    if focal_gamma > 0 or label_smoothing > 0:
        criterion = FocalLoss(
            gamma=focal_gamma, weight=weights, label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if warmup_epochs > 0:
        warmup = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: min(1.0, (e + 1) / max(1, warmup_epochs)),
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    scaler = torch.amp.GradScaler("cuda") if use_amp else None  # type: ignore[attr-defined]

    # EMA helper that shadows BOTH parameters and BN running statistics.
    ema_state: dict | None = None
    if ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update_ema():
        if ema_state is None:
            return
        msd = model.state_dict()
        for k, v in msd.items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
            else:
                # BN num_batches_tracked etc — copy verbatim.
                ema_state[k].copy_(v.detach())

    rng = np.random.default_rng(seed + 1)

    def run_epoch(loader, *, mixup: float):
        model.train()
        for morph_b, rr_b, y_b in loader:
            morph_b = morph_b.to(torch_device, non_blocking=True)
            rr_b = rr_b.to(torch_device, non_blocking=True)
            y_b = y_b.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if mixup > 0:
                morph_m, rr_m, y_a, y_b2, lam = mixup_batch(morph_b, rr_b, y_b, alpha=mixup, rng=rng)
                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(morph_m, rr_m)
                        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b2)
                else:
                    logits = model(morph_m, rr_m)
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b2)
            else:
                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(morph_b, rr_b)
                        loss = criterion(logits, y_b)
                else:
                    logits = model(morph_b, rr_b)
                    loss = criterion(logits, y_b)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            update_ema()

    # Two-stage schedule: stage 1 uses balanced sampler, stage 2 reverts to
    # natural distribution and (optionally) higher label-smoothing.
    if two_stage:
        stage1_epochs = int(round(0.7 * epochs))
        stage2_epochs = max(epochs - stage1_epochs, 1)
        loader_stage1 = make_loader(use_balanced=True)
        loader_stage2 = make_loader(use_balanced=False)
    else:
        stage1_epochs = epochs
        stage2_epochs = 0
        loader_stage1 = make_loader(use_balanced=balanced_sampler)
        loader_stage2 = None

    best_val_f1 = -1.0
    bad_epochs = 0
    best_state: dict | None = None
    last_epoch = 0
    epoch_idx = 0

    def eval_val_f1() -> float:
        model.eval()
        with torch.no_grad():
            val_logits = model(morph_val_t.to(torch_device), rr_val_t.to(torch_device))
            val_pred_local = val_logits.argmax(dim=1).cpu().numpy()
        return f1_score(y_val, val_pred_local, average="macro", zero_division=0)

    for stage_loader, stage_epochs, stage_label in [
        (loader_stage1, stage1_epochs, "stage1"),
        (loader_stage2, stage2_epochs, "stage2"),
    ]:
        if stage_loader is None or stage_epochs == 0:
            continue
        for _ in range(stage_epochs):
            last_epoch = epoch_idx
            run_epoch(stage_loader, mixup=mixup_alpha if stage_label == "stage1" else 0.0)
            scheduler.step()
            cur_f1 = eval_val_f1()
            if cur_f1 > best_val_f1 + 1e-4:
                best_val_f1 = cur_f1
                bad_epochs = 0
                # Save best-so-far weights — EMA-shadowed if EMA is in use,
                # otherwise the live model.
                if ema_state is not None:
                    best_state = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
                else:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
            epoch_idx += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits_t = model(morph_val_t.to(torch_device), rr_val_t.to(torch_device))
        val_logits_np = val_logits_t.cpu().numpy()
        val_proba = torch.softmax(val_logits_t, dim=1).cpu().numpy()
        val_pred = val_proba.argmax(axis=1)
        test_logits_t = model(morph_test_t.to(torch_device), rr_test_t.to(torch_device))
        test_logits_np = test_logits_t.cpu().numpy()
        test_proba = torch.softmax(test_logits_t, dim=1).cpu().numpy()
        test_pred = test_proba.argmax(axis=1)

    # ----- Post-hoc temperature scaling (fit on val, apply to test) -----
    from .calibration import TemperatureScaler

    fitted_temperature: float | None = None
    test_proba_calibrated = test_proba
    val_proba_calibrated = val_proba
    try:
        ts = TemperatureScaler()
        fitted_temperature = ts.fit(val_logits_np, y_val, max_iter=200)
        ts.save(out_dir / "temperature.pt")
        val_proba_calibrated = ts.apply_logits(val_logits_np)
        test_proba_calibrated = ts.apply_logits(test_logits_np)
    except (RuntimeError, ValueError):
        # If LBFGS fails (rare; e.g. degenerate val set), continue uncalibrated.
        fitted_temperature = None

    cls_range = range(scheme.n_classes)
    id_to_label = scheme.id_to_label
    metrics = {
        "model": "resnet1d",
        "scheme": scheme.name,
        "leads": list(leads),
        "config": model_config,
        "epochs_run": last_epoch + 1,
        "best_val_macro_f1": float(best_val_f1),
        "device": device,
        "temperature": fitted_temperature,
        "class_distribution_train": {id_to_label[k]: int((y_train == k).sum()) for k in cls_range},
        "class_distribution_val": {id_to_label[k]: int((y_val == k).sum()) for k in cls_range},
        "class_distribution_test": {id_to_label[k]: int((y_test == k).sum()) for k in cls_range},
        "val": _full_report(
            y_val, val_pred, val_proba_calibrated, id_to_label=id_to_label, bootstrap_seed=seed
        ),
        "test": _full_report(
            y_test, test_pred, test_proba_calibrated, id_to_label=id_to_label, bootstrap_seed=seed
        ),
        # Pre-calibration metrics, for comparison.
        "val_uncalibrated": _full_report(
            y_val, val_pred, val_proba, id_to_label=id_to_label, bootstrap_seed=seed
        ),
        "test_uncalibrated": _full_report(
            y_test, test_pred, test_proba, id_to_label=id_to_label, bootstrap_seed=seed
        ),
    }

    # ----- Plots (best-effort; gated on matplotlib being available) -----
    try:
        from .plots import confusion_matrix_plot, reliability_diagram

        confusion_matrix_plot(
            metrics["test"]["confusion_matrix"],
            scheme.labels,
            out_dir / "confusion_matrix_test.png",
            title=f"Confusion matrix (test, {scheme.name})",
        )
        confusion_matrix_plot(
            metrics["val"]["confusion_matrix"],
            scheme.labels,
            out_dir / "confusion_matrix_val.png",
            title=f"Confusion matrix (val, {scheme.name})",
        )
        if "calibration_bins" in metrics["test"]:
            cb = metrics["test"]["calibration_bins"]
            reliability_diagram(
                cb["centers"], cb["confidences"], cb["accuracies"], cb["counts"],
                out_dir / "reliability_diagram.png",
                title=f"Reliability diagram (test, {scheme.name})",
                ece=metrics["test"].get("ece"),
            )
    except ImportError:
        pass

    weights_path = out_dir / "resnet1d.pt"
    config_path = out_dir / "model_config.json"
    metrics_path = out_dir / "metrics_resnet.json"
    split_path = out_dir / "record_split.json"
    torch.save(model.state_dict(), weights_path)
    # Persist the v0.4 architecture flags + scheme alongside the bare model
    # config so ECGClassifier.load_resnet can rebuild the right ResNet1D and
    # tag it with the correct LabelScheme.
    saved_config = {
        **model_config,
        "architecture": architecture,
        "scheme": scheme.name,
    }
    if architecture == "resnet":
        saved_config["use_se"] = bool(use_se)
        saved_config["stem"] = str(stem_kind)
    config_path.write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
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
