"""ONNX export utilities for the ECG models.

ONNX is the lingua franca for deploying ML models to embedded targets, mobile
runtimes (Core ML / TFLite via converters), and language ecosystems other than
Python. This module provides one-shot exporters for both backends:

- :func:`export_baseline_to_onnx` — sklearn LR pipeline (scaler + classifier) → ONNX.
  Requires the ``[onnx]`` extra (``skl2onnx``, ``onnxruntime``).
- :func:`export_resnet_to_onnx` — PyTorch ResNet-1D → ONNX.
  Requires the ``[deep,onnx]`` extras.

Both exporters round-trip the model through ``onnxruntime`` and assert the
output matches the original within a small numerical tolerance, so a successful
export is also a parity test.

Example
-------
>>> from ecg_arrhythmia.export import export_baseline_to_onnx
>>> export_baseline_to_onnx(  # doctest: +SKIP
...     "artifacts/baseline/baseline_lr_mitdb.joblib",
...     "artifacts/baseline/baseline_lr_scaler.joblib",
...     "artifacts/baseline/baseline_lr.onnx",
... )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import WINDOW_SIZE

N_FEATURES = WINDOW_SIZE + 2  # 180 morphology samples + 2 RR ratios


@dataclass
class ExportReport:
    onnx_path: Path
    n_features_or_samples: int
    max_abs_diff: float
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "onnx_path": str(self.onnx_path),
            "n_features_or_samples": int(self.n_features_or_samples),
            "max_abs_diff": float(self.max_abs_diff),
            "notes": self.notes,
        }


def _require(pkg: str, hint: str) -> None:
    try:
        __import__(pkg)
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            f"{pkg} is required for ONNX export. Install with: pip install {hint}"
        ) from exc


def export_baseline_to_onnx(
    model_path: str | Path,
    scaler_path: str | Path,
    out_path: str | Path,
    *,
    n_features: int = N_FEATURES,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ExportReport:
    """Export the sklearn baseline (scaler + LR) as a single ONNX graph.

    The exported graph takes a single ``float[N, n_features]`` input named
    ``"features"`` and produces:
    - ``"label"``: ``int64[N]`` argmax class id (0=N, 1=V, 2=a)
    - ``"probabilities"``: ``float[N, 3]`` softmax probabilities

    Returns an :class:`ExportReport` containing the max absolute difference
    between sklearn and onnxruntime probabilities on a reference input.
    """
    import joblib

    _require("skl2onnx", "'ecg-arrhythmia-mitbih[onnx]'")
    _require("onnxruntime", "'ecg-arrhythmia-mitbih[onnx]'")

    from skl2onnx import to_onnx
    from sklearn.pipeline import Pipeline

    model = joblib.load(Path(model_path))
    scaler = joblib.load(Path(scaler_path))
    pipeline = Pipeline([("scaler", scaler), ("clf", model)])

    rng = np.random.default_rng(0)
    sample = rng.standard_normal((4, n_features)).astype(np.float32)

    onnx_model = to_onnx(
        pipeline,
        sample,
        target_opset=15,
        options={id(model): {"zipmap": False}},
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onnx_model.SerializeToString())

    # Parity check
    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_inputs = {session.get_inputs()[0].name: sample}
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_proba = np.asarray(onnx_outputs[1], dtype=float)
    sk_proba = pipeline.predict_proba(sample)
    max_diff = float(np.max(np.abs(onnx_proba - sk_proba)))
    if not np.allclose(onnx_proba, sk_proba, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"ONNX/sklearn parity check failed: max abs diff {max_diff} exceeds "
            f"rtol={rtol}, atol={atol}"
        )

    return ExportReport(
        onnx_path=out,
        n_features_or_samples=n_features,
        max_abs_diff=max_diff,
        notes="sklearn pipeline (scaler + LR) → ONNX, opset 15",
    )


def export_resnet_to_onnx(
    weights_path: str | Path,
    config_path: str | Path,
    out_path: str | Path,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    dynamic_batch: bool = True,
) -> ExportReport:
    """Export the ResNet-1D + RR head to ONNX.

    The exported graph has two named inputs:
    - ``"morph"``: ``float[N, in_channels, samples_per_lead]``
    - ``"rr"``:    ``float[N, rr_features]``

    And one named output:
    - ``"logits"``: ``float[N, n_classes]``

    Apply softmax in your runtime to recover probabilities.
    """
    import json

    _require("torch", "'ecg-arrhythmia-mitbih[deep]'")
    _require("onnxruntime", "'ecg-arrhythmia-mitbih[onnx]'")

    import torch

    from .models.resnet1d import ResNet1D

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    # Strip v0.4 bookkeeping fields the constructor doesn't accept.
    model_kwargs = {
        k: v for k, v in config.items()
        if k in {"in_channels", "samples_per_lead", "rr_features", "n_classes",
                 "channels", "blocks_per_stage", "use_se", "stem"}
    }
    model = ResNet1D(**model_kwargs)
    state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n = 4
    in_channels = config["in_channels"]
    n_samples = config["samples_per_lead"]
    rr_features = config["rr_features"]
    morph = torch.randn(n, in_channels, n_samples, dtype=torch.float32)
    rr = torch.randn(n, rr_features, dtype=torch.float32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "morph": {0: "batch"},
            "rr": {0: "batch"},
            "logits": {0: "batch"},
        }

    # ``dynamo=False`` forces the legacy tracer-based exporter, which produces a
    # single self-contained ONNX file. The newer dynamo path emits external
    # weights ("model.onnx" + "model.onnx.data") which is awkward for distribution.
    torch.onnx.export(
        model,
        (morph, rr),
        str(out),
        input_names=["morph", "rr"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Parity check
    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(
        None,
        {"morph": morph.numpy(), "rr": rr.numpy()},
    )[0]
    with torch.no_grad():
        torch_logits = model(morph, rr).numpy()
    max_diff = float(np.max(np.abs(onnx_logits - torch_logits)))
    if not np.allclose(onnx_logits, torch_logits, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"ONNX/torch parity check failed: max abs diff {max_diff} exceeds "
            f"rtol={rtol}, atol={atol}"
        )

    return ExportReport(
        onnx_path=out,
        n_features_or_samples=n_samples,
        max_abs_diff=max_diff,
        notes=f"ResNet-1D → ONNX, opset 17, dynamic_batch={dynamic_batch}",
    )


__all__ = ["export_baseline_to_onnx", "export_resnet_to_onnx", "ExportReport"]
