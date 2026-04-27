"""CLI to export trained ECG models to ONNX.

Examples
--------
Export the LR baseline:

    python scripts/export_onnx.py baseline \
        --model artifacts/baseline/baseline_lr_mitdb.joblib \
        --scaler artifacts/baseline/baseline_lr_scaler.joblib \
        --out artifacts/baseline/baseline_lr.onnx

Export the ResNet-1D:

    python scripts/export_onnx.py resnet \
        --weights artifacts/resnet/resnet1d.pt \
        --config artifacts/resnet/model_config.json \
        --out artifacts/resnet/resnet1d.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_arrhythmia.export import export_baseline_to_onnx, export_resnet_to_onnx


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export trained ECG models to ONNX.")
    sub = p.add_subparsers(dest="backend", required=True)

    bp = sub.add_parser("baseline", help="Export the sklearn LR baseline.")
    bp.add_argument("--model", type=Path, required=True)
    bp.add_argument("--scaler", type=Path, required=True)
    bp.add_argument("--out", type=Path, required=True)

    rp = sub.add_parser("resnet", help="Export the ResNet-1D model.")
    rp.add_argument("--weights", type=Path, required=True)
    rp.add_argument("--config", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="Disable dynamic batch axis (export with fixed batch size 4).",
    )

    args = p.parse_args(argv)

    if args.backend == "baseline":
        report = export_baseline_to_onnx(args.model, args.scaler, args.out)
    else:
        report = export_resnet_to_onnx(
            args.weights,
            args.config,
            args.out,
            dynamic_batch=not args.no_dynamic_batch,
        )

    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
