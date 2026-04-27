"""Console-script entry point for ONNX export.

Mirrors ``scripts/export_onnx.py`` so users who installed the package via pip
can run ``ecg-export-onnx baseline ...`` without checking out the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .export import export_baseline_to_onnx, export_resnet_to_onnx


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
