"""CLI for training ECG arrhythmia models."""

from __future__ import annotations

import argparse
from pathlib import Path

from .training import train_baseline_lr, train_baseline_quick_smoke, train_resnet_1d


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia models on MIT-BIH.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to MIT-BIH dataset directory containing RECORDS.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/baseline"),
        help="Output directory for model artifacts (default: artifacts/baseline).",
    )
    parser.add_argument("--model", choices=["baseline", "resnet"], default="baseline")
    parser.add_argument(
        "--leads",
        nargs="+",
        default=["MLII"],
        help="Lead name(s) to train on (default: MLII). Multi-lead concatenates morphology windows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run a synthetic smoke-training path (no MIT-BIH data required).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="ResNet only: epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="ResNet only: batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="ResNet only: learning rate.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="ResNet only: training device.",
    )
    args = parser.parse_args()

    if args.model == "baseline":
        if args.quick_smoke:
            out_dir = args.out_dir if "smoke" in str(args.out_dir) else Path("artifacts/smoke")
            artifacts = train_baseline_quick_smoke(out_dir, seed=args.seed)
        else:
            if args.data_dir is None:
                raise SystemExit("--data-dir is required unless --quick-smoke is set.")
            artifacts = train_baseline_lr(
                args.data_dir,
                args.out_dir,
                seed=args.seed,
                leads=tuple(args.leads),
            )
    else:  # resnet
        if args.quick_smoke:
            out_dir = args.out_dir if "smoke" in str(args.out_dir) else Path("artifacts/resnet_smoke")
            artifacts = train_resnet_1d(
                Path("."),
                out_dir,
                seed=args.seed,
                leads=tuple(args.leads),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                quick_smoke=True,
            )
        else:
            if args.data_dir is None:
                raise SystemExit("--data-dir is required unless --quick-smoke is set.")
            out_dir = args.out_dir if args.out_dir != Path("artifacts/baseline") else Path("artifacts/resnet")
            artifacts = train_resnet_1d(
                args.data_dir,
                out_dir,
                seed=args.seed,
                leads=tuple(args.leads),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                quick_smoke=False,
            )

    print(f"Saved model:   {artifacts.model_path}")
    print(f"Saved scaler:  {artifacts.scaler_path}")
    print(f"Saved metrics: {artifacts.metrics_path}")
    print(f"Saved split:   {artifacts.split_path}")


if __name__ == "__main__":
    main()
