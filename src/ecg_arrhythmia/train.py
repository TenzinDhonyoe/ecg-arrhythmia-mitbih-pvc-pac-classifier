"""CLI for training ECG arrhythmia models."""

from __future__ import annotations

import argparse
from pathlib import Path

from .training import train_baseline_lr, train_baseline_quick_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia models on MIT-BIH.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to MIT-BIH dataset directory containing RECORDS.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"), help="Output directory for model artifacts.")
    parser.add_argument("--model", choices=["baseline"], default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-smoke", action="store_true", help="Run a synthetic smoke-training path (no MIT-BIH data required).")
    args = parser.parse_args()

    if args.model == "baseline":
        if args.quick_smoke:
            artifacts = train_baseline_quick_smoke(args.out_dir, seed=args.seed)
        else:
            if args.data_dir is None:
                raise SystemExit("--data-dir is required unless --quick-smoke is set.")
            artifacts = train_baseline_lr(args.data_dir, args.out_dir, seed=args.seed)
        print(f"Saved model: {artifacts.model_path}")
        print(f"Saved scaler: {artifacts.scaler_path}")
        print(f"Saved metrics: {artifacts.metrics_path}")
        print(f"Saved split: {artifacts.split_path}")


if __name__ == "__main__":
    main()
