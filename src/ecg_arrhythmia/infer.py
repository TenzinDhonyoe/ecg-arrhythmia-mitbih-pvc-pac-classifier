"""CLI for inference on custom ECG CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .inference import run_baseline_inference, run_resnet_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECG beat classification on a custom CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV file with ECG signal.")
    parser.add_argument(
        "--model",
        choices=["baseline", "resnet"],
        default="baseline",
        help="Which classifier to use.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/baseline/baseline_lr_mitdb.joblib"),
        help="Baseline only: path to the LR joblib model.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("artifacts/baseline/baseline_lr_scaler.joblib"),
        help="Baseline only: path to the StandardScaler joblib.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("artifacts/resnet/resnet1d.pt"),
        help="ResNet only: path to the .pt weights.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("artifacts/resnet/model_config.json"),
        help="ResNet only: path to the model_config.json that the weights were trained with.",
    )
    parser.add_argument("--input-fs", type=float, default=360.0, help="Sampling rate of input CSV.")
    parser.add_argument(
        "--lead",
        choices=["I", "II", "III", "MLII", "V1", "V2", "V5", "unknown"],
        default="unknown",
        help="Which lead the input represents. Affects the domain-shift warning.",
    )
    parser.add_argument(
        "--auto-polarity",
        dest="auto_polarity",
        action="store_true",
        default=True,
        help="Auto-detect inverted signal and flip it (default: on).",
    )
    parser.add_argument(
        "--no-auto-polarity",
        dest="auto_polarity",
        action="store_false",
        help="Disable auto polarity detection.",
    )
    parser.add_argument(
        "--invert-polarity",
        action="store_true",
        help="Force-invert the input before peak detection.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    if args.model == "baseline":
        rows = run_baseline_inference(
            args.csv,
            args.model_path,
            args.scaler_path,
            args.input_fs,
            lead_name=args.lead,
            auto_polarity=args.auto_polarity,
            invert_polarity=args.invert_polarity,
        )
    else:
        rows = run_resnet_inference(
            args.csv,
            args.weights_path,
            args.config_path,
            args.input_fs,
            lead_name=args.lead,
            auto_polarity=args.auto_polarity,
            invert_polarity=args.invert_polarity,
        )

    if not rows:
        print("No valid beats detected.")
        return
    print(f"Detected and classified {len(rows)} beats with model={args.model}.")
    for row in rows[:10]:
        print(
            f"beat={row['beat_index']:4d} label={row['label']} "
            f"probs=({row['prob_N']:.3f}, {row['prob_V']:.3f}, {row['prob_a']:.3f})"
        )
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
