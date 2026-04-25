"""CLI for inference on custom ECG CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .inference import run_baseline_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECG beat classification on a custom CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV file with ECG signal.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/baseline_lr_mitdb.joblib"))
    parser.add_argument("--scaler-path", type=Path, default=Path("artifacts/baseline_lr_scaler.joblib"))
    parser.add_argument("--input-fs", type=float, default=360.0, help="Sampling rate of input CSV.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    rows = run_baseline_inference(args.csv, args.model_path, args.scaler_path, args.input_fs)
    if not rows:
        print("No valid beats detected.")
        return
    print(f"Detected and classified {len(rows)} beats.")
    for row in rows[:10]:
        print(f"beat={row['beat_index']:4d} label={row['label']} probs=({row['prob_N']:.3f}, {row['prob_V']:.3f}, {row['prob_a']:.3f})")
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
