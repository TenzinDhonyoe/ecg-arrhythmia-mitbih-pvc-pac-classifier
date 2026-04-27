"""CLI for inference on custom ECG CSV files.

Output formats
--------------
- ``--output-format text`` (default): human-readable, prints first 10 beats and
  a one-line summary.
- ``--output-format json``: a single JSON object with ``summary`` and ``beats``
  keys, suitable for piping into other tools.
- ``--output-format csv``: just the beats, written to ``--out`` (or stdout).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .api import ECGClassifier, PredictionResult


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "csv"],
        default="text",
        help="How to print results to stdout. --out is used by both text and csv modes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Show top-k labels per beat in text/json output (default: 1).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV path.")
    return parser


def _load_classifier(args: argparse.Namespace) -> ECGClassifier:
    if args.model == "baseline":
        return ECGClassifier.load_baseline(args.model_path, args.scaler_path)
    return ECGClassifier.load_resnet(args.weights_path, args.config_path)


def _print_text(result: PredictionResult, top_k: int) -> None:
    if not result.beats:
        print("No valid beats detected.")
        return
    summary = result.summary()
    hr = summary["mean_hr_bpm"]
    hr_str = f"{hr:.1f} bpm" if hr is not None else "n/a"
    print(
        f"Detected and classified {summary['n_beats']} beats "
        f"(backend={result.info.get('backend','?')}, mean HR={hr_str}, "
        f"class counts={summary['class_counts']})."
    )
    for beat in result.beats[:10]:
        if top_k <= 1:
            print(
                f"beat={beat.beat_index:4d} t={beat.peak_time_s:6.2f}s "
                f"label={beat.label} conf={beat.confidence:.3f}"
            )
        else:
            top = beat.topk(top_k)
            top_str = ", ".join(f"{lab}={p:.3f}" for lab, p in top)
            print(
                f"beat={beat.beat_index:4d} t={beat.peak_time_s:6.2f}s "
                f"top{top_k}=[{top_str}]"
            )


def _print_json(result: PredictionResult, top_k: int) -> str:
    beats = []
    for beat in result.beats:
        rec = beat.to_dict()
        if top_k > 1:
            rec["topk"] = [{"label": lab, "probability": p} for lab, p in beat.topk(top_k)]
        beats.append(rec)
    payload = {
        "summary": result.summary(),
        "info": result.info,
        "beats": beats,
    }
    return json.dumps(payload, indent=2)


def _write_csv(result: PredictionResult, out_path: Path | None) -> None:
    rows = result.to_records()
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if out_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    classifier = _load_classifier(args)
    result = classifier.predict_csv(
        args.csv,
        input_fs=args.input_fs,
        lead=args.lead,
        auto_polarity=args.auto_polarity,
        invert_polarity=args.invert_polarity,
    )

    if args.output_format == "json":
        print(_print_json(result, args.top_k))
    elif args.output_format == "csv":
        _write_csv(result, args.out)
    else:  # text
        _print_text(result, args.top_k)
        if args.out and result.beats:
            with args.out.open("w", newline="", encoding="utf-8") as f:
                rows = result.to_records()
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
