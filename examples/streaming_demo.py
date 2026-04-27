"""Streaming-inference demo.

Reads ``examples/sample_ecg.csv`` (or any CSV passed via --csv) and feeds
samples to :class:`StreamingClassifier` in 0.5-second chunks, printing each
beat as it is emitted. This mimics how a wearable would call into the library:
samples arrive over time, the classifier maintains its own buffer, and you get
a stream of :class:`BeatPrediction` objects.

Usage
-----
    python examples/streaming_demo.py
    python examples/streaming_demo.py --csv path/to/wearable.csv --input-fs 250 --lead I
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ecg_arrhythmia import ECGClassifier
from ecg_arrhythmia.streaming import StreamingClassifier


def _load_signal(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return np.array([float(row[0]) for row in reader if row], dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("examples/sample_ecg.csv"))
    p.add_argument("--input-fs", type=float, default=360.0)
    p.add_argument(
        "--lead",
        choices=["I", "II", "III", "MLII", "V1", "V2", "V5", "unknown"],
        default="MLII",
    )
    p.add_argument("--artifacts", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--chunk-s", type=float, default=0.5, help="Streaming chunk size (s).")
    args = p.parse_args()

    classifier = ECGClassifier.from_artifacts(args.artifacts)
    stream = StreamingClassifier(classifier, input_fs=args.input_fs, lead=args.lead)

    sig = _load_signal(args.csv)
    chunk_size = max(1, int(args.chunk_s * args.input_fs))
    print(
        f"Streaming {len(sig)} samples ({len(sig)/args.input_fs:.1f}s) "
        f"in {chunk_size}-sample chunks ({args.chunk_s}s)..."
    )

    for i in range(0, len(sig), chunk_size):
        for beat in stream.push_samples(sig[i : i + chunk_size]):
            print(
                f"  t={beat.peak_time_s:6.2f}s  beat #{beat.beat_index:3d}  "
                f"{beat.label}  conf={beat.confidence:.3f}"
            )
    for beat in stream.flush():
        print(
            f"  t={beat.peak_time_s:6.2f}s  beat #{beat.beat_index:3d}  "
            f"{beat.label}  conf={beat.confidence:.3f}  (flushed)"
        )

    print(f"Total beats emitted: {stream.state.n_beats_emitted}")


if __name__ == "__main__":
    main()
