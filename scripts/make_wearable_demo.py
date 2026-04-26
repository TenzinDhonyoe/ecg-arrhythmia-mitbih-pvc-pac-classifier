"""Generate a synthetic Apple-Watch-style Lead-I ECG CSV from a MIT-BIH record.

The output is *not* a real wearable recording — it is a simulation that
applies the kinds of transformations a Lead-I consumer device performs on a
ground-truth ECG: low-pass filtering (~40 Hz), downsampling to 250 Hz, mild
amplitude scaling, and small Gaussian noise. We use channel 1 (commonly a
precordial lead, e.g. V5) of a known MIT-BIH record because its R-amplitude
profile is closer to wearable single-lead than MLII is.

Usage:
    python scripts/make_wearable_demo.py [--data-dir data/mitdb] \
        [--record 100] [--out examples/wearable_lead_i_synthetic.csv] \
        [--seconds 30]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--record", type=str, default="100")
    parser.add_argument("--out", type=Path, default=Path("examples/wearable_lead_i_synthetic.csv"))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--target-fs", type=float, default=250.0)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.data_dir / "RECORDS").exists():
        print(
            f"ERROR: {args.data_dir}/RECORDS missing. Run scripts/fetch_mitbih.py first.",
            file=sys.stderr,
        )
        return 1

    cwd = Path.cwd()
    try:
        os.chdir(args.data_dir)
        record = wfdb.rdrecord(args.record)
    finally:
        os.chdir(cwd)

    src_fs = float(record.fs)
    n_samples = int(args.seconds * src_fs)
    channel = 1 if record.p_signal.shape[1] >= 2 else 0
    sig = record.p_signal[:n_samples, channel].astype(float)

    # Low-pass to ~40 Hz to mimic wearable analog front-end bandwidth.
    nyq = 0.5 * src_fs
    b, a = butter(4, 40.0 / nyq, btype="low")
    sig = filtfilt(b, a, sig)

    # Downsample to target_fs.
    n_out = int(len(sig) * args.target_fs / src_fs)
    sig = resample(sig, n_out)

    # Mild amplitude rescale + small noise.
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    rng = np.random.default_rng(args.seed)
    sig = sig + rng.normal(0.0, args.noise_std, size=sig.shape)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["signal"])
        for v in sig:
            writer.writerow([f"{v:.6f}"])

    print(f"Wrote {len(sig)} samples at {args.target_fs:.0f} Hz to {args.out}")
    print("Run inference with:")
    print(
        f"  python -m ecg_arrhythmia.infer --csv {args.out} "
        f"--input-fs {args.target_fs:.0f} --lead I "
        "--model-path artifacts/baseline/baseline_lr_mitdb.joblib "
        "--scaler-path artifacts/baseline/baseline_lr_scaler.joblib"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
