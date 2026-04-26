"""Download the MIT-BIH Arrhythmia Database from PhysioNet to a local directory.

Usage:
    python scripts/fetch_mitbih.py [--dest data/mitdb]

This is a one-time setup step (~100 MB, ~2-5 min depending on connection). The
data is *not* committed to the repository — every contributor downloads their
own copy and points training at it via ``--data-dir``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import wfdb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/mitdb"),
        help="Destination directory (default: data/mitdb).",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    abs_dest = args.dest.resolve()
    print(f"Downloading MIT-BIH Arrhythmia Database to {abs_dest} ...")
    t0 = time.time()
    wfdb.dl_database("mitdb", dl_dir=str(args.dest))
    elapsed = time.time() - t0

    records_path = args.dest / "RECORDS"
    if not records_path.exists():
        # ``wfdb.dl_database`` does not always pull the RECORDS index. Re-create
        # it from the .hea files so downstream code works.
        record_stems = sorted(p.stem for p in args.dest.glob("*.hea"))
        if not record_stems:
            print("ERROR: no .hea files found — download appears incomplete.", file=sys.stderr)
            return 1
        records_path.write_text("\n".join(record_stems) + "\n", encoding="utf-8")
        print(f"Reconstructed RECORDS index from {len(record_stems)} .hea files.")
    n_records = len([line for line in records_path.read_text().splitlines() if line.strip()])
    files = sorted(p.name for p in args.dest.glob("*"))
    expected = 48
    print(f"Done in {elapsed:.1f}s. RECORDS lists {n_records} records, {len(files)} files on disk.")
    if n_records != expected:
        print(f"WARNING: expected {expected} records, found {n_records}.", file=sys.stderr)
        return 2
    print()
    print("Train the LR baseline on real MIT-BIH:")
    print(
        f"  python -m ecg_arrhythmia.train --model baseline "
        f"--data-dir {abs_dest} --out-dir artifacts/baseline --seed 42"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
