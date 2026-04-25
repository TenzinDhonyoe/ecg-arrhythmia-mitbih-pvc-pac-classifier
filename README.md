# ECG Arrhythmia MIT-BIH PVC/PAC Classifier

Research-grade ECG beat classifier for MIT-BIH Arrhythmia data.

Classes:
- `N` = Normal
- `V` = PVC
- `a` = PAC (MIT-BIH `A` and `a` merged)

## Medical Safety Disclaimer

This repository is for research and education only. It is **not** a medical device and must not be used for diagnosis, treatment, triage, or clinical decision-making.

## Features

- Reusable package under `src/ecg_arrhythmia`
- Baseline model: logistic regression with morphology + RR-ratio features
- Patient-safe split by record ID (no record overlap across train/val/test)
- Train-only scaling and saved split metadata
- CLI training and inference workflows
- Tests + CI

## Repository Name

Recommended public repository name: `ecg-arrhythmia-mitbih-pvc-pac-classifier`

## Installation

### Prerequisites

- Python 3.10+
- `pip`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start (No Dataset Required)

Run a complete smoke pipeline with synthetic training data:

```bash
python -m ecg_arrhythmia.train --model baseline --quick-smoke --out-dir artifacts
```

This writes:
- `artifacts/baseline_lr_mitdb.joblib`
- `artifacts/baseline_lr_scaler.joblib`
- `artifacts/metrics_baseline.json`
- `artifacts/record_split.json`

## Train on MIT-BIH

1) Download MIT-BIH from PhysioNet and keep it local (do not commit dataset files).  
2) Point `--data-dir` to the folder containing `RECORDS`.

```bash
python -m ecg_arrhythmia.train \
  --model baseline \
  --data-dir /path/to/mit-bih-arrhythmia-database-1.0.0 \
  --out-dir artifacts \
  --seed 42
```

## Inference on Custom ECG CSV

Input CSV requirements:
- A `signal` column, or signal values in the first column
- Sampling rate provided with `--input-fs`

```bash
python -m ecg_arrhythmia.infer \
  --csv /path/to/your_ecg.csv \
  --model-path artifacts/baseline_lr_mitdb.joblib \
  --scaler-path artifacts/baseline_lr_scaler.joblib \
  --input-fs 360 \
  --out predictions.csv
```

Notes:
- If the CLI reports `No valid beats detected`, verify signal quality, duration, and `--input-fs`.
- Consumer device ECG can differ substantially from MIT-BIH morphology and lead configuration.

## CLI Entry Points

After install, these are also available:

```bash
ecg-train --help
ecg-infer --help
```

## Verification Checklist

```bash
pytest -q
python -m ruff check src tests
python -m ecg_arrhythmia.train --model baseline --quick-smoke --out-dir artifacts
python -m ecg_arrhythmia.infer --csv examples/sample_ecg.csv --input-fs 360 --model-path artifacts/baseline_lr_mitdb.joblib --scaler-path artifacts/baseline_lr_scaler.joblib
```

## Documentation

- Data and usage constraints: `docs/DATA.md`
- Model limitations and intended use: `docs/MODEL_CARD.md`
- Reproducibility details: `docs/REPRODUCIBILITY.md`
- Contribution guide: `CONTRIBUTING.md`

## Data and Artifact Policy

- Do not commit raw MIT-BIH data
- Do not commit personal ECG data without explicit provenance/consent
- Release pretrained weights only with complete model documentation
