# ECG Arrhythmia MIT-BIH PVC PAC Classifier

Suggested repository name: `ecg-arrhythmia-mitbih-pvc-pac-classifier`

This repository provides a reproducible ECG beat-classification pipeline trained on the MIT-BIH Arrhythmia Database for 3 classes:

- `N` (normal)
- `V` (PVC)
- `a` (PAC, merged from MIT-BIH `A` and `a`)

## Important Medical Disclaimer

This project is for research and education only and is **not a medical device**.  
Do not use it for diagnosis, treatment, or any clinical decision-making.

## What Is Included

- Reusable Python package in `src/ecg_arrhythmia`
- Record-level split training pipeline (patient-safe split by record)
- Baseline logistic regression model with morphology + RR features
- Inference CLI for custom ECG CSV files
- Tests and CI for core preprocessing/inference behavior

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Train baseline model:

```bash
python -m ecg_arrhythmia.train --model baseline --data-dir /path/to/mit-bih-arrhythmia-database-1.0.0 --out-dir artifacts

# Quick smoke mode (no dataset required)
python -m ecg_arrhythmia.train --model baseline --quick-smoke --out-dir artifacts
```

Run inference on a CSV (column `signal`, or first column if unnamed):

```bash
python -m ecg_arrhythmia.infer --csv examples/sample_ecg.csv --model-path artifacts/baseline_lr_mitdb.joblib --scaler-path artifacts/baseline_lr_scaler.joblib --input-fs 360
```

## Data Access

MIT-BIH data is not committed in this repository. See `docs/DATA.md` for how to obtain it and required citation/usage terms.

## Reproducibility

- Fixed random seed is available via `--seed`
- Record split is saved to `artifacts/record_split.json`
- Validation/test metrics are saved to JSON in `artifacts/`

See `docs/REPRODUCIBILITY.md` for full details and limitations.

## Repository Policy

- No raw MIT-BIH data in git
- No personal/custom ECG data in git unless provenance and consent are explicitly documented
- Pretrained artifacts should be released only with a model card (`docs/MODEL_CARD.md`)
