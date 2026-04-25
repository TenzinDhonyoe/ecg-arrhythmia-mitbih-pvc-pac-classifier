# Reproducibility

## Determinism and Seeds

- Training supports `--seed` for reproducible split order and estimator RNG.
- Record split is saved to `artifacts/record_split.json`.

## Leakage Controls

- Split is by record ID (patient-safe proxy).
- StandardScaler is fit on train only and applied to val/test.
- Test set is never used for model selection in the baseline pipeline.

## Outputs

- `artifacts/baseline_lr_mitdb.joblib`
- `artifacts/baseline_lr_scaler.joblib`
- `artifacts/record_split.json`
- `artifacts/metrics_baseline.json`
