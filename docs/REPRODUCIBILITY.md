# Reproducibility

## End-to-end reproduction

```bash
# 1. Set up env
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,deep,bench]"

# 2. Fetch data (~100 MB, ~5–15 min)
python scripts/fetch_mitbih.py

# 3. Reproduce LR baseline (~1–2 min on a laptop CPU)
python -m ecg_arrhythmia.train --model baseline \
  --data-dir data/mitdb --out-dir artifacts/baseline --seed 42

# 4. Reproduce ResNet-1D (~5–15 min on CPU; faster on Apple-MPS / CUDA)
python -m ecg_arrhythmia.train --model resnet \
  --data-dir data/mitdb --out-dir artifacts/resnet \
  --seed 42 --epochs 30 --batch-size 256
```

## Determinism and seeds

- `--seed 42` controls:
  - Record-level shuffle in `split_by_record` (NumPy generator).
  - LR `random_state` (`lbfgs` is deterministic given the same seed and
    inputs).
  - ResNet PyTorch RNG via the default global state (model init, dropout,
    DataLoader shuffle ordering when `num_workers=0`).
- Bootstrap CI uses `seed` and `seed + 1` for balanced-accuracy and
  macro-F1 separately so the two CIs are independent.

> Caveat: full bit-exactness across machines is **not** promised — BLAS
> non-determinism in CPU math libraries and CUDA/MPS kernel choice can
> shift the last 1–2 decimal places. Bootstrap CIs cover this comfortably.

## Leakage controls

- **Record-level split.** Train, val, and test sets share **no** record
  IDs. This is a patient-safe proxy on MIT-BIH (one record == one patient).
- **Train-only scaling.** `StandardScaler` is fit on the training set and
  applied unchanged to val and test.
- **Per-record RR features.** RR-interval ratios are normalised by the
  median RR within a single record to avoid cross-record timing leakage.

## Expected outputs

| Path                                                    | What                |
|---------------------------------------------------------|---------------------|
| `artifacts/baseline/baseline_lr_mitdb.joblib`           | Trained LR (sklearn pickle) |
| `artifacts/baseline/baseline_lr_scaler.joblib`          | Fitted StandardScaler |
| `artifacts/baseline/metrics_baseline.json`              | Full report (val + test) |
| `artifacts/baseline/record_split.json`                  | Train/val/test record IDs |
| `artifacts/baseline/benchmarks.json`                    | Latency + size benchmarks |
| `artifacts/baseline/confusion_matrix_test.png`          | Confusion matrix PNG (with `[bench]`) |
| `artifacts/resnet/resnet1d.pt`                          | PyTorch state dict |
| `artifacts/resnet/model_config.json`                    | Architecture hyperparameters |
| `artifacts/resnet/metrics_resnet.json`                  | Full report (val + test) |
| `artifacts/resnet/record_split.json`                    | Train/val/test record IDs |
| `artifacts/resnet/benchmarks.json`                      | Latency + size benchmarks |

## Expected metric ranges (sanity check)

If your reproduction lands far outside these bands, something is off
(wrong lead, wrong sampling rate, broken split, BLAS issue, etc):

| Metric (test)                   | Expected         | LR    | ResNet |
|---------------------------------|------------------|-------|--------|
| Balanced accuracy               | 0.65–0.78        | 0.673 | 0.749  |
| Macro F1                        | 0.42–0.60        | 0.440 | 0.586  |
| ROC-AUC (OvR macro)             | 0.80–0.96        | 0.820 | 0.950  |
| LR inference latency, CPU p50   | <0.1 ms          | 0.04 ms | —    |
| ResNet inference latency, CPU p50 | <2 ms          | —     | 0.66 ms |

## Hardware used for the shipped numbers

- Apple Silicon laptop CPU (LR baseline; deterministic).
- Apple Silicon `mps` device (ResNet training; ~5 minutes for 11 epochs
  with early stopping).
- Inference benchmarks always recorded on CPU regardless of training
  device, so they are portable.

## Re-generating confusion-matrix PNGs

The PNGs are produced by a small post-processing helper that ships with
the `[bench]` extra (matplotlib). You can regenerate them from any
`metrics_*.json` with:

```bash
python -c "import json,numpy as np,matplotlib;matplotlib.use('Agg');\
import matplotlib.pyplot as plt;\
m=json.load(open('artifacts/baseline/metrics_baseline.json'));\
cm=np.array(m['test']['confusion_matrix']);labels=m['test']['labels'];\
fig,ax=plt.subplots();ax.imshow(cm,cmap='Blues');\
ax.set_xticks(range(len(labels)));ax.set_yticks(range(len(labels)));\
ax.set_xticklabels(labels);ax.set_yticklabels(labels);\
fig.savefig('cm.png',dpi=120)"
```
