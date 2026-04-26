# Model Card — ECG Beat Classifier (LR baseline + ResNet-1D)

## Summary

Two MIT-BIH MLII beat classifiers for three classes:

- `N` — Normal
- `V` — PVC (Premature Ventricular Contraction)
- `a` — PAC (Premature Atrial Contraction; MIT-BIH `A` and `a` merged)

| Model        | What it is                                              |
| ------------ | ------------------------------------------------------- |
| `baseline_lr` | Class-weighted scikit-learn `LogisticRegression`       |
| `resnet1d`    | 6-block 1D ResNet (32→64→128 ch) + RR-feature FC head  |

Both models share preprocessing, splits, and CSV inference path. The LR is
intended as a fast, interpretable baseline; the ResNet-1D is the best-accuracy
model that ships with this repo.

## Inputs

- 180-sample beat window centered on a detected R-peak (target FS 360 Hz).
- 2 RR-ratio timing features (`RR_pre/median`, `RR_post/median`) computed
  per record to avoid cross-record timing leakage.
- Multi-lead training is supported by concatenating per-lead morphology
  windows before the head; the shipped models use a single MLII channel.

## Output

Class probabilities for `N`, `V`, `a`. Argmax → predicted label. The
inference CLI (`ecg-infer`) writes per-beat predictions with R-peak sample
index, predicted class, and per-class probability.

## Training data

- Source: MIT-BIH Arrhythmia Database (PhysioNet, v1.0.0).
- 48 records × 30 min, 360 Hz, single-lead MLII channel.
- Patient-safe record-level split (seed 42, ratios 0.7 / 0.15 / 0.15):
  - 32 train records (54 507 beats) → N: 48 389 · V: 5 710 · a (PAC): 2 408
  - 6 val records (11 716 beats) → N: 11 230 · V: 309 · a: 177
  - 8 test records (16 367 beats) → N: 15 152 · V: 1 104 · a: 111

Class imbalance is severe (PAC = 2.4% of training beats). Both models use
class-weighted training to compensate.

## Performance (test set, seed 42)

| Metric                       | LR baseline  | ResNet-1D    |
|------------------------------|--------------|--------------|
| Balanced accuracy            | 0.673        | **0.749**    |
| 95 % bootstrap CI            | [0.640, 0.703] | [0.715, 0.779] |
| Macro F1                     | 0.440        | **0.586**    |
| 95 % bootstrap CI            | [0.432, 0.448] | [0.571, 0.602] |
| ROC-AUC (OvR macro)          | 0.820        | **0.950**    |
| F1 N                         | 0.744        | **0.948**    |
| F1 V (PVC)                   | 0.552        | **0.655**    |
| F1 a (PAC)                   | 0.025        | **0.156**    |
| Parameters                   | ~550         | 551 975      |
| Inference latency (CPU, p50) | **0.04 ms**  | 0.66 ms      |
| Model size                   | 5.2 KB       | 2.2 MB       |

Full per-class precision/recall/F1, confusion matrices, and bootstrap CIs:
- `artifacts/baseline/metrics_baseline.json`
- `artifacts/resnet/metrics_resnet.json`
- `artifacts/comparison.md`

## How these were produced

```bash
python scripts/fetch_mitbih.py
python -m ecg_arrhythmia.train --model baseline \
  --data-dir data/mitdb --out-dir artifacts/baseline --seed 42
python -m ecg_arrhythmia.train --model resnet \
  --data-dir data/mitdb --out-dir artifacts/resnet --seed 42 --epochs 30
```

## Training protocol

- Record-level split (no record overlap between train / val / test).
- StandardScaler fit on train only and applied to val / test
  (LR baseline; ResNet uses BatchNorm internally).
- Class weights: `(n_samples / (n_classes * count))`.
- LR: `lbfgs`, `max_iter=8000`, `random_state=42`.
- ResNet: Adam + cosine LR schedule, class-weighted cross-entropy,
  early stopping on val macro-F1 (patience 5, max epochs 30, batch 256).
- Bootstrap 95% CIs over 1000 resamples of the val/test predictions.

## Intended use

- Research, method benchmarking, and education.
- Reproducible 3-class N/PVC/PAC baseline for downstream ECG tooling.
- A working multi-lead and wearable-inference *path* you can extend.

## Not intended for

- Clinical diagnosis, treatment decisions, emergency triage, or
  unsupervised medical deployment.
- Comprehensive arrhythmia classification (we cover only N / PVC / PAC).
- AAMI EC57 reporting (we use raw MIT-BIH symbols, not the N/S/V/F/Q
  superclass mapping).

## Limitations

### Honest weak points (visible in the numbers above)
- **PAC F1 is 0.025 (LR) / 0.156 (ResNet).** Only 111 PAC beats in the
  test split, morphology overlaps with `N`, and the dataset just does not
  have enough PAC-rich training data. **Do not gate clinical workflow on
  PAC predictions from these models.**
- **Class-weighted LR over-predicts rare classes.** The LR baseline pays
  for its high PVC recall with N-recall of 0.60, which produces noisy
  predictions on long traces with mostly Normal beats. The ResNet model
  largely fixes this (N recall 0.91).

### Distributional caveats
- **Single-lead MLII training.** The model has never seen Lead I, V1,
  V2, or precordial morphologies as primary inputs. Expect material
  accuracy drops on consumer wearable signals.
- **Sampling rate.** Trained at 360 Hz; the inference helpers resample
  inputs but cannot recover spectral content lost below ~100 Hz cutoff
  on cheap analog front-ends.
- **Polarity.** Some consumer devices output an inverted Lead-I signal.
  The wearable inference path includes an `auto_polarity_check` heuristic
  that flips the gross case; subtle inversions still hurt.
- **Motion artefact and 50 / 60 Hz mains hum** are not modelled.
- **Label simplification.** MIT-BIH `A` and `a` are merged into one PAC
  class. We do not predict fusion, paced, or unknown beats.

## Wearable use

If you point either model at a single-lead Lead-I signal (Apple Watch,
KardiaMobile, Withings) the pipeline will:

1. Resample to 360 Hz.
2. Optionally flip polarity (`--auto-polarity` is on by default; the
   heuristic compares the upper and lower 1 % tails of the bandpassed
   signal over a 5 s window).
3. Run the standard QRS detector + 180-sample window extraction.
4. Emit a domain-shift `UserWarning` so the lead mismatch is on the
   record, not silent.

You can reproduce a representative wearable trace from MIT-BIH with
`scripts/make_wearable_demo.py`. Cross-domain numbers (Lead-I synthetic
vs MLII real) are not currently shipped — they would require a real
wearable-device dataset to be trustworthy.

## Reproducibility

- Deterministic seed: 42 (controls split + LR `random_state` + ResNet
  weight init via `torch`).
- Bootstrap CIs use a separate seed family (`seed`, `seed+1`) for
  balanced-accuracy and macro-F1 respectively.
- Real metrics live under `artifacts/baseline/` and `artifacts/resnet/`
  and are re-generated end-to-end by the commands in "How these were
  produced".

## Future work

- AAMI EC57 5-class (N / S / V / F / Q) variant.
- Cross-domain evaluation on a real consumer-wearable dataset (PhysioNet
  Computing in Cardiology 2017 or similar).
- Beat-level uncertainty estimates (e.g. Monte Carlo dropout for the
  ResNet head) so downstream code can defer on low-confidence predictions
  rather than always argmax.
