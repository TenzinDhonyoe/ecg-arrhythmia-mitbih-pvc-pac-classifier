# Model Card — ECG Beat Classifier

## Summary

This repo ships beat classifiers under two label schemes:

**AAMI EC57 5-class** (default since v0.4):
- `N` — Normal beats and bundle-branch blocks (AAMI N ← MIT-BIH `N, L, R, e, j`)
- `S` — Supraventricular ectopic (AAMI S ← MIT-BIH `A, a, J, S`)
- `V` — Ventricular (AAMI V ← MIT-BIH `V, E`)
- `F` — Fusion of V and N (AAMI F ← MIT-BIH `F`)
- `Q` — Unknown / paced (AAMI Q ← MIT-BIH `Q, /, f, ?`)

**Legacy 3-class** (v0.1–v0.3, still selectable via `--scheme mitbih3`):
`N` / `V` (PVC) / `a` (PAC; MIT-BIH `A` and `a` merged).

| Model              | Scheme | What it is                                               |
| ------------------ | ------ | -------------------------------------------------------- |
| `baseline_lr`      | either | Class-weighted scikit-learn `LogisticRegression`         |
| `resnet1d` (v0.3)  | mitbih3 | 6-block 1D ResNet (32→64→128 ch) + RR-feature FC head   |
| `se_resnet_aami5` (v0.4 headline) | aami5 | Same backbone + Squeeze-Excitation attention + calibrated probabilities |
| `cnn_transformer_aami5` (v0.4 experimental) | aami5 | SE-ResNet stem + 2-layer Transformer encoder (d_model=64) + RR fusion |

The v0.4 headline model is the **calibrated** SE-ResNet under the AAMI 5-class
scheme on the canonical de Chazal DS1/DS2 inter-patient split — the
configuration most published MIT-BIH benchmarks use, so v0.4 numbers are
directly comparable to the literature.

### AAMI mapping notes

- **`J` is part of S, not N.** Junctional/nodal premature beats are
  supraventricular ectopic per AAMI EC57.
- **`|` is excluded.** Some annotation tools use it as an artifact marker
  rather than a beat label, so we don't include it in the Q class. If you
  re-train with a private dataset where `|` is a real beat label, add it to
  `AAMI5.label_to_id` explicitly.
- **Paced records** (MIT-BIH 102, 104, 107, 217) contain the bulk of Q-class
  beats; the v0.4 headline metrics use the canonical de Chazal split which
  already excludes them. The `--exclude-paced-records` flag exists for the
  random split only.

## Inputs

- 180-sample beat window centered on a detected R-peak (target FS 360 Hz).
- 2 RR-ratio timing features (`RR_pre/median`, `RR_post/median`) computed
  per record to avoid cross-record timing leakage.
- Multi-lead training is supported by concatenating per-lead morphology
  windows before the head; the shipped models use a single MLII channel.

## Output

Class probabilities for the active scheme (`{N, V, a}` for `mitbih3`,
`{N, S, V, F, Q}` for `aami5`). Argmax → predicted label. For
ResNet-backed classifiers with a fitted `temperature.pt` next to the
weights, `ECGClassifier.from_artifacts` auto-applies temperature scaling
to the logits at inference, so the reported probabilities are the
calibrated ones; pass `temperature=None` to disable. `predict_with_uncertainty()`
adds a per-beat **MC-dropout epistemic entropy** in the `uncertainty`
field of each `BeatPrediction`.

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

- Record-level or de Chazal DS1/DS2 split — no record overlap between
  train / val / test in either case.
- StandardScaler fit on train only and applied to val / test
  (LR baseline; ResNet uses BatchNorm internally).
- Class weights: `(n_samples / (n_classes * count))`, **capped at 50** since v0.4.
  Without the cap, AAMI 5-class produces weights >1000 for the Q class
  (~6 train beats), which causes the model to collapse to predicting that
  class always. The cap leaves minorities meaningfully upweighted without
  letting six examples dominate the loss.
- LR: `lbfgs`, `max_iter=8000`, `random_state=42`.
- ResNet (v0.3, 3-class default): Adam + cosine LR schedule,
  class-weighted cross-entropy, early stopping on val macro-F1
  (patience 5, max epochs 30, batch 128).
- ResNet (v0.4, AAMI 5-class default): same backbone with optional
  Squeeze-Excitation attention; opt-in flags for warmup→cosine schedule,
  signal augmentation, focal loss, label smoothing, mixup,
  WeightedRandomSampler, two-stage training, EMA weights (with BN-buffer
  shadowing), gradient clipping, and AMP (CUDA-only). See `make train-aami`
  for the published-numbers config.
- **Post-hoc temperature scaling** (LBFGS on val NLL) for the ResNet path.
  Saved as `temperature.pt` and applied automatically at inference by
  `ECGClassifier.from_artifacts`. Argmax accuracy is preserved.
- Bootstrap 95% CIs over 1000 resamples of the val/test predictions.

## Intended use

- Research, method benchmarking, and education.
- Reproducible 3-class N/PVC/PAC baseline for downstream ECG tooling.
- A working multi-lead and wearable-inference *path* you can extend.

## Not intended for

- Clinical diagnosis, treatment decisions, emergency triage, or
  unsupervised medical deployment.
- Anything beyond beat-level classification on MIT-BIH morphology.
  Rhythm-level / multi-second arrhythmia classification, AF detection,
  and 12-lead diagnosis are out of scope.

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
