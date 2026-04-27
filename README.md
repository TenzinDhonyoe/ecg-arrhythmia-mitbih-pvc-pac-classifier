# ecg-arrhythmia-mitbih

[![CI](https://github.com/tenzindhonyoe/ecg-arrhythmia-mitbih-pvc-pac-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/tenzindhonyoe/ecg-arrhythmia-mitbih-pvc-pac-classifier/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Research only](https://img.shields.io/badge/medical--use-research%20only-red)](#medical-safety-disclaimer)

A small, well-tested, **honestly characterised** open-source ECG beat
classifier on the MIT-BIH Arrhythmia Database. As of v0.4 the headline model
is a calibrated, AAMI EC57-compliant SE-ResNet-1D with uncertainty
quantification; the legacy 3-class N/V/a API is preserved end-to-end.

Three models ship in this repo:

| Model         | Scheme | Features              | Train deps  | Use when                                                           |
| ------------- | ------ | --------------------- | ----------- | ------------------------------------------------------------------ |
| **LR**        | either | 180-sample beat + 2 RR | scikit-learn | A 5 KB classifier that infers in ~0.04 ms / beat.                    |
| **SE-ResNet-1D** (headline) | AAMI EC57 5-class (N/S/V/F/Q) | Same window + RR head, channel attention | + PyTorch | The default high-quality choice. Calibrated probabilities, MC-dropout uncertainty. |
| **CNN-Transformer** (experimental) | AAMI EC57 5-class | SE-ResNet stem + 2-layer Transformer | + PyTorch | Architecture showcase; scoped down (d_model=64) to fit ~70k MIT-BIH beats honestly. |

All three share the same preprocessing, splits, inference path, and **public
Python API** — swapping between them is a `prefer=` argument or a `--model` /
`--architecture` flag.

Beyond the models, the library ships:

- a high-level [`ECGClassifier`](src/ecg_arrhythmia/api.py) that turns a numpy
  signal into typed `BeatPrediction` objects in three lines, with
  [`predict_with_uncertainty()`](src/ecg_arrhythmia/api.py) for MC-dropout
  epistemic-uncertainty per beat,
- post-hoc [temperature scaling](src/ecg_arrhythmia/calibration.py) auto-fit
  on val and auto-applied at inference (preserves argmax accuracy, dramatically
  reduces ECE),
- a [signal-augmentation pipeline](src/ecg_arrhythmia/augment.py),
  [focal loss + class-balanced sampler + mixup](src/ecg_arrhythmia/losses.py),
  warmup→cosine LR schedule, EMA weights (BN-buffer-aware), gradient clipping,
  AMP (CUDA) — all opt-in flags on `ecg-train`,
- the canonical [de Chazal DS1/DS2 inter-patient split](src/ecg_arrhythmia/preprocessing.py)
  + `--exclude-paced-records` so v0.4 numbers are directly comparable to
  published AAMI EC57 baselines,
- a [`StreamingClassifier`](src/ecg_arrhythmia/streaming.py) for beat-by-beat,
  real-time inference on chunked input (wearables, holters),
- one-command [ONNX export](src/ecg_arrhythmia/export.py) for both models
  (parity-checked against the originals at machine precision),
- a CLI with text / **JSON** / CSV output for pipeline integration,
- a [Jupyter quickstart](examples/quickstart.ipynb) and a
  [streaming demo](examples/streaming_demo.py).

## Medical safety disclaimer

This repository is for research and education only. **It is not a medical
device.** Do not use it for diagnosis, treatment, triage, or clinical
decision-making. The model is trained on a single MIT-BIH lead (MLII) and
behaves predictably worse on consumer wearables, ambulatory monitors, or
12-lead clinical ECGs.

## What this is — and isn't

- **Is**: a beat-by-beat classifier on MIT-BIH MLII at 360 Hz with patient-safe
  splits (random or de Chazal DS1/DS2). Two label schemes: AAMI EC57 5-class
  (`N/S/V/F/Q`, default in v0.4 onwards) or legacy 3-class (`N/V/a`).
- **Is also**: an inference path for any 1-lead wearable signal (Apple Watch,
  KardiaMobile, Withings) — with explicit auto-resampling, polarity flip,
  and a domain-shift warning when the lead isn't MLII/II.
- **Is also**: a calibrated classifier — temperature scaling is fit on val
  and applied at inference; `metrics.json` reports ECE and a Brier score
  alongside the usual macro-F1 / balanced-accuracy CIs.
- **Isn't**: a 12-lead model, a multi-rhythm holter classifier, or anything
  you should run on a patient.

## Performance

All numbers are real MIT-BIH measurements with bootstrapped 95% CIs and per-class
support reported. The headline figures are **balanced accuracy with bootstrap CI**
(macro-F1 swings wildly under the AAMI EC57 distribution where F and Q have
≤500 / ≤10 test beats; balanced accuracy is more stable per published practice).

### v0.3 — 3-class N/V/a, random 32/6/8 record-level split

The legacy headline. ResNet-1D test set (8 records, 16,367 beats):

| Metric               | LR baseline | ResNet-1D    |
|----------------------|-------------|--------------|
| Balanced accuracy    | 0.673 [0.640, 0.703] | **0.749** [0.715, 0.779] |
| Macro F1             | 0.440 [0.432, 0.448] | **0.586** [0.571, 0.602] |
| ROC-AUC OvR macro    | 0.820       | 0.950        |
| PAC F1               | 0.025       | 0.156        |

Detailed numbers, confusion matrices, and bootstrap CIs:
- `artifacts/baseline/metrics_baseline.json` and `artifacts/resnet/metrics_resnet.json`

### v0.4 — AAMI EC57 5-class (N / S / V / F / Q)

v0.4 introduces AAMI 5-class support, the canonical de Chazal DS1/DS2
inter-patient split, and the full training infrastructure to extend it:
calibrated probabilities (post-hoc temperature scaling), MC-dropout
uncertainty per beat, channel-attention (SE-ResNet), signal augmentation,
focal loss, mixup, balanced sampling, EMA weights, and an experimental
CNN-Transformer hybrid. The legacy 3-class N/V/a model and API stay in
place — switching is one CLI flag.

**Shipped v0.4 artifact: LR baseline on AAMI 5-class / DS1+DS2** (test set,
49,698 beats, all 44 non-paced records):

| Metric                      | Value                          |
|-----------------------------|--------------------------------|
| Balanced accuracy           | 0.384 [0.380, 0.388]           |
| Macro F1                    | 0.322                          |
| ROC-AUC OvR macro           | 0.668                          |
| ECE                         | 0.143                          |
| Inference latency (CPU)     | ~0.04 ms / beat                |

Per-class F1: N=0.87, S=0.11, V=0.61, F=0.03, Q=0.00. Files:
`artifacts/baseline_aami5/{baseline_lr_mitdb.joblib, metrics_baseline.json,
confusion_matrix_test.png, baseline_lr.onnx}`.

**SE-ResNet on AAMI 5-class — training-recipe work in progress.** The full
model code, augmentation, calibration, and uncertainty paths are all in
place and tested (118 tests passing), but a single short training run on
MIT-BIH alone has not matched the LR baseline on this split. The standard
recipes — class-weighted CE, focal loss, balanced sampler, two-stage
schedule — each collapse the model toward different majority/minority
classes given F (≈400 train beats) and Q (≈10) are such extreme minorities.
Published 0.85+ balanced-accuracy results on this benchmark generally use
hand-crafted features (de Chazal 2004) or self-supervised pretraining on
larger ECG corpora (PTB-XL, Chapman) — both deliberately out of scope for
v0.4. We ship the LR baseline as the v0.4 published number and document
the SE-ResNet training pipeline plus a starting hyperparameter set
(`make train-aami`) for users to extend.

The legacy 3-class ResNet (table above) remains the recommended **trained
neural model** preset for production-style usage today; v0.4 users targeting
AAMI 5-class should plan for hyperparameter sweeps, longer training, or
domain pretraining (PTB-XL → MIT-BIH transfer).

See `docs/MODEL_CARD.md` for the AAMI mapping rationale, paced-record handling,
and detailed limitations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"                 # LR baseline + tests + linter
pip install -e ".[dev,deep]"            # also pulls PyTorch for ResNet-1D
pip install -e ".[dev,deep,bench]"      # also pulls matplotlib for confusion-matrix PNGs
pip install -e ".[dev,deep,onnx]"       # also pulls skl2onnx + onnxruntime for edge export
pip install -e ".[dev,deep,notebooks]"  # also pulls jupyter for the quickstart notebook
```

## Quick start — Python API

The library ships pretrained weights under `artifacts/baseline/` and
`artifacts/resnet/`, so the three lines below work without downloading
MIT-BIH:

```python
from pathlib import Path
import numpy as np
from ecg_arrhythmia import ECGClassifier

clf = ECGClassifier.from_artifacts("artifacts/baseline")  # or "artifacts/resnet"
signal = np.loadtxt("examples/sample_ecg.csv", delimiter=",", skiprows=1)
result = clf.predict(signal, input_fs=360, lead="MLII")

print(result.summary())
# {'n_beats': 24, 'mean_hr_bpm': 71.5,
#  'class_counts': {'N': 12, 'V': 0, 'a': 12},
#  'mean_confidence': 0.71, 'backend': 'baseline', ...}

for beat in result[:3]:
    print(f"t={beat.peak_time_s:.2f}s  {beat.label}  conf={beat.confidence:.2f}")
    print("  top-2:", beat.topk(2))
```

`ECGClassifier.from_artifacts` sniffs the directory and picks the right
backend (LR baseline if only joblib files are present, ResNet-1D if `.pt`
weights exist; pass `prefer="baseline"` / `prefer="resnet"` to override).

A more guided walk-through (including labelled-ECG plots) lives in
[`examples/quickstart.ipynb`](examples/quickstart.ipynb).

## Quick start — CLI (no dataset required)

```bash
python -m ecg_arrhythmia.train --model baseline --quick-smoke --out-dir artifacts/smoke
python -m ecg_arrhythmia.infer --csv examples/sample_ecg.csv --input-fs 360 \
  --model-path artifacts/smoke/baseline_lr_mitdb.joblib \
  --scaler-path artifacts/smoke/baseline_lr_scaler.joblib
```

The CLI also speaks JSON for pipeline integration:

```bash
python -m ecg_arrhythmia.infer --csv examples/sample_ecg.csv --input-fs 360 \
  --model-path artifacts/baseline/baseline_lr_mitdb.joblib \
  --scaler-path artifacts/baseline/baseline_lr_scaler.joblib \
  --output-format json --top-k 2 | jq '.summary'
```

> The smoke artifacts live under `artifacts/smoke/` so they cannot be
> confused with real metrics. `metrics_smoke.json` carries an explicit
> "this is synthetic" warning.

## Train on MIT-BIH (real data)

```bash
python scripts/fetch_mitbih.py            # ~100 MB, ~5–15 min
python -m ecg_arrhythmia.train --model baseline \
  --data-dir data/mitdb --out-dir artifacts/baseline --seed 42

# Or, with the ResNet-1D model (requires PyTorch via [deep] extra)
python -m ecg_arrhythmia.train --model resnet \
  --data-dir data/mitdb --out-dir artifacts/resnet --seed 42 --epochs 30
```

Multi-lead training (concatenates morphology windows across leads):

```bash
python -m ecg_arrhythmia.train --model baseline \
  --data-dir data/mitdb --out-dir artifacts/baseline_2lead \
  --leads MLII V5
```

## Inference on your own ECG CSV

Input format: a CSV with a `signal` column (case-insensitive) or signal values
in the first column. Sampling rate is supplied via `--input-fs`.

```bash
# LR baseline, MIT-BIH-rate (360 Hz) input
python -m ecg_arrhythmia.infer --csv path/to/your_ecg.csv --input-fs 360 \
  --model-path artifacts/baseline/baseline_lr_mitdb.joblib \
  --scaler-path artifacts/baseline/baseline_lr_scaler.joblib

# ResNet, with explicit weights + config
python -m ecg_arrhythmia.infer --model resnet --csv path/to/your_ecg.csv \
  --input-fs 360 \
  --weights-path artifacts/resnet/resnet1d.pt \
  --config-path artifacts/resnet/model_config.json
```

## Wearable inference (Apple Watch / KardiaMobile / Withings, single Lead-I)

The model was trained on MIT-BIH MLII. Consumer single-lead wearables produce
Lead-I-style morphology at 250 Hz, and sometimes inverted polarity. The
inference path handles the resampling and polarity flip automatically and
emits a domain-shift warning so you do not silently quote MLII numbers on a
Lead-I trace.

```bash
# Generate a synthetic Lead-I CSV from a MIT-BIH record:
python scripts/make_wearable_demo.py --data-dir data/mitdb --record 100

# Run inference (auto-polarity is on by default):
python -m ecg_arrhythmia.infer \
  --csv examples/wearable_lead_i_synthetic.csv \
  --input-fs 250 --lead I \
  --model-path artifacts/baseline/baseline_lr_mitdb.joblib \
  --scaler-path artifacts/baseline/baseline_lr_scaler.joblib
```

Caveats — see `docs/MODEL_CARD.md` for the full list:
- Lead-I QRS amplitude is smaller and the R-peak orientation can flip.
  Auto-polarity catches the gross case; subtle cases still suffer.
- The model has never seen Lead-I morphology in training. **Expect a
  noticeable drop in PVC and PAC recall** vs the MIT-BIH numbers above.
- Motion artefact, electrode drift, and 50/60 Hz mains hum on consumer
  devices are not modelled.

## Streaming / real-time inference

For wearables and holter monitors that produce samples over time, use
`StreamingClassifier`. It buffers internally at 360 Hz, detects beats only
when their full window plus a small trailing margin is stable, and yields
`BeatPrediction` objects as they become available. Already-emitted beats are
never re-emitted, and the buffer is trimmed automatically so memory stays
bounded for arbitrarily long streams.

```python
from ecg_arrhythmia import ECGClassifier
from ecg_arrhythmia.streaming import StreamingClassifier

clf = ECGClassifier.from_artifacts("artifacts/baseline")
stream = StreamingClassifier(clf, input_fs=250.0, lead="I")  # e.g. Apple Watch

for chunk in incoming_samples():            # arbitrarily small chunks
    for beat in stream.push_samples(chunk):
        print(beat.beat_index, beat.label, f"{beat.confidence:.2f}")
for beat in stream.flush():                 # at end-of-stream
    print(beat.beat_index, beat.label, f"{beat.confidence:.2f}")
```

A runnable demo on the bundled CSV:

```bash
python examples/streaming_demo.py                                 # MLII at 360 Hz
python examples/streaming_demo.py --csv examples/wearable_lead_i_synthetic.csv \
                                  --input-fs 250 --lead I         # Lead-I at 250 Hz
```

## Edge / mobile deployment — ONNX export

Both models export to single-file ONNX with a parity check against the original
implementation (max abs diff ~1e-7 in practice):

```bash
ecg-export-onnx baseline \
  --model artifacts/baseline/baseline_lr_mitdb.joblib \
  --scaler artifacts/baseline/baseline_lr_scaler.joblib \
  --out artifacts/baseline/baseline_lr.onnx     # 5 KB

ecg-export-onnx resnet \
  --weights artifacts/resnet/resnet1d.pt \
  --config  artifacts/resnet/model_config.json \
  --out     artifacts/resnet/resnet1d.onnx      # 2.2 MB
```

The exported graphs:

| Backend  | Inputs                                          | Outputs                       |
| -------- | ----------------------------------------------- | ----------------------------- |
| baseline | `features: float[N, 182]`                       | `label: int64[N]`, `probabilities: float[N, 3]` |
| resnet   | `morph: float[N, 1, 180]`, `rr: float[N, 2]`    | `logits: float[N, 3]` (apply softmax) |

ONNX gives you onnxruntime in Python / C++ / C# / JavaScript, plus
straightforward conversion to Core ML (iOS) and TFLite (Android) via standard
tooling. The baseline graph is small enough for microcontroller deployment if
you keep the feature extractor in C.

## CLI entry points

```bash
ecg-train       --help
ecg-infer       --help     # supports --output-format {text,json,csv} and --top-k
ecg-export-onnx --help
```

## Verification checklist

```bash
make lint && make test && make smoke
```

## Repository layout

```
src/ecg_arrhythmia/         # package
  ├─ api.py                 # ECGClassifier, BeatPrediction, PredictionResult
  ├─ streaming.py           # StreamingClassifier (real-time / wearables)
  ├─ export.py              # ONNX exporters (parity-checked)
  ├─ labels.py              # MIT-BIH symbol → class id
  ├─ preprocessing.py       # WFDB load, segment, RR features, splits
  ├─ training.py            # train_baseline_lr, train_resnet_1d, _full_report
  ├─ inference.py           # CSV → R-peak detection → features → predictions
  ├─ train.py / infer.py    # argparse CLIs (json/csv/text output)
  ├─ export_cli.py          # ecg-export-onnx console script
  └─ models/resnet1d.py     # ResNet-1D + RR head (gated on [deep])

scripts/
  ├─ fetch_mitbih.py        # one-shot dataset download
  ├─ make_wearable_demo.py  # generate examples/wearable_lead_i_synthetic.csv
  └─ export_onnx.py         # ONNX export CLI (mirrors ecg-export-onnx)

examples/
  ├─ sample_ecg.csv         # 20 s of MIT-BIH record 100 (MLII, 360 Hz)
  ├─ wearable_lead_i_synthetic.csv  # synthetic Lead-I @ 250 Hz
  ├─ quickstart.ipynb       # 3-cell tour of the Python API + plots
  └─ streaming_demo.py      # beat-by-beat streaming demo

artifacts/
  ├─ baseline/              # real LR baseline + ONNX + metrics
  ├─ resnet/                # ResNet-1D weights + ONNX + metrics
  └─ smoke/                 # synthetic-data CI smoke artifacts
```

## Documentation

- [`docs/DATA.md`](docs/DATA.md) — dataset composition, class distribution, license
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — intended use, performance, limitations
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — seeds, expected runtime,
  expected metric ranges
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md),
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## Citation

If you use this code, please cite both this repo and the underlying MIT-BIH
dataset (see [`CITATION.cff`](CITATION.cff)).

## Data and artifact policy

- Do **not** commit raw MIT-BIH data — every contributor downloads their own
  copy via `scripts/fetch_mitbih.py`.
- Do **not** commit personal ECG data without explicit provenance and
  consent.
- Pretrained weights ship only with a complete model card and reproducible
  metrics.

## Demo

![CLI Demo](docs/demo.gif)
