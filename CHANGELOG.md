# Changelog

All notable changes to this project are documented in this file.

## [0.4.0] - 2026-04-27

A major ML upgrade: AAMI EC57 5-class support, calibrated probabilities,
uncertainty quantification, attention-based architecture, and a robust
training stack. The legacy 3-class N/V/a API is preserved end-to-end —
existing artifacts continue to work, and the v0.3 ResNet weights load via
the new `ResNet1D` without state-dict warnings.

### Added — Label space
- **AAMI EC57 5-class scheme** (`N` / `S` / `V` / `F` / `Q`) selectable via
  `--scheme aami5` and `LabelScheme`/`get_scheme` in `src/ecg_arrhythmia/labels.py`.
  Mapping follows de Chazal 2004 (`N←{N,L,R,e,j}`, `S←{A,a,J,S}`, `V←{V,E}`,
  `F←{F}`, `Q←{Q,/,f,?}`; `|` deliberately excluded — see `docs/MODEL_CARD.md`).
- **DE_CHAZAL_DS1/DS2 inter-patient split** via `split_de_chazal()` and
  `--split ds1ds2`. Lets v0.4 numbers be directly comparable to published
  AAMI EC57 baselines.
- **`--exclude-paced-records`** flag drops MIT-BIH 102/104/107/217 (paced),
  matching standard AAMI evaluation practice.

### Added — Architecture
- **SE-ResNet-1D**: Squeeze-Excitation channel attention (`ResNet1D(use_se=True)`)
  and an optional **multi-scale Inception stem** (`stem="inception"`).
  `use_se=False` is bit-identical to v0.3 — old weights load unchanged.
- **CNN-Transformer hybrid (experimental)** in
  `src/ecg_arrhythmia/models/cnn_transformer.py`: SE-ResNet stem → 2-layer
  Transformer encoder (d_model=64, 4 heads) → CLS token + RR fusion → head.
  Same `(morph, rr) -> logits` contract as the ResNet, so it slots into
  `ECGClassifier` and ONNX export unchanged. Selectable via
  `--architecture cnn_transformer`.

### Added — Training stack
- **Signal augmentation pipeline** in `src/ecg_arrhythmia/augment.py`
  (amplitude scaling, Gaussian noise at controllable SNR, baseline wander,
  time-warp, temporal shift). Deterministic given a seed; RR features are
  invariant under augmentation by construction.
- **AugmentedECGDataset** + multi-worker `DataLoader` (workers never touch
  WFDB — they consume pre-loaded numpy arrays).
- **Focal loss + label smoothing + mixup** in `src/ecg_arrhythmia/losses.py`.
  `FocalLoss(γ=0)` is bit-identical to weighted CE.
- **WeightedRandomSampler** for class-balanced batches; **two-stage training**
  (`--two-stage`) runs balanced + focal in stage 1, natural distribution +
  label smoothing in stage 2.
- **Linear LR warmup → cosine annealing** via `--warmup-epochs`.
- **Mixed precision** (`torch.amp`, gated to CUDA — MPS is not safe).
- **EMA weights** (`--ema-decay 0.999`) that shadow both parameters and
  BatchNorm running statistics.
- **Gradient clipping** (`--grad-clip 1.0`) and bigger default `--epochs 50`.

### Added — Calibration + uncertainty
- **Post-hoc temperature scaling** in `src/ecg_arrhythmia/calibration.py`
  (LBFGS on val NLL). Fits automatically at the end of `train_resnet_1d`,
  saves `temperature.pt` next to the weights, and `ECGClassifier.from_artifacts`
  auto-loads + applies it at inference.
- **MC dropout** (`MCDropoutEnsemble`) and a new
  `ECGClassifier.predict_with_uncertainty(...)` that decorates each
  `BeatPrediction` with an epistemic-entropy `uncertainty` field.
- **Expected Calibration Error (ECE)** + **Brier score** added to
  `metrics_*.json`; `metrics["test_uncalibrated"]` is recorded alongside
  the calibrated `metrics["test"]` so calibration's effect is auditable.

### Added — Reporting
- 5×5 confusion matrices, reliability diagrams, and (optional) t-SNE plots
  in `src/ecg_arrhythmia/plots.py`. Saved to each artifact dir.
- `metrics["test"]["balanced_accuracy_ci95"]` is the headline figure for the
  README; macro-F1 stays as a secondary number.

### Changed — API back-compat
- `BeatPrediction.to_dict()` emits `prob_<label>` for every class in the
  loaded scheme. For 3-class it produces the literal `prob_N`/`prob_V`/`prob_a`
  identically to v0.3.
- `BeatPrediction.uncertainty` is a new optional field, `None` by default.
- `ECGClassifier` now carries a `LabelScheme` (auto-sniffed from a sibling
  `metrics_*.json`) and an optional `temperature` (auto-loaded from
  `temperature.pt`).
- `class_weights(y, n_classes=...)`: the previously hardcoded `n_classes=3`
  is gone.

### Tests
- +66 new tests across `test_label_scheme.py`, `test_aami_labels.py`,
  `test_augment.py`, `test_losses.py`, `test_se_resnet.py`,
  `test_calibration.py`, `test_cnn_transformer.py`. Total 113+ passing.
- `test_se_resnet.py` pins **bit-level back-compat** of the v0.3 ResNet
  weights against the v0.4 `ResNet1D` (no missing/unexpected keys, finite
  forward pass).

### Known limitations (current state of v0.4.0)
- **Trained AAMI 5-class SE-ResNet not shipped** in `artifacts/`. Single-pass
  training on MIT-BIH alone collapses to the majority N class under all the
  recipes we tried (weighted CE, focal+balanced sampler, two-stage). Published
  AAMI 5-class results on MIT-BIH inter-patient typically use hand-crafted
  features or self-supervised pretraining; both are deliberately out of scope
  for v0.4. The LR baseline on AAMI 5-class is shipped as the published
  `artifacts/baseline_aami5/`. Users wanting to train an AAMI 5-class neural
  model can start from `make train-aami` and tune from there.
- **Class-weight cap** (`max_weight=50` in `class_weights()`) is a
  numerical-safety default since AAMI 5-class on MIT-BIH produces uncapped
  weights >1000 for the Q class (~6 train beats), which causes the model to
  collapse to predicting Q always. Pass `max_weight=None` to opt out.

### Deferred to v0.4.1
- AAMI 5-class trained SE-ResNet artifact, with longer training + larger
  hyperparameter sweep / PTB-XL pretraining.
- 1D Grad-CAM saliency notebook.
- Conformal prediction sets.
- Cross-dataset evaluation on PTB-XL / CPSC.

## [0.3.0] - 2026-04-27

### Added
- **High-level Python API.** New `ECGClassifier`, `BeatPrediction`, and
  `PredictionResult` classes in `src/ecg_arrhythmia/api.py`, exposed at the
  package root. Three lines from clone to predictions:
  `clf = ECGClassifier.from_artifacts("artifacts/baseline"); clf.predict(signal, input_fs=360)`.
  `from_artifacts` auto-detects baseline vs ResNet from the directory contents
  and falls back gracefully when PyTorch is missing.
- **Streaming / online inference.** `ecg_arrhythmia.streaming.StreamingClassifier`
  accepts samples chunk-by-chunk, maintains a sample buffer, locks polarity
  once, emits `BeatPrediction` objects only when the trailing window is
  stable, and trims the buffer so memory stays bounded for long streams.
  Companion `examples/streaming_demo.py`.
- **ONNX export with parity check.** `ecg_arrhythmia.export` exports the LR
  baseline (sklearn pipeline → single ONNX graph, ~5 KB) and the ResNet-1D
  (TorchScript path, single self-contained file with dynamic batch axis,
  ~2.2 MB), both with onnxruntime parity assertions at machine precision.
  New `ecg-export-onnx` console script and `scripts/export_onnx.py` CLI.
  Optional `[onnx]` extra (`skl2onnx`, `onnxruntime`, `onnxscript`).
  Shipped artifacts now include `artifacts/{baseline,resnet}/*.onnx`.
- **CLI: JSON / CSV output and per-run summary.**
  - `ecg-infer --output-format {text,json,csv}` for pipeline-friendly output.
  - `--top-k` shows the K most likely classes per beat.
  - JSON payloads include a `summary` block (n_beats, mean HR, class counts,
    mean confidence, backend) plus a per-run `info` block.
- **Confidence-aware results.** `BeatPrediction` carries `confidence` (the
  argmax probability) and a full `probabilities` dict; `PredictionResult`
  exposes `.confidences`, `.probabilities`, `.heart_rate_bpm()`,
  `.class_counts()`, `.summary()`, and iteration / slicing.
- **Quickstart Jupyter notebook** at `examples/quickstart.ipynb` (load
  pretrained → predict → labelled-ECG plot + confidence histogram).
- **Tests.** `tests/test_api.py`, `test_streaming.py`, `test_export_onnx.py`
  (auto-skipped without `[onnx]`), `test_cli_infer.py`. +17 tests, all green.

### Changed
- `ecg_arrhythmia.infer.main()` is now a function with explicit `argv`,
  re-implemented on top of the public `ECGClassifier` API. Existing CLI
  flags and default output remain backward compatible.
- Package `__init__` re-exports `ECGClassifier`, `BeatPrediction`,
  `PredictionResult` for `from ecg_arrhythmia import ECGClassifier`.
- Version bumped to `0.3.0`.

## [0.2.0] - 2026-04-25

### Added
- **ResNet-1D + RR-feature head** under the optional `[deep]` extra (PyTorch).
  - New `src/ecg_arrhythmia/models/resnet1d.py` (~6 residual blocks, GAP, FC head).
  - `train_resnet_1d()` in `training.py` with class-weighted CE, cosine LR, and
    early-stopping on val macro-F1.
  - `run_resnet_inference()` and `--model resnet` CLI flag in `infer.py`.
- **Multi-lead support.**
  - `segment_record(..., lead=...)` accepts a channel index or signal-name
    string (`"MLII"`, `"V1"`, `"V5"`, …).
  - `available_leads()` helper exposes the channels in a record.
  - `train_baseline_lr(..., leads=("MLII",))` and `--leads MLII V1` CLI flag;
    multi-lead concatenates morphology windows along the feature axis.
- **Wearable inference path.**
  - `WearableInferenceConfig`, `auto_polarity_check()`.
  - `--lead {I,II,III,MLII,V1,V2,V5,unknown}`, `--auto-polarity /
    --no-auto-polarity`, `--invert-polarity` flags in `infer.py`.
  - `examples/wearable_lead_i_synthetic.csv` plus
    `scripts/make_wearable_demo.py` to regenerate it from a MIT-BIH record.
  - Domain-shift warning emitted when the requested lead is not MLII/II.
- **Honest, real metrics on MIT-BIH.**
  - `_full_report()` writes per-class precision/recall/F1/support, confusion
    matrix, ROC-AUC (one-vs-rest macro), and 1000-sample bootstrap 95% CIs
    for balanced accuracy and macro F1.
  - `train_baseline_lr` and `train_resnet_1d` write inference latency
    benchmarks and model sizes to `artifacts/<model>/benchmarks.json`.
- **Reproducible setup.**
  - `scripts/fetch_mitbih.py` downloads MIT-BIH via `wfdb.dl_database` and
    reconstructs the `RECORDS` index when needed.
  - `Makefile` with `make install`, `make test`, `make lint`, `make smoke`,
    `make fetch-mitbih`, `make train`, `make train-resnet`, `make wearable-demo`.
- **Open-source repo polish.**
  - `SECURITY.md`, `CODE_OF_CONDUCT.md`,
    `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md`,
    `.github/PULL_REQUEST_TEMPLATE.md`.
  - CI matrix expanded to Python 3.10/3.11/3.12 × Ubuntu/macOS, plus a
    `smoke-cli` integration job and a `test-deep` job that exercises the
    ResNet path under `[dev,deep]`.
  - `pyproject.toml` gains `[deep]` and `[bench]` extras, ruff config, and
    `pytest-cov`.
- **Tests.**
  - `test_training_smoke.py`, `test_metrics_report.py`,
    `test_wearable_inference.py`, `test_resnet.py` (auto-skipped when torch
    is absent).

### Changed
- Smoke artifacts now live at `artifacts/smoke/` (`metrics_smoke.json` carries
  an explicit "this is synthetic" warning) so they cannot be confused with the
  real LR baseline numbers at `artifacts/baseline/metrics_baseline.json`.
- README reframed: explicit "single-lead MLII training, single-lead wearable
  inference with documented domain shift" — replacing the previous implicit
  "3-lead wearable" framing.
- Default `--out-dir` for `ecg-train` is now `artifacts/baseline/`.
- `training.py` docstring corrected (previously referenced ResNet that did
  not exist; now both LR and ResNet ship).

### Removed
- The misleading committed `metrics_baseline.json` containing the synthetic
  `1.0 / 1.0` smoke metrics.

## [0.1.0] - 2026-04-25

### Added
- Refactored codebase into reusable package modules under `src/ecg_arrhythmia`.
- CLI entry points for training and inference:
  - `python -m ecg_arrhythmia.train`
  - `python -m ecg_arrhythmia.infer`
- Synthetic quick smoke mode for fast setup validation:
  - `python -m ecg_arrhythmia.train --model baseline --quick-smoke`
- Documentation set for open-source readiness:
  - `README.md`
  - `docs/DATA.md`
  - `docs/MODEL_CARD.md`
  - `docs/REPRODUCIBILITY.md`
  - `CONTRIBUTING.md`
  - `CITATION.cff`
- Continuous integration workflow with lint and tests.
- Test suite for preprocessing and inference utilities.
- Demo asset at `docs/demo.gif`.

### Changed
- Strengthened repository policy and `.gitignore` to avoid committing sensitive data and model artifacts.
- Training pipeline now saves reusable artifacts and split/metrics metadata in `artifacts/`.

### Removed
- Legacy root-level scripts, ad-hoc binaries, and tracked sample/personal data files from the initial capstone layout.
