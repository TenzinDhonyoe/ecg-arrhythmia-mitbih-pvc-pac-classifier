# Changelog

All notable changes to this project are documented in this file.

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
