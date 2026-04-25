# Changelog

All notable changes to this project are documented in this file.

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

