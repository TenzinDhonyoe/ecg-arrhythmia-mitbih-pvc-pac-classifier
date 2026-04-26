# Contributing

Thanks for contributing.

## Setup

```bash
# Lean install (LR baseline only)
pip install -e ".[dev]"

# Full install (also builds and tests the ResNet-1D path)
pip install -e ".[dev,deep,bench]"
```

## Development checks

```bash
make lint           # python -m ruff check src tests
make test           # pytest -q
make smoke          # train smoke + run inference end-to-end
```

The Makefile is a thin convenience layer; the `python -m ecg_arrhythmia.*`
commands also work directly.

## Reproducing real-data numbers locally

```bash
make fetch-mitbih   # one-time download (~100 MB, ~5–15 min)
make train          # LR baseline → artifacts/baseline/
make train-resnet   # ResNet-1D → artifacts/resnet/  (needs [deep] extra)
```

Expected metric ranges are in `docs/REPRODUCIBILITY.md`. If your
reproduction lands far outside them, check your lead config, sampling
rate, and seed.

## PR guidelines

- Keep changes scoped and documented. One PR, one workstream.
- Add or update tests when behaviour changes.
- Run `make lint && make test` before pushing.
- Update `docs/MODEL_CARD.md` and `artifacts/comparison.md` whenever
  performance numbers move — the README's headline metrics are read by
  downstream users, please don't let them drift.
- **Do not commit** raw ECG datasets, real patient data, or large model
  weights without prior discussion. The curated `artifacts/baseline/`
  and `artifacts/resnet/` directories are the only place we ship
  pre-trained pickles, and only with full reproducibility metadata.
- Keep medical-safety disclaimers intact.

## Reporting safety concerns

See [`SECURITY.md`](SECURITY.md). Clinical-safety concerns (the model
producing outputs that could mislead a downstream alerting system) get
priority over feature requests.
