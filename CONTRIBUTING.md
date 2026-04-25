# Contributing

Thanks for contributing.

## Setup

```bash
pip install -e ".[dev]"
```

## Development Checks

```bash
pytest
python -m ruff check src tests
```

## PR Guidelines

- Keep changes scoped and documented.
- Add/update tests when behavior changes.
- Do not commit raw ECG datasets or private patient data.
- Keep medical-safety disclaimers intact.
