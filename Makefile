.PHONY: install install-deep test lint smoke fetch-mitbih train train-resnet eval clean

PYTHON ?= python3
DATA_DIR ?= data/mitdb

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-deep:
	$(PYTHON) -m pip install -e ".[dev,deep,bench]"

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m ecg_arrhythmia.train --model baseline --quick-smoke --out-dir artifacts/smoke
	$(PYTHON) -m ecg_arrhythmia.infer --csv examples/sample_ecg.csv --input-fs 360 \
		--model-path artifacts/smoke/baseline_lr_mitdb.joblib \
		--scaler-path artifacts/smoke/baseline_lr_scaler.joblib

fetch-mitbih:
	$(PYTHON) scripts/fetch_mitbih.py --dest $(DATA_DIR)

train:
	$(PYTHON) -m ecg_arrhythmia.train --model baseline \
		--data-dir $(DATA_DIR) --out-dir artifacts/baseline --seed 42

train-resnet:
	$(PYTHON) -m ecg_arrhythmia.train --model resnet \
		--data-dir $(DATA_DIR) --out-dir artifacts/resnet --seed 42 --epochs 30

wearable-demo:
	$(PYTHON) scripts/make_wearable_demo.py --data-dir $(DATA_DIR)
	$(PYTHON) -m ecg_arrhythmia.infer \
		--csv examples/wearable_lead_i_synthetic.csv --input-fs 250 --lead I \
		--model-path artifacts/baseline/baseline_lr_mitdb.joblib \
		--scaler-path artifacts/baseline/baseline_lr_scaler.joblib

clean:
	rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info build dist
