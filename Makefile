.PHONY: install install-deep install-onnx test lint smoke fetch-mitbih train train-resnet eval export-onnx streaming-demo clean

PYTHON ?= python3
DATA_DIR ?= data/mitdb

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-deep:
	$(PYTHON) -m pip install -e ".[dev,deep,bench]"

install-onnx:
	$(PYTHON) -m pip install -e ".[dev,deep,onnx]"

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

# v0.4 headline: AAMI EC57 5-class SE-ResNet on the de Chazal DS1/DS2 split.
# Requires the [deep,bench] extras for matplotlib confusion-matrix output.
train-aami:
	$(PYTHON) -m ecg_arrhythmia.train --model resnet \
		--data-dir $(DATA_DIR) --out-dir artifacts/se_resnet_aami5 \
		--scheme aami5 --split ds1ds2 \
		--architecture resnet --use-se --augment \
		--focal-gamma 2.0 --grad-clip 1.0 --warmup-epochs 5 \
		--ema-decay 0.999 --num-workers 2 \
		--epochs 60 --batch-size 256 --lr 1e-3 \
		--patience 10 --seed 42 --device auto

# Baseline LR for the same scheme + split, for the README comparison row.
train-aami-baseline:
	$(PYTHON) -m ecg_arrhythmia.train --model baseline \
		--data-dir $(DATA_DIR) --out-dir artifacts/baseline_aami5 \
		--scheme aami5 --split ds1ds2 --seed 42

# Experimental CNN-Transformer (scoped down: 2 layers, d_model=64).
train-aami-transformer:
	$(PYTHON) -m ecg_arrhythmia.train --model resnet \
		--data-dir $(DATA_DIR) --out-dir artifacts/cnn_transformer_aami5 \
		--scheme aami5 --split ds1ds2 \
		--architecture cnn_transformer --use-se \
		--epochs 30 --batch-size 128 --lr 5e-4 \
		--patience 8 --seed 42 --device auto

# Export the v0.4 AAMI5 artifacts to ONNX.
export-aami-onnx:
	$(PYTHON) scripts/export_onnx.py baseline \
		--model artifacts/baseline_aami5/baseline_lr_mitdb.joblib \
		--scaler artifacts/baseline_aami5/baseline_lr_scaler.joblib \
		--out artifacts/baseline_aami5/baseline_lr.onnx
	$(PYTHON) scripts/export_onnx.py resnet \
		--weights artifacts/se_resnet_aami5/resnet1d.pt \
		--config artifacts/se_resnet_aami5/model_config.json \
		--out artifacts/se_resnet_aami5/se_resnet.onnx

wearable-demo:
	$(PYTHON) scripts/make_wearable_demo.py --data-dir $(DATA_DIR)
	$(PYTHON) -m ecg_arrhythmia.infer \
		--csv examples/wearable_lead_i_synthetic.csv --input-fs 250 --lead I \
		--model-path artifacts/baseline/baseline_lr_mitdb.joblib \
		--scaler-path artifacts/baseline/baseline_lr_scaler.joblib

export-onnx:
	$(PYTHON) scripts/export_onnx.py baseline \
		--model artifacts/baseline/baseline_lr_mitdb.joblib \
		--scaler artifacts/baseline/baseline_lr_scaler.joblib \
		--out artifacts/baseline/baseline_lr.onnx
	$(PYTHON) scripts/export_onnx.py resnet \
		--weights artifacts/resnet/resnet1d.pt \
		--config artifacts/resnet/model_config.json \
		--out artifacts/resnet/resnet1d.onnx

streaming-demo:
	$(PYTHON) examples/streaming_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info build dist
