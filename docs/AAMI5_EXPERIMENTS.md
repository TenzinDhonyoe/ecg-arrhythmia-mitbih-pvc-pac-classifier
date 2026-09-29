# AAMI 5-class neural-model reproduction

This is an independent, **exploratory** reproduction of the two v0.4 neural
architectures on MIT-BIH. It records outcomes of the repository's existing
training pipeline, including unsuccessful minority-class behavior. These runs
do not establish a new recommended model or improve the published LR baseline
across all metrics.

The metrics, configs, record IDs, CPU benchmarks, confusion matrices, and
reliability diagrams are checked in under
`artifacts/se_resnet_aami5/` and `artifacts/cnn_transformer_aami5/`. No ECG
signals, checkpoints, or temperature-scaling weights are included. To run
inference, first reproduce a run to generate `resnet1d.pt` and
`temperature.pt` in its artifact directory. The common trainer calls both
output files `resnet1d.pt`; `model_config.json` identifies the actual
architecture.

## Dataset and protocol

- MIT-BIH Arrhythmia Database v1.0.0, MLII lead, 360 Hz; AAMI EC57
  `N/S/V/F/Q` labels and the repository's `--split ds1ds2` implementation.
- Seed 42; 18 training, 4 validation, and 22 test records. The three models
  below have identical record IDs. No record appears in more than one split.
- Test set: 49,698 beats (`N=44,246`, `S=1,837`, `V=3,220`, `F=388`, `Q=7`).
- Selection and early stopping use validation macro F1. Test metrics are
  computed only after model selection. Temperature scaling is fit on the
  validation set and applied to the neural-model probabilities.
- The LR row comes from the repository's released
  `artifacts/baseline_aami5/` files; it was **not** retrained in this run.

## Test results

| Metric | Released LR baseline | SE-ResNet-1D | CNN-Transformer |
| --- | ---: | ---: | ---: |
| Accuracy | 0.775 | 0.647 | 0.706 |
| Balanced accuracy | 0.384 | 0.363 | 0.389 |
| Macro F1 | 0.322 | 0.344 | 0.343 |
| ROC-AUC (OvR macro) | 0.668 | 0.749 | 0.673 |
| ECE | 0.098 | 0.175 | 0.058 |
| F1 N | 0.873 | 0.786 | 0.822 |
| F1 S | 0.105 | 0.148 | 0.235 |
| F1 V | 0.605 | 0.768 | 0.654 |
| F1 F | 0.028 | 0.015 | 0.002 |
| F1 Q | 0.000 | 0.000 | 0.000 |

SE-ResNet improves macro F1 and ROC-AUC over LR but reduces accuracy and
balanced accuracy. CNN-Transformer improves balanced accuracy, macro F1, and
S-class F1 but reduces accuracy. Neither is uniformly better than LR. The
test-set macro-F1 95% *beat bootstrap* intervals in the saved reports are
`[0.318, 0.326]` (LR), `[0.339, 0.348]` (SE-ResNet), and `[0.339, 0.347]`
(CNN-Transformer). These intervals resample beats, not patients, so they
should not be interpreted as patient-level uncertainty.

The SE-ResNet run predicted `F` for 14,724 test beats although only 388
test beats are truly `F`; the CNN-Transformer predicted `F` for 7,533.
Neither neural model correctly identified a `Q` beat. The training split
contains only 42 `F` and 6 `Q` beats after segmentation, while validation
contains 372 `F` and 2 `Q`. That imbalance is a plausible contributor to
the failure modes, but these runs do not isolate its causal effect. In
particular, do not use these checkpoints for clinical decisions or as
evidence of reliable F/Q detection.

The full per-class precision, recall, F1, bootstrap intervals, calibration
bins, and confusion matrices are in each `metrics_resnet.json`. The raw
metrics' `model` field is `resnet1d` for both neural runs because the
existing shared trainer writes that value; check `model_config.json` for
`architecture: cnn_transformer` in the Transformer run.

## Reproduce

Install the repository's `[dev,deep,bench]` extras in an isolated environment,
then fetch MIT-BIH using the existing downloader:

```bash
python scripts/fetch_mitbih.py --dest data/mitdb
```

The following commands are the **exact configurations** used here. They are
deliberately shorter than `make train-aami` (which specifies 60 epochs,
5 warmup epochs, and patience 10). Early stopping ended the SE-ResNet run
after 11 epochs and the CNN-Transformer run after 14 epochs.

```bash
python -m ecg_arrhythmia.train --model resnet \
  --data-dir data/mitdb --out-dir artifacts/se_resnet_aami5 \
  --scheme aami5 --split ds1ds2 --architecture resnet --use-se --augment \
  --focal-gamma 2.0 --grad-clip 1.0 --warmup-epochs 3 --ema-decay 0.999 \
  --epochs 20 --batch-size 256 --lr 0.001 --patience 5 --seed 42 \
  --device cpu --num-workers 0

python -m ecg_arrhythmia.train --model resnet \
  --data-dir data/mitdb --out-dir artifacts/cnn_transformer_aami5 \
  --scheme aami5 --split ds1ds2 --architecture cnn_transformer --use-se \
  --epochs 30 --batch-size 128 --lr 0.0005 --patience 8 --seed 42 \
  --device cpu --num-workers 0
```

Run environment: Windows, AMD Ryzen 7 7840HS CPU, Python 3.11.16,
PyTorch 2.14.0+cpu, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.7.2,
WFDB 4.3.1. Exact numbers and CPU latency may vary across environments.
The checked-in `benchmarks.json` files reflect this CPU, not a standardized
cross-machine performance comparison.

## Next questions

The F/Q training support and validation composition warrant a closer look
before promoting either neural model. Useful follow-ups would be a
record-level validation sensitivity study, a prespecified alternative loss
or sampling ablation, and evaluation on an external dataset. Those studies
would require new training runs and are not claimed by this report.
