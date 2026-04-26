# Model comparison — LR baseline vs ResNet-1D

Both models trained on the same MIT-BIH MLII record-level split (seed 42).

| Test metric                  | LR baseline  | ResNet-1D    | Δ      |
|------------------------------|--------------|--------------|--------|
| Balanced accuracy            | 0.673        | **0.749**    | +0.076 |
| Macro F1                     | 0.440        | **0.586**    | +0.146 |
| ROC-AUC (one-vs-rest, macro) | 0.820        | **0.950**    | +0.130 |
| F1 (N)                       | 0.744        | **0.948**    | +0.204 |
| F1 (V — PVC)                 | 0.552        | **0.655**    | +0.103 |
| F1 (a — PAC)                 | 0.025        | **0.156**    | +0.131 |
| Parameters                   | ~550         | 551 975      |        |
| Model size                   | 5.2 KB       | 2.2 MB       |        |
| Inference (CPU, p50)         | **0.04 ms**  | 0.66 ms      | +16×   |
| Inference (CPU, p95)         | **0.04 ms**  | 0.74 ms      | +18×   |

## Per-class test report

### LR baseline

| Class    | Support | Precision | Recall | F1    |
|----------|---------|-----------|--------|-------|
| N        | 15 152  | 0.981     | 0.599  | 0.744 |
| V (PVC)  |  1 104  | 0.406     | 0.861  | 0.552 |
| a (PAC)  |    111  | 0.013     | 0.559  | 0.025 |

### ResNet-1D

| Class    | Support | Precision | Recall | F1    |
|----------|---------|-----------|--------|-------|
| N        | 15 152  | 0.991     | 0.910  | 0.948 |
| V (PVC)  |  1 104  | 0.513     | 0.905  | 0.655 |
| a (PAC)  |    111  | 0.095     | 0.432  | 0.156 |

## What changed

- **Normal beats**: ResNet recall jumps from 0.60 to 0.91. The LR baseline
  with class-weighted training over-predicts the rare classes (especially
  PAC) at the cost of N recall; the ResNet's learned features separate
  PAC-like noise from N more cleanly.
- **PVC**: F1 +0.10. Both models pick up most PVCs (high recall ~0.86–0.90),
  but the ResNet's precision more than doubles.
- **PAC**: still poor on both. Only 111 PAC beats in the test set, the
  morphology is a small perturbation on N, and the LR + RR-features pair
  doesn't carry enough signal. ResNet F1 rises 6× but is still only 0.156.
  This is the honest weak point of the project: do not trust either model
  for PAC isolation in production.

## Choosing between them

| Use case                                             | Pick     |
|------------------------------------------------------|----------|
| Embedded / battery / sub-millisecond budget          | LR       |
| Best general accuracy, can spend 1 ms/beat on CPU    | ResNet-1D |
| Highest recall on rare PVCs in offline post-hoc QA   | ResNet-1D |
| You need PAC detection that you can rely on          | **Neither** — both are below the bar; this needs more PAC-rich training data and an AAMI-aware label scheme |
