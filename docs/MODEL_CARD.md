# Model Card (Baseline LR)

## Summary

Baseline logistic regression classifier for ECG beat classes `N`, `V`, and `a` trained on MIT-BIH.

## Inputs

- 180-sample beat window centered on detected R-peak
- 2 RR-ratio timing features (`RR_pre/median`, `RR_post/median`)

## Output

Class probabilities for:
- `N` (normal)
- `V` (PVC)
- `a` (PAC, merged MIT-BIH `A` + `a`)

## Training Protocol

- Record-level split (no record overlap between train/val/test)
- Feature scaler fit on train only
- Class-weighted logistic regression

## Intended Use

Research, method benchmarking, and education.

## Not Intended For

Clinical diagnosis, treatment decisions, emergency triage, or unsupervised medical deployment.

## Limitations

- Domain shift from MIT-BIH to wearable or custom hardware signals
- Lead configuration and polarity mismatch sensitivity
- Label simplification (PAC merged classes)
