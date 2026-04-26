# Security and Safety

## Reporting a vulnerability

If you discover a security issue (e.g. a way the inference pipeline can be
made to load arbitrary code, a vulnerable dependency, or a path-traversal
in the CSV reader) please email **tendhon3015@gmail.com** with a
description and reproduction steps. Do not open a public issue for
security reports.

We will acknowledge within 5 business days.

## Medical-safety disclaimer

This repository is research and educational software. **It is not a
medical device.** It is not approved by the FDA, EMA, MHRA, or any other
regulatory body. Do not use it for diagnosis, treatment, triage, or
clinical decision-making. The model is trained on a limited public
dataset (MIT-BIH MLII), which does not represent the morphology, lead
configuration, or signal quality of consumer wearables, ambulatory
monitors, or 12-lead clinical ECGs.

If you build something on top of this code that is intended to influence
medical care, you are responsible for the appropriate clinical
validation, regulatory clearance, and user-facing safety guarantees.

## Reporting clinical-safety concerns

If a downstream user reports that the model has produced a clinically
dangerous output (e.g. confidently calling PVCs as Normal in a way that
could mislead a downstream alerting system) please open a public issue
with the label `safety`. Include the input signal in fully de-identified
form. We treat safety reports as higher priority than feature requests.
