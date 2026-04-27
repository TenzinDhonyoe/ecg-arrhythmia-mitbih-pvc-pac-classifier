"""ECG beat classification package (MIT-BIH N / PVC / PAC).

Public API:
- :class:`ECGClassifier` — high-level wrapper that loads either backend and
  predicts on a numpy signal or CSV path.
- :class:`BeatPrediction`, :class:`PredictionResult` — typed result objects.
- :mod:`ecg_arrhythmia.streaming` — online beat-by-beat classification.
- :mod:`ecg_arrhythmia.export` — ONNX export utilities for edge deployment.
"""

from .api import BeatPrediction, ECGClassifier, PredictionResult

__all__ = [
    "ECGClassifier",
    "BeatPrediction",
    "PredictionResult",
    "labels",
    "preprocessing",
    "training",
    "inference",
]

__version__ = "0.4.0"
