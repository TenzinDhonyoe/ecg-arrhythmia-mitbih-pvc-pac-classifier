"""Post-hoc calibration and Monte Carlo dropout uncertainty.

Two helpers, both designed to plug into existing :class:`ECGClassifier`
artifacts without retraining:

- :class:`TemperatureScaler` — fits a single scalar ``T`` on a held-out
  validation set by minimising NLL with LBFGS, then applies
  ``softmax(logits / T)`` at inference. Preserves argmax accuracy and
  dramatically reduces ECE on a single CNN; standard practice for AAMI
  classifiers (Guo et al., "On Calibration of Modern Neural Networks," 2017).

- :class:`MCDropoutEnsemble` — toggles dropout layers to ``train`` mode at
  inference, runs ``n_passes`` forward passes, and returns ``(mean_probs,
  epistemic_entropy)``. Lets downstream systems abstain on uncertain beats.

Plus :func:`expected_calibration_error` for reporting and the reliability bins
needed to draw a calibration diagram.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute ECE and per-bin reliability statistics.

    Parameters
    ----------
    probs : (N, C) array
        Per-class probabilities.
    y_true : (N,) integer array of ground-truth labels.
    n_bins : int
        Number of equal-width bins on confidence in ``[0, 1]``.

    Returns
    -------
    dict with keys: ``ece``, ``bin_centers``, ``bin_confidences``,
    ``bin_accuracies``, ``bin_counts``. The diagram is
    ``bin_accuracies`` vs. ``bin_confidences``; ECE is the count-weighted
    average gap between them.
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    accuracies = np.zeros(n_bins, dtype=float)
    confs = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n = int(mask.sum())
        counts[i] = n
        if n > 0:
            accuracies[i] = float(correct[mask].mean())
            confs[i] = float(confidences[mask].mean())

    total = max(int(counts.sum()), 1)
    ece = float(np.sum(counts / total * np.abs(accuracies - confs)))
    return {
        "ece": ece,
        "bin_centers": centers.tolist(),
        "bin_confidences": confs.tolist(),
        "bin_accuracies": accuracies.tolist(),
        "bin_counts": counts.tolist(),
    }


class TemperatureScaler:
    """Fit and apply a single scalar logit-temperature.

    Usage::

        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_y)
        calibrated_probs = scaler.predict_proba(test_logits)
        scaler.save(Path("artifacts/.../temperature.pt"))
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        self.temperature: float = float(init_temperature)
        self._fitted: bool = False

    def fit(self, logits: np.ndarray, y_true: np.ndarray, *, max_iter: int = 200) -> float:
        """Fit ``T`` on a held-out logits/labels split by minimising NLL."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover
            raise ImportError("TemperatureScaler.fit requires PyTorch") from exc
        logits_t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y_true, dtype=np.int64))
        log_T = torch.zeros(1, requires_grad=True)  # parametrise log(T) so T > 0

        optimizer = torch.optim.LBFGS(
            [log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            T = log_T.exp()
            scaled = logits_t / T
            loss = F.cross_entropy(scaled, y_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_T.exp().detach().item())
        self._fitted = True
        return self.temperature

    def apply_logits(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for a logits matrix."""
        scaled = np.asarray(logits, dtype=float) / float(self.temperature)
        # Numerically stable softmax.
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return self.apply_logits(logits)

    # ------------------------------------------------------------------
    # Persistence — uses torch.save so the tensor and our scalar stay
    # in one file alongside the rest of the artifacts.
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("TemperatureScaler.save requires PyTorch") from exc
        torch.save({"temperature": float(self.temperature)}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("TemperatureScaler.load requires PyTorch") from exc
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        out = cls(init_temperature=float(state["temperature"]))
        out._fitted = True
        return out


class MCDropoutEnsemble:
    """Monte Carlo dropout uncertainty wrapper for a torch model.

    Toggles ``Dropout`` modules to train mode at inference (BatchNorm stays in
    eval mode so running statistics aren't perturbed), runs ``n_passes``
    forward passes, and returns the mean of the per-pass softmax distributions
    plus the predictive entropy (``-Σ p log p``) as an epistemic-uncertainty
    proxy.
    """

    def __init__(self, model, n_passes: int = 20) -> None:
        if n_passes < 2:
            raise ValueError(f"n_passes must be >= 2, got {n_passes}")
        self.model = model
        self.n_passes = int(n_passes)

    @staticmethod
    def _enable_dropout(model) -> None:
        try:
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("MCDropoutEnsemble requires PyTorch") from exc
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                m.train()

    def predict_proba(self, morph, rr) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean_probs (N,C), entropy (N,))`` for a (morph, rr) batch."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("MCDropoutEnsemble requires PyTorch") from exc

        self.model.eval()
        self._enable_dropout(self.model)
        with torch.no_grad():
            samples = []
            for _ in range(self.n_passes):
                logits = self.model(morph, rr)
                samples.append(torch.softmax(logits, dim=1).cpu().numpy())
        # Restore deterministic eval mode before we hand control back.
        self.model.eval()
        stacked = np.stack(samples, axis=0)             # (T, N, C)
        mean = stacked.mean(axis=0)                     # (N, C)
        # Predictive entropy of the mean distribution.
        eps = 1e-12
        entropy = -np.sum(mean * np.log(mean + eps), axis=1)
        return mean, entropy


__all__ = [
    "TemperatureScaler",
    "MCDropoutEnsemble",
    "expected_calibration_error",
]
