"""Loss functions and mixup utilities for imbalanced ECG classification.

The MIT-BIH AAMI 5-class distribution is dramatically imbalanced (~70% N,
~3% V, ~1% S, <0.5% F/Q in the inter-patient split). This module provides:

- :class:`FocalLoss` — focal CE with optional class weights and label
  smoothing. Implemented from scratch so it composes correctly with mixup
  (PyTorch's built-in ``CrossEntropyLoss`` does not accept the ``(y_a, y_b,
  lam)`` decomposition we want).
- :func:`mixup_batch` — apply mixup to a batch of (morph, rr, y) tensors and
  return both label streams plus the mixing coefficient. The caller computes
  the loss as ``lam * loss(logits, y_a) + (1 - lam) * loss(logits, y_b)``.
- :func:`make_balanced_sampler` — build a :class:`WeightedRandomSampler` that
  yields a roughly class-balanced stream of indices each epoch. Empirically
  the strongest single lever for AAMI imbalance.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def make_balanced_sampler(y: np.ndarray, n_classes: int, seed: int = 0):
    """Create a torch ``WeightedRandomSampler`` with inverse-frequency weights.

    Each sample's weight is ``1 / count(class[sample])``. Drawing
    ``len(y)`` samples without replacement is too restrictive; we draw
    ``len(y)`` *with* replacement so each epoch is roughly class-balanced
    on expectation.
    """
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover
        raise ImportError("make_balanced_sampler requires PyTorch") from exc

    y = np.asarray(y)
    counts = np.array([int((y == k).sum()) for k in range(n_classes)], dtype=float)
    # Avoid div-by-zero on classes with no samples (the sampler simply never
    # picks that class because its weight is 0).
    safe_counts = np.where(counts > 0, counts, 1.0)
    inv = np.where(counts > 0, 1.0 / safe_counts, 0.0)
    weights = inv[y]
    g = torch.Generator()
    g.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(y),
        replacement=True,
        generator=g,
    )


class FocalLoss:
    """Focal cross-entropy loss with optional class weights and label smoothing.

    .. math::

        \\mathrm{FL}(p_t) = -\\alpha_t \\, (1 - p_t)^\\gamma \\, \\log(p_t)

    With ``gamma=0`` and no label smoothing, this is exactly weighted CE.
    Label smoothing is applied in the standard way: the one-hot target is
    softened to ``(1 - eps) * one_hot + eps / n_classes``.

    Implemented as a plain callable (not ``nn.Module``) because it has no
    parameters and we never need to ship it across devices.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight=None,
        label_smoothing: float = 0.0,
    ) -> None:
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.gamma = float(gamma)
        self.weight = weight  # torch.Tensor of shape (n_classes,) or None
        self.label_smoothing = float(label_smoothing)

    def __call__(self, logits, target_ids):  # type: ignore[no-untyped-def]
        try:
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover
            raise ImportError("FocalLoss requires PyTorch") from exc
        # Soft targets: support both int target_ids and pre-mixed soft targets.
        if target_ids.dim() == 1:
            n_classes = logits.shape[1]
            soft = F.one_hot(target_ids, num_classes=n_classes).to(logits.dtype)
            if self.label_smoothing > 0:
                eps = self.label_smoothing
                soft = (1.0 - eps) * soft + eps / n_classes
        else:
            soft = target_ids.to(logits.dtype)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        # Element-wise focal weight; broadcast over classes.
        focal_weight = (1.0 - probs).clamp(min=0.0, max=1.0) ** self.gamma
        loss = -(focal_weight * log_probs * soft)

        if self.weight is not None:
            # Weight per true class id (or per soft mass if soft target).
            w = self.weight.to(logits.device, dtype=logits.dtype)
            loss = loss * w.unsqueeze(0)

        # Sum over classes, mean over batch (matches torch's default reduction).
        return loss.sum(dim=1).mean()


def mixup_batch(
    morph,
    rr,
    y,
    *,
    alpha: float = 0.2,
    rng: np.random.Generator | None = None,
):  # type: ignore[no-untyped-def]
    """Apply mixup to a batch.

    Returns
    -------
    morph_mixed, rr_mixed, y_a, y_b, lam
        ``morph_mixed = lam * morph + (1 - lam) * morph[perm]`` (same for rr).
        ``y_a == y`` and ``y_b == y[perm]``. The caller computes the loss as
        ``lam * loss(logits, y_a) + (1 - lam) * loss(logits, y_b)``.

    Notes
    -----
    Mixup of one-hot targets does not compose with ``nn.CrossEntropyLoss(weight=...)``
    because the soft targets break the per-sample class-weight lookup. Use
    :class:`FocalLoss` (which we wrote ourselves) instead.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mixup_batch requires PyTorch") from exc
    if alpha <= 0:
        return morph, rr, y, y, torch.tensor(1.0, device=morph.device)
    if rng is None:
        rng = np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    n = morph.shape[0]
    perm = torch.randperm(n, device=morph.device)
    morph_mixed = lam * morph + (1.0 - lam) * morph[perm]
    rr_mixed = lam * rr + (1.0 - lam) * rr[perm]
    return morph_mixed, rr_mixed, y, y[perm], torch.tensor(lam, device=morph.device)


__all__ = ["FocalLoss", "mixup_batch", "make_balanced_sampler"]


# Quiet unused-import linter
_ = Iterable
