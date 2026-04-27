"""Plotting helpers used by the training pipeline.

All functions here are gated on ``matplotlib`` being importable. If it isn't,
they raise an ``ImportError`` with a clear "install [bench] extra" message —
they are never silently no-ops.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # safe headless default for CI / training boxes
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plot helpers. "
            "Install with: pip install -e '.[bench]'"
        ) from exc


def confusion_matrix_plot(
    cm: np.ndarray | list[list[int]],
    labels: list[str],
    out_path: str | Path,
    *,
    title: str = "Confusion matrix (test)",
    normalize: bool = True,
) -> Path:
    """Save a confusion-matrix figure with per-row normalisation.

    Works for any class cardinality (3-class, 5-class, …). Cells are coloured by
    fraction; raw counts are annotated on top.
    """
    plt = _require_matplotlib()
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm

    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 2, 0.9 * len(labels) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v_norm = norm[i, j]
            count = int(cm[i, j])
            color = "white" if v_norm > 0.5 else "black"
            ax.text(
                j, i,
                f"{count}\n({v_norm:.2f})" if normalize else f"{count}",
                ha="center", va="center", color=color, fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def reliability_diagram(
    bin_centers: list[float],
    bin_confidences: list[float],
    bin_accuracies: list[float],
    bin_counts: list[int],
    out_path: str | Path,
    *,
    title: str = "Reliability diagram",
    ece: float | None = None,
) -> Path:
    """Plot a reliability diagram from ECE bin statistics."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(5, 5))
    centers = np.asarray(bin_centers)
    confs = np.asarray(bin_confidences)
    accs = np.asarray(bin_accuracies)
    counts = np.asarray(bin_counts)
    mask = counts > 0
    width = (centers[1] - centers[0]) if len(centers) > 1 else 0.05

    ax.bar(centers[mask], accs[mask], width=width * 0.9,
           color="#1f77b4", edgecolor="black", alpha=0.85, label="Accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    # Gap markers between accuracy and confidence per populated bin.
    for c, a, conf in zip(centers[mask], accs[mask], confs[mask], strict=False):
        ax.plot([c, c], [a, conf], "-", color="red", alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max softmax)")
    ax.set_ylabel("Accuracy")
    ax_title = title if ece is None else f"{title}  (ECE = {ece:.3f})"
    ax.set_title(ax_title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def tsne_embedding(
    embeddings: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    out_path: str | Path,
    *,
    title: str = "Penultimate-layer embedding (t-SNE)",
    perplexity: float = 30.0,
    seed: int = 0,
    max_points: int = 3000,
) -> Path:
    """Run t-SNE on penultimate embeddings and scatter-plot by class.

    ``max_points`` caps the embedding sample to keep this responsive on
    real MIT-BIH (~70k beats); we stratify the subsample so every class is
    represented.
    """
    plt = _require_matplotlib()
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(embeddings) > max_points:
        rng = np.random.default_rng(seed)
        per_class = max(max_points // max(len(labels), 1), 50)
        idxs = []
        for k in range(len(labels)):
            cls_idx = np.flatnonzero(y == k)
            if len(cls_idx) == 0:
                continue
            keep = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
            idxs.append(keep)
        idx = np.concatenate(idxs)
        embeddings = embeddings[idx]
        y = y[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed)
    z = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for k, lab in enumerate(labels):
        mask = y == k
        ax.scatter(z[mask, 0], z[mask, 1], s=10, alpha=0.6,
                   color=palette[k % len(palette)], label=lab)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9, markerscale=1.5)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


__all__ = ["confusion_matrix_plot", "reliability_diagram", "tsne_embedding"]
