"""
Visualization utilities for Ga2O3-Net training and results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(
    history: dict,
    save_path: str | None = None,
    title: str = "Training Curves",
):
    """
    Plot train and validation loss curves.

    Args:
        history: dict with 'train_loss' and 'val_loss' lists.
        save_path: If given, save figure to this path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_parity(
    pred: np.ndarray,
    target: np.ndarray,
    target_name: str = "Property",
    save_path: str | None = None,
):
    """
    Parity plot: predicted vs. actual.

    Args:
        pred: predicted values [N].
        target: true values [N].
        target_name: label for axes.
        save_path: optional save path.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target, pred, alpha=0.8, edgecolors="k", linewidths=0.5)
    mn = min(target.min(), pred.min())
    mx = max(target.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="y=x")
    ax.set_xlabel(f"True {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"Parity Plot — {target_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: list[str],
    save_path: str | None = None,
    title: str = "Stage-2 Embedding (PCA)",
):
    """
    2-D PCA scatter of multi-modal embeddings, colored by dopant element.

    Args:
        embeddings: [N, D] array.
        labels: list of N string labels (dopant element).
        save_path: optional save path.
        title: figure title.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys, label=lbl, color=color_map[lbl],
                   s=80, edgecolors="k", linewidths=0.5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_cv_bar(
    cv_summary: dict,
    target_col: str,
    metric: str = "mae",
    save_path: str | None = None,
):
    """
    Bar chart of cross-validation mean ± std for one target.

    Args:
        cv_summary: dict from FinetuneTrainer.run_cv().
        target_col: target property name.
        metric: 'mae', 'rmse', or 'r2'.
        save_path: optional save path.
    """
    mean_key = f"{target_col}_{metric}_mean"
    std_key = f"{target_col}_{metric}_std"
    if mean_key not in cv_summary:
        print(f"Key not found: {mean_key}")
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([target_col], [cv_summary[mean_key]],
           yerr=[cv_summary[std_key]], capsize=5, color="steelblue")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"CV {metric.upper()} — {target_col}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
