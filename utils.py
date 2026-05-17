"""
utils.py — CyberSentinel-LLM Utilities
=======================================
Helper functions for training, evaluation, metrics, and visualization.
"""

import os
import random
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility — Section 4.1.4"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# AVERAGE METER
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ─────────────────────────────────────────────────────────────────────────────
# METRICS — as reported in paper Section 4.2
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    scores: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute all metrics used in the paper:
    Accuracy, Precision, Recall, F1 (macro), AUC-ROC
    """
    # One-vs-rest binarization for AUC
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    metrics = {
        "accuracy":          accuracy_score(labels, preds),
        "f1_macro":          f1_score(labels, preds, average="macro",  zero_division=0),
        "f1_weighted":       f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro":   precision_score(labels, preds, average="macro",   zero_division=0),
        "recall_macro":      recall_score(labels, preds, average="macro",    zero_division=0),
    }

    # AUC-ROC (macro OvR)
    try:
        metrics["auc_roc_macro"] = roc_auc_score(
            labels_bin, scores, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["auc_roc_macro"] = 0.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, is_best: bool,
                    path_best: str, path_last: str):
    torch.save(state, path_last)
    if is_best:
        shutil.copy(path_last, path_best)


def load_checkpoint(model: torch.nn.Module, path: str) -> Tuple[int, float]:
    ckpt  = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    return ckpt.get("epoch", 0) + 1, ckpt.get("best_f1", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT — Figure 5 in paper
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds:  np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix — CyberSentinel-LLM",
):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n       = len(class_names)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    im      = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                        vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=7, color=color)

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(title,              fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ROC CURVE PLOT — Figure 6 in paper
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    labels:      np.ndarray,
    scores:      np.ndarray,
    class_names: List[str],
    save_path:   Optional[str] = None,
    title:       str = "ROC Curves — CyberSentinel-LLM",
):
    n_cls      = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_cls)))
    colors     = ["#1A73E8", "#E53935", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

    auc_all = []
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        try:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], scores[:, i])
            auc_val     = roc_auc_score(labels_bin[:, i], scores[:, i])
            auc_all.append(auc_val)
            ax.plot(fpr, tpr, color=color, lw=1.8,
                    label=f"{cls} (AUC={auc_val:.3f})")
            ax.fill_between(fpr, tpr, alpha=0.06, color=color)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(title,                  fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: dict, title: str = "Evaluation Results"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics.get('accuracy', 0) * 100:.2f}%")
    print(f"  F1 (macro): {metrics.get('f1_macro', 0) * 100:.2f}%")
    print(f"  Precision : {metrics.get('precision_macro', 0) * 100:.2f}%")
    print(f"  Recall    : {metrics.get('recall_macro', 0) * 100:.2f}%")
    print(f"  AUC-ROC   : {metrics.get('auc_roc_macro', 0) * 100:.2f}%")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVE PLOT — Figure 3 in paper
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_accs:     List[float],
    save_path:    Optional[str] = None,
):
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")

    ax1.plot(epochs, train_losses, "#1A73E8", lw=1.8, marker="o", ms=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(epochs, val_accs, "#E53935", lw=1.8, marker="s", ms=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE DRL REWARD — Section 3.5, Eq.(24)
# ─────────────────────────────────────────────────────────────────────────────

def compute_drl_reward(
    is_tp: bool, response_time_s: float, is_fp: bool,
    alpha1: float = 1.0, alpha2: float = 0.8, alpha3: float = 0.3
) -> float:
    """
    R(s_t, a_t) = α1 R_detect + α2 R_response − α3 C_penalty  — Eq.(24)
    """
    R_detect   = 1.0 if is_tp else -0.5
    if response_time_s <= 5.0:
        R_response = 0.8
    elif response_time_s >= 60.0:
        R_response = 0.0
    else:
        R_response = 0.8 * (1.0 - (response_time_s - 5.0) / 55.0)
    C_penalty  = 0.3 if is_fp else 0.0

    return alpha1 * R_detect + alpha2 * R_response - alpha3 * C_penalty
