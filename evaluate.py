"""
evaluate.py — CyberSentinel-LLM Evaluation
===========================================
Loads a trained checkpoint and evaluates on the test set,
computing all metrics reported in the paper (Table 3, Table 4).

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/cybersentinel_best.pt
    python evaluate.py --dataset BGL
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

from config import (
    CHECKPOINT, RESULTS, NUM_CLASSES, THREAT_CLASSES,
    ACTIVE_DATASET, TRAINING
)
from dataset import build_dataloaders
from model import build_model
from utils import (
    print_metrics_table, plot_confusion_matrix, plot_roc_curves,
    compute_metrics, set_seed
)

# ─────────────────────────────────────────────────────────────────────────────
# PAPER BASELINE RESULTS (Table 3) for comparison printing
# ─────────────────────────────────────────────────────────────────────────────
PAPER_TABLE_3 = {
    "HDFS": {
        "LogFiT [1]":              {"Acc": 93.2, "Prec": 93.5, "Rec": 92.1, "F1": 92.8},
        "DeepLog [7]":             {"Acc": 88.7, "Prec": 89.2, "Rec": 86.0, "F1": 87.5},
        "LogBERT [2]":             {"Acc": 90.1, "Prec": 90.8, "Rec": 89.5, "F1": 90.1},
        "LogEDL [5]":              {"Acc": 89.5, "Prec": 90.0, "Rec": 88.8, "F1": 89.5},
        "TLA-Net [4]":             {"Acc": 91.2, "Prec": 91.8, "Rec": 90.5, "F1": 91.1},
        "MADMM [13]":              {"Acc": 91.5, "Prec": 92.0, "Rec": 90.8, "F1": 91.4},
        "CNN-LogAD [6]":           {"Acc": 86.3, "Prec": 87.0, "Rec": 85.5, "F1": 86.3},
        "LogRobust [3]":           {"Acc": 85.8, "Prec": 86.5, "Rec": 85.0, "F1": 85.8},
        "ELFA-Log [9]":            {"Acc": 92.1, "Prec": 92.5, "Rec": 91.5, "F1": 92.0},
        "LogRESP [12]":            {"Acc": 91.8, "Prec": 92.2, "Rec": 91.3, "F1": 91.8},
        "CyberSentinel-LLM (Ours)":{"Acc": 96.8, "Prec": 97.1, "Rec": 95.9, "F1": 96.5},
    },
    "BGL": {
        "LogFiT [1]":              {"Acc": 91.8, "Prec": 92.0, "Rec": 91.5, "F1": 91.8},
        "DeepLog [7]":             {"Acc": 86.4, "Prec": 87.0, "Rec": 85.8, "F1": 86.4},
        "LogBERT [2]":             {"Acc": 88.5, "Prec": 89.0, "Rec": 88.0, "F1": 88.5},
        "LogEDL [5]":              {"Acc": 87.2, "Prec": 87.8, "Rec": 86.5, "F1": 87.2},
        "TLA-Net [4]":             {"Acc": 89.8, "Prec": 90.2, "Rec": 89.3, "F1": 89.8},
        "MADMM [13]":              {"Acc": 90.2, "Prec": 90.8, "Rec": 89.5, "F1": 90.2},
        "CNN-LogAD [6]":           {"Acc": 84.1, "Prec": 85.0, "Rec": 83.2, "F1": 84.1},
        "LogRobust [3]":           {"Acc": 83.5, "Prec": 84.2, "Rec": 82.8, "F1": 83.5},
        "ELFA-Log [9]":            {"Acc": 90.5, "Prec": 91.0, "Rec": 90.0, "F1": 90.5},
        "LogRESP [12]":            {"Acc": 89.8, "Prec": 90.5, "Rec": 89.2, "F1": 89.8},
        "CyberSentinel-LLM (Ours)":{"Acc": 95.5, "Prec": 95.8, "Rec": 94.5, "F1": 95.2},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Run full evaluation on a DataLoader.
    Returns predictions, true labels, and probability scores.
    """
    model.eval()
    all_preds, all_labels, all_scores, all_det_scores = [], [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=(device.type != "cpu")):
            out    = model(x)
            logits = out["logits"]
            det    = out["det_score"]

        probs = torch.softmax(logits, dim=-1)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_scores.append(probs.cpu().numpy())
        all_det_scores.append(det.cpu().numpy())

    preds      = np.concatenate(all_preds)
    labels     = np.concatenate(all_labels)
    scores     = np.concatenate(all_scores)
    det_scores = np.concatenate(all_det_scores)

    return preds, labels, scores, det_scores


def print_paper_comparison_table(model_metrics: dict, dataset_name: str):
    """Print results table matching paper Table 3 format."""
    ref = PAPER_TABLE_3.get(dataset_name, {})

    header  = f"\n{'='*72}"
    header += f"\n  Performance Comparison — {dataset_name} Dataset (Table 3)"
    header += f"\n{'='*72}"
    fmt     = "  {:<28} {:>7} {:>7} {:>7} {:>7}"
    print(header)
    print(fmt.format("Method", "Acc (%)", "Prec(%)", "Rec (%)", "F1  (%)"))
    print(f"  {'-'*68}")

    for method, vals in ref.items():
        marker = " ★" if "Ours" in method else ""
        print(fmt.format(
            method + marker,
            f"{vals['Acc']:.1f}", f"{vals['Prec']:.1f}",
            f"{vals['Rec']:.1f}",  f"{vals['F1']:.1f}",
        ))

    print(f"  {'-'*68}")
    print(fmt.format(
        "  → Our model (measured)",
        f"{model_metrics['accuracy']*100:.1f}",
        f"{model_metrics['precision_macro']*100:.1f}",
        f"{model_metrics['recall_macro']*100:.1f}",
        f"{model_metrics['f1_macro']*100:.1f}",
    ))
    print(f"{'='*72}\n")


def print_per_class_table(labels, preds, class_names):
    """Print per-class results matching Table 4."""
    print(f"\n{'='*65}")
    print("  Per-Class Detection Performance (Table 4 — HDFS Dataset)")
    print(f"{'='*65}")
    report = classification_report(labels, preds, target_names=class_names,
                                    output_dict=True)
    fmt = "  {:<20} {:>10} {:>8} {:>10} {:>10}"
    print(fmt.format("Threat Class", "Precision", "Recall", "F1-Score", "Support"))
    print(f"  {'-'*61}")
    for cls in class_names:
        if cls in report:
            r = report[cls]
            print(fmt.format(
                cls,
                f"{r['precision']*100:.1f}",
                f"{r['recall']*100:.1f}",
                f"{r['f1-score']*100:.1f}",
                str(int(r["support"])),
            ))
    print(f"  {'-'*61}")
    r = report["macro avg"]
    print(fmt.format(
        "Macro Average",
        f"{r['precision']*100:.1f}",
        f"{r['recall']*100:.1f}",
        f"{r['f1-score']*100:.1f}",
        str(int(report["weighted avg"]["support"])),
    ))
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    set_seed(TRAINING["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[evaluate] Device: {device}")

    # Data
    _, _, test_loader = build_dataloaders(
        dataset_name=args.dataset,
        use_synthetic=args.synthetic,
        batch_size=args.batch_size,
    )
    x_sample, _ = next(iter(test_loader))
    input_dim    = x_sample.shape[-1]

    # Model
    model = build_model(input_dim=input_dim, device=str(device))

    # Load checkpoint
    ckpt_path = args.checkpoint or str(CHECKPOINT["best_model"])
    if torch.serialization and not torch.cuda.is_available():
        pass
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[evaluate] Loaded checkpoint: {ckpt_path}")
    except FileNotFoundError:
        print(f"[evaluate] No checkpoint found at {ckpt_path}")
        print("[evaluate] Running with random weights for demonstration …")

    # Evaluate
    print(f"[evaluate] Evaluating on {args.dataset} test set …")
    preds, labels, scores, det_scores = evaluate_model(model, test_loader, device)

    # Metrics
    metrics = compute_metrics(labels, preds, scores, num_classes=NUM_CLASSES)

    # Print tables
    print_paper_comparison_table(metrics, args.dataset)
    print_per_class_table(labels, preds, THREAT_CLASSES)

    # Save confusion matrix
    cm_path = str(RESULTS["confusion_matrix"])
    plot_confusion_matrix(labels, preds, THREAT_CLASSES, save_path=cm_path)
    print(f"[evaluate] Confusion matrix saved → {cm_path}")

    # Save ROC curve
    roc_path = str(RESULTS["roc_curve"])
    plot_roc_curves(labels, scores, THREAT_CLASSES, save_path=roc_path)
    print(f"[evaluate] ROC curve saved → {roc_path}")

    # Save metrics JSON
    json_path = str(RESULTS["metrics_table"]).replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"[evaluate] Metrics saved → {json_path}")

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CyberSentinel-LLM")
    p.add_argument("--dataset",    type=str, default=ACTIVE_DATASET,
                   choices=["HDFS", "BGL"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--synthetic",  action="store_true", default=True)
    p.add_argument("--real_data",  dest="synthetic", action="store_false")
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
