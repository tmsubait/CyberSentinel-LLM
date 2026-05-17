"""
train.py — CyberSentinel-LLM Training Script
=============================================
Full training pipeline matching paper Section 4.1.5 (Training Configuration).

Usage:
    python train.py
    python train.py --dataset BGL --epochs 50 --batch_size 32
"""

import argparse
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import (
    TRAINING, TRANSFORMER, LORA, LOSS, DATASET, ACTIVE_DATASET,
    CHECKPOINT, LOG_DIR, NUM_CLASSES
)
from dataset import build_dataloaders
from model import build_model, CyberSentinelLoss
from utils import (
    AverageMeter, compute_metrics, save_checkpoint, load_checkpoint,
    print_metrics_table, set_seed
)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER & SCHEDULER — Section 4.1.5
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    AdamW with β1=0.9, β2=0.999, weight_decay=0.01 — Section 4.1.5
    """
    # Separate LoRA params (higher LR) from other params
    lora_params  = [p for n, p in model.named_parameters()
                    if "lora_" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if "lora_" not in n and p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": lora_params,  "lr": TRAINING["peak_lr"]},
            {"params": other_params, "lr": TRAINING["peak_lr"] * 0.5},
        ],
        betas=TRAINING["betas"],
        weight_decay=TRAINING["weight_decay"],
    )


def build_scheduler(optimizer: torch.optim.Optimizer,
                    total_steps: int = TRAINING["total_steps"],
                    warmup_steps: int = TRAINING["warmup_steps"]):
    """
    Cosine annealing with linear warmup — Section 4.1.5.
    peak_lr=5e-5, warmup_steps=500, total_steps=10,000
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler, scaler,
    criterion, device, epoch, writer, global_step
):
    model.train()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    t0         = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # Mixed precision forward pass — Section 4.1.5
        with autocast(enabled=(device != "cpu")):
            out     = model(x)
            logits  = out["logits"]
            h_LLM   = out["h_LLM"]
            total_loss, loss_parts = criterion(logits, h_LLM, y)
            # Scale by grad accumulation
            total_loss = total_loss / TRAINING["grad_accum_steps"]

        scaler.scale(total_loss).backward()

        # Gradient accumulation — Section 4.1.5
        if (batch_idx + 1) % TRAINING["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        acc  = (logits.argmax(-1) == y).float().mean().item()
        loss_meter.update(loss_parts["total"], x.size(0))
        acc_meter.update(acc, x.size(0))

        if writer and global_step % 50 == 0:
            writer.add_scalar("train/loss",    loss_parts["total"], global_step)
            writer.add_scalar("train/ce_loss", loss_parts["ce"],    global_step)
            writer.add_scalar("train/cont_loss", loss_parts["contrastive"], global_step)
            writer.add_scalar("train/acc",     acc,                 global_step)
            writer.add_scalar("train/lr",
                              optimizer.param_groups[0]["lr"],      global_step)

        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d} [{batch_idx:4d}/{len(loader):4d}]"
                f"  Loss: {loss_meter.avg:.4f}"
                f"  Acc: {acc_meter.avg*100:.2f}%"
                f"  LR: {optimizer.param_groups[0]['lr']:.2e}"
                f"  [{elapsed:.1f}s]"
            )

    return loss_meter.avg, acc_meter.avg, global_step


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, global_step):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels, all_scores = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=(device != "cpu")):
            out        = model(x)
            logits     = out["logits"]
            h_LLM      = out["h_LLM"]
            total_loss, _ = criterion(logits, h_LLM, y)

        loss_meter.update(total_loss.item(), x.size(0))
        probs = torch.softmax(logits, dim=-1)
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(y.cpu())
        all_scores.append(probs.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    scores = torch.cat(all_scores).numpy()

    metrics = compute_metrics(labels, preds, scores, num_classes=NUM_CLASSES)

    if writer:
        writer.add_scalar("val/loss",     loss_meter.avg,       global_step)
        writer.add_scalar("val/accuracy", metrics["accuracy"],   global_step)
        writer.add_scalar("val/f1",       metrics["f1_macro"],   global_step)
        writer.add_scalar("val/auc_roc",  metrics.get("auc_roc_macro", 0.0), global_step)

    return loss_meter.avg, metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # Reproducibility — Section 4.1.4
    set_seed(TRAINING["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[train] Device: {device}")

    # Data
    print("[train] Loading data …")
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=args.dataset,
        use_synthetic=args.synthetic,
        batch_size=args.batch_size,
    )

    # Infer input dim from first batch
    x_sample, _ = next(iter(train_loader))
    input_dim    = x_sample.shape[-1]

    # Model
    model     = build_model(input_dim=input_dim, device=str(device))
    criterion = CyberSentinelLoss(
        temperature=LOSS["temperature"],
        lam=LOSS["lambda_contrastive"],
        num_classes=NUM_CLASSES,
    )

    # Resume from checkpoint if requested
    start_epoch  = 1
    global_step  = 0
    best_f1      = 0.0

    if args.resume and CHECKPOINT["best_model"].exists():
        start_epoch, best_f1 = load_checkpoint(model, str(CHECKPOINT["best_model"]))
        print(f"[train] Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"[train] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer  = build_optimizer(model)
    scheduler  = build_scheduler(
        optimizer,
        total_steps=args.epochs * len(train_loader) // TRAINING["grad_accum_steps"],
        warmup_steps=TRAINING["warmup_steps"],
    )
    scaler     = GradScaler(enabled=(device.type != "cpu"))
    writer     = SummaryWriter(log_dir=str(LOG_DIR / "tensorboard")) if not args.no_tb else None

    print(f"[train] Starting training for {args.epochs} epochs …")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()

        # Training
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, epoch, writer, global_step
        )

        # Validation
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer, global_step
        )

        # Checkpoint
        val_f1 = val_metrics["f1_macro"]
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1

        _model = model.module if isinstance(model, nn.DataParallel) else model
        save_checkpoint({
            "epoch":     epoch,
            "model":     _model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_f1":   best_f1,
        }, is_best=is_best,
           path_best=str(CHECKPOINT["best_model"]),
           path_last=str(CHECKPOINT["last_model"]))

        elapsed = time.time() - t_epoch
        print(
            f"\n[Epoch {epoch:3d}/{args.epochs}]"
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%"
            f"  |  Val Loss: {val_loss:.4f}"
            f"  Acc: {val_metrics['accuracy']*100:.2f}%"
            f"  F1: {val_metrics['f1_macro']*100:.2f}%"
            f"  AUC: {val_metrics.get('auc_roc_macro',0.)*100:.2f}%"
            f"  {'★ BEST' if is_best else ''}"
            f"  [{elapsed:.1f}s]"
        )
        print("-" * 70)

    print(f"\n[train] Training complete. Best Val F1: {best_f1*100:.2f}%")

    # Final evaluation on test set
    print("\n[train] Evaluating on test set …")
    _, test_metrics = validate(model, test_loader, criterion, device,
                               args.epochs + 1, None, global_step)
    print_metrics_table(test_metrics, title="Final Test Results")

    if writer:
        writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CyberSentinel-LLM")
    p.add_argument("--dataset",    type=str,  default=ACTIVE_DATASET,
                   choices=["HDFS", "BGL"],
                   help="Dataset to train on (default: HDFS)")
    p.add_argument("--epochs",     type=int,  default=TRAINING["num_epochs"])
    p.add_argument("--batch_size", type=int,  default=TRAINING["batch_size"])
    p.add_argument("--lr",         type=float, default=TRAINING["peak_lr"])
    p.add_argument("--synthetic",  action="store_true", default=True,
                   help="Use synthetic data (for quick testing)")
    p.add_argument("--real_data",  dest="synthetic", action="store_false",
                   help="Use real downloaded data")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--cpu",        action="store_true",
                   help="Force CPU even if GPU available")
    p.add_argument("--no_tb",      action="store_true",
                   help="Disable TensorBoard logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
