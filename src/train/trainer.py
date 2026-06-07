"""
src/train/trainer.py
Full training loop with:
  • Multi-task loss (uncertainty weighting)
  • W&B logging
  • Early stopping
  • Checkpoint saving
"""

import os
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import f1_score, classification_report

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.models.model import build_model, MultiTaskLoss
from src.data.dataset import build_dataloaders, load_config


# ── helpers ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[train] Device: {dev}")
    return dev


def compute_metrics(preds: np.ndarray, labels: np.ndarray,
                    avg: str = "macro") -> Dict[str, float]:
    return {
        "f1":       f1_score(labels, preds, average=avg, zero_division=0),
        "accuracy": (preds == labels).mean(),
    }


# ── one epoch ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scheduler,
              device, train: bool = True) -> Dict[str, float]:
    model.train(train)
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_cl_pred, all_cl_true = [], []
    all_pr_pred, all_pr_true = [], []
    all_st_pred, all_st_true = [], []

    for ctx, cl_lbl, pr_lbl, st_lbl in loader:
        ctx    = ctx.to(device)
        cl_lbl = cl_lbl.to(device)
        pr_lbl = pr_lbl.to(device)
        st_lbl = st_lbl.to(device)

        cl_log, pr_log, st_log = model(ctx)

        loss_cl = ce(cl_log, cl_lbl)
        loss_pr = ce(pr_log, pr_lbl)
        loss_st = ce(st_log, st_lbl)
        loss    = criterion((loss_cl, loss_pr, loss_st))

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

        total_loss += loss.item()
        all_cl_pred.extend(cl_log.argmax(-1).cpu().numpy())
        all_cl_true.extend(cl_lbl.cpu().numpy())
        all_pr_pred.extend(pr_log.argmax(-1).cpu().numpy())
        all_pr_true.extend(pr_lbl.cpu().numpy())
        all_st_pred.extend(st_log.argmax(-1).cpu().numpy())
        all_st_true.extend(st_lbl.cpu().numpy())

    n = len(loader)
    return {
        "loss":          total_loss / n,
        "claim_f1":      compute_metrics(np.array(all_cl_pred), np.array(all_cl_true))["f1"],
        "premise_f1":    compute_metrics(np.array(all_pr_pred), np.array(all_pr_true))["f1"],
        "stance_f1":     compute_metrics(np.array(all_st_pred), np.array(all_st_true))["f1"],
        "claim_acc":     compute_metrics(np.array(all_cl_pred), np.array(all_cl_true))["accuracy"],
        "premise_acc":   compute_metrics(np.array(all_pr_pred), np.array(all_pr_true))["accuracy"],
        "stance_acc":    compute_metrics(np.array(all_st_pred), np.array(all_st_true))["accuracy"],
    }


# ── main trainer ──────────────────────────────────────────────────────────────
def train(config_path: str = "configs/config.yaml"):
    cfg    = load_config(config_path)
    device = get_device()

    train_loader, val_loader, test_loader, le = build_dataloaders(cfg)

    model     = build_model(cfg).to(device)
    criterion = MultiTaskLoss().to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["training"]["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["training"]["epochs"],
    )

    # W&B init
    use_wandb = WANDB_AVAILABLE and cfg["wandb"].get("entity") is not None
    if use_wandb:
        wandb.init(project=cfg["wandb"]["project"], config=cfg)

    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1   = 0.0
    patience_cnt  = 0
    patience      = cfg["training"]["early_stopping_patience"]
    history       = []

    print(f"\n[train] Starting training for {cfg['training']['epochs']} epochs...\n")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        tr_metrics = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        va_metrics = run_epoch(model, val_loader,   criterion, None,      None,      device, train=False)

        elapsed = time.time() - t0
        avg_val_f1 = (va_metrics["claim_f1"] + va_metrics["premise_f1"] + va_metrics["stance_f1"]) / 3

        log = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in tr_metrics.items()},
            **{f"val/{k}":   v for k, v in va_metrics.items()},
            "val/avg_f1": avg_val_f1,
        }
        history.append(log)
        if use_wandb:
            wandb.log(log)

        print(f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
              f"loss={tr_metrics['loss']:.4f} | "
              f"val_stance_f1={va_metrics['stance_f1']:.4f} | "
              f"val_avg_f1={avg_val_f1:.4f} | "
              f"{elapsed:.1f}s")

        # checkpoint
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "criterion_state": criterion.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "label_encoder_classes": le.classes_,
                "config": cfg,
            }, model_dir / "best_model.pt")
            print(f"  ✓ Saved best model (val_avg_f1={best_val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    print(f"\n[train] Best val avg F1: {best_val_f1:.4f}")

    # ── final test evaluation ──────────────────────────────────────────────
    
    ckpt = torch.load(
    model_dir / "best_model.pt",
    map_location=device,
    weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_metrics = run_epoch(model, test_loader, criterion, None, None, device, train=False)
    print("\n── Test Set Results ───────────────────────────────────────────")
    for k, v in te_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("──────────────────────────────────────────────────────────────\n")

    if use_wandb:
        wandb.log({f"test/{k}": v for k, v in te_metrics.items()})
        wandb.finish()

    return model, history, te_metrics


if __name__ == "__main__":
    train()
