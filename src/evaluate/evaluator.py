"""
src/evaluate/evaluator.py
  • Full classification reports for all three tasks
  • Confusion matrices
  • Ablation: BiLSTM context ON vs OFF
  • Error analysis: where does stance classifier fail?
  • Confidence threshold sweep
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from typing import List, Dict

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)


# ── full evaluation ───────────────────────────────────────────────────────────
def evaluate_all_tasks(model, loader, device, stance_classes: List[str]) -> Dict:
    model.eval()
    cl_preds, cl_trues = [], []
    pr_preds, pr_trues = [], []
    st_preds, st_trues = [], []
    st_probs_all = []

    with torch.no_grad():
        for ctx, cl_lbl, pr_lbl, st_lbl in loader:
            ctx = ctx.to(device)
            cl_log, pr_log, st_log = model(ctx)

            cl_preds.extend(cl_log.argmax(-1).cpu().numpy())
            cl_trues.extend(cl_lbl.numpy())
            pr_preds.extend(pr_log.argmax(-1).cpu().numpy())
            pr_trues.extend(pr_lbl.numpy())
            st_preds.extend(st_log.argmax(-1).cpu().numpy())
            st_trues.extend(st_lbl.numpy())
            st_probs_all.append(F.softmax(st_log, dim=-1).cpu().numpy())

    st_probs_all = np.concatenate(st_probs_all, axis=0)

    results = {}
    for name, preds, trues, classes in [
        ("Claim",   cl_preds, cl_trues, ["Non-Claim", "Claim"]),
        ("Premise", pr_preds, pr_trues, ["Non-Premise", "Premise"]),
        ("Stance",  st_preds, st_trues, stance_classes),
    ]:
        print(f"\n── {name} Classification Report ──")
        print(classification_report(trues, preds, target_names=classes, digits=4))
        results[name.lower()] = {
            "f1":        f1_score(trues, preds, average="macro", zero_division=0),
            "precision": precision_score(trues, preds, average="macro", zero_division=0),
            "recall":    recall_score(trues, preds, average="macro", zero_division=0),
            "preds":     np.array(preds),
            "trues":     np.array(trues),
        }

    results["stance"]["probs"] = st_probs_all
    return results


# ── confusion matrices ────────────────────────────────────────────────────────
def plot_confusion_matrices(results: Dict, stance_classes: List[str], save: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    configs = [
        ("claim",   ["Non-Claim", "Claim"],   "Claim Detection"),
        ("premise", ["Non-Premise","Premise"], "Premise Detection"),
        ("stance",  stance_classes,            "Stance Classification"),
    ]
    for ax, (key, labels, title) in zip(axes, configs):
        cm = confusion_matrix(results[key]["trues"], results[key]["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS / "confusion_matrices.png", dpi=150)
        print("[eval] Saved → results/confusion_matrices.png")
    plt.show()


# ── error analysis ────────────────────────────────────────────────────────────
def stance_error_analysis(results: Dict, sentences: List[str],
                           stance_classes: List[str], save: bool = True):
    """Find sentences where stance is wrong and confidence is high (hard errors)."""
    preds  = results["stance"]["preds"]
    trues  = results["stance"]["trues"]
    probs  = results["stance"]["probs"]
    confs  = probs.max(axis=1)

    errors_idx = np.where(preds != trues)[0]
    correct_idx = np.where(preds == trues)[0]

    print(f"\n[eval] Stance errors: {len(errors_idx)} / {len(preds)} "
          f"({100*len(errors_idx)/len(preds):.1f}%)")

    # high-confidence errors
    hc_errors = errors_idx[confs[errors_idx] > 0.8]
    print(f"[eval] High-confidence (>0.8) errors: {len(hc_errors)}")

    # error distribution per true class
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ec = pd.Series([stance_classes[t] for t in trues[errors_idx]]).value_counts()
    axes[0].bar(ec.index, ec.values, color="#C44E52")
    axes[0].set_title("Errors by True Stance")
    axes[0].set_xlabel("True Label")
    axes[0].set_ylabel("Error Count")

    # confidence distribution: correct vs wrong
    axes[1].hist(confs[correct_idx], bins=30, alpha=0.6, label="Correct", color="#4C72B0")
    axes[1].hist(confs[errors_idx],  bins=30, alpha=0.6, label="Wrong",   color="#C44E52")
    axes[1].set_title("Confidence: Correct vs Wrong")
    axes[1].set_xlabel("Max Softmax Confidence")
    axes[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS / "stance_error_analysis.png", dpi=150)
        print("[eval] Saved → results/stance_error_analysis.png")
    plt.show()


# ── confidence threshold sweep ────────────────────────────────────────────────
def threshold_sweep(results: Dict, stance_classes: List[str], save: bool = True):
    probs = results["stance"]["probs"]
    trues = results["stance"]["trues"]
    thresholds = np.linspace(0.3, 0.95, 30)
    coverages, f1s = [], []

    for t in thresholds:
        confident_mask = probs.max(axis=1) >= t
        coverage = confident_mask.mean()
        if confident_mask.sum() == 0:
            f1s.append(0)
        else:
            f1 = f1_score(trues[confident_mask],
                          probs[confident_mask].argmax(axis=1),
                          average="macro", zero_division=0)
            f1s.append(f1)
        coverages.append(coverage)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(thresholds, f1s, "b-o", ms=4, label="Macro F1")
    ax2.plot(thresholds, coverages, "r--s", ms=4, label="Coverage")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Macro F1", color="blue")
    ax2.set_ylabel("Coverage", color="red")
    ax1.set_title("Confidence Threshold vs F1 / Coverage")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS / "threshold_sweep.png", dpi=150)
        print("[eval] Saved → results/threshold_sweep.png")
    plt.show()
    return thresholds, f1s, coverages


# ── ablation table ────────────────────────────────────────────────────────────
def print_ablation_table(ablation_results: Dict):
    """
    ablation_results = {
      "BiLSTM context ON":  {"claim_f1": ..., "premise_f1": ..., "stance_f1": ...},
      "BiLSTM context OFF": {...},
    }
    """
    print("\n── Ablation Study ─────────────────────────────────────────────")
    rows = []
    for variant, metrics in ablation_results.items():
        rows.append({
            "Variant":    variant,
            "Claim F1":   f"{metrics['claim_f1']:.4f}",
            "Premise F1": f"{metrics['premise_f1']:.4f}",
            "Stance F1":  f"{metrics['stance_f1']:.4f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "ablation_table.csv", index=False)
    print("[eval] Saved → results/ablation_table.csv\n")
