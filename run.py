"""
run.py
Master script — run any stage of the DebateIQ pipeline.

Usage::
  python run.py eda          # exploratory data analysis
  python run.py train        # train the model
  python run.py evaluate     # evaluate on test set + plots
  python run.py ablation     # ablation: context ON vs OFF
  python run.py export       # export to ONNX
  python run.py api          # start FastAPI server
  python run.py streamlit    # start Streamlit dashboard
  python run.py all          # eda → train → evaluate → export
"""

import sys
import yaml
import torch
from pathlib import Path


def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


# ── EDA ───────────────────────────────────────────────────────────────────────
def run_eda():
    print("\n════ EDA ════════════════════════════════════════════════════")
    from src.data.dataset import load_ibm_dataset, SBERTEncoder
    from src.data.eda import run_full_eda
    cfg = load_cfg()
    df = load_ibm_dataset(cfg)
    encoder = SBERTEncoder(cfg["model"]["sbert_model"])
    emb = encoder.encode(df["sentence"].tolist()[:500])   # subsample for speed
    run_full_eda(df, emb)


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def run_train():
    print("\n════ TRAINING ═══════════════════════════════════════════════")
    from src.train.trainer import train
    model, history, test_metrics = train()
    return model, test_metrics


# ── EVALUATE ──────────────────────────────────────────────────────────────────
def run_evaluate():
    print("\n════ EVALUATION ══════════════════════════════════════════════")
    cfg    = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.data.dataset import build_dataloaders
    from src.models.model import build_model
    from src.evaluate.evaluator import (evaluate_all_tasks, plot_confusion_matrices,
                                         stance_error_analysis, threshold_sweep)

    _, _, test_loader, le = build_dataloaders(cfg)
    ckpt = torch.load(
    Path(cfg["paths"]["model_dir"]) / "best_model.pt",
    map_location="cpu",
    weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    stance_classes = list(le.classes_)
    results = evaluate_all_tasks(model, test_loader, device, stance_classes)
    plot_confusion_matrices(results, stance_classes)
    threshold_sweep(results, stance_classes)
    return results


# ── ABLATION ──────────────────────────────────────────────────────────────────
def run_ablation():
    print("\n════ ABLATION ════════════════════════════════════════════════")
    cfg = load_cfg()
    results_dict = {}

    for context_on in [True, False]:
        label = "BiLSTM context ON" if context_on else "BiLSTM context OFF"
        print(f"\n── Variant: {label} ──")
        cfg["data"]["context_window"] = 3 if context_on else 0

        from src.train.trainer import train as _train
        _, _, te_metrics = _train.__wrapped__(cfg) if hasattr(_train, "__wrapped__") else _train()
        results_dict[label] = {
            "claim_f1":   te_metrics.get("claim_f1", 0),
            "premise_f1": te_metrics.get("premise_f1", 0),
            "stance_f1":  te_metrics.get("stance_f1", 0),
        }

    from src.evaluate.evaluator import print_ablation_table
    print_ablation_table(results_dict)


# ── EXPORT ONNX ───────────────────────────────────────────────────────────────
def run_export():
    print("\n════ ONNX EXPORT ═════════════════════════════════════════════")
    cfg    = load_cfg()
    device = torch.device("cpu")  # ONNX export on CPU

    from src.models.model import build_model
    from src.models.export_onnx import export_to_onnx

    ckpt  = torch.load(Path(cfg["paths"]["model_dir"]) / "best_model.pt",
                       map_location=device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    export_to_onnx(model, cfg)


# ── API ───────────────────────────────────────────────────────────────────────
def run_api():
    import uvicorn
    print("\n════ FastAPI server on http://localhost:8000 ═════════════════")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


# ── STREAMLIT ─────────────────────────────────────────────────────────────────
def run_streamlit():
    import subprocess, sys
    print("\n════ Streamlit dashboard on http://localhost:8501 ═══════════")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "streamlit_app/app.py"])


# ── dispatch ──────────────────────────────────────────────────────────────────
COMMANDS = {
    "eda":       run_eda,
    "train":     run_train,
    "evaluate":  run_evaluate,
    "ablation":  run_ablation,
    "export":    run_export,
    "api":       run_api,
    "streamlit": run_streamlit,
}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "all":
        run_eda()
        run_train()
        run_evaluate()
        run_export()
        print("\n✅ Full pipeline complete!")
        return

    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
