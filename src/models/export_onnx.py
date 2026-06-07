"""
src/models/export_onnx.py
Export trained DebateIQ model to ONNX for fast CPU inference (<3s).
"""

import torch
import numpy as np
from pathlib import Path


def export_to_onnx(model, cfg: dict, output_path: str = None):
    model.eval()
    m = cfg["model"]
    seq_len  = 2 * cfg["data"]["context_window"] + 1
    sbert_dim = m["sbert_dim"]

    if output_path is None:
        onnx_dir = Path(cfg["paths"]["onnx_dir"])
        onnx_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(onnx_dir / "debateiq.onnx")

    dummy_input = torch.zeros(1, seq_len, sbert_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["context_emb"],
        output_names=["claim_logits", "premise_logits", "stance_logits"],
        dynamic_axes={
            "context_emb":     {0: "batch_size"},
            "claim_logits":    {0: "batch_size"},
            "premise_logits":  {0: "batch_size"},
            "stance_logits":   {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[onnx] Model exported → {output_path}")
    return output_path


def load_onnx_session(onnx_path: str):
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, sess_options,
                                   providers=["CPUExecutionProvider"])
    print(f"[onnx] Session loaded from {onnx_path}")
    return session


def onnx_predict(session, context_emb: np.ndarray) -> dict:
    """
    context_emb : (batch, seq_len, sbert_dim) float32
    """
    import scipy.special
    cl_log, pr_log, st_log = session.run(
        None, {"context_emb": context_emb.astype(np.float32)}
    )
    cl_probs = scipy.special.softmax(cl_log, axis=-1)
    pr_probs = scipy.special.softmax(pr_log, axis=-1)
    st_probs = scipy.special.softmax(st_log, axis=-1)
    return {
        "claim_label":   cl_probs.argmax(-1),
        "claim_conf":    cl_probs.max(-1),
        "premise_label": pr_probs.argmax(-1),
        "premise_conf":  pr_probs.max(-1),
        "stance_label":  st_probs.argmax(-1),
        "stance_conf":   st_probs.max(-1),
        "stance_probs":  st_probs,
    }
