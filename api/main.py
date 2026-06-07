"""
api/main.py
FastAPI backend for DebateIQ.
POST /analyze  — accepts raw text, returns structured JSON with
                 claims, premises, stances, and argument tree.
GET  /health   — liveness probe
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── lazy imports (avoid crashing if torch not installed in test env) ──────────
_model       = None
_onnx_sess   = None
_sbert_enc   = None
_le_classes  = None
_cfg         = None


def _load_pipeline():
    global _model, _onnx_sess, _sbert_enc, _le_classes, _cfg
    import yaml
    from sentence_transformers import SentenceTransformer
    from src.data.dataset import build_context_sequences
    from src.models.export_onnx import load_onnx_session

    with open("configs/config.yaml") as f:
        _cfg = yaml.safe_load(f)

    _sbert_enc = SentenceTransformer(_cfg["model"]["sbert_model"])

    onnx_path = Path(_cfg["paths"]["onnx_dir"]) / "debateiq.onnx"
    model_pt  = Path(_cfg["paths"]["model_dir"]) / "best_model.pt"

    if onnx_path.exists():
        _onnx_sess = load_onnx_session(str(onnx_path))
        print("[api] ONNX session loaded.")
    elif model_pt.exists():
        from src.models.model import build_model
        ckpt = torch.load(str(model_pt),map_location="cpu",weights_only=False)
        _model = build_model(ckpt["config"])
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()
        _le_classes = ckpt.get("label_encoder_classes", ["CON", "NONE", "PRO"])
        print("[api] PyTorch model loaded.")
    else:
        print("[api] WARNING: No model found. Returning dummy predictions.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_pipeline()
    yield


app = FastAPI(
    title="DebateIQ API",
    description="Argument mining: claim detection, premise detection, stance classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str
    topic: Optional[str] = None
    confidence_threshold: float = 0.5


class SentenceResult(BaseModel):
    index: int
    sentence: str
    is_claim: bool
    is_premise: bool
    stance: str
    claim_confidence: float
    premise_confidence: float
    stance_confidence: float
    stance_probs: dict


class ArgumentNode(BaseModel):
    id: int
    text: str
    node_type: str
    stance: str
    confidence: float


class ArgumentEdge(BaseModel):
    source: int
    target: int
    stance: str
    color: str


class AnalyzeResponse(BaseModel):
    sentences:       List[SentenceResult]
    argument_tree:   dict
    summary:         dict
    processing_ms:   float


# ── helpers ───────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def _encode_and_predict(sentences: List[str], threshold: float) -> dict:
    from src.data.dataset import build_context_sequences
    from src.models.export_onnx import onnx_predict
    import scipy.special

    emb = _sbert_enc.encode(sentences, show_progress_bar=False,
                             normalize_embeddings=True)
    window = _cfg["data"]["context_window"] if _cfg else 3
    ctx = build_context_sequences(emb, window=window).astype(np.float32)

    if _onnx_sess is not None:
        preds = onnx_predict(_onnx_sess, ctx)
    elif _model is not None:
        with torch.no_grad():
            t = torch.from_numpy(ctx)
            cl_l, pr_l, st_l = _model(t)
        preds = {
            "claim_label":   cl_l.argmax(-1).numpy(),
            "claim_conf":    scipy.special.softmax(cl_l.numpy(), axis=-1).max(-1),
            "premise_label": pr_l.argmax(-1).numpy(),
            "premise_conf":  scipy.special.softmax(pr_l.numpy(), axis=-1).max(-1),
            "stance_label":  st_l.argmax(-1).numpy(),
            "stance_conf":   scipy.special.softmax(st_l.numpy(), axis=-1).max(-1),
            "stance_probs":  scipy.special.softmax(st_l.numpy(), axis=-1),
        }
    else:
        # dummy
        n = len(sentences)
        preds = {
            "claim_label":   np.zeros(n, dtype=int),
            "claim_conf":    np.ones(n) * 0.5,
            "premise_label": np.zeros(n, dtype=int),
            "premise_conf":  np.ones(n) * 0.5,
            "stance_label":  np.ones(n, dtype=int),
            "stance_conf":   np.ones(n) * 0.5,
            "stance_probs":  np.ones((n, 3)) / 3,
        }
    return preds


def _get_stance_classes():
    if _le_classes is not None:
        return list(_le_classes)
    return ["CON", "NONE", "PRO"]


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (_onnx_sess or _model) is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    sentences = _split_sentences(req.text)
    if not sentences:
        raise HTTPException(400, "No sentences found in text.")
    if len(sentences) > 50:
        raise HTTPException(400, "Max 50 sentences per request.")

    preds = _encode_and_predict(sentences, req.confidence_threshold)
    stance_classes = _get_stance_classes()

    results = []
    for i, sent in enumerate(sentences):
        sl = int(preds["stance_label"][i])
        sp = preds["stance_probs"][i]
        results.append(SentenceResult(
            index=i,
            sentence=sent,
            is_claim=bool(preds["claim_label"][i] == 1),
            is_premise=bool(preds["premise_label"][i] == 1),
            stance=stance_classes[sl] if sl < len(stance_classes) else "NONE",
            claim_confidence=round(float(preds["claim_conf"][i]), 4),
            premise_confidence=round(float(preds["premise_conf"][i]), 4),
            stance_confidence=round(float(preds["stance_conf"][i]), 4),
            stance_probs={c: round(float(sp[j]), 4)
                          for j, c in enumerate(stance_classes)},
        ))

    # argument tree
    from src.evaluate.argument_tree import (build_argument_graph, graph_to_json,
                                             STANCE_COLOR)
    G = build_argument_graph(
        sentences,
        preds["claim_label"], preds["premise_label"], preds["stance_label"],
        stance_classes, preds["stance_conf"],
    )
    tree_json = graph_to_json(G)

    summary = {
        "total_sentences": len(sentences),
        "claims":          int(preds["claim_label"].sum()),
        "premises":        int(preds["premise_label"].sum()),
        "stance_counts": {
            c: int((preds["stance_label"] == j).sum())
            for j, c in enumerate(stance_classes)
        },
        "topic": req.topic,
    }

    return AnalyzeResponse(
        sentences=results,
        argument_tree=tree_json,
        summary=summary,
        processing_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
