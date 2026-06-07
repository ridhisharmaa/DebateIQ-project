"""
src/models/model.py
Architecture: SBERT context window → BiLSTM → three task heads
  • Task 1 – Claim Detection      (binary)
  • Task 2 – Premise Detection    (binary)
  • Task 3 – Stance Classification (PRO / CON / NONE)

The BiLSTM receives a sequence of SBERT embeddings (context window)
and returns the centre-token hidden state for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM that encodes a window of SBERT embeddings.
    Returns the hidden state at the centre position (the target sentence).
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_dim)
        returns : (batch, hidden_dim * 2)  — centre token representation
        """
        out, _ = self.bilstm(x)                   # (B, seq, H*2)
        centre_idx = x.size(1) // 2
        centre = out[:, centre_idx, :]             # (B, H*2)
        return self.layer_norm(centre)


class TaskHead(nn.Module):
    """Shared two-layer MLP classifier head."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DebateIQModel(nn.Module):
    """
    Multi-task argument mining model.

    Forward input  : context_emb (B, seq_len, sbert_dim)
    Forward outputs: claim_logits, premise_logits, stance_logits
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        sbert_dim   = m["sbert_dim"]
        hidden      = m["bilstm_hidden"]
        layers      = m["bilstm_layers"]
        dropout     = m["bilstm_dropout"]
        fc_hidden   = m["fc_hidden"]

        self.encoder = BiLSTMEncoder(sbert_dim, hidden, layers, dropout)
        enc_out_dim  = hidden * 2  # bidirectional

        self.claim_head   = TaskHead(enc_out_dim, fc_hidden, m["num_claim_labels"],   dropout)
        self.premise_head = TaskHead(enc_out_dim, fc_hidden, m["num_premise_labels"], dropout)
        self.stance_head  = TaskHead(enc_out_dim, fc_hidden, m["num_stance_labels"],  dropout)

    def forward(self, context_emb: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(context_emb)
        return (self.claim_head(enc),
                self.premise_head(enc),
                self.stance_head(enc))

    def predict(self, context_emb: torch.Tensor,
                stance_threshold: float = 0.5
                ) -> dict:
        """Convenience method for inference — returns dicts with labels + confidences."""
        self.eval()
        with torch.no_grad():
            cl_logits, pr_logits, st_logits = self(context_emb)
            cl_probs = F.softmax(cl_logits, dim=-1)
            pr_probs = F.softmax(pr_logits, dim=-1)
            st_probs = F.softmax(st_logits, dim=-1)
        return {
            "claim_label":    cl_logits.argmax(-1).cpu().numpy(),
            "claim_conf":     cl_probs.max(-1).values.cpu().numpy(),
            "premise_label":  pr_logits.argmax(-1).cpu().numpy(),
            "premise_conf":   pr_probs.max(-1).values.cpu().numpy(),
            "stance_label":   st_logits.argmax(-1).cpu().numpy(),
            "stance_conf":    st_probs.max(-1).values.cpu().numpy(),
            "stance_probs":   st_probs.cpu().numpy(),
        }


# ── multi-task loss ───────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Weighted sum of three cross-entropy losses.
    Weights are learnable (uncertainty weighting — Kendall et al., 2018).
    """

    def __init__(self, n_tasks: int = 3):
        super().__init__()
        # log(sigma^2) initialised to 0 → sigma=1
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total


def build_model(cfg: dict) -> DebateIQModel:
    model = DebateIQModel(cfg)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DebateIQ — trainable params: {total:,}")
    return model
