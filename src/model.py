"""Phase 4 — M1 temporal model (PLAN.md §6).

Small GRU over per-frame feature sequences, < 100k parameters, CPU-trainable
in minutes per fold. Undetected frames arrive zeroed with a boolean mask; the
mask is appended as an extra input channel so the model can tell "eyes closed"
from "no face".
"""
from __future__ import annotations

import torch
from torch import nn


class DrowsinessGRU(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_features + 1, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden_size, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) float32, mask: (B, T) bool → logits (B, n_classes)."""
        inp = torch.cat([x, mask.float().unsqueeze(-1)], dim=-1)
        out, _ = self.gru(inp)
        return self.head(out[:, -1])

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_model(n_features: int, n_classes: int, cfg: dict) -> DrowsinessGRU:
    m = cfg["model"]
    if m["kind"] != "gru":
        raise NotImplementedError(f"model kind {m['kind']!r} (TCN is a planned ablation)")
    model = DrowsinessGRU(n_features, n_classes, m["hidden_size"],
                          m["num_layers"], m["dropout"])
    n = model.n_parameters()
    assert n < 100_000, f"model has {n} params — PLAN.md §6 caps at 100k"
    return model


class FeatureScaler:
    """Per-feature standardization fitted on TRAIN windows' valid steps only."""

    def fit(self, X, mask):
        import numpy as np

        valid = X[mask]                      # (n_valid_steps, F)
        self.mean_ = valid.mean(axis=0)
        self.std_ = np.clip(valid.std(axis=0), 1e-6, None)
        return self

    def transform(self, X, mask):
        out = (X - self.mean_) / self.std_
        out[~mask] = 0.0
        return out.astype("float32")
