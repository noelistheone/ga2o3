"""Phase 54 V54-B1 — Charge-state V_O expert loader + module.

Defines `ChargeStateVOHead` (3-head MLP on a 64-d crystal embedding) and the
`load_charge_vo_expert` helper that reconstructs it from the checkpoint
saved by scripts/02g_pretrain_charge_vo.py. Caller is responsible for
freezing.

The Ga2O3Net constructor wraps it via `aux_distill_charge_vo_head` and
adds a soft-distill MSE on the V5 fused embedding → expert prediction.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ChargeStateVOHead(nn.Module):
    """3-head trunk: shared 2-layer MLP → (E_f^0, E_f^+1, E_f^+2).

    Matches the architecture from scripts/02g_pretrain_charge_vo.py.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        h = self.trunk(emb)
        return torch.cat([head(h) for head in self.heads], dim=-1)


def load_charge_vo_expert(ckpt_path: str | Path) -> nn.Module:
    """Load encoder_charge_vo.pt. Returns the frozen 3-head trunk.

    Robust to ambiguous `in_dim` in older checkpoints: re-derives both
    in_dim and hidden_dim from the actual saved trunk.0.weight shape.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["head_state_dict"]
    # trunk.0.weight is the first Linear's [out=hidden_dim, in=in_dim]
    if "trunk.0.weight" in sd:
        hidden_dim = int(sd["trunk.0.weight"].shape[0])
        in_dim = int(sd["trunk.0.weight"].shape[1])
    else:
        # Fallback to checkpoint metadata
        in_dim = int(state.get("in_dim", 64))
        hidden_dim = 128
    head = ChargeStateVOHead(in_dim=in_dim, hidden_dim=hidden_dim, dropout=0.0)
    head.load_state_dict(sd, strict=False)
    head.eval()
    return head
