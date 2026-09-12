"""Phase 54 V54-B1 — V_Ga charge-state expert loader.

Defines `VGaEncoderHead` (3-head MLP on standardized element descriptors)
and `load_vga_expert`. Matches scripts/02h_pretrain_vga.py.

The expert outputs [B, 3] = (E_f^-1, E_f^-2, E_f^-3) — V_Ga formation
energy for each negative charge state. Distill aux loss applies MSE between
fused embedding → linear-head → 3-vec and this expert's prediction
(after target standardization to match scale).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class VGaEncoderHead(nn.Module):
    """3-head MLP: standardized 11-d element descriptors → 3 charge-state E_f.

    Matches scripts/02h_pretrain_vga.py architecture.
    """

    def __init__(self, in_dim: int = 11, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])

    def forward(self, elem_emb: torch.Tensor) -> torch.Tensor:
        h = self.trunk(elem_emb)
        return torch.cat([head(h) for head in self.heads], dim=-1)


def load_vga_expert(ckpt_path: str | Path) -> nn.Module:
    """Load encoder_vga.pt. Returns the frozen 3-head expert."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_dim = int(state.get("table_dim", 11))
    hidden_dim = int(state.get("hidden_dim", 64))
    head = VGaEncoderHead(in_dim=in_dim, hidden_dim=hidden_dim, dropout=0.0)
    head.load_state_dict(state["head_state_dict"], strict=False)
    head.eval()
    # Attach target stats as buffers so trainer can de-standardize if needed
    target_mean = torch.tensor(state.get("target_mean", [0., 0., 0., 0.]),
                                 dtype=torch.float32)
    target_std = torch.tensor(state.get("target_std", [1., 1., 1., 1.]),
                                dtype=torch.float32)
    head.register_buffer("target_mean", target_mean)
    head.register_buffer("target_std", target_std)
    return head
