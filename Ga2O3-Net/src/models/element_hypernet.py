"""Phase 46 V11 — Element-descriptor hypernetwork (HN-GNN).

A small MLP that maps a per-element 5-dim chemistry descriptor
[EN, ionic_radius, carrier_type, Δv, Δperiod] to a scalar log_VO offset,
producing an element-specific correction that interpolates continuously in
chemistry space — so unseen rare-dopant elements (Ge/Bi/Sb at N=1) inherit
physics from labeled neighbors along the descriptor manifold rather than
collapsing to a class-wide bias (the V10 failure mode).

Replaces V10's `class_offset` (per-valence-class scalar Embedding(4, 1)) with
a continuous element-descriptor → offset mapping. The Δv channel in the input
descriptor (added in Phase 46) is what gives this module its super-donor
signal: at random init Bi/Sb/Ta sit close in descriptor space, far from
acceptors — see scripts/sanity_check_hypernet_init.py.

Usage:
    H = ElementHyperNet(desc_dim=5, hidden=16, mid=8)
    descriptors  # [B, K, 5]  per-row, per-cation descriptor lookups
    weights      # [B, K]      normalized fractions (Σ_k w=1) — NOT raw conc
    offset = H(descriptors, weights)  # [B, 1]
    log_VO = ... + offset

Design notes:
  * Final layer zero-initialized → V11 starts identical to V5+aux at epoch 0.
    Only learns deviations from there.
  * Weighted sum uses *normalized fractions* (Σ_k w=1), not raw concentrations,
    so H stays an element-identity correction rather than a hidden c-response.
    The Brouwer head's `dopant_term` already owns concentration response.
  * Trainer applies weight_decay=1e-3 to this module's `net` parameters
    specifically (param-group split) — prevents H from absorbing α_method's
    contribution under element/method correlation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ElementHyperNet(nn.Module):
    """N-dim element descriptor → scalar log_VO offset hypernetwork."""

    def __init__(
        self,
        desc_dim: int = 5,
        hidden: int = 16,
        mid: int = 8,
        final_init: str = "zeros",
        out_dim: int = 1,
    ):
        super().__init__()
        self.desc_dim = int(desc_dim)
        self.final_init = str(final_init)
        # Phase 53 V53-γ1: out_dim=2 produces (a_e, b_e) for the conc-dependent
        # dopant_term in BrouwerHeadVC; default 1 is V52b's scalar log_VO offset.
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.desc_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, mid),
            nn.SiLU(),
            nn.Linear(mid, self.out_dim),
        )
        # Phase 46 V11 used "zeros" (identity at epoch 0). V11 collapsed: hidden
        # weights crushed to ~0 before final layer could route gradient. Phase
        # 47 V13 uses "small_random" (small Gaussian init, std=0.01) so all
        # layers receive useful gradient from the first epoch.
        if self.final_init == "zeros":
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif self.final_init == "small_random":
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(
                f"ElementHyperNet final_init must be 'zeros' or 'small_random', "
                f"got {self.final_init!r}"
            )

    def forward(
        self,
        descriptors: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        descriptors: [B, K, desc_dim] — per-row, per-cation descriptors
        weights:     [B, K]           — *normalized* fractions, Σ_k weights[b,k] = 1
        returns:     [B, out_dim]     — weighted-sum offset for log_VO (out_dim=1)
                                        or (a_e, b_e) tuple (out_dim=2 for V53-γ1)
        """
        if descriptors.dim() != 3 or descriptors.shape[-1] != self.desc_dim:
            raise ValueError(
                f"ElementHyperNet expects descriptors [B, K, {self.desc_dim}], "
                f"got {tuple(descriptors.shape)}"
            )
        if weights.dim() != 2 or weights.shape != descriptors.shape[:2]:
            raise ValueError(
                f"ElementHyperNet expects weights [B, K] matching descriptors[:2], "
                f"got weights={tuple(weights.shape)}, descriptors={tuple(descriptors.shape)}"
            )
        per_elem_offset = self.net(descriptors)              # [B, K, out_dim]
        weighted = (weights.unsqueeze(-1) * per_elem_offset).sum(dim=1)  # [B, out_dim]
        return weighted
