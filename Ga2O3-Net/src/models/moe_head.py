"""
Mixture-of-Experts (MoE) prediction head for Ga2O3-Net (Phase 11).

Replaces the single shared PredictionHead with K soft-routed expert heads.
A small gating network reads chemistry-only features (CompositionStream output
or DopantStream output, depending on what's available) and produces softmax
weights over the K experts. Final prediction = Σ wₖ · expertₖ(fused).

Design rationale:
  * Routing is data-driven (no element-name hardcoding) — generalises to
    unseen dopants whose chemistry features fall in known regions.
  * Soft (not top-k) routing — every expert gets gradient on every sample,
    avoiding expert collapse on small training sets.
  * Gate input uses chemistry-only features (NOT the full fused vector),
    so process / structure context doesn't pollute routing decisions.
  * Load balancing via entropy regularisation in CombinedLoss
    (see src/training/losses.py).

Diagnostic: every forward pass caches `self.last_gate_weights` [B, K] —
the trainer / evaluator can read this to:
  - compute load-balancing loss
  - export per-element routing signature heatmap
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MoEHead(nn.Module):
    """
    Soft Mixture-of-Experts head.

    Args:
        in_dim:        Fused embedding dim (typically 160 or 176).
        gate_in_dim:   Chemistry-feature dim for gate input
                       (e.g. 64 for CompositionStream, 16 for DopantStream).
        hidden_dims:   Hidden layer sizes for each expert MLP.
        out_dim:       Output dim per expert (usually 1 for regression).
        num_experts:   K — number of expert heads.
        dropout:       Dropout rate inside experts.
        gate_hidden:   Hidden size of the gating MLP (small by design).
    """

    def __init__(
        self,
        in_dim: int,
        gate_in_dim: int,
        hidden_dims: list[int] = [64, 32],
        out_dim: int = 1,
        num_experts: int = 4,
        dropout: float = 0.3,
        gate_hidden: int = 32,
        routing: str = "soft",       # "soft" | "top1"  (12N)
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.routing = routing

        # K independent expert MLPs
        self.experts = nn.ModuleList([
            self._make_expert(in_dim, hidden_dims, out_dim, dropout)
            for _ in range(num_experts)
        ])

        # Gating network — small, sees only chemistry features
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )

        # Cache for diagnostics + load-balancing loss
        self.last_gate_weights: torch.Tensor | None = None

    @staticmethod
    def _make_expert(
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        """One expert = standard PredictionHead (matches src/models/ga2o3_net.py)."""
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        fused: torch.Tensor,           # [B, in_dim]
        gate_features: torch.Tensor,   # [B, gate_in_dim]
    ) -> torch.Tensor:
        """
        Returns weighted sum of K expert predictions.

        Side-effect: caches gate weights at self.last_gate_weights for the
        load-balancing loss and downstream diagnostics.
        """
        gate_logits = self.gate(gate_features)                  # [B, K]

        if self.routing == "top1":
            # 12N: hard top-1 routing with straight-through estimator (STE)
            #   forward = argmax (one-hot), backward = softmax gradient
            soft = torch.softmax(gate_logits, dim=-1)           # [B, K]
            top_idx = soft.argmax(dim=-1, keepdim=True)         # [B, 1]
            hard = torch.zeros_like(soft).scatter_(1, top_idx, 1.0)
            gate_weights = hard - soft.detach() + soft          # STE
        else:
            gate_weights = torch.softmax(gate_logits, dim=-1)   # [B, K]
        self.last_gate_weights = gate_weights                   # cache

        # Compute all K expert outputs
        # Stack: [B, K, out_dim]
        expert_outs = torch.stack(
            [exp(fused) for exp in self.experts], dim=1
        )

        # Weighted sum across experts
        # [B, K, 1] × [B, K, out_dim] → [B, out_dim]
        out = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out

    def get_gate_weights(self) -> torch.Tensor | None:
        """Returns cached gate weights from the last forward pass, or None."""
        return self.last_gate_weights


def load_balancing_loss(
    gate_weights: torch.Tensor,
    target_uniformity: bool = True,
) -> torch.Tensor:
    """
    Entropy-based load-balancing loss for MoE.

    Penalises expert collapse (one expert getting all the routing weight).
    The minimum (= 0) is reached when batch-average gate usage is uniform
    across experts.

    Args:
        gate_weights: [B, K] softmax weights from MoEHead.
        target_uniformity: If True, returns max_entropy − actual_entropy
                           (smaller is better, ≥ 0).

    Returns:
        Scalar tensor.
    """
    if gate_weights is None or gate_weights.numel() == 0:
        return torch.zeros((), device=gate_weights.device if gate_weights is not None else "cpu")

    # Average gate usage per expert across the batch
    avg_w = gate_weights.mean(dim=0)             # [K]
    K = avg_w.shape[0]

    # Negative entropy of the average usage distribution
    # Uniform → entropy = log(K) (max). Collapsed → entropy → 0.
    entropy = -(avg_w * (avg_w + 1e-9).log()).sum()
    max_entropy = torch.log(torch.tensor(float(K), device=avg_w.device))

    # Loss = max_entropy − entropy (≥ 0, minimised when balanced)
    return max_entropy - entropy
