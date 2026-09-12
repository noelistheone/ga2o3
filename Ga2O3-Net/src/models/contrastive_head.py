"""Phase 53 V53-ζ — PDR↔V_O latent InfoNCE contrastive head.

Aligns PDR-latent and V_O-latent on samples that share (DOI, dopant_label) by
adding a SimCLR-style symmetric InfoNCE term on the FiLM output. PDR has 52
labels; V_O has 14 — sharing latent axes raises the effective N for V_O
regularization to ~66.

Design choices (matching plan §Step 1b):
  - Projection head: 160 → 64 → 32, L2-normalized output
  - Temperature τ = 0.5
  - Positives = same group_id (DOI + dopant_label hash)
  - V_O regression loss is unchanged — InfoNCE acts as auxiliary regularizer
  - PDR regression head is NOT enabled (V44-V8 multitask is NEG)

The projection is added as a sibling module to BrouwerHeadVC inside Ga2O3Net.
The trainer (`finetune_trainer.py`) calls `model.contrastive_proj(fused)` and
passes group_ids extracted from the batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCEProjector(nn.Module):
    """SimCLR-style projection: in_dim → hidden → out_dim (L2-normalized)."""

    def __init__(self, in_dim: int = 160, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return F.normalize(z, dim=-1)


def info_nce_loss(
    z: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Symmetric InfoNCE with group-id positives.

    Args:
        z:           [B, D] L2-normalized projections.
        group_ids:   [B] long tensor; positives are pairs with matching id.
        temperature: softmax temperature.

    Returns:
        Scalar contrastive loss. Returns 0.0 if no sample has any positive
        (i.e., all group_ids unique within the batch — common for tiny batches).
    """
    if z.dim() != 2:
        raise ValueError(f"info_nce_loss expects [B, D], got {tuple(z.shape)}")
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    sim = z @ z.T / float(temperature)                      # [B, B]
    # Build positive mask (same group_id, excluding self)
    pos_mask = (group_ids[:, None] == group_ids[None, :]).float()
    pos_mask.fill_diagonal_(0.0)
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # Numerical-stable log-softmax over rows
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_exp = torch.exp(sim - sim_max)
    sim_exp = sim_exp - torch.diag_embed(sim_exp.diagonal())  # remove self-similarity
    pos_sum = (sim_exp * pos_mask).sum(dim=1)
    all_sum = sim_exp.sum(dim=1)
    loss = -torch.log(pos_sum[valid] / (all_sum[valid] + 1e-12) + 1e-12)
    return loss.mean()


# ---------------------------------------------------------------------------
# Phase 54 V54-A1: SINCERE multi-positive contrastive loss
# Feeney & Hughes, arXiv:2309.14277, Eq. 5.
#
# Key difference vs vanilla supervised InfoNCE:
#   Positives are REMOVED from the denominator. For each anchor i and each
#   positive p in P(i), the contrast is between p (numerator) and the negative
#   pool N(i) (denominator). Vanilla supervised InfoNCE puts all positives in
#   both numerator AND denominator, which cancels gradient when |P(i)| >= 2
#   and drives intra-class repulsion. SINCERE eliminates that pathology.
#
# Formula:
#   L = − (1/|A|) Σ_{i∈A} (1/|P(i)|) Σ_{p∈P(i)}
#         log( e^{s_ip/τ} / ( e^{s_ip/τ} + Σ_{n∈N(i)} e^{s_in/τ} ) )
# where A = {i : |P(i)| > 0}, P(i) = same-group neighbors (i excluded),
# N(i) = different-group neighbors.
# ---------------------------------------------------------------------------

def sincere_loss_per_row(
    z: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.5,
    hard_neg_log_weights: torch.Tensor | None = None,
    sentinel: int = -1,
) -> torch.Tensor:
    """Per-row SINCERE loss; returns [B], zero on rows without positives.

    Args:
        z:                    [B, D] L2-normalized projections.
        group_ids:            [B] long tensor. Rows with id == sentinel are
                              treated as having NO valid positives or negatives
                              from them (excluded from both num and denom).
        temperature:          softmax temperature.
        hard_neg_log_weights: optional [B, B] additive log-weighting on the
                              NEGATIVE pool only (added to s_in/τ before
                              exponentiation). Use to up-weight hard negatives
                              (chem-close + proc-far). Diagonal should be 0.
        sentinel:             group-id value used to mark "no contrastive
                              participation" (e.g. undoped rows).

    Returns:
        [B] per-row loss tensor. Rows without valid positives → 0.
    """
    if z.dim() != 2:
        raise ValueError(f"sincere_loss_per_row expects [B, D], got {tuple(z.shape)}")
    B = z.shape[0]
    device, dtype = z.device, z.dtype
    if B < 2:
        return torch.zeros(B, device=device, dtype=dtype)

    # Cosine similarity / τ
    sim = (z @ z.T) / float(temperature)                          # [B, B]

    # Sentinel rows participate as neither anchor nor pair partner.
    active = (group_ids != sentinel)                              # [B]
    if active.sum() < 2:
        return torch.zeros(B, device=device, dtype=dtype)

    # Build positive / negative masks.
    eye = torch.eye(B, dtype=torch.bool, device=device)
    same = (group_ids[:, None] == group_ids[None, :])             # [B, B]
    both_active = active[:, None] & active[None, :]
    pos_mask = same & both_active & ~eye                          # [B, B] bool
    neg_mask = ~same & both_active & ~eye                         # [B, B] bool

    # Per-row negative logsumexp = log Σ_{n∈N(i)} e^{s_in/τ}.
    sim_neg = sim.masked_fill(~neg_mask, float("-inf"))           # [B, B]
    if hard_neg_log_weights is not None:
        sim_neg = sim_neg + hard_neg_log_weights * neg_mask.float()
    neg_lse = torch.logsumexp(sim_neg, dim=1)                     # [B]
    has_neg = neg_mask.any(dim=1)
    # Rows with no negatives at all: skip (per-row loss = 0).

    # For each (i, p) positive pair, log-prob = s_ip/τ − log(e^{s_ip/τ} + e^{neg_lse[i]})
    # = s_ip/τ − logsumexp([s_ip/τ, neg_lse[i]])
    sim_pos = sim                                                 # [B, B]
    neg_lse_b = neg_lse.unsqueeze(1).expand_as(sim_pos)            # [B, B]
    # Stack along new dim and logsumexp over it = log(e^{s_ip} + e^{neg_lse_i})
    denom = torch.logsumexp(
        torch.stack([sim_pos, neg_lse_b], dim=-1), dim=-1
    )                                                              # [B, B]
    log_prob = sim_pos - denom                                     # [B, B]

    # Average over positives for each anchor.
    pos_count = pos_mask.sum(dim=1)                                # [B]
    valid_row = pos_count.gt(0) & has_neg                          # [B]
    if valid_row.sum() == 0:
        return torch.zeros(B, device=device, dtype=dtype)

    # Sum log_prob over positives; divide by count.
    log_prob_pos = log_prob.masked_fill(~pos_mask, 0.0)            # [B, B]
    sum_log_prob = log_prob_pos.sum(dim=1)                         # [B]
    per_row = torch.zeros(B, device=device, dtype=dtype)
    per_row[valid_row] = -sum_log_prob[valid_row] / pos_count[valid_row].clamp(min=1).to(dtype)
    return per_row


def sincere_loss(
    z: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.5,
    hard_neg_log_weights: torch.Tensor | None = None,
    sentinel: int = -1,
) -> torch.Tensor:
    """Scalar SINCERE loss = mean over rows with valid positives.

    Returns 0.0 if no row has any positive (matches info_nce_loss behavior).
    """
    per_row = sincere_loss_per_row(
        z, group_ids, temperature=temperature,
        hard_neg_log_weights=hard_neg_log_weights, sentinel=sentinel,
    )
    valid = per_row != 0.0
    if valid.sum() == 0:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    return per_row[valid].mean()


def hard_neg_log_weighting(
    chem_emb: torch.Tensor,
    proc_emb: torch.Tensor,
    alpha: float = 2.0,
    log_clamp: float = 4.0,
) -> torch.Tensor:
    """Robinson et al. arXiv:2010.04592 §3.2 hard-negative reweighting.

    Hard negative = chemically SIMILAR but process-DIFFERENT (the kind of
    sample that looks like a positive but should be repelled).

    Returns [B, B] log-weight matrix to ADD to sim/τ in the SINCERE denom:
        log w_ij = α · (proc_dist_ij − chem_dist_ij)
    clamped to ±log_clamp to bound gradient magnitude (Robinson recommends
    upper bound on importance ratio to avoid the unbounded estimator).

    Args:
        chem_emb:  [B, K] standardized per-row chemistry descriptors.
        proc_emb:  [B, P] standardized process tensor.
        alpha:     temperature on the (proc - chem) margin. α=0 disables.
        log_clamp: max absolute value of the log-weight (default 4 ≈ ×55).
    """
    if alpha <= 0.0:
        return torch.zeros(chem_emb.shape[0], chem_emb.shape[0],
                            device=chem_emb.device, dtype=chem_emb.dtype)
    chem_dist = torch.cdist(chem_emb, chem_emb, p=2)              # [B, B]
    proc_dist = torch.cdist(proc_emb, proc_emb, p=2)              # [B, B]
    log_w = alpha * (proc_dist - chem_dist)
    log_w = log_w.clamp(min=-log_clamp, max=log_clamp)
    # Zero diagonal so self never gets re-weighted (it's masked anyway, but
    # keeps the gradient clean).
    eye = torch.eye(log_w.shape[0], dtype=torch.bool, device=log_w.device)
    log_w = log_w.masked_fill(eye, 0.0)
    return log_w

