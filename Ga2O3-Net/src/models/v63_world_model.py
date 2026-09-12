"""Phase 63 — LeWorldModel-style doping-type-general surrogate (distilled from the V63 solver oracle).

A permutation-invariant SET model over dopant tokens that maps any doping recipe (single / co / multi /
mixed -- a cation multiset) + host knobs (T, log pO2) to the physical state (E_F, log10[V_O], bound DAP
fraction). Design (docs/phase63_design.md sec 4; research Angle-2/4: SIMPLER than JEPA -- Deep Sets +
host/dopant decoupling + physical encodings, validated by attribution + a frozen linear probe, NOT a heavy
JEPA or a physics-residual loss).

Aggregation is the load-bearing inductive bias:
  * 'sum'  : Deep Sets sum-pool (Zaheer 2017). The solver's only dopant coupling is the ADDITIVE charge
             sum sum_i z_i c_i f_i(E_F); a sum over per-token embeddings represents it exactly and is both
             order-invariant AND cardinality-agnostic -> can extrapolate the LAW from D in {1,2} to D in
             {3,4}. This is the hypothesis Phase 63 tests.
  * 'mean' : normalizes out cardinality (breaks the additive-charge analogy) -> expected to FAIL cardinality
             extrapolation. Ablation control.
  * 'attn' : masked self-attention then sum-pool -> can represent PAIRWISE interactions (the DAP-pairing /
             bound_frac mechanism) that a pure sum cannot. Tests whether the pairwise 'complex mechanism'
             is learnable.

The decoder's pre-head vector is the LATENT = physical state; a frozen linear probe decodes E_F from it
(research caveat: a passing probe means 'saw the input', the mechanism claim rests on the counterfactual
law tests). All-torch, GPU. No DFT/LLM gradient (the labels are the detached solver oracle; Rule 1 spirit).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.codoping_oracle import TOKEN_DIM


class V63WorldModel(nn.Module):
    def __init__(self, token_dim: int = TOKEN_DIM, hidden: int = 96, latent: int = 32,
                 agg: str = "sum", n_heads: int = 4, t_center: float = 950.0, t_scale: float = 200.0,
                 conc_gate: bool = False, logc_idx: int = TOKEN_DIM - 2):
        super().__init__()
        assert agg in ("sum", "mean", "attn")
        self.agg = agg
        # conc_gate: multiply each token embedding by a smooth presence gate g(log10 c) that -> 0 as c -> 0,
        # so a vanishing dopant contributes nothing (enforces the c->0 reduction law L6 exactly in the limit).
        # Centered at log10 c = -5 (real dopants are >=1e-4 -> gate ~1; the artificial c=1e-7 -> gate ~0).
        self.conc_gate = bool(conc_gate); self.logc_idx = int(logc_idx)
        self.phi = nn.Sequential(nn.Linear(token_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        if agg == "attn":
            self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
            self.attn_ln = nn.LayerNorm(hidden)
        self.rho = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.SiLU(),
                                 nn.Linear(hidden, latent), nn.SiLU())
        self.head_EF = nn.Linear(latent, 1)
        self.head_VO = nn.Linear(latent, 1)
        self.head_BF = nn.Linear(latent, 1)
        self.register_buffer("t_center", torch.tensor(float(t_center)))
        self.register_buffer("t_scale", torch.tensor(float(t_scale)))

    def encode(self, tokens, mask, T, lpo2) -> torch.Tensor:
        """Return the latent (physical-state) vector [B, latent]."""
        B, D, _ = tokens.shape
        m = mask.unsqueeze(-1)                                   # [B,D,1]
        if self.conc_gate:                                       # presence gate: ->0 as log10 c -> -inf
            g = torch.sigmoid((tokens[..., self.logc_idx:self.logc_idx + 1] + 5.0) / 0.3)
            m = m * g                                            # vanishing-conc dopant contributes nothing
        e = self.phi(tokens) * m                                 # masked token embeddings [B,D,h]
        if self.agg == "attn":
            key_pad = (mask < 0.5)                               # True where padded
            # guard fully-padded rows (never happens: D>=1) — still safe
            a, _ = self.attn(e, e, e, key_padding_mask=key_pad, need_weights=False)
            e = self.attn_ln(e + a * m)                          # residual, re-mask
            S = (e * m).sum(dim=1)                               # sum-pool the attended tokens
        elif self.agg == "mean":
            S = (e).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:                                                    # sum
            S = e.sum(dim=1)
        knobs = torch.stack([(T - self.t_center) / self.t_scale, lpo2 / 4.0], dim=-1)  # [B,2]
        return self.rho(torch.cat([S, knobs], dim=-1))          # [B, latent]

    def forward(self, tokens, mask, T, lpo2) -> dict:
        z = self.encode(tokens, mask, T, lpo2)
        return dict(latent=z,
                    E_F=self.head_EF(z).squeeze(-1),
                    log10_VO=self.head_VO(z).squeeze(-1),
                    bound_frac=self.head_BF(z).squeeze(-1))


class SlotMLP(nn.Module):
    """Non-permutation-invariant control: flatten the padded D_MAX slots into one fixed-length MLP input.

    Sorting is NOT applied (raw slot order), so it is order-sensitive and cardinality-bound -> the ablation
    baseline that should FAIL the permutation-invariance and cardinality-extrapolation gates.
    """
    def __init__(self, token_dim: int = TOKEN_DIM, d_max: int = 4, hidden: int = 96, latent: int = 32,
                 t_center: float = 950.0, t_scale: float = 200.0):
        super().__init__()
        self.d_max = d_max
        self.net = nn.Sequential(nn.Linear(token_dim * d_max + 2, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, latent), nn.SiLU())
        self.head_EF = nn.Linear(latent, 1); self.head_VO = nn.Linear(latent, 1); self.head_BF = nn.Linear(latent, 1)
        self.register_buffer("t_center", torch.tensor(float(t_center)))
        self.register_buffer("t_scale", torch.tensor(float(t_scale)))

    def encode(self, tokens, mask, T, lpo2):
        B = tokens.shape[0]
        flat = (tokens * mask.unsqueeze(-1)).reshape(B, -1)
        knobs = torch.stack([(T - self.t_center) / self.t_scale, lpo2 / 4.0], dim=-1)
        return self.net(torch.cat([flat, knobs], dim=-1))

    def forward(self, tokens, mask, T, lpo2):
        z = self.encode(tokens, mask, T, lpo2)
        return dict(latent=z, E_F=self.head_EF(z).squeeze(-1),
                    log10_VO=self.head_VO(z).squeeze(-1), bound_frac=self.head_BF(z).squeeze(-1))
