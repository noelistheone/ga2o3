"""
Phase 59 — V59-CALM zero-gated Δ-residual head.

Implements §3.4 of docs/phase59_llm_multiexpert_physics_law_design.md.

The whole point (C1 + C2 + the no-regression-at-init guarantee):

    ŷ = ŷ_phys(u₀)  +  Δ
    Δ = w_out · σ_act( W_h [ u₀ ⊕ c_new ⊕ log c ] )
    c_new = g_L·tanh(W_L z_expert) + g_D·tanh(W_D φ_DFT) + g_M·tanh(W_M z_MACE)

with

  * **w_out zero-initialized** ⇒ Δ = 0 at init *regardless of c_new* ⇒ at step 0
    the model is byte-identical to V55-Ext (the closed-form Brouwer floor). This
    is the formal "adding a modality cannot regress at init" guarantee (§3.4).
  * **gates g_* zero-initialized** (ReZero / Flamingo tanh-gate / LayerScale,
    arXiv:2003.04887). A one-step gradient lag follows from w_out=0 (∂L/∂g_m ∝
    w_out = 0 at step 0); w_out opens first (step 1), gates from step 2. Harmless,
    documented in §3.4-R2. Set `gate_init` > 0 to remove the lag if desired.
  * **All modality inputs are detached** (Rule 1) and NaN-imputed to 0 (cache
    miss → that modality simply contributes nothing for that row).
  * The new modalities feed ONLY this residual, never the magnitude floor u₀ →
    path separation (C2). u₀ is produced by the UNCHANGED V55-Ext FiLM fusion.
  * **Low-rank fusion** (`fusion_rank`, LMF arXiv:1806.00064) keeps params tiny:
    with the defaults the module adds ≈ 3.0–3.1 k trainable params (vs V58's
    +198,760), well inside the C1 < 10 k ceiling.
  * **Asymmetric modality dropout** (`modality_dropout`): during training the
    whole new-modality block c_new is zeroed with prob p, forcing Δ to either
    lean on the physics floor or genuinely use the modality — the standard
    tiny-N robustifier that augments the N=14 set with single-modality views.

Per-modality enable flags (`use_expert`/`use_dft`/`use_mace`) exist so the §9
ablations B/D/E can switch a modality on/off without changing the param count of
the active sub-modules.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CALMResidual(nn.Module):
    """Zero-gated, path-separated Δ-residual over a closed-form physics floor."""

    def __init__(
        self,
        u0_dim: int = 160,
        expert_dim: int = 16,
        dft_dim: int = 16,
        mace_dim: int = 64,
        fusion_rank: int = 4,
        hidden: int = 16,
        dropout: float = 0.0,
        modality_dropout: float = 0.3,
        gate_init: float = 0.0,
        use_expert: bool = True,
        use_dft: bool = True,
        use_mace: bool = True,
        act: str = "silu",
    ):
        super().__init__()
        self.u0_dim = int(u0_dim)
        self.expert_dim = int(expert_dim)
        self.dft_dim = int(dft_dim)
        self.mace_dim = int(mace_dim)
        self.fusion_rank = int(fusion_rank)
        self.modality_dropout = float(modality_dropout)
        self.use_expert = bool(use_expert)
        self.use_dft = bool(use_dft)
        self.use_mace = bool(use_mace)

        # Low-rank modality projections → fusion_rank-d (the c_new space).
        # Only built for enabled modalities so disabled ones cost zero params.
        self.W_L = nn.Linear(self.expert_dim, self.fusion_rank) if self.use_expert else None
        self.W_D = nn.Linear(self.dft_dim, self.fusion_rank) if self.use_dft else None
        self.W_M = nn.Linear(self.mace_dim, self.fusion_rank) if self.use_mace else None

        # Zero-init scalar gates (ReZero/LayerScale). One per enabled modality.
        self.g_L = nn.Parameter(torch.full((1,), float(gate_init))) if self.use_expert else None
        self.g_D = nn.Parameter(torch.full((1,), float(gate_init))) if self.use_dft else None
        self.g_M = nn.Parameter(torch.full((1,), float(gate_init))) if self.use_mace else None

        # Δ_θ: one hidden layer over [u₀ ⊕ c_new ⊕ log c].
        in_dim = self.u0_dim + self.fusion_rank + 1
        self.W_h = nn.Linear(in_dim, hidden)
        self.act = nn.SiLU() if act == "silu" else nn.Tanh()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # THE guarantee: zero-init output layer ⇒ Δ = 0 at init.
        self.w_out = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.w_out.weight)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _clean(x: torch.Tensor | None) -> torch.Tensor | None:
        """Detach (Rule 1) + replace NaN/Inf (cache miss) with 0."""
        if x is None:
            return None
        x = x.detach()
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── forward ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        u0: torch.Tensor,                 # [B, u0_dim]  (gradient flows — it's the V55-Ext latent)
        z_expert: torch.Tensor | None,    # [B, expert_dim] (detached, cached)
        phi_dft: torch.Tensor | None,     # [B, dft_dim]    (detached, cached; NaN on miss)
        z_mace: torch.Tensor | None,      # [B, mace_dim]   (detached, cached)
        logc: torch.Tensor,               # [B, 1] log10 concentration (DIFFERENTIABLE — DCC reads ∂Δ/∂logc)
    ) -> torch.Tensor:
        B = u0.shape[0]
        device, dtype = u0.device, u0.dtype

        c_new = torch.zeros(B, self.fusion_rank, device=device, dtype=dtype)
        if self.use_expert and z_expert is not None:
            z = self._clean(z_expert).to(device=device, dtype=dtype)
            c_new = c_new + self.g_L * torch.tanh(self.W_L(z))
        if self.use_dft and phi_dft is not None:
            d = self._clean(phi_dft).to(device=device, dtype=dtype)
            c_new = c_new + self.g_D * torch.tanh(self.W_D(d))
        if self.use_mace and z_mace is not None:
            m = self._clean(z_mace).to(device=device, dtype=dtype)
            c_new = c_new + self.g_M * torch.tanh(self.W_M(m))

        # Asymmetric modality dropout (training only): drop the new-modality block.
        if self.training and self.modality_dropout > 0.0:
            if torch.rand((), device=device) < self.modality_dropout:
                c_new = c_new * 0.0

        if logc.dim() == 1:
            logc = logc.unsqueeze(-1)
        logc = logc.to(device=device, dtype=dtype)

        inp = torch.cat([u0, c_new, logc], dim=-1)
        delta = self.w_out(self.drop(self.act(self.W_h(inp))))   # [B, 1]
        return delta
