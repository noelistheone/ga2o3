"""Phase 60 V60-DCWM — the LeJEPA-stable, action-conditioned defect-chemistry world model.

Implements, faithfully to docs/phase60_dcwm_jepa_design.md §3-§5 and the verified LeJEPA/LeWM math
(docs/phase60_research/), the following pieces:

  StateEncoder      f_θ : x (state features) -> z ∈ R^d            (+ frozen input standardizer)
  ActionEncoder     E_a : a (intervention)   -> a_emb
  AdaLNZeroBlock         : DiT/LeWM zero-gated residual; identity at init (ReZero gate α=0)
  WorldModelPredictor g_φ: (z, a_emb) -> ẑ'  (ẑ'=z at init -> zero counterfactual slope at init)
  SIGReg                 : sliced Epps-Pulley characteristic-function test -> push Z -> N(0,I)
                           (Cramér–Wold + Epps–Pulley; bounded gradients; O(N); no covariance inversion)
  DualReadout       r_ψ : z -> (measured_logVO, bulk_logVO)  (two heads, §Q1 decision)
  DCWM                   : the full module + Stage-A loss + counterfactual probe.

Anti-collapse is SIGReg only (LeWM): NO EMA, NO stop-gradient, NO predictor asymmetry. The target
latent z' = f_θ(x_t) is the encoder's own gradient-carrying output (end-to-end).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA arXiv:2511.08544)
# ──────────────────────────────────────────────────────────────────────────────
class SIGReg(nn.Module):
    """Push a batch of embeddings Z∈R^{N×d} toward N(0,I).

    SIGReg(Z) = (1/M) Σ_m EP( Z u_m ),  u_m ~ Unif(S^{d-1})
    EP(h)     = ∫ w(t) |φ̂_h(t) − e^{−t²/2}|² dt,   φ̂_h(t)=(1/N)Σ_n e^{i t h_n},  w(t)=e^{−t²/σ²}
    (Cramér–Wold: SIGReg→0 ⇔ ℙ_Z→N(0,I).) Trapezoidal quadrature on a symmetric t-grid.
    Differentiable in Z (hence f_θ); bounded gradients (Thm 4) -> stable end-to-end.
    """

    def __init__(self, n_slices: int = 256, n_quad: int = 17, t_max: float = 5.0, sigma: float = 1.0):
        super().__init__()
        self.n_slices = int(n_slices)
        self.sigma = float(sigma)
        t = torch.linspace(-t_max, t_max, int(n_quad))
        w = torch.exp(-(t ** 2) / (sigma ** 2))
        self.register_buffer("t_grid", t)                  # [Q]
        self.register_buffer("w", w)                        # [Q]
        # trapezoid weights for ∫ over the grid
        dt = (2 * t_max) / (n_quad - 1)
        trap = torch.full((int(n_quad),), dt)
        trap[0] = trap[-1] = dt / 2
        self.register_buffer("trap", trap)                  # [Q]

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: [N, d] -> scalar regularizer (mean over slices, integrated over t)."""
        N, d = Z.shape
        U = torch.randn(d, self.n_slices, device=Z.device, dtype=Z.dtype)
        U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-8)         # [d, M] unit columns
        H = Z @ U                                                   # [N, M] projections
        # empirical characteristic function at each t: φ̂(t) = mean_n exp(i t h)
        # th: [Q, N, M]
        th = self.t_grid.view(-1, 1, 1) * H.unsqueeze(0)
        phi_re = torch.cos(th).mean(dim=1)                          # [Q, M]
        phi_im = torch.sin(th).mean(dim=1)                          # [Q, M]
        target = torch.exp(-(self.t_grid ** 2) / 2.0).view(-1, 1)   # [Q, 1]
        integrand = self.w.view(-1, 1) * ((phi_re - target) ** 2 + phi_im ** 2)   # [Q, M]
        ep = (self.trap.view(-1, 1) * integrand).sum(dim=0)         # [M]  ∫ over t per slice
        return ep.mean()                                            # mean over slices


# ──────────────────────────────────────────────────────────────────────────────
# Encoders
# ──────────────────────────────────────────────────────────────────────────────
class InputStandardizer(nn.Module):
    """Frozen (mean,std) standardization of the raw state features. Set via .fit()."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self._fitted = False

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        self.mean.copy_(X.mean(dim=0))
        self.std.copy_(X.std(dim=0).clamp_min(1e-6))
        self._fitted = True

    def forward(self, x):
        return (x - self.mean) / self.std


class StateEncoder(nn.Module):
    """f_θ: standardized state features -> latent z ∈ R^d (no output LayerNorm — SIGReg owns the
    output distribution)."""

    def __init__(self, in_dim: int, d: int = 64, hidden: int = 192, dropout: float = 0.1):
        super().__init__()
        self.std = InputStandardizer(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, x):
        return self.net(self.std(x))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, a_emb: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, a_emb),
        )

    def forward(self, a):
        return self.net(a)


# ──────────────────────────────────────────────────────────────────────────────
# Zero-gated AdaLN predictor (LeWM / DiT AdaLN-Zero; ReZero scalar gate α init 0)
# ──────────────────────────────────────────────────────────────────────────────
class AdaLNZeroBlock(nn.Module):
    """out = z + α · MLP( (1+γ(a))⊙LN(z) + β(a) ),   α (ReZero gate) init 0 -> identity at init.
    Action conditions LayerNorm scale/shift (AdaLN); α opens first, then γ/β/MLP (one-step lag)."""

    def __init__(self, d: int, a_emb: int, mlp_mult: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.cond = nn.Linear(a_emb, 2 * d)                # -> (γ, β); small init
        nn.init.zeros_(self.cond.bias)
        nn.init.normal_(self.cond.weight, std=1e-3)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d), nn.SiLU(), nn.Linear(mlp_mult * d, d)
        )
        self.alpha = nn.Parameter(torch.zeros(1))          # ReZero gate (identity at init)

    def forward(self, z, a_emb):
        gamma, beta = self.cond(a_emb).chunk(2, dim=-1)
        h = (1.0 + gamma) * self.ln(z) + beta
        return z + self.alpha * self.mlp(h)


class WorldModelPredictor(nn.Module):
    """g_φ: (z, a_emb) -> ẑ'. Stack of AdaLN-Zero blocks. ẑ'=z at init."""

    def __init__(self, d: int, a_emb: int, n_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d, a_emb) for _ in range(n_blocks)])

    def forward(self, z, a_emb):
        for blk in self.blocks:
            z = blk(z, a_emb)
        return z


# ──────────────────────────────────────────────────────────────────────────────
# Dual readout (measured-proxy + bulk-equilibrium V_O); standardized output
# ──────────────────────────────────────────────────────────────────────────────
class DualReadout(nn.Module):
    def __init__(self, d: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
            )
        self.meas = head()
        self.bulk = head()

    def forward(self, z):
        return self.meas(z).squeeze(-1), self.bulk(z).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Full module
# ──────────────────────────────────────────────────────────────────────────────
class DCWM(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, d: int = 64, a_emb: int = 32,
                 enc_hidden: int = 192, n_blocks: int = 4, dropout: float = 0.1,
                 sig_slices: int = 256, sig_quad: int = 17):
        super().__init__()
        self.d = d
        self.encoder = StateEncoder(in_dim, d=d, hidden=enc_hidden, dropout=dropout)
        self.action_enc = ActionEncoder(action_dim, a_emb=a_emb)
        self.predictor = WorldModelPredictor(d, a_emb, n_blocks=n_blocks)
        self.readout = DualReadout(d, dropout=dropout)
        self.sigreg = SIGReg(n_slices=sig_slices, n_quad=sig_quad)
        # target standardization (set by trainer from synthetic dist); real log10[V_O] = mean + std*out
        self.register_buffer("y_meas_mean", torch.zeros(1))
        self.register_buffer("y_meas_std", torch.ones(1))
        self.register_buffer("y_bulk_mean", torch.zeros(1))
        self.register_buffer("y_bulk_std", torch.ones(1))
        # Stage-B Δ-correction on the measured head (zero-init -> no change until fit on experiment)
        self.delta_meas = nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.delta_meas[-1].weight); nn.init.zeros_(self.delta_meas[-1].bias)

    # --- standardization helpers ---
    def set_target_stats(self, ym_mean, ym_std, yb_mean, yb_std):
        self.y_meas_mean.fill_(float(ym_mean)); self.y_meas_std.fill_(max(float(ym_std), 1e-6))
        self.y_bulk_mean.fill_(float(yb_mean)); self.y_bulk_std.fill_(max(float(yb_std), 1e-6))

    def encode(self, x):
        return self.encoder(x)

    def predict(self, z, a):
        return self.predictor(z, self.action_enc(a))

    def readout_std(self, z):
        """standardized readouts (meas, bulk)."""
        return self.readout(z)

    def readout_real(self, z, apply_delta: bool = False):
        """real log10[V_O] readouts (meas, bulk)."""
        m_s, b_s = self.readout(z)
        meas = self.y_meas_mean + self.y_meas_std * m_s
        if apply_delta:
            meas = meas + self.delta_meas(z).squeeze(-1)
        bulk = self.y_bulk_mean + self.y_bulk_std * b_s
        return meas, bulk

    # --- Stage-A forward (returns everything the loss needs) ---
    def forward_stageA(self, x_s, a, x_t):
        z_s = self.encode(x_s)
        z_t = self.encode(x_t)
        z_pred = self.predict(z_s, a)
        m_s, b_s = self.readout(z_s)
        m_t, b_t = self.readout(z_t)
        return dict(z_s=z_s, z_t=z_t, z_pred=z_pred, m_s=m_s, b_s=b_s, m_t=m_t, b_t=b_t)

    # --- counterfactual probe (eval): apply action through the WM, read measured head ---
    @torch.no_grad()
    def counterfactual_measured(self, x_s, a, apply_delta: bool = True):
        """Sweep an action through the WM. The magnitude Δ-correction is evaluated at the BASE
        recipe latent (a constant offset held fixed across the sweep) so the world model owns the
        counterfactual SLOPE and Δ owns only the magnitude OFFSET (design: 'Δ corrects magnitude,
        not the law')."""
        z = self.encode(x_s)
        zc = self.predict(z, a)
        meas, _ = self.readout_real(zc, apply_delta=False)
        if apply_delta:
            meas = meas + self.delta_meas(z).squeeze(-1)      # Δ at base z -> constant offset
        return meas
