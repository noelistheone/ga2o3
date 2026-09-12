"""V61 model — mechanism-explicit, measurement-aware predictor (design docs/phase61_v61_design.md §2).

Pipeline (per recipe):
  frozen detached features  ──► (a) ΔE_f head (bounded tanh, Rule 2) ──► dH_f^q
                                                                         │
  (T, pO2, dopant charge)  ──────────────────────────────────────────► (a) DIFFERENTIABLE
                                                                         SELF-CONSISTENT SOLVER
                                                                         (defect_equilibrium)
                                                                         │ bulk log10[V_O^••], E_F* (the MECHANISM)
  (element, log c)  ──► (c) SIGN-CONSTRAINED measurement response ◄──────┘
                          r = w_phys·z(bulk) + g_elem·monotone(log c)   (per-element sign data-adjudicated)
                          + (d) heteroscedastic log σ head

The per-DOI measurement operator y ≈ α_DOI + β_DOI·r + σ·ε (component b) is fit in the TRAINER
(closed-form per fold / hierarchical), not here, so this module stays a pure recipe→(mechanism, r, σ) map.

Rule 1: DFT/MACE features enter the ΔE_f & σ heads detached. Rule 2: dH_f = closed-form baseline +
bounded free ΔE_f (never DFT-hard-anchored). Sign is a LEARNED per-element gate, never locked (≠ V53-γ).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.models.defect_equilibrium import solve_equilibrium, NATIVE_VO_DHF, EG_GA2O3  # noqa: F401

C_REF = 1e-2  # 1 at% reference (matches kroger_synthetic / dcwm_transitions)

# fixed element vocab (carrier-sign table + undoped + an OTHER bucket); index 0 = undoped, last = other
_ELEM_VOCAB = ["undoped", "Mg", "Zn", "Cu", "Ni", "N", "F", "Al", "Fe", "B", "Er", "Eu", "Cr", "In",
               "Sb", "Bi", "V", "Si", "Sn", "Ti", "Ge", "Ta", "W", "H", "Zr", "OTHER"]
_ELEM2IDX = {e: i for i, e in enumerate(_ELEM_VOCAB)}
# carrier sign for the BULK Fermi shift (donor +1, acceptor −1, isovalent 0) — sets z_dop into the solver
_CARRIER = {"Mg": -1, "Zn": -1, "Cu": -1, "Ni": -1, "N": -1, "F": -1,
            "Al": 0, "Fe": 0, "B": 0, "Er": 0, "Eu": 0, "Cr": 0, "In": 0,
            "Sb": +1, "Bi": +1, "V": +1, "Si": +1, "Sn": +1, "Ti": +1, "Ge": +1,
            "Ta": +1, "W": +1, "H": +1, "Zr": +1}


def elem_to_idx(elem: str) -> int:
    return _ELEM2IDX.get(elem, _ELEM2IDX["OTHER"])


def carrier_sign(elem: str) -> float:
    return float(_CARRIER.get(elem, 0.0))


class _MonotoneScalar(nn.Module):
    """Guaranteed monotone-INCREASING scalar function R→R (abs weights + softplus activation).

    out(x) = Σ_j |w2_j|·softplus(|w1_j|·x + b1_j) + b2.  d out/dx ≥ 0 everywhere (per-feature
    monotonicity by construction, Constrained-Monotonic-NN family, Runje & Shankaranarayana ICML'23).
    Sign/scale is applied OUTSIDE via a per-element gate, so this is the shared monotone SHAPE.
    """
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.randn(hidden) * 0.5 / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: [B]
        h = torch.nn.functional.softplus(self.w1.abs().view(1, -1) * x.view(-1, 1) + self.b1.view(1, -1))
        return (self.w2.abs().view(1, -1) * h).sum(dim=1) + self.b2  # [B], monotone ↑ in x


class V61Net(nn.Module):
    """Recipe → (bulk mechanism, measurement response r, heteroscedastic log σ)."""

    def __init__(self, feat_dim: int = 71, ef_swing: float = 0.6, hidden: int = 96,
                 n_elem: int = len(_ELEM_VOCAB), bulk_center: float = 12.0, bulk_scale: float = 4.0,
                 logc_center: float = -2.0, logc_scale: float = 1.0,
                 use_solver: bool = True, sign_mode: str = "constrained"):
        super().__init__()
        # ablation switches: use_solver (bulk mechanism in r), sign_mode (constrained|free|nonmono)
        self.use_solver = bool(use_solver)
        self.sign_mode = sign_mode
        emb_dim = 8
        self.elem_emb = nn.Embedding(n_elem, emb_dim)            # for the free / nonmono ablation heads
        self.free_head = nn.Sequential(nn.Linear(1 + emb_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        # (a) bounded ΔE_f head over DETACHED frozen features → 3 charge states (q=0,1,2)
        self.delta_ef = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3))
        self.ef_swing = float(ef_swing)
        self.register_buffer("dHf_base", torch.tensor([NATIVE_VO_DHF[0], NATIVE_VO_DHF[1],
                                                        NATIVE_VO_DHF[2]], dtype=torch.float32))
        # zero-init last layer ⇒ ΔE_f = 0 at init ⇒ solver = DFT-anchored closed form (bounded downside)
        nn.init.zeros_(self.delta_ef[-1].weight); nn.init.zeros_(self.delta_ef[-1].bias)

        # (c) sign-constrained measurement response
        self.mono = _MonotoneScalar(hidden=16)                  # monotone↑ shape in log c
        self.g_elem = nn.Embedding(n_elem, 1)                   # per-element signed scale (sign = data-adjudicated)
        nn.init.zeros_(self.g_elem.weight)
        self.w_phys = nn.Parameter(torch.zeros(1))              # weight on standardized bulk mechanism

        # (d) heteroscedastic log σ head (detached features); zero-init last layer ⇒ σ=1 at start
        self.sigma_head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.sigma_head[-1].weight); nn.init.zeros_(self.sigma_head[-1].bias)

        self.register_buffer("bulk_center", torch.tensor(bulk_center))
        self.register_buffer("bulk_scale", torch.tensor(bulk_scale))
        self.register_buffer("logc_center", torch.tensor(logc_center))
        self.register_buffer("logc_scale", torch.tensor(logc_scale))

    def dHf(self, feats: torch.Tensor) -> torch.Tensor:
        """dH_f^q = closed-form baseline + bounded free correction (Rule 2). feats DETACHED upstream."""
        return self.dHf_base.view(1, -1) + self.ef_swing * torch.tanh(self.delta_ef(feats))

    def forward(self, feats: torch.Tensor, T_K: torch.Tensor, log_pO2: torch.Tensor,
                z_dop: torch.Tensor, c_frac: torch.Tensor, elem_idx: torch.Tensor) -> dict:
        feats = feats.detach()                                  # Rule 1: no grad to frozen features
        dH = self.dHf(feats)                                    # [B,3]
        sol = solve_equilibrium(dH, T_K, log_pO2, z_dop, c_frac)
        bulk = sol["log10_VO"]                                  # [B] the MECHANISM (bulk equilibrium)
        # measurement response r (per-DOI α,β handled in trainer)
        z_bulk = (bulk - self.bulk_center) / self.bulk_scale
        log_c = torch.log10(c_frac.clamp_min(1e-6) / C_REF)
        log_c_std = (log_c - self.logc_center) / self.logc_scale
        if self.sign_mode == "constrained":                     # GUARANTEED per-element sign (default)
            shape = self.mono(log_c_std)                        # [B] monotone↑ in log c
            r_dop = self.g_elem(elem_idx).squeeze(-1) * shape   # signed per-element scale × monotone shape
        else:                                                   # free | nonmono: no sign guarantee (ablation)
            emb = self.elem_emb(elem_idx)
            r_dop = self.free_head(torch.cat([log_c_std.unsqueeze(-1), emb], dim=-1)).squeeze(-1)
        bulk_term = self.w_phys * z_bulk if self.use_solver else torch.zeros_like(r_dop)
        r = bulk_term + r_dop                                   # [B]
        log_sigma = self.sigma_head(feats).squeeze(-1)          # [B]
        return dict(bulk_logVO=bulk, E_F=sol["E_F"], residual=sol["residual"],
                    r=r, log_sigma=log_sigma, dHf=dH)
