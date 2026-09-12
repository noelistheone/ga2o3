"""V62 model — mechanism-aware CO-DOPING predictor (design docs/phase62_codoping_design.md sec 4).

Strict generalization of V61Net (src/models/v61_net.py) to D simultaneous dopants. The measurement
response is the SUM of per-element signed monotone contributions (the per-element signs/scales are
learned on single-dopant data and transfer additively to co-doping), plus the shared-Fermi-level
mechanism term from the MULTI-DOPANT solver. At D=1 with V61-default transition levels this reduces to
V61Net bit-for-bit (regression-tested: scripts/79_v62_reduction_test.py).

Parameter names mirror V61Net exactly so a V61 state_dict transfers directly. Rule 1 (detached feats),
Rule 2 (bounded ΔE_f around the closed-form baseline) preserved.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.defect_equilibrium_codoping import solve_equilibrium_codoping
from src.models.v61_net import _MonotoneScalar, _ELEM_VOCAB, C_REF, elem_to_idx, carrier_sign  # noqa: F401
from src.models.defect_equilibrium import NATIVE_VO_DHF, EG_GA2O3


class V62CoDopingNet(nn.Module):
    """Recipe (1..D dopants) -> (bulk mechanism, measurement response r, heteroscedastic log sigma)."""

    def __init__(self, feat_dim: int = 71, ef_swing: float = 0.6, hidden: int = 96,
                 n_elem: int = len(_ELEM_VOCAB), bulk_center: float = 12.0, bulk_scale: float = 4.0,
                 logc_center: float = -2.0, logc_scale: float = 1.0,
                 use_solver: bool = True, sign_mode: str = "constrained"):
        super().__init__()
        self.use_solver = bool(use_solver)
        self.sign_mode = sign_mode
        emb_dim = 8
        self.elem_emb = nn.Embedding(n_elem, emb_dim)
        self.free_head = nn.Sequential(nn.Linear(1 + emb_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        # (a) bounded ΔE_f head over DETACHED frozen features -> 3 V_O charge states
        self.delta_ef = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3))
        self.ef_swing = float(ef_swing)
        self.register_buffer("dHf_base", torch.tensor([NATIVE_VO_DHF[0], NATIVE_VO_DHF[1],
                                                        NATIVE_VO_DHF[2]], dtype=torch.float32))
        nn.init.zeros_(self.delta_ef[-1].weight); nn.init.zeros_(self.delta_ef[-1].bias)
        # (c) sign-constrained measurement response (shared monotone shape + per-element signed gate)
        self.mono = _MonotoneScalar(hidden=16)
        self.g_elem = nn.Embedding(n_elem, 1)
        nn.init.zeros_(self.g_elem.weight)
        self.w_phys = nn.Parameter(torch.zeros(1))
        # (d) heteroscedastic log sigma head
        self.sigma_head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.sigma_head[-1].weight); nn.init.zeros_(self.sigma_head[-1].bias)
        self.register_buffer("bulk_center", torch.tensor(bulk_center))
        self.register_buffer("bulk_scale", torch.tensor(bulk_scale))
        self.register_buffer("logc_center", torch.tensor(logc_center))
        self.register_buffer("logc_scale", torch.tensor(logc_scale))

    def dHf(self, feats: torch.Tensor) -> torch.Tensor:
        return self.dHf_base.view(1, -1) + self.ef_swing * torch.tanh(self.delta_ef(feats))

    def forward(self, feats: torch.Tensor, T_K: torch.Tensor, log_pO2: torch.Tensor,
                z_dop: torch.Tensor, c_frac: torch.Tensor, elem_idx: torch.Tensor,
                e_level: torch.Tensor, dop_mask: torch.Tensor | None = None) -> dict:
        """feats [B,F]; T_K,log_pO2 [B]; z_dop,c_frac,elem_idx,e_level [B,D]; dop_mask [B,D] or None."""
        feats = feats.detach()                                   # Rule 1
        if z_dop.dim() == 1:                                     # tolerate [B] -> [B,1]
            z_dop = z_dop.view(-1, 1); c_frac = c_frac.view(-1, 1)
            elem_idx = elem_idx.view(-1, 1); e_level = e_level.view(-1, 1)
        if dop_mask is None:
            dop_mask = torch.ones_like(z_dop, dtype=feats.dtype)
        dop_mask = dop_mask.to(feats.dtype)
        B, D = z_dop.shape

        dH = self.dHf(feats)                                     # [B,3]
        sol = solve_equilibrium_codoping(dH, T_K, log_pO2, z_dop, c_frac, e_level, dop_mask=dop_mask)
        bulk = sol["log10_VO"]                                   # [B] the MECHANISM
        z_bulk = (bulk - self.bulk_center) / self.bulk_scale

        log_c = torch.log10(c_frac.clamp_min(1e-6) / C_REF)      # [B,D]
        log_c_std = (log_c - self.logc_center) / self.logc_scale # [B,D]
        if self.sign_mode == "constrained":
            shape = self.mono(log_c_std.reshape(-1)).reshape(B, D)        # [B,D] monotone↑ in log c
            g = self.g_elem(elem_idx).squeeze(-1)                         # [B,D] signed per-element scale
            r_dop = (dop_mask * g * shape).sum(dim=1)                     # SUM over co-dopants [B]
        else:                                                            # free | nonmono ablations
            emb = self.elem_emb(elem_idx)                                # [B,D,emb]
            inp = torch.cat([log_c_std.unsqueeze(-1), emb], dim=-1)      # [B,D,1+emb]
            r_each = self.free_head(inp).squeeze(-1)                      # [B,D]
            r_dop = (dop_mask * r_each).sum(dim=1)                        # [B]
        bulk_term = self.w_phys * z_bulk if self.use_solver else torch.zeros_like(r_dop)
        r = bulk_term + r_dop                                            # [B]
        log_sigma = self.sigma_head(feats).squeeze(-1)
        return dict(bulk_logVO=bulk, E_F=sol["E_F"], residual=sol["residual"],
                    r=r, log_sigma=log_sigma, dHf=dH, dop_ion=sol["dop_ion"])
