"""V57-B1-Quartet+1 — Loader for the 4th frozen DFT-scan expert.

Loads `checkpoints/encoder_dft_scan.pt` (trained by scripts/35_v57_quartet_plus_pretrain.py).
Returns a `FrozenEncoderDFTScan` whose forward takes
  (element_onehot, log_c, log_T, log_pO2, q_onehot)
and returns predicted log10[V_O^q] in real (un-standardized) units.

Same Phase 54 V54-B1 pattern as `load_charge_vo_expert` / `load_vga_expert`:
  - frozen at inference (`requires_grad_(False) + .eval()`)
  - distilled via MSE on a small `aux_distill_dft_scan_head(fused)` Linear
  - Rule 1 compliant — `detach()` enforced inside Ga2O3Net.dft_scan_predict()
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class FrozenEncoderDFTScan(nn.Module):
    """Replays the architecture from scripts/35_v57_quartet_plus_pretrain.py.

    Input layout (matches that script's `df_to_tensors`):
      - elem_onehot: [B, n_elements]
      - log_c:       [B] log10(dopant_atomic_fraction)
      - log_T:       [B] log10(T_K)
      - log_pO2:     [B] log10(O2 partial pressure / atm)
      - q_onehot:    [B, n_charges]
    """

    def __init__(self, state: dict):
        super().__init__()
        elements = list(state["elements"])
        n_elements = len(elements)
        n_charges = int(state.get("n_charges", 3))
        hidden = int(state.get("hidden", 96))
        in_dim = n_elements + 3 + n_charges
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Load weights — saved keys prefixed with "net." in pretrain script
        net_state = {k.replace("net.", ""): v for k, v in state["model_state"].items()
                      if k.startswith("net.")}
        self.net.load_state_dict(net_state)

        self.elements = elements
        self.elem_to_idx = {e: i for i, e in enumerate(elements)}
        self.n_elements = n_elements
        self.n_charges = n_charges
        cm = torch.tensor(state["cont_mean"], dtype=torch.float32)
        cs = torch.tensor(state["cont_std"],  dtype=torch.float32)
        self.register_buffer("cont_mean", cm)
        self.register_buffer("cont_std",  cs)
        self.target_mean = float(state["target_mean"])
        self.target_std  = float(state["target_std"])

    def forward(
        self,
        elem_onehot: torch.Tensor,
        log_c: torch.Tensor,
        log_T: torch.Tensor,
        log_pO2: torch.Tensor,
        q_onehot: torch.Tensor,
    ) -> torch.Tensor:
        cont = torch.stack([log_c, log_T, log_pO2], dim=-1)
        cont_norm = (cont - self.cont_mean) / self.cont_std
        x = torch.cat([elem_onehot.float(), cont_norm, q_onehot.float()], dim=-1)
        pred_norm = self.net(x).squeeze(-1)
        return pred_norm * self.target_std + self.target_mean


def load_frozen_dft_scan(path: str | Path) -> FrozenEncoderDFTScan:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    return FrozenEncoderDFTScan(state)
