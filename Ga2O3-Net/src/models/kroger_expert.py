"""Phase 53 Stage 2 — Loader for frozen encoder_kroger expert.

Loads the EncoderKroger MLP from `checkpoints/encoder_kroger.pt`. Returns a
wrapped frozen model that takes batch (process_tensor, dopant_specs) and
returns predicted log10[V_O].
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from src.data.kroger_synthetic import ELEMENTS, C_REF


class FrozenEncoderKroger(nn.Module):
    """Wraps the trained EncoderKroger MLP. Takes (process, dopant_specs, total_conc)
    and returns predicted log10[V_O] in real units (not standardized).

    Process tensor layout (17-dim, see the project docs):
      idx 0 = temperature_C
      idx 2 = o2_fraction (0–1, can map to log_pO2 ≈ log10(o2_fraction))
    """

    def __init__(self, state: dict):
        super().__init__()
        # Reconstruct the same architecture as scripts/02f_pretrain_kroger.py::EncoderKroger
        n_elements = len(state["elements"])
        hidden = 64
        in_dim = n_elements + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4), nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )
        self.net.load_state_dict({
            k.replace("net.", ""): v for k, v in state["model_state"].items()
            if k.startswith("net.")
        })
        self.elements = state["elements"]
        self.elem_to_idx = {e: i for i, e in enumerate(self.elements)}
        self.n_elements = n_elements
        self.register_buffer("cont_mean", torch.tensor(state["cont_mean"], dtype=torch.float32))
        self.register_buffer("cont_std", torch.tensor(state["cont_std"], dtype=torch.float32))
        self.target_mean = float(state["target_mean"])
        self.target_std = float(state["target_std"])

    def _parse_dopant_to_idx(self, spec: str) -> int:
        """Find first known element in dopant spec; default to first element if unknown."""
        from src.data.dopant_spec import DopantSpec
        try:
            ds = DopantSpec.parse(spec)
            if ds.is_undoped or not ds.components:
                return 0  # Default to first element (Mg)
            cation = max(ds.components, key=lambda c: c.conc).cation
            return self.elem_to_idx.get(cation, 0)
        except Exception:
            return 0

    def _total_conc(self, spec: str) -> float:
        """Return total atomic-fraction dopant concentration."""
        from src.data.dopant_spec import DopantSpec
        try:
            ds = DopantSpec.parse(spec)
            if ds.is_undoped or not ds.components:
                return float(C_REF)  # default 1 at% so log_c=0
            tot = sum(max(float(c.conc), 0.0) for c in ds.components)
            return max(tot, 1e-6)
        except Exception:
            return float(C_REF)

    def forward(self, process: torch.Tensor, dopant_specs: list[str]) -> torch.Tensor:
        """
        Args:
            process: [B, 17+] process tensor (cols 0=T_C, 2=o2_fraction)
            dopant_specs: list of B dopant spec strings.
        Returns:
            [B] tensor of predicted log10[V_O] in real units.
        """
        device = process.device
        B = len(dopant_specs)

        # Build features: log10(c), log10(T_K), log10(p_O2)
        elem_idx = torch.tensor([self._parse_dopant_to_idx(s) for s in dopant_specs],
                                dtype=torch.long, device=device)
        c_total = torch.tensor([self._total_conc(s) for s in dopant_specs],
                               dtype=torch.float32, device=device)
        log_c = torch.log10(c_total.clamp(min=1e-6))
        T_C = process[:, 0]
        T_K = (T_C + 273.15).clamp(min=100.0)
        log_T = torch.log10(T_K)
        # o2_fraction in [0, 1]; log10(p_O2) ≈ log10(o2_fraction). Clamp to avoid -inf.
        o2 = process[:, 2].clamp(min=1e-4, max=1.0)
        log_pO2 = torch.log10(o2)

        cont = torch.stack([log_c, log_T, log_pO2], dim=-1)  # [B, 3]
        cont_norm = (cont - self.cont_mean) / self.cont_std

        one_hot = torch.nn.functional.one_hot(elem_idx, self.n_elements).float()
        x = torch.cat([one_hot, cont_norm], dim=-1)
        pred_norm = self.net(x).squeeze(-1)  # [B]
        pred_real = pred_norm * self.target_std + self.target_mean
        return pred_real


def load_encoder_kroger(path: str) -> FrozenEncoderKroger:
    """Load encoder_kroger checkpoint and return frozen FrozenEncoderKroger."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return FrozenEncoderKroger(state)
