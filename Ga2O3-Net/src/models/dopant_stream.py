"""Phase 5G / 6A: explicit per-dopant identity channel + chemistry prior.

Architecture
============
DopantStream is the 4th fusion stream (alongside structure, composition,
process). For each sample it:

1. Parses the dopant_spec string via DopantSpec.
2. Looks up a learned embedding per dopant component element, weight-sums them
   by fractional concentration: e = Σ_i emb(elem_i) * frac_i / Σ_i frac_i.
   Undoped samples route to index 0.
3. (Phase 6A; Phase 46 V11 extended to 5-dim) Looks up a 5-vector of physical
   descriptors (electronegativity, Shannon ionic radius, carrier-type, Δv,
   Δperiod) per element and weight-sums the same way — gives the MLP a
   continuous chemistry axis so rare/unseen dopants extrapolate from physical
   similarity, not just class-embedding lookup. The Δv channel separates
   super-donors (Sb/Ta/Bi/W, Δv=+2) from regular donors (Δv=+1) explicitly.
4. Concatenates the weight-summed embedding + descriptors with
   [log10(total_conc), is_undoped_flag].
5. Passes through a small MLP to produce the `out_dim` output vector fused with
   the other streams.

Why this exists
---------------
The existing CompositionStream consumes the full doped-crystal XenonPy
features (290-dim), which are weighted by the full formula. With 99% Ga
weight, dopant identity (Sn vs Mg vs Zn) enters only as a perturbation
to elemental property averages. On the external set this encoding fails
for Sn (Pearson r ≈ −0.87). DopantStream gives the fusion head a dedicated
categorical channel whose signal is orthogonal to XenonPy and concentration.

Phase 6A family-group cos regularizer
-------------------------------------
`family_groups` is a fixed buffer mapping each dopant index → family-group
integer (II/III/IV/V/d-block/H/other). The fine-tune trainer pulls same-family
class embeddings together via a small cos-similarity penalty — rare dopants
(Cu, Ge, Bi) thereby borrow representation from abundant same-family ones
(Fe, Sn, …). See `src/training/finetune_trainer.py::family_cos_regularizer`.

Dopant class enumeration
------------------------
13 classes covering ≥95% of PDR-labeled rows. "other" is the catch-all for
unseen elements so the module never raises on new dopants.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from src.data.dopant_spec import DopantSpec
from src.data.element_descriptors import (
    ELEMENT_DESCRIPTORS,
    FAMILY_GROUPS,
    FAMILY_OTHER_IDX,
    DESCRIPTOR_DIM,
)

logger = logging.getLogger(__name__)

# Order matters: index 0 = undoped; rest ordered by training frequency.
ELEMENT_TO_IDX: dict[str, int] = {
    "undoped": 0,
    "Sn": 1,
    "Mg": 2,
    "Zn": 3,
    "Si": 4,
    "N":  5,
    "Ta": 6,
    "H":  7,
    "In": 8,
    "Al": 9,
    "Zr": 10,
    "Fe": 11,
    # 12 = "other" catch-all
}
N_DOPANTS_DEFAULT = 13
OTHER_IDX = 12


def _build_descriptor_and_family_tables(n_dopants: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (descriptors [n_dopants, DESCRIPTOR_DIM], family_groups [n_dopants])."""
    desc = torch.zeros(n_dopants, DESCRIPTOR_DIM, dtype=torch.float32)
    fam = torch.full((n_dopants,), FAMILY_OTHER_IDX, dtype=torch.long)
    # Reverse ELEMENT_TO_IDX
    idx_to_elem = {i: e for e, i in ELEMENT_TO_IDX.items()}
    other_desc = torch.tensor(ELEMENT_DESCRIPTORS["other"], dtype=torch.float32)
    for i in range(n_dopants):
        elem = idx_to_elem.get(i, "other")
        d = ELEMENT_DESCRIPTORS.get(elem, ELEMENT_DESCRIPTORS["other"])
        desc[i] = torch.tensor(d, dtype=torch.float32)
        fam[i] = FAMILY_GROUPS.get(elem, FAMILY_OTHER_IDX)
    # OTHER_IDX may exceed ELEMENT_TO_IDX range; ensure fallback.
    if OTHER_IDX < n_dopants and idx_to_elem.get(OTHER_IDX, "other") == "other":
        desc[OTHER_IDX] = other_desc
        fam[OTHER_IDX] = FAMILY_OTHER_IDX
    return desc, fam


class DopantStream(nn.Module):
    """Learned per-dopant embedding + physical descriptors + log10(c) context → out_dim vector."""

    def __init__(
        self,
        n_dopants: int = N_DOPANTS_DEFAULT,
        embed_dim: int = 8,
        hidden_dim: int = 32,
        out_dim: int = 16,
        dropout: float = 0.2,
        use_descriptors: bool = True,
    ):
        super().__init__()
        self.n_dopants = n_dopants
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_descriptors = use_descriptors

        self.element_embedding = nn.Embedding(n_dopants, embed_dim)
        nn.init.normal_(self.element_embedding.weight, std=0.1)

        # Fixed per-element physical descriptors + family-group ids.
        desc, fam = _build_descriptor_and_family_tables(n_dopants)
        self.register_buffer("element_descriptors", desc)   # [n_dopants, DESCRIPTOR_DIM=5]
        self.register_buffer("family_groups", fam)          # [n_dopants]

        desc_dim = DESCRIPTOR_DIM if use_descriptors else 0
        # MLP input: [weight-summed embedding | weight-summed descriptors? | log10(c), is_undoped]
        mlp_in = embed_dim + desc_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode_one(self, spec_str: str, device: torch.device) -> torch.Tensor:
        """Return a single [embed_dim + (DESCRIPTOR_DIM if use_descriptors) + 2] feature tensor."""
        spec = DopantSpec.parse(spec_str.strip())

        if spec.is_undoped:
            idx = ELEMENT_TO_IDX["undoped"]
            emb = self.element_embedding(
                torch.tensor(idx, device=device)
            )
            parts = [emb]
            if self.use_descriptors:
                parts.append(self.element_descriptors[idx].to(device))
            parts.append(torch.tensor([0.0, 1.0], device=device))   # log_c=0, is_undoped=1
            return torch.cat(parts)

        total = spec.total_conc
        if total <= 0:
            total = 1e-4
        log_c = math.log10(max(total, 1e-4))

        weighted_emb = torch.zeros(self.embed_dim, device=device)
        weighted_desc = torch.zeros(DESCRIPTOR_DIM, device=device)
        weight_sum = 0.0
        for comp in spec.components:
            cation = comp.cation
            idx = ELEMENT_TO_IDX.get(cation, OTHER_IDX)
            idx_t = torch.tensor(idx, device=device)
            weighted_emb = weighted_emb + self.element_embedding(idx_t) * comp.conc
            weighted_desc = weighted_desc + self.element_descriptors[idx].to(device) * comp.conc
            weight_sum += comp.conc
        if weight_sum > 0:
            weighted_emb = weighted_emb / weight_sum
            weighted_desc = weighted_desc / weight_sum

        parts = [weighted_emb]
        if self.use_descriptors:
            parts.append(weighted_desc)
        parts.append(torch.tensor([log_c, 0.0], device=device))
        return torch.cat(parts)

    def forward(self, dopant_specs: list[str]) -> torch.Tensor:
        """[B] list of spec strings → [B, out_dim] tensor."""
        device = next(self.parameters()).device
        feats = torch.stack([self._encode_one(s, device) for s in dopant_specs])
        return self.mlp(feats)
