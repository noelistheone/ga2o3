"""V58 — M4: DFT physics stream.

Design ref: phase58_v58_dft_llm_hybrid_design.md §3.4.

Takes the 16-d cached `dft_features` vector (per dopant×atmosphere×T, built by
build_dft_cache.py from QE ΔE_f + Reuter-Scheffler μ_O + competing-phase μ_M) and
applies a TINY MLP [16 → 24 → 16] to produce a 16-d fusion feature φ_DFT.

H1 (manifold-dim) compliance: this stream outputs a FUSION FEATURE ONLY. It has NO
prediction head and is NEVER a target of any loss. This is the explicit contrast with
the failed V57-A1 `dft_scan_expert`, which had its own prediction head that competed
with the V_O loss (doc §3.4 NOTE lines 293-296).

Missing-DFT handling: rows whose dopant has no QE anchor carry a NaN/zero feature
vector; a learnable `missing_token` (16-d) replaces all-NaN rows so the stream is
robust, and a `dft_valid` mask is returned for the BrouwerHead to switch to its
learned-ΔE_f fallback (ablation A5 / §12.2). The MLP itself never sees NaN.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DFTStream(nn.Module):
    def __init__(self, feature_dim: int = 16, hidden: int = 24, out_dim: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        # Learnable replacement for all-NaN (no-DFT) rows; keeps the MLP NaN-free.
        self.missing_token = nn.Parameter(torch.zeros(feature_dim))

    # Dims that gate the M9 DFT-anchor switch: the V_O ΔE_f^{0,+1,+2} (0:3). A row is
    # "DFT-anchored" iff these are present, EVEN IF other dims (e.g. M_Ga dims 3-4, which
    # we have no QE substitution calc for) are NaN. Those secondary dims are imputed to 0
    # for the φ_DFT fusion feature. Using "any NaN over all 16 dims" here would wrongly
    # force every covered-dopant row into the learned fallback (M_Ga is always NaN). See
    # design-doc §3.4/§3.9 + Audit 3 follow-up.
    ANCHOR_DIMS = slice(0, 3)

    @classmethod
    def valid_mask(cls, dft_features: torch.Tensor) -> torch.Tensor:
        """[B] bool: True iff the V_O ΔE_f anchor dims (0:3) are all non-NaN."""
        return ~torch.isnan(dft_features[..., cls.ANCHOR_DIMS]).any(dim=-1)

    def forward(self, dft_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """dft_features: [B, 16] (cached, requires_grad=False). May contain NaN rows
        (fully-NaN for un-covered dopants; partially-NaN, e.g. M_Ga dims 3-4, for
        covered dopants).

        Returns (phi_DFT [B, out_dim], valid_mask [B]).
        """
        valid = self.valid_mask(dft_features)          # gates M9 anchor (dims 0:3)
        fully_missing = torch.isnan(dft_features).all(dim=-1)
        x = dft_features
        if fully_missing.any():
            # replace ALL-NaN rows (un-covered dopant) with the learnable token
            x = torch.where(fully_missing.unsqueeze(-1),
                            self.missing_token.to(x.dtype).expand_as(x), x)
        # impute any remaining per-dim NaN (e.g. M_Ga dims 3-4) to 0 before the MLP
        x = torch.nan_to_num(x, nan=0.0)
        return self.net(x), valid


if __name__ == "__main__":
    torch.manual_seed(0)
    m = DFTStream()
    n_train = sum(p.numel() for p in m.parameters())
    print(f"DFTStream params = {n_train}")  # expect ~1.2k + 16 token
    x = torch.randn(5, 16)
    x[2] = float("nan")       # fully-NaN row (un-covered dopant) -> invalid + token
    x[4, 3:5] = float("nan")  # covered dopant w/ M_Ga (dims 3-4) NaN -> still VALID, imputed
    out, valid = m(x)
    print("out shape", tuple(out.shape), "valid", valid.tolist())
    assert out.shape == (5, 16)
    # row 2 fully NaN -> invalid; row 4 only dims 3-4 NaN -> valid (anchor dims 0:3 present)
    assert valid.tolist() == [True, True, False, True, True], valid.tolist()
    assert not torch.isnan(out).any(), "NaN leaked through DFTStream"
    print("DFTStream OK (anchor-dim validity + partial-NaN imputation)")
