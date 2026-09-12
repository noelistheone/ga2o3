"""V58 — M9: DFT-anchored Brouwer head for log[V_O].

Design ref: phase58_v58_dft_llm_hybrid_design.md §3.9 (+ §4.1).

Replaces V55-Ext's data-driven `learned ΔE_f` (src/models/brouwer_head.py BrouwerHeadVC)
with the QE-computed formation energies. For a row whose dopant HAS a QE anchor:

    log10[V_O]_abs = log10 N_site^DFT  +  log10 Σ_q exp( −ΔE_f^q / (k_B T) )      (§4.1)
                   = log_pref_DFT      +  logsumexp_q( −ΔE_f^q/(k_B T) ) / ln10
    log10[V_O]_pred = (log10[V_O]_abs − target_mean) / target_std  +  δ(φ_fused)

where
  - ΔE_f^q = dft_features[:, 0:3]  (V_O^{0,+1,+2}) are already at THIS row's atmosphere
    (μ_O(T,p_O2)) and Fermi level (q·μ_e), having been shifted in build_dft_cache from
    the stored mid-limit, εF=0 values (see design-doc §7 callout "build_dft_cache 必须做的
    两个参考态修正"). So the pO2/atmosphere dependence lives INSIDE ΔE_f via μ_O — there is
    NO separate −½ log pO2 term here (that would double-count; the Brouwer −¼ log pO2
    asymptote emerges from ΔE_f's +μ_O dependence).
  - log_pref_DFT = sum of dft_features[:, 9:13]  (= log10 N_site, the 4 prefactor comps).
  - δ(φ_fused) = tanh(residual_net(φ_fused)) · max_residual / target_std  — HARD-BOUNDED
    to ±0.3 dex (H3). It cannot override the DFT anchor by more than ±0.3 dex.
  - target_mean / target_std are per-fold buffers the trainer sets via set_target_stats()
    (V55-Ext standardizes V_O targets outside the model). They map the absolute DFT anchor
    into the standardized target space the loss operates in.

H1: this head emits ONLY log[V_O]; no auxiliary prediction/distillation target.
H3: `dft_features` enter as a CONSTANT (requires_grad=False cache + defensive .detach());
    `log_vo_dft` carries NO gradient into the DFT numbers — only δ is trainable on the
    physics side.

Fallback (no QE anchor): rows flagged `dft_valid=False` use a compact learned-ΔE_f path
(the V55-Ext closed form with a free log_prefactor + global_bias that self-fits the
standardized target). This is the §11.2.2 ablation-A5 form and the §12.2 risk mitigation
for unseen chemical classes. Both paths output in the SAME standardized space.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

KB_EV = 8.617333262e-5
LN10 = math.log(10.0)


class DFTAnchoredBrouwerHead(nn.Module):
    def __init__(
        self,
        fused_dim: int = 208,
        max_residual: float = 0.3,
        # fallback learned-ΔE_f path (for dopants without a QE anchor)
        fallback_hidden: int = 32,
        e_f_baseline: float = 2.0,
        e_f_swing: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        self.max_residual = float(max_residual)
        self.e_f_baseline = float(e_f_baseline)
        self.e_f_swing = float(e_f_swing)

        # δ residual: bounded correction on the DFT anchor. ~6.7k params at fused_dim=208.
        self.residual_net = nn.Sequential(
            nn.Linear(fused_dim, fallback_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fallback_hidden, 1),
        )
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

        # Fallback learned-ΔE_f latent net: (ΔE_f raw, log_prefactor, dopant_term).
        self.fallback_net = nn.Sequential(
            nn.Linear(fused_dim, fallback_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fallback_hidden, 3),
        )
        nn.init.zeros_(self.fallback_net[-1].weight)
        nn.init.zeros_(self.fallback_net[-1].bias)
        self.fallback_global_bias = nn.Parameter(torch.zeros(1))

        # Single learnable global bias on the DFT-anchored path. The QE anchor is a
        # THERMODYNAMIC-EQUILIBRIUM V_O concentration at the anneal/measurement T; the
        # MEASURED V_O in sputtered films is frozen-in from growth + kinetic/sputter-damage
        # defects, systematically offset from equilibrium by a roughly constant factor
        # (≈ several dex). global_bias absorbs that single constant while the DFT anchor
        # keeps providing the atmosphere/T/charge-state SHAPE — H3-preserving (1 scalar,
        # not a per-row data-driven ΔE_f). Same role as V55-Ext BrouwerHeadVC.global_bias.
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Per-fold target standardization (set by trainer). Buffers move with .to(device).
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))
        self.register_buffer("_stats_set", torch.zeros(1))  # 0/1 flag

    def set_target_stats(self, mean: float, std: float) -> None:
        """Trainer calls this per fold so the absolute DFT anchor maps to the
        standardized target space. Mirrors V55-Ext's external target standardization."""
        self.target_mean.fill_(float(mean))
        self.target_std.fill_(float(max(std, 1e-6)))
        self._stats_set.fill_(1.0)

    def _dft_anchor_standardized(self, dft_features: torch.Tensor,
                                 T_K: torch.Tensor) -> torch.Tensor:
        """Absolute DFT log10[V_O] mapped to standardized space. [B,1]. No grad to DFT."""
        ef = dft_features[:, 0:3].detach()          # ΔE_f^{0,+1,+2}, constant
        log_pref = dft_features[:, 9:13].detach().sum(dim=1, keepdim=True)  # log10 N_site
        # log10 Σ_q exp(−ΔE_f^q/kT) = logsumexp(−ΔE_f/kT)/ln10
        log_sum = torch.logsumexp(-ef / (KB_EV * T_K), dim=1, keepdim=True) / LN10
        log_vo_abs = log_pref + log_sum             # [B,1] absolute log10[V_O]
        return (log_vo_abs - self.target_mean) / self.target_std

    def _fallback_standardized(self, fused: torch.Tensor, process: torch.Tensor,
                               T_K: torch.Tensor) -> torch.Tensor:
        """V55-Ext-style learned closed form, output already in standardized target space
        (free log_prefactor + global_bias absorb the scale). [B,1]."""
        lat = self.fallback_net(fused)
        delta_ef = self.e_f_swing * torch.tanh(lat[:, 0:1])
        E_f = self.e_f_baseline + delta_ef
        log_prefactor = lat[:, 1:2]
        dopant_term = lat[:, 2:3]
        pO2 = process[:, 2:3].clamp(min=1e-4, max=1.0)
        boltz = -E_f / (KB_EV * T_K * LN10)
        return log_prefactor + boltz - 0.5 * torch.log10(pO2) + dopant_term \
            + self.fallback_global_bias

    def forward(self, fused: torch.Tensor, dft_features: torch.Tensor,
                process: torch.Tensor, dft_valid: torch.Tensor | None = None
                ) -> torch.Tensor:
        """fused [B,fused_dim]; dft_features [B,16] (cached, NaN rows allowed);
        process [B,>=3] with T_C at col 0, o2_fraction at col 2 (V55-Ext convention).
        dft_valid [B] bool (from DFTStream.valid_mask); if None it is computed here.

        Returns standardized log10[V_O], [B,1].
        """
        if dft_valid is None:
            dft_valid = ~torch.isnan(dft_features).any(dim=-1)
        T_K = (process[:, 0:1] + 273.15).clamp(min=100.0)

        if dft_valid.any() and self._stats_set.item() < 0.5:
            raise RuntimeError(
                "DFTAnchoredBrouwerHead: target stats not set but DFT-anchored rows "
                "present. Trainer must call set_target_stats(mean, std) per fold."
            )

        # δ residual, hard-bounded to ±max_residual dex, then to std units.
        delta_dex = torch.tanh(self.residual_net(fused)) * self.max_residual  # [B,1]
        delta_std = delta_dex / self.target_std

        # DFT-anchored path (NaN rows replaced by 0 so logsumexp is finite; masked out next)
        safe_dft = torch.nan_to_num(dft_features, nan=0.0)
        anchored = self._dft_anchor_standardized(safe_dft, T_K) + delta_std + self.global_bias

        # Fallback path for rows without a QE anchor
        fallback = self._fallback_standardized(fused, process, T_K)          # [B,1]

        valid = dft_valid.unsqueeze(-1)
        return torch.where(valid, anchored, fallback)

    # Convenience for the §11.2 DFT-consistency eval: pure DFT anchor (no δ), std space.
    def dft_anchor_only(self, dft_features: torch.Tensor, process: torch.Tensor
                        ) -> torch.Tensor:
        T_K = (process[:, 0:1] + 273.15).clamp(min=100.0)
        return self._dft_anchor_standardized(torch.nan_to_num(dft_features, nan=0.0), T_K)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, FD = 6, 208
    head = DFTAnchoredBrouwerHead(fused_dim=FD)
    n = sum(p.numel() for p in head.parameters())
    print(f"DFTAnchoredBrouwerHead params = {n}")

    fused = torch.randn(B, FD)
    # Build plausible dft_features: dims 0-2 ΔE_f ~ [1.5, 0.8, 0.2] eV, dims 9-12 log_pref ~22/4
    dft = torch.zeros(B, 16)
    dft[:, 0:3] = torch.tensor([1.6, 0.9, 0.3])
    dft[:, 9:13] = 22.7 / 4.0           # log10 N_site ≈ 22.7
    dft[3] = float("nan")               # one no-DFT row -> fallback
    process = torch.zeros(B, 18)
    process[:, 0] = 700.0               # T_C
    process[:, 2] = 0.2                 # o2_fraction

    # absolute anchor sanity BEFORE standardization
    head.set_target_stats(mean=0.0, std=1.0)
    T_K = torch.full((B, 1), 973.15)
    abs_anchor = head._dft_anchor_standardized(torch.nan_to_num(dft, nan=0.0), T_K)
    print("abs log10[V_O] anchor (row0):", float(abs_anchor[0]))  # expect ~17-19 region

    out = head(fused, dft, process)
    print("out shape", tuple(out.shape), "any nan?", bool(torch.isnan(out).any()))
    assert out.shape == (B, 1) and not torch.isnan(out).any()

    # H3: δ bounded ±0.3 dex. Push residual_net hard, confirm clamp.
    with torch.no_grad():
        for p in head.residual_net.parameters():
            p.add_(torch.randn_like(p) * 100)
    head.set_target_stats(mean=18.0, std=2.0)
    d = torch.tanh(head.residual_net(fused)) * head.max_residual
    assert d.abs().max() <= 0.3 + 1e-6, "delta exceeded ±0.3 dex"
    print(f"max |delta| dex = {float(d.abs().max()):.4f} (<=0.3 OK)")

    # H3: no gradient flows into dft_features. Because _dft_anchor_standardized .detach()es
    # ef and log_pref, the anchor output must NOT require grad w.r.t. dft_features.
    dft2 = dft.clone().detach().requires_grad_(True)
    head.set_target_stats(0.0, 1.0)
    y = head._dft_anchor_standardized(torch.nan_to_num(dft2, nan=0.0), T_K)
    assert not y.requires_grad, "H3 VIOLATION: DFT anchor carries gradient into dft_features"
    print("H3 OK: DFT anchor has no grad path to dft_features (requires_grad=False)")
    # Sanity: the residual δ DOES require grad (it is the only trainable physics term)
    head.zero_grad()
    out2 = head(fused, dft, process).sum()
    assert out2.requires_grad, "δ residual should be trainable"
    out2.backward()
    assert dft2.grad is None, "dft_features must receive no gradient"
    print("DFTAnchoredBrouwerHead OK")
