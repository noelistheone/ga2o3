"""V58 — M8: FiLM fusion at 208-d (widened from V55-Ext's 160-d).

Design ref: phase58_v58_dft_llm_hybrid_design.md §3.8 (+ §4.4).

Input is the concat of the five streams:
    [φ_struct(64), φ_comp(64), φ_proc(32), z_LLM(32), φ_DFT(16)] = 208-d
4-head self-attention over the (length-1) feature sequence predicts FiLM γ, β which
gate the concatenated vector:
    φ_fused = γ(attn(x)) ⊙ x + β(attn(x))     (eq §4.4)

Internal hidden_dim stays 128 (same as V55-Ext) to avoid overfitting; the only param
growth vs V55-Ext is from the 160→208 width (~+50k), which is the bulk of the H2
"≤50k new trainable params" budget (doc §3.8.2 line 542-545).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMFusion208(nn.Module):
    def __init__(self, input_dim: int = 208, hidden_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.gamma_proj = nn.Linear(hidden_dim, input_dim)
        self.beta_proj = nn.Linear(hidden_dim, input_dim)
        # Init γ→1, β→0 so fusion starts as identity (stable training).
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 208]
        x_seq = x.unsqueeze(1)  # [B, 1, 208]
        q = self.q_proj(x_seq)
        k = self.k_proj(x_seq)
        v = self.v_proj(x_seq)
        attn_out, _ = self.attn(q, k, v)          # [B, 1, hidden]
        a = attn_out.squeeze(1)                    # [B, hidden]
        gamma = self.gamma_proj(a)                 # [B, 208]
        beta = self.beta_proj(a)                   # [B, 208]
        return gamma * x + beta                    # [B, 208]


if __name__ == "__main__":
    torch.manual_seed(0)
    m = FiLMFusion208()
    n = sum(p.numel() for p in m.parameters())
    print(f"FiLMFusion208 params = {n}")
    x = torch.randn(7, 208)
    out = m(x)
    print("out shape", tuple(out.shape))
    assert out.shape == (7, 208)
    # identity init check: at init γ=1,β=0 BEFORE attn contribution -> output≈x only if
    # attn adds nothing; attn is random-init so not exact identity, but gamma/beta proj
    # zero-weight means γ=1,β=0 regardless of attn output -> output == x exactly.
    assert torch.allclose(out, x, atol=1e-5), "FiLM not identity at init"
    print("FiLMFusion208 OK (identity at init)")
