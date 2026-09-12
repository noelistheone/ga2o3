"""
M7: Cross-attention bridge (LLMBridge).

Fuses the Process-LLM embedding ``z_proto`` (M5) and the Physics-LLM
embedding ``z_phys`` (M6) into the final LLM sidechannel embedding
``z_LLM`` used by the V58 FiLM fusion.

Architecture (verbatim from design doc §3.7.1):
  - 2 cross-attention layers, dim=32, n_heads=4, dropout=0.1
  - per-layer FFN(32 → 4*32=128 → 32) with GELU + dropout
  - pre/post LayerNorm residual structure
  - ``z_proto`` is the query, ``z_phys`` is the key/value
  - forward(z_proto[B,32], z_phys[B,32]) -> z_llm[B,32]

Training (§3.7.2): bridge is trained in **Stage 2** alongside the
Process-LLM LoRA. Per the H2 parameter-budget constraint (§1.4), the
bridge is FROZEN for Stage 3 downstream regression — see
``freeze_for_stage3()``.

Param count is ≈25k (design doc §3.7.1); the self-test asserts this.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LLMBridge(nn.Module):
    """
    Cross-attention bridge fusing ``z_proto`` (query) with ``z_phys`` (k/v).

    Args:
        dim     : embedding dimension of both inputs and the output (32).
        n_heads : number of attention heads (4).
        n_layers: number of stacked cross-attention + FFN blocks (2).
        dropout : dropout used in attention and FFN (0.1).
    """

    def __init__(
        self,
        dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim, num_heads=n_heads,
                batch_first=True, dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4 * dim), nn.GELU(),
                nn.Linear(4 * dim, dim), nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

    def forward(self, z_proto: torch.Tensor, z_phys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_proto: [B, dim] Process-LLM embedding (used as query).
            z_phys : [B, dim] Physics-LLM embedding (used as key/value).

        Returns:
            z_llm: [B, dim] fused LLM embedding.
        """
        # treat each [B, dim] vector as a length-1 sequence
        q = z_proto.unsqueeze(1)         # [B, 1, dim]
        k = v = z_phys.unsqueeze(1)      # [B, 1, dim]
        for ca, ffn, n1, n2 in zip(
            self.cross_attn_layers, self.ffn_layers, self.norm1, self.norm2
        ):
            attn_out, _ = ca(q, k, v)
            q = n1(q + attn_out)
            ffn_out = ffn(q)
            q = n2(q + ffn_out)
        return q.squeeze(1)              # [B, dim]

    # ── Stage-3 freezing (H2 parameter-budget constraint, §1.4) ───────────────

    def n_trainable(self) -> int:
        """Number of trainable (requires_grad=True) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_for_stage3(self) -> None:
        """
        Freeze ALL bridge params and switch to eval mode for Stage 3.

        After this call ``n_trainable()`` must be 0: the bridge is trained
        only in Stage 2 (§3.7.2) and contributes no trainable params to the
        downstream V_O/PDR regression (§1.4 H2).
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


def _self_test() -> None:
    """CPU self-test: shapes + param count, runs in well under a second."""
    torch.manual_seed(0)
    bridge = LLMBridge(dim=32, n_heads=4, n_layers=2, dropout=0.1)

    total = sum(p.numel() for p in bridge.parameters())
    trainable = bridge.n_trainable()
    print(f"[LLMBridge self-test]")
    print(f"  total params     : {total:,}")
    print(f"  trainable params : {trainable:,}")
    # Per-layer breakdown for transparency.
    attn = sum(p.numel() for p in bridge.cross_attn_layers.parameters())
    ffn = sum(p.numel() for p in bridge.ffn_layers.parameters())
    ln = (sum(p.numel() for p in bridge.norm1.parameters())
          + sum(p.numel() for p in bridge.norm2.parameters()))
    print(f"    cross-attn     : {attn:,}")
    print(f"    ffn            : {ffn:,}")
    print(f"    layernorm      : {ln:,}")

    # Doc §3.7.1 estimates ≈25k; the exact value includes MHA in/out-proj
    # biases + LayerNorm params, so allow a generous band around 25k.
    assert 20_000 <= total <= 35_000, f"param count {total} far from ~25k"

    B = 5
    z_proto = torch.randn(B, 32)
    z_phys = torch.randn(B, 32)
    z_llm = bridge(z_proto, z_phys)
    assert z_llm.shape == (B, 32), f"bad output shape {z_llm.shape}"
    assert torch.isfinite(z_llm).all(), "non-finite output"
    print(f"  forward output   : {tuple(z_llm.shape)} (OK)")

    # gradient flows to all params
    bridge.train()
    loss = bridge(z_proto, z_phys).pow(2).sum()
    loss.backward()
    n_with_grad = sum(
        1 for p in bridge.parameters() if p.requires_grad and p.grad is not None
    )
    print(f"  params with grad : {n_with_grad}")

    # freeze_for_stage3 drops trainable to 0
    bridge.freeze_for_stage3()
    assert bridge.n_trainable() == 0, "freeze_for_stage3 left trainable params"
    assert not bridge.training, "freeze_for_stage3 did not switch to eval()"
    print(f"  after freeze     : trainable={bridge.n_trainable()} eval={not bridge.training}")
    print("[LLMBridge self-test] PASS")


if __name__ == "__main__":
    _self_test()
