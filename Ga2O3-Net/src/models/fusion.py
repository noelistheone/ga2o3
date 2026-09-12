"""
Fusion layers that combine the three feature streams into a single vector.

Two variants:
  ConcatFusion         — simple concatenation, output dim = sum of inputs
  SelfAttentionFusion  — treat each stream as a token; apply multi-head
                         self-attention, then flatten to a fixed output dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """
    Trivial fusion: concatenate all stream embeddings along dim=-1.

    Input:  list of tensors, each [B, d_i]
    Output: Tensor [B, sum(d_i)]
    """

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        return torch.cat(streams, dim=-1)

    @staticmethod
    def output_dim(*dims: int) -> int:
        return sum(dims)


class FiLMFusion(nn.Module):
    """
    Phase 13A: FiLM-style element-conditioned process modulation.

    γ, β = MLP(comp_emb)
    proc' = (1 + γ) ⊙ proc + β
    out   = concat(struct, comp, [dopant,] proc')

    Stream order assumption (matches Ga2O3Net): comp is at index 1, proc is last.
    Output dim equals ConcatFusion (sum of stream dims) — drop-in replacement.

    Initialization: γ→0, β→0 → at init this collapses to ConcatFusion, so
    training starts from the same point as a concat baseline and only
    diverges when modulation actually helps. This stabilises training on
    small data (~170 samples).
    """

    def __init__(
        self,
        comp_dim: int,
        proc_dim: int,
        film_hidden: int = 64,
        film_dropout: float = 0.1,
    ):
        super().__init__()
        self.proc_dim = proc_dim
        self.film_net = nn.Sequential(
            nn.Linear(comp_dim, film_hidden),
            nn.GELU(),
            nn.Dropout(film_dropout),
            nn.Linear(film_hidden, 2 * proc_dim),
        )
        # Zero-init the final projection so γ=0, β=0 at start.
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        # streams = (struct, comp, [dopant,] proc)
        comp = streams[1]
        proc = streams[-1]
        gamma, beta = self.film_net(comp).chunk(2, dim=-1)
        proc_mod = (1.0 + gamma) * proc + beta
        # Replace last stream with modulated process embedding
        new_streams = streams[:-1] + (proc_mod,)
        return torch.cat(new_streams, dim=-1)

    @staticmethod
    def output_dim(*dims: int) -> int:
        return sum(dims)


class BiFiLMFusion(nn.Module):
    """
    Phase 14A: Bidirectional FiLM — comp ↔ proc cross-modulation.

    γ_p, β_p = MLP_p(comp)            proc' = (1 + γ_p) ⊙ proc + β_p
    γ_c, β_c = MLP_c(proc)            comp' = (1 + γ_c) ⊙ comp + β_c
    out = concat(struct, comp', [dopant,] proc')

    Both directions zero-init so init = ConcatFusion. ~16k extra params total.
    Allows process to also condition the composition embedding (e.g. a deposition
    method might emphasise different element-property channels), which 13A's
    one-way FiLM couldn't do.
    """

    def __init__(
        self,
        comp_dim: int,
        proc_dim: int,
        film_hidden: int = 64,
        film_dropout: float = 0.1,
    ):
        super().__init__()
        self.proc_dim = proc_dim
        self.comp_dim = comp_dim
        self.film_p = nn.Sequential(           # comp → modulate proc
            nn.Linear(comp_dim, film_hidden), nn.GELU(),
            nn.Dropout(film_dropout),
            nn.Linear(film_hidden, 2 * proc_dim),
        )
        self.film_c = nn.Sequential(           # proc → modulate comp
            nn.Linear(proc_dim, film_hidden), nn.GELU(),
            nn.Dropout(film_dropout),
            nn.Linear(film_hidden, 2 * comp_dim),
        )
        for net in (self.film_p, self.film_c):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        # streams = (struct, comp, [dopant,] proc)
        comp = streams[1]
        proc = streams[-1]
        gp, bp = self.film_p(comp).chunk(2, dim=-1)
        gc, bc = self.film_c(proc).chunk(2, dim=-1)
        proc_mod = (1.0 + gp) * proc + bp
        comp_mod = (1.0 + gc) * comp + bc
        new_streams = (streams[0], comp_mod) + streams[2:-1] + (proc_mod,)
        return torch.cat(new_streams, dim=-1)

    @staticmethod
    def output_dim(*dims: int) -> int:
        return sum(dims)


class CrossStreamAttentionFusion(nn.Module):
    """
    Phase 13B: 1-layer cross-stream Transformer with [CLS] pooling.

    Each stream is projected to a common ``token_dim``, plus a learnable type
    embedding. A learnable [CLS] token attends to (and is attended by) the
    stream tokens. Output is the [CLS] token after one self-attention block
    and one FFN block, then projected to ``out_dim``.

    Designed for ~170-sample fine-tuning: 1 layer, dropout 0.2, LayerNorm
    everywhere, total ~40k params.
    """

    def __init__(
        self,
        stream_dims: list[int],
        token_dim: int = 64,
        num_heads: int = 4,
        out_dim: int = 64,
        dropout: float = 0.2,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.num_streams = len(stream_dims)
        self.token_dim = token_dim
        self.projections = nn.ModuleList([
            nn.Linear(d, token_dim) for d in stream_dims
        ])
        self.type_emb = nn.Parameter(torch.randn(self.num_streams, token_dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * ffn_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * ffn_mult, token_dim),
        )
        self.ln2 = nn.LayerNorm(token_dim)
        self.output_proj = nn.Linear(token_dim, out_dim)
        self.out_dim_value = out_dim

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        B = streams[0].shape[0]
        toks = [proj(s) + self.type_emb[i]
                for i, (proj, s) in enumerate(zip(self.projections, streams))]
        cls = self.cls.expand(B, -1, -1)                           # [B, 1, D]
        x = torch.cat([cls] + [t.unsqueeze(1) for t in toks], 1)   # [B, 1+N, D]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return self.output_proj(x[:, 0])                           # [B, out_dim]

    def output_dim(self) -> int:
        return self.out_dim_value


class SelfAttentionFusion(nn.Module):
    """
    Self-attention fusion over N stream tokens.

    Each stream is first projected to a common `token_dim`, then treated
    as one token in a sequence.  Multi-head self-attention lets the model
    learn cross-modal interactions.  The output is flattened and projected
    back to `out_dim`.

    Args:
        stream_dims: list of input dims for each stream (e.g. [64, 64, 32]).
        token_dim: common projection dimension per token.
        num_heads: number of attention heads (must divide token_dim evenly).
        out_dim: final output dimension.
        dropout: dropout on attention weights.
    """

    def __init__(
        self,
        stream_dims: list[int],
        token_dim: int = 64,
        num_heads: int = 4,
        out_dim: int = 160,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_streams = len(stream_dims)
        self.token_dim = token_dim

        # Project each stream to a common token dim
        self.projections = nn.ModuleList([
            nn.Linear(d, token_dim) for d in stream_dims
        ])

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # input shape [B, seq_len, token_dim]
        )
        self.layer_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.layer_norm2 = nn.LayerNorm(token_dim)

        # Flatten + project to out_dim
        self.output_proj = nn.Linear(self.num_streams * token_dim, out_dim)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *streams: each Tensor [B, d_i]

        Returns:
            Tensor [B, out_dim]
        """
        # Project to common dim and stack as sequence: [B, num_streams, token_dim]
        tokens = torch.stack(
            [proj(s) for proj, s in zip(self.projections, streams)],
            dim=1,
        )

        # Self-attention with residual
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.layer_norm(tokens + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(tokens)
        tokens = self.layer_norm2(tokens + ffn_out)

        # Flatten: [B, num_streams * token_dim]
        flat = tokens.reshape(tokens.size(0), -1)
        return self.output_proj(flat)   # [B, out_dim]
