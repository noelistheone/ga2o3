"""Phase 51 — Frozen Multi-Expert Fusion via Self-Attention.

The user's proposed architecture (Phase 51):
    1. Pretrain N CGCNN encoders independently, each on its own task:
         - encoder_mp:      bandgap + E_form  (pretrained_encoder_v2.pt)
         - encoder_witman:  V_O formation enthalpy
         - encoder_jarvis:  TBmBJ bandgap + dielectric
    2. FREEZE all N encoders (no gradient back to them at fine-tune time).
    3. At inference: each frozen encoder produces a 64-dim graph
       representation. Stack them as a sequence of N tokens, run through
       a small self-attention layer, then pool.

This dodges every Phase 47-49 failure mode:
    - No "tug-of-war" — encoders never co-train.
    - No "encoder feature shift" — frozen encoders preserve V5's chemistry.
    - No "capacity dilution" — each task has its own 64-dim space.
    - No "prototype pulling" — attention is per-sample dynamic, not a
      shared latent target.

Reference: HackNIP (arXiv:2506.18497) shows frozen-NIP-embedding +
shallow head BEATS end-to-end fine-tune at downstream N<10⁴.

This is a NEW file — does not modify CGCNNEncoder or Ga2O3Net.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from src.models.cgcnn_encoder import CGCNNEncoder

logger = logging.getLogger(__name__)


def _load_encoder(ckpt_path: str, encoder_kwargs: dict | None = None) -> CGCNNEncoder:
    """Load a CGCNN encoder from a flat state-dict checkpoint."""
    encoder_kwargs = encoder_kwargs or {}
    enc = CGCNNEncoder(
        atom_embedding_dim=encoder_kwargs.get("atom_embedding_dim", 64),
        edge_rbf_dim=encoder_kwargs.get("edge_rbf_dim", 40),
        num_conv_layers=encoder_kwargs.get("num_conv_layers", 4),
        residual=encoder_kwargs.get("residual", False),
        pool=encoder_kwargs.get("pool", "mean"),
        activation=encoder_kwargs.get("activation", "softplus"),
        pretrain=False,   # we don't need the pretrain head at fine-tune
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "encoder_state" in state:
        state = state["encoder_state"]
    # Drop pretrain head weights (they're not used)
    state = {k: v for k, v in state.items()
             if not k.startswith("pretrain_head")
             and not k.startswith("pretrain_heads")}
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Encoder load_state_dict missing keys: {list(missing)[:5]}")
    if unexpected:
        logger.warning(f"Encoder load_state_dict unexpected keys: {list(unexpected)[:5]}")
    return enc


class MultiEncoderFusion(nn.Module):
    """Frozen multi-encoder fusion via self-attention.

    Args:
        encoder_specs: list of dicts, each with keys:
            'name': str — used for logging/diagnostics
            'ckpt_path': str — path to flat state-dict
            'freeze': bool (default True)
            'encoder_kwargs': dict for CGCNNEncoder ctor (defaults to V5)
        attn_dim: dimension of attention space (= encoder out_dim, default 64)
        n_heads: attention heads (default 4)
        n_layers: number of self-attention layers (default 1)
        pool: 'cls' | 'mean' | 'concat' — how to reduce sequence to vector
        attn_dropout: dropout in attention (default 0.1)
        ffn_hidden_dim: feedforward hidden dim if n_layers>0 (default 128)

    Forward(graph) → tensor of shape:
        [B, attn_dim]                if pool='cls' or pool='mean'
        [B, attn_dim * n_encoders]   if pool='concat'

    Use `out_dim` property to query the output dim.

    Provides .freeze()/.unfreeze() shims so it can drop into Ga2O3Net's
    encoder slot. The encoders themselves are always frozen by design;
    only the attention layers and (if pool='cls') the CLS token are
    learnable.
    """

    def __init__(self,
                 encoder_specs: list[dict],
                 attn_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 1,
                 pool: str = "cls",
                 attn_dropout: float = 0.1,
                 ffn_hidden_dim: int = 128):
        super().__init__()
        if pool not in ("cls", "mean", "concat"):
            raise ValueError(f"pool must be cls/mean/concat, got {pool!r}")
        self.pool = pool
        self.attn_dim = attn_dim
        self.n_encoders = len(encoder_specs)

        self.encoder_names = [s["name"] for s in encoder_specs]
        # ModuleDict to register encoders so .to(device) works
        self.encoders = nn.ModuleDict()
        for spec in encoder_specs:
            enc = _load_encoder(spec["ckpt_path"], spec.get("encoder_kwargs"))
            if spec.get("freeze", True):
                for p in enc.parameters():
                    p.requires_grad = False
                enc.eval()
            self.encoders[spec["name"]] = enc

        # Self-attention layer(s)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=n_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

    @property
    def out_dim(self) -> int:
        if self.pool in ("cls", "mean"):
            return self.attn_dim
        return self.attn_dim * self.n_encoders

    @property
    def embedding_dim(self) -> int:
        # Ga2O3Net.__init__ reads encoder.embedding_dim — alias to out_dim
        return self.out_dim

    def get_encoder_embeddings(self, graph) -> torch.Tensor:
        """Return [B, n_encoders, attn_dim] tensor of per-encoder graph embeds.

        Encoders are always run in no_grad context (frozen by design).
        """
        embs = []
        for name in self.encoder_names:
            enc = self.encoders[name]
            # Even if .freeze() wasn't called, we always treat as eval+no_grad
            # to ensure stability at fine-tune time.
            with torch.no_grad():
                e = enc.get_embedding(graph)
            embs.append(e)
        return torch.stack(embs, dim=1)        # [B, n_encoders, attn_dim]

    def forward(self, graph) -> torch.Tensor:
        embs = self.get_encoder_embeddings(graph)   # [B, N, D]
        B = embs.size(0)

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)        # [B, 1, D]
            seq = torch.cat([cls, embs], dim=1)            # [B, N+1, D]
            out = self.attn(seq)                           # [B, N+1, D]
            return out[:, 0]                                # [B, D]
        elif self.pool == "mean":
            out = self.attn(embs)                          # [B, N, D]
            return out.mean(dim=1)                         # [B, D]
        else:   # concat
            out = self.attn(embs)                          # [B, N, D]
            return out.flatten(start_dim=1)                # [B, N*D]

    def get_embedding(self, graph) -> torch.Tensor:
        """Alias matching CGCNNEncoder.get_embedding signature."""
        return self.forward(graph)

    # ---- API shims for Ga2O3Net compat ---------------------------------------
    def freeze(self):
        """Freezes the encoders (already frozen by default) AND the attention.
        Use .unfreeze() to make attention learnable again.
        """
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        """Re-enable gradient on attention layers + CLS token only.
        Encoders STAY frozen — that is the whole architecture point.
        """
        for p in self.parameters():
            p.requires_grad = True
        # Re-freeze the encoders
        for enc in self.encoders.values():
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()
        self.train()

    def unfreeze_last_n(self, n: int) -> None:
        """No-op for compat. The whole point is encoders stay frozen."""
        pass
