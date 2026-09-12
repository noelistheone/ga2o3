"""Phase 51 — V51 model factory using MultiEncoderFusion.

Builds a Ga2O3Net instance whose `.encoder` is a MultiEncoderFusion
(N frozen pretrained encoders + small self-attention) instead of a
single CGCNN encoder.

The downstream Ga2O3Net code (composition stream, process stream, FiLM
fusion, Brouwer head) is reused unchanged because MultiEncoderFusion
exposes the same interface as CGCNNEncoder
(`.get_embedding(graph) → [B, D]`, `.out_dim`, `.freeze()`,
`.unfreeze()`, `.unfreeze_last_n()`).

This is a NEW file — does not modify Ga2O3Net or CGCNNEncoder.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from src.models.ga2o3_net import Ga2O3Net, CompositionStream, ProcessStream, DopantStream
from src.models.multi_encoder_fusion import MultiEncoderFusion

logger = logging.getLogger(__name__)


def build_v51_model(
    encoder_specs: list[dict],
    target_cols: list[str],
    fusion_dim: int = 64,
    fusion_pool: str = "cls",
    n_attn_heads: int = 4,
    n_attn_layers: int = 1,
    attn_dropout: float = 0.1,
    ffn_hidden_dim: int = 128,
    fusion_mode: str = "film",
    composition_kwargs: dict | None = None,
    process_kwargs: dict | None = None,
    dopant_kwargs: dict | None = None,
    head_hidden_dims: list[int] = (64, 32),
    head_dropout: float = 0.30,
    head_type: str = "brouwer",
    num_experts: int = 4,
    moe_routing: str = "soft",
    **physics_kwargs,
) -> Ga2O3Net:
    """Construct a V51 Ga2O3Net with MultiEncoderFusion as the structure stream.

    Args mirror the V5 fine-tune config (`fusion_head128_phase43_v5_distill_vc.yaml`)
    so the existing trainer paths work without modification.

    NOTE: V5's `pretrain_distill_aux` uses `encoder.pretrain_head` for the
    distillation target. MultiEncoderFusion doesn't expose this — set
    `pretrain_distill_aux=False` (already the default) when using V51.
    """
    composition_kwargs = composition_kwargs or {}
    process_kwargs = process_kwargs or {}

    # ---- Build the multi-encoder fusion structure stream ----
    encoder = MultiEncoderFusion(
        encoder_specs=encoder_specs,
        attn_dim=fusion_dim,
        n_heads=n_attn_heads,
        n_layers=n_attn_layers,
        pool=fusion_pool,
        attn_dropout=attn_dropout,
        ffn_hidden_dim=ffn_hidden_dim,
    )

    # ---- Build composition + process streams (same as V5) ----
    composition = CompositionStream(**composition_kwargs)
    process = ProcessStream(**process_kwargs)
    dopant = DopantStream(**dopant_kwargs) if dopant_kwargs else None

    # Make sure V5's distill aux is disabled — MultiEncoderFusion has no
    # `pretrain_head` to distill from. Caller is supposed to ensure this,
    # but we double-check.
    if physics_kwargs.get("pretrain_distill_aux", False):
        logger.warning("V51 ignores pretrain_distill_aux=True — "
                       "MultiEncoderFusion has no pretrain_head. Disabling.")
        physics_kwargs["pretrain_distill_aux"] = False

    model = Ga2O3Net(
        encoder=encoder,
        composition_stream=composition,
        process_stream=process,
        target_cols=target_cols,
        fusion_mode=fusion_mode,
        dopant_stream=dopant,
        head_hidden_dims=head_hidden_dims,
        head_dropout=head_dropout,
        head_type=head_type,
        num_experts=num_experts,
        moe_routing=moe_routing,
        **physics_kwargs,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"V51 model built: {n_trainable:,}/{n_total:,} trainable params")
    logger.info(f"  encoders ({len(encoder_specs)}): "
                f"{[s['name'] for s in encoder_specs]} (frozen)")
    logger.info(f"  fusion: {fusion_pool}-pool, {n_attn_heads} heads × "
                f"{n_attn_layers} layers, attn_dim={fusion_dim} → "
                f"struct_emb_dim={encoder.out_dim}")
    return model
