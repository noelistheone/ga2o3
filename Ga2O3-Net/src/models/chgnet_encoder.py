"""Phase 49 V17 — CHGNet-based structure encoder (frozen, cached features).

Drop-in replacement for `CGCNNEncoder` that returns CHGNet's pretrained
`crystal_fea` (64-dim) instead of running CGCNN forward. CHGNet was
pretrained on MPtrj — 1.5M oxide structures with energy/force/stress/magmom
labels — so its crystal-level features encode oxidation-state-aware
geometry that our 82k-MP-oxide CGCNN pretrain cannot.

Embeddings are pre-extracted by `scripts/extract_chgnet_embeddings.py` and
attached to each PyG `Data` object as `data.chgnet_emb` (shape [64]). At
training time, this encoder simply reads `graph.chgnet_emb` from the batch.
No CHGNet weights are loaded into the live model — pure cache lookup.

Reference: HackNIP (arXiv:2506.18497) — for downstream tasks at N<10⁴,
frozen NIP embedding + shallow head BEATS end-to-end fine-tune of the same
NIP. The agent's report (2026-05-04) ranked this as Rec 2.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class CHGNetCachedEncoder(nn.Module):
    """Frozen-by-construction encoder that looks up CHGNet `crystal_fea`
    via the `chgnet_emb` attribute attached to each graph batch.

    Interface mirrors CGCNNEncoder so `Ga2O3Net` doesn't need to care.
    """

    out_dim: int = 64

    def __init__(self, cache_path: str | Path | None = None,
                 embedding_dim: int = 64, allow_missing: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cache_path = str(cache_path) if cache_path is not None else None
        self.allow_missing = allow_missing
        # Single learnable parameter so the optimizer always sees something
        # — projection scale that lets the downstream optimizer rescale
        # the frozen feature magnitudes without retraining the backbone.
        self.scale = nn.Parameter(torch.ones(1))
        # Register a dummy buffer so `.to(device)` and `.device` work.
        self.register_buffer("_device_anchor", torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def get_embedding(self, batch) -> torch.Tensor:
        """Read `chgnet_emb` attribute from the PyG batch.

        Args:
            batch: PyG Batch with `chgnet_emb` of shape [B, 64] (or [B*64]
                   stacked along node dim — we reshape).

        Returns:
            Tensor [B, 64].
        """
        if not hasattr(batch, "chgnet_emb") or batch.chgnet_emb is None:
            if self.allow_missing:
                # Return zero embedding (debug only; never on production runs)
                bs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
                return torch.zeros(bs, self.embedding_dim,
                                   device=self.device)
            raise RuntimeError(
                "Batch is missing `chgnet_emb` attribute. Did you build the "
                "dataset with `chgnet_emb_cache_path=...`?"
            )
        emb = batch.chgnet_emb
        # PyG concatenates per-graph attrs along dim 0 when present.
        # Shape after batching is [B, 64].
        if emb.dim() == 1:
            emb = emb.view(-1, self.embedding_dim)
        if emb.shape[-1] != self.embedding_dim:
            raise RuntimeError(
                f"chgnet_emb has dim {emb.shape[-1]}, expected "
                f"{self.embedding_dim}"
            )
        emb = emb.to(self.device)
        return emb * self.scale

    def freeze(self) -> None:
        # No backbone weights to freeze; only `scale` is trainable.
        # Trainer can call freeze() to disable scale learning if desired.
        self.scale.requires_grad_(False)

    def unfreeze(self) -> None:
        self.scale.requires_grad_(True)

    def unfreeze_last_n(self, n: int) -> None:  # noqa: ARG002
        # No-op for cached encoder; provided for API compatibility.
        pass

    @classmethod
    def load_cache(cls, cache_path: str | Path) -> dict[str, torch.Tensor]:
        """Load the cache dict from disk; returns {cache_key: torch.Tensor[64]}."""
        data = torch.load(str(cache_path), weights_only=False)
        if isinstance(data, dict) and "embeddings" in data:
            raw = data["embeddings"]
        else:
            raw = data
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in raw.items()}
