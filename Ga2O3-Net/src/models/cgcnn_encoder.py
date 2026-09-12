"""
CGCNN-style Crystal Graph Convolutional Neural Network encoder.

Supports multiple architectural variants for hyperparameter search:
  - residual      : skip connections between CGConv layers
  - pool          : 'mean' | 'max' | 'mean_max' | 'attention'
  - activation    : 'softplus' | 'silu' | 'relu'
  - num_conv_layers: depth (4–8)
  - atom_embedding_dim: width (64 | 128 | 256)
  - pre_pool_dim  : optional linear projection before pooling

Default (run00 baseline):
  4 layers, 64 dim, mean pool, softplus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool, global_max_pool, GlobalAttention


class CGCNNEncoder(nn.Module):
    """
    Transferable GNN encoder for crystal structures.

    Args:
        atom_embedding_dim : Dimension of learnable atom-type embeddings.
        edge_rbf_dim       : Dimension of edge features (Gaussian RBF size).
        num_conv_layers    : Number of CGConv message-passing layers.
        residual           : If True, add residual skip connections.
        pool               : Pooling method — 'mean', 'max', 'mean_max', 'attention'.
        activation         : Node activation — 'softplus', 'silu', 'relu'.
        pre_pool_dim       : If > 0, insert Linear(embedding_dim → pre_pool_dim)
                             before pooling. Output embedding dim becomes pre_pool_dim
                             (or 2*pre_pool_dim for mean_max pool).
        pretrain           : If True, attach regression head for pre-training.
        num_targets        : Number of pre-training regression targets (default 2).
    """

    def __init__(
        self,
        atom_embedding_dim: int = 64,
        edge_rbf_dim: int = 40,
        num_conv_layers: int = 4,
        residual: bool = False,
        pool: str = "mean",
        activation: str = "softplus",
        pre_pool_dim: int = 0,
        pretrain: bool = True,
        num_targets: int = 2,
        multi_head_targets: dict[str, int] | None = None,
    ):
        super().__init__()
        self.embedding_dim = atom_embedding_dim
        self.residual = residual
        self.pool_type = pool
        self.pre_pool_dim = pre_pool_dim

        # Activation function
        _act_map = {
            "softplus": F.softplus,
            "silu": F.silu,
            "relu": F.relu,
        }
        if activation not in _act_map:
            raise ValueError(f"activation must be one of {list(_act_map)}; got {activation!r}")
        self.act = _act_map[activation]

        # Atom embedding: maps atomic number (1–94) → learned vector
        self.atom_embedding = nn.Embedding(95, atom_embedding_dim, padding_idx=0)

        # CGConv layers with BatchNorm
        self.conv_layers = nn.ModuleList([
            CGConv(atom_embedding_dim, dim=edge_rbf_dim, batch_norm=False)
            for _ in range(num_conv_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(atom_embedding_dim)
            for _ in range(num_conv_layers)
        ])

        # Optional pre-pooling projection
        if pre_pool_dim > 0:
            self.pre_pool_proj = nn.Sequential(
                nn.Linear(atom_embedding_dim, pre_pool_dim),
                nn.SiLU(),
            )
            pool_in_dim = pre_pool_dim
        else:
            self.pre_pool_proj = None
            pool_in_dim = atom_embedding_dim

        # Pooling
        if pool == "attention":
            # GlobalAttention: gate_nn scores each node, then weighted sum
            self.attention_pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(pool_in_dim, pool_in_dim),
                    nn.SiLU(),
                    nn.Linear(pool_in_dim, 1),
                )
            )
            self._out_dim = pool_in_dim
        elif pool == "mean_max":
            self._out_dim = pool_in_dim * 2
        else:
            self._out_dim = pool_in_dim

        # Pre-training regression head(s)
        self.pretrain = pretrain
        # Phase 50: multi-head pretrain — train the encoder against multiple
        # databases simultaneously (MP bandgap+E_form, JARVIS dielectric,
        # Witman/Goyal V_O formation enthalpy). Each head is a Linear
        # projection from the encoder embedding to a database-specific target.
        # When `multi_head_targets` is None we fall back to the single-head
        # legacy path so existing checkpoints / V5 distill aux keep working.
        self.multi_head_targets = multi_head_targets
        if pretrain:
            if multi_head_targets:
                self.pretrain_heads = nn.ModuleDict({
                    name: nn.Linear(self._out_dim, n_out)
                    for name, n_out in multi_head_targets.items()
                })
                # Backwards-compat alias — V5's aux_distill points at
                # `encoder.pretrain_head` (legacy 2-d MP head). When multi-head
                # is on, alias to the "mp" head so V5 keeps working unchanged.
                if "mp" in multi_head_targets:
                    self.pretrain_head = self.pretrain_heads["mp"]
                else:
                    # No "mp" key — pick the first; useful for ablations
                    first_name = next(iter(multi_head_targets))
                    self.pretrain_head = self.pretrain_heads[first_name]
            else:
                self.pretrain_head = nn.Linear(self._out_dim, num_targets)

    @property
    def out_dim(self) -> int:
        """Embedding dimension output by the encoder (before pretrain head)."""
        return self._out_dim

    def forward(self, data, return_embedding: bool = False,
                head: str | None = None):
        """
        Args:
            data: PyG Data with x, edge_index, edge_attr, batch.
                Phase 5D: ``data.x`` is [N, 4] float32 with columns
                ``(host_Z, dopant_Z, dopant_occ, is_disordered_flag)``. Node
                features become ``emb(host_Z) * (1 - occ) + emb(dopant_Z) * occ``.
                Legacy [N] long input (pre-5D checkpoints, pretest fixtures) is
                still accepted and falls through to the simple embedding lookup.
            return_embedding: If True, return graph embedding (skip pretrain head).

        Returns:
            [B, out_dim] embedding  or  [B, num_targets] predictions.
        """
        raw_x = data.x
        if raw_x.dim() == 2 and raw_x.size(-1) >= 3:
            host_z = raw_x[:, 0].long().clamp(min=0, max=94)
            dopant_z = raw_x[:, 1].long().clamp(min=0, max=94)
            dopant_occ = raw_x[:, 2].clamp(min=0.0, max=1.0).unsqueeze(-1)   # [N, 1]
            host_occ = 1.0 - dopant_occ
            x = self.atom_embedding(host_z) * host_occ \
                + self.atom_embedding(dopant_z) * dopant_occ                 # [N, D]
        else:
            x = self.atom_embedding(raw_x.long())                            # [N, D]

        for conv, bn in zip(self.conv_layers, self.batch_norms):
            h = conv(x, data.edge_index, data.edge_attr)
            h = bn(h)
            h = self.act(h)
            x = x + h if self.residual else h     # residual skip or replace

        # Optional pre-pooling projection
        if self.pre_pool_proj is not None:
            x = self.pre_pool_proj(x)

        # Pooling
        if self.pool_type == "mean":
            embedding = global_mean_pool(x, data.batch)
        elif self.pool_type == "max":
            embedding = global_max_pool(x, data.batch)
        elif self.pool_type == "mean_max":
            embedding = torch.cat([
                global_mean_pool(x, data.batch),
                global_max_pool(x, data.batch),
            ], dim=-1)
        elif self.pool_type == "attention":
            embedding = self.attention_pool(x, data.batch)
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type!r}")

        if return_embedding or not self.pretrain:
            return embedding

        # Multi-head routing: pass head="mp" / "jarvis" / "vacancy" to pick.
        if head is not None and self.multi_head_targets:
            return self.pretrain_heads[head](embedding)
        return self.pretrain_head(embedding)

    def get_embedding(self, data) -> torch.Tensor:
        return self.forward(data, return_embedding=True)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def unfreeze_last_n(self, n: int):
        """
        Partially unfreeze the encoder: freeze all params, then unfreeze only
        the last `n` CGConv layers + their BatchNorms + optional pre_pool_proj
        + attention pooling gate (if used).

        Atom embedding stays frozen to preserve pretrained element representations.
        Use encoder_lr (e.g. 5e-6) for the unfrozen params to avoid catastrophic
        forgetting of pretrained crystal geometry features.

        Args:
            n: Number of CGConv layers to unfreeze from the end (e.g. 2 of 4).
        """
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        n = min(n, len(self.conv_layers))
        # Unfreeze last n conv layers and their corresponding batch norms
        for layer, bn in zip(self.conv_layers[-n:], self.batch_norms[-n:]):
            for p in layer.parameters():
                p.requires_grad = True
            for p in bn.parameters():
                p.requires_grad = True

        # Unfreeze pre-pooling projection if present
        if self.pre_pool_proj is not None:
            for p in self.pre_pool_proj.parameters():
                p.requires_grad = True

        # Unfreeze attention pooling gate if used
        if self.pool_type == "attention":
            for p in self.attention_pool.parameters():
                p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        import logging
        logging.getLogger(__name__).info(
            f"CGCNNEncoder: last {n} conv layers unfrozen "
            f"({n_trainable:,}/{n_total:,} params trainable)."
        )
