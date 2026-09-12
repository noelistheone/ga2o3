"""
Latent Synthesis Bank — Phase 12A.

Pre-computed pool of synthetic (fused_embedding, target) pairs,
each anchored to one real CSV row whose `doi` is recorded so the
trainer can filter out rows whose anchor leaks the val fold.

The bank is loaded once at trainer init from a single .npz file
(produced by `/tmp/phase12_prep.py`).
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch


class LatentSynthBank:
    """
    Provides per-fold sampling of synthetic (embedding, target) pairs.

    npz schema:
      synth_embeddings:  [M, fused_dim]  float32
      synth_pdr:         [M]             float32 (NaN allowed)
      synth_vc:          [M]             float32 (NaN allowed)
      synth_anchor_idx:  [M]             int32   (row idx in the source CSV)
      synth_anchor_doi:  [M]             object  (DOI string per anchor)
      synth_cluster_id:  [M]             int32   (cluster id, for diagnostics)
    """

    def __init__(
        self,
        npz_path: str | pathlib.Path,
        target_col: str = "photo_dark_ratio",
        synth_weight: float = 0.3,
    ):
        d = np.load(npz_path, allow_pickle=True)
        self.embeddings = torch.from_numpy(d["synth_embeddings"].astype(np.float32))
        if target_col == "photo_dark_ratio":
            self.targets = torch.from_numpy(d["synth_pdr"].astype(np.float32))
        elif target_col == "vacancy_concentration":
            self.targets = torch.from_numpy(d["synth_vc"].astype(np.float32))
        else:
            raise ValueError(f"Unknown target_col: {target_col!r}")
        # Mask entries with NaN target (target column may differ from npz column)
        self.target_valid = ~torch.isnan(self.targets)
        self.anchor_doi = np.array(d["synth_anchor_doi"], dtype=object)
        self.anchor_idx = np.asarray(d["synth_anchor_idx"], dtype=np.int64)
        self.cluster_id = np.asarray(d["synth_cluster_id"], dtype=np.int64)
        self.synth_weight = float(synth_weight)
        self.target_col = target_col

        n_total = len(self.targets)
        n_valid = int(self.target_valid.sum())
        print(f"[LatentSynthBank] loaded {n_total} synth rows, "
              f"{n_valid} valid for {target_col}, "
              f"weight={self.synth_weight}")

    @staticmethod
    def _doi_in_allowed(doi_str: str, allowed: set[str]) -> bool:
        """Handle both single-anchor DOIs and pair-anchor DOIs ('A||B').

        For pair-anchor SMOTE (12E), the doi field stores 'doi_a||doi_b';
        we require BOTH parts to be in allowed_dois to keep fold integrity.
        """
        if "||" in doi_str:
            parts = doi_str.split("||")
            return all(p in allowed for p in parts)
        return doi_str in allowed

    def sample(
        self,
        n: int,
        allowed_dois: set[str] | None = None,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Sample n synthetic rows whose anchor DOI(s) are in allowed_dois (if
        given) and whose target value is non-NaN.

        Returns (embs [n, dim], targets [n], weights [n]) all on `device`,
        or None if the eligible pool is empty.
        """
        if allowed_dois is not None:
            mask = self.target_valid & torch.tensor(
                [self._doi_in_allowed(d, allowed_dois) for d in self.anchor_doi],
                dtype=torch.bool,
            )
        else:
            mask = self.target_valid
        eligible = mask.nonzero(as_tuple=True)[0]
        if eligible.numel() == 0:
            return None
        # Sample with replacement (when n > eligible)
        if n > eligible.numel():
            idx = eligible[torch.randint(0, eligible.numel(), (n,))]
        else:
            idx = eligible[torch.randperm(eligible.numel())[:n]]
        embs    = self.embeddings[idx].to(device)
        targets = self.targets[idx].to(device).unsqueeze(-1)   # [n, 1]
        weights = torch.full((n,), self.synth_weight, device=device)
        return embs, targets, weights

    def n_eligible_for_dois(self, allowed_dois: set[str] | None) -> int:
        """How many synth rows pass the fold filter?"""
        if allowed_dois is None:
            return int(self.target_valid.sum())
        mask = self.target_valid & torch.tensor(
            [self._doi_in_allowed(d, allowed_dois) for d in self.anchor_doi],
            dtype=torch.bool,
        )
        return int(mask.sum())
