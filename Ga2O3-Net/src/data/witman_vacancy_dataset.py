"""Phase 50.B — Witman/Goyal V_O formation enthalpy dataset.

Reads pristine host POSCARs + V_O formation enthalpy targets from the
Witman/Goyal database (Zenodo 10.5281/zenodo.8087871, Witman et al.
Nat Comp Sci 2023). Targets are mean V_O formation enthalpy per compound
(averaging over multi-site entries) — this gives the encoder a per-host
signal for "how much V_O does this oxide want to form".

Used as Stage 2 pretrain target alongside MP bandgap+E_form to teach
the encoder defect-chemistry physics that's missing from MP's bulk
thermodynamic targets.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _poscar_to_graph(poscar_path: str, cutoff: float = 6.0,
                     num_rbf: int = 40) -> Data:
    """Convert a VASP POSCAR to a PyG graph using the same node-feature
    layout as src/data/graph_builder.py (4-dim node features:
    [host_Z, dopant_Z, dopant_occ, is_disordered_flag]).
    """
    from pymatgen.core import Structure
    from src.data.graph_builder import structure_to_graph
    s = Structure.from_file(poscar_path)
    return structure_to_graph(s, cutoff=cutoff, num_rbf=num_rbf)


class WitmanVacancyDataset(Dataset):
    """Witman/Goyal V_O formation enthalpy database.

    One sample per (compound_id, V_O site). Multi-site entries are kept
    (so a compound with 3 inequivalent O sites contributes 3 examples,
    each with the same host structure but a different target).

    Args:
        master_csv : Path to the master table built by extract_witman_master.
            Must have columns: compound_id, site, dH_eV, formula,
            bandgap_eV, poscar_path.
        cache_dir  : Directory for cached PyG .pt files
            (one per compound_id, since structure is shared across sites).
        cutoff     : Neighbour search radius (Å). Should match MP dataset.
        num_rbf    : RBF basis dim for edge features (default 40).
    """

    def __init__(self, master_csv: str, cache_dir: str,
                 cutoff: float = 6.0, num_rbf: int = 40,
                 normalize: bool = True):
        super().__init__()
        self.master_csv = master_csv
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cutoff = cutoff
        self.num_rbf = num_rbf

        self.df = pd.read_csv(master_csv).reset_index(drop=True)
        # One graph per unique compound — share across multi-site rows
        self._graph_cache: dict[str, Data] = {}
        self._preload()

        # Normalisation stats over training targets
        targets = self.df["dH_eV"].to_numpy(dtype=np.float32)
        self._target_mean = float(np.mean(targets))
        self._target_std = float(np.std(targets) + 1e-9)
        self.normalize = normalize
        logger.info(f"Witman dataset: {len(self)} entries, "
                    f"{len(self._graph_cache)} unique compounds, "
                    f"target mean={self._target_mean:.3f} std={self._target_std:.3f}")

    def _preload(self):
        """Convert each unique POSCAR to a PyG graph, with on-disk cache."""
        unique = self.df.drop_duplicates(subset=["compound_id"])
        n_built = 0
        for _, row in unique.iterrows():
            cid = row["compound_id"]
            cache_path = self.cache_dir / f"{cid}.pt"
            if cache_path.exists():
                try:
                    self._graph_cache[cid] = torch.load(str(cache_path),
                                                         weights_only=False)
                    continue
                except Exception:
                    pass
            try:
                graph = _poscar_to_graph(row["poscar_path"],
                                          cutoff=self.cutoff,
                                          num_rbf=self.num_rbf)
                torch.save(graph, str(cache_path))
                self._graph_cache[cid] = graph
                n_built += 1
            except Exception as e:
                logger.warning(f"Witman skip {cid}: {type(e).__name__}: {e}")
        logger.info(f"Witman preload: {n_built} new graphs built, "
                    f"{len(self._graph_cache)} total in cache")

    def __len__(self) -> int:
        return sum(1 for _, r in self.df.iterrows()
                   if r["compound_id"] in self._graph_cache)

    def __getitem__(self, idx: int) -> Data:
        # Filter to rows whose graph is in cache (skip failed compounds)
        if not hasattr(self, "_valid_idx") or self._valid_idx is None:
            self._valid_idx = [i for i in range(len(self.df))
                               if self.df.iloc[i]["compound_id"] in self._graph_cache]
        actual_idx = self._valid_idx[idx]
        row = self.df.iloc[actual_idx]
        graph = self._graph_cache[row["compound_id"]].clone()
        target = float(row["dH_eV"])
        if self.normalize:
            target = (target - self._target_mean) / self._target_std
        graph.y = torch.tensor([target], dtype=torch.float32)
        graph.compound_id = row["compound_id"]
        return graph

    def target_stats(self) -> tuple[float, float]:
        return self._target_mean, self._target_std
