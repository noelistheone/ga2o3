"""Phase 51 — JARVIS-DFT dataset for single-task encoder pretraining.

Loads JARVIS-DFT (Choudhary 2020 npj CM, NIST) — 76k structures with
TBmBJ bandgap (~HSE quality), dielectric tensor, formation energy, etc.
Used to pretrain a CGCNN encoder specialized in *electronic structure*
features — complementary to V5's bandgap+E_form encoder (PBE only) and
the Witman/Goyal V_O-formation-enthalpy encoder.

Default targets:
  - mbj_bandgap (corrects PBE underestimate; covers β-Ga₂O₃ regime)
  - epsx, epsy, epsz (orientation-averaged static dielectric)

Backed by dft_3d.json (downloaded via `jarvis.db.figshare.data('dft_3d')`).

This is a *new file* — does not touch existing src/data/ modules.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _atoms_dict_to_structure(atoms_dict: dict):
    """Convert JARVIS atoms dict → pymatgen Structure."""
    from pymatgen.core import Structure, Lattice
    lattice = Lattice(atoms_dict["lattice_mat"])
    species = atoms_dict["elements"]
    coords = atoms_dict["coords"]
    coords_are_cartesian = bool(atoms_dict.get("cartesian", False))
    return Structure(lattice, species, coords,
                     coords_are_cartesian=coords_are_cartesian)


def _atoms_to_graph(atoms_dict: dict, cutoff: float = 6.0,
                    num_rbf: int = 40) -> Data:
    """Convert JARVIS atoms dict → PyG graph using project graph_builder."""
    from src.data.graph_builder import structure_to_graph
    s = _atoms_dict_to_structure(atoms_dict)
    return structure_to_graph(s, cutoff=cutoff, num_rbf=num_rbf)


class JarvisDFTDataset(Dataset):
    """JARVIS-DFT 76k oxide+nonoxide structure-property dataset.

    Args:
        json_path : path to dft_3d.json (downloaded once via
                    `from jarvis.db.figshare import data; data('dft_3d')`).
        cache_dir : directory for cached PyG graph .pt files.
        targets   : list of JARVIS field names to predict, e.g.
                    ['mbj_bandgap', 'epsx', 'epsy', 'epsz']. NaN samples
                    are excluded per-target.
        require_all_targets : if True, only keep entries with ALL targets
                    present (smaller, cleaner). If False, allow per-target
                    NaN masks — caller's loss must handle NaNs.
        oxide_only : if True, filter to entries containing oxygen.
        max_atoms : skip very large supercells (>this many atoms).
    """

    def __init__(self, json_path: str, cache_dir: str,
                 targets: list[str] = None,
                 require_all_targets: bool = True,
                 oxide_only: bool = True,
                 max_atoms: int = 80,
                 cutoff: float = 6.0,
                 num_rbf: int = 40,
                 normalize: bool = True):
        super().__init__()
        if targets is None:
            targets = ["mbj_bandgap", "epsx", "epsy", "epsz"]
        self.targets = list(targets)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.normalize = normalize

        with open(json_path) as f:
            raw = json.load(f)

        # Filter by target availability + oxide_only + max_atoms
        kept = []
        for entry in raw:
            # Targets present (na or None checks)
            target_vals = []
            ok = True
            for t in self.targets:
                v = entry.get(t)
                if v is None or v == "na" or (isinstance(v, float) and np.isnan(v)):
                    if require_all_targets:
                        ok = False
                        break
                    target_vals.append(np.nan)
                else:
                    target_vals.append(float(v))
            if not ok:
                continue
            # oxide filter (atoms_dict.elements contains 'O')
            atoms_dict = entry.get("atoms")
            if atoms_dict is None:
                continue
            elements = atoms_dict.get("elements", [])
            if oxide_only and "O" not in elements:
                continue
            n_atoms = len(elements)
            if n_atoms == 0 or n_atoms > max_atoms:
                continue
            kept.append((entry["jid"], atoms_dict, target_vals))
        self._raw = kept
        logger.info(f"JARVIS-DFT: {len(raw)} → {len(kept)} kept "
                    f"(targets={targets}, oxide_only={oxide_only}, max_atoms={max_atoms})")

        # Normalisation stats (per target)
        all_targets = np.array([t for _, _, t in kept], dtype=np.float32)
        self._target_mean = np.nanmean(all_targets, axis=0)
        self._target_std = np.nanstd(all_targets, axis=0) + 1e-9
        logger.info(f"  target means: {self._target_mean.tolist()}")
        logger.info(f"  target stds:  {self._target_std.tolist()}")

        # Build/load graph cache
        self._cache_keys: list[str] = []
        for jid, atoms_dict, _ in kept:
            cache_path = self.cache_dir / f"{jid}.pt"
            self._cache_keys.append(jid)
            if not cache_path.exists():
                try:
                    g = _atoms_to_graph(atoms_dict, self.cutoff, self.num_rbf)
                    torch.save(g, str(cache_path))
                except Exception as e:
                    # Mark as None — will skip in __getitem__
                    logger.debug(f"Skip {jid}: {type(e).__name__}: {e}")
                    self._cache_keys[-1] = None
        n_valid = sum(1 for k in self._cache_keys if k is not None)
        logger.info(f"  built {n_valid}/{len(kept)} graph caches in {cache_dir}")

        # Map valid index → kept index
        self._valid_indices = [i for i, k in enumerate(self._cache_keys) if k is not None]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Data:
        kept_idx = self._valid_indices[idx]
        jid, _, target_vals = self._raw[kept_idx]
        graph = torch.load(str(self.cache_dir / f"{jid}.pt"), weights_only=False)
        target_arr = np.asarray(target_vals, dtype=np.float32)
        if self.normalize:
            target_arr = (target_arr - self._target_mean) / self._target_std
        graph.y = torch.tensor(target_arr, dtype=torch.float32).view(1, -1)
        graph.jid = jid
        return graph

    def target_stats(self) -> tuple[np.ndarray, np.ndarray]:
        return self._target_mean.copy(), self._target_std.copy()
