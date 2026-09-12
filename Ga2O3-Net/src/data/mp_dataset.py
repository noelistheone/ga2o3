"""
PyTorch Geometric Dataset wrapping Materials Project (and AFLOW) oxide structures.
Supports two prediction targets: band_gap and formation_energy_per_atom.

For large datasets (> 5000 structures), process() uses parallel workers to
convert CIFs to PyG graphs. Set num_process_workers > 1 to enable.
"""

from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Dataset, Data

from src.data.graph_builder import cif_to_graph
from src.data.mp_downloader import load_labels_csv

logger = logging.getLogger(__name__)


# ── Worker function (must be top-level for multiprocessing) ───────────────────

def _process_one(args: tuple) -> Optional[dict]:
    """Convert one CIF to a graph and return serialized bytes (or None on error)."""
    import warnings
    warnings.filterwarnings("ignore")   # suppress pymatgen CIF parse warnings in workers

    mp_id, cif_path, label, cutoff, num_rbf = args
    if not os.path.exists(cif_path):
        return None
    try:
        graph = cif_to_graph(cif_path, cutoff=cutoff, num_rbf=num_rbf)
        graph.y = label
        graph.material_id = mp_id
        return {"mp_id": mp_id, "graph": graph}
    except Exception:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class MPOxideDataset(Dataset):
    """
    Dataset of oxide crystal structures for Stage 1 pre-training.

    Each sample is a PyG Data object with:
        x          [N]        atomic numbers
        edge_index [2, E]     bonds (periodic neighbor list)
        edge_attr  [E, 40]    Gaussian RBF distances
        y          [1, 2]     [band_gap_norm, formation_energy_norm]
        material_id           identifier string (mp-XXXXX or aflow-...)

    Args:
        raw_dir: Directory containing .cif files.
        processed_dir: Directory for cached .pt graph files.
        labels_csv: CSV with columns: material_id, band_gap, formation_energy_per_atom, cif_path.
        cutoff: Neighbor search radius in Angstroms.
        num_rbf: Number of Gaussian RBF centers for edge distances.
        num_process_workers: Workers for parallel CIF→graph conversion (0 = serial).
                             Set to os.cpu_count() or -1 for all cores.
    """

    def __init__(
        self,
        raw_dir: str,
        processed_dir: str,
        labels_csv: str,
        cutoff: float = 6.0,
        num_rbf: int = 40,
        num_process_workers: int = 0,
        transform=None,
        pre_transform=None,
    ):
        self.raw_cif_dir = raw_dir
        self.labels_csv = labels_csv
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.num_process_workers = num_process_workers
        self._processed_dir_path = processed_dir
        self._label_rows: list[dict] | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        super().__init__(root=processed_dir, transform=transform,
                         pre_transform=pre_transform)
        self._load_normalization()

    # ── PyG Dataset interface ─────────────────────────────────────────────────

    @property
    def raw_file_names(self):
        return []   # downloading handled externally

    @property
    def processed_file_names(self):
        rows = self._get_label_rows()
        return [f"{row['material_id']}.pt" for row in rows]

    def download(self):
        pass  # Use scripts/01_download_mp.py

    def process(self):
        rows = self._get_label_rows()
        Path(self._processed_dir_path).mkdir(parents=True, exist_ok=True)

        # Normalization stats
        band_gaps = [r["band_gap"] for r in rows]
        fes = [r["formation_energy_per_atom"] for r in rows]
        self._mean = torch.tensor([
            sum(band_gaps) / len(band_gaps),
            sum(fes) / len(fes),
        ], dtype=torch.float32)
        self._std = torch.tensor([
            float(torch.std(torch.tensor(band_gaps, dtype=torch.float32))),
            float(torch.std(torch.tensor(fes, dtype=torch.float32))),
        ], dtype=torch.float32)
        torch.save({"mean": self._mean, "std": self._std},
                   os.path.join(self._processed_dir_path, "normalization.pt"))

        # Build job list (skip already-processed)
        jobs = []
        for row in rows:
            out_path = os.path.join(self._processed_dir_path,
                                    f"{row['material_id']}.pt")
            if os.path.exists(out_path):
                continue
            # cif_path may be in the CSV or derived from raw_cif_dir
            cif_path = row.get("cif_path") or os.path.join(
                self.raw_cif_dir, f"{row['material_id']}.cif"
            )
            label = torch.tensor(
                [[row["band_gap"], row["formation_energy_per_atom"]]],
                dtype=torch.float32,
            )
            jobs.append((row["material_id"], cif_path, label, self.cutoff, self.num_rbf))

        if not jobs:
            logger.info("All graphs already processed — skipping.")
            return

        logger.info(f"Processing {len(jobs)} CIF files...")

        n_workers = self.num_process_workers
        if n_workers == -1:
            import os as _os
            n_workers = _os.cpu_count() or 1

        if n_workers > 1:
            self._process_parallel(jobs, n_workers)
        else:
            self._process_serial(jobs)

    def _process_serial(self, jobs: list[tuple]):
        from tqdm import tqdm
        for job in tqdm(jobs, desc="CIF→graph"):
            result = _process_one(job)
            if result is not None:
                out_path = os.path.join(
                    self._processed_dir_path, f"{result['mp_id']}.pt"
                )
                torch.save(result["graph"], out_path)

    def _process_parallel(self, jobs: list[tuple], n_workers: int):
        import multiprocessing as _mp
        from tqdm import tqdm

        logger.info(f"Using {n_workers} parallel workers for CIF processing.")
        saved = 0
        failed = 0

        # Use 'spawn' context: starts fresh Python interpreters without inheriting
        # parent file descriptors or CUDA state — avoids deadlocks on Linux when
        # orphaned worker processes from a previous run are still alive.
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_one, jobs, chunksize=16),
                total=len(jobs),
                desc="CIF→graph (parallel)",
            ):
                if result is not None:
                    out_path = os.path.join(
                        self._processed_dir_path, f"{result['mp_id']}.pt"
                    )
                    torch.save(result["graph"], out_path)
                    saved += 1
                else:
                    failed += 1

        logger.info(f"Parallel processing done: {saved} saved, {failed} failed.")

    def len(self):
        return len(self._get_label_rows())

    def get(self, idx: int) -> Data:
        rows = self._get_label_rows()
        mp_id = rows[idx]["material_id"]
        path = os.path.join(self._processed_dir_path, f"{mp_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed graph not found: {path}\n"
                f"Run MPOxideDataset.process() first."
            )
        data = torch.load(path, weights_only=False)
        # Normalize labels on-the-fly
        if self._mean is not None and data.y is not None:
            data.y = (data.y - self._mean) / (self._std + 1e-8)
        return data

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_label_rows(self) -> list[dict]:
        if self._label_rows is None:
            self._label_rows = load_labels_csv(self.labels_csv)
            # Filter: only include rows where cif exists
            valid = []
            for row in self._label_rows:
                cif = row.get("cif_path") or os.path.join(
                    self.raw_cif_dir, f"{row['material_id']}.cif"
                )
                if os.path.exists(cif):
                    valid.append(row)
                else:
                    # Silently skip missing CIFs (can happen after partial download)
                    pass
            if len(valid) < len(self._label_rows):
                logger.warning(
                    f"{len(self._label_rows) - len(valid)} entries skipped "
                    f"(CIF files not found)."
                )
            self._label_rows = valid
        return self._label_rows

    def _load_normalization(self):
        norm_path = os.path.join(self._processed_dir_path, "normalization.pt")
        if os.path.exists(norm_path):
            norm = torch.load(norm_path, weights_only=False)
            self._mean = norm["mean"]
            self._std = norm["std"]

    @property
    def normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mean, self._std

    @property
    def dataset_stats(self) -> dict:
        """Summary statistics for logging."""
        rows = self._get_label_rows()
        band_gaps = [r["band_gap"] for r in rows]
        fes = [r["formation_energy_per_atom"] for r in rows]
        sources = {}
        for r in rows:
            src = r.get("source", "mp")
            sources[src] = sources.get(src, 0) + 1
        return {
            "total": len(rows),
            "band_gap_mean": sum(band_gaps) / len(band_gaps) if band_gaps else 0,
            "band_gap_max": max(band_gaps) if band_gaps else 0,
            "formation_energy_mean": sum(fes) / len(fes) if fes else 0,
            "sources": sources,
        }
