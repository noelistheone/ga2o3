"""Phase 55 V55-MM — MACE-MP-0 Frozen Embeddings for β-Ga2O3 Doped Supercells.

Uses the MACE-MP-0 universal foundation model (89 elements, trained on 1.6M
bulk crystals in MPTrj per Batatia et al. arXiv:2401.00096) to extract
detach()-only chemistry-rich embeddings for each unique dopant in the
training dataset.

Pipeline:
  1. Load MACE-MP-0 medium checkpoint via mace-torch
  2. For each unique (dopant, dopant_at_pct) row, build a β-Ga2O3 supercell
     (96 atoms) with the dopant substituted at 1-2 Ga sites
  3. Run MACE-MP-0 forward → extract last-layer atomic features (per atom)
  4. Pool: mean over Ga + mean over dopant + global mean → 3 × node_feat_dim
  5. Cache to data/processed/mace_mp_features.npz

Output: per-row MACE-MP embedding vector (~384-d after concat) usable as
auxiliary feature for V54-A2 retrain.

Falsification gate (V55 Roadmap §V55-MM):
  - V54-A2 + MACE retrain VC R²_Platt ≥ 0.774 (V54-A2 0.754 + 0.02)

Note: MACE-MP-0 medium uses ~1M params on MPTrj 1.6M oxide-rich crystals;
expected to be especially useful for unseen elements (Hf/Sc/Y/RE in V54-E1).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v55_mm")


def _load_mace_mp(device: str = "cuda:0", model_size: str = "medium"):
    """Load MACE-MP-0 foundation model (auto-downloads first time)."""
    from mace.calculators import mace_mp
    logger.info(f"Loading MACE-MP-0 {model_size} on {device}...")
    calc = mace_mp(model=model_size, device=device, default_dtype="float32")
    logger.info("MACE-MP-0 loaded.")
    return calc


def _build_doped_supercell(host_cif: Path, dopant_symbol: str,
                            target_dopant_count: int = 1):
    """Build β-Ga2O3 supercell with `target_dopant_count` Ga atoms replaced by dopant.

    Uses pymatgen Structure for clean atom substitution.
    """
    from pymatgen.core import Structure
    struct = Structure.from_file(str(host_cif))
    # Replicate to ~80-100 atoms
    if len(struct) < 60:
        struct.make_supercell([2, 1, 2])
    # Substitute Ga -> dopant
    ga_indices = [i for i, site in enumerate(struct.sites) if site.specie.symbol == "Ga"]
    if not ga_indices:
        raise ValueError(f"No Ga sites in {host_cif}")
    # Take first N Ga indices (deterministic)
    n_sub = min(target_dopant_count, len(ga_indices))
    for idx in ga_indices[:n_sub]:
        struct.replace(idx, dopant_symbol)
    return struct


def _extract_mace_embedding(calc, atoms_or_struct, return_per_node: bool = False):
    """Run MACE-MP-0 on an ASE Atoms or pymatgen Structure; return embeddings.

    Returns dict with:
      mean_all   = (node_feat_dim,)     global mean over all atoms
      mean_ga    = (node_feat_dim,)     mean over Ga atoms only
      mean_dop   = (node_feat_dim,)     mean over dopant atom(s)
      mean_o     = (node_feat_dim,)     mean over O atoms
      node_feats = (n_atoms, node_feat_dim)  per-atom features (if return_per_node)
    """
    from ase import Atoms
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    if isinstance(atoms_or_struct, Structure):
        atoms = AseAtomsAdaptor().get_atoms(atoms_or_struct)
    else:
        atoms = atoms_or_struct

    # Use MACE's get_descriptors or invariant_node_features
    atoms.calc = calc
    descriptors = calc.get_descriptors(atoms)  # (n_atoms, descriptor_dim)
    # descriptors has dtype float32 of shape [n_atoms, feat_dim]
    symbols = atoms.get_chemical_symbols()
    descriptors = np.asarray(descriptors)
    if descriptors.ndim != 2:
        raise RuntimeError(f"Unexpected MACE descriptor shape: {descriptors.shape}")

    n_atoms, feat_dim = descriptors.shape
    # Get atom-type masks
    sym_arr = np.array(symbols)
    mask_ga = sym_arr == "Ga"
    mask_o = sym_arr == "O"
    mask_dop = ~(mask_ga | mask_o)

    def safe_mean(arr, mask):
        if mask.sum() == 0:
            return np.zeros(feat_dim, dtype=np.float32)
        return arr[mask].mean(axis=0).astype(np.float32)

    out = {
        "mean_all": descriptors.mean(axis=0).astype(np.float32),
        "mean_ga":  safe_mean(descriptors, mask_ga),
        "mean_dop": safe_mean(descriptors, mask_dop),
        "mean_o":   safe_mean(descriptors, mask_o),
        "node_feat_dim": int(feat_dim),
    }
    if return_per_node:
        out["node_feats"] = descriptors.astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="V55-MM MACE-MP-0 feature extraction")
    parser.add_argument("--csv", type=Path,
                        default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")
    parser.add_argument("--host-cif", type=Path,
                        default=PROJ / "data" / "structures" / "Ga2O3_base.cif")
    parser.add_argument("--output", type=Path,
                        default=PROJ / "data" / "processed" / "mace_mp_features.npz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-size", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--max-elements", type=int, default=0,
                        help="Cap unique dopants (0 = all)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV + collect unique dopants
    df = pd.read_csv(args.csv)
    if "usable_flag" in df.columns:
        df = df[df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    df = df[~df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    elements = sorted(df["element"].dropna().unique())
    # Filter to single-element dopants in MACE-MP-0 species (89 elements)
    valid_elements = []
    for e in elements:
        s = str(e).strip()
        if "+" in s or " " in s or s in ("undoped", "other", "nan"):
            continue
        valid_elements.append(s)
    if args.max_elements > 0:
        valid_elements = valid_elements[:args.max_elements]
    logger.info(f"Unique dopant elements ({len(valid_elements)}): {valid_elements}")

    # Load MACE-MP-0
    calc = _load_mace_mp(device=args.device, model_size=args.model_size)

    # For each dopant, build supercell + extract embedding
    embeddings = {}
    t0 = time.time()
    for elem in valid_elements:
        try:
            struct = _build_doped_supercell(args.host_cif, elem, target_dopant_count=1)
            emb_dict = _extract_mace_embedding(calc, struct)
            embeddings[elem] = emb_dict
            elapsed = time.time() - t0
            logger.info(f"  [{len(embeddings)}/{len(valid_elements)}] {elem}: "
                        f"node_feat_dim={emb_dict['node_feat_dim']} ({elapsed:.0f}s)")
        except Exception as exc:
            logger.warning(f"  {elem}: FAILED - {exc}")
            continue

    # Save NPZ — one row per dopant
    if embeddings:
        save_dict = {
            "elements": np.array(list(embeddings.keys())),
        }
        for key in ("mean_all", "mean_ga", "mean_dop", "mean_o"):
            arr = np.stack([embeddings[e][key] for e in save_dict["elements"]])
            save_dict[key] = arr.astype(np.float32)
        save_dict["node_feat_dim"] = np.array([list(embeddings.values())[0]["node_feat_dim"]])

        np.savez(args.output, **save_dict)
        logger.info(f"\nWrote MACE-MP-0 features for {len(embeddings)} elements to {args.output}")
        logger.info(f"  Shapes: mean_all={save_dict['mean_all'].shape}, "
                    f"mean_dop={save_dict['mean_dop'].shape}")
    else:
        logger.error("No embeddings extracted; nothing to save")


if __name__ == "__main__":
    main()
