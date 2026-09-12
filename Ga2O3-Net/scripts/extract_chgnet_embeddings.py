"""Phase 49 V17 — pre-extract CHGNet `crystal_fea` for every unique
experimental DopantSpec.

CHGNet (https://github.com/CederGroupHub/chgnet) is pretrained on MPtrj
(1.5M oxide structures with energy/force/stress/magmom labels). Its
crystal-level feature (`crystal_fea`, 64-dim) is exactly the same shape as
our existing CGCNN encoder output — clean drop-in replacement.

Usage:
    conda activate ga2o3
    python scripts/extract_chgnet_embeddings.py \
        --csv data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv \
        --out data/processed/chgnet_emb_cache.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.experimental_dataset import _parse_row_spec  # noqa: E402
from src.data.graph_builder import dopant_spec_to_structure  # noqa: E402
from pymatgen.core import Element, Composition  # noqa: E402


def order_disordered(structure):
    """Convert a disordered Ga₂O₃-with-substitution structure to an ordered
    one suitable for CHGNet: replace ONE disordered cation site with the
    dominant non-Ga dopant species, others fall back to Ga.

    For pure (already-ordered) structures, returns a copy unchanged.
    """
    from pymatgen.core import Structure
    s = structure.copy()
    if all(site.is_ordered for site in s.sites):
        return s
    placed_dopant = False
    new_species = []
    for site in s.sites:
        if site.is_ordered:
            new_species.append(site.species)
            continue
        # Disordered: pick majority non-Ga species (the dopant)
        comp = site.species
        non_ga = {sp: occ for sp, occ in comp.items() if str(sp) != "Ga"}
        if non_ga and not placed_dopant:
            dopant_sp = max(non_ga, key=non_ga.get)
            new_species.append(Composition(str(dopant_sp)))
            placed_dopant = True
        else:
            new_species.append(Composition("Ga"))
    return Structure(
        lattice=s.lattice,
        species=new_species,
        coords=[site.frac_coords for site in s.sites],
        coords_are_cartesian=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv")
    p.add_argument("--out", default="data/processed/chgnet_emb_cache.pt")
    p.add_argument("--base-cif", default="data/structures/Ga2O3_base.cif")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    csv_path = (PROJ / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    out_path = (PROJ / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    base_cif = (PROJ / args.base_cif).resolve() if not Path(args.base_cif).is_absolute() else Path(args.base_cif)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "usable_flag" in df.columns:
        df = df[df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    df = df[~df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    print(f"[info] {len(df)} rows after standard V5 filters")

    # Build unique cache_keys
    specs_by_key: dict[str, object] = {}
    for _, row in df.iterrows():
        spec = _parse_row_spec(row)
        specs_by_key.setdefault(spec.cache_key(), spec)
    print(f"[info] {len(specs_by_key)} unique structures to embed")

    # Load CHGNet
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from chgnet.model.model import CHGNet

    print("[info] loading CHGNet...")
    model = CHGNet.load()
    model.eval()
    device = next(model.parameters()).device
    print(f"[info] CHGNet device: {device}")

    cache: dict[str, np.ndarray] = {}
    t0 = time.time()
    n_failed = 0
    for i, (key, spec) in enumerate(sorted(specs_by_key.items())):
        try:
            structure = dopant_spec_to_structure(spec, base_cif=str(base_cif))
            structure = order_disordered(structure)
            # CHGNet needs grad enabled to compute forces internally; we
            # detach the returned crystal_fea immediately.
            out = model.predict_structure(
                structure,
                return_atom_feas=True,
                return_crystal_feas=True,
                return_site_energies=False,
            )
            crystal_fea = out["crystal_fea"]
            if isinstance(crystal_fea, torch.Tensor):
                crystal_fea = crystal_fea.detach().cpu().numpy()
            else:
                crystal_fea = np.asarray(crystal_fea)
            cache[key] = crystal_fea.astype(np.float32).reshape(-1)
        except Exception as e:
            print(f"[warn] {key}: {type(e).__name__}: {e}")
            n_failed += 1
            continue
        if (i + 1) % 25 == 0 or (i + 1) == len(specs_by_key):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  [{i+1}/{len(specs_by_key)}] elapsed {elapsed:.1f}s, "
                  f"{rate:.1f} structs/s")

    if n_failed:
        print(f"[warn] {n_failed} structures failed to embed")

    # Sanity: verify all cached embeddings are 64-dim
    dims = {v.shape[0] for v in cache.values()}
    print(f"[info] embedding dims: {dims}")
    assert dims == {64}, f"unexpected embedding dims: {dims}"

    print(f"[info] saving {len(cache)} embeddings → {out_path}")
    torch.save({
        "version": "chgnet_v0.3.0_crystal_fea",
        "n_structures": len(cache),
        "embedding_dim": 64,
        "embeddings": cache,
    }, out_path)
    print(f"[done] saved {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
