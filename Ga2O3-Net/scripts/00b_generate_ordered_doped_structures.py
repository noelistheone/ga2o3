"""Phase 50.A — Generate ORDERED doped β-Ga₂O₃ structures for every
unique (element, concentration) DopantSpec in the experimental CSV.

For each unique cache_key:
  • pick the smallest supercell where round(conc*N_Ga) ≥ 1
  • Ewald-minimise the dopant placement
  • CHGNet-relax with FIRE optimiser (F<0.1 eV/Å, ≤200 steps)
  • cache to data/structures/ordered/<cache_key>.relaxed.cif

Wall time on RTX 3090: ~3-4 hours for 170 unique specs.

Usage:
    conda activate ga2o3
    python scripts/00b_generate_ordered_doped_structures.py
    python scripts/00b_generate_ordered_doped_structures.py --no-relax  # skip CHGNet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.experimental_dataset import _parse_row_spec  # noqa: E402
from src.data.ordered_structure_builder import (  # noqa: E402
    build_ordered_doped_structure, manifest_save,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv")
    p.add_argument("--base-cif", default="data/structures/Ga2O3_base.cif")
    p.add_argument("--out-dir", default="data/structures/ordered")
    p.add_argument("--manifest", default="data/structures/ordered/manifest.json")
    p.add_argument("--no-relax", action="store_true",
                   help="Skip CHGNet relaxation (10× faster, less physical)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N unique specs (for testing)")
    p.add_argument("--resume", action="store_true",
                   help="Skip specs already in manifest.json")
    args = p.parse_args()

    csv_path = (PROJ / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    base_cif = (PROJ / args.base_cif).resolve() if not Path(args.base_cif).is_absolute() else Path(args.base_cif)
    out_dir = (PROJ / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    manifest_path = (PROJ / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest for resume
    existing: dict[str, dict] = {}
    if args.resume and manifest_path.exists():
        import json as _json
        with open(manifest_path) as f:
            entries = _json.load(f)
        existing = {e["cache_key"]: e for e in entries
                    if e.get("cif_path") and Path(e["cif_path"]).exists()}
        logger.info(f"Resume: {len(existing)} existing entries skipped")

    # Filter rows like V5
    df = pd.read_csv(csv_path)
    if "usable_flag" in df.columns:
        df = df[df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    df = df[~df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    logger.info(f"{len(df)} rows after standard V5 filters")

    # Build unique cache_keys
    specs_by_key: dict[str, tuple[str, float]] = {}
    for _, row in df.iterrows():
        spec = _parse_row_spec(row)
        key = spec.cache_key()
        if key in specs_by_key:
            continue
        if spec.is_undoped:
            specs_by_key[key] = ("undoped", 0.0)
        else:
            elem = spec.components[0].cation if len(spec.components) == 1 else \
                   spec.components[0].cation
            conc = float(spec.total_conc)
            specs_by_key[key] = (elem, conc)
    logger.info(f"{len(specs_by_key)} unique structures")

    if args.limit:
        keys_subset = list(specs_by_key.keys())[:args.limit]
        specs_by_key = {k: specs_by_key[k] for k in keys_subset}
        logger.info(f"Limited to first {len(specs_by_key)} (--limit)")

    # Init CHGNet optimizer once
    optimizer = None
    if not args.no_relax:
        if args.gpu >= 0 and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        from chgnet.model import StructOptimizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"CHGNet StructOptimizer on {device}")
        optimizer = StructOptimizer(use_device=device)

    manifests = []
    t0 = time.time()
    n_relaxed = 0
    n_failed = 0
    items = sorted(specs_by_key.items())
    for i, (key, (element, conc_frac)) in enumerate(items):
        if key in existing:
            from src.data.ordered_structure_builder import OrderedStructureManifest
            manifests.append(OrderedStructureManifest(**{
                k: v for k, v in existing[key].items()
                if k in OrderedStructureManifest.__dataclass_fields__
            }))
            continue
        try:
            warnings.filterwarnings("ignore")
            m = build_ordered_doped_structure(
                cache_key=key, element=element, conc_frac=conc_frac,
                base_cif=base_cif, out_dir=out_dir,
                relax=(not args.no_relax), optimizer=optimizer,
            )
            manifests.append(m)
            if m.error:
                n_failed += 1
                logger.warning(f"  [{i+1}/{len(items)}] {key}: {m.error}")
            elif m.relaxed:
                n_relaxed += 1
        except Exception as e:
            n_failed += 1
            logger.error(f"  [{i+1}/{len(items)}] {key}: "
                         f"{type(e).__name__}: {e}")
            from src.data.ordered_structure_builder import OrderedStructureManifest
            manifests.append(OrderedStructureManifest(
                cache_key=key, element=element,
                target_conc_frac=conc_frac, supercell="?",
                n_atoms=0, n_ga=0, n_dopant_substituted=0,
                achieved_conc_frac=0.0, relaxed=False,
                relax_steps=0, relax_max_force=None, relax_energy=None,
                cif_path="", relaxed_cif_path=None,
                error=f"{type(e).__name__}: {e}",
            ))
        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(items) - i - 1) / max(rate, 1e-3)
            logger.info(f"  [{i+1}/{len(items)}] elapsed {elapsed/60:.1f} min, "
                        f"{rate*60:.1f}/min, ETA {eta/60:.1f} min, "
                        f"relaxed {n_relaxed}, failed {n_failed}")
            # Periodic manifest dump for crash recovery
            manifest_save(manifests, manifest_path)

    manifest_save(manifests, manifest_path)
    logger.info(f"Done. Total {len(manifests)} structures, "
                f"{n_relaxed} relaxed, {n_failed} failed. "
                f"Wall time: {(time.time()-t0)/60:.1f} min")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
