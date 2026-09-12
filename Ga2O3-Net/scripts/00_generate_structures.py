"""
Script 00: Download β-Ga₂O₃ base structure from Materials Project and
           generate doped unit cells for all dopant specifications.

Supports:
  Single element    : "Fe"  or  "Fe:0.0265"
  Multi-element     : "Fe:0.0133,Sn:0.0132"
  Compound          : "SnO2:0.0265"
  Multi-compound    : "SnO2:0.013,MgO:0.013"
  Arbitrary extras  : --extra "Ti:0.02" "Cu:0.015,In:0.015"

Generated files go to data/structures/:
  Ga2O3_base.cif
  Fe-0.0265.cif
  Fe-0.0133_Sn-0.0132.cif
  ...

Usage:
    conda activate ga2o3
    python scripts/00_generate_structures.py

    # Add extra specs:
    python scripts/00_generate_structures.py --extra "Ti:0.02" "Cu:0.015,In:0.015"

    # Overwrite existing files:
    python scripts/00_generate_structures.py --overwrite
"""

from __future__ import annotations

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BETA_GA2O3_MP_ID = "mp-886"

# Default specs derived from the experimental CSV
DEFAULT_SPECS = [
    "Fe:0.0265",
    "Sn:0.0265",
    "Mg:0.0265",
    "Zn:0.0265",
]


def download_base(api_key: str, out_dir: str) -> str:
    """Download undoped β-Ga₂O₃ from MP, return CIF path."""
    from mp_api.client import MPRester

    out_path = os.path.join(out_dir, "Ga2O3_base.cif")
    if os.path.exists(out_path):
        logger.info(f"Base structure already exists: {out_path}")
        return out_path

    logger.info(f"Downloading β-Ga₂O₃ ({BETA_GA2O3_MP_ID}) ...")
    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(BETA_GA2O3_MP_ID)

    structure.to(fmt="cif", filename=out_path)
    logger.info(
        f"Saved → {out_path}  ({structure.formula}, "
        f"a={structure.lattice.a:.3f} b={structure.lattice.b:.3f} "
        f"c={structure.lattice.c:.3f} β={structure.lattice.beta:.2f}°)"
    )
    return out_path


def generate_cif(spec_str: str, base_cif: str, out_dir: str, overwrite: bool) -> str:
    """Generate a doped Ga₂O₃ CIF from a DopantSpec string."""
    from src.data.dopant_spec import parse_spec
    from src.data.graph_builder import dopant_spec_to_structure

    spec = parse_spec(spec_str)
    out_path = os.path.join(out_dir, spec.to_cif_filename())

    if os.path.exists(out_path) and not overwrite:
        logger.info(f"  skip (exists): {os.path.basename(out_path)}")
        return out_path

    structure = dopant_spec_to_structure(spec, base_cif=base_cif)
    structure.to(fmt="cif", filename=out_path)

    # Also write legacy single-element CIF (e.g. Fe.cif) for backward compat
    if len(spec.components) == 1:
        legacy = os.path.join(out_dir, f"{spec.components[0].cation}.cif")
        if not os.path.exists(legacy) or overwrite:
            structure.to(fmt="cif", filename=legacy)
            logger.info(
                f"  [{spec.label():>20s}]  "
                f"{os.path.basename(out_path)}  +  {os.path.basename(legacy)}"
                f"  ({structure.formula})"
            )
        else:
            logger.info(
                f"  [{spec.label():>20s}]  {os.path.basename(out_path)}"
                f"  ({structure.formula})"
            )
    else:
        logger.info(
            f"  [{spec.label():>20s}]  {os.path.basename(out_path)}"
            f"  ({structure.formula})"
        )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate doped β-Ga₂O₃ CIF files for Ga2O3-Net"
    )
    parser.add_argument("--out-dir", default="data/structures")
    parser.add_argument(
        "--extra", nargs="*", default=[],
        metavar="SPEC",
        help='Extra dopant specs, e.g. --extra "Ti:0.02" "Cu:0.015,In:0.015"',
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        logger.error("MP_API_KEY not set.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Base structure
    base_cif = download_base(api_key, args.out_dir)

    # 2. Generate doped structures
    all_specs = DEFAULT_SPECS + [s for s in args.extra if s not in DEFAULT_SPECS]
    logger.info(f"\nGenerating {len(all_specs)} doped structure(s):")
    for spec_str in all_specs:
        try:
            generate_cif(spec_str.strip(), base_cif, args.out_dir, args.overwrite)
        except Exception as e:
            logger.error(f"  Failed [{spec_str}]: {e}")

    # 3. Summary
    cifs = sorted(f for f in os.listdir(args.out_dir) if f.endswith(".cif"))
    logger.info(f"\nDone. {len(cifs)} CIF file(s) in {args.out_dir}/:")
    for f in cifs:
        logger.info(f"  {f}")


if __name__ == "__main__":
    main()
