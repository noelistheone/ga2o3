"""
Script 01: Download oxide structures for Stage 1 pre-training.

Downloads two complementary pools from Materials Project:
  Pool A — broad oxide dataset (band_gap > 0, ~35k structures)
            Remove --band-gap-filter to get ~80k (includes metals)
  Pool B — targeted multi-element systems (Fe-Ga-O, Sn-Ga-O, Mg-Ga-O, …)

Optionally downloads additional structures from AFLOW (pool C):
  Pool C — AFLOW ICSD oxides (independent DFT data, ~10k–100k structures)
            Requires: pip install requests

Data scale reference:
  MP oxides (band_gap > 0.01 eV)  →  ~35 000 structures
  MP oxides (no band_gap filter)   →  ~80 000 structures
  MP all stable structures         → ~150 000 structures
  AFLOW ICSD oxides                →  ~30 000–100 000 structures

Usage:
    conda activate ga2o3
    # Standard run (MP oxides only, ~35k):
    python scripts/01_download_mp.py

    # Maximum MP oxides (~80k, includes metallic oxides):
    python scripts/01_download_mp.py --no-band-gap-filter --max-broad 0

    # All MP oxides + targeted + AFLOW:
    python scripts/01_download_mp.py --no-band-gap-filter --aflow --aflow-max 20000

    # Quick test (500 structures):
    python scripts/01_download_mp.py --max-broad 500 --no-targeted

    # Add new element to Pool B (incremental, safe to re-run):
    python scripts/01_download_mp.py --extra-chemsys "Cu-Ga-O" --max-broad 0
"""

from __future__ import annotations

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.mp_downloader import (
    download_mp_oxides,
    download_mp_all,
    download_mp_chemsys,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Pool B: targeted ternary/quaternary systems for Ga₂O₃ doping context
TARGETED_CHEMSYS = [
    "Fe-Ga-O",
    "Sn-Ga-O",
    "Mg-Ga-O",
    "Zn-Ga-O",
    "Fe-Sn-Ga-O",   # co-doped
    "Fe-Mg-Ga-O",
    "Sn-Mg-Ga-O",
    "Ga-O",          # undoped Ga₂O₃ polymorphs
    "In-Ga-O",       # related transparent conductor
    "Al-Ga-O",       # related wide-bandgap oxide
]


def main():
    parser = argparse.ArgumentParser(
        description="Download oxide structures for Ga2O3-Net pre-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/pretrain.yaml")

    # ── Pool A: broad MP oxides ──────────────────────────────────────────────
    pool_a = parser.add_argument_group("Pool A — broad MP oxide search")
    pool_a.add_argument(
        "--max-broad", type=int, default=None,
        metavar="N",
        help="Max structures from broad oxide search (default: all available ~35k). "
             "Set 0 to skip Pool A entirely.",
    )
    pool_a.add_argument(
        "--no-band-gap-filter", action="store_true",
        help="Remove band_gap > 0 filter — roughly doubles dataset (~80k oxides) "
             "by including metallic oxides. Formation energy is still a valid target.",
    )
    pool_a.add_argument(
        "--all-mp", action="store_true",
        help="Download ALL stable MP structures (not just oxides). ~150k structures. "
             "Overrides --no-band-gap-filter and element=O filter.",
    )

    # ── Pool B: targeted chemsys ─────────────────────────────────────────────
    pool_b = parser.add_argument_group("Pool B — targeted chemical systems")
    pool_b.add_argument(
        "--no-targeted", action="store_true",
        help="Skip Pool B (targeted gallate chemical systems).",
    )
    pool_b.add_argument(
        "--max-targeted", type=int, default=500,
        metavar="N",
        help="Max structures per targeted chemical system (default 500).",
    )
    pool_b.add_argument(
        "--extra-chemsys", nargs="*", default=[],
        metavar="CHEMSYS",
        help='Extra chemical systems, e.g. --extra-chemsys "Cu-Ga-O" "Ti-Ga-O"',
    )

    # ── Pool C: AFLOW ───────────────────────────────────────────────────────
    pool_c = parser.add_argument_group("Pool C — AFLOW database (optional)")
    pool_c.add_argument(
        "--aflow", action="store_true",
        help="Also download oxide structures from AFLOW (ICSD catalog). "
             "Requires: pip install requests",
    )
    pool_c.add_argument(
        "--aflow-max", type=int, default=10000,
        metavar="N",
        help="Max AFLOW structures to download (default 10000).",
    )
    pool_c.add_argument(
        "--aflow-band-gap-min", type=float, default=0.01,
        help="Min band gap for AFLOW query (default 0.01 eV).",
    )

    # ── General ──────────────────────────────────────────────────────────────
    parser.add_argument("--api-key", default=None,
                        help="MP API key (overrides MP_API_KEY env var).")
    parser.add_argument("--band-gap-min", type=float, default=0.01,
                        help="Min band gap for Pool A (default 0.01 eV). "
                             "Ignored if --no-band-gap-filter is set.")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key and not args.aflow:
        logger.error("MP_API_KEY not set. Export it or pass --api-key.")
        sys.exit(1)

    output_dir = config["paths"]["raw_data"]
    labels_csv = config["paths"]["labels_csv"]
    total = 0

    # ── Pool A ──────────────────────────────────────────────────────────────
    skip_a = (args.max_broad == 0)
    if not skip_a and api_key:
        logger.info("=== Pool A: broad oxide download ===")

        if args.all_mp:
            logger.info("Mode: ALL stable MP structures (no element filter).")
            n = download_mp_all(
                api_key=api_key,
                output_dir=output_dir,
                labels_csv=labels_csv,
                max_structures=args.max_broad,
            )
        else:
            band_gap_filter = None if args.no_band_gap_filter else args.band_gap_min
            if band_gap_filter is None:
                logger.info("Band gap filter: DISABLED (~80k oxide structures expected).")
            else:
                logger.info(f"Band gap filter: > {band_gap_filter} eV (~35k structures expected).")
            n = download_mp_oxides(
                api_key=api_key,
                output_dir=output_dir,
                labels_csv=labels_csv,
                max_structures=args.max_broad,
                band_gap_min=band_gap_filter,
            )

        total += n
        logger.info(f"Pool A complete: {n} new structures.")

    # ── Pool B ──────────────────────────────────────────────────────────────
    if not args.no_targeted and api_key:
        chemsys_list = TARGETED_CHEMSYS + [
            c for c in args.extra_chemsys if c not in TARGETED_CHEMSYS
        ]
        logger.info(f"=== Pool B: targeted chemsys ({len(chemsys_list)} systems) ===")
        n = download_mp_chemsys(
            api_key=api_key,
            chemsys_list=chemsys_list,
            output_dir=output_dir,
            labels_csv=labels_csv,
            max_per_chemsys=args.max_targeted,
        )
        total += n
        logger.info(f"Pool B complete: {n} new structures.")

    # ── Pool C: AFLOW ────────────────────────────────────────────────────────
    if args.aflow:
        logger.info(f"=== Pool C: AFLOW ICSD oxides (max {args.aflow_max}) ===")
        try:
            from src.data.aflow_downloader import download_aflow_oxides
            n = download_aflow_oxides(
                output_dir=output_dir,
                labels_csv=labels_csv,
                max_structures=args.aflow_max,
                band_gap_min=args.aflow_band_gap_min,
                catalog="ICSD",
            )
            total += n
            logger.info(f"Pool C complete: {n} new structures.")
        except ImportError as e:
            logger.error(f"AFLOW download failed: {e}")
            logger.error("Install with: pip install requests")

    logger.info(f"=== Total new structures this run: {total} ===")
    logger.info(f"CIFs saved to: {output_dir}")
    logger.info(f"Labels CSV:    {labels_csv}")
    logger.info("")
    logger.info("Next step:")
    logger.info("  python scripts/02_pretrain_encoder.py --gpu 0")


if __name__ == "__main__":
    main()
