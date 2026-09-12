"""
Download oxide structures from the Materials Project database.
Saves CIF files and a CSV of labels (band_gap, formation_energy_per_atom).

Scale guide (approximate MP counts as of 2024):
  elements=["O"], band_gap > 0.01 eV   →  ~35 000 structures
  elements=["O"], no band_gap filter    →  ~80 000 structures (includes metals)
  all structures (no element filter)    → ~150 000 structures
"""

from __future__ import annotations

import os
import csv
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def download_mp_oxides(
    api_key: str,
    output_dir: str,
    labels_csv: str,
    max_structures: Optional[int] = None,
    band_gap_min: Optional[float] = 0.01,
) -> int:
    """
    Download oxide structures from Materials Project.

    Args:
        api_key: MP API key from materialsproject.org.
        output_dir: Directory to write .cif files.
        labels_csv: Path to write/append labels CSV.
        max_structures: Cap on number of structures (None = all available, ~35k–80k).
        band_gap_min: Minimum band gap (eV); set to None to include metals.
                      Setting None roughly doubles the dataset (~80k oxides).

    Returns:
        Number of structures newly saved.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load existing IDs so incremental runs skip already-downloaded CIFs
    existing_ids = _load_existing_ids(labels_csv)

    logger.info("Connecting to Materials Project (oxide query)...")
    bg_filter = (band_gap_min, None) if band_gap_min is not None else None

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=["O"],
            band_gap=bg_filter,
            fields=["material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "structure"],
        )

    logger.info(f"MP returned {len(docs)} oxide documents.")

    if max_structures is not None:
        docs = docs[:max_structures]
        logger.info(f"Capping at {max_structures} structures.")

    return _save_docs(docs, output_dir, labels_csv, existing_ids, desc="MP oxides")


def download_mp_all(
    api_key: str,
    output_dir: str,
    labels_csv: str,
    max_structures: Optional[int] = None,
    formation_energy_max: float = 0.0,
) -> int:
    """
    Download ALL thermodynamically stable MP structures (not just oxides).

    Uses formation_energy_per_atom ≤ formation_energy_max as stability proxy.
    Stable structures (≤ 0 eV/atom) number ~60k; relaxing to 0.1 gives ~80k.

    This gives the encoder broader chemistry knowledge but may introduce noise
    from non-oxide crystal chemistry. Recommended only when you have > 10k
    Ga₂O₃-related experimental samples for fine-tuning.

    Args:
        formation_energy_max: Stability cutoff in eV/atom (default 0.0 = on hull).
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_csv).parent.mkdir(parents=True, exist_ok=True)

    existing_ids = _load_existing_ids(labels_csv)

    logger.info("Connecting to Materials Project (all structures query)...")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formation_energy_per_atom=(None, formation_energy_max),
            fields=["material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "structure"],
        )

    logger.info(f"MP returned {len(docs)} stable structures.")
    if max_structures is not None:
        docs = docs[:max_structures]
        logger.info(f"Capping at {max_structures} structures.")

    return _save_docs(docs, output_dir, labels_csv, existing_ids, desc="MP all")


def download_mp_chemsys(
    api_key: str,
    chemsys_list: list[str],
    output_dir: str,
    labels_csv: str,
    max_per_chemsys: int = 500,
) -> int:
    """
    Download structures for specific multi-element chemical systems.

    New entries are appended to `labels_csv` (deduplicating by material_id).

    Args:
        chemsys_list: e.g. ["Fe-Ga-O", "Sn-Ga-O"].
        max_per_chemsys: Max structures per chemical system.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_csv).parent.mkdir(parents=True, exist_ok=True)

    existing_ids = _load_existing_ids(labels_csv)
    existing_rows = load_labels_csv(labels_csv) if Path(labels_csv).exists() else []
    new_rows: list[dict] = []
    total_saved = 0

    with MPRester(api_key) as mpr:
        for chemsys in chemsys_list:
            logger.info(f"  Querying {chemsys} ...")
            try:
                docs = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=["material_id", "formula_pretty", "band_gap",
                            "formation_energy_per_atom", "structure"],
                )
            except Exception as e:
                logger.warning(f"  Query failed for {chemsys}: {e}")
                continue

            if max_per_chemsys:
                docs = docs[:max_per_chemsys]

            saved_this = 0
            for doc in tqdm(docs, desc=f"  {chemsys}", leave=False):
                mp_id = doc.material_id
                if mp_id in existing_ids:
                    continue
                cif_path = os.path.join(output_dir, f"{mp_id}.cif")
                if not os.path.exists(cif_path):
                    try:
                        doc.structure.to(fmt="cif", filename=cif_path)
                    except Exception as e:
                        logger.warning(f"  Failed to save {mp_id}: {e}")
                        continue
                new_rows.append({
                    "material_id": mp_id,
                    "formula": doc.formula_pretty,
                    "band_gap": doc.band_gap if doc.band_gap else 0.0,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "cif_path": cif_path,
                    "source": "mp_chemsys",
                })
                existing_ids.add(mp_id)
                saved_this += 1
                total_saved += 1
            logger.info(f"  {chemsys}: {saved_this} new structures")

    if new_rows:
        _append_rows(existing_rows, new_rows, labels_csv)
        logger.info(f"Labels CSV updated: {len(existing_rows) + len(new_rows)} total entries")

    return total_saved


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_existing_ids(labels_csv: str) -> set[str]:
    if not Path(labels_csv).exists():
        return set()
    rows = load_labels_csv(labels_csv)
    return {r["material_id"] for r in rows}


def _save_docs(docs, output_dir: str, labels_csv: str,
               existing_ids: set[str], desc: str = "MP") -> int:
    """Write CIFs and update labels CSV. Skips already-downloaded IDs."""
    existing_rows = load_labels_csv(labels_csv) if Path(labels_csv).exists() else []
    new_rows: list[dict] = []
    skipped = 0

    for doc in tqdm(docs, desc=desc):
        mp_id = doc.material_id
        if mp_id in existing_ids:
            skipped += 1
            continue
        cif_path = os.path.join(output_dir, f"{mp_id}.cif")
        if not os.path.exists(cif_path):
            try:
                doc.structure.to(fmt="cif", filename=cif_path)
            except Exception as e:
                logger.warning(f"Failed to save {mp_id}: {e}")
                continue
        new_rows.append({
            "material_id": mp_id,
            "formula": doc.formula_pretty,
            "band_gap": doc.band_gap if doc.band_gap is not None else 0.0,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "cif_path": cif_path,
            "source": "mp",
        })
        existing_ids.add(mp_id)

    if new_rows:
        _append_rows(existing_rows, new_rows, labels_csv)

    logger.info(f"{desc}: {len(new_rows)} new, {skipped} skipped (already downloaded).")
    return len(new_rows)


_LABELS_FIELDNAMES = [
    "material_id", "formula", "band_gap",
    "formation_energy_per_atom", "cif_path", "source",
]


def _append_rows(existing: list[dict], new: list[dict], labels_csv: str):
    # Strip None keys left by csv.DictReader when a row has more fields than headers
    clean_existing = [{k: v for k, v in row.items() if k is not None}
                      for row in existing]
    all_rows = clean_existing + new
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_LABELS_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(all_rows)


def load_labels_csv(labels_csv: str) -> list[dict]:
    """Load the labels CSV into a list of dicts."""
    if not Path(labels_csv).exists():
        return []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    good_rows = []
    for row in rows:
        try:
            row["band_gap"] = float(row.get("band_gap") or 0.0)
            row["formation_energy_per_atom"] = float(
                row.get("formation_energy_per_atom") or 0.0
            )
            good_rows.append(row)
        except (ValueError, TypeError):
            # Skip corrupted rows (e.g. two CSV rows merged into one)
            pass
    return good_rows
