"""
Download oxide structures from the AFLOW database for Stage 1 pre-training.

AFLOW (Automatic FLOW for Materials Discovery) hosts ~3.5M DFT-computed
structures. We query only:
  - Oxides (must contain 'O' in species list)
  - Finite band gap (Egap_fit > band_gap_min)
  - ICSD-derived prototypes (higher quality, experimental geometry basis)

Labels used as pre-training targets:
  - Egap_fit      →  band_gap  (eV)
  - enthalpy_formation_atom  →  formation_energy_per_atom  (eV/atom)

Requirements:
    pip install aflow requests

AFLOW REST API docs: https://aflow.org/documentation/AFLOWLIB_REST_API/
"""

from __future__ import annotations

import os
import csv
import logging
import time
from io import StringIO
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# AFLOW REST API endpoint
_AFLOW_API = "http://aflow.org/API/aflowlib.py"
_AFLOW_ENTRY = "http://aflow.org/API"
_PAGE_SIZE = 1000   # records per API page


def download_aflow_oxides(
    output_dir: str,
    labels_csv: str,
    max_structures: Optional[int] = 10000,
    band_gap_min: float = 0.01,
    catalog: str = "ICSD",
    request_delay: float = 0.5,
) -> int:
    """
    Download oxide CIFs and labels from the AFLOW database.

    Args:
        output_dir: Directory to write .cif files.
        labels_csv: Path to write/append labels CSV.
        max_structures: Max structures to download (None = all, could be 100k+).
        band_gap_min: Minimum band gap in eV (filters out metals).
        catalog: AFLOW catalog — "ICSD" (experimental prototypes, higher quality)
                 or "LIB1"/"LIB2"/"LIB3" (enumeration libraries).
        request_delay: Seconds to wait between HTTP pages (be polite to server).

    Returns:
        Number of structures newly saved.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.core import Structure
    except ImportError:
        raise ImportError("pip install pymatgen")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_csv).parent.mkdir(parents=True, exist_ok=True)

    from src.data.mp_downloader import load_labels_csv, _append_rows
    existing_rows = load_labels_csv(labels_csv)
    existing_ids: set[str] = {r["material_id"] for r in existing_rows}

    new_rows: list[dict] = []
    total_fetched = 0
    page = 1

    logger.info(f"Querying AFLOW ({catalog}) for oxides with Egap_fit > {band_gap_min} eV...")

    while True:
        paging_start = (page - 1) * _PAGE_SIZE + 1

        # Build AFLOW REST query
        params = {
            "format": "aflow",
            "catalog": catalog,
            "NSPECIES": "2,3,4,5",        # binary to quinary
            "species": "O",               # must contain oxygen
            f"Egap_fit": f"[{band_gap_min},999]",
            "$select": "compound,auid,Egap_fit,enthalpy_formation_atom,files",
            "paging": str(page),
        }

        try:
            resp = requests.get(_AFLOW_API, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"AFLOW API request failed (page {page}): {e}")
            break

        data = resp.json()
        if not data or not isinstance(data, list):
            logger.info("No more data from AFLOW.")
            break

        entries = data if isinstance(data[0], dict) else [data]

        for entry in entries:
            auid = entry.get("auid", "")
            compound = entry.get("compound", "")
            egap = entry.get("Egap_fit")
            hf = entry.get("enthalpy_formation_atom")

            if not auid or egap is None or hf is None:
                continue

            # Use auid as material ID, prefixed with "aflow-"
            mat_id = f"aflow-{auid.replace(':', '-')}"
            if mat_id in existing_ids:
                continue

            cif_path = os.path.join(output_dir, f"{mat_id}.cif")
            if not os.path.exists(cif_path):
                # Try to download CIF from AFLOW entry URL
                success = _download_aflow_cif(
                    auid=auid,
                    files=entry.get("files", ""),
                    cif_path=cif_path,
                )
                if not success:
                    continue

            new_rows.append({
                "material_id": mat_id,
                "formula": compound,
                "band_gap": float(egap),
                "formation_energy_per_atom": float(hf),
                "cif_path": cif_path,
                "source": f"aflow_{catalog}",
            })
            existing_ids.add(mat_id)
            total_fetched += 1

            if max_structures and total_fetched >= max_structures:
                break

        logger.info(f"  Page {page}: {len(entries)} entries, {total_fetched} total saved.")

        if max_structures and total_fetched >= max_structures:
            break
        if len(entries) < _PAGE_SIZE:
            break  # last page

        page += 1
        time.sleep(request_delay)

    if new_rows:
        _append_rows(existing_rows, new_rows, labels_csv)
        logger.info(f"AFLOW: {len(new_rows)} new structures saved.")
    else:
        logger.info("AFLOW: 0 new structures (all already downloaded or query empty).")

    return len(new_rows)


def _download_aflow_cif(auid: str, files: str, cif_path: str) -> bool:
    """
    Download a CIF file for an AFLOW entry.

    AFLOW entries have a URL like:
        http://aflow.org/API/<catalog>/<auid>/
    CIF files are listed in the `files` field as comma-separated names.
    We look for *_CONTCAR.cif or POSCAR.cif or CONTCAR files.
    """
    try:
        import requests
        from pymatgen.io.vasp import Poscar
        from pymatgen.io.cif import CifWriter
    except ImportError:
        return False

    # Build entry URL from auid
    # auid format: "aflow:aXXXXXXXXXXXXXXXX" or "catalog/path/to/entry"
    # Try direct CIF download from AFLOW files list
    file_list = [f.strip() for f in files.split(",") if f.strip()]
    cif_candidates = [f for f in file_list
                      if f.endswith(".cif") or "CONTCAR" in f or "POSCAR" in f]

    # Construct base URL for the entry
    # AFLOW URLs: http://aflow.org/API/aflowlib.py?auid=<auid>&format=json
    # CIF file: http://aflow.org/API/<path>/<filename>
    # The auid encodes the path: "aflow:aXXX" → look up entry URL
    entry_url_base = f"http://aflow.org/API/aflowlib.py?auid={auid}&format=aflow"

    try:
        resp = requests.get(entry_url_base, timeout=20)
        resp.raise_for_status()
        entry_data = resp.json()
        if isinstance(entry_data, list) and entry_data:
            entry_data = entry_data[0]
        aurl = entry_data.get("aurl", "")
    except Exception:
        return False

    if not aurl:
        return False

    # aurl format: "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/..."
    # Convert to HTTP URL
    aurl_http = "http://" + aurl.replace(":", "/", 1)

    for fname in cif_candidates:
        file_url = f"{aurl_http}/{fname}"
        try:
            r = requests.get(file_url, timeout=20)
            r.raise_for_status()
            content = r.text

            if fname.endswith(".cif"):
                with open(cif_path, "w") as f:
                    f.write(content)
                return True
            else:
                # POSCAR/CONTCAR → convert to CIF via pymatgen
                from pymatgen.io.vasp import Poscar
                from pymatgen.io.cif import CifWriter
                from io import StringIO
                try:
                    structure = Poscar.from_str(content).structure
                    CifWriter(structure).write_file(cif_path)
                    return True
                except Exception:
                    continue
        except Exception:
            continue

    return False
