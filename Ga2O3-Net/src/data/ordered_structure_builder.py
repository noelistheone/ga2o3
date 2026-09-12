"""Phase 50.A — Generate ORDERED doped β-Ga₂O₃ structures with
concentration-dependent geometry.

Pipeline (per (element, concentration) spec):
  1. Choose smallest supercell where ⌊conc × N_Ga + 0.5⌋ ≥ 1, so each
     spec maps to a unique integer dopant count. Ladder:
       conc ≤ 1%   → 3×3×2 (180 atoms, 72 Ga, 1.39%/Ga)
       1%–3%       → 3×2×2 (120 atoms, 48 Ga, 2.08%/Ga)
       3%–6%       → 2×2×2  (80 atoms, 32 Ga, 3.13%/Ga)
       6%–12%      → 2×2×1  (40 atoms, 16 Ga, 6.25%/Ga)
       12%–25%     → 2×1×1  (20 atoms,  8 Ga, 12.5%/Ga)
       ≥25%        → 1×1×1  (10 atoms,  4 Ga, 25%/Ga)
  2. Build disordered substitution structure (pymatgen
     SubstitutionTransformation auto-decorated with oxidation states).
  3. Order via Ewald minimisation (`OrderDisorderedStructureTransformation`,
     ALGO_FAST). For very small N_dopant=1 this is essentially "place
     dopant at the lowest-electrostatic-energy Ga site".
  4. Relax with CHGNet (FIRE optimiser, F<0.1 eV/Å, ≤200 steps). This
     gives DFT-quality local distortion around the dopant.

Concentration variation is now encoded structurally via:
  (a) supercell shape (anisotropic),
  (b) number of substituted sites,
  (c) deterministic placement (Ewald → SQS-like for low conc),
  (d) CHGNet-relaxed local distortion (Bi-O > Ga-O > Mg-O bond lengths).

Outputs:
  data/structures/ordered/<cache_key>.cif            — pre-relaxation ordered
  data/structures/ordered/<cache_key>.relaxed.cif    — CHGNet-relaxed
  data/structures/ordered/manifest.json              — per-spec metadata
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import (
    SubstitutionTransformation, OxidationStateDecorationTransformation,
)
from pymatgen.transformations.advanced_transformations import (
    OrderDisorderedStructureTransformation,
)

logger = logging.getLogger(__name__)


SUPERCELL_LADDER = [
    # (max_conc_frac, scaling_matrix, N_Ga, label)
    (0.01,  (3, 3, 2), 72, "3x3x2"),
    (0.03,  (3, 2, 2), 48, "3x2x2"),
    (0.06,  (2, 2, 2), 32, "2x2x2"),
    (0.12,  (2, 2, 1), 16, "2x2x1"),
    (0.25,  (2, 1, 1),  8, "2x1x1"),
    (1.00,  (1, 1, 1),  4, "1x1x1"),
]


@dataclass
class OrderedStructureManifest:
    cache_key: str
    element: str
    target_conc_frac: float
    supercell: str
    n_atoms: int
    n_ga: int
    n_dopant_substituted: int
    achieved_conc_frac: float
    relaxed: bool
    relax_steps: int
    relax_max_force: float | None
    relax_energy: float | None
    cif_path: str
    relaxed_cif_path: str | None
    error: str | None = None


def _pick_supercell(conc_frac: float) -> tuple[tuple[int, int, int], int, str]:
    """Return (scaling, N_Ga_in_supercell, label)."""
    if conc_frac <= 0:
        return (1, 1, 1), 4, "1x1x1"
    for max_c, scaling, n_ga, label in SUPERCELL_LADDER:
        if conc_frac <= max_c:
            # Sanity: this supercell has enough Ga for ≥1 dopant
            if int(np.round(conc_frac * n_ga)) >= 1 or conc_frac < 1e-4:
                return scaling, n_ga, label
    return (1, 1, 1), 4, "1x1x1"


def _build_ordered_substituted(
    base_structure: Structure,
    dopant_element: str,
    n_dopant: int,
    n_ga_total: int,
) -> Structure:
    """Replace `n_dopant` Ga sites with `dopant_element` via Ewald-min ordering.

    Strategy: build a disordered structure with uniform partial occupancy
    `frac = n_dopant / n_ga_total`, then run OrderDisorderedStructureTransformation
    to pick the lowest-electrostatic-energy ordered configuration.
    """
    s = base_structure.copy()
    frac = n_dopant / n_ga_total
    # Build replacement spec: replace each Ga with mixture {dopant: frac, Ga: 1-frac}
    # SubstitutionTransformation expects {old_site_str: new_species_dict}
    sub = SubstitutionTransformation({"Ga": {dopant_element: frac, "Ga": 1.0 - frac}})
    s_disordered = sub.apply_transformation(s)

    # Decorate with oxidation states (needed for Ewald)
    try:
        ox_decorate = OxidationStateDecorationTransformation({
            "Ga": 3, "O": -2, dopant_element: _guess_oxidation(dopant_element),
        })
        s_disordered = ox_decorate.apply_transformation(s_disordered)
    except Exception as e:
        logger.warning(f"  Could not decorate oxidation states: {e}")

    if n_dopant == 0:
        # Truly undoped — return just the base supercell, no substitution
        return base_structure

    # Order via Ewald minimisation
    odt = OrderDisorderedStructureTransformation(algo=2)  # ALGO_FAST
    s_ordered = odt.apply_transformation(s_disordered)
    if isinstance(s_ordered, list):
        # When return_ranked_list=True or when multiple equivalent orderings;
        # take the lowest-energy one
        s_ordered = s_ordered[0]["structure"] if isinstance(s_ordered[0], dict) else s_ordered[0]
    return s_ordered


def _guess_oxidation(element: str) -> int:
    """Guess oxidation state for substitutional dopant on Ga (3+) site."""
    valence_map = {
        "Mg": 2, "Zn": 2, "Cu": 2,
        "Al": 3, "Fe": 3, "B": 3, "V": 3, "Er": 3, "Eu": 3,
        "Si": 4, "Sn": 4, "Ti": 4, "Ge": 4,
        "Sb": 5, "Ta": 5, "Bi": 3, "Nb": 5,
        "W": 6, "F": -1, "N": -3,
    }
    return valence_map.get(element, 3)


def _relax_with_chgnet(struct: Structure, optimizer=None,
                       fmax: float = 0.1, steps: int = 200) -> dict:
    """Relax with CHGNet's force field. Returns dict with keys:
    relaxed_structure, max_force, energy, n_steps, converged.
    """
    if optimizer is None:
        from chgnet.model import StructOptimizer
        optimizer = StructOptimizer(use_device="cuda" if torch.cuda.is_available() else "cpu")
    result = optimizer.relax(struct, fmax=fmax, steps=steps, verbose=False)
    final = result["final_structure"]
    traj = result["trajectory"]
    energies = traj.energies if hasattr(traj, 'energies') else []
    forces = traj.forces if hasattr(traj, 'forces') else []
    n_steps = len(energies) if len(energies) > 0 else 0
    last_force = forces[-1] if len(forces) > 0 else None
    max_force = float(np.max(np.linalg.norm(last_force, axis=-1))) if last_force is not None else float("nan")
    energy = float(energies[-1]) if len(energies) > 0 else float("nan")
    return dict(
        relaxed_structure=final,
        max_force=max_force,
        energy=energy,
        n_steps=n_steps,
        converged=(max_force < fmax),
    )


def build_ordered_doped_structure(
    cache_key: str,
    element: str,
    conc_frac: float,
    base_cif: str | Path,
    out_dir: str | Path,
    relax: bool = True,
    optimizer=None,
) -> OrderedStructureManifest:
    """Top-level entry: takes a (cache_key, element, conc) and produces
    an ordered doped β-Ga₂O₃ supercell, optionally relaxed by CHGNet.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_path = out_dir / f"{cache_key}.cif"
    relaxed_path = out_dir / f"{cache_key}.relaxed.cif"

    if conc_frac <= 0 or element in ("undoped", ""):
        # Pure undoped — just use the base structure
        s = Structure.from_file(str(base_cif))
        s.to(filename=str(cif_path))
        if relax:
            r = _relax_with_chgnet(s, optimizer=optimizer)
            r["relaxed_structure"].to(filename=str(relaxed_path))
            return OrderedStructureManifest(
                cache_key=cache_key, element="undoped",
                target_conc_frac=0.0, supercell="1x1x1",
                n_atoms=len(s), n_ga=4, n_dopant_substituted=0,
                achieved_conc_frac=0.0, relaxed=True,
                relax_steps=r["n_steps"], relax_max_force=r["max_force"],
                relax_energy=r["energy"],
                cif_path=str(cif_path), relaxed_cif_path=str(relaxed_path),
            )
        return OrderedStructureManifest(
            cache_key=cache_key, element="undoped",
            target_conc_frac=0.0, supercell="1x1x1",
            n_atoms=len(s), n_ga=4, n_dopant_substituted=0,
            achieved_conc_frac=0.0, relaxed=False,
            relax_steps=0, relax_max_force=None, relax_energy=None,
            cif_path=str(cif_path), relaxed_cif_path=None,
        )

    # Pick supercell to fit the target concentration
    scaling, n_ga, label = _pick_supercell(conc_frac)
    base = Structure.from_file(str(base_cif))
    super_struct = base * scaling   # pymatgen supercell construction
    n_ga_super = sum(1 for site in super_struct.sites if str(site.species.elements[0]) == "Ga")
    assert n_ga_super == n_ga, f"expected {n_ga} Ga sites in {label}, got {n_ga_super}"

    # Determine integer dopant count via rounding
    n_dopant = max(1, int(np.round(conc_frac * n_ga)))
    n_dopant = min(n_dopant, n_ga - 1)   # leave at least one Ga
    achieved_conc = n_dopant / n_ga

    try:
        ordered = _build_ordered_substituted(super_struct, element, n_dopant, n_ga)
        ordered.to(filename=str(cif_path))
    except Exception as e:
        logger.error(f"Failed ordering for {cache_key}: {type(e).__name__}: {e}")
        return OrderedStructureManifest(
            cache_key=cache_key, element=element,
            target_conc_frac=conc_frac, supercell=label,
            n_atoms=len(super_struct), n_ga=n_ga, n_dopant_substituted=n_dopant,
            achieved_conc_frac=achieved_conc, relaxed=False,
            relax_steps=0, relax_max_force=None, relax_energy=None,
            cif_path="", relaxed_cif_path=None,
            error=f"ordering: {type(e).__name__}: {e}",
        )

    relax_steps = 0
    relax_max_force = None
    relax_energy = None
    relaxed_cif_str = None
    if relax:
        try:
            r = _relax_with_chgnet(ordered, optimizer=optimizer)
            r["relaxed_structure"].to(filename=str(relaxed_path))
            relax_steps = r["n_steps"]
            relax_max_force = r["max_force"]
            relax_energy = r["energy"]
            relaxed_cif_str = str(relaxed_path)
        except Exception as e:
            logger.warning(f"CHGNet relax failed for {cache_key}: "
                           f"{type(e).__name__}: {e} — keeping unrelaxed.")

    return OrderedStructureManifest(
        cache_key=cache_key, element=element,
        target_conc_frac=conc_frac, supercell=label,
        n_atoms=len(ordered), n_ga=n_ga, n_dopant_substituted=n_dopant,
        achieved_conc_frac=achieved_conc, relaxed=(relaxed_cif_str is not None),
        relax_steps=relax_steps, relax_max_force=relax_max_force,
        relax_energy=relax_energy,
        cif_path=str(cif_path),
        relaxed_cif_path=relaxed_cif_str,
    )


def manifest_save(manifests: list[OrderedStructureManifest], path: str | Path):
    """Save list of manifests to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump([asdict(m) for m in manifests], f, indent=2)
