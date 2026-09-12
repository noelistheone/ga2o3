"""
Convert pymatgen Structure objects into PyTorch Geometric Data objects.

Graph convention (Phase 5D):
  - Nodes : atoms in the unit cell; node feature is a 4-vector
              [host_Z, dopant_Z, dopant_occupancy, is_disordered_flag]
            stored as float32 shape [N, 4]. Ordered sites have dopant_Z=0,
            occupancy=0, flag=0 (the padding_idx=0 slot of the encoder's
            atom_embedding contributes nothing). Disordered sites produced
            by SubstitutionTransformation carry their minority component in
            the dopant channel, so the encoder can linearly mix host and
            dopant embeddings — previously the dominant-species rule in
            ``_site_atomic_number`` hid all <50% dopants.
            See docs/experiment_log.md §13.30 for the Phase 5B attribution.
  - Edges : all atom pairs within ``cutoff`` Å (periodic boundary conditions)
  - Edge features : 40-centre Gaussian RBF expansion of distance

On-the-fly doped structure generation:
  Given a DopantSpec (single/multi-element, compound, multi-compound), the module
  can synthesise a β-Ga₂O₃ unit cell with the requested substitution pattern via
  pymatgen's SubstitutionTransformation — no pre-existing CIF required.
"""

from __future__ import annotations

import os
import logging

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Base β-Ga₂O₃ CIF path (relative to project root or absolute)
_DEFAULT_BASE_CIF = "data/structures/Ga2O3_base.cif"
_BASE_STRUCTURE_CACHE: dict[str, object] = {}   # path → Structure


# ── pymatgen compatibility ────────────────────────────────────────────────────

def _site_node_features(site) -> tuple[int, int, float, float]:
    """
    Return ``(host_Z, dopant_Z, dopant_occupancy, is_disordered_flag)`` for a
    pymatgen PeriodicSite. Ordered sites report ``(Z, 0, 0.0, 0.0)``; disordered
    sites sort species by occupancy, keeping the majority as the host and the
    largest minority species as the dopant.

    Phase 5D: replaces ``_site_atomic_number``'s dominant-species collapse so
    the CGCNN encoder can distinguish sub-50% dopants instead of mapping every
    doped CIF to the undoped Ga₂O₃ embedding.
    """
    try:
        z = site.specie.Z                       # pymatgen < 2024, ordered site
        return (int(z), 0, 0.0, 0.0)
    except AttributeError:
        pass

    species_items = list(site.species.items())
    if not species_items:
        return (0, 0, 0.0, 0.0)

    sorted_items = sorted(species_items, key=lambda kv: kv[1], reverse=True)
    host_Z = int(sorted_items[0][0].Z)

    if len(sorted_items) == 1:
        return (host_Z, 0, 0.0, 0.0)

    dopant_Z = int(sorted_items[1][0].Z)
    dopant_occ = float(sorted_items[1][1])
    # Fold any further minority species into the dopant occupancy so the
    # weighted embedding does not silently drop them.
    for sp, occ in sorted_items[2:]:
        dopant_occ += float(occ)
    dopant_occ = max(0.0, min(dopant_occ, 1.0))
    return (host_Z, dopant_Z, dopant_occ, 1.0)


def _site_atomic_number(site) -> int:
    """Backward-compatible wrapper retained for non-encoder callers."""
    return _site_node_features(site)[0]


# ── Gaussian RBF expansion ────────────────────────────────────────────────────

def gaussian_rbf(
    distances: np.ndarray,
    num_centers: int = 40,
    cutoff: float = 8.0,
    sigma: float = 0.5,
) -> np.ndarray:
    """
    Expand scalar distances to a fixed-length RBF feature vector.

    Args:
        distances : [E] inter-atomic distances in Å.
        num_centers: number of Gaussian centres evenly spaced in [0, cutoff].
        cutoff     : max distance for centre placement (Å).
        sigma      : Gaussian width (Å).

    Returns:
        rbf : [E, num_centers]
    """
    centers = np.linspace(0.0, cutoff, num_centers)
    diff = distances[:, None] - centers[None, :]
    return np.exp(-(diff ** 2) / (2 * sigma ** 2))


# ── Structure → PyG graph ─────────────────────────────────────────────────────

def structure_to_graph(
    structure,
    cutoff: float = 6.0,
    num_rbf: int = 40,
    rbf_cutoff: float = 8.0,
    rbf_sigma: float = 0.5,
    label: float | None = None,
    label_key: str = "y",
) -> Data:
    """
    Convert a pymatgen Structure to a torch_geometric.data.Data object.

    Returns:
        Data with:
            x          [N, 4]     [host_Z, dopant_Z, dopant_occ, is_disordered]
                                  (float32; ordered sites have dopant cols = 0)
            edge_index [2, E]     directed edges (both directions)
            edge_attr  [E, num_rbf] RBF-expanded distances
            y          [1, 1] or absent
    """
    num_atoms = len(structure)

    node_features = torch.tensor(
        [_site_node_features(site) for site in structure], dtype=torch.float32
    )   # [N, 4]

    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)

    src_list, dst_list, dist_list = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        for nb in neighbors:
            src_list.append(i)
            dst_list.append(nb[2])
            dist_list.append(nb[1])

    if not dist_list:
        src_list = list(range(num_atoms))
        dst_list = list(range(num_atoms))
        dist_list = [0.0] * num_atoms

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    distances = np.array(dist_list, dtype=np.float32)
    rbf = gaussian_rbf(distances, num_centers=num_rbf,
                       cutoff=rbf_cutoff, sigma=rbf_sigma)
    edge_attr = torch.tensor(rbf, dtype=torch.float32)

    data = Data(x=node_features, edge_index=edge_index,
                edge_attr=edge_attr, num_nodes=num_atoms)

    if label is not None:
        setattr(data, label_key, torch.tensor([[label]], dtype=torch.float32))

    return data


def cif_to_graph(cif_path: str, **kwargs) -> Data:
    """Load a CIF file and convert to graph."""
    from pymatgen.core import Structure
    return structure_to_graph(Structure.from_file(cif_path), **kwargs)


# ── Base structure cache ──────────────────────────────────────────────────────

def _get_base_structure(base_cif: str = _DEFAULT_BASE_CIF):
    """Return (and cache) the β-Ga₂O₃ base structure."""
    if base_cif not in _BASE_STRUCTURE_CACHE:
        from pymatgen.core import Structure
        _BASE_STRUCTURE_CACHE[base_cif] = Structure.from_file(base_cif)
    return _BASE_STRUCTURE_CACHE[base_cif]


# ── On-the-fly doped structure generation ────────────────────────────────────

_ANION_SITE_DOPANTS: set[str] = set()    # DISABLED: frozen encoder can't handle novel graph topologies
_INTERSTITIAL_DOPANTS: set[str] = set()  # DISABLED: frozen encoder can't handle interstitial graphs
# Re-enable when encoder unfreezing is viable (requires more data or deeper unfreeze)

# Known interstitial site in β-Ga₂O₃ (monoclinic C2/m, mp-886) for hydrogen.
# Fractional coordinates from DFT literature (H_i near the O(I) channel).
_H_INTERSTITIAL_FRAC_COORDS = [0.17, 0.0, 0.61]


def dopant_spec_to_structure(
    spec: "DopantSpec",  # noqa: F821 – forward ref; import lazily
    base_cif: str = _DEFAULT_BASE_CIF,
):
    """
    Generate a doped β-Ga₂O₃ pymatgen Structure from a DopantSpec.

    Handles single element, multi-element co-doping, compound dopants,
    and multi-compound dopants.  Site assignment follows physics:
      - Most cations (Fe, Sn, Mg, Zn, Si, …) → Ga-site substitution
      - N → O-site substitution (nitrogen on oxygen site, acceptor)
      - H → interstitial site (hydrogen in lattice void, shallow donor)

    For co-doped specs like Mg+N, Mg goes to Ga sites and N to O sites.

    Args:
        spec    : DopantSpec instance (see src/data/dopant_spec.py).
        base_cif: Path to undoped β-Ga₂O₃ CIF.

    Returns:
        pymatgen.core.Structure
    """
    from pymatgen.transformations.standard_transformations import (
        SubstitutionTransformation,
    )

    base = _get_base_structure(base_cif)

    # Partition components by target site
    ga_site = [c for c in spec.components
               if c.cation not in _ANION_SITE_DOPANTS | _INTERSTITIAL_DOPANTS]
    o_site  = [c for c in spec.components if c.cation in _ANION_SITE_DOPANTS]
    interst = [c for c in spec.components if c.cation in _INTERSTITIAL_DOPANTS]

    structure = base.copy()

    # ── Ga-site substitution (Fe, Sn, Mg, Zn, Si, Ge, …) ────────────────────
    if ga_site:
        ga_sub: dict[str, float] = {}
        total_replaced = 0.0
        for c in ga_site:
            ga_sub[c.cation] = c.conc
            total_replaced += c.conc
        ga_sub["Ga"] = max(0.0, 1.0 - total_replaced)
        # Normalise so fractions sum to exactly 1.0
        s = sum(ga_sub.values())
        ga_sub = {k: v / s for k, v in ga_sub.items()}
        trans = SubstitutionTransformation({"Ga": ga_sub})
        structure = trans.apply_transformation(structure)

    # ── O-site substitution (N) ──────────────────────────────────────────────
    if o_site:
        total_o_replaced = sum(c.conc for c in o_site)
        total_o_replaced = min(total_o_replaced, 0.999)
        o_sub: dict[str, float] = {}
        for c in o_site:
            o_sub[c.cation] = c.conc
        o_sub["O"] = max(0.001, 1.0 - total_o_replaced)
        s = sum(o_sub.values())
        o_sub = {k: v / s for k, v in o_sub.items()}
        trans = SubstitutionTransformation({"O": o_sub})
        structure = trans.apply_transformation(structure)

    # ── Interstitial insertion (H) ───────────────────────────────────────────
    if interst:
        try:
            from pymatgen.transformations.site_transformations import (
                InsertSitesTransformation,
            )
            insert_trans = InsertSitesTransformation(
                species=["H"],
                coords=[_H_INTERSTITIAL_FRAC_COORDS],
            )
            structure = insert_trans.apply_transformation(structure)
            logger.debug("H interstitial inserted at fractional coords "
                         f"{_H_INTERSTITIAL_FRAC_COORDS}")
        except Exception as e:
            # Fallback: model H as O-site substitution (less accurate but
            # still far better than Ga-site for CGCNN embedding)
            logger.warning(f"H interstitial insertion failed ({e}); "
                           "falling back to O-site substitution")
            total_h = min(sum(c.conc for c in interst), 0.5)
            h_sub = {"H": total_h, "O": 1.0 - total_h}
            trans = SubstitutionTransformation({"O": h_sub})
            structure = trans.apply_transformation(structure)

    # ── Fallback: pure Ga-site (undoped or unrecognised) ─────────────────────
    if not ga_site and not o_site and not interst:
        sub_dict = spec.to_substitution_dict()
        trans = SubstitutionTransformation({"Ga": sub_dict})
        structure = trans.apply_transformation(structure)

    return structure


def dopant_spec_to_graph(
    spec: "DopantSpec",
    base_cif: str = _DEFAULT_BASE_CIF,
    **kwargs,
) -> Data:
    """Generate a doped β-Ga₂O₃ PyG graph from a DopantSpec on-the-fly."""
    structure = dopant_spec_to_structure(spec, base_cif)
    return structure_to_graph(structure, **kwargs)


# ── High-level entry point ────────────────────────────────────────────────────

def get_graph_for_spec(
    spec: "DopantSpec",
    structures_dir: str = "data/structures",
    base_cif: str | None = None,
    prefer_ordered: bool = False,
    **kwargs,
) -> Data:
    """
    Return a PyG graph for a DopantSpec.

    Priority:
      0. (Phase 50) If prefer_ordered=True, load from
         `structures_dir/ordered/<cache_key>.relaxed.cif` — concentration-
         dependent SQS+CHGNet-relaxed ordered supercell.
      1. Load from  structures_dir/<spec.to_cif_filename()>  (manually vetted CIF)
      2. Load from  structures_dir/<single_cation>.cif        (legacy single-element CIF)
      3. Generate on-the-fly via SubstitutionTransformation

    Args:
        spec          : DopantSpec instance.
        structures_dir: Directory containing pre-saved CIF files.
        base_cif      : Path to β-Ga₂O₃ base CIF for on-the-fly generation.
                        Defaults to structures_dir/Ga2O3_base.cif.
        prefer_ordered: Phase 50.A — try the ordered/relaxed CIF first. Falls
                        back to legacy paths if the ordered file is missing.
        **kwargs      : Passed to structure_to_graph (cutoff, num_rbf, …).

    Returns:
        torch_geometric.data.Data
    """
    if base_cif is None:
        base_cif = os.path.join(structures_dir, "Ga2O3_base.cif")

    # 0. Phase 50.A — prefer the ordered (CHGNet-relaxed) supercell when
    # available. Concentration enters geometry via supercell shape + dopant
    # placement + relaxed local distortion.
    if prefer_ordered:
        ordered_cif = os.path.join(structures_dir, "ordered",
                                   f"{spec.cache_key()}.relaxed.cif")
        if os.path.exists(ordered_cif):
            logger.debug(f"Loading ordered CIF: {ordered_cif}")
            return cif_to_graph(ordered_cif, **kwargs)
        # Fallback to unrelaxed ordered CIF if relaxation failed
        ordered_unrelaxed = os.path.join(structures_dir, "ordered",
                                          f"{spec.cache_key()}.cif")
        if os.path.exists(ordered_unrelaxed):
            logger.debug(f"Loading ordered unrelaxed CIF: {ordered_unrelaxed}")
            return cif_to_graph(ordered_unrelaxed, **kwargs)

    # 1. Exact match by spec filename
    spec_cif = os.path.join(structures_dir, spec.to_cif_filename())
    if os.path.exists(spec_cif):
        logger.debug(f"Loading pre-saved CIF: {spec_cif}")
        return cif_to_graph(spec_cif, **kwargs)

    # 2. Legacy single-element CIF (e.g. "Fe.cif")
    if len(spec.components) == 1:
        legacy_cif = os.path.join(structures_dir, f"{spec.components[0].cation}.cif")
        if os.path.exists(legacy_cif):
            logger.debug(f"Loading legacy CIF: {legacy_cif}")
            return cif_to_graph(legacy_cif, **kwargs)

    # 3. On-the-fly generation
    logger.debug(f"Generating structure on-the-fly for: {spec.cache_key()}")
    return dopant_spec_to_graph(spec, base_cif=base_cif, **kwargs)
