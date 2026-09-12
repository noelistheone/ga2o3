"""
Phase 6A: chemical family grouping + physical descriptors per dopant element.
Phase 46 V11: extended from 3-vector to 5-vector (added Δv and Δperiod) so
the element-descriptor hypernetwork can distinguish super-donors (Sb/Ta/Bi/W)
from regular donors (Si/Sn/Ti/Ge) on a continuous axis.

Used by `DopantStream` and `ElementHyperNet` to:
  1. Inject a 5-vector of physical descriptors per element (electronegativity,
     ionic radius, carrier-type hint, Δv vs Ga³⁺, Δperiod vs Ga's row 4)
     directly into the MLP input.
  2. Expose a per-element family-group index so the fine-tune trainer can add
     a cos-similarity regularizer pulling same-family dopant embeddings closer.

Goal: let rare-dopant embeddings (Cu/Ge/Bi/V/Sb at N=1-5) borrow representation
from abundant same-family dopants (Fe/Sn/Ta) through the regularizer, and give
the model a continuous chemistry axis that extrapolates to dopants unseen at
train time.

Carrier-type convention (on Ga³⁺ site, except N on O²⁻ site and H interstitial):
  +1 = donor   (adds an electron vs Ga, e.g. Sn⁴⁺, Si⁴⁺, Ta⁵⁺)
   0 = isovalent (B³⁺, Al³⁺, In³⁺, Fe³⁺, Bi³⁺)
  −1 = acceptor (Mg²⁺, Zn²⁺, Cu²⁺; N on O-site is hole-like)
H interstitial treated as shallow donor.

Δv convention: v_dopant − v_host (Ga³⁺ = +3; O²⁻ = −2 for anion-site dopants).
  Aligned with `_ELEM_CLASS_MAP` in src/models/ga2o3_net.py — super-donors
  (Sb⁵⁺/Ta⁵⁺/Bi⁵⁺/W⁶⁺) get Δv=+2, regular donors (Si⁴⁺/Sn⁴⁺/Ti⁴⁺/Ge⁴⁺)
  get Δv=+1, acceptors get Δv=−1.

Δperiod convention: period − 4 (Ga sits in row 4); separates lighter (B/Si/Al)
from heavier (In/Sn/Sb/Bi/Ta) homologues that often behave differently
(orbital diffuseness, ionic radius mismatch with Ga).
"""

from __future__ import annotations

import torch


# Family groups — elements that should share representation under the
# chem-family cos-similarity prior. "other"/"undoped" are bucketed separately
# (no pull applied) because we don't want miscellaneous elements forced close.
FAMILY_GROUPS: dict[str, int] = {
    "undoped": 6,       # neutral bucket, regularizer skips
    # II-group behavior on Ga site (acceptors)
    "Mg": 0, "Zn": 0,
    # III-group (isovalent with Ga)
    "B": 1, "Al": 1, "In": 1,
    # IV-group (donors on Ga site)
    "Si": 2, "Ge": 2, "Sn": 2,
    # V-group
    "N": 3, "Sb": 3, "Bi": 3,
    # d-block (transition metals on Ga site)
    "Fe": 4, "Cu": 4, "Ta": 4, "V": 4, "Zr": 4,
    # Interstitial (H in Ga₂O₃ channel)
    "H": 5,
    "other": 6,
}
N_FAMILY_GROUPS = 7
FAMILY_OTHER_IDX = 6   # regularizer skips this bucket


# Per-element 11-vector (V52b 2026-05-06):
#   0:  Pauling electronegativity (EN)
#   1:  Shannon 6-coord ionic radius (Å) of most common oxid state
#   2:  carrier_type ∈ {-1, 0, +1}
#   3:  Δv = v_dopant − 3 (V52: Sb/Bi=0)
#   4:  Δperiod = period − 4
#   5:  lone_pair_bool (1 if active ns² lone pair: Bi³⁺, Sb³⁺; else 0)  ← V52b NEW
#   6:  walsh_D — off-centering metric (Han et al., JACS 147:1291, 2025); 0 for non-LP  ← V52b NEW
#   7:  Pearson_hardness η (eV)                                                          ← V52b NEW
#   8:  χ_dopant − χ_Ga (Ga χ=1.81)                                                      ← V52b NEW
#   9:  (r_dopant − r_Ga) / r_Ga (fractional radius mismatch, r_Ga=0.620)                ← V52b NEW
#  10:  preferred_oxidation_state − 3 (oxidation offset from Ga³⁺)                       ← V52b NEW
# Sources: Pauling EN (standard), Shannon radii (Acta Cryst. A32, 1976),
# Pearson hardness (Pearson 1985 / Cardona 1996), Walsh lone-pair theory
# (Walsh & Watson J. Solid State Chem. 2005, 178:1422), JPCC 2025 lone-pair
# off-centering analysis. Indices 8-10 are linearly derivable from 0,1,3 but
# explicit features speed convergence (NN doesn't need to learn the linear
# combination).
_RAW_DESCRIPTORS: dict[str, tuple] = {
    # element : ( EN, radius, carrier, Δv, Δperiod, lone_pair, walsh_D, η, χ_mm, r_mm, ox_off)
    "undoped": (1.81, 0.620,  0.0,  0.0,  0.0,  0.0, 0.00, 5.5,  0.00,  0.00,  0.0),
    "Mg":      (1.31, 0.720, -1.0, -1.0, -1.0,  0.0, 0.00, 3.9, -0.50,  0.16, -1.0),
    "Zn":      (1.65, 0.740, -1.0, -1.0,  0.0,  0.0, 0.00, 5.5, -0.16,  0.19, -1.0),
    "Cu":      (1.90, 0.730, -1.0, -1.0,  0.0,  0.0, 0.00, 5.2,  0.09,  0.18, -1.0),
    "B":       (2.04, 0.270,  0.0,  0.0, -2.0,  0.0, 0.00, 8.0,  0.23, -0.56,  0.0),
    "Al":      (1.61, 0.535,  0.0,  0.0, -1.0,  0.0, 0.00, 6.5, -0.20, -0.14,  0.0),
    "In":      (1.78, 0.800,  0.0,  0.0,  1.0,  0.0, 0.00, 6.0, -0.03,  0.29,  0.0),
    "Fe":      (1.83, 0.645,  0.0,  0.0,  0.0,  0.0, 0.00, 7.0,  0.02,  0.04,  0.0),
    "V":       (1.63, 0.540,  1.0,  0.0,  0.0,  0.0, 0.00, 5.0, -0.18, -0.13,  0.0),
    "Si":      (1.90, 0.400,  1.0,  1.0, -1.0,  0.0, 0.00, 6.0,  0.09, -0.35,  1.0),
    "Ge":      (2.01, 0.530,  1.0,  1.0,  0.0,  0.0, 0.00, 5.5,  0.20, -0.15,  1.0),
    "Sn":      (1.96, 0.690,  1.0,  1.0,  1.0,  0.0, 0.00, 5.0,  0.15,  0.11,  1.0),
    "Zr":      (1.33, 0.720,  1.0,  1.0,  1.0,  0.0, 0.00, 3.2, -0.48,  0.16,  1.0),
    "Ti":      (1.54, 0.605,  1.0,  1.0,  0.0,  0.0, 0.00, 3.4, -0.27, -0.02,  1.0),
    # V52 isovalent + lone pair
    "Sb":      (2.05, 0.760,  0.0,  0.0,  1.0,  1.0, 0.30, 4.4,  0.24,  0.23,  0.0),  # 5s²
    "Bi":      (2.02, 1.030,  0.0,  0.0,  2.0,  1.0, 0.45, 4.5,  0.21,  0.66,  0.0),  # 6s²
    # Super-donors (no active lone pair)
    "Ta":      (1.50, 0.640,  1.0,  2.0,  2.0,  0.0, 0.00, 4.2, -0.31,  0.03,  2.0),
    "W":       (2.36, 0.600,  1.0,  3.0,  2.0,  0.0, 0.00, 4.5,  0.55, -0.03,  3.0),
    # Anion-site / interstitial / fallback
    "N":       (3.04, 1.460, -1.0, -1.0, -2.0,  0.0, 0.00, 7.0,  1.23,  1.35, -1.0),
    "F":       (3.98, 1.330, -1.0, -1.0, -2.0,  0.0, 0.00, 7.0,  2.17,  1.15, -1.0),
    "H":       (2.20, 0.200,  1.0, -2.0, -3.0,  0.0, 0.00, 7.0,  0.39, -0.68, -2.0),
    "other":   (1.90, 0.650,  0.0,  0.0,  0.0,  0.0, 0.00, 5.0,  0.09,  0.05,  0.0),
}


def _standardized_descriptor_table() -> dict[str, tuple[float, ...]]:
    """Z-score each column across the element set so the 5 dims enter the MLP
    on comparable scales. Computed once at module load."""
    keys = list(_RAW_DESCRIPTORS.keys())
    arr = torch.tensor([_RAW_DESCRIPTORS[k] for k in keys], dtype=torch.float32)
    mean = arr.mean(dim=0, keepdim=True)
    std = arr.std(dim=0, keepdim=True).clamp(min=1e-6)
    normed = (arr - mean) / std
    return {k: tuple(normed[i].tolist()) for i, k in enumerate(keys)}


ELEMENT_DESCRIPTORS: dict[str, tuple[float, ...]] = _standardized_descriptor_table()
DESCRIPTOR_DIM = 11   # V52b (was 5): added lone_pair, walsh_D, η, χ_mismatch, r_mismatch, ox_offset


# ──────────────────────────────────────────────────────────────────────────
# Phase 6C: generalizable carrier-type lookup for ANY element (seen or unseen).
# Used by the monotonic-constraint loss so the physics prior applies at
# deployment to new dopants without a manual table update.
# ──────────────────────────────────────────────────────────────────────────

# Extended carrier-type table covering common candidate β-Ga₂O₃ dopants that
# may appear at deployment but were not in the training set. Convention:
# +1 donor, 0 isovalent/amphoteric/unknown, -1 acceptor.
#   Ga site (host = Ga³⁺): compare most common oxid state to +3.
#     +1 → donor (e.g. Sn⁴⁺, Si⁴⁺, Ta⁵⁺, Nb⁵⁺, W⁶⁺, Mo⁶⁺, Zr⁴⁺, Hf⁴⁺)
#      0 → isovalent / ambiguous (B³⁺, Al³⁺, In³⁺, Fe³⁺, Cr³⁺, Bi³⁺, Sc³⁺)
#     -1 → acceptor (Mg²⁺, Zn²⁺, Cu²⁺, Ni²⁺, Co²⁺, Ca²⁺)
#   O site (host = O²⁻):
#     -1 → hole-producing (N³⁻, P³⁻)  [treated as acceptor for slope sign]
#   Interstitial:
#     +1 → shallow donor (H, Li)

_EXTENDED_CARRIER_TYPE: dict[str, int] = {
    # Donors on Ga site
    "Sn": +1, "Si": +1, "Ge": +1, "Ta": +1, "Nb": +1, "V": +1, "W": +1, "Mo": +1,
    "Zr": +1, "Hf": +1, "Sb": +1, "Ti": +1,
    # Isovalent on Ga site
    "B": 0, "Al": 0, "In": 0, "Fe": 0, "Cr": 0, "Bi": 0, "Sc": 0, "Y": 0, "La": 0,
    "Ga": 0,  # host itself
    # Acceptors on Ga site
    "Mg": -1, "Zn": -1, "Cu": -1, "Ni": -1, "Co": -1, "Ca": -1, "Sr": -1, "Ba": -1,
    "Li": -1, "Na": -1,  # 1+ valence → strong acceptors on 3+ site, though ionic size hurts
    # O-site anion acceptors
    "N": -1, "P": -1,
    # Interstitial donors
    "H": +1,
    # Neutral / unknown defaults
    "undoped": 0,
    "other": 0,
}


def get_carrier_type(element_symbol: str) -> float:
    """Return carrier_type (+1 / 0 / -1) for any element symbol.

    Falls back to 0 (no monotonic constraint) for unknown elements so the
    constraint is safe — an unseen dopant just won't contribute monotonic
    loss, matching the original (4F/5-series) behaviour.
    """
    if element_symbol in _EXTENDED_CARRIER_TYPE:
        return float(_EXTENDED_CARRIER_TYPE[element_symbol])
    # Last-resort attempt: use pymatgen's common oxidation state if available.
    try:
        from pymatgen.core import Element
        e = Element(element_symbol)
        oxid = e.common_oxidation_states
        if not oxid:
            return 0.0
        # Pick the most-positive common oxidation state (typical for cation
        # dopants on Ga site). For strongly anionic elements (N, P) this gives
        # a wrong-sign answer but they're already in the explicit table above.
        most_common = max(oxid, key=abs)
        if most_common > 3:
            return +1.0
        elif most_common < 3 and most_common > 0:
            return -1.0
        else:
            return 0.0
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────
# Phase 17: VC monotonicity prior — substitutional cation dopants.
# Physics: any Ga-site substitutional cation dopant requires V_O for charge
# balance (acceptor strongly, donor weakly via vacancy/dopant association).
# So d(V_O concentration)/d(dopant_conc) ≥ 0 universally for this class.
# Anion (N, F, P) and interstitial (H, Li) dopants have different mechanisms
# and are excluded from this prior.
# ──────────────────────────────────────────────────────────────────────────

_ANION_OR_INTERSTITIAL = {"N", "F", "P", "H", "Li", "O", "Cl", "Br"}


def is_cation_substitutional(element_symbol: str) -> bool:
    """Return True iff the element is a substitutional cation dopant on the Ga site.

    Used by the VC-monotonicity prior in finetune_trainer to apply the
    "more dopant → more V_O" charge-balance physics. Returns False for
    anion-site dopants, interstitials, the host Ga itself, undoped, and
    any element where the assignment is ambiguous (defaults to False).
    """
    if not element_symbol or element_symbol in ("undoped", "other", "Ga"):
        return False
    if element_symbol in _ANION_OR_INTERSTITIAL:
        return False
    if element_symbol in _EXTENDED_CARRIER_TYPE:
        # Anything in the explicit cation table (donor / isovalent / acceptor) is OK
        return True
    # Unknown element: try pymatgen — accept if it has a positive common oxidation state
    try:
        from pymatgen.core import Element
        e = Element(element_symbol)
        oxid = e.common_oxidation_states
        if oxid and max(oxid) > 0:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Phase 54 V54-A1: SINCERE multi-positive chemical-similarity groups.
#
# These groups define MULTI-positive sets for the SINCERE contrastive loss
# (Feeney & Hughes arXiv:2309.14277). Distinct from FAMILY_GROUPS above,
# which is consumed by _family_cos_loss at a different code path —
# DO NOT merge or rebind.
#
# Sets chosen to (a) cluster lone-pair Bi/Sb (V52 moved both to
# _ELEM_CLASS_MAP=1 because they prefer +3 substitution with ns² lone pair);
# (b) preserve V53-α's Mg-acceptor cluster; (c) keep classical IV-donors
# (Si/Sn/Ge) together; (d) keep III-isovalent (Al/Fe/B/V) together.
# ---------------------------------------------------------------------------
CHEM_SIM_GROUP_MAP: dict[str, int] = {
    # Lone-pair (active ns²): Bi 6s², Sb 5s² — both prefer +3 substitution
    "Bi": 0, "Sb": 0,
    # Classical acceptors on Ga site (Mg/Zn/Cu = v=2; Ca placeholder)
    "Mg": 1, "Zn": 1, "Cu": 1, "Ca": 1,
    # Classical IV-donors (v=4): Si/Sn/Ge
    "Si": 2, "Sn": 2, "Ge": 2,
    # Group-III / first-row d-block isovalent (v=3)
    "Al": 3, "Fe": 3, "B": 3, "V": 3,
    # Super-donors (v=5/6): Ta/W (no active lone pair)
    "Ta": 4, "W": 4,
    # Group-IV transition donors (Ti/Zr/Hf — distinct from Si/Sn/Ge due to
    # d-character; Hf is unseen in current training but listed for future AL)
    "Ti": 5, "Zr": 5, "Hf": 5,
    # Rare-earth isovalent (Er/Eu — sparse in dataset but distinct chemistry)
    "Er": 6, "Eu": 6,
}
CHEM_SIM_OTHER_OFFSET = 100
N_CHEM_SIM_BASE_GROUPS = 7
CHEM_SIM_UNDOPED_SENTINEL = -1


def chemical_similarity_group_id(dopant_label: str) -> int:
    """Map a cation label to a SINCERE-positive bucket.

    Known elements use CHEM_SIM_GROUP_MAP. Unknown / co-doped / blended
    labels (e.g. "Sn+Fe", "Cu0.5Zn0.5") fall into a stable hash bucket
    (offset by CHEM_SIM_OTHER_OFFSET) so two batches with the same unknown
    label still group together — but unknowns never accidentally collide
    with known buckets 0–6.

    Undoped / blank labels map to a sentinel (-1) so the trainer treats
    them as 'no positives'.
    """
    if not dopant_label or dopant_label in ("", "undoped", "other"):
        return CHEM_SIM_UNDOPED_SENTINEL
    if dopant_label in CHEM_SIM_GROUP_MAP:
        return CHEM_SIM_GROUP_MAP[dopant_label]
    # Stable, deterministic fallback (Python hash is salted per-process).
    import hashlib
    h = int(hashlib.md5(dopant_label.encode("utf-8")).hexdigest(), 16) % 1000
    return CHEM_SIM_OTHER_OFFSET + h


def chem_sim_group_ids_for_batch(labels: list[str]) -> list[int]:
    """Vectorized helper for trainer hot path."""
    return [chemical_similarity_group_id(l) for l in labels]



