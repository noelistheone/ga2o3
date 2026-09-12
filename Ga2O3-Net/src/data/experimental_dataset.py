"""
Few-shot experimental dataset for doped β-Ga₂O₃.

CSV format (ga2o3_exp.csv)
──────────────────────────
Required columns:
  dopant_spec     — dopant specification string (authoritative)
                    e.g. "Fe:0.0265" | "Fe:0.0133,Sn:0.0132" | "SnO2:0.0265"
  concentration_at% — total dopant atomic percent (used when dopant_spec has no
                    embedded concentrations, e.g. "Fe,Sn")
  atmosphere      — annealing atmosphere string; supports:
                      single gas : "Ar" | "O2" | "N2" | "air" | "O2_plasma"
                      mixed gas  : "Ar:O2=3:1" | "N2:O2=4:1" | "O2:Ar=1:9"
                    Parsed to three features: o2_fraction, is_plasma_enhanced,
                    has_nitrogen_species.
  method          — fabrication method free text (e.g. "RF magnetron sputtering",
                    "PLD", "MOCVD"). One-hot encoded into 5 groups.
  temperature_C   — annealing temperature in °C
  time_min        — annealing time in minutes

Optional target columns (any subset):
  photocurrent_uA, dark_current_pA, photo_dark_ratio,
  sensitivity, vacancy_concentration

Legacy column (kept for compatibility, not used by model):
  element         — primary dopant element symbol

Each row becomes one sample with:
  graph       : PyG Data from β-Ga₂O₃ doped structure (on-the-fly if needed)
  dopant_spec : spec string passed to CompositionStream
  process     : 18-dim raw scalar tensor (Phase 5C layout):
                 [temperature_C, time_min, o2_fraction,
                  is_plasma_enhanced, has_nitrogen_species,
                  has_plasma_dep, has_anneal, has_buffer, on_foreign_substrate,
                  measurement_voltage_V, measurement_wavelength_nm,
                  log10_concentration,
                  method_sputtering, method_pld, method_cvd_ald,
                  method_wet, method_evaporation,
                  substrate_category_idx]
  target      : [n_targets] tensor (NaN where data is missing)
"""

from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dopant_spec import DopantSpec, parse_spec
from src.data.graph_builder import get_graph_for_spec

logger = logging.getLogger(__name__)

TARGET_COLS = [
    "photocurrent_uA",
    "dark_current_pA",
    "photo_dark_ratio",
    "sensitivity",
    "vacancy_concentration",
]

# ── Atmosphere parser ─────────────────────────────────────────────────────────

# Gases whose contribution to O₂ partial pressure is known
_O2_FRACTION: dict[str, float] = {
    "o2":         1.0,
    "air":        0.21,   # standard atmosphere
    "o3":         1.0,    # ozone → treat as pure oxidiser
    "o2plasma":   1.0,    # O₂ plasma: highly oxidising
    "o2_plasma":  1.0,
    "n2o":        0.5,    # N₂O: partial oxidiser (mono-oxygen source)
    "f-plasma":   0.0,    # fluorine plasma: inert from O₂ perspective
    "fplasma":    0.0,
    "inert/o₂":   0.5,    # unclear mixed: assume ~50/50
    "inert/o2":   0.5,
    "ar:o₂":      0.5,    # "Ar:O₂" without ratio → assume 50/50
    "ar:o2":      0.5,
    # v6 new atmosphere strings
    "n2_plasma":  0.0,    # nitrogen plasma: no O₂ (plasma flag handled separately)
    "n2plasma":   0.0,
    "ar+air":     0.105,  # Ar diluted with air: ~50% air → O₂ ≈ 0.105
    "wet":        0.0,    # aqueous / electrolyte environments: no gas-phase O₂
}
# Inert / reducing gases contribute 0 O₂
_INERT_GASES = {
    "ar", "n2", "he", "ne", "kr", "xe", "h2", "co",
    "formingas", "forming_gas",
    "vacuum", "h2o", "wet", "inert",
}


def _parse_atmosphere_o2_fraction(atm_str: str) -> float:
    """
    Convert an atmosphere string to O₂ mole fraction [0.0, 1.0].

    Supported formats
    -----------------
    Single gas  : "Ar"  →  0.0
                  "O2"  →  1.0
                  "air" →  0.21
    Mixed gas   : "Ar:O2=3:1"   → O2 fraction = 1/(3+1) = 0.25
                  "N2:O2=4:1"   → 0.2
                  "O2:Ar=1:9"   → 0.1
                  "N2:O2:Ar=2:1:7" → 0.1  (multi-component)

    The mixed-gas format is  "Gas1:Gas2:...=ratio1:ratio2:..."
    Ratios are molar (or volumetric, equivalent for ideal gases).
    The O₂ mole fraction is  sum(O₂-like ratios) / sum(all ratios).

    Unknown single gases default to 0.0 with a warning.
    """
    if not isinstance(atm_str, str) or not atm_str.strip():
        return 0.0

    atm_str = atm_str.strip()

    # ── Mixed gas: contains "=" ───────────────────────────────────────────────
    if "=" in atm_str:
        try:
            gases_part, ratios_part = atm_str.split("=", 1)
            gas_names = [g.strip() for g in gases_part.split(":")]
            ratios    = [float(r.strip()) for r in ratios_part.split(":")]
            if len(gas_names) != len(ratios) or not ratios:
                raise ValueError("gas/ratio count mismatch")
            total = sum(ratios)
            if total <= 0:
                raise ValueError("ratios sum to zero")
            o2_frac = sum(
                _O2_FRACTION.get(g.lower().replace("_", ""), 0.0) * r
                for g, r in zip(gas_names, ratios)
            ) / total
            return float(np.clip(o2_frac, 0.0, 1.0))
        except Exception as exc:
            logger.warning(
                f"Could not parse mixed atmosphere '{atm_str}' ({exc}); "
                "falling back to 0.0 (inert)."
            )
            return 0.0

    # ── Single gas ────────────────────────────────────────────────────────────
    key = atm_str.lower().replace("_", "").replace("-", "")
    if key in _O2_FRACTION:
        return _O2_FRACTION[key]
    if key in _INERT_GASES:
        return 0.0

    # ── Common compound / shorthand forms ─────────────────────────────────────
    if key in ("n2+o2", "n2o2"):        return 0.20   # MOCVD N₂+O₂ without ratio
    if key in ("ar+n2", "arn2"):        return 0.0
    if key in ("ar+h2", "arh2"):        return 0.0    # reducing
    if key in ("ar+air", "arair"):      return 0.105  # ≈ half-air
    if key == "wet":                    return 0.0
    if key == "vacuum":                 return 0.0
    if key == "inert":                  return 0.0

    logger.warning(
        f"Unknown atmosphere '{atm_str}'; assuming inert (O₂ fraction = 0.0)."
    )
    return 0.0


def _parse_atmosphere_flags(atm_str: str) -> tuple[float, float]:
    """
    Return (is_plasma_enhanced, has_nitrogen_species) for an atmosphere string.

    is_plasma_enhanced  : 1.0 if the atmosphere includes plasma activation
                          (e.g. "O2_plasma", "N2_plasma").  Plasma strongly
                          increases surface reactivity beyond the O₂ fraction.
    has_nitrogen_species: 1.0 if N₂, N₂O, NH₃, or other N-bearing gas is present.
                          Nitrogen affects Ga₂O₃ defect chemistry differently from
                          pure inert or oxidising environments.

    *Legacy v2 (18-dim process tensor) helper.* Use ``_parse_atmosphere_flags_v3``
    for the v3 layout (19-dim) which adds a hydrogen-species bit.
    """
    if not isinstance(atm_str, str) or not atm_str.strip():
        return 0.0, 0.0
    key = atm_str.lower()
    is_plasma = 1.0 if "plasma" in key else 0.0
    has_nitrogen = 1.0 if any(n in key for n in ("n2", "n2o", "nh3", "no2", "nitrogen")) else 0.0
    return is_plasma, has_nitrogen


def parse_atmosphere_ordinal(atm_str: str) -> int:
    """
    Map atmosphere string to oxidising-power ordinal (Phase 32).

    Lower ordinal = more oxidising = predicts LOWER V_O density.
    Higher ordinal = more reducing  = predicts HIGHER V_O density.

    Order (from OA β-Ga₂O₃ sputter literature, agent scan 2026-04-30):
        0 = O₂_plasma  (most oxidising — atomic O activity)
        1 = O₂          (pure oxygen, no plasma)
        2 = Ar:O₂ mix   (e.g. Ar:O2=1:1, Ar:O2=3:1)
        3 = Ar+H₂       (mildly reducing — atomic H gettering)
        4 = Ar+N₂       (inert with passive N₂)
        5 = Ar / N₂ pure  (inert sputter standard)
        6 = vacuum      (most reducing — V_O highest)

    Falls back to ordinal 5 (Ar-equivalent) for unknown strings.
    """
    if not isinstance(atm_str, str) or not atm_str.strip():
        return 5
    key = atm_str.lower().strip()
    if "plasma" in key:
        return 0
    has_h = any(h in key for h in ("h2", "nh3", "hydrogen", "forming"))
    has_n = any(n in key for n in ("n2", "n2o", "nh3", "no2", "nitrogen"))
    is_o2 = key in ("o2",) or key == "oxygen"
    is_ar_o2 = ("ar" in key and "o2" in key)
    is_air = key in ("air", "atmosphere")
    is_vac = "vacuum" in key or "vac" == key
    if is_o2:        return 1
    if is_ar_o2:     return 2
    if is_air:       return 2  # ~21% O₂
    if has_h:        return 3
    if has_n and "ar" in key: return 4
    if is_vac:       return 6
    if key.startswith("ar") or key == "n2": return 5
    return 5  # default to Ar-like


def _parse_atmosphere_flags_v3(atm_str: str) -> tuple[float, float, float]:
    """
    v3 (19-dim process tensor) — adds ``has_hydrogen_species``.

    Returns (is_plasma_enhanced, has_nitrogen_species, has_hydrogen_species).

    has_hydrogen_species: 1.0 if H₂, NH₃, or other H-bearing gas is present.
                          Hydrogen passivates V_O dangling bonds and is reported
                          to reduce dark current in oxide UV photodetectors —
                          previously invisible to the 18-dim layout because
                          ``has_nitrogen_species`` was the only N/H-bit available
                          (NH₃ also tripped that bit but Ar+H₂ did not).
    """
    if not isinstance(atm_str, str) or not atm_str.strip():
        return 0.0, 0.0, 0.0
    key = atm_str.lower()
    is_plasma = 1.0 if "plasma" in key else 0.0
    has_nitrogen = 1.0 if any(n in key for n in ("n2", "n2o", "nh3", "no2", "nitrogen")) else 0.0
    # Hydrogen detection: explicit "h2", or hydrogen-bearing molecules. NH3
    # already trips has_nitrogen but contains H so trip both.
    has_hydrogen = 1.0 if any(h in key for h in ("h2", "nh3", "hydrogen", "forming gas")) else 0.0
    return is_plasma, has_nitrogen, has_hydrogen


_METHOD_KEYWORDS: dict[str, list[str]] = {
    "sputtering":  ["sputter", "magnetron", "pvd"],
    "pld":         ["pld", "pulsed laser", "laser ablation", "laser deposition"],
    "cvd_ald":     ["cvd", "mocvd", "ald", "pecvd", "lpcvd", "mbe"],
    "wet":         ["hydrothermal", "sol-gel", "sol gel", "solgel",
                    "chemical bath", "cbd", "spray", "electrodeposition",
                    "precipitation"],
    "evaporation": ["evapor", "e-beam", "ebeam", "electron beam", "vacuum dep"],
}


def _parse_method_onehot(method_str: str) -> list[float]:
    """
    Map a free-text fabrication method string to a 5-element one-hot vector.

    Order: [sputtering, pld, cvd_ald, wet, evaporation]
    Unknown / missing method → all zeros (implicit 'other' baseline).
    """
    if not isinstance(method_str, str) or not method_str.strip():
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    key = method_str.lower()
    return [
        1.0 if any(kw in key for kw in kws) else 0.0
        for kws in _METHOD_KEYWORDS.values()
    ]


# ── Phase 2C: Hierarchical method flags ──────────────────────────────────────
# The 5-category one-hot above collapses ~90 unique method strings into very
# broad buckets; we lose distinctions the physics cares about (plasma activation
# during growth, post-growth anneal presence, buffer layer, foreign substrate).
# These flags are extracted alongside the one-hot and added to the continuous
# branch so they go through StandardScaler like any other numeric.

_METHOD_FLAG_KEYWORDS: dict[str, list[str]] = {
    # Plasma-assisted deposition (growth-time plasma, not anneal atmosphere).
    "has_plasma_dep":       ["pe-ald", "peald", "pecvd", "plasma-assisted",
                             "plasma assisted", "pa-mbe", "pambe",
                             "plasma-enhanced", "plasma enhanced"],
    # Post-growth anneal step.
    "has_anneal":           ["anneal", "annealed", "+ann", "+rta", "rta",
                             "post-anneal", "post anneal", "post-ann"],
    # Buffer / interlayer between film and substrate.
    "has_buffer":           ["buffer", "aln", "azo", "interlayer", "seed layer",
                             "seed-layer"],
    # Foreign substrate (not Ga₂O₃ homoepitaxy). Indicates lattice-mismatch /
    # thermal-expansion-mismatch defect channels.
    "on_foreign_substrate": ["on si", "on sic", "on gan", "on sapphire",
                             "sapphire", "mgo", "c-plane", "c plane",
                             "srtio3", "strontium titanate", "α-al2o3",
                             "al2o3", "zno substrate", "znga2o4"],
}


def _parse_method_flags(method_str: str) -> list[float]:
    """
    Extract 4 hierarchical process flags from a free-text method string.

    Returns ``[has_plasma_dep, has_anneal, has_buffer, on_foreign_substrate]``
    with each entry in {0.0, 1.0}. Unknown / missing → all zeros.

    These flags are *complementary* to the 5-way method one-hot: they encode
    physics-relevant modifiers (plasma, anneal, buffer, substrate) that the
    broad category collapse would otherwise lose.
    """
    if not isinstance(method_str, str) or not method_str.strip():
        return [0.0, 0.0, 0.0, 0.0]
    key = method_str.lower()
    return [
        1.0 if any(kw in key for kw in kws) else 0.0
        for kws in _METHOD_FLAG_KEYWORDS.values()
    ]


# ── Substrate categorical encoding (Phase 2E) ────────────────────────────────
# Substrate identity determines lattice/thermal mismatch, interface defect density,
# and back-contact behaviour — all physically upstream of PDR/VC, NOT derived from
# them (no target leakage). We map method+notes free text to one of 7 categories
# and feed as a single integer → nn.Embedding(7, 3) in ProcessStream.
#
# Ordering chosen so bulk/microwire (no foreign substrate) is highest index; the
# "other" bucket (index 6) catches unmatched cases without biasing via position.

# Keyword priority matters: when a string contains multiple substrate mentions
# (e.g. "MOCVD on GaN/sapphire"), the *direct contact* surface is the one the
# β-Ga₂O₃ film actually nucleates on. GaN/AlGaN/MgO/oxide-templates are scanned
# before sapphire/Si so stacked structures resolve to the intermediate template,
# not the ultimate base wafer. `native_bulk` has the highest priority because
# self-supporting microwires/bulk crystals have no substrate at all.
_SUBSTRATE_KEYWORDS: dict[str, list[str]] = {
    # 0 native_bulk — self-supporting microwires/nanowires/bulk EFG crystals.
    "native_bulk":     ["microwire", "nanowire", "microbelt", "bulk",
                        "self-support", "self support", "self-supporting",
                        "efg", "czochralski", "edge-defined",
                        "free-standing", "freestanding"],
    # 1 GaN / AlGaN — epitaxial templates with small mismatch.
    "GaN":             ["gan", "algan"],
    # 2 MgO — rocksalt (100), unusual template, Mg-diffusion source.
    "MgO":             ["mgo", "mgo(100)", "on mgo"],
    # 3 oxide_template — spinels, perovskites, ZnO, TiN buffers, transparent oxides.
    "oxide_template":  ["znga2o4", "srtio3", "strontium titanate",
                        "zno substrate", "tin buffer", "ito", "fto"],
    # 4 Si — (100)/(111); triggers large mismatch + different buffer pathways.
    "Si":              ["on si", "si(100)", "si(111)", "si substrate",
                        "p-si", "n-si", "silicon substrate"],
    # 5 sapphire — α-Al₂O₃, c-/a-/m-plane. Most common β-Ga₂O₃ substrate;
    #   placed last so stacked structures (GaN/sapphire) resolve to GaN.
    "sapphire":        ["sapphire", "c-plane", "c plane", "a-plane",
                        "α-al2o3", "alpha-al2o3", "al2o3", "c-sapphire"],
    # 6 "other" is the implicit fallback when no keyword matches.
}


def _parse_substrate_category(text: str) -> int:
    """
    Map a free-text method+notes string to a substrate category index in [0, 6].

    Returns 6 ("other") when no keyword matches. The index ordering encodes
    priority: when multiple substrate keywords match (e.g. a GaN/sapphire stack),
    the first match wins, which corresponds to the direct film–substrate
    interface rather than the ultimate base wafer.

    Substrate is a fabrication choice available before any measurement, so using
    it as a feature does NOT introduce target leakage.
    """
    if not isinstance(text, str) or not text.strip():
        return 6
    key = text.lower()
    for idx, kws in enumerate(_SUBSTRATE_KEYWORDS.values()):
        if any(kw in key for kw in kws):
            return idx
    return 6


_N_SUBSTRATE_CATEGORIES = 7


_DEFAULT_MEASURE_VOLTAGE = 10.0       # V — typical MSM bias
_DEFAULT_MEASURE_WAVELENGTH = 254.0   # nm — dominant solar-blind line


def build_process_tensor(
    temperature_C: float,
    time_min: float,
    atmosphere: str = "",
    method: str = "",
    measurement_voltage_V: float | None = None,
    measurement_wavelength_nm: float | None = None,
    concentration_total_frac: float = 0.0,
) -> "torch.Tensor":
    """
    Build an 18-dim process tensor from human-readable experimental conditions.

    Phase 5C adds a dedicated ``log10_concentration`` slot (dim 11) so total
    dopant concentration reaches the fusion head directly — required because
    graph_builder's dominant-species collapse hides sub-50% dopants from the
    encoder (see Phase 5B, docs/experiment_log.md §13.30).

      * 12 continuous features        (indices 0–11, StandardScaler)
      * 5 method one-hot bits         (indices 12–16, → Embedding via argmax)
      * 1 substrate category index    (index 17,     → Embedding directly)

    Layout:
        [0]  temperature_C
        [1]  time_min
        [2]  o2_fraction
        [3]  is_plasma_enhanced        (atmosphere-plasma flag)
        [4]  has_nitrogen_species
        [5]  has_plasma_dep            (growth-time plasma, e.g. PE-ALD, PAMBE)
        [6]  has_anneal                (post-growth anneal)
        [7]  has_buffer                (AlN/AZO/seed layer)
        [8]  on_foreign_substrate      (sapphire / Si / GaN / MgO …)
        [9]  measurement_voltage_V
        [10] measurement_wavelength_nm
        [11] log10_concentration       (log10 of total dopant fraction, floor 1e-4)
        [12] method_sputtering
        [13] method_pld
        [14] method_cvd_ald
        [15] method_wet
        [16] method_evaporation
        [17] substrate_category_idx    (0=sapphire, 1=Si, 2=GaN, 3=MgO,
                                        4=oxide_template, 5=native_bulk, 6=other)

    Args:
        temperature_C: Annealing temperature in °C.
        time_min:      Annealing duration in minutes.
        atmosphere:    Annealing atmosphere string; empty → inert.
        method:        Fabrication method free text. Also parsed for hierarchical
                       flags (plasma-dep / anneal / buffer / foreign substrate)
                       and substrate category.
        measurement_voltage_V:     Measurement bias (V). None → default 10.0 V.
        measurement_wavelength_nm: Probe wavelength (nm). None → default 254 nm.
        concentration_total_frac:  Total dopant concentration as a fraction
                                   (at%/100). Undoped → 0.0, floored at 1e-4
                                   before log10 so undoped maps to −4.

    Returns:
        Tensor [18] of dtype float32.
    """
    o2_frac = _parse_atmosphere_o2_fraction(atmosphere)
    is_plasma, has_nitrogen = _parse_atmosphere_flags(atmosphere)
    method_bits = _parse_method_onehot(method)
    method_flags = _parse_method_flags(method)
    v = float(measurement_voltage_V) if measurement_voltage_V is not None else _DEFAULT_MEASURE_VOLTAGE
    wl = float(measurement_wavelength_nm) if measurement_wavelength_nm is not None else _DEFAULT_MEASURE_WAVELENGTH
    log10_conc = float(np.log10(max(float(concentration_total_frac), 1e-4)))
    sub_idx = float(_parse_substrate_category(method))
    return torch.tensor(
        [temperature_C, time_min, o2_frac, is_plasma, has_nitrogen]
        + method_flags
        + [v, wl, log10_conc]
        + method_bits
        + [sub_idx],
        dtype=torch.float32,
    )


def build_process_tensor_v3(
    temperature_C: float,
    time_min: float,
    atmosphere: str = "",
    method: str = "",
    measurement_voltage_V: float | None = None,
    measurement_wavelength_nm: float | None = None,
    concentration_total_frac: float = 0.0,
) -> "torch.Tensor":
    """
    Build a 19-dim process tensor (v3 layout) — adds ``has_hydrogen_species`` at
    index 5 vs v2. Everything else shifts by +1 vs ``build_process_tensor``.

    Layout:
        [0]  temperature_C
        [1]  time_min
        [2]  o2_fraction
        [3]  is_plasma_enhanced
        [4]  has_nitrogen_species
        [5]  has_hydrogen_species          ← NEW: Ar+H₂ / NH₃ / forming gas
        [6]  has_plasma_dep
        [7]  has_anneal
        [8]  has_buffer
        [9]  on_foreign_substrate
        [10] measurement_voltage_V
        [11] measurement_wavelength_nm
        [12] log10_concentration
        [13] method_sputtering
        [14] method_pld
        [15] method_cvd_ald
        [16] method_wet
        [17] method_evaporation
        [18] substrate_category_idx

    continuous_dim = 13 (indices 0–12 are scaled by StandardScaler).
    in_dim         = 19.
    """
    o2_frac = _parse_atmosphere_o2_fraction(atmosphere)
    is_plasma, has_nitrogen, has_hydrogen = _parse_atmosphere_flags_v3(atmosphere)
    method_bits = _parse_method_onehot(method)
    method_flags = _parse_method_flags(method)
    v = float(measurement_voltage_V) if measurement_voltage_V is not None else _DEFAULT_MEASURE_VOLTAGE
    wl = float(measurement_wavelength_nm) if measurement_wavelength_nm is not None else _DEFAULT_MEASURE_WAVELENGTH
    log10_conc = float(np.log10(max(float(concentration_total_frac), 1e-4)))
    sub_idx = float(_parse_substrate_category(method))
    return torch.tensor(
        [temperature_C, time_min, o2_frac, is_plasma, has_nitrogen, has_hydrogen]
        + method_flags
        + [v, wl, log10_conc]
        + method_bits
        + [sub_idx],
        dtype=torch.float32,
    )


def _parse_row_spec(row: pd.Series) -> DopantSpec:
    """Build a DopantSpec from a CSV row, handling both formats."""
    if "dopant_spec" in row.index and pd.notna(row.get("dopant_spec")):
        spec_str = str(row["dopant_spec"]).strip()

        # Undoped sentinel — no concentration needed
        if spec_str.lower() == "undoped":
            return parse_spec("undoped")

        # no_conc rows: spec_str is empty string → treat as undoped
        if not spec_str:
            return parse_spec("undoped")

        total_conc = (
            float(row["concentration_at%"]) / 100.0
            if "concentration_at%" in row.index
            and pd.notna(row.get("concentration_at%")) else None
        )
        return parse_spec(spec_str, total_conc)

    # Legacy: only element + concentration columns
    elem = str(row["element"]).strip()
    # Guard against placeholder "—" / "–" / empty element (no_conc rows written
    # as "" by prepare_data.py; pandas read_csv converts "" back to NaN, so the
    # dopant_spec branch above was skipped; treat as undoped here).
    if not elem or elem in ("—", "–", "-", "nan"):
        return parse_spec("undoped")
    conc_raw = row.get("concentration_at%")
    if pd.notna(conc_raw):
        conc = float(conc_raw) / 100.0
        return parse_spec(f"{elem}:{conc:.6f}")
    else:
        # no_conc row — element known but concentration missing; use dataset mean
        return parse_spec(f"{elem}:0.026500")


class Ga2O3ExpDataset(Dataset):
    """
    Few-shot experimental dataset for doped β-Ga₂O₃.

    Supports all dopant modes via the `dopant_spec` CSV column:
      - Single element: "Fe:0.0265"
      - Multi-element:  "Fe:0.0133,Sn:0.0132"
      - Compound:       "SnO2:0.0265"
      - Multi-compound: "SnO2:0.013,MgO:0.013"

    Args:
        csv_path      : Path to ga2o3_exp.csv.
        structures_dir: Directory with pre-saved CIF files and Ga2O3_base.cif.
        target_cols   : Subset of TARGET_COLS to predict.
        augment       : Apply jitter during __getitem__.
        conc_jitter   : Concentration noise std (fraction).
        label_noise_std: Label noise std (fraction of range).
    """

    def __init__(
        self,
        csv_path: str,
        structures_dir: str,
        target_cols: list[str] = TARGET_COLS,
        cutoff: float = 6.0,
        num_rbf: int = 40,
        augment: bool = False,
        conc_jitter: float = 0.005,
        label_noise_std: float = 0.01,
        method_shuffle_prob: float = 0.0,
        substrate_shuffle_prob: float = 0.0,
        log_transform_targets: bool = True,
        mask_noconc_labels: bool = False,
        mask_undoped_labels: bool = False,
        min_conc_threshold: float = 0.0,
        mask_carrier_vc: bool = False,
        exclude_elements: list[str] | None = None,
        include_elements: list[str] | None = None,
        include_keep_undoped: bool = True,
        process_layout: str = "v2",
        chgnet_emb_cache_path: str | None = None,
        prefer_ordered: bool = False,
    ):
        """
        Args:
            log_transform_targets: API flag; when using the CSV produced by
                ``scripts/00c_prepare_data.py`` targets are already log10-
                transformed, so this is a no-op.  Set to False only if
                supplying a raw (un-transformed) CSV.
            mask_noconc_labels: If True, target values for rows with
                ``usable_flag == 'no_conc'`` are replaced with NaN during
                training.  These rows have unknown dopant concentrations so
                their composition features are approximated as "undoped",
                creating label noise.  Masking removes the MSE gradient while
                keeping the samples available for InfoNCE contrastive signal.
            mask_undoped_labels: If True, target values for undoped rows
                (``dopant_spec == 'undoped'``) are replaced with NaN during
                training.  Undoped samples all share identical composition
                features (pure Ga₂O₃), so the model can only differentiate
                them via the 10-dim process features.  When many undoped samples
                exist with different targets, they create label noise because the
                50+ unique fabrication methods collapse to 6 categories.
                Masking focuses MSE on doped samples where composition features
                provide real differentiation.
            min_conc_threshold: If > 0, mask MSE labels for doped samples
                whose total dopant concentration (fraction, not at%) is below
                this threshold.  Samples with very low concentrations (<0.001
                i.e. <0.1 at%) have XenonPy composition features nearly
                identical to undoped Ga₂O₃, so they behave as pseudo-undoped
                and add label noise.  Also excludes MBE/PAMBE samples whose
                concentrations were back-calculated from carrier concentration
                (circular dependency with the VC target).
                Typical value: 0.001 (0.1 at%).
            mask_carrier_vc: If True, mask the vacancy_concentration label
                (set to NaN) for rows where ``conc_source == 'carrier'``.
                These samples have dopant concentrations back-calculated
                from carrier concentration (≈ vacancy concentration),
                creating a circular dependency between feature and target.
                Only the VC label is masked; PDR labels are kept.
        """
        self.log_transform_targets = log_transform_targets
        self.mask_noconc_labels = mask_noconc_labels
        self.mask_undoped_labels = mask_undoped_labels
        self.min_conc_threshold = min_conc_threshold
        self.mask_carrier_vc = mask_carrier_vc
        self.exclude_elements = set(exclude_elements) if exclude_elements else set()
        self.include_elements = set(include_elements) if include_elements else set()
        self.include_keep_undoped = bool(include_keep_undoped)
        self.df = pd.read_csv(csv_path)

        # Drop rows flagged as exotic (non-β-Ga₂O₃ host phases)
        if "usable_flag" in self.df.columns:
            n_before = len(self.df)
            self.df = self.df[self.df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
            n_skip = n_before - len(self.df)
            if n_skip:
                logger.info(f"Dropped {n_skip} exotic_skip rows; {len(self.df)} remain.")

        # Exclude rows containing specified dopant elements
        if self.exclude_elements:
            n_before = len(self.df)
            keep_mask = []
            for _, row in self.df.iterrows():
                spec = _parse_row_spec(row)
                cations = {c.cation for c in spec.components}
                keep_mask.append(not bool(cations & self.exclude_elements))
            self.df = self.df[keep_mask].reset_index(drop=True)
            n_excl = n_before - len(self.df)
            if n_excl:
                logger.info(f"Excluded {n_excl} rows containing "
                            f"{self.exclude_elements}; {len(self.df)} remain.")

        if self.include_elements:
            n_before = len(self.df)
            keep_mask = []
            for _, row in self.df.iterrows():
                spec = _parse_row_spec(row)
                cations = {c.cation for c in spec.components}
                if not cations and self.include_keep_undoped:
                    keep_mask.append(True)
                    continue
                keep_mask.append(bool(cations & self.include_elements))
            self.df = self.df[keep_mask].reset_index(drop=True)
            n_kept = len(self.df)
            n_drop = n_before - n_kept
            if n_drop:
                logger.info(
                    f"Include filter {self.include_elements} "
                    f"(keep_undoped={self.include_keep_undoped}): "
                    f"dropped {n_drop} rows, {n_kept} remain."
                )

        self.structures_dir = structures_dir
        self.target_cols = target_cols
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.augment = augment
        self.conc_jitter = conc_jitter
        self.label_noise_std = label_noise_std
        self.method_shuffle_prob = float(method_shuffle_prob)
        self.substrate_shuffle_prob = float(substrate_shuffle_prob)
        self.process_layout = str(process_layout).lower()
        if self.process_layout not in ("v2", "v3"):
            raise ValueError(f"process_layout must be 'v2' or 'v3', got {process_layout!r}")

        # Compute process imputation values (used for NaN temperature/time rows)
        temps_raw = pd.to_numeric(self.df["temperature_C"], errors="coerce").values
        times_raw = pd.to_numeric(self.df["time_min"], errors="coerce").values
        self._mean_temp = float(np.nanmean(temps_raw)) if not np.all(np.isnan(temps_raw)) else 800.0
        self._mean_time = float(np.nanmean(times_raw)) if not np.all(np.isnan(times_raw)) else 60.0

        # Phase 2C: measurement-condition imputation (voltage, wavelength)
        if "measurement_voltage_V" in self.df.columns:
            v_raw = pd.to_numeric(self.df["measurement_voltage_V"], errors="coerce").values
            self._mean_voltage = (
                float(np.nanmean(v_raw))
                if not np.all(np.isnan(v_raw))
                else _DEFAULT_MEASURE_VOLTAGE
            )
        else:
            self._mean_voltage = _DEFAULT_MEASURE_VOLTAGE
        if "measurement_wavelength_nm" in self.df.columns:
            wl_raw = pd.to_numeric(self.df["measurement_wavelength_nm"], errors="coerce").values
            self._mean_wavelength = (
                float(np.nanmean(wl_raw))
                if not np.all(np.isnan(wl_raw))
                else _DEFAULT_MEASURE_WAVELENGTH
            )
        else:
            self._mean_wavelength = _DEFAULT_MEASURE_WAVELENGTH

        # Parse DopantSpec for every row
        self._specs: list[DopantSpec] = [
            _parse_row_spec(row) for _, row in self.df.iterrows()
        ]

        # Phase 50.A: when True, _preload_graphs uses the SQS+CHGNet-relaxed
        # ordered CIFs at structures_dir/ordered/<key>.relaxed.cif (concentration
        # encoded as supercell shape + dopant placement + relaxed distortion).
        self.prefer_ordered = prefer_ordered

        # Pre-cache graphs keyed by spec.cache_key()
        self._graph_cache: dict[str, object] = {}
        self._preload_graphs()

        # Phase 49 V17: optional CHGNet pre-extracted crystal_fea attachment.
        # When provided, each cached graph gets a `.chgnet_emb` attribute
        # (shape [64]) which the CHGNetCachedEncoder reads at training time.
        # PyG Batch.from_data_list propagates per-graph tensor attrs as [B, 64].
        self.chgnet_emb_cache_path = chgnet_emb_cache_path
        if chgnet_emb_cache_path is not None:
            self._attach_chgnet_emb(chgnet_emb_cache_path)

        # Pre-warm XenonPy lru_cache for all unique specs (avoids cold-start
        # penalty during first training epoch; subsequent jitter hits the cache).
        self._prewarm_xenonpy_cache()

        self._label_ranges = self._compute_label_ranges()

        # Phase 6D: pre-compute method-onehot and substrate pools for shuffle
        # augmentation. Each pool row is a realisation observed in this dataset,
        # so shuffled values respect the training distribution (no invented
        # combinations). Ordered by row so we can optionally filter by mask.
        self._method_pool = np.asarray(
            [_parse_method_onehot(str(m) if pd.notna(m) else "")
             for m in self.df.get("method", pd.Series([""] * len(self.df)))],
            dtype=np.float32,
        )
        self._substrate_pool = np.asarray(
            [_parse_substrate_category(
                (str(m) if pd.notna(m) else "") + " "
                + (str(n) if pd.notna(n) else ""))
             for m, n in zip(
                 self.df.get("method", pd.Series([""] * len(self.df))),
                 self.df.get("notes", pd.Series([""] * len(self.df))),
             )],
            dtype=np.float32,
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        spec = self._specs[idx]

        # Optional concentration jitter (perturbs each component equally)
        if self.augment and not spec.is_undoped and spec.total_conc > 0:
            noise = np.random.normal(0, self.conc_jitter)
            # Build a jittered spec string with adjusted concentrations
            parts = []
            for c in spec.components:
                new_conc = float(np.clip(c.conc + noise / len(spec.components),
                                         1e-4, 0.49))
                # 3-decimal precision keeps XenonPy lru_cache effective:
                # ±0.005 jitter → only ~10 unique concentrations per element.
                parts.append(f"{c.formula}:{new_conc:.3f}")
            jittered_spec_str = ",".join(parts)
        else:
            jittered_spec_str = spec.raw

        graph = self._graph_cache.get(spec.cache_key())
        if graph is None:
            raise RuntimeError(
                f"No graph found for spec '{spec.cache_key()}'. "
                "Check that Ga2O3_base.cif exists in structures_dir."
            )

        atm_str = str(row["atmosphere"]) if "atmosphere" in row.index and pd.notna(row.get("atmosphere")) else "Ar"
        # Use column mean for as-grown samples that have no annealing step
        temp_c = float(row["temperature_C"]) if pd.notna(row.get("temperature_C")) else getattr(self, "_mean_temp", 800.0)
        time_m = float(row["time_min"]) if pd.notna(row.get("time_min")) else getattr(self, "_mean_time", 60.0)
        o2_frac = _parse_atmosphere_o2_fraction(atm_str)
        if self.process_layout == "v3":
            is_plasma, has_nitrogen, has_hydrogen = _parse_atmosphere_flags_v3(atm_str)
        else:
            is_plasma, has_nitrogen = _parse_atmosphere_flags(atm_str)
            has_hydrogen = 0.0  # unused in v2 path
        method_str = str(row["method"]) if "method" in row.index and pd.notna(row.get("method")) else ""
        # Combine method + notes for flag extraction (anneal/substrate info often
        # appears in the notes free-text column; the 5-way method one-hot keeps
        # using method alone to avoid cross-contamination).
        notes_str = str(row["notes"]) if "notes" in row.index and pd.notna(row.get("notes")) else ""
        method_flags = _parse_method_flags(method_str + " " + notes_str)
        # Phase 2C: measurement conditions (imputed with column mean when absent)
        v_raw = row.get("measurement_voltage_V") if "measurement_voltage_V" in row.index else None
        wl_raw = row.get("measurement_wavelength_nm") if "measurement_wavelength_nm" in row.index else None
        v_val = float(v_raw) if (v_raw is not None and pd.notna(v_raw)) else self._mean_voltage
        wl_val = float(wl_raw) if (wl_raw is not None and pd.notna(wl_raw)) else self._mean_wavelength
        # Phase 2E: substrate category parsed from method + notes combined.
        sub_idx = float(_parse_substrate_category(method_str + " " + notes_str))
        # Phase 5C: direct concentration signal at index 11 (continuous block).
        # spec.total_conc is a fraction; undoped/no_conc rows floor to 1e-4 → log10=−4.
        log10_conc = float(np.log10(max(float(getattr(spec, "total_conc", 0.0) or 0.0), 1e-4)))
        method_onehot = _parse_method_onehot(method_str)
        # Phase 6D: method/substrate shuffle — break spurious
        # method↔dopant and substrate↔dopant correlations by replacing the
        # method bits (and/or substrate index) with a random observed
        # realisation from the training distribution. Active only when
        # augment=True, so inference and validation are unaffected.
        if self.augment and self.method_shuffle_prob > 0.0 \
                and np.random.random() < self.method_shuffle_prob \
                and len(self._method_pool) > 0:
            method_onehot = list(
                self._method_pool[np.random.randint(len(self._method_pool))]
            )
        if self.augment and self.substrate_shuffle_prob > 0.0 \
                and np.random.random() < self.substrate_shuffle_prob \
                and len(self._substrate_pool) > 0:
            sub_idx = float(
                self._substrate_pool[np.random.randint(len(self._substrate_pool))]
            )
        if self.process_layout == "v3":
            process = torch.tensor(
                [temp_c, time_m, o2_frac, is_plasma, has_nitrogen, has_hydrogen]
                + method_flags
                + [v_val, wl_val, log10_conc]
                + method_onehot
                + [sub_idx],
                dtype=torch.float32,
            )
        else:
            process = torch.tensor(
                [temp_c, time_m, o2_frac, is_plasma, has_nitrogen]
                + method_flags
                + [v_val, wl_val, log10_conc]
                + method_onehot
                + [sub_idx],
                dtype=torch.float32,
            )

        labels = []
        for col in self.target_cols:
            val = float(row[col]) if col in row.index and pd.notna(row.get(col)) else float("nan")
            if self.augment and not np.isnan(val):
                val += np.random.normal(0, self.label_noise_std * self._label_ranges[col])
            labels.append(val)
        target = torch.tensor(labels, dtype=torch.float32)

        # Mask regression labels for no_conc rows: these samples have unknown
        # dopant concentrations so their composition features are approximated
        # as "undoped", making their labels unreliable for regression.
        if self.mask_noconc_labels and row.get("usable_flag") == "no_conc":
            target = torch.full_like(target, float("nan"))

        # Mask regression labels for undoped rows: all undoped samples share
        # identical composition features (pure Ga₂O₃), so they can only be
        # differentiated by process features. When many undoped samples exist
        # with diverse targets, they create label noise.
        if self.mask_undoped_labels and spec.is_undoped:
            target = torch.full_like(target, float("nan"))

        # Mask regression labels for near-undoped doped samples: if total
        # dopant concentration is below min_conc_threshold, XenonPy composition
        # features are indistinguishable from undoped Ga₂O₃, and MBE/PAMBE
        # samples at very low doping may have circular concentration→VC
        # derivation. Both effects create label noise.
        if (self.min_conc_threshold > 0
                and not spec.is_undoped
                and spec.total_conc < self.min_conc_threshold):
            target = torch.full_like(target, float("nan"))

        # Mask VC labels for samples where concentration was back-derived from
        # carrier concentration (circular dependency: feature ← VC ← label)
        if self.mask_carrier_vc:
            conc_source = row.get("conc_source", "direct") if "conc_source" in row.index else "direct"
            if conc_source == "carrier" and "vacancy_concentration" in self.target_cols:
                vc_idx = self.target_cols.index("vacancy_concentration")
                target = target.clone()
                target[vc_idx] = float("nan")

        # Per-sample weight for weighted MSE (default 1.0 if column absent)
        sw_raw = row.get("sample_weight") if "sample_weight" in row.index else None
        sw_val = float(sw_raw) if (sw_raw is not None and pd.notna(sw_raw)) else 1.0

        return {
            "graph": graph,
            "dopant_spec": jittered_spec_str,
            "process": process,
            "target": target,
            "sample_weight": torch.tensor(sw_val, dtype=torch.float32),
            # Convenience fields for logging / InfoNCE class labels
            "dopant_label": spec.cation_label(),
            # Phase 20: DOI for within-DOI pair-aware monotonic loss
            "doi": str(row.get("doi", "")) if "doi" in row.index else "",
            # Phase 32: raw atmosphere string for atmosphere-ordinal monotone loss
            "atmosphere": atm_str,
            # Phase 54 V54-A1: global sample index for self-distill cache lookup
            "sample_idx": int(idx),
        }

    # ── Accessors for trainer ─────────────────────────────────────────────────

    def get_dopant_specs(self) -> list[str]:
        """Return raw spec string for each sample."""
        return [s.raw for s in self._specs]

    def get_dopant_labels(self) -> list[str]:
        """Return cation-label for each sample (used for InfoNCE class grouping)."""
        return [s.cation_label() for s in self._specs]

    def get_process_array(self) -> np.ndarray:
        """Return [N, 18] raw process array for StandardScaler fitting (Phase 5C).

        Columns (18):
          [0]  temperature_C
          [1]  time_min
          [2]  o2_fraction
          [3]  is_plasma_enhanced      (atmosphere plasma)
          [4]  has_nitrogen_species
          [5]  has_plasma_dep          (growth-time plasma: PE-ALD/PAMBE/...)
          [6]  has_anneal              (post-growth anneal)
          [7]  has_buffer              (AlN/AZO/seed layer)
          [8]  on_foreign_substrate
          [9]  measurement_voltage_V
          [10] measurement_wavelength_nm
          [11] log10_concentration     (Phase 5C; log10 of total dopant fraction,
                                        floor 1e-4 so undoped → −4)
          [12] method_sputtering
          [13] method_pld
          [14] method_cvd_ald
          [15] method_wet
          [16] method_evaporation
          [17] substrate_category_idx  (0-6; → nn.Embedding in ProcessStream)

        NaN values (as-grown / no-measurement rows) are imputed with the column
        mean so the scaler receives finite input.
        """
        temps = pd.to_numeric(self.df["temperature_C"], errors="coerce").values.astype(np.float64)
        times = pd.to_numeric(self.df["time_min"], errors="coerce").values.astype(np.float64)

        # Impute NaN with column mean (computed once in __init__)
        temps = np.where(np.isnan(temps), self._mean_temp, temps).astype(np.float32)
        times = np.where(np.isnan(times), self._mean_time, times).astype(np.float32)

        # fillna before str() conversion to avoid str(NaN)="nan" leaking into parsers
        atm_col = (
            self.df["atmosphere"].fillna("Ar")
            if "atmosphere" in self.df.columns
            else pd.Series(["Ar"] * len(self.df))
        )
        o2_fracs = np.array(
            [_parse_atmosphere_o2_fraction(str(a)) for a in atm_col],
            dtype=np.float32,
        )

        # Atmosphere flags: [N, 2] for v2 (is_plasma, has_nitrogen) or
        # [N, 3] for v3 (is_plasma, has_nitrogen, has_hydrogen).
        if self.process_layout == "v3":
            atm_flags = np.array(
                [_parse_atmosphere_flags_v3(str(a)) for a in atm_col],
                dtype=np.float32,
            )
        else:
            atm_flags = np.array(
                [_parse_atmosphere_flags(str(a)) for a in atm_col],
                dtype=np.float32,
            )

        # Method strings → flag block (method+notes) and one-hot block (method only).
        method_col = (
            self.df["method"].fillna("")
            if "method" in self.df.columns
            else pd.Series([""] * len(self.df))
        )
        notes_col = (
            self.df["notes"].fillna("")
            if "notes" in self.df.columns
            else pd.Series([""] * len(self.df))
        )
        # Hierarchical method flags: [N, 4]  (parsed from method + notes)
        method_flags = np.array(
            [_parse_method_flags(str(m) + " " + str(n))
             for m, n in zip(method_col, notes_col)],
            dtype=np.float32,
        )
        # Method one-hot: [N, 5]  (method string only; notes avoided to keep the
        # categorical embedding tied to the fabrication technique itself)
        methods = np.array(
            [_parse_method_onehot(str(m)) for m in method_col],
            dtype=np.float32,
        )

        # Measurement-condition columns (voltage, wavelength) with mean imputation.
        if "measurement_voltage_V" in self.df.columns:
            v_raw = pd.to_numeric(self.df["measurement_voltage_V"], errors="coerce").values.astype(np.float64)
            v = np.where(np.isnan(v_raw), self._mean_voltage, v_raw).astype(np.float32)
        else:
            v = np.full(len(self.df), self._mean_voltage, dtype=np.float32)
        if "measurement_wavelength_nm" in self.df.columns:
            wl_raw = pd.to_numeric(self.df["measurement_wavelength_nm"], errors="coerce").values.astype(np.float64)
            wl = np.where(np.isnan(wl_raw), self._mean_wavelength, wl_raw).astype(np.float32)
        else:
            wl = np.full(len(self.df), self._mean_wavelength, dtype=np.float32)

        # Phase 2E: substrate category index (one integer per sample, 0–6).
        sub_idx = np.array(
            [_parse_substrate_category(str(m) + " " + str(n))
             for m, n in zip(method_col, notes_col)],
            dtype=np.float32,
        )

        # Phase 5C: log10(total dopant concentration as fraction). Pulled from
        # the parsed DopantSpec per row so undoped/no_conc cases floor to −4
        # consistently with __getitem__.
        log10_conc = np.array(
            [np.log10(max(float(getattr(s, "total_conc", 0.0) or 0.0), 1e-4))
             for s in self._specs],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                temps[:, None], times[:, None], o2_fracs[:, None],
                atm_flags,            # [N, 2] for v2 / [N, 3] for v3
                method_flags,         # [N, 4]
                v[:, None], wl[:, None],
                log10_conc[:, None],  # [N, 1]  — Phase 5C concentration signal
                methods,              # [N, 5]
                sub_idx[:, None],     # [N, 1]  — index into substrate embedding
            ],
            axis=1,
        )  # [N, 18] for v2 / [N, 19] for v3

    # ── Private ───────────────────────────────────────────────────────────────

    def _preload_graphs(self):
        """
        Load or generate structure graphs for every unique DopantSpec.

        Priority: pre-saved spec CIF → legacy single-element CIF → on-the-fly.
        """
        seen: set[str] = set()
        base_cif = os.path.join(self.structures_dir, "Ga2O3_base.cif")

        for spec in self._specs:
            key = spec.cache_key()
            if key in seen:
                continue
            seen.add(key)
            try:
                # Undoped → directly use the base CIF (or the ordered/relaxed
                # supercell when prefer_ordered=True)
                if spec.is_undoped:
                    from src.data.graph_builder import cif_to_graph
                    cif_path = base_cif
                    if getattr(self, "prefer_ordered", False):
                        ordered_undoped = os.path.join(self.structures_dir,
                                                       "ordered", "undoped.relaxed.cif")
                        if os.path.exists(ordered_undoped):
                            cif_path = ordered_undoped
                    graph = cif_to_graph(
                        cif_path,
                        cutoff=self.cutoff,
                        num_rbf=self.num_rbf,
                    )
                    self._graph_cache["undoped"] = graph
                    logger.info(f"Graph ready for [undoped] via {cif_path}")
                    continue

                graph = get_graph_for_spec(
                    spec,
                    structures_dir=self.structures_dir,
                    base_cif=base_cif,
                    cutoff=self.cutoff,
                    num_rbf=self.num_rbf,
                    prefer_ordered=getattr(self, "prefer_ordered", False),
                )
                self._graph_cache[key] = graph

                cif_path = os.path.join(self.structures_dir, spec.to_cif_filename())
                legacy_cif = (
                    os.path.join(self.structures_dir,
                                 f"{spec.components[0].cation}.cif")
                    if len(spec.components) == 1 else None
                )
                if os.path.exists(cif_path):
                    src = f"CIF ({spec.to_cif_filename()})"
                elif legacy_cif and os.path.exists(legacy_cif):
                    src = f"legacy CIF ({os.path.basename(legacy_cif)})"
                else:
                    src = "on-the-fly generation"
                logger.info(f"Graph ready for [{spec.label()}] via {src}")

            except Exception as e:
                logger.error(f"Failed to build graph for [{spec.label()}]: {e}")

    def _attach_chgnet_emb(self, cache_path: str) -> None:
        """Attach pre-extracted CHGNet `crystal_fea` (64-dim) to each cached
        graph as `graph.chgnet_emb`. PyG Batch will then concatenate these
        along the batch dim, giving [B, 64] when the CHGNetCachedEncoder
        reads `batch.chgnet_emb`.
        """
        cache_data = torch.load(cache_path, weights_only=False)
        if isinstance(cache_data, dict) and "embeddings" in cache_data:
            emb_dict = cache_data["embeddings"]
        else:
            emb_dict = cache_data
        n_attached = 0
        n_missing = 0
        for spec in self._specs:
            key = spec.cache_key()
            graph = self._graph_cache.get(key)
            if graph is None:
                continue
            if hasattr(graph, "chgnet_emb") and graph.chgnet_emb is not None:
                continue   # already attached (shared across rows with same key)
            emb = emb_dict.get(key)
            if emb is None:
                n_missing += 1
                # Fallback: zero vector (lets training proceed; trainer logs)
                emb = np.zeros(64, dtype=np.float32)
            # Shape [1, 64] so PyG batches as [B, 64] along dim 0
            graph.chgnet_emb = torch.as_tensor(emb, dtype=torch.float32).view(1, -1)
            n_attached += 1
        logger.info(
            f"CHGNet emb cache attached: {n_attached} graphs from {cache_path}"
            + (f" (warning: {n_missing} keys missing in cache)" if n_missing else "")
        )

    def _prewarm_xenonpy_cache(self):
        """
        Pre-compute XenonPy features for all unique base specs.

        With jitter enabled, training batches use slightly perturbed concentrations
        at 3-decimal precision (≈10 distinct values per element).  Warming the cache
        here with the exact base specs covers the common case and avoids the
        cold-start penalty for fold-1 scaler fitting.
        """
        try:
            from src.models.composition_stream import spec_to_xenonpy_features
        except ImportError:
            return
        unique_raws = {s.raw for s in self._specs}
        logger.info(
            f"Pre-warming XenonPy cache for {len(unique_raws)} unique specs..."
        )
        for raw in unique_raws:
            try:
                spec_to_xenonpy_features(raw)
            except Exception as e:
                logger.debug(f"XenonPy cache warm-up skipped for '{raw}': {e}")
        logger.info("XenonPy cache warmed.")

    def get_target_array(self) -> np.ndarray:
        """
        Return [N, num_targets] float32 array with NaN for missing labels.

        Useful for pseudo-labeling: identifies which training samples lack
        which targets so that pseudo-labels can be inserted selectively.
        """
        rows = []
        for _, row in self.df.iterrows():
            vals = []
            for col in self.target_cols:
                v = (
                    float(row[col])
                    if col in row.index and pd.notna(row.get(col))
                    else float("nan")
                )
                vals.append(v)
            rows.append(vals)
        return np.array(rows, dtype=np.float32)

    def _compute_label_ranges(self) -> dict[str, float]:
        ranges = {}
        for col in self.target_cols:
            if col in self.df.columns:
                vals = pd.to_numeric(self.df[col], errors="coerce").dropna().values
                ranges[col] = float(vals.max() - vals.min()) if len(vals) > 1 else 1.0
            else:
                ranges[col] = 1.0
        return ranges


# ── Pseudo-label helper ───────────────────────────────────────────────────────

class PseudoLabeledSubset(Dataset):
    """
    A Dataset wrapper that substitutes NaN targets with pseudo-labels for
    selected indices, while keeping all other fields from the base dataset.

    Used in iterative pseudo-labeling (Stage 3 phase 2) to expand the
    effective training set for sparsely-labeled targets.

    Args:
        base_dataset:    Ga2O3ExpDataset (should have augment=True for phase 2).
        indices:         List of integer indices into base_dataset.
        pseudo_targets:  np.ndarray [len(indices), num_targets].
                         NaN entries remain NaN; non-NaN entries override the
                         base dataset's target tensor for that sample/task.
    """

    def __init__(
        self,
        base_dataset: Ga2O3ExpDataset,
        indices: list[int],
        pseudo_targets: np.ndarray,
    ):
        self.base = base_dataset
        self.indices = indices
        self.pseudo_targets = torch.tensor(pseudo_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_idx: int) -> dict:
        global_idx = self.indices[local_idx]
        item = self.base[global_idx]
        item["target"] = self.pseudo_targets[local_idx]
        return item


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate for Ga2O3ExpDataset batches.
    PyG graphs are batched with Batch.from_data_list; everything else stacked.
    """
    from torch_geometric.data import Batch

    return {
        "graph": Batch.from_data_list([s["graph"] for s in batch]),
        "dopant_spec": [s["dopant_spec"] for s in batch],
        "process": torch.stack([s["process"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
        "sample_weight": torch.stack([s["sample_weight"] for s in batch]),
        "dopant_label": [s["dopant_label"] for s in batch],
        "doi": [s.get("doi", "") for s in batch],   # Phase 20
        "atmosphere": [s.get("atmosphere", "") for s in batch],  # Phase 32
        # Phase 54 V54-A1: sample_idx for self-distill cache lookup
        "sample_idx": torch.tensor(
            [int(s.get("sample_idx", -1)) for s in batch], dtype=torch.long
        ),
    }
