"""
DopantSpec — unified dopant specification parser.

Supports all dopant modes:
  "Fe"                         → single element, concentration from CSV column
  "Fe:0.0265"                  → single element with explicit fraction
  "Fe:0.0133,Sn:0.0132"       → multi-element co-doping
  "SnO2"                       → compound (Sn cation extracted automatically)
  "SnO2:0.0265"                → compound with explicit fraction
  "SnO2:0.013,MgO:0.013"      → multi-compound co-doping
  "Fe2O3:0.01,SnO2:0.01,MgO:0.005" → triple compound

Usage:
    spec = DopantSpec.parse("Fe:0.0133,Sn:0.0132")
    spec.to_xenonpy_formula()       # "Fe0.026600Sn0.026400Ga1.947000O3"
    spec.to_substitution_dict()     # {"Fe": 0.0133, "Sn": 0.0132, "Ga": 0.9735}
    spec.cache_key()                # "Fe-0.0133_Sn-0.0132"
    spec.label()                    # "Fe+Sn"

Convention for formula construction (consistent with Ga₂O₃ per-formula-unit):
  dopant_count_i = 2 × conc_i       (per Ga₂ in Ga₂O₃)
  ga_count       = 2 - Σ dopant_count_i
  xenonpy_formula = Σ[cation_i + count_i] + "Ga{ga_count}O3"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


# ── Cation extraction ─────────────────────────────────────────────────────────

def _extract_cation(formula: str) -> str:
    """
    Extract the primary cation from a formula string.

    Rules (in order):
      1. If it's a single element symbol → return it directly.
      2. If it's a compound (e.g. SnO2, Fe2O3, MgAl2O4):
         → return the first non-oxygen metal element.

    Examples:
        "Fe"      → "Fe"
        "SnO2"    → "Sn"
        "Fe2O3"   → "Fe"
        "MgAl2O4" → "Mg"   (first listed metal)
    """
    # Lazy-import to avoid hard dependency at import time
    from pymatgen.core import Composition, Element

    # Simple element symbol (1-2 uppercase/lowercase chars, no digits)
    if re.fullmatch(r"[A-Z][a-z]?", formula):
        return formula

    try:
        comp = Composition(formula)
        # Filter out oxygen; take the first element by composition order
        metals = [str(el) for el in comp.elements if str(el) != "O"]
        if metals:
            return metals[0]
    except Exception:
        pass

    # Fallback: grab first capitalised token from the string
    match = re.match(r"([A-Z][a-z]?)", formula)
    if match:
        return match.group(1)

    raise ValueError(f"Cannot extract cation from formula: {formula!r}")


# ── Component dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class DopantComponent:
    """A single dopant component with its formula and concentration."""

    formula: str    # original formula: "Fe", "SnO2", "MgAl2O4"
    cation: str     # substituting element: "Fe", "Sn", "Mg"
    conc: float     # fractional concentration (0 < conc < 1)

    def __post_init__(self):
        object.__setattr__(self, "conc", float(np.clip(self.conc, 1e-5, 0.49)))

    @classmethod
    def from_token(cls, token: str, default_conc: float | None = None) -> "DopantComponent":
        """
        Parse a single token like "Fe", "Fe:0.0265", "SnO2:0.013".

        If no concentration is given, `default_conc` is used (required in that case).
        """
        if ":" in token:
            formula_part, conc_part = token.split(":", 1)
            conc = float(conc_part)
        elif default_conc is not None:
            formula_part = token
            conc = default_conc
        else:
            raise ValueError(
                f"Token {token!r} has no concentration and no default_conc was provided."
            )
        formula_part = formula_part.strip()
        cation = _extract_cation(formula_part)
        return cls(formula=formula_part, cation=cation, conc=conc)


# ── DopantSpec ────────────────────────────────────────────────────────────────

@dataclass
class DopantSpec:
    """
    Parsed representation of a (possibly complex) dopant specification.

    Attributes:
        raw: The original spec string.
        components: List of DopantComponent, one per dopant formula.
    """

    raw: str
    components: list[DopantComponent] = field(default_factory=list)

    # ── Parsing ───────────────────────────────────────────────────────────────

    @classmethod
    def parse(
        cls,
        spec_str: str,
        total_conc_frac: float | None = None,
    ) -> "DopantSpec":
        """
        Parse a dopant specification string into a DopantSpec.

        Args:
            spec_str: Specification string (see module docstring for formats).
            total_conc_frac: Total dopant concentration fraction.
                Used when individual concentrations are not given in spec_str
                (e.g. "Fe,Sn" → split equally). Ignored when all components
                embed their own concentration.

        Returns:
            DopantSpec instance.
        """
        spec_str = spec_str.strip()

        # ── Undoped sentinel ──────────────────────────────────────────────────
        if spec_str.lower() == "undoped":
            return cls(raw="undoped", components=[])

        tokens = [t.strip() for t in spec_str.split(",") if t.strip()]

        # Count how many tokens lack an explicit concentration
        missing = sum(1 for t in tokens if ":" not in t)

        if missing == 0:
            # All tokens have explicit concentrations
            components = [DopantComponent.from_token(t) for t in tokens]
        elif total_conc_frac is not None:
            # Distribute remaining concentration equally among un-specified tokens
            explicit_sum = sum(
                float(t.split(":")[1]) for t in tokens if ":" in t
            )
            per_missing = max(
                (total_conc_frac - explicit_sum) / max(missing, 1), 1e-5
            )
            components = [
                DopantComponent.from_token(t, default_conc=per_missing)
                for t in tokens
            ]
        else:
            raise ValueError(
                f"Spec {spec_str!r} contains tokens without concentration and "
                "no total_conc_frac was provided."
            )

        return cls(raw=spec_str, components=components)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_undoped(self) -> bool:
        """True when no dopant components are present (pure Ga₂O₃ host)."""
        return not self.components

    @property
    def total_conc(self) -> float:
        """Sum of all component concentrations."""
        return sum(c.conc for c in self.components)

    def cache_key(self) -> str:
        """
        Deterministic string key for graph/feature caching.
        Sorts by cation name so "Fe:0.013,Sn:0.013" == "Sn:0.013,Fe:0.013".
        """
        if not self.components:
            return "undoped"
        parts = sorted(
            f"{c.cation}-{c.conc:.6f}" for c in self.components
        )
        return "_".join(parts)

    def label(self) -> str:
        """Human-readable label: "Fe", "Fe+Sn", "SnO2+MgO"."""
        if not self.components:
            return "undoped"
        return "+".join(c.formula for c in self.components)

    def cation_label(self) -> str:
        """Label using only cation symbols: "Fe", "Fe+Sn"."""
        if not self.components:
            return "undoped"
        return "+".join(sorted(c.cation for c in self.components))

    # ── Formula builders ──────────────────────────────────────────────────────

    def to_xenonpy_formula(self) -> str:
        """
        Build the full doped Ga₂O₃ formula string for XenonPy.

        Convention: concentration_frac is relative to the Ga₂ sub-lattice
        (so 2 × conc atoms per formula unit replace Ga).

        Example:
            "Fe:0.0265"              → "Fe0.053000Ga1.947000O3"
            "Fe:0.0133,Sn:0.0132"   → "Fe0.026600Sn0.026400Ga1.947000O3"
        """
        dopant_parts = ""
        total_dopant = 0.0
        for comp in sorted(self.components, key=lambda c: c.cation):
            count = 2.0 * comp.conc
            dopant_parts += f"{comp.cation}{count:.6f}"
            total_dopant += count
        ga_count = max(2.0 - total_dopant, 0.01)
        return f"{dopant_parts}Ga{ga_count:.6f}O3"

    def to_substitution_dict(self) -> dict[str, float]:
        """
        Build substitution dict for pymatgen SubstitutionTransformation.

        Maps each Ga site to either a dopant cation or remaining Ga.

        Example:
            "Fe:0.0133,Sn:0.0132" →
            {"Fe": 0.0133, "Sn": 0.0132, "Ga": 0.9735}
        """
        total = sum(c.conc for c in self.components)
        ga_remaining = max(1.0 - total, 1e-4)
        d: dict[str, float] = {c.cation: c.conc for c in self.components}
        d["Ga"] = ga_remaining
        # Normalise so values sum to exactly 1.0
        s = sum(d.values())
        return {k: v / s for k, v in d.items()}

    def to_cif_filename(self) -> str:
        """Filename-safe string for saving a CIF, e.g. "Fe-Sn_0.0133-0.0132.cif"."""
        if not self.components:
            return "Ga2O3_base.cif"
        parts = sorted(self.components, key=lambda c: c.cation)
        name = "_".join(f"{c.cation}-{c.conc:.4f}" for c in parts)
        return f"{name}.cif"

    def __repr__(self) -> str:
        return f"DopantSpec({self.raw!r})"


# ── Convenience function ──────────────────────────────────────────────────────

def parse_spec(spec_str: str, total_conc_frac: float | None = None) -> DopantSpec:
    """Shorthand for DopantSpec.parse()."""
    return DopantSpec.parse(spec_str, total_conc_frac)
