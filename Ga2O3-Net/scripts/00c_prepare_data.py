"""
scripts/00c_prepare_data.py
───────────────────────────
One-shot xlsx → CSV cleaning script for the experimental dataset.

Run once to produce:
    data/raw/experimental/ga2o3_exp.csv

Targets (photo_dark_ratio, vacancy_concentration) are stored as log10(value)
in the output CSV. Expected ranges after transform:
    photo_dark_ratio      ∈ [0.85, 8.3]
    vacancy_concentration ∈ [9.97, 19.3]

Usage:
    conda activate ga2o3
    python scripts/00c_prepare_data.py
"""

from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
import pandas as pd
import openpyxl

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX_PATH = os.path.join(ROOT, "data", "raw", "experimental", "ga2o3_exp_v7_conc_filled.xlsx")
OUT_CSV   = os.path.join(ROOT, "data", "raw", "experimental", "ga2o3_exp.csv")

# ── Unicode superscript → ASCII digit mapping ──────────────────────────────
_SUPERSCRIPT = str.maketrans({
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
    "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
    "\u2078": "8", "\u2079": "9", "\u207b": "-",
    # subscript minus (cm⁻³)
    "\u208b": "-",
})

# Qualitative strings that should always parse to None
_QUALITATIVE_RE = re.compile(
    r"^(high|low|moderate|very|n-type|p-type|tunable|↓|↑|suppressed|enhanced|"
    r"gradient|improved|degraded|amorphous|signif|deep|surface|bandgap|"
    r"partially|crystal|intrinsic|balanced|ohmic|plasmon|donor|acceptor|"
    r"phase|nanorod|cryst|undoped|bg$|n-type$|p-type$)",
    re.IGNORECASE,
)

# Patterns that indicate exotic (non-β-Ga₂O₃) host phases
_EXOTIC_SPECS = [
    # AlGaO alloys
    "algao", "(al₀", "(al0", "alga₂o₃", "alga2o3",
    # ZnGaO / ZnGa₂O₄ compounds (not β-Ga₂O₃)
    "znga₂o₄", "znga2o4", "zngao", "znga",
    # InGaO / InAlGaO
    "ingao", "inal-ga", "inzn-ga",
    # Non-β phases: α (U+03B1), ε (U+03B5), κ (U+03BA)
    "\u03b1-ga", "α-ga", "alpha-ga",
    "\u03b5-ga", "ε-ga", "epsilon-ga",
    "\u03ba-ga", "κ-ga", "kappa-ga",
    # Other hosts
    "sno₂:f", "sno2:f", "cu₂o", "igzo",
    # Mixed oxides that are not β-Ga₂O₃ host
    "(ga,sn)o",   # (Ga,Sn)O₃ mixed oxide
]

# Patterns in dopant_spec that indicate legend/category rows (not data)
_LEGEND_PATTERNS = [
    "colour legend", "color legend",
    "original csv rows",
    "individual cell",
    "rf/dc magnetron",
    "pld (pulsed",
    "cvd / mocvd",
    "ald / sald",
    "mbe (molecular",
    "sol-gel / spray",
]


# ── Helper: extract measurement voltage/wavelength from `notes` ────────────

_VOLTAGE_BLACKLIST_PREFIXES = ("br", "bd", "ds", "gs", "th", "on", "off", "tj")

def extract_measurement_voltage(notes: str) -> float | None:
    """
    Extract the measurement (bias) voltage from the `notes` field.

    Handles patterns like:
        "10V;254nm"                → 10.0
        "0V self-powered;Id=1.43pA" → 0.0
        "20V;254nm;R=323 A/W"      → 20.0

    Skips obvious false positives like breakdown voltage:
        "SBD Vbr=497V"  → None  (preceded by 'br')
        "300-425V;CW"   → None  (voltage range, not bias)
    """
    if not notes or not isinstance(notes, str):
        return None
    # Primary: leading "{num}V;" or "{num}V " pattern
    m = re.match(r'^\s*(\d+\.?\d*)\s*V\b', notes)
    if m:
        return float(m.group(1))
    # Secondary: "{num}V" after a semicolon/space delimiter, but not preceded by letter
    # We capture leading context; reject if the char just before digit-run is a letter.
    for m in re.finditer(r'([;,]\s*|^)(\d+\.?\d*)\s*V\b', notes):
        val = float(m.group(2))
        # Reject voltage ranges like "300-425V"
        context = notes[max(0, m.start()-5):m.end()]
        if '-' in context.split('V')[0]:
            continue
        return val
    # Tertiary: "@30V" or "at 30V" patterns — common measurement notation
    m = re.search(r'(?:@|at\s+)(\d+\.?\d*)\s*V\b', notes, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def extract_measurement_wavelength(notes: str) -> float | None:
    """
    Extract the UV illumination wavelength (nm) from the `notes` field.

    Handles: "10V;254nm", "220nm DUV", "0V;265nm;R=0.94mA/W"

    DUV/UV typical wavelengths: 213, 220, 222, 230, 254, 255, 265, 280,
    365, 374, 401. Filter to 150-800 nm range to reject obvious false matches.
    """
    if not notes or not isinstance(notes, str):
        return None
    # Search for all "{num}nm" occurrences, take first in the sensible range.
    for m in re.finditer(r'(\d+\.?\d*)\s*nm\b', notes):
        val = float(m.group(1))
        if 150.0 <= val <= 800.0:
            return val
    return None


# ── Helper: parse scientific notation from vacancy_conc strings ────────────

def parse_sci_notation(s) -> float | None:
    """
    Convert a vacancy_conc cell value to a plain float (cm⁻³), or None.

    Handles:
      "3.166×10¹⁷"        → 3.166e17
      "~1×10¹⁸ cm⁻³"      → 1e18
      "~5e17 cm⁻³"        → 5e17
      "n-type 6.1×10¹⁷"   → 6.1e17
      "~SI (~1e17 Fe)"     → 1e17
      "~1e16-18"           → 10^17  (geometric mean of range)
      "~1.6×10¹⁵ bg"       → 1.6e15
      "2×10¹⁸"             → 2e18
      qualitative strings  → None
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s) if not math.isnan(float(s)) else None

    s = str(s).strip()
    if not s or s in ("—", "-", ""):
        return None

    # Translate Unicode superscripts to ASCII first
    s2 = s.translate(_SUPERSCRIPT)

    # Replace multiplication × with e-style (works before stripping)
    s2_sci = s2.replace("×10", "e").replace("x10", "e").replace("×", "e")

    # ── Try to find embedded scientific notation BEFORE qualitative check ──
    # This handles "n-type 6.1×10¹⁷" and "~SI (~1e17 Fe)" etc.

    # Detect range like "1e16-18" → geometric mean
    range_m = re.search(r"(\d+\.?\d*)[eE](\d+)-(\d+)", s2_sci, re.IGNORECASE)
    if range_m:
        mantissa = float(range_m.group(1)) if range_m.group(1) else 1.0
        exp_lo   = float(range_m.group(2))
        exp_hi   = float(range_m.group(3))
        mean_exp = (exp_lo + exp_hi) / 2.0
        return mantissa * 10 ** mean_exp

    # Parenthesised extract: "(~1e17 Fe)" → 1e17
    inner = re.search(r"\(~?([\d.]+[eE][\d]+)", s2_sci, re.IGNORECASE)
    if inner:
        try:
            return float(inner.group(1))
        except ValueError:
            pass

    # General embedded scientific notation search (covers "n-type 6.1e17")
    m = re.search(r"(\d+\.?\d*)[eE]([-]?\d+)", s2_sci)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass

    # ── Qualitative check (only if no numeric pattern found above) ─────────
    if _QUALITATIVE_RE.match(s):
        return None

    # Strip leading approximation markers and unit suffixes, try plain float
    s2 = s2_sci.lstrip("~≈≤≥<>")
    s2 = re.sub(r"\s*(cm[-\u207b]3|cm-3|cm⁻³)\s*$", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\s+\S+$", "", s2.strip())  # remove trailing word

    try:
        val = float(s2)
        if val > 0:
            return val
    except ValueError:
        pass

    return None


# ── Helper: normalize atmosphere string to ASCII-safe form ─────────────────

_ATM_MAP = {
    "O₂ plasma":    "O2_plasma",
    "O2 plasma":    "O2_plasma",
    "N₂":           "N2",
    "N₂+O₂":        "N2+O2",
    "N2+O₂":        "N2+O2",
    "N₂+O2":        "N2+O2",
    "Ar+N₂":        "Ar+N2",
    "Ar+N2":        "Ar+N2",
    "Ar+H₂":        "Ar+H2",
    "Ar+H₂ excess": "Ar+H2",
    "Ar+H2 excess": "Ar+H2",
    "Ar+H2":        "Ar+H2",
    "Ar:O₂=3:1":   "Ar:O2=3:1",
    "Ar:O2=3:1":   "Ar:O2=3:1",
    "Ar+O₂":        "Ar:O2=1:1",   # unknown ratio → assume 50/50
    "Ar+O2":        "Ar:O2=1:1",
    "O₂":           "O2",
    "H₂O+N₂O":      "N2O",
    "H2O+N2O":      "N2O",
    "N₂O":          "N2O",
    "N2O":          "N2O",
    "NaOH aq":      "wet",
    "seawater":     "wet",
    "vacuum":       "vacuum",
    "inert":        "inert",
    "—":            "Ar",           # missing → assume inert
    "-":            "Ar",
    "Ar+N₂→O₂":    "O2",           # sequential: Ar+N₂ dep → O₂ anneal; O₂ step dominates
    "Ar+N2→O2":    "O2",
    # New atmosphere strings from v5 dataset
    "O₃":           "O2",           # ozone → treat as pure oxidiser (same O₂ fraction)
    "O3":           "O2",
    "F-plasma":     "Ar",           # fluorine plasma: no O₂ contribution
    "inert/O₂":     "Ar:O2=1:1",   # ambiguous → assume 50/50
    "inert/O2":     "Ar:O2=1:1",
    "Ar:O₂":        "Ar:O2=1:1",   # bare Ar:O₂ without ratio → assume 50/50
    "Ar:O2":        "Ar:O2=1:1",
    "O₂+Ar":        "Ar:O2=1:1",
    "O2+Ar":        "Ar:O2=1:1",
    # New atmosphere strings from v6 dataset
    "NaOH":         "wet",          # aqueous NaOH electrolyte (PEC measurements)
    "N₂ plasma":    "N2_plasma",    # nitrogen plasma: N-species, no O₂, plasma-enhanced
    "N2 plasma":    "N2_plasma",
    "Ar+air":       "Ar+air",       # Ar diluted with air; O₂ fraction ≈ 0.105 (handled in dataset)
}


def normalize_atmosphere(atm) -> str:
    if atm is None:
        return "Ar"
    atm_s = str(atm).strip()
    if atm_s in _ATM_MAP:
        return _ATM_MAP[atm_s]
    # Strip trailing context words like "anneal" after the gas name
    base = atm_s.split()[0] if " " in atm_s else atm_s
    if base in _ATM_MAP:
        return _ATM_MAP[base]
    # Trailing context words: "Ar anneal" → "Ar"
    if " " in atm_s:
        first = atm_s.split()[0]
        if first in _ATM_MAP:
            return _ATM_MAP[first]
        # Passthrough the first word (e.g. "Ar" from "Ar anneal")
        return first

    # Passthrough for already-ASCII forms ("Ar", "air", "O2", etc.)
    return atm_s


# ── Helper: determine if a row is exotic ──────────────────────────────────

def is_exotic(spec: str, elem: str) -> bool:
    spec_lower = spec.lower()
    for pat in _EXOTIC_SPECS:
        if pat in spec_lower:
            return True
    # "(AlGaO ...)/GaN" style
    if spec.startswith("(AlGaO") or spec.startswith("(algao"):
        return True
    return False


def is_legend_row(raw_spec) -> bool:
    """Return True if this row is a legend/category entry, not real data."""
    if raw_spec is None:
        return True
    s = str(raw_spec).strip().lower()
    if not s or s == "nan":
        return True
    return any(pat in s for pat in _LEGEND_PATTERNS)


# ── Helper: build the canonical DopantSpec string ─────────────────────────

_NUMERIC_IN_SPEC_RE = re.compile(
    r"^[A-Za-zα-ωΑ-Ω\-]+:~?(\d+\.?\d*)\s*(?:%|wt%|at%|[\s(,]|$)"
)

def normalize_dopant_spec(
    raw_spec, element_str, conc_at_pct_val
) -> tuple[str, str]:
    """
    Returns (spec_str, usable_flag).

    flag ∈ {'ok', 'no_conc', 'exotic_skip'}

    spec_str format: "Fe:0.026500" or "Fe:0.013,Sn:0.013" or "undoped" or ""
    Concentrations in spec_str are FRACTIONS (divided by 100 from at%).
    """
    spec = str(raw_spec).strip() if raw_spec is not None else ""
    elem = str(element_str).strip() if element_str not in (None, "—", "–") else ""

    # Numeric concentration from the concentration (at%) column
    try:
        col_conc_at = float(conc_at_pct_val) if conc_at_pct_val not in (None, "—", "–", "") else None
    except (ValueError, TypeError):
        col_conc_at = None

    # ── 1. Exotic check ───────────────────────────────────────────────────
    if is_exotic(spec, elem):
        return "", "exotic_skip"

    # ── 2. Undoped check ─────────────────────────────────────────────────
    spec_lower = spec.lower()
    if spec_lower.startswith("undoped"):
        return "undoped", "ok"
    # Zero concentration (column) with no meaningful element
    if col_conc_at is not None and col_conc_at == 0.0:
        return "undoped", "ok"

    # ── 3. Parse element(s) ───────────────────────────────────────────────
    # Element column: "Fe", "Bi+Cu", "Mg+N", "In+Zn", etc.
    # Drop "Ga" (host), "—" (none)
    elements: list[str] = []
    if elem and elem not in ("—", "–", "-"):
        raw_elems = re.split(r"[+,]", elem)
        elements = [e.strip() for e in raw_elems if e.strip() and e.strip() != "Ga"]

    if not elements:
        # Fall back: try to extract element prefix from spec string
        # "Fe:2.65", "Si:α-Ga₂O₃", "B:interstitial", etc.
        m = re.match(r"^([A-Za-z]{1,3})[:\-]", spec)
        if m:
            candidate = m.group(1)
            # Exclude Greek-letter prefixes misread as element
            if candidate not in ("", "a", "b", "k"):
                elements = [candidate]

    if not elements:
        # Pure undoped host
        return "undoped", "ok"

    # ── 4. Determine concentration ────────────────────────────────────────
    # Try to extract embedded numeric concentration from spec string
    spec_conc_at: float | None = None
    m = _NUMERIC_IN_SPEC_RE.match(spec)
    if m:
        spec_conc_at = float(m.group(1))

    # Choose best concentration (prefer spec-embedded, fall back to column)
    best_at = spec_conc_at if spec_conc_at is not None else col_conc_at

    if best_at is None or best_at <= 0:
        # No usable concentration
        return "", "no_conc"

    # Convert at% → fraction
    conc_frac = best_at / 100.0

    # ── 5. Build spec string ──────────────────────────────────────────────
    if len(elements) == 1:
        return f"{elements[0]}:{conc_frac:.6f}", "ok"
    else:
        per_elem = conc_frac / len(elements)
        parts = [f"{e}:{per_elem:.6f}" for e in elements]
        return ",".join(parts), "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {XLSX_PATH} …")
    wb = openpyxl.load_workbook(XLSX_PATH)
    ws = wb.active

    # Read header row and strip whitespace / newlines
    raw_headers = [str(c.value).strip().replace("\n", " ") for c in ws[1]]
    print(f"  Columns: {raw_headers}")

    # Column-index mapping (0-based after we convert to list)
    COL = {h: i for i, h in enumerate(raw_headers)}

    def _get(row_vals, key, default=None):
        idx = COL.get(key)
        if idx is None:
            return default
        v = row_vals[idx]
        return v if v is not None else default

    rows_out = []
    stats = {"ok": 0, "no_conc": 0, "exotic_skip": 0, "legend_skip": 0}

    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True)):
        r = list(row)

        raw_spec    = _get(r, "dopant_spec")

        # Skip legend / category / header rows
        if is_legend_row(raw_spec):
            stats["legend_skip"] += 1
            continue
        element_str = _get(r, "element(s)")
        conc_at_pct = _get(r, "concentration (at%)")
        atmosphere  = _get(r, "atmosphere")
        temp_c      = _get(r, "temperature (°C)")
        time_min    = _get(r, "time (min)")
        photo_uA    = _get(r, "photocurrent (μA)")
        dark_pA     = _get(r, "dark_current (pA)")
        photo_dark  = _get(r, "photo_dark ratio")
        sensitivity = _get(r, "responsivity (A/W)")
        vac_raw     = _get(r, "vacancy_conc (cm⁻³)")
        notes       = _get(r, "notes", "")
        doi         = _get(r, "DOI / source", "")

        # Build canonical spec + flag
        clean_spec, flag = normalize_dopant_spec(raw_spec, element_str, conc_at_pct)
        stats[flag] += 1

        # Normalize atmosphere
        atm_clean = normalize_atmosphere(atmosphere)

        # ── photo_dark_ratio ───────────────────────────────────────────────
        pdr_raw = photo_dark
        if pdr_raw in (None, "—", "–", ""):
            pdr_log10 = float("nan")
        else:
            try:
                pdr_val = float(pdr_raw)
                pdr_log10 = math.log10(pdr_val) if pdr_val > 0 else float("nan")
            except (ValueError, TypeError):
                pdr_log10 = float("nan")

        # ── vacancy_concentration ──────────────────────────────────────────
        vac_val = parse_sci_notation(vac_raw)
        vac_log10 = math.log10(vac_val) if (vac_val is not None and vac_val > 0) else float("nan")

        # ── photocurrent, dark current, sensitivity (kept as-is, optional) ─
        def _to_float_or_nan(v):
            if v in (None, "—", "–", ""):
                return float("nan")
            try:
                return float(v)
            except (ValueError, TypeError):
                return float("nan")

        notes_str = str(notes).strip() if notes else ""
        meas_v  = extract_measurement_voltage(notes_str)
        meas_nm = extract_measurement_wavelength(notes_str)

        rows_out.append({
            "dopant_spec":          clean_spec,
            "element":              str(element_str).strip() if element_str else "—",
            "method":               str(_get(r, "method", "")).strip(),
            "concentration_at%":    float(conc_at_pct) if conc_at_pct not in (None, "—") else float("nan"),
            "atmosphere":           atm_clean,
            "temperature_C":        _to_float_or_nan(temp_c),
            "time_min":             _to_float_or_nan(time_min),
            "photocurrent_uA":      _to_float_or_nan(photo_uA),
            "dark_current_pA":      _to_float_or_nan(dark_pA),
            "photo_dark_ratio":     pdr_log10,
            "sensitivity":          _to_float_or_nan(sensitivity),
            "vacancy_concentration": vac_log10,
            "measurement_voltage_V": meas_v if meas_v is not None else float("nan"),
            "measurement_wavelength_nm": meas_nm if meas_nm is not None else float("nan"),
            "notes":                notes_str,
            "doi":                  str(doi).strip() if doi else "",
            "usable_flag":          flag,
        })

    df = pd.DataFrame(rows_out)

    print(f"\nRows: {len(df)}")
    print(f"Legend rows skipped: {stats['legend_skip']}")
    print(f"usable_flag distribution:\n{df['usable_flag'].value_counts().to_string()}")
    n_pdr = df["photo_dark_ratio"].notna().sum()
    n_vac = df["vacancy_concentration"].notna().sum()
    print(f"\nphoto_dark_ratio (log10) non-NaN: {n_pdr}")
    print(f"vacancy_concentration (log10) non-NaN: {n_vac}")

    # Range checks for non-exotic rows
    ok_df = df[df["usable_flag"] != "exotic_skip"]
    pdr_vals = ok_df["photo_dark_ratio"].dropna()
    vac_vals = ok_df["vacancy_concentration"].dropna()
    if len(pdr_vals):
        print(f"  photo_dark_ratio log10 range: [{pdr_vals.min():.2f}, {pdr_vals.max():.2f}]")
    if len(vac_vals):
        print(f"  vacancy_conc     log10 range: [{vac_vals.min():.2f}, {vac_vals.max():.2f}]")

    # Write CSV (NaN → empty string)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, na_rep="")
    print(f"\nWrote {len(df)} rows → {OUT_CSV}")

    # ── Verbose row report ─────────────────────────────────────────────────
    print("\nPer-row summary (spec, flag, pdr_log10, vac_log10):")
    for i, row_d in enumerate(rows_out):
        pdr_s = f"{row_d['photo_dark_ratio']:.2f}" if not math.isnan(row_d['photo_dark_ratio']) else "NaN"
        vac_s = f"{row_d['vacancy_concentration']:.2f}" if not math.isnan(row_d['vacancy_concentration']) else "NaN"
        print(
            f"  Row {i+1:3d}: [{row_d['usable_flag']:11s}] "
            f"spec={row_d['dopant_spec'][:35]:35s} "
            f"pdr={pdr_s:8s} vac={vac_s}"
        )


if __name__ == "__main__":
    main()
