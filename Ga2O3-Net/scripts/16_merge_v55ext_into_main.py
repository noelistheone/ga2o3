"""Phase 55 V55-Ext integration — merge LLM-extracted rows into main CSV.

Reads `data/v55_llm_extracted.csv` (V55-Ext output) and merges high-confidence
rows into `data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v55.csv` with a
`sample_weight = confidence_weight` column. Then retrains V54-A1 on the
merged dataset.

V55 Roadmap Rule 3 (Confidence weighting):
  - confidence_weight >= 0.9 + manual verify → V_O regression head usage
  - confidence_weight in [0.5, 0.9) → SINCERE contrastive only (filter VC regression)
  - confidence_weight < 0.5 → drop

Output:
  - `data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v55.csv` — merged
  - retrain ready: scripts/04_finetune_predict.py with updated CSV path
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v55_merge")


def _normalize_v55ext_row(row: dict, default_method: str = "RF magnetron sputtering") -> dict | None:
    """Convert V55-Ext row → ga2o3 main CSV row format.

    Filters: must have at least one of (PDR, V_O, responsivity) numeric;
             must be a sputter method;
             dopant must be a single-element symbol (no co-doping for first pass).
    """
    # Required: numeric PDR or V_O or responsivity
    pdr = row.get("PDR")
    v_o = row.get("V_O_per_cm3")
    resp = row.get("responsivity_AW")
    dark = row.get("dark_current_A")
    photo = row.get("photo_current_A")

    def _coerce_number_or_corrected(v):
        """Coerce v to float, or extract 'corrected:<value>' from a verification annotation string."""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v)
        # Patterns like 'unstable|corrected:168.0' or 'corrected:1.23e-5'
        import re
        m = re.search(r"corrected:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        # Strip non-numeric prefix/suffix
        m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    # Skip rows with no usable measurement
    pdr = _coerce_number_or_corrected(pdr)
    v_o = _coerce_number_or_corrected(v_o)
    resp = _coerce_number_or_corrected(resp)
    dark = _coerce_number_or_corrected(dark)
    photo = _coerce_number_or_corrected(photo)

    valid_pdr = False
    if pdr is not None and pdr > 0:
        valid_pdr = True

    valid_v_o = False
    if v_o is not None and v_o > 0:
        valid_v_o = True

    if not (valid_pdr or valid_v_o):
        # Try deriving PDR from photo/dark
        if photo is not None and dark is not None and photo > 0 and dark > 0:
            pdr = photo / dark
            valid_pdr = True
    if not (valid_pdr or valid_v_o):
        return None

    # Dopant: single element only
    dopant = str(row.get("dopant", "")).strip()
    if not dopant or dopant.lower() in ("none", "undoped", "nan"):
        return None
    if "+" in dopant or "," in dopant:
        return None  # skip co-doping for v1
    # Sanitize: must be 1-2 char chemical symbol
    if not dopant[0].isupper() or len(dopant) > 3:
        return None

    # Concentration
    c_at = row.get("dopant_at_pct")
    try:
        c_at = float(c_at) if c_at is not None and not pd.isna(c_at) else 0.0
    except (TypeError, ValueError):
        c_at = 0.0

    # Standardize method
    method = str(row.get("method", "")).lower()
    if "sputter" not in method and "magnetron" not in method:
        return None
    out_method = default_method

    # Atmosphere from Ar_O2_ratio
    ar_o2 = str(row.get("Ar_O2_ratio", "") or "")
    if ":" in ar_o2:
        ar, o2 = ar_o2.split(":")[:2]
        try:
            ar_f = float(ar)
            o2_f = float(o2)
            tot = ar_f + o2_f
            if tot > 0:
                o2_frac = o2_f / tot
                if o2_frac < 0.05:
                    atmosphere = "Ar"
                elif o2_frac < 0.5:
                    atmosphere = f"Ar:O2={int(ar_f):d}:{int(o2_f):d}"
                else:
                    atmosphere = "O2"
            else:
                atmosphere = "Ar"
        except (TypeError, ValueError):
            atmosphere = "Ar"
    else:
        atmosphere = "Ar"

    # Anneal info
    anneal_T = row.get("post_anneal_T_C")
    anneal_atm = row.get("post_anneal_atm")
    anneal_time = row.get("post_anneal_time_min")

    return {
        "dopant_spec": f"{dopant}:{c_at / 100:.6f}" if c_at > 0 else f"{dopant}:0.001000",
        "element": dopant,
        "method": out_method,
        "concentration_at%": c_at,
        "atmosphere": atmosphere,
        "temperature_C": float(anneal_T) if anneal_T not in (None, "", "null") and not pd.isna(anneal_T) else np.nan,
        "time_min": float(anneal_time) if anneal_time not in (None, "", "null") and not pd.isna(anneal_time) else np.nan,
        "photocurrent_uA": float(photo) * 1e6 if photo and not pd.isna(photo) else np.nan,
        "dark_current_pA": float(dark) * 1e12 if dark and not pd.isna(dark) else np.nan,
        "photo_dark_ratio": np.log10(pdr) if valid_pdr else np.nan,
        "sensitivity": np.nan,
        "vacancy_concentration": np.log10(v_o) if valid_v_o else np.nan,
        "measurement_voltage_V": float(row.get("bias_V") or 10.0),
        "measurement_wavelength_nm": float(row.get("wavelength_nm") or 254.0),
        "notes": f"V55-Ext from {row.get('source_pdf', '')[:50]}",
        "doi": f"V55-Ext:{row.get('source_pdf_dir', '')}/{row.get('source_pdf', '')[:30]}",
        "usable_flag": "v55_ext",
        "sample_weight": float(row.get("confidence_weight", 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V55-Ext merge into main CSV")
    parser.add_argument("--extracted-csv", type=Path,
                        default=PROJ / "data" / "v55_llm_extracted.csv")
    parser.add_argument("--main-csv", type=Path,
                        default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")
    parser.add_argument("--output-csv", type=Path,
                        default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v55.csv")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Drop rows with confidence_weight < this")
    parser.add_argument("--vo-min-confidence", type=float, default=0.9,
                        help="V_O column only kept if confidence ≥ this")
    args = parser.parse_args()

    main_df = pd.read_csv(args.main_csv)
    logger.info(f"Main CSV: {len(main_df)} rows")
    # Ensure sample_weight column exists in main CSV
    if "sample_weight" not in main_df.columns:
        main_df["sample_weight"] = 1.0
    if "usable_flag" not in main_df.columns:
        main_df["usable_flag"] = "ok"

    if not args.extracted_csv.exists():
        logger.error(f"V55-Ext CSV not found: {args.extracted_csv}")
        return
    ext_df = pd.read_csv(args.extracted_csv)
    logger.info(f"V55-Ext CSV: {len(ext_df)} rows")

    if len(ext_df) == 0:
        logger.warning("V55-Ext CSV is empty; writing main_csv unchanged.")
        main_df.to_csv(args.output_csv, index=False)
        return

    # Confidence distribution
    conf = ext_df.get("confidence_weight", pd.Series([0.5] * len(ext_df))).astype(float)
    logger.info(f"V55-Ext confidence: median {conf.median():.2f}, "
                f"≥0.9: {(conf >= 0.9).sum()}, "
                f"≥0.5: {(conf >= 0.5).sum()}, "
                f"≥0.4: {(conf >= 0.4).sum()}")

    # Filter + normalize
    new_rows = []
    n_skipped = 0
    for _, row in ext_df.iterrows():
        if row.get("confidence_weight", 0) < args.min_confidence:
            n_skipped += 1
            continue
        normalized = _normalize_v55ext_row(row.to_dict())
        if normalized is None:
            n_skipped += 1
            continue
        # If V_O present but confidence below VO threshold, mask V_O
        if normalized["sample_weight"] < args.vo_min_confidence:
            if not np.isnan(normalized.get("vacancy_concentration", np.nan)):
                logger.debug(f"  Masking V_O for low-confidence row (conf={normalized['sample_weight']:.2f})")
                normalized["vacancy_concentration"] = np.nan
        new_rows.append(normalized)

    logger.info(f"V55-Ext normalized: {len(new_rows)} rows kept, {n_skipped} skipped")

    if not new_rows:
        logger.warning("No usable V55-Ext rows; writing main CSV unchanged.")
        main_df.to_csv(args.output_csv, index=False)
        return

    new_df = pd.DataFrame(new_rows)
    # Align columns with main_df
    for col in main_df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan if col not in ("usable_flag", "sample_weight") else (
                "v55_ext" if col == "usable_flag" else 0.5
            )
    new_df = new_df[main_df.columns]

    merged = pd.concat([main_df, new_df], ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote merged CSV with {len(merged)} rows to {args.output_csv}")
    logger.info(f"  Original: {len(main_df)} | V55-Ext added: {len(new_df)}")
    logger.info(f"  V_O-labeled in merged: "
                f"{merged['vacancy_concentration'].notna().sum()} "
                f"(was {main_df['vacancy_concentration'].notna().sum()})")
    logger.info(f"  PDR-labeled in merged: "
                f"{merged['photo_dark_ratio'].notna().sum()} "
                f"(was {main_df['photo_dark_ratio'].notna().sum()})")


if __name__ == "__main__":
    main()
