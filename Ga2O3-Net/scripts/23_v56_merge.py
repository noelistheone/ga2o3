"""V56-Ext-2 merge — append extracted SputterRow rows to main training CSV.

Reads `data/v56_llm_extracted.csv` (V56-Ext-2 output) and merges high-confidence
rows into `data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv` with
`sample_weight = confidence_weight`. Output goes to
`data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v56.csv`.

V56 Rule 3 (confidence weighting):
  - confidence_weight ≥ 0.9 + manual verify → V_O regression head usage
  - confidence_weight ∈ [0.5, 0.9)         → SINCERE contrastive only (mask V_O)
  - confidence_weight < 0.5                → drop

V56-Ext-2 row schema (SputterRow) → main CSV schema:
  - dopant + dopant_at_pct        → dopant_spec, element, concentration_at%
  - deposition_method "RF sputter" → method "RF magnetron sputtering"
  - O2_Ar_ratio                   → atmosphere string ("Ar"/"Ar:O2=3:1"/"O2")
  - anneal_T_C, anneal_atm, anneal_time_min → temperature_C, atmosphere, time_min
  - PDR_log10                     → photo_dark_ratio (already log10)
  - V_O_log10_cm3                 → vacancy_concentration (already log10)
  - photo_current_A × 1e6         → photocurrent_uA
  - dark_current_A × 1e12         → dark_current_pA
  - source_paper_id               → doi field (prefixed)
  - confidence_weight             → sample_weight
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v56_merge")


def _normalize_v56_row(row: dict, vo_min_confidence: float) -> dict | None:
    """Convert V56-Ext-2 row → main CSV row format.

    Filters:
      - must have at least one of (PDR_log10, V_O_log10_cm3) numeric
      - must be a sputter method (else SputterRow schema would have caught it)
      - dopant must be single-element symbol (Pydantic Literal ensures this)
    """
    pdr_log10 = row.get("PDR_log10")
    v_o_log10 = row.get("V_O_log10_cm3")
    photo = row.get("photo_current_A")
    dark = row.get("dark_current_A")

    def _f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    pdr_log10 = _f(pdr_log10)
    v_o_log10 = _f(v_o_log10)
    photo = _f(photo)
    dark = _f(dark)

    if pdr_log10 is None and v_o_log10 is None:
        # Try deriving PDR from photo/dark
        if photo is not None and dark is not None and photo > 0 and dark > 0:
            pdr_log10 = float(np.log10(photo / dark))

    if pdr_log10 is None and v_o_log10 is None:
        return None

    # Sputter check (V56 schema constrains deposition_method via Literal)
    method = str(row.get("deposition_method") or "").lower()
    if "sputter" not in method:
        return None

    dopant = (str(row.get("dopant") or "")).strip()
    if not dopant or dopant.lower() in ("none", "undoped", "nan", "null"):
        return None

    c_at = _f(row.get("dopant_at_pct")) or 0.0

    # Atmosphere from O2_Ar_ratio (V56 stores ratio as float; build string)
    ratio = _f(row.get("O2_Ar_ratio"))
    if ratio is None:
        atmosphere = "Ar"
    elif ratio <= 0.02:
        atmosphere = "Ar"
    elif ratio >= 4.0:
        atmosphere = "O2"
    else:
        # Convert O2/Ar ratio to Ar:O2=A:B with small integers
        o2 = ratio
        ar = 1.0
        # Scale to integer ratio
        if ratio < 1.0:
            ar = round(1.0 / ratio)
            o2_part = 1
            atmosphere = f"Ar:O2={int(ar)}:{o2_part}"
        else:
            ar_part = 1
            o2 = round(ratio)
            atmosphere = f"Ar:O2={ar_part}:{int(o2)}"

    # Anneal info → primary temperature_C / atmosphere / time_min slot
    anneal_T = _f(row.get("anneal_T_C"))
    anneal_atm = row.get("anneal_atm")
    anneal_time = _f(row.get("anneal_time_min"))

    if anneal_T is not None and isinstance(anneal_atm, str) and anneal_atm:
        # Anneal supersedes deposition atmosphere for "atmosphere" column
        if anneal_atm.lower() in ("o2", "n2", "ar", "air", "vacuum"):
            atmosphere = anneal_atm.upper() if anneal_atm.lower() == "o2" else anneal_atm.capitalize()

    return {
        "dopant_spec": f"{dopant}:{c_at / 100:.6f}" if c_at > 0 else f"{dopant}:0.001000",
        "element": dopant,
        "method": "RF magnetron sputtering",
        "concentration_at%": c_at,
        "atmosphere": atmosphere,
        "temperature_C": anneal_T if anneal_T is not None else np.nan,
        "time_min": anneal_time if anneal_time is not None else np.nan,
        "photocurrent_uA": photo * 1e6 if photo is not None else np.nan,
        "dark_current_pA": dark * 1e12 if dark is not None else np.nan,
        "photo_dark_ratio": pdr_log10 if pdr_log10 is not None else np.nan,
        "sensitivity": np.nan,
        "vacancy_concentration": v_o_log10 if v_o_log10 is not None else np.nan,
        "measurement_voltage_V": _f(row.get("bias_V")) or 10.0,
        "measurement_wavelength_nm": _f(row.get("wavelength_nm")) or 254.0,
        "notes": f"V56-Ext-2 confidence={row.get('confidence', 'low')} | {(row.get('quote') or '')[:80]}",
        "doi": f"V56-Ext:{row.get('source_paper_id', 'unknown')}",
        "usable_flag": "v56_ext",
        "sample_weight": float(row.get("confidence_weight", 0.5)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V56-Ext-2 merge into main CSV")
    ap.add_argument("--extracted-csv", type=Path,
                    default=PROJ / "data" / "v56_llm_extracted.csv")
    ap.add_argument("--main-csv", type=Path,
                    default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")
    ap.add_argument("--output-csv", type=Path,
                    default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v56.csv")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="Drop rows with confidence_weight < this")
    ap.add_argument("--vo-min-confidence", type=float, default=0.9,
                    help="V_O column only kept if confidence ≥ this")
    args = ap.parse_args()

    main_df = pd.read_csv(args.main_csv)
    logger.info(f"Main CSV: {len(main_df)} rows")
    if "sample_weight" not in main_df.columns:
        main_df["sample_weight"] = 1.0
    if "usable_flag" not in main_df.columns:
        main_df["usable_flag"] = "ok"

    if not args.extracted_csv.exists():
        logger.error(f"V56-Ext-2 CSV not found: {args.extracted_csv}")
        return
    ext_df = pd.read_csv(args.extracted_csv)
    logger.info(f"V56-Ext-2 CSV: {len(ext_df)} rows")

    if len(ext_df) == 0:
        logger.warning("V56-Ext-2 CSV is empty; writing main_csv unchanged.")
        main_df.to_csv(args.output_csv, index=False)
        return

    conf = ext_df["confidence_weight"].astype(float)
    logger.info(
        f"V56-Ext-2 confidence: median {conf.median():.2f}, "
        f"≥0.9: {(conf >= 0.9).sum()}, "
        f"≥0.5: {(conf >= 0.5).sum()}, "
        f"V_O field nonnull: {ext_df['V_O_log10_cm3'].notna().sum()}, "
        f"PDR field nonnull: {ext_df['PDR_log10'].notna().sum()}"
    )

    new_rows = []
    n_skipped = 0
    for _, row in ext_df.iterrows():
        if float(row.get("confidence_weight", 0)) < args.min_confidence:
            n_skipped += 1
            continue
        normalized = _normalize_v56_row(row.to_dict(), args.vo_min_confidence)
        if normalized is None:
            n_skipped += 1
            continue
        if normalized["sample_weight"] < args.vo_min_confidence:
            if not (isinstance(normalized.get("vacancy_concentration"), float)
                    and np.isnan(normalized["vacancy_concentration"])):
                logger.debug(f"  Masking V_O for low-conf row (conf={normalized['sample_weight']:.2f})")
                normalized["vacancy_concentration"] = np.nan
        new_rows.append(normalized)

    logger.info(f"V56-Ext-2 normalized: {len(new_rows)} kept, {n_skipped} skipped")

    if not new_rows:
        logger.warning("No usable V56-Ext-2 rows; writing main_csv unchanged.")
        main_df.to_csv(args.output_csv, index=False)
        return

    new_df = pd.DataFrame(new_rows)
    for col in main_df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan if col not in ("usable_flag", "sample_weight") else (
                "v56_ext" if col == "usable_flag" else 0.5
            )
    new_df = new_df[main_df.columns]

    merged = pd.concat([main_df, new_df], ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote merged CSV with {len(merged)} rows → {args.output_csv}")
    logger.info(f"  Original: {len(main_df)} | V56-Ext-2 added: {len(new_df)}")
    logger.info(
        f"  V_O-labeled in merged: {merged['vacancy_concentration'].notna().sum()} "
        f"(was {main_df['vacancy_concentration'].notna().sum()})"
    )
    logger.info(
        f"  PDR-labeled in merged: {merged['photo_dark_ratio'].notna().sum()} "
        f"(was {main_df['photo_dark_ratio'].notna().sum()})"
    )


if __name__ == "__main__":
    main()
