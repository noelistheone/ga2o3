"""
prepare_maux_csv.py — generate MAUX (multi-method auxiliary) variants of
the augmented experimental CSVs for Phase 40 training.

Phase 40 strategy: train on the FULL multi-method dataset (488 / 494 rows)
instead of the sputter-only subset (194 / 215 rows). The Brouwer / main
MSE supervision is restricted to sputter rows by zeroing the
`sample_weight` for everything else; auxiliary losses (InfoNCE, monotone,
atmosphere-ordinal, class-aware monotone, reconstruction) continue to
operate on the full batch regardless of sample_weight, so non-sputter
rows shape the encoder + composition + process + fusion latent without
contaminating the sputter-tuned head.

We MULTIPLY the existing sample_weight by an indicator (1.0 if sputter,
0.0 otherwise) so the original quality weighting (real=1.0, interp=0.5,
already-masked=0.0) is preserved.

Run once:
    python scripts/prepare_maux_csv.py

Outputs (next to source):
  data/raw/experimental/ga2o3_exp_aug3x_vcinterp_maux.csv
  data/raw/experimental/ga2o3_exp_aug3x_interp_maux.csv

The _sputter.csv subset rule used elsewhere in the project is:
    method.lower().contains("sputter")
We use the same rule here so the sample_weight=1.0 rows in the MAUX file
are exactly the rows in the corresponding _sputter.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "raw" / "experimental"

SOURCES = [
    ("ga2o3_exp_aug3x_vcinterp.csv", "ga2o3_exp_aug3x_vcinterp_maux.csv"),
    ("ga2o3_exp_aug3x_interp.csv",   "ga2o3_exp_aug3x_interp_maux.csv"),
]


def main() -> None:
    for src_name, dst_name in SOURCES:
        src = DATA / src_name
        dst = DATA / dst_name
        if not src.exists():
            print(f"[skip] {src} not found")
            continue

        df = pd.read_csv(src)
        n_total = len(df)

        method_str = df["method"].fillna("").str.lower()
        sputter_mask = method_str.str.contains("sputter")
        n_sputter = int(sputter_mask.sum())

        # Existing sample_weight already encodes data quality (real=1.0,
        # interp=0.5, mask=0.0). Multiply by sputter indicator so rows
        # outside sputter scope contribute zero to the main MSE / Brouwer
        # supervision but still flow through auxiliary losses.
        existing_w = df["sample_weight"].fillna(1.0).astype(float)
        sputter_w  = sputter_mask.astype(float)
        new_w = (existing_w * sputter_w).clip(0.0, 1.0)

        df_out = df.copy()
        df_out["sample_weight"] = new_w
        df_out.to_csv(dst, index=False)

        n_active = int((new_w > 0).sum())
        n_zero   = int((new_w == 0).sum())
        print(f"{src_name:50s} → {dst_name}")
        print(f"  total rows         : {n_total}")
        print(f"  sputter rows       : {n_sputter}")
        print(f"  active (w > 0)     : {n_active}  (sputter ∩ existing_w>0)")
        print(f"  masked (w == 0)    : {n_zero}    (non-sputter ∪ already-masked)")
        # Sanity: how many of those active rows have valid VC label?
        for tcol in ("vacancy_concentration", "photo_dark_ratio"):
            if tcol in df_out.columns:
                with_label_active = int(
                    ((new_w > 0) & df_out[tcol].notna()).sum()
                )
                with_label_total = int(df_out[tcol].notna().sum())
                print(f"  rows w/ {tcol:25s} active label: "
                      f"{with_label_active} / total {with_label_total}")
        print()


if __name__ == "__main__":
    main()
