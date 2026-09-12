"""Phase 54 V54-A1 — Build V53-ζ pseudo-target cache for self-distillation.

Reads V53-ζ 5-seed OOF predictions and stores Platt-calibrated V_O pseudo-
targets for V_O-UNLABELED rows. The cache is consumed by
finetune_trainer._pdr_self_distill_loss in V54-A1 training.

The cache is keyed by `sample_idx`, which matches Ga2O3ExpDataset row order
AFTER the standard filters (usable_flag != exotic_skip, exclude elements
[N, H, Nb, In], reset_index). This is the same filter applied in
scripts/eval_phase42_vs_baselines.py:46-49 and produces 420 rows for the
current data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv (488 raw).

Output: data/processed/v53z_pseudo_targets_vo.npz with arrays
  sample_idx: int32[N]
  pseudo_log_vo: float32[N]   (NaN on rows that already have a true label)
  pseudo_std: float32[N]      (V53-ζ MC-dropout std; used for inverse-var weighting)
  dopant_label: str[N]        (audit only — joins back to dataset.df)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parent.parent
OOF_PATH = PROJ / "results" / "phase53v53z_contrastive_vc_5seed" / "oof_predictions.csv"
SRC_CSV = PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv"
OUT_PATH = PROJ / "data" / "processed" / "v53z_pseudo_targets_vo.npz"


def _apply_dataset_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate Ga2O3ExpDataset / eval_phase42 post-filter row ordering."""
    if "usable_flag" in df.columns:
        df = df[df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    df = df[~df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    return df


def main() -> None:
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"V53-ζ OOF missing: {OOF_PATH}")
    if not SRC_CSV.exists():
        raise FileNotFoundError(f"Source CSV missing: {SRC_CSV}")

    oof = pd.read_csv(OOF_PATH)
    src = _apply_dataset_filter(pd.read_csv(SRC_CSV))
    src["sample_idx"] = np.arange(len(src), dtype=int)

    if len(oof) != len(src):
        raise RuntimeError(
            f"OOF / src length mismatch: OOF={len(oof)} src={len(src)}. "
            "Filter recipe drifted; re-check dataset.py."
        )

    merged = oof.merge(
        src[["sample_idx", "vacancy_concentration", "element"]],
        on="sample_idx",
        how="left",
        suffixes=("", "_src"),
    )

    # Pseudo only on V_O-UNLABELED rows (true target NaN in source CSV).
    # If we distilled onto labeled rows, we'd reinforce V53-ζ's own residuals
    # there and squash the supervised gradient.
    src_true_nan = merged["vacancy_concentration"].isna().values
    pseudo = merged["vacancy_concentration_pred_platt"].astype(np.float32).values.copy()
    pseudo[~src_true_nan] = np.nan
    std = merged["vacancy_concentration_std"].astype(np.float32).values.copy()
    std[~src_true_nan] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        sample_idx=merged["sample_idx"].astype(np.int32).values,
        pseudo_log_vo=pseudo,
        pseudo_std=std,
        dopant_label=merged["dopant_label"].astype(str).values,
    )

    n_unlabeled = int(np.isnan(merged["vacancy_concentration"].values).sum())
    n_pseudo = int((~np.isnan(pseudo)).sum())
    print(f"Wrote {OUT_PATH}")
    print(f"  Total rows           : {len(merged)}")
    print(f"  V_O-labeled (skip)   : {len(merged) - n_unlabeled}")
    print(f"  V_O-unlabeled (used) : {n_unlabeled}")
    print(f"  Pseudo non-NaN       : {n_pseudo}")
    print(f"  Pseudo log V_O range : [{np.nanmin(pseudo):.3f}, {np.nanmax(pseudo):.3f}]")
    print(f"  Pseudo std median    : {np.nanmedian(std):.3f}")


if __name__ == "__main__":
    main()
