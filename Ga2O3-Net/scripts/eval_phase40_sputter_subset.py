"""Filter Phase 40 OOF to sputter-only rows and report metrics for
apples-to-apples comparison with Phase 33/27 baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 2 or np.std(y_true) < 1e-9:
        return dict(n=len(y_true), r=float("nan"), r2=float("nan"),
                    mae=float("nan"), rmse=float("nan"))
    r = float(stats.pearsonr(y_true, y_pred).statistic)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(n=len(y_true), r=r, r2=r2, mae=mae, rmse=rmse)


def evaluate_phase40_sputter(target: str) -> None:
    target_col = "vacancy_concentration" if target == "VC" else "photo_dark_ratio"
    bundle_name = f"phase40_maux_{'vc' if target == 'VC' else 'pdr'}_5seed"
    csv_name = (
        "ga2o3_exp_aug3x_vcinterp_maux.csv"
        if target == "VC"
        else "ga2o3_exp_aug3x_interp_maux.csv"
    )

    oof = pd.read_csv(PROJ / "results" / bundle_name / "oof_predictions.csv")
    src = pd.read_csv(PROJ / "data" / "raw" / "experimental" / csv_name)
    src = src[src.get("usable_flag", "") != "exotic_skip"].reset_index(drop=True)
    src = src[~src["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    src["sample_idx"] = np.arange(len(src), dtype=int)

    merged = oof.merge(
        src[["sample_idx", "method", "sample_weight"]], on="sample_idx", how="left"
    )
    merged["is_sputter"] = (
        merged["method"].fillna("").str.lower().str.contains("sputter")
    )

    pred_col       = f"{target_col}_pred"
    pred_platt_col = f"{target_col}_pred_platt"
    true_col       = f"{target_col}_true"

    valid = merged.dropna(subset=[pred_col, pred_platt_col, true_col])

    print(f"\n=== Phase 40 MAUX {target} — sputter-only subset of OOF ===")
    sub = valid[valid.is_sputter]
    print(f"  total OOF rows         : {len(merged)}")
    print(f"  rows w/ valid label    : {len(valid)}")
    print(f"  sputter rows w/ label  : {len(sub)}")
    print()

    print(f"  Phase 40 MAUX (sputter subset of OOF):")
    for label, col in [("raw  ", pred_col), ("Platt", pred_platt_col)]:
        m = metrics(sub[true_col].to_numpy(), sub[col].to_numpy())
        print(f"    {label} N={m['n']:3d}  R²={m['r2']:+.3f}  r={m['r']:+.3f}  "
              f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")

    print(f"\n  Per-element (sputter subset):")
    for elem in sorted(sub.dopant_label.unique(), key=lambda e: -len(sub[sub.dopant_label == e])):
        ee = sub[sub.dopant_label == elem]
        if len(ee) < 2: continue
        m = metrics(ee[true_col].to_numpy(), ee[pred_platt_col].to_numpy())
        print(f"    {elem:8s} N={m['n']:3d}  r={m['r']:+.3f}  MAE={m['mae']:.3f}")


if __name__ == "__main__":
    evaluate_phase40_sputter("VC")
    evaluate_phase40_sputter("PDR")
