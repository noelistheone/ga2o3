"""V52g — Gaussian Process residual on top of V5 OOF predictions.

Trains a Matern-5/2 GP on the residual (true - V5_pred) using physical
descriptors that V5 doesn't see directly. Adds the GP's predicted residual
back to V5's prediction → corrected output.

Features per labeled sample:
  - V5 Platt-calibrated prediction (1-d)
  - lone_pair_bool (0 / 1) — Bi/Sb=1
  - χ_mismatch = χ_dopant - 1.81 (Ga)
  - r_mismatch = (r_dopant - 0.62) / 0.62
  - oxidation_state_offset (0 for isovalent, +1 donor, -1 acceptor)
  - one-hot for method (sputter is the only one in scope, so this is 1)

Cross-validation: GroupKFold by DOI to avoid leakage. Output:
  results/<bundle>/oof_predictions_v52g.csv  (V5 + GP corrected)

CPU only — runs in parallel with GPU training.

Usage:
    conda activate ga2o3
    python scripts/run_gp_residual.py [--bundle results/phase43v5_distill_vc_5seed]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel,
)
from sklearn.model_selection import GroupKFold

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.eval_physics_diag_table import load_oof_with_meta, fit_sputter_platt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Physical descriptors per dopant element (V52 corrected with Bi/Sb isovalent)
# Format: (lone_pair_bool, χ_dopant, r_dopant_Å, ox_state)
_ELEMENT_FEATURES = {
    # acceptors (ox=2)
    "Mg":      (0.0, 1.31, 0.720, 2),
    "Zn":      (0.0, 1.65, 0.740, 2),
    "Cu":      (0.0, 1.90, 0.730, 2),
    # isovalent (ox=3)
    "Al":      (0.0, 1.61, 0.535, 3),
    "Fe":      (0.0, 1.83, 0.645, 3),
    "B":       (0.0, 2.04, 0.270, 3),
    "V":       (0.0, 1.63, 0.540, 3),
    "Er":      (0.0, 1.24, 0.890, 3),
    "Eu":      (0.0, 1.20, 0.947, 3),
    # isovalent with lone pair
    "Sb":      (1.0, 2.05, 0.760, 3),  # Sb³⁺, 5s² lone pair
    "Bi":      (1.0, 2.02, 1.030, 3),  # Bi³⁺, 6s² lone pair
    # donors (ox=4)
    "Si":      (0.0, 1.90, 0.400, 4),
    "Sn":      (0.0, 1.96, 0.690, 4),
    "Ti":      (0.0, 1.54, 0.605, 4),
    "Ge":      (0.0, 2.01, 0.530, 4),
    # super-donors (ox=5/6, no lone pair)
    "Ta":      (0.0, 1.50, 0.640, 5),
    "W":       (0.0, 2.36, 0.600, 6),
    # anion / interstitial
    "F":       (0.0, 3.98, 1.330, -1),
    "undoped": (0.0, 1.81, 0.620, 3),
}
GA_CHI = 1.81
GA_RADIUS = 0.620


def _build_features(elem: str, v5_pred_platt: float) -> np.ndarray:
    """Compute physical-descriptor feature vector for a labeled sample."""
    feats = _ELEMENT_FEATURES.get(elem, _ELEMENT_FEATURES["undoped"])
    lone, chi, r, ox = feats
    chi_mismatch = chi - GA_CHI
    r_mismatch = (r - GA_RADIUS) / GA_RADIUS
    ox_offset = ox - 3
    return np.array([
        v5_pred_platt,    # 0: V5 prediction (anchors GP to V5's level)
        lone,             # 1: lone pair indicator
        chi_mismatch,     # 2: electronegativity mismatch
        r_mismatch,       # 3: radius mismatch (fractional)
        ox_offset,        # 4: oxidation state offset
    ], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle",
                   default="results/phase43v5_distill_vc_5seed")
    p.add_argument("--full-csv",
                   default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv")
    p.add_argument("--target", default="vacancy_concentration")
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--out-suffix", default="v52g")
    args = p.parse_args()

    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = PROJ / bundle
    csv_path = (PROJ / args.full_csv) if not Path(args.full_csv).is_absolute() else Path(args.full_csv)

    target_col = args.target
    pred_col = f"{target_col}_pred"
    pred_platt_col = f"{target_col}_pred_platt"
    true_col = f"{target_col}_true"

    # Load V5 OOF + source meta
    df_full = load_oof_with_meta(bundle, str(csv_path), target=target_col)
    slope, intercept = fit_sputter_platt(df_full, target=target_col)
    df_full[pred_platt_col] = df_full[pred_col] * slope + intercept
    logger.info(f"V5 Platt: slope={slope:+.3f}, intercept={intercept:+.3f}")

    # Honor source-CSV NaN labels (post-data-fix authority)
    src_df = pd.read_csv(csv_path)
    if "usable_flag" in src_df.columns:
        src_df = src_df[src_df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    src_df = src_df[~src_df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    src_df["sample_idx"] = np.arange(len(src_df), dtype=int)
    true_map = dict(zip(src_df["sample_idx"], src_df[target_col]))
    n_masked = 0
    for i, sidx in enumerate(df_full["sample_idx"].astype(int)):
        src_val = true_map.get(int(sidx))
        if src_val is None or pd.isna(src_val):
            if pd.notna(df_full.iloc[i][true_col]):
                df_full.at[df_full.index[i], true_col] = np.nan
                n_masked += 1
    if n_masked > 0:
        logger.info(f"Masked {n_masked} OOF labels per source CSV (data fixes)")

    # Restrict to sputter labeled sputter rows
    sub = df_full[df_full.is_sputter & df_full[true_col].notna()].copy()
    sub = sub[sub.dopant_label.isin(_ELEMENT_FEATURES.keys())].copy()
    logger.info(f"Sputter labeled rows for GP residual: {len(sub)}")

    # Build feature matrix + targets
    X = np.stack([
        _build_features(elem, v5)
        for elem, v5 in zip(sub.dopant_label, sub[pred_platt_col])
    ])
    y_true = sub[true_col].to_numpy(dtype=np.float64)
    y_v5 = sub[pred_platt_col].to_numpy(dtype=np.float64)
    residual = y_true - y_v5
    groups = sub["doi"].fillna("__NA__").to_numpy()
    logger.info(f"Feature matrix: {X.shape}, residual mean={residual.mean():+.3f} "
                f"std={residual.std():+.3f}")

    # GroupKFold by DOI for leakage-free GP residual evaluation
    n_unique_groups = len(set(groups))
    n_folds_use = min(args.n_folds, n_unique_groups)
    if n_folds_use < 2:
        raise RuntimeError(f"Insufficient DOI groups for {n_folds_use}-fold CV: {n_unique_groups}")
    gkf = GroupKFold(n_splits=n_folds_use)

    # Per-fold OOF GP residual prediction
    gp_resid_oof = np.full(len(sub), np.nan, dtype=np.float64)
    gp_std_oof = np.full(len(sub), np.nan, dtype=np.float64)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, residual, groups)):
        if len(train_idx) < 5 or len(val_idx) < 1:
            continue
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.01, 10.0))
            * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
            + WhiteKernel(0.1, noise_level_bounds=(1e-3, 1.0))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-3, normalize_y=True,
            n_restarts_optimizer=2, random_state=42,
        )
        gp.fit(X[train_idx], residual[train_idx])
        pred_resid, pred_std = gp.predict(X[val_idx], return_std=True)
        gp_resid_oof[val_idx] = pred_resid
        gp_std_oof[val_idx] = pred_std
        logger.debug(f"Fold {fold_idx+1}/{n_folds_use}: train n={len(train_idx)}, "
                     f"val n={len(val_idx)}, GP kernel={gp.kernel_}")

    # Final corrected predictions
    valid_mask = np.isfinite(gp_resid_oof)
    n_valid = valid_mask.sum()
    if n_valid < 5:
        raise RuntimeError(f"GP OOF only {n_valid} valid predictions")
    sub_valid = sub.iloc[valid_mask].copy()
    y_v52g = y_v5[valid_mask] + gp_resid_oof[valid_mask]
    y_true_valid = y_true[valid_mask]

    # Metrics
    from scipy import stats as _stats
    pearson_r = _stats.pearsonr(y_true_valid, y_v52g).statistic
    pearson_v5 = _stats.pearsonr(y_true_valid, y_v5[valid_mask]).statistic
    mae_v52g = float(np.mean(np.abs(y_true_valid - y_v52g)))
    mae_v5 = float(np.mean(np.abs(y_true_valid - y_v5[valid_mask])))
    ss_res_v52g = float(np.sum((y_true_valid - y_v52g) ** 2))
    ss_res_v5 = float(np.sum((y_true_valid - y_v5[valid_mask]) ** 2))
    ss_tot = float(np.sum((y_true_valid - y_true_valid.mean()) ** 2))
    r2_v52g = 1.0 - ss_res_v52g / ss_tot if ss_tot > 0 else float("nan")
    r2_v5 = 1.0 - ss_res_v5 / ss_tot if ss_tot > 0 else float("nan")

    print(f"\n{'='*60}\nV52g GP Residual on {bundle.name}\n{'='*60}")
    print(f"Sputter labeled rows used: {n_valid}/{len(sub)}")
    print(f"\n         V5 alone    V5+GP (V52g)    Δ")
    print(f"  R²:    {r2_v5:+.3f}     {r2_v52g:+.3f}        {r2_v52g-r2_v5:+.3f}")
    print(f"  r:     {pearson_v5:+.3f}     {pearson_r:+.3f}        {pearson_r-pearson_v5:+.3f}")
    print(f"  MAE:   {mae_v5:.3f}      {mae_v52g:.3f}         {mae_v52g-mae_v5:+.3f}")

    # Per-element comparison
    sub_valid_with_pred = sub_valid.copy()
    sub_valid_with_pred[f"{target_col}_pred_v5"] = y_v5[valid_mask]
    sub_valid_with_pred[f"{target_col}_pred_v52g"] = y_v52g
    sub_valid_with_pred[f"{target_col}_gp_resid"] = gp_resid_oof[valid_mask]
    sub_valid_with_pred[f"{target_col}_gp_std"] = gp_std_oof[valid_mask]

    print(f"\nPer-element parity Pearson r (V5 → V52g):")
    for elem in sorted(sub_valid_with_pred.dopant_label.unique()):
        ee = sub_valid_with_pred[sub_valid_with_pred.dopant_label == elem]
        if len(ee) < 3:
            continue
        r_v5 = float(_stats.pearsonr(ee[true_col], ee[f"{target_col}_pred_v5"]).statistic)
        r_v52g = float(_stats.pearsonr(ee[true_col], ee[f"{target_col}_pred_v52g"]).statistic)
        print(f"  {elem:5s} (n={len(ee):2d}):  V5 r={r_v5:+.3f}   V52g r={r_v52g:+.3f}   "
              f"Δ={r_v52g-r_v5:+.3f}")

    # Save corrected OOF
    out_df = df_full[["sample_idx", "dopant_label", pred_col,
                       pred_platt_col, "vacancy_concentration_std",
                       true_col]].copy()
    out_df[f"{target_col}_pred_v5"] = out_df[pred_platt_col]
    out_df[f"{target_col}_pred_v52g"] = out_df[pred_platt_col]   # default = V5
    out_df[f"{target_col}_gp_resid"] = 0.0
    out_df[f"{target_col}_gp_std"] = 0.0
    # Insert GP corrections at matching sample_idx
    for i, sidx in enumerate(sub.sample_idx[valid_mask]):
        ridx = out_df.index[out_df.sample_idx == sidx]
        if len(ridx) > 0:
            out_df.loc[ridx[0], f"{target_col}_pred_v52g"] = y_v52g[i]
            out_df.loc[ridx[0], f"{target_col}_gp_resid"] = gp_resid_oof[valid_mask][i]
            out_df.loc[ridx[0], f"{target_col}_gp_std"] = gp_std_oof[valid_mask][i]
    # For evaluation compatibility, also write V52g as the primary "pred"
    out_df[f"{target_col}_pred"] = out_df[f"{target_col}_pred_v52g"]
    out_df[f"{target_col}_pred_platt"] = out_df[f"{target_col}_pred_v52g"]

    out_path = bundle / f"oof_predictions_{args.out_suffix}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
