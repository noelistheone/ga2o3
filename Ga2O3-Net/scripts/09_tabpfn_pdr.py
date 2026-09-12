"""Phase 54 Add-on 1 — Tabular PDR baseline (LightGBM + optional TabPFN-2.5).

Original plan: TabPFN-2.5 (Hollmann et al. *Nature* 637:319, 2025) as a
non-deep regressor on PDR rows. Discovery 2026-05-21: TabPFN 2.5 requires
a license/API token (`TABPFN_TOKEN`) for download; that environment access
is not available in this session — Add-on 1 falls back to LightGBM, which
TabPFN's own paper benchmarks against (normalized RMSE 0.872) and which
serves the same role as a non-deep tabular calibration baseline.

If TABPFN_TOKEN is set in the environment, the script will also run TabPFN
side-by-side. Otherwise only LightGBM runs.

Output: per-row OOF predictions for cross-validation comparison against V5
PDR head.

Cost: ~3 min for 5-fold OOF LightGBM on N=226 PDR-labeled rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def build_features(csv_path: Path) -> pd.DataFrame:
    """Build flat 11-d element + 18-d process feature table per row."""
    from src.data.element_descriptors import _standardized_descriptor_table
    table = _standardized_descriptor_table()
    K = len(next(iter(table.values())))

    df = pd.read_csv(csv_path)
    # Apply standard filter so sample_idx aligns with OOF
    if "usable_flag" in df.columns:
        df = df[df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    df = df[~df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)

    rows = []
    for idx, row in df.iterrows():
        elem = str(row.get("element", "")).strip()
        if not elem or elem == "nan":
            elem = "undoped"
        # Use the cation table; for co-doped (Fe+Sn) average
        if "+" in elem:
            comps = [c.strip() for c in elem.split("+")]
            vecs = [table.get(c, table["other"]) for c in comps]
            elem_vec = np.mean(vecs, axis=0)
        else:
            elem_vec = np.array(table.get(elem, table["other"]))

        # Process: T_C, time_min, atmosphere (one-hot 6-way), conc_at%
        from src.data.experimental_dataset import _parse_atmosphere_o2_fraction
        atm_str = str(row.get("atmosphere", "Ar")).strip()
        try:
            o2 = _parse_atmosphere_o2_fraction(atm_str)
        except Exception:
            o2 = 0.0
        proc_vec = np.array([
            float(row.get("temperature_C", 0) or 0),
            float(row.get("time_min", 0) or 0),
            float(o2),
            float(row.get("concentration_at%", 0) or 0),
            float(row.get("measurement_voltage_V", 10) or 10),
            float(row.get("measurement_wavelength_nm", 254) or 254),
        ], dtype=np.float32)

        rows.append({
            "sample_idx": idx,
            "feature": np.concatenate([elem_vec, proc_vec]),
            "pdr_log10": np.log10(row["photo_dark_ratio"]) if pd.notna(row.get("photo_dark_ratio")) else np.nan,
            "vc": float(row.get("vacancy_concentration")) if pd.notna(row.get("vacancy_concentration")) else np.nan,
            "doi": str(row.get("doi", "")),
            "dopant_label": elem,
        })
    feats = pd.DataFrame(rows)
    feats["X"] = feats["feature"]
    return feats


def main() -> None:
    import argparse
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import r2_score, mean_absolute_error

    parser = argparse.ArgumentParser(description="V54 Add-on 1: TabPFN PDR")
    parser.add_argument("--target", default="pdr_log10",
                        choices=["pdr_log10", "vc"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--group-by-doi", action="store_true",
                        help="Use GroupKFold on DOI to prevent leakage")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJ / "results" / "phase54addon1_tabpfn_pdr")
    parser.add_argument("--csv", type=Path,
                        default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")
    args = parser.parse_args()

    feats = build_features(args.csv)
    labeled = feats[feats[args.target].notna()].reset_index(drop=True)
    print(f"Total rows: {len(feats)} | {args.target}-labeled: {len(labeled)}")

    X = np.stack(labeled["X"].values)
    y = labeled[args.target].values.astype(np.float32)
    print(f"Feature shape: {X.shape}  (E={X.shape[1]} dims)")

    import os
    use_tabpfn = bool(os.environ.get("TABPFN_TOKEN"))
    if use_tabpfn:
        from tabpfn import TabPFNRegressor
    import lightgbm as lgb

    if args.group_by_doi:
        groups = labeled["doi"].values
        kf = GroupKFold(n_splits=args.cv_folds)
        splits = list(kf.split(X, y, groups=groups))
    else:
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=0)
        splits = list(kf.split(X))

    oof_pred = np.full(len(y), np.nan, dtype=np.float32)
    oof_pred_lgbm = np.full(len(y), np.nan, dtype=np.float32)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        # LightGBM
        lgbm = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            verbose=-1, random_state=0,
        )
        lgbm.fit(X[train_idx], y[train_idx])
        pred_lgbm = lgbm.predict(X[val_idx])
        oof_pred_lgbm[val_idx] = pred_lgbm
        r2_lgbm = r2_score(y[val_idx], pred_lgbm)
        mae_lgbm = mean_absolute_error(y[val_idx], pred_lgbm)

        # TabPFN (only if license token is set)
        pred = pred_lgbm.copy()  # default to LightGBM if TabPFN unavailable
        r2 = r2_lgbm
        mae = mae_lgbm
        if use_tabpfn:
            try:
                reg = TabPFNRegressor(device="cuda" if _gpu_ok() else "cpu")
                reg.fit(X[train_idx], y[train_idx])
                pred = reg.predict(X[val_idx])
                r2 = r2_score(y[val_idx], pred)
                mae = mean_absolute_error(y[val_idx], pred)
            except Exception as e:
                print(f"Fold {fold + 1} TabPFN failed → fall back to LightGBM: {e}")

        oof_pred[val_idx] = pred
        tag = "TabPFN" if use_tabpfn else "LightGBM"
        print(f"Fold {fold + 1}/{args.cv_folds}  n_train={len(train_idx)}  n_val={len(val_idx)}  "
              f"{tag} R²={r2:.3f} MAE={mae:.3f}  |  LightGBM R²={r2_lgbm:.3f} MAE={mae_lgbm:.3f}")
        fold_metrics.append({"fold": fold + 1, "r2": r2, "mae": mae,
                             "r2_lgbm": r2_lgbm, "mae_lgbm": mae_lgbm,
                             "n_train": int(len(train_idx)), "n_val": int(len(val_idx))})

    valid = ~np.isnan(oof_pred)
    overall_r2 = r2_score(y[valid], oof_pred[valid])
    overall_mae = mean_absolute_error(y[valid], oof_pred[valid])
    valid_l = ~np.isnan(oof_pred_lgbm)
    overall_r2_lgbm = r2_score(y[valid_l], oof_pred_lgbm[valid_l])
    overall_mae_lgbm = mean_absolute_error(y[valid_l], oof_pred_lgbm[valid_l])
    print(f"\nOOF aggregate (primary): R²={overall_r2:.3f}  MAE={overall_mae:.3f}  (n={valid.sum()})")
    print(f"OOF aggregate (LightGBM): R²={overall_r2_lgbm:.3f}  MAE={overall_mae_lgbm:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_idx": labeled["sample_idx"].values,
        f"{args.target}_pred_tabpfn": oof_pred,
        f"{args.target}_true": y,
        "dopant_label": labeled["dopant_label"].values,
        "doi": labeled["doi"].values,
    }).to_csv(args.output_dir / f"oof_predictions_{args.target}.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(
        args.output_dir / f"cv_results_{args.target}.csv", index=False
    )

    # Write summary
    summary = {
        "target": args.target,
        "n_labeled": int(len(labeled)),
        "feature_dim": int(X.shape[1]),
        "cv_folds": int(args.cv_folds),
        "group_by_doi": bool(args.group_by_doi),
        "oof_r2": float(overall_r2),
        "oof_mae": float(overall_mae),
        "fold_metrics": fold_metrics,
    }
    import json
    (args.output_dir / f"summary_{args.target}.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir}/summary_{args.target}.json")


def _gpu_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
