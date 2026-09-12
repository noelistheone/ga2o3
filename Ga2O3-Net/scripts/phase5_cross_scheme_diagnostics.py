"""Cross-scheme diagnostic table builder for Phase 5C/5D/5E verdict.

For each OOF file:
  - Merge with source CSV on sample_idx → attach DOI, element, concentration_at%.
  - Compute global R² / r / MAE (Platt-calibrated OOF, nested LOO-Platt, raw).
  - Compute Mg-subset R² and Pearson r (raw predictions).
  - Compute within-DOI Pearson ρ(conc, pred) on Mg rows (multi-concentration DOIs).
  - Compute var(pred|doi)/var(true|doi), averaged across DOIs with ≥2 labeled rows.

Emit a single CSV at results/phase5_comparison/diagnostics.csv with one row per scheme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCHEMES = [
    ("4F_baseline", "results/phase4f_pdr_aug3x_5seed/oof_predictions.csv",
     "data/raw/experimental/ga2o3_exp_augmented_3x.csv",
     "results/phase4f_pdr_aug3x_5seed/external/external_predictions.csv"),
    ("5D_gbv2", "results/phase5d_pdr_gbv2_5seed/oof_predictions.csv",
     "data/raw/experimental/ga2o3_exp_augmented_3x.csv",
     "results/phase5d_pdr_gbv2_5seed/external/external_predictions.csv"),
    ("5E_interp", "results/phase5e_pdr_interp_5seed/oof_predictions.csv",
     "data/raw/experimental/ga2o3_exp_aug3x_interp.csv",
     "results/phase5e_pdr_interp_5seed/external/external_predictions.csv"),
    ("5F_gbv2_interp", "results/phase5f_pdr_gbv2_interp_5seed/oof_predictions.csv",
     "data/raw/experimental/ga2o3_exp_aug3x_interp.csv",
     "results/phase5f_pdr_gbv2_interp_5seed/external/external_predictions.csv"),
    ("5C_conc18_clean", "results/phase5c_pdr_conc18_5seed_clean/oof_predictions.csv",
     "data/raw/experimental/ga2o3_exp_augmented_3x.csv",
     "results/phase5c_pdr_conc18_5seed_clean/external/external_predictions.csv"),
    ("5G_dopant_v1", "results/phase5g_pdr_dopant_5seed/oof_predictions.csv",
     "data/raw/experimental/ga2o3_exp_augmented_3x.csv",
     "results/phase5g_pdr_dopant_5seed/external/external_predictions.csv"),
    ("5G_dopant_combo", "results/phase5g_pdr_dopant_interp_v2_5seed/oof_predictions.csv",
     "data/raw/experimental/ga2o3_exp_aug3x_interp.csv",
     "results/phase5g_pdr_dopant_interp_v2_5seed/external/external_predictions.csv"),
]


def _loo_platt(y_pred, y_true):
    n = len(y_pred)
    out = np.empty(n)
    X = y_pred.reshape(-1, 1)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        lr = LinearRegression().fit(X[m], y_true[m])
        out[i] = float(lr.predict(X[i:i+1])[0])
    return out


def _external_r2(path):
    if not Path(path).exists():
        return (float("nan"), float("nan"), 0)
    ext = pd.read_csv(path)
    for pred_col, true_col in [
        ("photo_dark_ratio_pred", "photo_dark_ratio_true"),
        ("pred", "true"),
    ]:
        if pred_col in ext.columns and true_col in ext.columns:
            break
    else:
        return (float("nan"), float("nan"), 0)
    m = ext[pred_col].notna() & ext[true_col].notna()
    if m.sum() < 3:
        return (float("nan"), float("nan"), int(m.sum()))
    yp = ext.loc[m, pred_col].to_numpy()
    yt = ext.loc[m, true_col].to_numpy()
    return (float(r2_score(yt, yp)), float(pearsonr(yp, yt)[0]), int(m.sum()))


def compute_row(name, oof_path, csv_path, ext_path):
    oof = pd.read_csv(oof_path)
    src = pd.read_csv(csv_path).reset_index(drop=True)
    src["sample_idx"] = src.index

    df = oof.merge(src[["sample_idx", "element", "concentration_at%", "doi"]],
                   on="sample_idx", how="left")

    m = df["photo_dark_ratio_pred"].notna() & df["photo_dark_ratio_true"].notna()
    d = df.loc[m].copy()
    yp_raw = d["photo_dark_ratio_pred"].to_numpy()
    yt = d["photo_dark_ratio_true"].to_numpy()

    # raw + Platt + nested LOO-Platt
    yp_platt = _loo_platt(yp_raw, yt)

    r2_raw = r2_score(yt, yp_raw)
    r_raw = pearsonr(yp_raw, yt)[0]
    mae_raw = mean_absolute_error(yt, yp_raw)
    r2_nest = r2_score(yt, yp_platt)
    r_nest = pearsonr(yp_platt, yt)[0]

    # Mg-subset
    mg = d[d["element"] == "Mg"].copy()
    if len(mg) >= 3:
        mg_yp = mg["photo_dark_ratio_pred"].to_numpy()
        mg_yt = mg["photo_dark_ratio_true"].to_numpy()
        mg_r2 = r2_score(mg_yt, mg_yp)
        mg_r = pearsonr(mg_yp, mg_yt)[0]
    else:
        mg_r2 = mg_r = float("nan")

    # within-DOI ρ(conc, pred) on Mg rows — only DOIs with ≥2 Mg rows of differing conc
    rhos = []
    for doi, sub in mg.groupby("doi"):
        if len(sub) < 2 or sub["concentration_at%"].nunique() < 2:
            continue
        c = sub["concentration_at%"].to_numpy()
        p = sub["photo_dark_ratio_pred"].to_numpy()
        if np.std(c) < 1e-9 or np.std(p) < 1e-9:
            continue
        rhos.append(pearsonr(c, p)[0])
    within_rho_mg = float(np.mean(rhos)) if rhos else float("nan")
    n_doi_mg = len(rhos)

    # var(pred|doi) / var(true|doi), averaged over DOIs with ≥2 labeled rows
    ratios = []
    for doi, sub in d.groupby("doi"):
        if len(sub) < 2:
            continue
        vt = float(np.var(sub["photo_dark_ratio_true"]))
        vp = float(np.var(sub["photo_dark_ratio_pred"]))
        if vt < 1e-9:
            continue
        ratios.append(vp / vt)
    var_ratio = float(np.mean(ratios)) if ratios else float("nan")
    n_doi_all = len(ratios)

    ext_r2, ext_r, ext_n = _external_r2(ext_path)

    return {
        "scheme": name,
        "N_oof": int(m.sum()),
        "R2_raw": round(r2_raw, 4),
        "R2_nested": round(r2_nest, 4),
        "r_raw": round(r_raw, 4),
        "r_nested": round(r_nest, 4),
        "MAE_raw": round(mae_raw, 4),
        "Mg_N": int(len(mg)),
        "Mg_R2": round(mg_r2, 4) if not np.isnan(mg_r2) else float("nan"),
        "Mg_r": round(mg_r, 4) if not np.isnan(mg_r) else float("nan"),
        "within_DOI_rho_Mg": round(within_rho_mg, 4) if not np.isnan(within_rho_mg) else float("nan"),
        "n_DOI_Mg_multiconc": n_doi_mg,
        "var_pred_over_var_true_per_DOI": round(var_ratio, 4) if not np.isnan(var_ratio) else float("nan"),
        "n_DOI_all": n_doi_all,
        "ext_R2_raw": round(ext_r2, 4) if not np.isnan(ext_r2) else float("nan"),
        "ext_r_raw": round(ext_r, 4) if not np.isnan(ext_r) else float("nan"),
        "ext_N": ext_n,
    }


def main():
    rows = []
    for name, oof, csv, ext in SCHEMES:
        if not Path(oof).exists():
            print(f"[skip] {name}: no {oof}")
            continue
        rows.append(compute_row(name, oof, csv, ext))
    out = pd.DataFrame(rows)
    Path("results/phase5_comparison").mkdir(parents=True, exist_ok=True)
    out_path = "results/phase5_comparison/diagnostics.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
