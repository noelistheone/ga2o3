"""Phase 42 V4 / Phase 43 V5 evaluation vs Phase 33 + Phase 41 V2 baselines.

Reports sputter-only OOF metrics (apples-to-apples comparison) with
re-fitted Platt calibration on the sputter subset for each model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]


def metrics(t, p):
    if len(t) < 2 or np.std(t) < 1e-9:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = float(stats.pearsonr(t, p).statistic)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    return r, r2, mae, rmse


def evaluate_bundle(bundle_dir: Path, label: str, full_csv: str | None) -> dict:
    """Return sputter-only metrics with refitted Platt."""
    oof_path = bundle_dir / "oof_predictions.csv"
    if not oof_path.exists():
        return {"label": label, "error": f"no OOF at {oof_path}"}
    oof = pd.read_csv(oof_path)

    if full_csv is None:
        # No method column needed — assume all rows are sputter (Phase 33 case)
        sub = oof.dropna(subset=["vacancy_concentration_pred",
                                 "vacancy_concentration_true"]).copy()
        is_sputter_count = len(sub)
    else:
        src = pd.read_csv(full_csv)
        if "usable_flag" in src.columns:
            src = src[src["usable_flag"] != "exotic_skip"].reset_index(drop=True)
        src = src[~src["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
        src["sample_idx"] = np.arange(len(src), dtype=int)
        merged = oof.merge(src[["sample_idx", "method"]], on="sample_idx", how="left")
        merged["is_sputter"] = (
            merged["method"].fillna("").str.lower().str.contains("sputter")
        )
        valid = merged.dropna(subset=["vacancy_concentration_pred",
                                       "vacancy_concentration_true"])
        sub = valid[valid.is_sputter].copy()
        is_sputter_count = len(sub)

    if len(sub) < 2:
        return {"label": label, "error": f"too few sputter rows: {len(sub)}"}

    y_true = sub["vacancy_concentration_true"].to_numpy()
    y_raw  = sub["vacancy_concentration_pred"].to_numpy()

    # Re-fit Platt on this sputter-only subset (apples-to-apples)
    slope, intercept = np.polyfit(y_raw, y_true, 1)
    y_platt = y_raw * slope + intercept

    r_raw, r2_raw, mae_raw, rmse_raw = metrics(y_true, y_raw)
    r_pl,  r2_pl,  mae_pl,  rmse_pl  = metrics(y_true, y_platt)

    # Per-element on Platt
    per_elem = {}
    for elem in sorted(sub.dopant_label.unique(),
                       key=lambda e: -len(sub[sub.dopant_label == e])):
        ee = sub[sub.dopant_label == elem]
        if len(ee) < 2: continue
        y_t = ee["vacancy_concentration_true"].to_numpy()
        y_p = ee["vacancy_concentration_pred"].to_numpy() * slope + intercept
        r, r2, mae, _ = metrics(y_t, y_p)
        per_elem[elem] = dict(n=len(ee), r=r, mae=mae)

    return dict(
        label=label,
        n_sputter=is_sputter_count,
        platt=dict(slope=slope, intercept=intercept),
        raw=dict(r=r_raw, r2=r2_raw, mae=mae_raw, rmse=rmse_raw),
        platt_metrics=dict(r=r_pl, r2=r2_pl, mae=mae_pl, rmse=rmse_pl),
        per_element=per_elem,
    )


def main():
    bundles = [
        ("Phase 33 frontier",
         PROJ / "results" / "deployment" / "vc_alt_33_sputter_brouwer_atmo_class",
         None),  # sputter-only training, no method merge needed
        ("Phase 41 V2 (Brouwer + α_method)",
         PROJ / "results" / "phase41v2_brouwer_method_offset_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 42 V4 (V2 + cross-method PULL)",
         PROJ / "results" / "phase42v4_pullaux_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 43 V5 (V2 + encoder distill aux)",
         PROJ / "results" / "phase43v5_distill_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 44 V6 (V5 + DopantStream + InfoNCE + Huber + family_cos)",
         PROJ / "results" / "phase44v6_lowcost_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 44 V8 (V6 + multi-task VC+PDR + SRH coupling)",
         PROJ / "results" / "phase44v8_multitask_vcpdr_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 45 V10 (V5 + class_offset)",
         PROJ / "results" / "phase45v10_class_offset_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 46 V11 (V5 + HN-GNN, replaces class_offset)",
         PROJ / "results" / "phase46v11_hypernet_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 46 V12 (V11 + composition capacity)",
         PROJ / "results" / "phase46v12_capacity_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 47 V13 (HN small-random init, WD=0)",
         PROJ / "results" / "phase47v13_hn_no_wd_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 47 V14 (V13 + multiplicative HN on dopant_term)",
         PROJ / "results" / "phase47v14_mult_hn_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 47 V15 (direct descriptor injection, no HN)",
         PROJ / "results" / "phase47v15_inject_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 47 V16 (proxy V_O f.e. anchor)",
         PROJ / "results" / "phase47v16_proxy_anchor_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 49 V17 (CHGNet frozen backbone)",
         PROJ / "results" / "phase49v17_chgnet_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 50 V18a (V5 + ordered structures only)",
         PROJ / "results" / "phase50v18a_ordered_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 50 V18b (V5 + multi-DB pretrain only)",
         PROJ / "results" / "phase50v18b_newpretrain_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 50 V18 (ordered + multi-DB pretrain)",
         PROJ / "results" / "phase50v18_combined_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 51 V51 (Frozen Multi-Expert + Self-Attention)",
         PROJ / "results" / "phase51v51_multiexpert_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
        ("Phase 52 V52a (V5 + Bi/Sb relabeled isovalent)",
         PROJ / "results" / "phase52v52a_relabel_vc_5seed",
         str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")),
    ]

    results = []
    for label, bd, full_csv in bundles:
        if not bd.exists():
            print(f"[SKIP] {label}: bundle missing at {bd}")
            continue
        try:
            r = evaluate_bundle(bd, label, full_csv)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {label}: {e}")

    print("=" * 80)
    print(f"{'Label':<45s} {'N_sp':>4s} {'R²_Pl':>8s} {'r':>7s} {'MAE':>6s} {'RMSE':>6s}")
    print("=" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<45s}  ERROR: {r['error']}")
            continue
        pm = r["platt_metrics"]
        print(f"{r['label']:<45s} {r['n_sputter']:>4d} "
              f"{pm['r2']:>+8.3f} {pm['r']:>+7.3f} "
              f"{pm['mae']:>6.3f} {pm['rmse']:>6.3f}")

    print()
    print("=" * 80)
    print("Per-element (Pearson r on Platt-calibrated sputter-subset):")
    print("=" * 80)
    elements = sorted({e for r in results if "error" not in r for e in r["per_element"]},
                      key=str)
    print(f"{'Element':<10s} " + "  ".join([f"{r['label'][:18]:<18s}" for r in results
                                              if "error" not in r]))
    for elem in elements:
        line = f"{elem:<10s}"
        for r in results:
            if "error" in r: continue
            pe = r["per_element"].get(elem)
            if pe is None:
                line += f"  {'(N/A)':<18s}"
            else:
                line += f"  N={pe['n']:>2d} r={pe['r']:+.3f} ".ljust(20)
        print(line)


if __name__ == "__main__":
    main()
