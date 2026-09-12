"""
Phase 6B: per-dopant ensemble across the 6 phase-5 schemes.

Learns weights on per-dopant-Platt-calibrated OOF predictions (leakage-free
because OOF is by definition never seen during training of each scheme), then
applies to the external set.

Three strategies compared:
  (A) Global NNLS  — one non-negative weight per scheme, shared across dopants.
  (B) Per-dopant NNLS — separate weights per dopant where N_OOF ≥ 10, fall back
      to global weights elsewhere.
  (C) Best-single-per-dopant — pick the scheme with best OOF MAE per dopant.

Outputs ``results/phase6_comparison/per_dopant_ensemble.csv`` and prints a
comparison table vs each individual scheme.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


SCHEMES = [
    ("5C", "phase5c_pdr_conc18_5seed_clean"),
    ("5D", "phase5d_pdr_gbv2_5seed"),
    ("5E", "phase5e_pdr_interp_5seed"),
    ("5F", "phase5f_pdr_gbv2_interp_5seed"),
    ("5G-v1",    "phase5g_pdr_dopant_5seed"),
    ("5G-combo", "phase5g_pdr_dopant_interp_v2_5seed"),
]

PRED_COL = "photo_dark_ratio_pred_per_dopant_platt"
TRUE_COL = "photo_dark_ratio_true"
RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "phase6_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(kind: str) -> pd.DataFrame:
    """kind ∈ {'oof', 'external'} → merged DataFrame with per-scheme pred columns."""
    merged = None
    for code, name in SCHEMES:
        if kind == "oof":
            path = RESULTS_DIR / name / "oof_per_dopant_platt.csv"
            join_key = "sample_idx"
        else:
            path = RESULTS_DIR / name / "external" / "external_per_dopant_platt.csv"
            join_key = "sample_idx"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path)
        keep = df[[join_key, TRUE_COL, "dopant_label", PRED_COL]].copy()
        keep = keep.rename(columns={PRED_COL: f"pred_{code}"})
        if merged is None:
            merged = keep
        else:
            merged = merged.merge(
                keep.drop(columns=[TRUE_COL, "dopant_label"]),
                on=join_key, how="outer", validate="one_to_one",
            )
    merged = merged.dropna(subset=[TRUE_COL] + [f"pred_{c}" for c, _ in SCHEMES])
    return merged


def nnls_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Non-negative least squares with tiny ridge for stability. Returns
    normalised weights that sum to 1."""
    if len(y) < 2:
        # degenerate: uniform
        w = np.ones(X.shape[1])
        return w / w.sum()
    Xa = np.vstack([X, ridge * np.eye(X.shape[1])])
    ya = np.concatenate([y, np.zeros(X.shape[1])])
    w, _ = nnls(Xa, ya)
    s = w.sum()
    return w / s if s > 1e-9 else np.ones_like(w) / len(w)


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")


def evaluate(name: str, y_true, y_pred) -> dict:
    from scipy.stats import pearsonr
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mae = np.abs(y_true - y_pred).mean()
    r = pearsonr(y_true, y_pred)[0] if len(y_true) >= 2 else float("nan")
    return {"name": name, "N": len(y_true), "R2": r2(y_true, y_pred), "MAE": mae, "r": r}


def run():
    oof = load_predictions("oof")
    ext = load_predictions("external")
    print(f"Loaded OOF={len(oof)} rows, external={len(ext)} rows (both per_dopant_platt).")

    scheme_codes = [c for c, _ in SCHEMES]
    pred_cols = [f"pred_{c}" for c in scheme_codes]

    # ── Baselines: each individual scheme ──
    rows_ind = []
    for c in scheme_codes:
        rows_ind.append({
            **evaluate(f"{c} (OOF)", oof[TRUE_COL], oof[f"pred_{c}"]),
            "set": "OOF",
        })
        rows_ind.append({
            **evaluate(f"{c} (ext)", ext[TRUE_COL], ext[f"pred_{c}"]),
            "set": "ext",
        })

    # ── Strategy A: global NNLS ──
    X_oof = oof[pred_cols].values; y_oof = oof[TRUE_COL].values
    w_global = nnls_fit(X_oof, y_oof)
    oof_A = X_oof @ w_global
    ext_A = ext[pred_cols].values @ w_global
    rows_ens = []
    rows_ens.append({**evaluate("A: global NNLS (OOF)", y_oof, oof_A), "set": "OOF"})
    rows_ens.append({**evaluate("A: global NNLS (ext)", ext[TRUE_COL], ext_A), "set": "ext"})
    print("\nStrategy A — global NNLS weights:")
    for c, w in zip(scheme_codes, w_global):
        print(f"  w_{c:9s} = {w:.3f}")

    # ── Strategy B: per-dopant NNLS with global fallback ──
    oof_B = np.zeros_like(oof_A); ext_B = np.zeros_like(ext_A)
    per_dopant_weights = {}
    fallback_dopants = []
    min_n = 10
    for d in sorted(oof["dopant_label"].unique()):
        mask_oof = (oof["dopant_label"] == d).values
        mask_ext = (ext["dopant_label"] == d).values
        if mask_oof.sum() >= min_n:
            Xd = X_oof[mask_oof]; yd = y_oof[mask_oof]
            wd = nnls_fit(Xd, yd)
            per_dopant_weights[d] = wd
        else:
            wd = w_global
            fallback_dopants.append(d)
        oof_B[mask_oof] = X_oof[mask_oof] @ wd
        if mask_ext.any():
            ext_B[mask_ext] = ext[pred_cols].values[mask_ext] @ wd
    rows_ens.append({**evaluate("B: per-dopant NNLS (OOF)", y_oof, oof_B), "set": "OOF"})
    rows_ens.append({**evaluate("B: per-dopant NNLS (ext)", ext[TRUE_COL], ext_B), "set": "ext"})
    print(f"\nStrategy B — per-dopant weights (fitted for {len(per_dopant_weights)}, "
          f"fallback to global for {len(fallback_dopants)}: {fallback_dopants}):")
    for d, wd in per_dopant_weights.items():
        wstr = " ".join(f"{w:.2f}" for w in wd)
        print(f"  w[{d:8s}] = {wstr}")

    # ── Strategy C: best-single-per-dopant ──
    oof_C = np.zeros_like(oof_A); ext_C = np.zeros_like(ext_A)
    per_dopant_best = {}
    for d in sorted(oof["dopant_label"].unique()):
        mask_oof = (oof["dopant_label"] == d).values
        mask_ext = (ext["dopant_label"] == d).values
        if mask_oof.sum() < 2:
            per_dopant_best[d] = scheme_codes[0]  # 5C fallback
        else:
            # lowest MAE on OOF wins
            yd = y_oof[mask_oof]
            maes = [np.abs(yd - X_oof[mask_oof, i]).mean() for i in range(len(scheme_codes))]
            per_dopant_best[d] = scheme_codes[int(np.argmin(maes))]
        best_i = scheme_codes.index(per_dopant_best[d])
        oof_C[mask_oof] = X_oof[mask_oof, best_i]
        if mask_ext.any():
            ext_C[mask_ext] = ext[pred_cols].values[mask_ext, best_i]
    rows_ens.append({**evaluate("C: best-single-per-dopant (OOF)", y_oof, oof_C), "set": "OOF"})
    rows_ens.append({**evaluate("C: best-single-per-dopant (ext)", ext[TRUE_COL], ext_C), "set": "ext"})
    print("\nStrategy C — best-single-per-dopant picks:")
    for d, c in per_dopant_best.items():
        print(f"  {d:8s} → {c}")

    # ── Per-dopant breakdown on external ──
    per_dopant_ext = []
    for d in sorted(ext["dopant_label"].unique()):
        m = (ext["dopant_label"] == d).values
        if m.sum() < 2:
            continue
        row = {"dopant": d, "N_ext": int(m.sum())}
        y = ext[TRUE_COL].values[m]
        for c in scheme_codes:
            row[f"{c}_r"] = evaluate("_", y, ext[f"pred_{c}"].values[m])["r"]
        row["A_r"] = evaluate("_", y, ext_A[m])["r"]
        row["B_r"] = evaluate("_", y, ext_B[m])["r"]
        row["C_r"] = evaluate("_", y, ext_C[m])["r"]
        per_dopant_ext.append(row)

    # ── Save + print ──
    df_ind = pd.DataFrame(rows_ind)
    df_ens = pd.DataFrame(rows_ens)
    df_ind.to_csv(OUT_DIR / "individual_schemes.csv", index=False)
    df_ens.to_csv(OUT_DIR / "ensemble_strategies.csv", index=False)
    pd.DataFrame(per_dopant_ext).to_csv(OUT_DIR / "per_dopant_external_r.csv", index=False)

    print("\n" + "=" * 72)
    print("INDIVIDUAL SCHEMES")
    print("=" * 72)
    print(df_ind.pivot_table(index="name", columns="set", values=["R2", "MAE", "r"]).round(3).to_string())
    print("\n" + "=" * 72)
    print("ENSEMBLE STRATEGIES")
    print("=" * 72)
    print(df_ens.pivot_table(index="name", columns="set", values=["R2", "MAE", "r"]).round(3).to_string())
    print("\n" + "=" * 72)
    print("PER-DOPANT EXTERNAL r")
    print("=" * 72)
    print(pd.DataFrame(per_dopant_ext).round(3).to_string(index=False))


if __name__ == "__main__":
    run()
