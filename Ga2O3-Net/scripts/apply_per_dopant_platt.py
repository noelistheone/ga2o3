"""Per-dopant Platt calibration — Phase 5I.

For a given scheme's OOF + external predictions, fit a LinearRegression per
dopant_label class on OOF rows (when N_train ≥ 5), else fall back to global
Platt. Apply to both OOF and external; report metrics with per-dopant Pearson r
table so we can see where the calibration flipped reversed predictions.

Usage::

    python scripts/apply_per_dopant_platt.py results/phase5X_session/

The script looks for:
    results/phase5X_session/oof_predictions.csv
    results/phase5X_session/external/external_predictions.csv
and writes:
    results/phase5X_session/oof_per_dopant_platt.csv
    results/phase5X_session/external/external_per_dopant_platt.csv
    results/phase5X_session/per_dopant_platt_metrics.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MIN_TRAIN = 5
TARGET = "photo_dark_ratio"   # overridable via --target CLI arg; see __main__


def _fit_global(y_pred, y_true):
    return LinearRegression().fit(y_pred.reshape(-1, 1), y_true)


def _fit_per_dopant(oof_df):
    m = oof_df[f"{TARGET}_pred"].notna() & oof_df[f"{TARGET}_true"].notna()
    oof = oof_df.loc[m]
    yp = oof[f"{TARGET}_pred"].to_numpy()
    yt = oof[f"{TARGET}_true"].to_numpy()

    global_model = _fit_global(yp, yt)
    per_dopant: dict[str, LinearRegression] = {}
    fit_summary: list[tuple[str, int, float, float]] = []

    for label, sub in oof.groupby("dopant_label"):
        if len(sub) < MIN_TRAIN:
            continue
        ypd = sub[f"{TARGET}_pred"].to_numpy()
        ytd = sub[f"{TARGET}_true"].to_numpy()
        if np.std(ypd) < 1e-9:
            continue
        lr = LinearRegression().fit(ypd.reshape(-1, 1), ytd)
        per_dopant[label] = lr
        fit_summary.append((label, len(sub), float(lr.coef_[0]), float(lr.intercept_)))

    return global_model, per_dopant, fit_summary


def _apply(pred_df, global_model, per_dopant):
    if pred_df.empty:
        return pred_df

    out = pred_df.copy()
    yp = out[f"{TARGET}_pred"].to_numpy()
    calibrated = np.full_like(yp, np.nan, dtype=np.float64)
    used_bucket = np.empty(len(out), dtype=object)

    for i, (pred, label) in enumerate(zip(yp, out["dopant_label"])):
        if np.isnan(pred):
            continue
        if label in per_dopant:
            calibrated[i] = float(per_dopant[label].predict([[pred]])[0])
            used_bucket[i] = f"per_dopant:{label}"
        else:
            calibrated[i] = float(global_model.predict([[pred]])[0])
            used_bucket[i] = "global_fallback"

    out[f"{TARGET}_pred_per_dopant_platt"] = calibrated
    out["_platt_bucket"] = used_bucket
    return out


def _metrics(label, y_pred, y_true):
    if len(y_true) < 3 or np.var(y_true) < 1e-12:
        return None
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r, _ = pearsonr(y_pred, y_true)
    return dict(label=label, n=len(y_true), r2=r2, mae=mae, rmse=rmse, r=r)


def _print_metrics(title, rows):
    print(title)
    for m in rows:
        if m is None:
            continue
        print(
            f"  {m['label']:22s}  N={m['n']:3d}  R²={m['r2']:+.3f}  "
            f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  r={m['r']:+.3f}"
        )


def run(session_dir: Path) -> dict:
    oof_path = session_dir / "oof_predictions.csv"
    ext_path = session_dir / "external" / "external_predictions.csv"

    oof = pd.read_csv(oof_path)
    if "dopant_label" not in oof.columns:
        raise ValueError(f"OOF file missing dopant_label column: {oof_path}")

    global_model, per_dopant, fit_summary = _fit_per_dopant(oof)

    print(f"\n=== {session_dir.name} ===")
    print(f"Global slope/intercept: a={global_model.coef_[0]:.3f}, b={global_model.intercept_:.3f}")
    print(f"Per-dopant calibrators fitted for {len(per_dopant)} classes (N≥{MIN_TRAIN}):")
    for label, n, a, b in sorted(fit_summary, key=lambda x: -x[1]):
        print(f"  {label:12s}  N_train={n:3d}  a={a:+.3f}  b={b:+.3f}")

    # Apply to OOF
    oof_cal = _apply(oof, global_model, per_dopant)
    oof_cal.to_csv(session_dir / "oof_per_dopant_platt.csv", index=False)

    # Global metrics pre/post
    m_oof = oof[oof[f"{TARGET}_pred"].notna() & oof[f"{TARGET}_true"].notna()]
    yp_raw = m_oof[f"{TARGET}_pred"].to_numpy()
    yt = m_oof[f"{TARGET}_true"].to_numpy()
    yp_cal = oof_cal.loc[m_oof.index, f"{TARGET}_pred_per_dopant_platt"].to_numpy()

    oof_metrics_raw = _metrics("OOF raw", yp_raw, yt)
    oof_metrics_cal = _metrics("OOF per-dopant-Platt", yp_cal, yt)

    # Per-dopant metrics
    oof_per_dopant_metrics = []
    for label, sub in m_oof.groupby("dopant_label"):
        if len(sub) < 3:
            continue
        idx = sub.index
        oof_per_dopant_metrics.append(
            _metrics(f"OOF {label}", sub[f"{TARGET}_pred"].to_numpy(),
                     sub[f"{TARGET}_true"].to_numpy())
        )
    oof_per_dopant_cal = []
    for label, sub in oof_cal.loc[m_oof.index].groupby("dopant_label"):
        if len(sub) < 3:
            continue
        oof_per_dopant_cal.append(
            _metrics(f"OOF+Platt {label}",
                     sub[f"{TARGET}_pred_per_dopant_platt"].to_numpy(),
                     oof.loc[sub.index, f"{TARGET}_true"].to_numpy())
        )

    print()
    _print_metrics("[OOF metrics]", [oof_metrics_raw, oof_metrics_cal])
    _print_metrics("[OOF per-dopant raw]", oof_per_dopant_metrics)
    _print_metrics("[OOF per-dopant after per-dopant Platt]", oof_per_dopant_cal)

    # External
    ext_result = {}
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        # Ensure dopant_label column — derive from element if missing
        if "dopant_label" not in ext.columns:
            if "element" in ext.columns:
                ext["dopant_label"] = ext["element"].fillna("—")
            else:
                raise ValueError(f"External missing dopant_label and element: {ext_path}")
        ext_cal = _apply(ext, global_model, per_dopant)
        ext_cal.to_csv(session_dir / "external" / "external_per_dopant_platt.csv", index=False)

        m_ext = ext[ext[f"{TARGET}_pred"].notna() & ext[f"{TARGET}_true"].notna()]
        if len(m_ext) >= 3:
            yp_raw_e = m_ext[f"{TARGET}_pred"].to_numpy()
            yt_e = m_ext[f"{TARGET}_true"].to_numpy()
            yp_cal_e = ext_cal.loc[m_ext.index, f"{TARGET}_pred_per_dopant_platt"].to_numpy()

            ext_m_raw = _metrics("External raw", yp_raw_e, yt_e)
            ext_m_cal = _metrics("External per-dopant-Platt", yp_cal_e, yt_e)
            print()
            _print_metrics("[External metrics]", [ext_m_raw, ext_m_cal])

            # Per-dopant external
            ext_dopant_raw = []
            ext_dopant_cal = []
            for label, sub in m_ext.groupby("dopant_label"):
                if len(sub) < 2:
                    continue
                ext_dopant_raw.append(
                    _metrics(f"Ext {label}",
                             sub[f"{TARGET}_pred"].to_numpy(),
                             sub[f"{TARGET}_true"].to_numpy())
                )
                cal_sub = ext_cal.loc[sub.index]
                ext_dopant_cal.append(
                    _metrics(f"Ext+Platt {label}",
                             cal_sub[f"{TARGET}_pred_per_dopant_platt"].to_numpy(),
                             sub[f"{TARGET}_true"].to_numpy())
                )
            print()
            _print_metrics("[Ext per-dopant raw]", ext_dopant_raw)
            _print_metrics("[Ext per-dopant after per-dopant Platt]", ext_dopant_cal)

            ext_result = dict(ext_raw=ext_m_raw, ext_cal=ext_m_cal,
                              ext_per_dopant_raw=ext_dopant_raw,
                              ext_per_dopant_cal=ext_dopant_cal)

    return dict(
        session=session_dir.name,
        oof_raw=oof_metrics_raw,
        oof_cal=oof_metrics_cal,
        oof_per_dopant_raw=oof_per_dopant_metrics,
        oof_per_dopant_cal=oof_per_dopant_cal,
        **ext_result,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Per-dopant Platt calibration + metrics")
    ap.add_argument("paths", nargs="*",
                    help="Session directories to process (default: five Phase-5 PDR sessions).")
    ap.add_argument("--target", default="photo_dark_ratio",
                    help="Target column to calibrate (photo_dark_ratio or "
                         "vacancy_concentration). Sets module-level TARGET.")
    args = ap.parse_args()

    TARGET = args.target   # overrides the module default for this run

    paths = [Path(p) for p in args.paths]
    if not paths:
        paths = [
            Path("results/phase4f_pdr_aug3x_5seed"),
            Path("results/phase5c_pdr_conc18_5seed_clean"),
            Path("results/phase5d_pdr_gbv2_5seed"),
            Path("results/phase5e_pdr_interp_5seed"),
            Path("results/phase5f_pdr_gbv2_interp_5seed"),
        ]
    for p in paths:
        if not p.exists():
            print(f"[skip] {p} does not exist")
            continue
        run(p)
