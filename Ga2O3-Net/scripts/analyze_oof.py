"""
Lightweight OOF metrics summariser.

Reads ``results/oof_predictions.csv`` (or a user-supplied file) and prints a
compact summary of R², MAE, RMSE, Pearson r, slope for each target and
per-dopant Pearson r breakdown. Used after each 5-seed ensemble run.

Reports three calibration variants:
  raw       — model output before any post-hoc calibration.
  Platt     — model output × linear calibration (a, b) fitted on ALL OOF rows.
              Fitted on the same rows it is evaluated on → optimistic (mild
              leakage), but the number most commonly cited in internal reports.
  nested    — leave-one-out (LOO) cross-validated Platt. For each sample i,
              (a_i, b_i) is fitted on all rows except i, then applied to row i.
              Strictly honest: calibration never sees the row it evaluates.
              Treat this as the upper-bound-of-performance number when writing
              external reports / papers.

Also emits a 1000-draw bootstrap 95% CI for R² of each variant to convey
uncertainty at small N.

Usage::

    python scripts/analyze_oof.py [path/to/oof_predictions.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_PATH = Path("results/oof_predictions.csv")
TARGETS = ["photo_dark_ratio", "vacancy_concentration"]
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0


def _loo_platt(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Leave-one-out cross-validated Platt calibration.

    For each sample i, fit LinearRegression on all other (y_pred, y_true) pairs
    and predict sample i. Returns calibrated predictions of shape [N].
    """
    n = len(y_pred)
    calibrated = np.empty(n, dtype=np.float64)
    x_all = y_pred.reshape(-1, 1)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        lr = LinearRegression().fit(x_all[mask], y_true[mask])
        calibrated[i] = float(lr.predict(x_all[i : i + 1])[0])
    return calibrated


def _bootstrap_r2_ci(
    y_pred: np.ndarray, y_true: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Return 95% percentile CI for R² via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(y_pred)
    r2s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yp = y_pred[idx]
        yt = y_true[idx]
        if np.var(yt) < 1e-12:
            r2s[b] = np.nan
            continue
        r2s[b] = r2_score(yt, yp)
    lo = float(np.nanpercentile(r2s, 2.5))
    hi = float(np.nanpercentile(r2s, 97.5))
    return lo, hi


def _metrics(label: str, y_pred: np.ndarray, y_true: np.ndarray) -> None:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r, _ = pearsonr(y_pred, y_true)
    a, _ = np.polyfit(y_pred, y_true, 1)
    lo, hi = _bootstrap_r2_ci(y_pred, y_true)
    print(
        f"  {label:8s}  N={len(y_true):3d}  R²={r2:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]  "
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  r={r:+.3f}  slope={a:.3f}"
    )


def summarise(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"OOF file: {path}  |  rows: {len(df)}")
    print("=" * 80)

    for tgt in TARGETS:
        col_t = tgt + "_true"
        col_p = tgt + "_pred"
        col_platt = tgt + "_pred_platt"

        if col_t not in df.columns or col_p not in df.columns:
            print(f"[skip] {tgt}: no *_true or *_pred column")
            continue

        m = df[col_p].notna() & df[col_t].notna()
        if m.sum() < 3:
            print(f"[skip] {tgt}: <3 valid rows")
            continue
        y_pred = df.loc[m, col_p].to_numpy()
        y_true = df.loc[m, col_t].to_numpy()

        print(f"\n### {tgt}")
        _metrics("raw", y_pred, y_true)

        if col_platt in df.columns:
            y_platt = df.loc[m, col_platt].to_numpy()
            _metrics("Platt", y_platt, y_true)

        # LOO nested Platt — honest calibration
        if len(y_pred) >= 5:
            y_nested = _loo_platt(y_pred, y_true)
            _metrics("nested", y_nested, y_true)

    if "dopant_label" not in df.columns:
        return

    for tgt in TARGETS:
        col_p = tgt + "_pred"
        col_t = tgt + "_true"
        if col_p not in df.columns:
            continue
        print(f"\n### Per-dopant Pearson r — {tgt}")
        for label, sub in df.groupby("dopant_label"):
            m = sub[col_p].notna() & sub[col_t].notna()
            if m.sum() < 3:
                continue
            y_p = sub.loc[m, col_p].to_numpy()
            y_t = sub.loc[m, col_t].to_numpy()
            r, _ = pearsonr(y_p, y_t)
            print(f"  {label:20s}  N={int(m.sum()):3d}  r={r:+.3f}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    summarise(p)
