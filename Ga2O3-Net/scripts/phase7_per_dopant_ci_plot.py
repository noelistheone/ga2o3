"""
Phase 7A — Figure 2: per-dopant R² with bootstrap 95% CI error bars.

Input  : oof_per_dopant_platt.csv (from scripts/apply_per_dopant_platt.py)
Output : {out_dir}/fig2_per_dopant_r2_ci.pdf

For each dopant with N_int >= MIN_N_BAR, bootstraps R²_raw and R²_per_dopant_Platt
1000 times and plots horizontal bars with 95% CI error bars, one subplot per target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

MIN_N_BAR = 3
BOOTSTRAP_DRAWS = 1000
RNG_SEED = 42

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def bootstrap_r2_ci(y_true: np.ndarray, y_pred: np.ndarray,
                    n: int = BOOTSTRAP_DRAWS, seed: int = RNG_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    N = len(y_true)
    if N < 3:
        return (float("nan"), float("nan"), float("nan"))
    point = float(r2_score(y_true, y_pred))
    draws = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        if len(np.unique(y_true[idx])) < 2:
            draws[i] = np.nan
            continue
        draws[i] = r2_score(y_true[idx], y_pred[idx])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def _panel(ax, df: pd.DataFrame, target: str, title: str):
    pred_raw = f"{target}_pred"
    pred_platt = f"{target}_pred_per_dopant_platt"
    true = f"{target}_true"
    sub = df.dropna(subset=[pred_raw, true]).copy()
    if pred_platt not in sub.columns:
        pred_platt = pred_raw

    rows = []
    for dop, g in sub.groupby("dopant_label"):
        if len(g) < MIN_N_BAR:
            continue
        yt = g[true].to_numpy()
        yp_raw = g[pred_raw].to_numpy()
        yp_pl = g[pred_platt].to_numpy()
        p_raw, lo_raw, hi_raw = bootstrap_r2_ci(yt, yp_raw)
        p_pl, lo_pl, hi_pl = bootstrap_r2_ci(yt, yp_pl)
        rows.append(dict(dopant=dop, N=len(g),
                         r2_raw=p_raw, lo_raw=lo_raw, hi_raw=hi_raw,
                         r2_platt=p_pl, lo_platt=lo_pl, hi_platt=hi_pl))
    if not rows:
        ax.text(0.5, 0.5, f"No dopant with N ≥ {MIN_N_BAR}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return

    d = pd.DataFrame(rows).sort_values("N", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    bar_h = 0.38

    raw_err = np.vstack([d["r2_raw"] - d["lo_raw"], d["hi_raw"] - d["r2_raw"]])
    pl_err = np.vstack([d["r2_platt"] - d["lo_platt"], d["hi_platt"] - d["r2_platt"]])

    ax.barh(y - bar_h / 2, d["r2_raw"], bar_h, xerr=raw_err,
            color="#4C72B0", label="raw", ecolor="black", capsize=3)
    ax.barh(y + bar_h / 2, d["r2_platt"], bar_h, xerr=pl_err,
            color="#55A868", label="per-dopant Platt", ecolor="black", capsize=3)

    ax.axvline(0.0, color="grey", lw=0.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.dopant} (N={r.N})" for r in d.itertuples()])
    ax.set_xlabel(r"$R^2$ (bootstrap 95% CI)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    # Sensible x range so negative CI ticks don't dominate
    xmin = min(-0.2, float(d[["lo_raw", "lo_platt"]].min().min()) - 0.05)
    xmax = max(1.0, float(d[["hi_raw", "hi_platt"]].max().max()) + 0.05)
    ax.set_xlim(xmin, xmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-oof", required=True, type=Path,
                    help="Path to PDR oof_per_dopant_platt.csv")
    ap.add_argument("--vc-oof", type=Path, default=None,
                    help="Path to VC oof_per_dopant_platt.csv (optional)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output PDF path")
    args = ap.parse_args()

    pdr_df = pd.read_csv(args.pdr_oof)
    if args.vc_oof is not None and args.vc_oof.exists():
        vc_df = pd.read_csv(args.vc_oof)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        _panel(axes[0], pdr_df, "photo_dark_ratio", r"PDR per-dopant $R^2$")
        _panel(axes[1], vc_df, "vacancy_concentration", r"VC per-dopant $R^2$")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
        _panel(ax, pdr_df, "photo_dark_ratio", r"PDR per-dopant $R^2$")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
