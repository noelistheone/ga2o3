"""
Phase 7A — Figure 3: per-dopant parity subplots for the major dopants.

One subplot per dopant (top N by N_labelled) showing:
  - Scatter of predictions (x) vs truths (y), color-coded by concentration_at%
  - 45° identity line (y = x)
  - Per-dopant best-fit line + shaded 95% CI band
  - Metric inset: N, R², Pearson r, MAE

Inputs: oof_per_dopant_platt.csv (from scripts/apply_per_dopant_platt.py) and
the source CSV (for concentration_at% lookup via sample_idx).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

MAX_DOPANTS = 6
MIN_N = 5

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _fit_ci_band(x, y, xs):
    """Return (mean, low, high) of the linear fit at query points xs."""
    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    mean = lr.predict(xs.reshape(-1, 1))
    # Residual SE → ±2*SE shaded band
    resid = y - lr.predict(x.reshape(-1, 1))
    se = float(np.std(resid, ddof=2)) if len(y) > 2 else 0.0
    return mean, mean - 2 * se, mean + 2 * se, lr.coef_[0], lr.intercept_


def _panel(ax, sub: pd.DataFrame, target: str, dopant: str, conc: np.ndarray):
    pred_col = f"{target}_pred_per_dopant_platt"
    if pred_col not in sub.columns:
        pred_col = f"{target}_pred"
    true_col = f"{target}_true"
    yp = sub[pred_col].to_numpy()
    yt = sub[true_col].to_numpy()

    # Scatter colored by concentration
    if np.all(np.isnan(conc)):
        sc = ax.scatter(yp, yt, c="#333333", s=28, alpha=0.75, edgecolors="white", linewidths=0.5)
        cbar = None
    else:
        valid_c = conc[~np.isnan(conc)]
        vmin = float(np.min(valid_c)) if len(valid_c) else 0.0
        vmax = float(np.max(valid_c)) if len(valid_c) else 1.0
        sc = ax.scatter(yp, yt, c=conc, cmap="viridis", s=28, alpha=0.85,
                        edgecolors="white", linewidths=0.5, vmin=vmin, vmax=vmax)
        cbar = sc

    # y=x reference (spanning combined range)
    lo = float(min(np.nanmin(yp), np.nanmin(yt)))
    hi = float(max(np.nanmax(yp), np.nanmax(yt)))
    pad = (hi - lo) * 0.06
    xlim = (lo - pad, hi + pad)
    ax.plot(xlim, xlim, "k:", lw=0.8, alpha=0.6, label="y = x")

    # Per-dopant fit + CI band
    if len(yp) >= 3 and np.std(yp) > 1e-9:
        xs = np.linspace(xlim[0], xlim[1], 40)
        mean, low, high, slope, icept = _fit_ci_band(yp, yt, xs)
        ax.plot(xs, mean, color="#C44E52", lw=1.2, label=f"fit (slope={slope:.2f})")
        ax.fill_between(xs, low, high, color="#C44E52", alpha=0.15, lw=0)

    # Metric inset
    if len(yp) >= 3 and np.std(yp) > 1e-9 and np.std(yt) > 1e-9:
        r2 = r2_score(yt, yp)
        r = pearsonr(yt, yp)[0]
        mae = mean_absolute_error(yt, yp)
        text = f"N={len(sub)}\n$R^2$={r2:+.3f}\nr={r:+.3f}\nMAE={mae:.2f}"
    else:
        text = f"N={len(sub)}\n(insufficient for metrics)"
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="grey"))

    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Predicted log$_{{10}}$({target})")
    ax.set_ylabel(f"Measured log$_{{10}}$({target})")
    ax.set_title(f"{dopant}")
    return cbar


def _load_merged(oof_path: Path, source_csv: Path) -> pd.DataFrame:
    oof = pd.read_csv(oof_path)
    src = pd.read_csv(source_csv).reset_index().rename(columns={"index": "sample_idx"})
    merge_cols = ["sample_idx"]
    for c in ("concentration_at%", "element"):
        if c in src.columns:
            merge_cols.append(c)
    merged = oof.merge(src[merge_cols], on="sample_idx", how="left")
    return merged


def _build_figure(merged: pd.DataFrame, target: str, out_path: Path):
    pred_col = f"{target}_pred_per_dopant_platt"
    if pred_col not in merged.columns:
        pred_col = f"{target}_pred"
    true_col = f"{target}_true"
    lab = merged.dropna(subset=[pred_col, true_col])

    # Rank dopants by N, take top MAX_DOPANTS with N >= MIN_N
    counts = lab["dopant_label"].value_counts()
    top = [d for d in counts.index if counts[d] >= MIN_N][:MAX_DOPANTS]
    if not top:
        print(f"[{target}] No dopant reached N >= {MIN_N}; skipping figure.")
        return

    n_cols = min(3, len(top))
    n_rows = int(np.ceil(len(top) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.4 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    last_cbar = None
    for ax, dop in zip(axes, top):
        sub = lab[lab["dopant_label"] == dop]
        conc = pd.to_numeric(sub.get("concentration_at%", pd.Series([np.nan]*len(sub))),
                             errors="coerce").to_numpy()
        last_cbar = _panel(ax, sub, target, dop, conc)
    for ax in axes[len(top):]:
        ax.axis("off")

    if last_cbar is not None:
        cb = fig.colorbar(last_cbar, ax=axes.tolist(), shrink=0.7, aspect=30,
                          location="right", pad=0.02)
        cb.set_label("concentration (at%)", rotation=90)

    fig.suptitle(f"Per-dopant parity — {target}  (Phase 6 deployment best)",
                 fontsize=13, y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-oof", type=Path)
    ap.add_argument("--vc-oof", type=Path, default=None)
    ap.add_argument("--pdr-source-csv", type=Path)
    ap.add_argument("--vc-source-csv", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.pdr_oof and args.pdr_source_csv:
        merged = _load_merged(args.pdr_oof, args.pdr_source_csv)
        _build_figure(merged, "photo_dark_ratio",
                      args.out_dir / "fig3_per_dopant_parity_pdr.pdf")
    if args.vc_oof and args.vc_source_csv:
        merged = _load_merged(args.vc_oof, args.vc_source_csv)
        _build_figure(merged, "vacancy_concentration",
                      args.out_dir / "fig3_per_dopant_parity_vc.pdf")


if __name__ == "__main__":
    main()
