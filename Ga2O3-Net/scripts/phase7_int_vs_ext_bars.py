"""
Phase 7A — Figure 6: internal OOF vs external R² per dopant.

For each dopant with ≥MIN_EXT external labelled rows, plots side-by-side bars:
internal OOF R²_Platt (with bootstrap 95% CI) vs external R²_Platt (with CI).
The Sn bar highlights the known domain-shift reversal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

MIN_EXT = 2
BOOTSTRAP_DRAWS = 1000
RNG_SEED = 42

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _r2_ci(y_true, y_pred, n=BOOTSTRAP_DRAWS, seed=RNG_SEED):
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


def _per_dopant(df: pd.DataFrame, target: str):
    pred_pl = f"{target}_pred_per_dopant_platt"
    if pred_pl not in df.columns:
        pred_pl = f"{target}_pred"
    true = f"{target}_true"
    sub = df.dropna(subset=[pred_pl, true])
    out = {}
    for dop, g in sub.groupby("dopant_label"):
        yt = g[true].to_numpy()
        yp = g[pred_pl].to_numpy()
        out[dop] = dict(N=len(g), r2=_r2_ci(yt, yp))
    return out


def _plot(ax, oof_stats: dict, ext_stats: dict, target_label: str):
    dopants = sorted([d for d in ext_stats.keys() if ext_stats[d]["N"] >= MIN_EXT])
    if not dopants:
        ax.text(0.5, 0.5, f"No dopant with N_ext ≥ {MIN_EXT}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(target_label)
        return

    x = np.arange(len(dopants))
    w = 0.4

    oof_pts = [oof_stats.get(d, {}).get("r2", (np.nan, np.nan, np.nan)) for d in dopants]
    ext_pts = [ext_stats[d]["r2"] for d in dopants]

    oof_vals = [p[0] for p in oof_pts]
    oof_err = np.array([
        [p[0] - p[1] if np.isfinite(p[1]) else 0.0 for p in oof_pts],
        [p[2] - p[0] if np.isfinite(p[2]) else 0.0 for p in oof_pts],
    ])
    ext_vals = [p[0] for p in ext_pts]
    ext_err = np.array([
        [p[0] - p[1] if np.isfinite(p[1]) else 0.0 for p in ext_pts],
        [p[2] - p[0] if np.isfinite(p[2]) else 0.0 for p in ext_pts],
    ])

    ax.bar(x - w / 2, oof_vals, w, yerr=oof_err, color="#4C72B0",
           label="internal OOF", ecolor="black", capsize=3)
    ax.bar(x + w / 2, ext_vals, w, yerr=ext_err, color="#C44E52",
           label="external", ecolor="black", capsize=3)

    # Sample-size annotations
    for xi, d in zip(x, dopants):
        n_int = oof_stats.get(d, {}).get("N", 0)
        n_ext = ext_stats[d]["N"]
        ax.annotate(f"$N_{{int}}$={n_int}\n$N_{{ext}}$={n_ext}",
                    xy=(xi, min(oof_vals[list(dopants).index(d)], ext_vals[list(dopants).index(d)]) if
                        np.isfinite(oof_vals[list(dopants).index(d)]) and np.isfinite(ext_vals[list(dopants).index(d)])
                        else 0),
                    xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=8, color="#555555")

        # Annotate Sn external with domain-shift note if reversed
        if d == "Sn" and np.isfinite(ext_vals[list(dopants).index(d)]) and ext_vals[list(dopants).index(d)] < 0:
            ax.annotate("see fig7\n(domain shift)",
                        xy=(xi + w / 2, ext_vals[list(dopants).index(d)]),
                        xytext=(12, -6), textcoords="offset points",
                        fontsize=8, color="#C44E52",
                        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.6))

    ax.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(dopants)
    ax.set_ylabel(r"$R^2$ (per-dopant Platt, bootstrap 95% CI)")
    ax.set_title(target_label)
    ax.legend(loc="upper right", frameon=False)


def _maybe_load(path):
    return pd.read_csv(path) if path and Path(path).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-oof", type=Path, required=True)
    ap.add_argument("--pdr-ext", type=Path, required=True)
    ap.add_argument("--vc-oof", type=Path, default=None)
    ap.add_argument("--vc-ext", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pdr_oof = pd.read_csv(args.pdr_oof)
    pdr_ext = pd.read_csv(args.pdr_ext)
    vc_oof = _maybe_load(args.vc_oof)
    vc_ext = _maybe_load(args.vc_ext)

    panels = [("PDR — internal vs external", pdr_oof, pdr_ext, "photo_dark_ratio")]
    if vc_oof is not None and vc_ext is not None:
        panels.append(("VC — internal vs external", vc_oof, vc_ext, "vacancy_concentration"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5),
                             constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, oof_df, ext_df, target) in zip(axes, panels):
        oof_stats = _per_dopant(oof_df, target)
        ext_stats = _per_dopant(ext_df, target)
        _plot(ax, oof_stats, ext_stats, title)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
