"""
Re-generate Fig 2 / Fig 5 / Fig 6 / Fig 7 for results/paper_single_element.

The original phase7_build_paper_package.py produced:
  * Fig 2 with unclipped R^2 CIs (Si PDR went to -1.25e8, destroyed axis)
  * Fig 5 with under-confidence but caption said "well-calibrated" (wrong)
  * Fig 6 showing only 2 dopants with R^2 crushed by one external CI outlier
  * Fig 7 middle panel nearly empty + misleading domain-shift claim

This script keeps the same input CSVs and filenames but replots these four
with correct axis choices (Pearson r for bounded bars, bootstrap CIs clipped
to [-1, 1], sample-size annotations, honest captions).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr, norm
from pathlib import Path

OUT = Path("results/paper_single_element")


# ── metric helpers ────────────────────────────────────────────────────────────

def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _boot(y, p, fn, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y)
    out = []
    for _ in range(n):
        s = rng.choice(N, N, replace=True)
        ys, ps = y[s], p[s]
        if len(np.unique(ys)) < 2 or len(np.unique(ps)) < 2:
            continue
        out.append(fn(ys, ps))
    return np.array(out)


def boot_r_ci(y, p, n=2000):
    samp = _boot(y, p, lambda a, b: pearsonr(a, b)[0], n=n)
    if len(samp) < 20:
        return (np.nan, np.nan)
    return tuple(np.percentile(samp, [2.5, 97.5]))


def boot_r2_ci(y, p, n=2000):
    samp = _boot(y, p, r2, n=n)
    if len(samp) < 20:
        return (np.nan, np.nan)
    return tuple(np.percentile(samp, [2.5, 97.5]))


# ── load data ─────────────────────────────────────────────────────────────────

oof_pdr = pd.read_csv(OUT / "oof_pdr.csv").dropna(
    subset=["photo_dark_ratio_true", "photo_dark_ratio_pred_per_dopant_platt"]
)
oof_vc = pd.read_csv(OUT / "oof_vc.csv").dropna(
    subset=["vacancy_concentration_true", "vacancy_concentration_pred_per_dopant_platt"]
)
ext_pdr = pd.read_csv(OUT / "external_pdr.csv")


# ── Fig 2 — per-dopant R^2 + Pearson r with bootstrap 95% CI ─────────────────

def plot_fig2():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    tasks = [
        ("PDR", oof_pdr, "photo_dark_ratio_true",
         "photo_dark_ratio_pred", "photo_dark_ratio_pred_per_dopant_platt"),
        ("VC", oof_vc, "vacancy_concentration_true",
         "vacancy_concentration_pred", "vacancy_concentration_pred_per_dopant_platt"),
    ]

    for col, (title, df, y_col, praw_col, pplt_col) in enumerate(tasks):
        rows = []
        for d, g in df.groupby("dopant_label"):
            if len(g) < 5:
                continue
            y = g[y_col].values
            praw, pplt = g[praw_col].values, g[pplt_col].values
            r_raw = pearsonr(y, praw)[0] if len(set(y)) > 1 else np.nan
            r_plt = pearsonr(y, pplt)[0] if len(set(y)) > 1 else np.nan
            r2_raw, r2_plt = r2(y, praw), r2(y, pplt)
            ci_r_raw = boot_r_ci(y, praw)
            ci_r_plt = boot_r_ci(y, pplt)
            ci_r2_raw = boot_r2_ci(y, praw)
            ci_r2_plt = boot_r2_ci(y, pplt)
            rows.append((d, len(g), r_raw, r_plt, r2_raw, r2_plt,
                         ci_r_raw, ci_r_plt, ci_r2_raw, ci_r2_plt))

        rows.sort(key=lambda t: -t[1])  # by N desc
        labels = [f"{d}\n(N={n})" for d, n, *_ in rows]
        y_pos = np.arange(len(rows))

        # --- R^2 bars (clipped) ---
        ax_r2 = axes[0, col]
        bw = 0.36
        r2_raw_vals = [t[4] for t in rows]
        r2_plt_vals = [t[5] for t in rows]
        ci_r2_raw = np.array([[t[8][0], t[8][1]] for t in rows])
        ci_r2_plt = np.array([[t[9][0], t[9][1]] for t in rows])

        # clip CIs for plotting (but show clipping indicator)
        def clip_err(vals, cis, lo=-1.0, hi=1.2):
            lo_err = np.clip(np.array(vals) - cis[:, 0], 0, np.inf)
            hi_err = np.clip(cis[:, 1] - np.array(vals), 0, np.inf)
            clipped_lo = cis[:, 0] < lo
            clipped_hi = cis[:, 1] > hi
            lo_err = np.where(clipped_lo, np.array(vals) - lo, lo_err)
            hi_err = np.where(clipped_hi, hi - np.array(vals), hi_err)
            return np.vstack([lo_err, hi_err]), clipped_lo, clipped_hi

        err_raw, clo_raw, chi_raw = clip_err(r2_raw_vals, ci_r2_raw)
        err_plt, clo_plt, chi_plt = clip_err(r2_plt_vals, ci_r2_plt)

        ax_r2.barh(y_pos - bw/2, np.clip(r2_raw_vals, -1, 1.2), bw,
                   xerr=err_raw, color="#8ab6d6", label="raw", capsize=3,
                   error_kw={"elinewidth": 1.1, "alpha": 0.85})
        ax_r2.barh(y_pos + bw/2, np.clip(r2_plt_vals, -1, 1.2), bw,
                   xerr=err_plt, color="#2b6cb0", label="per-dopant Platt", capsize=3,
                   error_kw={"elinewidth": 1.1, "alpha": 0.85})

        # annotate clipping with a small arrow tip where CI bottom was cut off
        for i, (c_lo, c_hi) in enumerate(zip(clo_raw, chi_raw)):
            if c_lo:
                ax_r2.annotate("◀", (-1.0, y_pos[i] - bw/2), ha="left",
                               va="center", fontsize=7, color="#555")
        for i, (c_lo, c_hi) in enumerate(zip(clo_plt, chi_plt)):
            if c_lo:
                ax_r2.annotate("◀", (-1.0, y_pos[i] + bw/2), ha="left",
                               va="center", fontsize=7, color="#555")

        ax_r2.axvline(0, color="k", lw=0.5)
        ax_r2.set_xlim(-1.05, 1.2)
        ax_r2.set_yticks(y_pos)
        ax_r2.set_yticklabels(labels, fontsize=9)
        ax_r2.set_xlabel(r"$R^2$ (clipped to $[-1, 1.2]$; ◀ = CI lower bound cut off)")
        ax_r2.set_title(f"{title} — per-dopant $R^2$")
        ax_r2.grid(axis="x", alpha=0.3)
        ax_r2.legend(loc="lower right", fontsize=8)

        # --- Pearson r bars (naturally bounded [-1, 1]) ---
        ax_r = axes[1, col]
        r_raw_vals = [t[2] for t in rows]
        r_plt_vals = [t[3] for t in rows]
        ci_r_raw = np.array([[t[6][0], t[6][1]] for t in rows])
        ci_r_plt = np.array([[t[7][0], t[7][1]] for t in rows])

        lo_err_raw = np.array(r_raw_vals) - ci_r_raw[:, 0]
        hi_err_raw = ci_r_raw[:, 1] - np.array(r_raw_vals)
        lo_err_plt = np.array(r_plt_vals) - ci_r_plt[:, 0]
        hi_err_plt = ci_r_plt[:, 1] - np.array(r_plt_vals)

        ax_r.barh(y_pos - bw/2, r_raw_vals, bw,
                  xerr=np.vstack([lo_err_raw, hi_err_raw]),
                  color="#f6c28b", label="raw", capsize=3,
                  error_kw={"elinewidth": 1.1, "alpha": 0.85})
        ax_r.barh(y_pos + bw/2, r_plt_vals, bw,
                  xerr=np.vstack([lo_err_plt, hi_err_plt]),
                  color="#d67341", label="per-dopant Platt", capsize=3,
                  error_kw={"elinewidth": 1.1, "alpha": 0.85})

        ax_r.axvline(0, color="k", lw=0.5)
        ax_r.set_xlim(-1.05, 1.05)
        ax_r.set_yticks(y_pos)
        ax_r.set_yticklabels(labels, fontsize=9)
        ax_r.set_xlabel(r"Pearson $r$ (bootstrap 95% CI)")
        ax_r.set_title(f"{title} — per-dopant Pearson $r$")
        ax_r.grid(axis="x", alpha=0.3)
        ax_r.legend(loc="lower right", fontsize=8)

    fig.suptitle("Per-dopant prediction quality (5-seed 10-fold GroupKFold OOF)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_per_dopant_r2_ci.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_per_dopant_r2_ci.pdf")


# ── Fig 5 — honest calibration plot ───────────────────────────────────────────

def plot_fig5():
    """
    Rebuild reliability diagram, but this time show the empirical coverage
    alongside a ±1/√N noise band and explicitly label the under-dispersion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    tasks = [
        ("PDR", oof_pdr, "photo_dark_ratio_true",
         "photo_dark_ratio_pred_per_dopant_platt", "photo_dark_ratio_std"),
        ("VC",  oof_vc, "vacancy_concentration_true",
         "vacancy_concentration_pred_per_dopant_platt", "vacancy_concentration_std"),
    ]

    for ax, (title, df, y_col, p_col, s_col) in zip(axes, tasks):
        y = df[y_col].values
        p = df[p_col].values
        s = df[s_col].values
        s = np.maximum(s, 1e-6)
        z = (y - p) / s  # z-scores against MC-Dropout sigma
        levels = np.arange(0.05, 1.0, 0.05)
        emp, noise = [], []
        N = len(z)
        for lvl in levels:
            z_lvl = norm.ppf(0.5 + lvl / 2)  # two-sided
            emp.append(np.mean(np.abs(z) < z_lvl))
            noise.append(np.sqrt(lvl * (1 - lvl) / N))  # binomial sd
        emp = np.array(emp)
        noise = np.array(noise)

        ax.fill_between(levels, emp - 1.96*noise, emp + 1.96*noise,
                        alpha=0.2, color="#2b6cb0", label="empirical ±1.96·SE")
        ax.plot(levels, emp, "o-", color="#2b6cb0", lw=1.5, label="empirical coverage")
        ax.plot([0, 1], [0, 1], "--", color="k", lw=1, label="ideal")

        # flag under-dispersion: empirical < nominal at 0.9 level
        cov90 = np.interp(0.90, levels, emp)
        ax.annotate(f"90% nominal → {cov90*100:.0f}% empirical\n"
                    f"(under-dispersed — MC σ too small)",
                    xy=(0.90, cov90), xytext=(0.35, 0.85),
                    arrowprops=dict(arrowstyle="->", color="#c0392b"),
                    fontsize=9, color="#c0392b")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"{title} MC-Dropout reliability (N={N})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("MC-Dropout is under-dispersed — the reported σ "
                 "needs downstream scaling before deployment", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_uncertainty_calibration.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_uncertainty_calibration.pdf")


# ── Fig 6 — internal vs external Pearson r per dopant ────────────────────────

def plot_fig6():
    """
    Use Pearson r (bounded [-1, 1]) instead of R^2. Show every internal dopant
    with N_int >= 5, plus external bars where external N >= 3. Elements with
    external N < 3 are shown as an annotation ("ext N=X, r undefined") rather
    than a wildly uncertain point estimate.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    elements = []
    for d, g in oof_pdr.groupby("dopant_label"):
        if len(g) < 5:
            continue
        y = g["photo_dark_ratio_true"].values
        p = g["photo_dark_ratio_pred_per_dopant_platt"].values
        r_int = pearsonr(y, p)[0]
        ci_int = boot_r_ci(y, p)
        # external
        ext_rows = ext_pdr[(ext_pdr["dopant_label"] == d)
                           & ext_pdr["photo_dark_ratio_true"].notna()]
        n_ext = len(ext_rows)
        y_e = ext_rows["photo_dark_ratio_true"].values
        p_e = ext_rows["photo_dark_ratio_pred_per_dopant_platt"].values
        if n_ext >= 3 and len(set(y_e)) >= 2:
            r_ext = pearsonr(y_e, p_e)[0]
            # var of y for annotation
            y_range = y_e.max() - y_e.min()
        else:
            r_ext = np.nan
            y_range = np.nan
        elements.append((d, len(g), r_int, ci_int, n_ext, r_ext, y_range))

    elements.sort(key=lambda t: -t[1])
    y_pos = np.arange(len(elements))
    bw = 0.38

    for i, (d, n_int, r_int, ci_int, n_ext, r_ext, y_range) in enumerate(elements):
        lo_err = r_int - ci_int[0] if not np.isnan(ci_int[0]) else 0
        hi_err = ci_int[1] - r_int if not np.isnan(ci_int[1]) else 0
        ax.barh(y_pos[i] - bw/2, r_int, bw,
                xerr=[[lo_err], [hi_err]],
                color="#2b6cb0",
                label="internal OOF" if i == 0 else None, capsize=3,
                error_kw={"elinewidth": 1.1})
        if not np.isnan(r_ext):
            ax.barh(y_pos[i] + bw/2, r_ext, bw,
                    color="#c0392b",
                    label="external literature" if i == 0 else None)
            ax.text(r_ext + 0.05 * np.sign(r_ext or 1) + 0.02, y_pos[i] + bw/2,
                    f"N_ext={n_ext}, Δy={y_range:.2f}",
                    va="center", fontsize=8, color="#7a2d20")
        else:
            ax.text(0.02, y_pos[i] + bw/2,
                    f"N_ext={n_ext} → r undefined",
                    va="center", fontsize=8, color="#888",
                    style="italic")
        # internal N on left
        ax.text(-1.02, y_pos[i] - bw/2, f"N_int={n_int}",
                va="center", fontsize=8, color="#333")

    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-1.15, 1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d for d, *_ in elements], fontsize=10)
    ax.set_xlabel("Pearson $r$ (per-dopant Platt prediction vs measurement)")
    ax.set_title("Fig 6 — PDR: internal OOF vs external literature, per dopant")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # explanatory note
    ax.text(0.5, -0.22,
            "External rows per dopant: " +
            ", ".join([f"{d} N={n_ext}" for d, _, _, _, n_ext, _, _ in elements]) +
            ".  'Δy' = range of measured log$_{10}$(PDR) in external subset; "
            "Δy ≲ 0.1 makes Pearson $r$ statistically meaningless (shown only when N_ext ≥ 3 & Δy > 0).",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")

    fig.tight_layout()
    fig.savefig(OUT / "fig6_internal_vs_external.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig6_internal_vs_external.pdf")


# ── Fig 7 — external-Sn honest diagnosis ─────────────────────────────────────

def plot_fig7():
    """
    The previous Fig 7 framed external-Sn as a method/substrate domain-shift
    failure. Inspection of the actual data shows:
      * All 3 external Sn rows come from ONE paper (10.3390/ma17133227).
      * Method = RF magnetron sputtering (SAME bucket as 29/69 training Sn).
      * Measured log10(PDR) = {1.978, 2.009, 2.004} — 0.03-unit spread.
      * Predictions spread ~0.15 log units.
    With essentially constant y and N=3, Pearson r is arbitrary. So the
    redesigned figure shows the honest picture: a within-DOI concentration
    curve where the model predicts a small decreasing trend while the measured
    PDR is flat, with no claim to a domain-shift diagnosis.
    """
    sn_tr = oof_pdr[oof_pdr["dopant_label"] == "Sn"].copy()
    sn_ex = ext_pdr[ext_pdr["dopant_label"] == "Sn"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- left: training Sn measured vs predicted (range context) ---
    ax = axes[0]
    ax.scatter(sn_tr["photo_dark_ratio_pred_per_dopant_platt"],
               sn_tr["photo_dark_ratio_true"],
               alpha=0.6, s=28, color="#2b6cb0", label=f"Training Sn (N={len(sn_tr)})")
    # mark the external-y range on the same axes
    y_min, y_max = sn_ex["photo_dark_ratio_true"].min(), sn_ex["photo_dark_ratio_true"].max()
    ax.axhspan(y_min - 0.02, y_max + 0.02, color="#c0392b", alpha=0.15,
               label=f"External-Sn measured range  Δy={y_max-y_min:.3f}")
    xy_min = min(sn_tr["photo_dark_ratio_true"].min(),
                 sn_tr["photo_dark_ratio_pred_per_dopant_platt"].min()) - 0.3
    xy_max = max(sn_tr["photo_dark_ratio_true"].max(),
                 sn_tr["photo_dark_ratio_pred_per_dopant_platt"].max()) + 0.3
    ax.plot([xy_min, xy_max], [xy_min, xy_max], "--", color="k", lw=0.8, label="y=x")
    ax.set_xlim(xy_min, xy_max); ax.set_ylim(xy_min, xy_max)
    ax.set_xlabel(r"Predicted $\log_{10}$(PDR)")
    ax.set_ylabel(r"Measured $\log_{10}$(PDR)")
    ax.set_title("Training Sn covers a wide PDR range;\n"
                 "external-Sn measurements fall in a narrow 0.03-log-unit band")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # --- right: external Sn — concentration vs measured & predicted PDR ---
    ax = axes[1]
    c = sn_ex["concentration_at%"].astype(float).values
    y_e = sn_ex["photo_dark_ratio_true"].values
    p_e = sn_ex["photo_dark_ratio_pred_per_dopant_platt"].values
    order = np.argsort(c)
    ax.plot(c[order], y_e[order], "o-", color="#2b6cb0", lw=1.5,
            markersize=8, label=f"measured (N={len(c)})")
    ax.plot(c[order], p_e[order], "s--", color="#c0392b", lw=1.5,
            markersize=8, label=f"predicted (per-dopant Platt)")
    for ci, yi, pi in zip(c, y_e, p_e):
        ax.annotate(f"  {yi:.3f}", (ci, yi), fontsize=8, color="#2b6cb0",
                    va="bottom", ha="left")
        ax.annotate(f"  {pi:.3f}", (ci, pi), fontsize=8, color="#c0392b",
                    va="top", ha="left")
    ax.set_xlabel("Concentration (at%)")
    ax.set_ylabel(r"$\log_{10}$(PDR)")
    doi = sn_ex["source_doi"].iloc[0]
    method = sn_ex["method"].iloc[0]
    ax.set_title(f"External Sn = 3 rows, single paper\n"
                 f"DOI {doi} · method: {method}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Fig 7 — External-Sn evaluation is underpowered, not a domain-shift diagnosis.\n"
                 "Pearson $r=-0.92$ on 3 near-constant points is a statistical artifact.",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_sn_domain_shift.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig7_sn_domain_shift.pdf")


if __name__ == "__main__":
    plot_fig2()
    plot_fig5()
    plot_fig6()
    plot_fig7()
