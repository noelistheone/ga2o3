"""
make_lab_report_figures.py — physics-audience figures showing the
model's single-element sputter prediction accuracy.

Two scopes:
  * scope='sputter' (DEFAULT) — full single-element sputter OOF, every
    dopant element with at least 2 labelled samples is shown. Best for
    a presentation about overall model breadth across the literature
    you have collected.
  * scope='lab' — restricted to the lab-deployable slice (Ar atmosphere,
    native β-Ga₂O₃ bulk, no post-anneal). Best for the "what can the lab
    actually do today" argument.

Avoids ML jargon. All quantities reported in physical units (log₁₀ of
V_O concentration in cm⁻³, log₁₀ of photo-to-dark current ratio).
Error bars are calibrated 80 % prediction intervals consumed from
`results/lab_regime_eval/conformal_quantiles.json`.

Outputs (PNG @ 200 dpi + PDF) to:
  results/lab_regime_eval/figures/full_sputter/  (scope='sputter')
  results/lab_regime_eval/figures/lab_regime/    (scope='lab')

Files in each subdirectory:
  fig1_parity.{png,pdf}
  fig2_concentration_response.{png,pdf}
  fig3_per_element_accuracy.{png,pdf}
  fig4_uncertainty_reliability.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.eval_lab_target_regime import _build_dataset, tag_proximity

# ── Style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "legend.fontsize":   11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    ":",
    "figure.dpi":        200,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# Colour-blind-safe palette (Bang Wong)
ELEM_COLORS = {
    "Mg":      "#0072B2",  # blue
    "Sn":      "#009E73",  # green
    "Fe":      "#D55E00",  # red-orange
    "Zn":      "#CC79A7",  # pink
    "Si":      "#56B4E9",  # sky blue
    "F":       "#E69F00",  # orange
    "W":       "#6A3D9A",  # purple
    "Ti":      "#B15928",  # brown
    "Al":      "#999999",  # grey
    "B":       "#FFD92F",  # yellow
    "undoped": "#444444",  # dark grey
    "Sb":      "#A6CEE3",
    "Bi":      "#B2DF8A",
    "V":       "#FB9A99",
}

# Lab-relevant single-element dopants the actual lab cares about most (used
# for the highlight in the parity plot — the rest of the dataset is shown
# alongside, just with a softer accent so the headline elements still pop).
LAB_ELEMENTS = ["Mg", "Sn"]

# Minimum N before we draw a per-element bar. Below this the Pearson r
# bootstrap CI is too wide to communicate anything (N=2 has trivial r=±1).
# Singletons + N=2 still show up in the parity plot, just not in the bars.
MIN_N_FOR_REPORT = 3

# Bootstrap CI bands on the per-element bars. 1000 draws is plenty for the
# ≤50-row strata we have here.
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_CI    = 0.90  # 5/95 % bands

PI80_Z = 1.2815515655446004

OUT_DIR_BASE = PROJ / "results" / "lab_regime_eval" / "figures"

# Bundles
BUNDLES = {
    "VC":  {
        "dir":    PROJ / "results" / "deployment" / "vc_alt_33_sputter_brouwer_atmo_class",
        "target_col": "vacancy_concentration",
        "ylabel": r"$\log_{10}\,[\,V_{\mathrm{O}}\,]\;(\mathrm{cm^{-3}})$",
        "name":   "Oxygen vacancy concentration",
    },
    "PDR": {
        "dir":    PROJ / "results" / "deployment" / "pdr_alt_27_sputter_cmixup",
        "target_col": "photo_dark_ratio",
        "ylabel": r"$\log_{10}\,(\,I_{\mathrm{photo}}\,/\,I_{\mathrm{dark}}\,)$",
        "name":   "Photo-to-dark current ratio",
    },
}


def _conformal_q(target: str, element: str) -> float:
    p = PROJ / "results" / "lab_regime_eval" / "conformal_quantiles.json"
    blk = next(t for t in json.loads(p.read_text())["targets"] if t["target"] == target)
    qd = blk["per_elem_tier0_q"].get(element, {})
    if qd.get("q_norm") is not None:
        return float(qd["q_norm"])
    if blk["per_tier_q"].get("0", {}).get("q_norm") is not None:
        return float(blk["per_tier_q"]["0"]["q_norm"])
    return float(blk["q_global_norm"])


def load_oof(target: str, scope: str = "sputter") -> pd.DataFrame:
    """Return OOF rows with calibrated PI half-widths and concentrations.

    scope:
        'sputter' — every OOF row with a valid label (full single-element
                    sputter dataset; every dopant the literature has data for).
        'lab'     — only tier-0 rows (Ar atmosphere × native bulk × no anneal).
    """
    bundle = BUNDLES[target]
    target_col = bundle["target_col"]

    oof = pd.read_csv(bundle["dir"] / "oof_predictions.csv")
    fu  = yaml.safe_load((bundle["dir"] / "fusion_head128.yaml").read_text())
    ds  = _build_dataset(fu)
    proc = ds.get_process_array()

    cols = [c for c in
            ("element", "atmosphere", "concentration_at%", "doi", "method", "notes")
            if c in ds.df.columns]
    src = ds.df[cols].copy().reset_index(drop=True)
    src["sample_idx"] = np.arange(len(src), dtype=int)

    merged = oof.merge(src, on="sample_idx", how="left")
    tagged = tag_proximity(merged, proc[merged.sample_idx.to_numpy()])
    if scope == "lab":
        tagged = tagged[tagged.tier == 0]
    elif scope != "sputter":
        raise ValueError(f"unknown scope: {scope!r}")

    sub = tagged.dropna(
        subset=[f"{target_col}_true", f"{target_col}_pred_platt", f"{target_col}_std"]
    ).copy()

    # Drop undoped — they share identical composition features and aren't a
    # "single dopant element" answer the professor asked about.
    sub = sub[sub.element != "undoped"].copy()

    # Apply per-element calibrated PI (fall back to global → naive z).
    half = np.empty(len(sub), dtype=float)
    for elem in sub.element.unique():
        m = sub.element == elem
        try:
            q = _conformal_q(target, elem)
        except (KeyError, StopIteration):
            q = PI80_Z
        half[m] = q * np.maximum(sub.loc[m, f"{target_col}_std"].to_numpy(), 1e-9)
    sub["pi80_half"] = half

    sub = sub.rename(columns={
        f"{target_col}_true":       "y_true",
        f"{target_col}_pred_platt": "y_pred",
        f"{target_col}_std":        "y_std",
    })
    return sub.reset_index(drop=True)


# ── Bootstrap helper ────────────────────────────────────────────────
def _bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray,
                  fn, n: int = BOOTSTRAP_DRAWS, ci: float = BOOTSTRAP_CI,
                  rng_seed: int = 0) -> tuple[float, float, float]:
    """(point_estimate, lo, hi) over n bootstrap resamples paired across a/b."""
    rng = np.random.default_rng(rng_seed)
    k = len(values_a)
    if k < 2:
        v = fn(values_a, values_b)
        return v, v, v
    boots = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, k, size=k)
        boots[i] = fn(values_a[idx], values_b[idx])
    point = fn(values_a, values_b)
    lo = float(np.nanpercentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(boots, (1 + ci) / 2 * 100))
    return float(point), lo, hi


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(a, b).statistic)


def _coverage_safe(true: np.ndarray, half: np.ndarray, pred: np.ndarray) -> float:
    if len(true) == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred) <= half))


# ── Figure 1: parity plot ─────────────────────────────────────────────
SCOPE_TITLE = {
    "sputter": "Single-element sputter dataset (all process conditions)",
    "lab":     "Lab regime: Ar atmosphere × native β-Ga$_2$O$_3$ bulk × no anneal",
}


def fig1_parity(vc: pd.DataFrame, pdr: pd.DataFrame,
                out_dir: Path, scope: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    panels = [
        (axes[0], vc,  "VC",
         BUNDLES["VC"]["ylabel"],  "Oxygen vacancy concentration"),
        (axes[1], pdr, "PDR",
         BUNDLES["PDR"]["ylabel"], "Photo-to-dark current ratio"),
    ]

    for ax, df, target, ylab, name in panels:
        # Diagonal y=x
        lo = min(df.y_true.min(), df.y_pred.min())
        hi = max(df.y_true.max(), df.y_pred.max())
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5,
                label="perfect prediction")
        ax.fill_between([lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5],
                        color="grey", alpha=0.10,
                        label="±0.5 dex band (factor 3.2)")

        # Sort elements by descending N so the bigger groups go onto the
        # canvas first and are visually prominent.
        elem_counts = df.element.value_counts()
        ordered_elems = list(elem_counts.index)

        for elem in ordered_elems:
            sub = df[df.element == elem]
            if len(sub) == 0:
                continue
            color = ELEM_COLORS.get(elem, "#888888")
            is_headline = elem in LAB_ELEMENTS
            r_str = ""
            if len(sub) >= 2 and np.std(sub.y_true) > 1e-9:
                r = float(stats.pearsonr(sub.y_true, sub.y_pred).statistic)
                r_str = f", r={r:+.2f}"
            else:
                r_str = " (singleton)"
            ax.errorbar(
                sub.y_true, sub.y_pred,
                yerr=sub.pi80_half,
                fmt="o" if is_headline else "^",
                color=color, ecolor=color,
                alpha=0.85 if is_headline else 0.7,
                markersize=9 if is_headline else 7,
                capsize=2.5, capthick=0.8, elinewidth=0.9,
                markeredgecolor="white", markeredgewidth=0.7,
                label=f"{elem} (N={len(sub)}{r_str})",
            )

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Measured {ylab}")
        ax.set_ylabel(f"Predicted {ylab}")
        ax.set_title(name)
        # legend outside on the right when there are many elements
        n_elems = df.element.nunique()
        if n_elems <= 5:
            ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=9.5)
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=True, framealpha=0.92, fontsize=9,
                      title="Dopant (N, r)")

    fig.suptitle(
        f"Predicted vs measured — {SCOPE_TITLE[scope]}\n"
        "Error bars: calibrated 80 % confidence interval",
        fontsize=13.5, y=1.02,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1_parity.png")
    fig.savefig(out_dir / "fig1_parity.pdf")
    plt.close(fig)
    print(f"  [{scope}] fig1: parity plot")


# ── Figure 2: concentration response per paper ────────────────────────
def _within_doi_score(df: pd.DataFrame, elem: str) -> int:
    """How many rows belong to a multi-row DOI for this element."""
    sub = df[df.element == elem].copy()
    sub["c"] = pd.to_numeric(sub["concentration_at%"], errors="coerce")
    sub = sub.dropna(subset=["c"])
    if len(sub) == 0:
        return 0
    return int(sub.groupby("doi").filter(lambda g: len(g) >= 2).shape[0])


def _pick_panels(vc: pd.DataFrame, pdr: pd.DataFrame, max_panels: int = 6
                 ) -> list[tuple]:
    """Pick the (element, target) cells with the richest within-DOI data."""
    candidates = []
    for elem in sorted(set(vc.element.unique()) | set(pdr.element.unique())):
        for target, df in [("VC", vc), ("PDR", pdr)]:
            score = _within_doi_score(df, elem)
            if score >= 3:  # need ≥3 rows in multi-row DOIs
                candidates.append((score, elem, target, df))
    candidates.sort(key=lambda x: -x[0])
    return [(elem, target, df) for _, elem, target, df in candidates[:max_panels]]


def fig2_concentration_response(vc: pd.DataFrame, pdr: pd.DataFrame,
                                out_dir: Path, scope: str) -> None:
    """For each (element, target) cell with within-DOI sweeps, plot
    measured values + model predictions with 80 % PI error bars."""
    panels = _pick_panels(vc, pdr, max_panels=6)
    if not panels:
        print(f"  [{scope}] fig2: no within-DOI sweeps → skipping")
        return

    n_panels = len(panels)
    nrows = (n_panels + 2) // 3
    ncols = min(n_panels, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows),
                             squeeze=False)

    cmap = plt.colormaps.get_cmap("tab10")
    for ax_i, (elem, target, df) in enumerate(panels):
        ax = axes[ax_i // ncols, ax_i % ncols]
        target_name = "V$_O$" if target == "VC" else "PDR"
        ylab = BUNDLES[target]["ylabel"]
        sub = df[df.element == elem].copy()
        sub["conc_at_pct"] = pd.to_numeric(sub["concentration_at%"], errors="coerce")
        sub = sub.dropna(subset=["conc_at_pct"])

        doi_groups = (sub.groupby("doi").filter(lambda g: len(g) >= 2)
                          .groupby("doi"))
        for i, (doi, grp) in enumerate(doi_groups):
            grp = grp.sort_values("conc_at_pct").reset_index(drop=True)
            color = cmap(i % 10)
            label = (str(doi)[:24] + "…") if len(str(doi)) > 25 else str(doi)
            label = f"{label}  (N={len(grp)})"
            ax.plot(grp.conc_at_pct, grp.y_true, "o-",
                    color=color, lw=1.4, ms=7,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=f"{label} measured", zorder=3)
            ax.errorbar(grp.conc_at_pct, grp.y_pred,
                        yerr=grp.pi80_half,
                        fmt="s--", color=color, mfc="white",
                        mec=color, mew=1.2, ms=7,
                        ecolor=color, alpha=0.85,
                        capsize=2.5, capthick=0.7, elinewidth=0.8,
                        label=f"{label} predicted", zorder=2)

        ax.set_xlabel("Dopant concentration (at%)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{elem} — {target_name} vs concentration")
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)

    # Blank out unused axes
    for k in range(n_panels, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle(
        f"Within-paper concentration response — measured vs predicted "
        f"({SCOPE_TITLE[scope]})\n"
        "Each colour = one published study; every prediction is out-of-fold "
        "(the model never saw that paper)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_concentration_response.png")
    fig.savefig(out_dir / "fig2_concentration_response.pdf")
    plt.close(fig)
    print(f"  [{scope}] fig2: concentration response ({n_panels} panels)")


# ── Figure 3: per-element accuracy bar (MAE — well-defined for any N ≥ 1) ─
def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def fig3_per_element(vc: pd.DataFrame, pdr: pd.DataFrame,
                     out_dir: Path, scope: str) -> None:
    """Mean absolute error in dex per element.

    MAE is preferred over Pearson r for this audience because:
      - it works for elements where every literature measurement happens
        at one fixed condition (no variance ⇒ Pearson undefined, but the
        model still has something to be right or wrong about);
      - 'dex' is a physically intuitive unit (0.5 dex = factor 3.2 error).
    Lower bars = more accurate predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

    for ax, df, target, name in [
        (axes[0], vc,  "VC",  "V$_O$ concentration"),
        (axes[1], pdr, "PDR", "Photo-to-dark current ratio"),
    ]:
        rows = []
        for elem in sorted(df.element.unique(),
                           key=lambda e: (-len(df[df.element == e]), e)):
            sub = df[df.element == elem]
            n = len(sub)
            if n < MIN_N_FOR_REPORT:
                continue
            mae, lo, hi = _bootstrap_ci(
                sub.y_true.to_numpy(), sub.y_pred.to_numpy(),
                _mae, rng_seed=hash(elem) & 0xFFFF)
            # Per-element Pearson r computed for annotation only — may be
            # NaN when y_true has no variance; that is fine for annotation.
            r = (float(stats.pearsonr(sub.y_true, sub.y_pred).statistic)
                 if np.std(sub.y_true) > 1e-9 else float("nan"))
            rows.append(dict(element=elem, mae=mae, mae_lo=lo, mae_hi=hi,
                             r=r, n=n))
        rows = pd.DataFrame(rows)

        if rows.empty:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="grey"); continue

        colors = [ELEM_COLORS.get(e, "#888888") for e in rows.element]
        yerr_lo = (rows.mae - rows.mae_lo).clip(lower=0).to_numpy()
        yerr_hi = (rows.mae_hi - rows.mae).clip(lower=0).to_numpy()
        bars = ax.bar(rows.element, rows.mae, color=colors,
                      edgecolor="black", lw=0.8,
                      yerr=[yerr_lo, yerr_hi],
                      capsize=4, ecolor="#333333", error_kw=dict(lw=1.2))

        for b, (_, r) in zip(bars, rows.iterrows()):
            top = max(r["mae"], r["mae_hi"]) if np.isfinite(r["mae_hi"]) else r["mae"]
            r_str = f", r={r['r']:+.2f}" if np.isfinite(r["r"]) else ""
            ax.text(b.get_x() + b.get_width() / 2, top + 0.05,
                    f"N={r['n']}{r_str}",
                    ha="center", va="bottom", fontsize=8.5)

        # Reference lines: 0.5 dex (factor 3.2) and 1.0 dex (factor 10)
        ax.axhline(0.5, color="#1f77b4", lw=0.8, ls=":", alpha=0.6,
                   label="0.5 dex (factor 3.2)")
        ax.axhline(1.0, color="#d62728", lw=0.8, ls=":", alpha=0.6,
                   label="1.0 dex (factor 10)")
        ax.set_ylim(0, max(2.2, rows.mae_hi.max() * 1.15))
        ax.set_ylabel("Mean absolute prediction error (dex)\n"
                      "lower = more accurate")
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=0 if len(rows) <= 7 else 35)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
        ax.text(0.99, -0.13,
                f"bars sorted by sample size N; error bar = bootstrap 90 % CI; "
                f"N ≥ {MIN_N_FOR_REPORT} only",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#555555", style="italic")

    fig.suptitle(
        f"Prediction–measurement agreement, per dopant element\n"
        f"{SCOPE_TITLE[scope]}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_per_element_accuracy.png")
    fig.savefig(out_dir / "fig3_per_element_accuracy.pdf")
    plt.close(fig)
    print(f"  [{scope}] fig3: per-element accuracy")


# ── Figure 4: uncertainty reliability ─────────────────────────────────
def _coverage_with_ci(true: np.ndarray, pred: np.ndarray, half: np.ndarray,
                      seed: int = 0) -> tuple[float, float, float]:
    inside = float(np.mean(np.abs(true - pred) <= half))
    rng = np.random.default_rng(seed)
    k = len(true)
    if k < 2:
        return inside, inside, inside
    boots = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for i in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, k, size=k)
        boots[i] = float(np.mean(np.abs(true[idx] - pred[idx]) <= half[idx]))
    lo = float(np.percentile(boots, (1 - BOOTSTRAP_CI) / 2 * 100))
    hi = float(np.percentile(boots, (1 + BOOTSTRAP_CI) / 2 * 100))
    return inside, lo, hi


def fig4_uncertainty(vc: pd.DataFrame, pdr: pd.DataFrame,
                     out_dir: Path, scope: str) -> None:
    """Show that the 80% confidence interval really covers ~80% of measurements."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

    for ax, df, name in [
        (axes[0], vc,  "V$_O$ concentration"),
        (axes[1], pdr, "Photo-to-dark current ratio"),
    ]:
        rows = []
        for elem in sorted(df.element.unique(),
                           key=lambda e: (-len(df[df.element == e]), e)):
            sub = df[df.element == elem]
            if len(sub) < MIN_N_FOR_REPORT: continue
            inside, lo, hi = _coverage_with_ci(
                sub.y_true.to_numpy(), sub.y_pred.to_numpy(), sub.pi80_half.to_numpy(),
                seed=hash(elem) & 0xFFFF)
            rows.append(dict(element=elem, inside=inside, lo=lo, hi=hi, n=len(sub)))
        all_in, all_lo, all_hi = _coverage_with_ci(
            df.y_true.to_numpy(), df.y_pred.to_numpy(), df.pi80_half.to_numpy())
        rows.append(dict(element="ALL", inside=all_in, lo=all_lo, hi=all_hi, n=len(df)))
        rdf = pd.DataFrame(rows)

        colors = [ELEM_COLORS.get(e, "#404040") for e in rdf.element]
        colors = ["#000000" if e == "ALL" else c for c, e in zip(colors, rdf.element)]
        yerr_lo = ((rdf.inside - rdf.lo) * 100).clip(lower=0).to_numpy()
        yerr_hi = ((rdf.hi - rdf.inside) * 100).clip(lower=0).to_numpy()
        bars = ax.bar(rdf.element, rdf.inside * 100, color=colors,
                      edgecolor="black", lw=0.8, alpha=0.9,
                      yerr=[yerr_lo, yerr_hi],
                      capsize=4, ecolor="#333333", error_kw=dict(lw=1.2))
        for b, (_, r) in zip(bars, rdf.iterrows()):
            top = max(r["inside"], r["hi"]) * 100
            ax.text(b.get_x() + b.get_width() / 2, top + 1.5,
                    f"N={r['n']}", ha="center", fontsize=9)
        ax.axhline(80, color="red", lw=2, ls="--", label="target = 80 %")
        ax.set_ylim(0, 115)
        ax.set_ylabel("Fraction of measurements inside\nthe 80 % confidence interval (%)")
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=0 if len(rdf) <= 7 else 35)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
        ax.text(0.99, -0.15,
                f"error bar = bootstrap 90 % CI; "
                f"N ≥ {MIN_N_FOR_REPORT} only",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#555555", style="italic")

    fig.suptitle(
        "Reliability check: when the model says “80 % confident”, "
        "what fraction of measurements actually fall inside?\n"
        f"{SCOPE_TITLE[scope]} · calibrated post-hoc on the full out-of-fold set",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_uncertainty_reliability.png")
    fig.savefig(out_dir / "fig4_uncertainty_reliability.pdf")
    plt.close(fig)
    print(f"  [{scope}] fig4: uncertainty reliability")


# ── Driver ────────────────────────────────────────────────────────────
def _run_scope(scope: str) -> None:
    out_dir = OUT_DIR_BASE / ("full_sputter" if scope == "sputter" else "lab_regime")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Scope: {scope} → {out_dir.relative_to(PROJ)} ===")
    vc  = load_oof("VC",  scope=scope)
    pdr = load_oof("PDR", scope=scope)
    print(f"  VC:  N={len(vc):3d}  elements={sorted(vc.element.unique())}")
    print(f"  PDR: N={len(pdr):3d}  elements={sorted(pdr.element.unique())}")
    fig1_parity(vc, pdr, out_dir, scope)
    fig2_concentration_response(vc, pdr, out_dir, scope)
    fig3_per_element(vc, pdr, out_dir, scope)
    fig4_uncertainty(vc, pdr, out_dir, scope)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="both",
                        choices=["sputter", "lab", "both"],
                        help="sputter = full single-element sputter OOF; "
                             "lab = restrict to Ar × native bulk × no anneal "
                             "tier-0 cell")
    args = parser.parse_args()

    OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
    if args.scope in ("sputter", "both"):
        _run_scope("sputter")
    if args.scope in ("lab", "both"):
        _run_scope("lab")
    print(f"\nAll figures saved under {OUT_DIR_BASE}/")


if __name__ == "__main__":
    main()
