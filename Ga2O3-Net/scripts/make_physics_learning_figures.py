"""
make_physics_learning_figures.py — additional figures showing how well
the model has learned **within-element** (concentration, temperature,
atmosphere) and **cross-element** (acceptor vs donor) physical
regularities. Uses out-of-fold predictions only — no fresh inference.

Outputs (PNG @ 200 dpi + PDF) to
`results/lab_regime_eval/figures/physics_learning/`:

  fig5_concentration_overlay.{png,pdf}
      Per-element concentration response — measured vs predicted, every
      element with ≥ 4 OOF rows in one panel.

  fig6_acceptor_donor_signature.{png,pdf}
      Per-element Spearman correlation between log₁₀(V_O / PDR) and
      log₁₀(concentration), MEASURED vs PREDICTED side-by-side. Sorted
      by formal valence so the audience can see acceptors (Mg, Zn) end
      up with negative slope and donors (Sn, Si, Ti) with positive
      slope, and that the model recovers that valence ordering.

  fig7_atmosphere_x_element_bias.{png,pdf}
      Heatmap of mean signed residual (predicted − measured) for each
      (atmosphere × dopant element) cell. Red = the model over-predicts
      that combination; blue = under-predicts; grey = no data. Reveals
      whether the model has internalised the atmosphere-ordinal signal
      learnt in Phase 32 (oxidising → fewer V_O / higher PDR).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.make_lab_report_figures import (
    BUNDLES, ELEM_COLORS, MIN_N_FOR_REPORT, load_oof, mpl as _mpl_styled,  # noqa
)


# Formal valence used for cation ordering. Acceptors (v<3) → expect
# negative dV_O/dconc; donors (v>3) → positive; isovalent (v=3) → ~0.
ELEM_VALENCE = {
    "Mg": 2, "Zn": 2,                  # acceptors
    "Al": 3, "Fe": 3, "B": 3, "V": 3,  # isovalent (or mixed)
    "Si": 4, "Sn": 4, "Ti": 4,         # donors
    "Sb": 5, "Ta": 5, "Bi": 5,
    "W":  6,
    "F": -1,                           # anion (out of cation framework)
}

OUT_DIR = PROJ / "results" / "lab_regime_eval" / "figures" / "physics_learning"


# ── Helpers ──────────────────────────────────────────────────────────
def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray,
                        n: int = 1000, ci: float = 0.90,
                        seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    k = len(x)
    if k < 2:
        v = _spearman_safe(x, y)
        return v, v, v
    boots = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, size=k)
        boots[i] = _spearman_safe(x[idx], y[idx])
    point = _spearman_safe(x, y)
    lo = float(np.nanpercentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(boots, (1 + ci) / 2 * 100))
    return float(point), lo, hi


# ── Figure 5: per-element concentration response, all elements overlaid ──
def fig5_concentration_overlay(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """One panel per target. Every element with ≥ 4 OOF rows is plotted in
    log–log: filled circles = measured, open squares + dashed line =
    predicted, with the calibrated 80 % PI as vertical bars."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, df, target, ylab, name in [
        (axes[0], vc,  "VC",  BUNDLES["VC"]["ylabel"],
         "Oxygen vacancy concentration"),
        (axes[1], pdr, "PDR", BUNDLES["PDR"]["ylabel"],
         "Photo-to-dark current ratio"),
    ]:
        df = df.copy()
        df["c"] = pd.to_numeric(df["concentration_at%"], errors="coerce")
        df = df.dropna(subset=["c"])
        df = df[df.c > 0]                     # log-x requires strictly positive
        df["logc"] = np.log10(df.c)

        elems = df.element.value_counts()
        elems = [e for e in elems.index if elems[e] >= 2]
        # Sort by valence for visual consistency
        elems.sort(key=lambda e: (ELEM_VALENCE.get(e, 99), -df[df.element == e].shape[0]))

        for elem in elems:
            sub = df[df.element == elem].sort_values("logc")
            color = ELEM_COLORS.get(elem, "#888888")
            # Measured points
            ax.plot(sub.logc, sub.y_true, "o",
                    color=color, ms=8, alpha=0.85,
                    markeredgecolor="white", markeredgewidth=0.7,
                    label=f"{elem} (N={len(sub)})  measured")
            # Predicted points + 80 % PI + linear fit
            ax.errorbar(sub.logc, sub.y_pred, yerr=sub.pi80_half,
                        fmt="s", color=color, mfc="white", mec=color, mew=1.2,
                        ms=7, ecolor=color, alpha=0.65,
                        capsize=2.5, capthick=0.7, elinewidth=0.7)
            if len(sub) >= 3 and np.std(sub.logc) > 1e-9:
                fit = np.polyfit(sub.logc, sub.y_pred, 1)
                xx = np.linspace(sub.logc.min(), sub.logc.max(), 50)
                ax.plot(xx, np.polyval(fit, xx), "--",
                        color=color, lw=1.2, alpha=0.55,
                        label=f"{elem} (predicted slope = {fit[0]:+.2f})")

        ax.set_xlabel(r"$\log_{10}\,[\,\mathrm{dopant\ concentration}\,]$  (at%)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{name} — every dopant with ≥ 2 measurements")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=True, framealpha=0.92, fontsize=8.5,
                  title="Dopant — slope in log₁₀ units")

    fig.suptitle(
        "Per-element concentration response — measured vs predicted "
        "(single-element sputter, all process conditions)\n"
        "Filled circles = measurements;  open squares + dashed = model "
        "predictions out-of-fold;  bar = 80 % CI",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig5_concentration_overlay.png")
    fig.savefig(OUT_DIR / "fig5_concentration_overlay.pdf")
    plt.close(fig)
    print("  fig5: concentration overlay")


# ── Figure 6: acceptor / donor signature ─────────────────────────────
def fig6_acceptor_donor_signature(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """Per-element Spearman ρ(target, log conc), measured vs predicted,
    side-by-side bars sorted by formal valence."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

    for ax, df, target, name in [
        (axes[0], vc,  "VC",  "V$_O$ concentration"),
        (axes[1], pdr, "PDR", "Photo-to-dark current ratio"),
    ]:
        df = df.copy()
        df["c"] = pd.to_numeric(df["concentration_at%"], errors="coerce")
        df = df.dropna(subset=["c"])
        df = df[df.c > 0]
        df["logc"] = np.log10(df.c)

        rows = []
        for elem in df.element.unique():
            sub = df[df.element == elem]
            if len(sub) < MIN_N_FOR_REPORT:
                continue
            xt = sub.logc.to_numpy()
            yt = sub.y_true.to_numpy()
            yp = sub.y_pred.to_numpy()
            r_meas, lo_m, hi_m = _bootstrap_spearman(
                xt, yt, seed=hash(elem + "_meas") & 0xFFFF)
            r_pred, lo_p, hi_p = _bootstrap_spearman(
                xt, yp, seed=hash(elem + "_pred") & 0xFFFF)
            if not (np.isfinite(r_meas) or np.isfinite(r_pred)):
                continue
            rows.append(dict(
                element=elem, valence=ELEM_VALENCE.get(elem, 99),
                n=len(sub),
                r_meas=r_meas, lo_m=lo_m, hi_m=hi_m,
                r_pred=r_pred, lo_p=lo_p, hi_p=hi_p))
        rdf = pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)

        if rdf.empty:
            ax.text(0.5, 0.5, "(insufficient multi-conc data)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey")
            ax.set_title(name)
            continue

        x_pos = np.arange(len(rdf))
        bw = 0.38
        # Measured bars (filled)
        m_lo = (rdf.r_meas - rdf.lo_m).clip(lower=0).to_numpy()
        m_hi = (rdf.hi_m - rdf.r_meas).clip(lower=0).to_numpy()
        ax.bar(x_pos - bw / 2, rdf.r_meas, bw,
               color=[ELEM_COLORS.get(e, "#888888") for e in rdf.element],
               edgecolor="black", lw=0.7,
               yerr=[m_lo, m_hi], capsize=3, ecolor="#333", error_kw=dict(lw=0.9),
               label="measured")
        # Predicted bars (hatched / outlined)
        p_lo = (rdf.r_pred - rdf.lo_p).clip(lower=0).to_numpy()
        p_hi = (rdf.hi_p - rdf.r_pred).clip(lower=0).to_numpy()
        ax.bar(x_pos + bw / 2, rdf.r_pred, bw,
               facecolor="white",
               edgecolor=[ELEM_COLORS.get(e, "#888888") for e in rdf.element],
               lw=1.6, hatch="///",
               yerr=[p_lo, p_hi], capsize=3, ecolor="#333", error_kw=dict(lw=0.9),
               label="model")

        # Element labels with valence annotation
        labels = [f"{e}\n(v={int(rdf.valence.iloc[i])})"
                  if rdf.valence.iloc[i] != 99 else e
                  for i, e in enumerate(rdf.element)]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="black", lw=0.9)
        ax.axhspan(-1.05, 0, color="#0072B2", alpha=0.05,
                   label="acceptor expectation (slope < 0)")
        ax.axhspan(0, 1.05, color="#009E73", alpha=0.05,
                   label="donor expectation (slope > 0)")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Spearman correlation\n"
                      f"between log₁₀(target) and log₁₀(conc)")
        ax.set_title(name)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
        ax.text(0.99, -0.18,
                "elements sorted by formal valence; "
                "filled = literature, hatched = model; "
                "error bar = bootstrap 90 % CI",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#555555", style="italic")

    fig.suptitle(
        "Cross-element regularity: does the model recover the acceptor "
        "(slope < 0) vs donor (slope > 0) signature?\n"
        "Per-element Spearman correlation between target and dopant "
        "concentration",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_acceptor_donor_signature.png")
    fig.savefig(OUT_DIR / "fig6_acceptor_donor_signature.pdf")
    plt.close(fig)
    print("  fig6: acceptor / donor signature")


# ── Figure 7: atmosphere × element residual heatmap ─────────────────
def _atm_group(atm: object) -> str:
    if pd.isna(atm) or str(atm).strip() == "":
        return "Ar"  # dataset default
    s = str(atm).strip().lower()
    if "plasma" in s:                     return "O2_plasma"
    if "o2" in s and "ar:o2" not in s:    return "O2"
    if "ar:o2" in s:                      return "Ar:O2 mix"
    if "h2" in s:                         return "Ar+H2"
    if "n2" in s:                         return "Ar+N2"
    if s == "vacuum":                     return "vacuum"
    if s == "ar":                         return "Ar"
    return "other"


def fig7_atmosphere_x_element_bias(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """Mean signed residual (predicted - measured) per (atmosphere × element)
    cell. Reveals systematic over/under-prediction by process condition."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

    # Atmosphere category order — left-to-right roughly inert → oxidising
    ATM_ORDER = ["vacuum", "Ar", "Ar+H2", "Ar+N2", "Ar:O2 mix", "O2", "O2_plasma"]

    for ax, df, target, name in [
        (axes[0], vc,  "VC",  "V$_O$ concentration"),
        (axes[1], pdr, "PDR", "Photo-to-dark current ratio"),
    ]:
        df = df.copy()
        df["atm_group"] = df.atmosphere.map(_atm_group)
        df["resid"] = df.y_pred - df.y_true

        # Element rows: sort by N descending (the most-supported elements
        # appear at the top so the audience reads them first).
        elem_counts = df.element.value_counts()
        elems = [e for e in elem_counts.index if elem_counts[e] >= 2]
        elems.sort(key=lambda e: (-elem_counts[e], ELEM_VALENCE.get(e, 99), e))

        # Build [n_elems × n_atm] matrices. Cells with fewer than 3 rows
        # are left empty — a single residual or pair is dominated by paper-
        # level bias and would mislead the audience.
        MIN_CELL = 3
        bias = np.full((len(elems), len(ATM_ORDER)), np.nan)
        cnts = np.zeros((len(elems), len(ATM_ORDER)), dtype=int)
        for i, elem in enumerate(elems):
            for j, atm in enumerate(ATM_ORDER):
                cell = df[(df.element == elem) & (df.atm_group == atm)]
                if len(cell) < MIN_CELL:
                    continue
                bias[i, j] = float(cell.resid.mean())
                cnts[i, j] = len(cell)
        # Drop rows that have NO surviving cell
        keep = np.array([np.any(cnts[i] > 0) for i in range(len(elems))])
        elems = [e for e, k in zip(elems, keep) if k]
        bias  = bias[keep, :]
        cnts  = cnts[keep, :]

        cmap = plt.colormaps.get_cmap("RdBu_r")
        # Symmetric colour scale around 0 using the range of populated cells
        vmax = np.nanmax(np.abs(bias)) if np.isfinite(np.nanmax(np.abs(bias))) else 1.0
        vmax = float(np.ceil(vmax * 10) / 10)  # round up to one decimal
        vmax = max(vmax, 0.5)
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=+vmax)
        im = ax.imshow(bias, cmap=cmap, norm=norm, aspect="auto")

        # Annotate each populated cell with N + bias value
        for i in range(len(elems)):
            for j in range(len(ATM_ORDER)):
                if cnts[i, j] == 0:
                    continue
                v = bias[i, j]
                ax.text(j, i, f"{v:+.2f}\nN={cnts[i, j]}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > vmax * 0.55 else "black")

        # Empty cells: lightly shade in grey by overlaying patches
        for i in range(len(elems)):
            for j in range(len(ATM_ORDER)):
                if cnts[i, j] == 0:
                    ax.add_patch(mpl.patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=True, facecolor="#EEEEEE",
                        edgecolor="white", lw=0.6, zorder=0))

        ax.set_xticks(range(len(ATM_ORDER)))
        ax.set_xticklabels(ATM_ORDER, rotation=20, ha="right")
        ax.set_yticks(range(len(elems)))
        ax.set_yticklabels(elems)
        ax.set_xlabel("atmosphere group  (inert → oxidising →)")
        ax.set_ylabel("dopant element")
        ax.set_title(name)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("predicted − measured (dex)\n"
                       "  ↑ over-predicts        ↓ under-predicts",
                       fontsize=9)

    fig.suptitle(
        "Within- and cross-element bias map — by dopant element × atmosphere\n"
        "Empty cells: no measurements with this combination in the dataset",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_atmosphere_x_element_bias.png")
    fig.savefig(OUT_DIR / "fig7_atmosphere_x_element_bias.pdf")
    plt.close(fig)
    print("  fig7: atmosphere × element bias heatmap")


# ── Figure 8: per-element baseline level (cross-element ranking) ────
def fig8_element_baselines(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """For each element with N ≥ 3, show measured vs predicted MEAN value
    of the target. Side-by-side bars sorted by formal valence so the
    audience can see whether the model places each element at the right
    absolute level (not just trend)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

    BASELINE_MIN_N = 2  # mean ± SD is meaningful even at N = 2
    for ax, df, target, ylab, name in [
        (axes[0], vc,  "VC",  BUNDLES["VC"]["ylabel"],
         "Oxygen vacancy concentration"),
        (axes[1], pdr, "PDR", BUNDLES["PDR"]["ylabel"],
         "Photo-to-dark current ratio"),
    ]:
        rows = []
        for elem in df.element.unique():
            sub = df[df.element == elem]
            if len(sub) < BASELINE_MIN_N:
                continue
            rows.append(dict(
                element=elem, valence=ELEM_VALENCE.get(elem, 99),
                n=len(sub),
                mean_meas=float(sub.y_true.mean()),
                std_meas=float(sub.y_true.std(ddof=0)),
                mean_pred=float(sub.y_pred.mean()),
                std_pred=float(sub.y_pred.std(ddof=0)),
            ))
        rdf = pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)

        if rdf.empty:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(name); continue

        x_pos = np.arange(len(rdf))
        bw = 0.38
        # Measured (filled) vs predicted (hatched) per element
        ax.bar(x_pos - bw / 2, rdf.mean_meas, bw,
               color=[ELEM_COLORS.get(e, "#888888") for e in rdf.element],
               edgecolor="black", lw=0.7,
               yerr=rdf.std_meas, capsize=3, ecolor="#333", error_kw=dict(lw=0.9),
               label="measured (mean ± SD across rows)")
        ax.bar(x_pos + bw / 2, rdf.mean_pred, bw,
               facecolor="white",
               edgecolor=[ELEM_COLORS.get(e, "#888888") for e in rdf.element],
               lw=1.6, hatch="///",
               yerr=rdf.std_pred, capsize=3, ecolor="#333", error_kw=dict(lw=0.9),
               label="model (mean ± SD)")

        labels = [f"{e}\n(v={int(rdf.valence.iloc[i])}, N={rdf.n.iloc[i]})"
                  if rdf.valence.iloc[i] != 99
                  else f"{e}\n(N={rdf.n.iloc[i]})"
                  for i, e in enumerate(rdf.element)]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel(ylab)
        ax.set_title(name)
        ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=9)
        ax.text(0.99, -0.15,
                "elements sorted by formal valence; "
                "filled = literature, hatched = model",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#555555", style="italic")

    fig.suptitle(
        "Cross-element baseline level — does the model place each "
        "element at the right absolute log₁₀ value?\n"
        "Mean target across all OOF measurements per element",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_element_baselines.png")
    fig.savefig(OUT_DIR / "fig8_element_baselines.pdf")
    plt.close(fig)
    print("  fig8: per-element baseline level")


# ── Figure 9: per-element parity grid (every element, even singletons) ─
def fig9_per_element_parity_grid(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """One mini-parity panel per element. Singletons get a panel showing the
    single (true, pred) point against the diagonal — so every element with
    at least one labelled measurement is represented."""
    # Combine VC + PDR side-by-side per element. We lay out one row per
    # element, two columns (V_O on the left, PDR on the right). Elements
    # are sorted by valence then by total N (richer first).
    all_elems = sorted(
        set(vc.element.unique()) | set(pdr.element.unique()),
        key=lambda e: (ELEM_VALENCE.get(e, 99),
                       -(len(vc[vc.element == e]) + len(pdr[pdr.element == e])),
                       e),
    )

    nrows = len(all_elems)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.5, 2.0 * nrows + 0.6),
                             squeeze=False)

    # Pre-compute global axis ranges per target so every element panel
    # is on the same scale (visual consistency across the column).
    def _range(df, pad=0.4):
        vals = np.concatenate([df.y_true.to_numpy(), df.y_pred.to_numpy()])
        return float(vals.min()) - pad, float(vals.max()) + pad

    vc_lo, vc_hi   = _range(vc)
    pdr_lo, pdr_hi = _range(pdr)

    for i, elem in enumerate(all_elems):
        for j, (df, lo, hi, target_name) in enumerate([
            (vc,  vc_lo,  vc_hi,  "V$_O$"),
            (pdr, pdr_lo, pdr_hi, "PDR"),
        ]):
            ax = axes[i, j]
            sub = df[df.element == elem]
            color = ELEM_COLORS.get(elem, "#888888")

            ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.45, zorder=1)
            ax.fill_between([lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5],
                            color="grey", alpha=0.07, zorder=0)

            if len(sub) == 0:
                ax.text(0.5, 0.5, "—  no measurements",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#888", fontsize=9, style="italic")
            else:
                ax.errorbar(sub.y_true, sub.y_pred, yerr=sub.pi80_half,
                            fmt="o", color=color, ms=8, alpha=0.85,
                            markeredgecolor="white", markeredgewidth=0.7,
                            ecolor=color, capsize=2, capthick=0.7,
                            elinewidth=0.7)
                # Annotate: N + per-element MAE in dex
                mae = float(np.mean(np.abs(sub.y_true - sub.y_pred)))
                ax.text(0.03, 0.97, f"N={len(sub)}\nMAE={mae:.2f} dex",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  alpha=0.85, edgecolor="none"))

            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"{target_name}", fontsize=11)
            if j == 0:
                v = ELEM_VALENCE.get(elem, 99)
                vstr = f"  (v={v})" if v != 99 else ""
                ax.set_ylabel(f"{elem}{vstr}\npredicted", fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel("measured", fontsize=10)
            ax.tick_params(axis="both", labelsize=8.5)

    fig.suptitle(
        "Per-element prediction quality — every dopant element with at "
        "least one measurement\n"
        "(diagonal = perfect prediction; grey band = ±0.5 dex; "
        "vertical bars = 80 % CI)",
        fontsize=12.5, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_per_element_parity_grid.png")
    fig.savefig(OUT_DIR / "fig9_per_element_parity_grid.pdf")
    plt.close(fig)
    print(f"  fig9: per-element parity grid ({len(all_elems)} elements)")


# ── Figure 10: dataset inventory (ALL elements, with N counts) ──────
def fig10_dataset_inventory(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """Bar chart of dopant element coverage: how many OOF rows per element
    in total, and how many have valid V_O / PDR labels. Provides the
    audience with a clear picture of data density (and sparsity)."""
    # Build per-element counts: total OOF rows in each target's slice.
    # Re-load OOF directly so we capture rows even where the *other* target
    # is the one that's masked out.
    vc_full  = pd.read_csv(BUNDLES["VC"] ["dir"] / "oof_predictions.csv")
    pdr_full = pd.read_csv(BUNDLES["PDR"]["dir"] / "oof_predictions.csv")

    all_elems = sorted(
        set(vc_full.dopant_label.unique()) | set(pdr_full.dopant_label.unique())
    )
    all_elems = [e for e in all_elems if e != "undoped"]
    # Sort by formal valence then by total OOF rows
    all_elems.sort(key=lambda e: (
        ELEM_VALENCE.get(e, 99),
        -(int((vc_full.dopant_label == e).sum())
          + int((pdr_full.dopant_label == e).sum())),
        e,
    ))

    rows = []
    for elem in all_elems:
        v_subset  = vc_full[vc_full.dopant_label == elem]
        p_subset  = pdr_full[pdr_full.dopant_label == elem]
        rows.append(dict(
            element=elem,
            valence=ELEM_VALENCE.get(elem, 99),
            vc_total=len(v_subset),
            vc_labelled=int(v_subset.vacancy_concentration_true.notna().sum()),
            pdr_total=len(p_subset),
            pdr_labelled=int(p_subset.photo_dark_ratio_true.notna().sum()),
        ))
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.0), sharex=True)
    x_pos = np.arange(len(rdf))
    bw = 0.42

    for ax, target_total_col, target_lbl_col, name, color_lit in [
        (axes[0], "vc_total",  "vc_labelled",
         "Oxygen vacancy concentration coverage",  "#0072B2"),
        (axes[1], "pdr_total", "pdr_labelled",
         "Photo-to-dark current ratio coverage",   "#009E73"),
    ]:
        # "Total" bar (light) shows every row the model produced a prediction
        # for; "labelled" bar (dark) is the subset with a valid measurement.
        ax.bar(x_pos, rdf[target_total_col], 0.7,
               color=[ELEM_COLORS.get(e, "#888") for e in rdf.element],
               alpha=0.30, edgecolor="black", lw=0.6,
               label="model produces a prediction")
        ax.bar(x_pos, rdf[target_lbl_col], 0.7,
               color=[ELEM_COLORS.get(e, "#888") for e in rdf.element],
               edgecolor="black", lw=0.7,
               label="labelled measurement available")

        for i, r in rdf.iterrows():
            if r[target_total_col] > 0:
                ax.text(i, r[target_total_col] + 0.6,
                        f"{r[target_lbl_col]}/{r[target_total_col]}",
                        ha="center", va="bottom", fontsize=8.5)

        ax.set_ylabel("number of rows in dataset")
        ax.set_title(name)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92,
                  fontsize=9)
        ax.set_ylim(0, max(rdf[target_total_col].max() * 1.15, 5))

    labels = [f"{e}\n(v={int(r.valence)})" if r.valence != 99 else e
              for r, e in zip(rdf.itertuples(), rdf.element)]
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_xlabel("dopant element  (sorted by formal valence)")

    fig.suptitle(
        "Dataset element inventory — every dopant element the model "
        "has been trained on\n"
        "Light bar: rows with predictions; dark bar: rows with a "
        "ground-truth measurement to compare against",
        fontsize=12.5, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig10_dataset_inventory.png")
    fig.savefig(OUT_DIR / "fig10_dataset_inventory.pdf")
    plt.close(fig)
    print(f"  fig10: dataset inventory ({len(all_elems)} elements)")


# ── Driver ───────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Always use the FULL sputter scope so every element the user has
    # collected literature for is represented (lab tier-0 is too thin for
    # cross-element analysis).
    print("Loading full sputter OOF predictions...")
    vc  = load_oof("VC",  scope="sputter")
    pdr = load_oof("PDR", scope="sputter")
    print(f"  VC:  N={len(vc):3d}  elements={sorted(vc.element.unique())}")
    print(f"  PDR: N={len(pdr):3d}  elements={sorted(pdr.element.unique())}")
    print("Generating physics-learning figures...")
    fig5_concentration_overlay(vc, pdr)
    fig6_acceptor_donor_signature(vc, pdr)
    fig7_atmosphere_x_element_bias(vc, pdr)
    fig8_element_baselines(vc, pdr)
    fig9_per_element_parity_grid(vc, pdr)
    fig10_dataset_inventory(vc, pdr)
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
