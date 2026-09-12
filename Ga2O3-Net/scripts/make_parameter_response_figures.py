"""
make_parameter_response_figures.py — two more figures answering:
  Fig 11 (within-element):  for the same element, does the model track
                            V_O / PDR variation across multiple
                            experimental parameters (concentration,
                            temperature, atmosphere, substrate)?
  Fig 12 (cross-element):   when both the element AND the experimental
                            parameter change, can the model still
                            capture the response?

Both figures use full single-element sputter OOF predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.eval_lab_target_regime import _build_dataset, tag_proximity
from scripts.make_lab_report_figures import (
    BUNDLES, ELEM_COLORS, PI80_Z, _conformal_q,
)
from scripts.make_physics_learning_figures import ELEM_VALENCE


OUT_DIR = PROJ / "results" / "lab_regime_eval" / "figures" / "physics_learning"


# ── Loader: include temperature / time / substrate / voltage columns ──
def load_oof_with_process(target: str) -> pd.DataFrame:
    bundle = BUNDLES[target]
    target_col = bundle["target_col"]

    oof = pd.read_csv(bundle["dir"] / "oof_predictions.csv")
    fu  = yaml.safe_load((bundle["dir"] / "fusion_head128.yaml").read_text())
    ds  = _build_dataset(fu)
    proc = ds.get_process_array()

    cols = [c for c in
            ("element", "atmosphere", "concentration_at%", "doi", "method",
             "notes", "temperature_C", "time_min",
             "measurement_voltage_V", "measurement_wavelength_nm")
            if c in ds.df.columns]
    src = ds.df[cols].copy().reset_index(drop=True)
    src["sample_idx"] = np.arange(len(src), dtype=int)

    merged = oof.merge(src, on="sample_idx", how="left")
    tagged = tag_proximity(merged, proc[merged.sample_idx.to_numpy()])
    sub = tagged.dropna(
        subset=[f"{target_col}_true", f"{target_col}_pred_platt", f"{target_col}_std"]
    ).copy()
    sub = sub[sub.element != "undoped"]

    # Calibrated PI half-widths
    half = np.empty(len(sub), dtype=float)
    for elem in sub.element.unique():
        m = sub.element == elem
        try:
            q = _conformal_q(target, elem)
        except (KeyError, StopIteration):
            q = PI80_Z
        half[m] = q * np.maximum(sub.loc[m, f"{target_col}_std"].to_numpy(), 1e-9)
    sub["pi80_half"] = half

    # tag_proximity adds boolean flags we can repurpose
    sub["on_foreign_substrate"] = 1 - sub["native_bulk_ok"].astype(int)
    sub["has_anneal"]           = 1 - sub["no_anneal_ok"].astype(int)

    sub = sub.rename(columns={
        f"{target_col}_true":       "y_true",
        f"{target_col}_pred_platt": "y_pred",
        f"{target_col}_std":        "y_std",
    })
    return sub.reset_index(drop=True)


# ── Atmosphere ordinal: more inert → larger value ────────────────────
ATMO_ORDER = ["O2_plasma", "O2", "Ar:O2", "Ar+N2", "Ar+H2", "Ar", "vacuum"]


def _atm_ordinal(a) -> int:
    if pd.isna(a) or str(a).strip() == "":
        return ATMO_ORDER.index("Ar")
    s = str(a).strip().lower()
    if "plasma" in s:                  return ATMO_ORDER.index("O2_plasma")
    if "ar:o2" in s:                   return ATMO_ORDER.index("Ar:O2")
    if "h2" in s:                      return ATMO_ORDER.index("Ar+H2")
    if "n2" in s:                      return ATMO_ORDER.index("Ar+N2")
    if s == "o2":                      return ATMO_ORDER.index("O2")
    if s == "vacuum":                  return ATMO_ORDER.index("vacuum")
    if s == "ar":                      return ATMO_ORDER.index("Ar")
    return ATMO_ORDER.index("Ar")  # default


# ── Parameter list each plot iterates over ──────────────────────────
PARAMS = [
    # (col, transform, x_label, x_kind)
    # x_kind: 'cont' (continuous numeric) or 'ord' (categorical, integer)
    ("concentration_at%", lambda x: np.log10(x), "log₁₀(concentration / at%)", "cont"),
    ("temperature_C",     lambda x: x,           "deposition T (°C)",          "cont"),
    ("atmosphere",        lambda x: x.map(_atm_ordinal), "atmosphere", "ord"),
    ("on_foreign_substrate", lambda x: x,        "substrate (0 = native bulk, 1 = foreign)", "ord"),
]


def _prepare_param(df: pd.DataFrame, col: str, transform):
    """Apply transform; drop rows where the resulting x is NaN/empty."""
    raw = df[col]
    if col == "concentration_at%":
        x = pd.to_numeric(raw, errors="coerce")
        x = pd.Series(np.where(x > 0, np.log10(x), np.nan), index=raw.index)
        return x
    if col == "atmosphere":
        return raw.map(_atm_ordinal)
    if col == "on_foreign_substrate":
        return raw.astype(int)
    if col == "temperature_C":
        return pd.to_numeric(raw, errors="coerce")
    return transform(raw)


# ── Figure 11: per-element parameter response panorama ──────────────
def fig11_within_element(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """For each element with sufficient rows in a target, plot how
    measured + predicted values respond to each experimental parameter.
    The element list is determined per-target so empty-data rows are
    avoided.

    Layout: dynamic — VC block above, PDR block below, with a thin
    separator. Each block stacks (element × parameter) panels.
    """
    blocks_data = []
    for target_label, df, ylab, name in [
        ("VC",  vc,
         BUNDLES["VC"]["ylabel"],
         "Oxygen vacancy concentration"),
        ("PDR", pdr,
         BUNDLES["PDR"]["ylabel"],
         "Photo-to-dark current ratio"),
    ]:
        # Include EVERY element that has at least one labelled row in
        # this target — even singletons get a row so the audience can see
        # the model's single prediction next to the single measurement.
        # Within each block, sort by formal valence then by N descending.
        elem_counts = df.element.value_counts().to_dict()
        elems = sorted(
            elem_counts.keys(),
            key=lambda e: (ELEM_VALENCE.get(e, 99), -elem_counts[e], e),
        )
        blocks_data.append((target_label, df, ylab, name, elems))

    n_params = len(PARAMS)
    block_sizes = [len(b[4]) for b in blocks_data]
    total_rows  = sum(block_sizes)
    if total_rows == 0:
        print("  fig11: no eligible (element × parameter) cells, skipping")
        return

    # With many rows, shrink each row's height a bit so the figure is
    # presentable (~2.0 in / row instead of 2.7 in / row).
    row_height = 2.0 if total_rows > 8 else 2.7
    fig, axes = plt.subplots(
        total_rows, n_params,
        figsize=(4.0 * n_params, row_height * total_rows),
        squeeze=False,
    )
    row_offsets = [0, block_sizes[0]]

    for blk_i, (target_label, df, ylab, name, elems) in enumerate(blocks_data):
        for elem_i, elem in enumerate(elems):
            row = row_offsets[blk_i] + elem_i
            for param_i, (col, transform, xlab, kind) in enumerate(PARAMS):
                ax = axes[row, param_i]
                sub = df[df.element == elem].copy()
                sub["x"] = _prepare_param(sub, col, transform)
                cell = sub.dropna(subset=["x"]).copy()
                color = ELEM_COLORS.get(elem, "#888")

                # Title on the very top row of each block
                if elem_i == 0:
                    ax.set_title(xlab, fontsize=10)

                if param_i == 0:
                    v = ELEM_VALENCE.get(elem, 99)
                    vstr = f"  v={v}" if v != 99 else ""
                    ax.set_ylabel(f"{target_label} · {elem}{vstr}\n"
                                  f"N={len(sub)}",
                                  fontsize=9.5)

                if len(cell) == 0:
                    ax.text(0.5, 0.5, "(no value\nfor this knob)",
                            ha="center", va="center", transform=ax.transAxes,
                            color="#aaa", fontsize=8.5, style="italic")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                if kind == "ord":
                    x_jit = cell["x"] + np.random.default_rng(
                        hash(elem + col) & 0xFFFF).uniform(-0.14, 0.14,
                                                            size=len(cell))
                else:
                    x_jit = cell["x"]

                ax.plot(x_jit, cell.y_true, "o",
                        color=color, ms=7, alpha=0.85,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label="measured", zorder=3)
                ax.errorbar(x_jit, cell.y_pred, yerr=cell.pi80_half,
                            fmt="s", color=color, mfc="white",
                            mec=color, mew=1.1, ms=6,
                            ecolor=color, alpha=0.7,
                            capsize=2, capthick=0.6, elinewidth=0.6,
                            label="model", zorder=2)

                if col == "atmosphere":
                    used = sorted(cell["x"].unique())
                    ax.set_xticks(used)
                    ax.set_xticklabels([ATMO_ORDER[int(u)] for u in used],
                                       fontsize=7.5, rotation=25, ha="right")
                if col == "on_foreign_substrate":
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["native", "foreign"], fontsize=8)
                ax.tick_params(axis="both", labelsize=8.5)

                # First non-empty cell in the figure gets a marker legend
                if blk_i == 0 and elem_i == 0 and param_i == 0:
                    ax.legend(loc="upper left", frameon=True,
                              framealpha=0.92, fontsize=8)

    # Separator between VC block and PDR block
    if total_rows > 1 and block_sizes[0] > 0 and block_sizes[1] > 0:
        sep_y = 1 - block_sizes[0] / total_rows
        fig.add_artist(plt.Line2D([0.05, 0.97], [sep_y, sep_y],
                                  color="#888", linewidth=0.7,
                                  linestyle="--", transform=fig.transFigure))

    fig.suptitle(
        "Within-element parameter response — for each dopant, does the "
        "model track every experimental knob?\n"
        "Filled circles = literature measurements;  open squares + 80 % CI "
        "bars = model predictions out-of-fold",
        fontsize=12.5, y=1.005,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig11_within_element_panorama.png")
    fig.savefig(OUT_DIR / "fig11_within_element_panorama.pdf")
    plt.close(fig)

    summary = ", ".join(
        f"{tag}=[{','.join(elems)}]"
        for tag, _, _, _, elems in blocks_data if elems
    )
    print(f"  fig11: within-element panorama ({summary})")


# ── Figure 12: cross-element parameter consistency ──────────────────
def fig12_cross_element(vc: pd.DataFrame, pdr: pd.DataFrame) -> None:
    """Each parameter gets a column. All elements are overlaid in the same
    panel, color-coded. Filled circles = measured, open squares = predicted.
    Demonstrates that no matter which element or which parameter changes,
    the model's predictions track the measurements."""
    blocks = [
        ("VC",  vc,  BUNDLES["VC"]["ylabel"],  "Oxygen vacancy concentration"),
        ("PDR", pdr, BUNDLES["PDR"]["ylabel"], "Photo-to-dark current ratio"),
    ]
    n_params = len(PARAMS)
    fig, axes = plt.subplots(2, n_params,
                              figsize=(4.5 * n_params, 8.5),
                              squeeze=False)

    for blk_i, (target_label, df, ylab, name) in enumerate(blocks):
        # Sort elements by N so plotting order is deterministic
        elems = df.element.value_counts().index.tolist()
        for param_i, (col, transform, xlab, kind) in enumerate(PARAMS):
            ax = axes[blk_i, param_i]

            for elem in elems:
                sub = df[df.element == elem].copy()
                if len(sub) < 1:
                    continue
                sub["x"] = _prepare_param(sub, col, transform)
                cell = sub.dropna(subset=["x"])
                if len(cell) == 0:
                    continue
                color = ELEM_COLORS.get(elem, "#888")

                if kind == "ord":
                    x_jit_m = cell["x"] + np.random.default_rng(
                        hash(elem + col + "_m") & 0xFFFF).uniform(-0.16, 0.16,
                                                                  size=len(cell))
                    x_jit_p = cell["x"] + np.random.default_rng(
                        hash(elem + col + "_p") & 0xFFFF).uniform(-0.16, 0.16,
                                                                  size=len(cell))
                else:
                    x_jit_m = cell["x"]
                    x_jit_p = cell["x"]
                ax.plot(x_jit_m, cell.y_true, "o",
                        color=color, ms=7, alpha=0.85,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=f"{elem} measured (N={len(cell)})", zorder=3)
                ax.plot(x_jit_p, cell.y_pred, "s",
                        color="white", mec=color, mew=1.2, ms=6,
                        alpha=0.75, label=f"{elem} predicted", zorder=2)

            if col == "atmosphere":
                ax.set_xticks(range(len(ATMO_ORDER)))
                ax.set_xticklabels(ATMO_ORDER, fontsize=8, rotation=25,
                                   ha="right")
            if col == "on_foreign_substrate":
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["native", "foreign"], fontsize=9)

            ax.set_xlabel(xlab, fontsize=10)
            if param_i == 0:
                ax.set_ylabel(f"{target_label}\n{ylab}", fontsize=10)
            ax.tick_params(axis="both", labelsize=8.5)

        # Single legend for the row, on the right
        handles, labels = axes[blk_i, 0].get_legend_handles_labels()
        # De-duplicate while preserving order — keep "measured" entries only
        seen = set()
        keep_h, keep_l = [], []
        for h, l in zip(handles, labels):
            base = l.split(" measured")[0].split(" predicted")[0]
            if "measured" in l and base not in seen:
                seen.add(base)
                keep_h.append(h)
                keep_l.append(base)
        # Add the marker-style legend entries
        from matplotlib.lines import Line2D
        keep_h.extend([
            Line2D([], [], color="black", marker="o", linestyle="",
                   markersize=7, label="measured"),
            Line2D([], [], color="white", marker="s", linestyle="",
                   markersize=7, mec="black", mew=1.2, label="model"),
        ])
        keep_l.extend(["── measured ──", "── model ──"])
        fig.legend(keep_h, keep_l,
                   loc="center left",
                   bbox_to_anchor=(1.005, 0.75 if blk_i == 0 else 0.27),
                   frameon=True, framealpha=0.92, fontsize=8.5,
                   title=f"{target_label}", title_fontsize=10)

    fig.suptitle(
        "Cross-element parameter consistency — does the model track the "
        "right response when both the dopant and the experimental knob change?\n"
        "Filled circles = literature measurements;  open squares = model "
        "predictions out-of-fold",
        fontsize=12.5, y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 1.0))
    fig.savefig(OUT_DIR / "fig12_cross_element_response.png")
    fig.savefig(OUT_DIR / "fig12_cross_element_response.pdf")
    plt.close(fig)
    print(f"  fig12: cross-element parameter consistency")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading full sputter OOF + process columns...")
    vc  = load_oof_with_process("VC")
    pdr = load_oof_with_process("PDR")
    print(f"  VC:  N={len(vc):3d}  elements={sorted(vc.element.unique())}")
    print(f"  PDR: N={len(pdr):3d}  elements={sorted(pdr.element.unique())}")
    print("Generating parameter-response figures...")
    fig11_within_element(vc, pdr)
    fig12_cross_element(vc, pdr)
    print(f"\nFigures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
