"""
Phase 7A — Figure 7: Sn external-reversal domain-shift honesty figure.

Two-panel:
  A. Method × substrate contingency heatmap for training-Sn vs external-Sn,
     showing the visible clustering difference (sputter+sapphire vs MOCVD/ALD+other).
  B. External-Sn parity scatter with per-row (method, substrate) annotation +
     y=x line. Makes the reversal interpretable at a glance.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHOD_BUCKETS = [
    ("sputter", re.compile(r"sputter", re.I)),
    ("PLD", re.compile(r"\bPLD\b|pulsed laser", re.I)),
    ("CVD/ALD/MOCVD", re.compile(r"CVD|ALD|MOCVD|HVPE", re.I)),
    ("sol-gel / wet", re.compile(r"sol[-\s]?gel|wet|hydrothermal|spin", re.I)),
    ("evap / MBE", re.compile(r"evap|MBE|pulsed vapor", re.I)),
]

SUBSTRATE_BUCKETS = [
    ("native bulk", re.compile(r"EFG|Czochralski|bulk|microwire|nanowire", re.I)),
    ("GaN", re.compile(r"GaN", re.I)),
    ("MgO", re.compile(r"MgO", re.I)),
    ("Si", re.compile(r"\bSi\b|silicon", re.I)),
    ("sapphire", re.compile(r"sapph|Al2O3|c-plane", re.I)),
    ("other", None),   # catch-all
]


def classify_method(text: str) -> str:
    text = str(text) if text is not None else ""
    for name, rx in METHOD_BUCKETS:
        if rx.search(text):
            return name
    return "other"


def classify_substrate(text: str) -> str:
    text = str(text) if text is not None else ""
    for name, rx in SUBSTRATE_BUCKETS:
        if rx is None:
            continue
        if rx.search(text):
            return name
    return "other"


def _contingency(df: pd.DataFrame) -> pd.DataFrame:
    m = [classify_method(str(r.method) if "method" in df.columns else "") for r in df.itertuples()]
    joint = []
    for i, r in enumerate(df.itertuples()):
        method_s = getattr(r, "method", "") if hasattr(r, "method") else ""
        notes_s = getattr(r, "notes", "") if hasattr(r, "notes") else ""
        joint.append(classify_substrate(str(method_s) + " " + str(notes_s)))
    out = pd.DataFrame({"method": m, "substrate": joint})
    counts = out.groupby(["method", "substrate"]).size().unstack(fill_value=0)
    methods = [n for n, _ in METHOD_BUCKETS] + ["other"]
    subs = [n for n, _ in SUBSTRATE_BUCKETS]
    counts = counts.reindex(index=methods, columns=subs, fill_value=0)
    return counts


def _heatmap(ax, counts: pd.DataFrame, title: str, vmax: int):
    im = ax.imshow(counts.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_xticklabels(counts.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_yticklabels(counts.index)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = counts.iat[i, j]
            if v:
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, str(int(v)), ha="center", va="center", color=color, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Substrate bucket")
    ax.set_ylabel("Method bucket")
    return im


def _parity_panel(ax, ext_df: pd.DataFrame, target: str):
    pred_pl = f"{target}_pred_per_dopant_platt"
    if pred_pl not in ext_df.columns:
        pred_pl = f"{target}_pred"
    true = f"{target}_true"
    sub = ext_df.dropna(subset=[pred_pl, true])
    if sub.empty:
        ax.text(0.5, 0.5, "No labelled Sn external rows", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("External Sn parity")
        return

    yt = sub[true].to_numpy()
    yp = sub[pred_pl].to_numpy()
    lo = float(min(yp.min(), yt.min())) - 0.3
    hi = float(max(yp.max(), yt.max())) + 0.3
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8, alpha=0.6, label="y = x")
    ax.scatter(yp, yt, color="#C44E52", s=80, edgecolor="white", linewidth=0.8, zorder=3)

    for r in sub.itertuples():
        method_s = classify_method(str(getattr(r, "method", "")))
        sub_s = classify_substrate(str(getattr(r, "method", "")) + " " + str(getattr(r, "notes", "")))
        conc = getattr(r, "concentration_at%", None)
        label = f"{method_s}\n{sub_s}"
        if conc is not None and pd.notna(conc):
            label = f"{float(conc):.1f}at%  {method_s}\n{sub_s}"
        ax.annotate(label,
                    xy=(getattr(r, f"{target}_pred_per_dopant_platt" if pred_pl.endswith("platt") else f"{target}_pred"),
                        getattr(r, f"{target}_true")),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color="#333333")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted log$_{10}$(PDR)")
    ax.set_ylabel("Measured log$_{10}$(PDR)")
    ax.set_title("External Sn parity (reversed ⇒ domain shift)")
    ax.legend(loc="lower right", frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True,
                    help="Training CSV (ga2o3_exp_aug3x_interp.csv) to get training-Sn rows")
    ap.add_argument("--ext-pred", type=Path, required=True,
                    help="External predictions CSV (with method/notes/truth/pred cols)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    ext = pd.read_csv(args.ext_pred)

    sn_train = train[train["element"].astype(str) == "Sn"].copy()
    sn_ext = ext[ext["dopant_label"].astype(str) == "Sn"].copy() if "dopant_label" in ext.columns \
             else ext[ext["element"].astype(str) == "Sn"].copy()

    fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.2])
    ax_tr = fig.add_subplot(gs[0, 0])
    ax_ex = fig.add_subplot(gs[0, 1])
    ax_par = fig.add_subplot(gs[0, 2])

    ct_tr = _contingency(sn_train)
    ct_ex = _contingency(sn_ext)
    vmax = max(int(ct_tr.values.max()), int(ct_ex.values.max()), 1)
    im_tr = _heatmap(ax_tr, ct_tr, f"Training Sn (N={len(sn_train)}) — method × substrate", vmax)
    im_ex = _heatmap(ax_ex, ct_ex, f"External Sn (N={len(sn_ext)}) — method × substrate", vmax)
    cb = fig.colorbar(im_ex, ax=[ax_tr, ax_ex], shrink=0.75, location="bottom", pad=0.08)
    cb.set_label("row count")

    _parity_panel(ax_par, sn_ext, "photo_dark_ratio")

    fig.suptitle("Figure 7 — Sn external reversal is a method/substrate domain shift, "
                 "not a model-capacity failure.",
                 fontsize=12, y=1.03)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
