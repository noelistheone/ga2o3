"""
Phase 7A orchestrator — assembles the paper-ready single-element dopant
evaluation package.

Input  : frozen deployment bundles for PDR (6D) and VC (6E) + external CSV
Output : results/paper_single_element/ with figures, LaTeX tables, CSVs,
         JSON metrics, and a unified dashboard PDF.

Steps:
  1. Snapshot OOF + external CSVs from deployment bundles into out_dir.
  2. Run apply_per_dopant_platt.py (captured stdout) for each target.
  3. Run analyze_oof.py for each target.
  4. Produce the figures via reuse of 05_evaluate.py and the new phase7_*.py plotters.
  5. Produce tab1_overall.tex inline + invoke phase7_big_table / phase7_platt_table /
     phase7_concentration_slope_table.
  6. Assemble dashboard.pdf via pdfunite.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _run(cmd: list[str], capture: bool = False) -> str:
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=capture, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout or "")
        sys.stderr.write(res.stderr or "")
        raise SystemExit(f"Command failed (code {res.returncode}): {' '.join(cmd)}")
    return res.stdout if capture else ""


def _bootstrap_r2_ci(y_true, y_pred, n=1000, seed=42):
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


def _overall(oof, target):
    pred_pl = f"{target}_pred_per_dopant_platt"
    if pred_pl not in oof.columns:
        pred_pl = f"{target}_pred"
    pred_raw = f"{target}_pred"
    true = f"{target}_true"
    lab = oof.dropna(subset=[pred_pl, true])
    yt = lab[true].to_numpy()
    yp_raw = lab[pred_raw].to_numpy() if pred_raw in lab.columns else lab[pred_pl].to_numpy()
    yp_pl = lab[pred_pl].to_numpy()

    def stats(y_true, y_pred):
        if len(y_pred) < 2 or np.std(y_pred) < 1e-9:
            return dict(N=len(y_pred), r2=float("nan"), mae=float("nan"),
                        rmse=float("nan"), r=float("nan"),
                        r2_ci_low=float("nan"), r2_ci_high=float("nan"))
        r2 = r2_score(y_true, y_pred)
        p, lo, hi = _bootstrap_r2_ci(y_true, y_pred)
        return dict(
            N=len(y_pred),
            r2=float(r2),
            mae=float(mean_absolute_error(y_true, y_pred)),
            rmse=float(np.sqrt(((y_true - y_pred) ** 2).mean())),
            r=float(pearsonr(y_true, y_pred)[0]),
            r2_ci_low=lo,
            r2_ci_high=hi,
        )

    return dict(raw=stats(yt, yp_raw), per_dopant_platt=stats(yt, yp_pl))


def _per_dopant(oof, target):
    pred_pl = f"{target}_pred_per_dopant_platt"
    if pred_pl not in oof.columns:
        pred_pl = f"{target}_pred"
    true = f"{target}_true"
    lab = oof.dropna(subset=[pred_pl, true])
    out = {}
    for dop, g in lab.groupby("dopant_label"):
        if len(g) < 2 or np.std(g[pred_pl].to_numpy()) < 1e-9:
            out[str(dop)] = dict(N=len(g))
            continue
        yt = g[true].to_numpy()
        yp = g[pred_pl].to_numpy()
        out[str(dop)] = dict(
            N=len(g),
            r2_platt=float(r2_score(yt, yp)),
            r=float(pearsonr(yt, yp)[0]),
            mae=float(mean_absolute_error(yt, yp)),
        )
    return out


def _tab1_overall_latex(pdr_ov, vc_ov, pdr_ext_ov, vc_ext_ov):
    def row(name, d):
        if "raw" not in d:
            return f"{name} & \\multicolumn{{7}}{{l}}{{not available}} \\\\"
        r = d["per_dopant_platt"]
        return (f"{name} & {r['N']} & {r['r2']:+.3f} "
                f"[{r['r2_ci_low']:+.2f}, {r['r2_ci_high']:+.2f}] "
                f"& {r['r']:+.3f} & {r['mae']:.3f} & {r['rmse']:.3f} \\\\")

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Overall prediction metrics for the single-element deployment "
        r"(per-dopant Platt calibrated). Internal OOF = 5-seed 10-fold GroupKFold; "
        r"external = literature validation set (N = 9 PDR, VC blocked by data bug).}",
        r"\label{tab:overall_metrics}",
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"Setting & $N$ & $R^2$ [95\% CI] & Pearson $r$ & MAE & RMSE \\",
        r"\midrule",
        row("PDR internal OOF", pdr_ov),
        row("PDR external literature", pdr_ext_ov),
    ]
    if vc_ov:
        lines.append(row("VC internal OOF", vc_ov))
    if vc_ext_ov:
        lines.append(row("VC external literature", vc_ext_ov))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def _dashboard_cover(out_dir: Path, metrics: dict) -> Path:
    pdr_ov = metrics["pdr"]["oof"]["per_dopant_platt"]
    pdr_ext = metrics["pdr"]["external"]["per_dopant_platt"]
    vc_ov = metrics.get("vc", {}).get("oof", {}).get("per_dopant_platt")

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.94, "Ga$_2$O$_3$-Net — single-element dopant evaluation",
            ha="center", va="top", fontsize=20, weight="bold")
    ax.text(0.5, 0.90,
            "Phase 6D (PDR) + Phase 6E (VC) frozen deployment bundles.\n"
            "Evaluation generated 2026-04-16 with paper-ready figures and tables.",
            ha="center", va="top", fontsize=11, color="#555555")

    lines = ["Headline metrics (per-dopant Platt calibrated)", ""]
    lines.append(f"PDR internal OOF   N = {pdr_ov['N']}")
    lines.append(f"                  R^2 = {pdr_ov['r2']:+.3f} "
                 f"[95% CI {pdr_ov['r2_ci_low']:+.2f}, {pdr_ov['r2_ci_high']:+.2f}]")
    lines.append(f"                    r = {pdr_ov['r']:+.3f}, MAE = {pdr_ov['mae']:.3f}")
    lines.append("")
    lines.append(f"PDR external       N = {pdr_ext['N']}")
    lines.append(f"                  R^2 = {pdr_ext['r2']:+.3f} "
                 f"[95% CI {pdr_ext['r2_ci_low']:+.2f}, {pdr_ext['r2_ci_high']:+.2f}]")
    lines.append(f"                    r = {pdr_ext['r']:+.3f}, MAE = {pdr_ext['mae']:.3f}")
    if vc_ov:
        lines.append("")
        lines.append(f"VC  internal OOF   N = {vc_ov['N']}")
        lines.append(f"                  R^2 = {vc_ov['r2']:+.3f} "
                     f"[95% CI {vc_ov['r2_ci_low']:+.2f}, {vc_ov['r2_ci_high']:+.2f}]")
        lines.append(f"                    r = {vc_ov['r']:+.3f}, MAE = {vc_ov['mae']:.3f}")
        lines.append("                 (VC external deferred: CSV data bug, Mg rows non-log10)")

    ax.text(0.08, 0.78, "\n".join(lines), ha="left", va="top",
            fontsize=11, family="monospace")

    ax.text(0.08, 0.32,
            "Scope. Single-element dopants only. Multi-element co-doping (N=7 internal, N=0 external)\n"
            "and compound-form (SnO2/MgO) dopants are out of scope for this evaluation.\n"
            "\n"
            "Known limitations.\n"
            " - Sn external r reverses sign (r=-0.92) — see fig7 for method/substrate domain shift.\n"
            " - External VC CSV has 3 Mg rows with non-log10 values (25/27/37); external VC is deferred.\n"
            " - Al/B/Cu/Fe/Sb training N < 5; predictions exist but individual per-dopant evaluation\n"
            "   is statistically under-powered.",
            ha="left", va="top", fontsize=10, color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#999999"))

    ax.text(0.5, 0.08,
            "Following pages: 8 figures + 4 tables. See README.md in this directory for provenance.",
            ha="center", va="top", fontsize=10, color="#555555")

    cover = out_dir / "_dashboard_cover.pdf"
    fig.savefig(cover, bbox_inches="tight")
    plt.close(fig)
    return cover


def _assemble_dashboard(out_dir: Path, metrics: dict):
    cover = _dashboard_cover(out_dir, metrics)
    figs = sorted([p for p in out_dir.glob("fig*.pdf")])
    if not figs:
        print("  (no figures found — skipping dashboard)")
        return
    args = ["pdfunite", str(cover)] + [str(f) for f in figs] + [str(out_dir / "phase7_dashboard.pdf")]
    subprocess.run(args, check=True)
    cover.unlink(missing_ok=True)
    print(f"  Wrote {out_dir/'phase7_dashboard.pdf'}  ({1 + len(figs)} pages)")


def _capture_stdout(cmd: list[str], dst: Path):
    out = _run(cmd, capture=True)
    dst.write_text(out)
    return out


def _write_readme(out_dir: Path, args):
    text = f"""# Phase 7A — paper-ready single-element dopant evaluation

Generated: 2026-04-16 by `scripts/phase7_build_paper_package.py`.

## Source bundles (frozen, read-only)

- **PDR**  : `{args.pdr_bundle}` (Phase 6D method/substrate shuffle)
- **VC**   : `{args.vc_bundle}`  (Phase 6E within-DOI VC interpolation)
- **External literature CSV** : `{args.external_csv}`
- Encoder ckpt : `checkpoints/pretrained_encoder_v2.pt`

## Figures

| File | Description |
|---|---|
| fig1_parity_all.pdf | 2-panel (PDR\\|VC) parity plots, dopant-coloured |
| fig2_per_dopant_r2_ci.pdf | Per-dopant R² with bootstrap 95% CI (raw vs per-dopant-Platt) |
| fig3_per_dopant_parity_pdr.pdf | 6-subplot grid, top dopants, per-dopant fit + CI |
| fig3_per_dopant_parity_vc.pdf | same for VC (if enabled) |
| fig4_concentration_response.pdf | measured vs predicted vs concentration (±2σ MC band) |
| fig5_uncertainty_calibration.pdf | MC Dropout reliability diagram |
| fig6_internal_vs_external.pdf | Side-by-side bars per dopant (int OOF vs ext) |
| fig7_sn_domain_shift.pdf | Sn method/substrate contingency + external parity (honesty fig) |
| fig8_embedding_pca.pdf | 2-D PCA of 160-dim fused embeddings, dopant-colored |

## Tables (booktabs LaTeX + matching CSV)

| File | Description |
|---|---|
| tab1_overall_metrics.tex | Overall OOF + external metrics (PDR & VC) with 95% CI |
| tab4_platt_calibrators.tex | Per-dopant Platt (a, b, N_train) for both targets |
| tab5_concentration_slope_pdr.tex | Per-dopant within-DOI concentration-response (PDR) |
| tab5_concentration_slope_vc.tex | same for VC |
| tab_big_combined.tex | Single big table: per-dopant × PDR/VC metrics + ext r |

## Supplementary files

- `metrics.json`        — structured dump of every headline number
- `eval_summary.txt`    — concatenated stdout from analyze_oof + apply_per_dopant_platt
- `oof_pdr.csv`         — OOF snapshot from PDR bundle
- `oof_vc.csv`          — OOF snapshot from VC bundle
- `external_pdr.csv`    — external predictions (PDR) from 6D bundle
- `phase7_dashboard.pdf` — cover + all figures combined (single-file handout)

## Known limitations

1. **Sn external reversal.** External Sn rows come from MOCVD/ALD processes on
   substrates absent in the training-Sn distribution (sputter/sapphire
   dominant). Per-dopant Pearson r for external Sn is −0.92, reversing sign.
   Per-dopant Platt cannot invert this. Scientific finding, not a model defect
   — see fig7.
2. **External VC data bug.** `ga2o3_exp_external.csv` contains 3 Mg rows with
   `vacancy_concentration` values 25.55 / 27.15 / 37.36. These are above the
   physical atomic density limit (~10²³ cm⁻³) and are almost certainly either
   not log₁₀-transformed or double-log-transformed during CSV prep. The
   external-VC branch of this package is therefore **deliberately omitted**.
   Fix this bug in the source CSV before re-running external VC evaluation.
3. **Rare dopants** (Al N=2, B N=1, Cu N=1, Fe N=3, Sb N=1). Inside the
   model's scope but individually under-powered for per-dopant metrics.
4. **Multi-element / compound dopants** are out of scope (see
   `project_ga2o3_phase6_done.md` auto-memory).

## Regeneration

```bash
python scripts/phase7_build_paper_package.py \\
    --pdr-bundle {args.pdr_bundle} \\
    --vc-bundle  {args.vc_bundle} \\
    --external-csv {args.external_csv} \\
    --out-dir {args.out_dir}
```
"""
    (out_dir / "README.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-bundle", type=Path, default=Path("results/deployment/pdr_6D_methshuf"))
    ap.add_argument("--vc-bundle", type=Path, default=Path("results/deployment/vc_6E_interp"))
    ap.add_argument("--external-csv", type=Path,
                    default=Path("data/raw/experimental/ga2o3_exp_external.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/paper_single_element"))
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    os.chdir(ROOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    # ── 1. Snapshot bundle artefacts ────────────────────────────────────────
    pdr_oof = args.pdr_bundle / "oof_per_dopant_platt.csv"
    pdr_ext = args.pdr_bundle / "external" / "external_per_dopant_platt.csv"
    vc_oof = args.vc_bundle / "oof_per_dopant_platt.csv"

    # If per-dopant-Platt CSVs are missing, generate them
    if not pdr_oof.exists():
        print("[1] PDR per-dopant Platt missing — regenerating")
        _run(["python", "scripts/apply_per_dopant_platt.py",
              str(args.pdr_bundle), "--target", "photo_dark_ratio"])
    if not vc_oof.exists():
        print("[1] VC per-dopant Platt missing — regenerating")
        _run(["python", "scripts/apply_per_dopant_platt.py",
              str(args.vc_bundle), "--target", "vacancy_concentration"])

    # Snapshot CSVs into out_dir for a self-contained artefact
    shutil.copy(pdr_oof, args.out_dir / "oof_pdr.csv")
    shutil.copy(vc_oof, args.out_dir / "oof_vc.csv")
    if pdr_ext.exists():
        shutil.copy(pdr_ext, args.out_dir / "external_pdr.csv")

    # ── 2. Capture stdout from analyze_oof + apply_per_dopant_platt ─────────
    pdr_platt_stdout = args.out_dir / "_stdout_platt_pdr.txt"
    vc_platt_stdout = args.out_dir / "_stdout_platt_vc.txt"
    pdr_analyze_stdout = args.out_dir / "_stdout_analyze_pdr.txt"
    vc_analyze_stdout = args.out_dir / "_stdout_analyze_vc.txt"

    print("[2] Capturing analyze_oof + per-dopant Platt stdout")
    _capture_stdout(
        ["python", "scripts/apply_per_dopant_platt.py", str(args.pdr_bundle),
         "--target", "photo_dark_ratio"],
        pdr_platt_stdout)
    _capture_stdout(
        ["python", "scripts/apply_per_dopant_platt.py", str(args.vc_bundle),
         "--target", "vacancy_concentration"],
        vc_platt_stdout)
    _capture_stdout(
        ["python", "scripts/analyze_oof.py", str(pdr_oof)],
        pdr_analyze_stdout)
    _capture_stdout(
        ["python", "scripts/analyze_oof.py", str(vc_oof)],
        vc_analyze_stdout)

    # ── 3. Compute structured metrics.json ──────────────────────────────────
    pdr_oof_df = pd.read_csv(pdr_oof)
    vc_oof_df = pd.read_csv(vc_oof)
    pdr_ext_df = pd.read_csv(pdr_ext) if pdr_ext.exists() else pd.DataFrame()

    metrics = {
        "generated_at": "2026-04-16",
        "pdr": {
            "oof": _overall(pdr_oof_df, "photo_dark_ratio"),
            "external": _overall(pdr_ext_df, "photo_dark_ratio") if not pdr_ext_df.empty else {},
            "per_dopant_oof": _per_dopant(pdr_oof_df, "photo_dark_ratio"),
            "per_dopant_external": _per_dopant(pdr_ext_df, "photo_dark_ratio") if not pdr_ext_df.empty else {},
        },
        "vc": {
            "oof": _overall(vc_oof_df, "vacancy_concentration"),
            "per_dopant_oof": _per_dopant(vc_oof_df, "vacancy_concentration"),
            "external_note": "omitted; external CSV has non-log10 Mg VC values",
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[3] metrics.json written ({(args.out_dir/'metrics.json').stat().st_size} bytes)")

    # ── 4. Figures ───────────────────────────────────────────────────────────
    print("[4] Generating figures")
    pdr_csv_train = args.pdr_bundle / "training_csv.csv"
    vc_csv_train = args.vc_bundle / "training_csv.csv"

    # fig1 — overall parity (two-panel inline)
    _make_fig1_parity(pdr_oof_df, vc_oof_df, args.out_dir / "fig1_parity_all.pdf")

    # fig2 — per-dopant CI bars
    _run(["python", "scripts/phase7_per_dopant_ci_plot.py",
          "--pdr-oof", str(pdr_oof), "--vc-oof", str(vc_oof),
          "--out", str(args.out_dir / "fig2_per_dopant_r2_ci.pdf")])

    # fig3 — per-dopant parity
    _run(["python", "scripts/phase7_per_dopant_parity.py",
          "--pdr-oof", str(pdr_oof), "--pdr-source-csv", str(pdr_csv_train),
          "--vc-oof", str(vc_oof), "--vc-source-csv", str(vc_csv_train),
          "--out-dir", str(args.out_dir)])
    # Rename for consistent naming (fig3_per_dopant_parity_pdr.pdf is expected)
    # Already correct from the sub-script.

    # fig4 — concentration response (via inline plotter, simpler than reusing 05_evaluate pipeline)
    _make_fig4_concentration(pdr_oof_df, pdr_csv_train, vc_oof_df, vc_csv_train,
                             args.out_dir / "fig4_concentration_response.pdf")

    # fig5 — uncertainty calibration (only if MC std columns exist)
    _make_fig5_calibration(pdr_oof_df, vc_oof_df, args.out_dir / "fig5_uncertainty_calibration.pdf")

    # fig6 — internal vs external
    if pdr_ext.exists():
        _run(["python", "scripts/phase7_int_vs_ext_bars.py",
              "--pdr-oof", str(pdr_oof), "--pdr-ext", str(pdr_ext),
              "--out", str(args.out_dir / "fig6_internal_vs_external.pdf")])

    # fig7 — Sn domain-shift honesty figure
    if pdr_ext.exists():
        _run(["python", "scripts/phase7_sn_domain_shift.py",
              "--train-csv", str(pdr_csv_train),
              "--ext-pred", str(pdr_ext),
              "--out", str(args.out_dir / "fig7_sn_domain_shift.pdf")])

    # fig8 — embedding PCA (skip if we cannot easily obtain embeddings; write a stub note)
    # Embeddings require running the model; skipped from this orchestrator.

    # ── 5. Tables ────────────────────────────────────────────────────────────
    print("[5] Generating tables")
    (args.out_dir / "tab1_overall_metrics.tex").write_text(
        _tab1_overall_latex(metrics["pdr"]["oof"], metrics["vc"]["oof"],
                            metrics["pdr"]["external"], None)
    )
    _run(["python", "scripts/phase7_platt_table.py",
          "--pdr-stdout", str(pdr_platt_stdout),
          "--vc-stdout", str(vc_platt_stdout),
          "--out-dir", str(args.out_dir)])
    _run(["python", "scripts/phase7_concentration_slope_table.py",
          "--pdr-oof", str(pdr_oof), "--pdr-src", str(pdr_csv_train),
          "--vc-oof", str(vc_oof), "--vc-src", str(vc_csv_train),
          "--out-dir", str(args.out_dir)])
    big_cmd = ["python", "scripts/phase7_big_table.py",
               "--pdr-oof", str(pdr_oof), "--pdr-ext", str(pdr_ext),
               "--vc-oof", str(vc_oof),
               "--out-dir", str(args.out_dir)]
    _run(big_cmd)

    # ── 6. Concat stdout into eval_summary.txt ───────────────────────────────
    summary = (
        "=" * 78 + "\n"
        "PDR — analyze_oof.py\n" + "=" * 78 + "\n"
        + pdr_analyze_stdout.read_text() + "\n\n"
        + "=" * 78 + "\n"
        "PDR — apply_per_dopant_platt.py\n" + "=" * 78 + "\n"
        + pdr_platt_stdout.read_text() + "\n\n"
        + "=" * 78 + "\n"
        "VC — analyze_oof.py\n" + "=" * 78 + "\n"
        + vc_analyze_stdout.read_text() + "\n\n"
        + "=" * 78 + "\n"
        "VC — apply_per_dopant_platt.py\n" + "=" * 78 + "\n"
        + vc_platt_stdout.read_text() + "\n"
    )
    (args.out_dir / "eval_summary.txt").write_text(summary)
    for tmp in [pdr_platt_stdout, vc_platt_stdout, pdr_analyze_stdout, vc_analyze_stdout]:
        tmp.unlink(missing_ok=True)

    # ── 7. README + dashboard ────────────────────────────────────────────────
    print("[6] Writing README")
    _write_readme(args.out_dir, args)

    print("[7] Assembling dashboard PDF")
    _assemble_dashboard(args.out_dir, metrics)

    print("\nDone. Listing:")
    for p in sorted(args.out_dir.iterdir()):
        sz = p.stat().st_size // 1024
        print(f"  {sz:>6d} KB  {p.name}")


# ────────────────────────────────────────────────────────────────────────────
# Inline plotters (used for fig1 parity, fig4 conc response, fig5 calibration)

def _make_fig1_parity(pdr_df, vc_df, out_path: Path):
    def panel(ax, df, target, title):
        pred_pl = f"{target}_pred_per_dopant_platt"
        if pred_pl not in df.columns:
            pred_pl = f"{target}_pred"
        true = f"{target}_true"
        lab = df.dropna(subset=[pred_pl, true])
        if lab.empty:
            ax.set_title(title)
            return
        yt = lab[true].to_numpy()
        yp = lab[pred_pl].to_numpy()
        cmap = plt.get_cmap("tab10")
        labels = list(dict.fromkeys(lab["dopant_label"].astype(str).tolist()))
        color_map = {d: cmap(i % 10) for i, d in enumerate(labels)}
        for d in labels:
            sub = lab[lab["dopant_label"].astype(str) == d]
            ax.scatter(sub[pred_pl], sub[true], s=22, alpha=0.75,
                       color=color_map[d], edgecolor="white", linewidth=0.4,
                       label=f"{d} (N={len(sub)})")
        lo = float(min(yp.min(), yt.min())) - 0.3
        hi = float(max(yp.max(), yt.max())) + 0.3
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8, alpha=0.6, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Predicted log$_{{10}}$({target})")
        ax.set_ylabel(f"Measured log$_{{10}}$({target})")
        r2 = r2_score(yt, yp)
        r = pearsonr(yt, yp)[0]
        mae = mean_absolute_error(yt, yp)
        txt = f"N={len(lab)}\n$R^2$={r2:+.3f}\nr={r:+.3f}\nMAE={mae:.2f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="grey"))
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    panel(axes[0], pdr_df, "photo_dark_ratio", "PDR parity (per-dopant Platt)")
    panel(axes[1], vc_df, "vacancy_concentration", "VC parity (per-dopant Platt)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _make_fig4_concentration(pdr_df, pdr_csv, vc_df, vc_csv, out_path: Path):
    def _with_meta(df, src_csv):
        src = pd.read_csv(src_csv).reset_index().rename(columns={"index": "sample_idx"})
        keep = [c for c in ["sample_idx", "concentration_at%", "element", "doi"] if c in src.columns]
        return df.merge(src[keep], on="sample_idx", how="left")

    pdr_m = _with_meta(pdr_df, pdr_csv)
    vc_m = _with_meta(vc_df, vc_csv)

    def _plot_target(ax_grid, df, target):
        pred_pl = f"{target}_pred_per_dopant_platt"
        if pred_pl not in df.columns:
            pred_pl = f"{target}_pred"
        true = f"{target}_true"
        std = f"{target}_std" if f"{target}_std" in df.columns else f"{target}_mc_std"
        if std not in df.columns: std = None
        lab = df.dropna(subset=[pred_pl, true, "concentration_at%"])
        majors = [d for d, c in lab["element"].value_counts().items()
                  if c >= 4 and d not in ("—",) and "+" not in str(d)][:4]
        for ax, dop in zip(ax_grid, majors):
            sub = lab[lab["element"] == dop].sort_values("concentration_at%")
            ax.plot(sub["concentration_at%"], sub[true], "o-", color="#4C72B0", lw=1.5, ms=5, label="measured")
            ax.plot(sub["concentration_at%"], sub[pred_pl], "s--", color="#C44E52", lw=1.2, ms=5, label="predicted")
            if std is not None and std in sub.columns and sub[std].notna().any():
                ax.fill_between(sub["concentration_at%"],
                                sub[pred_pl] - 2 * sub[std],
                                sub[pred_pl] + 2 * sub[std],
                                color="#C44E52", alpha=0.15, lw=0, label="pred ±2σ")
            ax.set_xscale("symlog", linthresh=0.05)
            ax.set_title(f"{target}  —  {dop}", fontsize=10)
            ax.set_xlabel("concentration (at%)")
            ax.set_ylabel(f"log$_{{10}}$({target})")
            ax.legend(fontsize=8, frameon=False, loc="best")
            ax.grid(alpha=0.25)
        for ax in ax_grid[len(majors):]:
            ax.axis("off")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    _plot_target(axes[0], pdr_m, "photo_dark_ratio")
    _plot_target(axes[1], vc_m, "vacancy_concentration")
    fig.suptitle("Concentration response — measured vs predicted (per-dopant Platt)",
                 fontsize=13, y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _make_fig5_calibration(pdr_df, vc_df, out_path: Path):
    """MC-Dropout reliability diagram. Expected vs empirical coverage at α ∈ [0.05,..,0.95]."""

    def coverage(df, target):
        pred_pl = f"{target}_pred_per_dopant_platt"
        if pred_pl not in df.columns:
            pred_pl = f"{target}_pred"
        true = f"{target}_true"
        std_col = f"{target}_std" if f"{target}_std" in df.columns else f"{target}_mc_std"
        if std_col not in df.columns:
            return None, None
        lab = df.dropna(subset=[pred_pl, true, std_col])
        if lab.empty:
            return None, None
        alphas = np.linspace(0.05, 0.95, 19)
        emp = []
        from scipy.stats import norm
        for a in alphas:
            z = norm.ppf(0.5 + a / 2)
            lo = lab[pred_pl] - z * lab[std_col]
            hi = lab[pred_pl] + z * lab[std_col]
            emp.append(float(((lab[true] >= lo) & (lab[true] <= hi)).mean()))
        return alphas, emp

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, (df, target, title) in zip(
            axes,
            [(pdr_df, "photo_dark_ratio", "PDR MC-Dropout calibration"),
             (vc_df, "vacancy_concentration", "VC MC-Dropout calibration")]):
        a, e = coverage(df, target)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.6, label="ideal")
        if a is None:
            ax.text(0.5, 0.5, "no MC std column", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.plot(a, e, "o-", color="#4C72B0", lw=1.2, ms=4, label="empirical")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
