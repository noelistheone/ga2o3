"""Driver for post-hoc RankRefine on a Phase X OOF bundle.

Loads OOF + meta, applies `src.inference.rank_refine.refine_oof`, then
re-runs the Tier-1A / Tier-1B / Tier-1C diagnostics on the *refined*
predictions WITHOUT refitting Platt (so the comparison is honest).

Usage:
    python scripts/run_rank_refine.py <bundle_dir> [<full_csv>] \
           [--target vacancy_concentration|photo_dark_ratio] \
           [--scope all|sputter] [--cross-class-weight 0.30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.inference.rank_refine import refine_oof  # noqa: E402
from scripts.eval_physics_diag_table import (  # noqa: E402
    load_oof_with_meta, fit_sputter_platt,
    tier1_acceptor_donor, tier1b_rare_element_direction,
    tier1c_unsupervised_element_direction, tier2_within_element_slope,
    df_to_md,
)


def _overall_metrics(df: pd.DataFrame, pred_col: str, true_col: str) -> dict:
    sub = df.dropna(subset=[pred_col, true_col])
    if len(sub) < 2:
        return dict(n=len(sub), r=float("nan"), r2=float("nan"), mae=float("nan"))
    y_t = sub[true_col].to_numpy()
    y_p = sub[pred_col].to_numpy()
    r = float(stats.pearsonr(y_t, y_p).statistic)
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(y_t - y_p)))
    return dict(n=len(sub), r=r, r2=r2, mae=mae)


def _summarize_tier(df_tier: pd.DataFrame, sign_col: str = "sign_match") -> dict:
    if df_tier is None or len(df_tier) == 0:
        return dict(ok=0, flip=0, total=0)
    vals = df_tier[sign_col].astype(str)
    return dict(
        ok=int(vals.str.contains("OK", case=False).sum()),
        flip=int(vals.str.contains("FLIP", case=True).sum()),
        total=int(len(df_tier)),
    )


def _summarize_dir(df_tier: pd.DataFrame, col: str) -> dict:
    if df_tier is None or len(df_tier) == 0:
        return dict(ok=0, flip=0, total=0)
    vals = df_tier[col].astype(str)
    return dict(
        ok=int((vals == "OK").sum()),
        flip=int(vals.str.contains("FLIP", case=True).sum()),
        total=int(len(df_tier)),
    )


def run(bundle_dir: Path, full_csv: str | None, target: str,
        scope: str = "all", cross_w: float = 0.30,
        sigma_floor: float = 0.20, min_dlogc: float = 0.10,
        sputter_only_refs: bool = True,
        unlabeled_only: bool = False, max_shift: float = 1.5,
        suffix: str = ""):
    print(f"\n{'='*78}")
    print(f"Bundle: {bundle_dir.name} | target={target} | scope={scope}")
    print(f"{'='*78}")

    df = load_oof_with_meta(bundle_dir, full_csv, target=target)
    slope, intercept = fit_sputter_platt(df, target=target)
    pred_col = f"{target}_pred"
    pred_platt_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    df[pred_platt_col] = df[pred_col] * slope + intercept

    if scope == "sputter":
        df = df[df.is_sputter].copy()

    # --- Apply RankRefine ---
    refined, diag = refine_oof(
        df, target=target,
        platt_slope=slope, platt_intercept=intercept,
        min_dlogc=min_dlogc, cross_class_weight=cross_w,
        sigma_reg_floor=sigma_floor, sputter_only_refs=sputter_only_refs,
        unlabeled_only=unlabeled_only, max_shift=max_shift,
    )
    refined_col = f"{target}_pred_refined"

    # --- Eval on labeled subset ---
    eval_base = df.dropna(subset=[true_col]).copy()
    eval_ref = refined.dropna(subset=[true_col]).copy()

    base_overall = _overall_metrics(eval_base, pred_platt_col, true_col)
    ref_overall = _overall_metrics(eval_ref, refined_col, true_col)

    # --- Tier-1A / Tier-1B / Tier-1C / Tier-2 on baseline ---
    t1a_b = tier1_acceptor_donor(eval_base, target=target)
    t1b_b = tier1b_rare_element_direction(eval_base, target=target)
    # Tier-1C uses ALL rows (incl. unlabeled); for refined use refined too
    t1c_b = tier1c_unsupervised_element_direction(df, target=target)
    t2_b = tier2_within_element_slope(eval_base, target=target)

    # --- Tier on refined: temporarily swap pred_platt column ---
    def _swap_pred(d: pd.DataFrame) -> pd.DataFrame:
        d2 = d.copy()
        d2[pred_platt_col] = d2[refined_col]
        return d2
    eval_ref_s = _swap_pred(eval_ref)
    refined_s = _swap_pred(refined)
    t1a_r = tier1_acceptor_donor(eval_ref_s, target=target)
    t1b_r = tier1b_rare_element_direction(eval_ref_s, target=target)
    t1c_r = tier1c_unsupervised_element_direction(refined_s, target=target)
    t2_r = tier2_within_element_slope(eval_ref_s, target=target)

    # --- Print comparison ---
    print(f"\n[Overall, scope={scope}, labeled n={base_overall['n']}]  "
          f"BASELINE  →  REFINED")
    print(f"  R²:  {base_overall['r2']:+.3f}  →  {ref_overall['r2']:+.3f}   "
          f"(Δ {ref_overall['r2']-base_overall['r2']:+.3f})")
    print(f"  r:   {base_overall['r']:+.3f}  →  {ref_overall['r']:+.3f}   "
          f"(Δ {ref_overall['r']-base_overall['r']:+.3f})")
    print(f"  MAE: {base_overall['mae']:.3f}  →  {ref_overall['mae']:.3f}   "
          f"(Δ {ref_overall['mae']-base_overall['mae']:+.3f})")

    s_a_b = _summarize_tier(t1a_b)
    s_a_r = _summarize_tier(t1a_r)
    s_b_b = _summarize_dir(t1b_b, "valence_dir")
    s_b_r = _summarize_dir(t1b_r, "valence_dir")
    s_c_b = _summarize_dir(t1c_b, "verdict")
    s_c_r = _summarize_dir(t1c_r, "verdict")

    print(f"\n[Tier-1A] (per-elem slope sign, N≥3)  BASELINE: "
          f"OK={s_a_b['ok']}/{s_a_b['total']} FLIP={s_a_b['flip']}  "
          f"→  REFINED: OK={s_a_r['ok']}/{s_a_r['total']} FLIP={s_a_r['flip']}")
    print(f"[Tier-1B] (labeled-elem direction)     BASELINE: "
          f"OK={s_b_b['ok']}/{s_b_b['total']} FLIP={s_b_b['flip']}  "
          f"→  REFINED: OK={s_b_r['ok']}/{s_b_r['total']} FLIP={s_b_r['flip']}")
    print(f"[Tier-1C] (unsupervised-elem direction) BASELINE: "
          f"OK={s_c_b['ok']}/{s_c_b['total']} FLIP={s_c_b['flip']}  "
          f"→  REFINED: OK={s_c_r['ok']}/{s_c_r['total']} FLIP={s_c_r['flip']}")

    # --- Per-element pearson r (Tier-2) ---
    if len(t2_b) > 0 and len(t2_r) > 0:
        merged = t2_b[["element", "n", "pearson_r"]].merge(
            t2_r[["element", "pearson_r"]], on="element",
            suffixes=("_base", "_ref"))
        merged["delta"] = merged["pearson_r_ref"] - merged["pearson_r_base"]
        print("\n[Tier-2 per-element Pearson r]")
        print(merged.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # --- Per-element direction flip detail (Tier-1B + 1C) ---
    if len(t1c_b) > 0 and len(t1c_r) > 0:
        c_merge = t1c_b[["element", "n_total", "n_labeled", "verdict"]].merge(
            t1c_r[["element", "verdict"]], on="element",
            suffixes=("_base", "_ref"))
        flips_changed = c_merge[c_merge.verdict_base != c_merge.verdict_ref]
        if len(flips_changed) > 0:
            print("\n[Tier-1C verdict changes]")
            print(flips_changed.to_string(index=False))
        else:
            print("\n[Tier-1C verdicts unchanged]")

    # --- Save refined OOF + diag ---
    out_dir = bundle_dir
    sfx = (suffix or "")
    refined_path = out_dir / f"oof_refined_{target}{sfx}.csv"
    refined.to_csv(refined_path, index=False)
    diag_path = out_dir / f"rank_refine_diag_{target}{sfx}.csv"
    diag.to_csv(diag_path, index=False)
    print(f"\n[Saved] {refined_path}")
    print(f"[Saved] {diag_path}")

    # --- Save markdown comparison ---
    md = []
    md.append(f"# RankRefine Comparison — {bundle_dir.name} ({target}, scope={scope})\n")
    md.append(f"cross_class_weight = {cross_w}, sigma_reg_floor = {sigma_floor}, "
              f"min_dlogc = {min_dlogc}, sputter_only_refs = {sputter_only_refs}\n\n")
    md.append("## Overall (refined uses no Platt re-fit)\n\n")
    md.append("| Metric | BASELINE | REFINED | Δ |\n|---|---:|---:|---:|\n")
    md.append(f"| R² | {base_overall['r2']:+.3f} | {ref_overall['r2']:+.3f} | "
              f"{ref_overall['r2']-base_overall['r2']:+.3f} |\n")
    md.append(f"| r | {base_overall['r']:+.3f} | {ref_overall['r']:+.3f} | "
              f"{ref_overall['r']-base_overall['r']:+.3f} |\n")
    md.append(f"| MAE | {base_overall['mae']:.3f} | {ref_overall['mae']:.3f} | "
              f"{ref_overall['mae']-base_overall['mae']:+.3f} |\n\n")
    md.append("## Tier counts\n\n")
    md.append(f"| Tier | BASELINE OK / FLIP / N | REFINED OK / FLIP / N |\n|---|---|---|\n")
    md.append(f"| 1A (slope sign) | {s_a_b['ok']}/{s_a_b['flip']}/{s_a_b['total']} "
              f"| {s_a_r['ok']}/{s_a_r['flip']}/{s_a_r['total']} |\n")
    md.append(f"| 1B (labeled dir) | {s_b_b['ok']}/{s_b_b['flip']}/{s_b_b['total']} "
              f"| {s_b_r['ok']}/{s_b_r['flip']}/{s_b_r['total']} |\n")
    md.append(f"| 1C (unsupervised dir) | {s_c_b['ok']}/{s_c_b['flip']}/{s_c_b['total']} "
              f"| {s_c_r['ok']}/{s_c_r['flip']}/{s_c_r['total']} |\n\n")
    md.append("## Tier-1A baseline\n\n")
    md.append(df_to_md(t1a_b))
    md.append("\n## Tier-1A refined\n\n")
    md.append(df_to_md(t1a_r))
    md.append("\n## Tier-1B baseline\n\n")
    md.append(df_to_md(t1b_b))
    md.append("\n## Tier-1B refined\n\n")
    md.append(df_to_md(t1b_r))
    md.append("\n## Tier-1C baseline\n\n")
    md.append(df_to_md(t1c_b))
    md.append("\n## Tier-1C refined\n\n")
    md.append(df_to_md(t1c_r))
    md_path = out_dir / f"rank_refine_compare_{target}{sfx}.md"
    md_path.write_text("\n".join(md))
    print(f"[Saved] {md_path}")

    # Summary dict for caller
    return dict(
        bundle=bundle_dir.name, target=target,
        base_r2=base_overall["r2"], ref_r2=ref_overall["r2"],
        base_r=base_overall["r"], ref_r=ref_overall["r"],
        base_mae=base_overall["mae"], ref_mae=ref_overall["mae"],
        t1a_base_flip=s_a_b["flip"], t1a_ref_flip=s_a_r["flip"],
        t1b_base_flip=s_b_b["flip"], t1b_ref_flip=s_b_r["flip"],
        t1c_base_flip=s_c_b["flip"], t1c_ref_flip=s_c_r["flip"],
        n_refined=int((diag.shape[0] if diag is not None else 0)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle_dir")
    p.add_argument("full_csv", nargs="?", default=None)
    p.add_argument("--target", default="vacancy_concentration",
                   choices=["vacancy_concentration", "photo_dark_ratio"])
    p.add_argument("--scope", default="all", choices=["all", "sputter"])
    p.add_argument("--cross-class-weight", type=float, default=0.30)
    p.add_argument("--sigma-floor", type=float, default=0.20)
    p.add_argument("--min-dlogc", type=float, default=0.10)
    p.add_argument("--no-sputter-only-refs", action="store_true",
                   help="Use ALL labeled rows as refs, not just sputter")
    p.add_argument("--unlabeled-only", action="store_true",
                   help="Only refine rows with NO true label (preserves parity r)")
    p.add_argument("--max-shift", type=float, default=1.5,
                   help="Cap |y_rank - y_reg| in log10 units (limits outlier damage)")
    p.add_argument("--suffix", default="",
                   help="Append to output filenames to keep multiple runs side-by-side")
    args = p.parse_args()
    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = PROJ / bundle
    run(bundle, args.full_csv, args.target, scope=args.scope,
        cross_w=args.cross_class_weight, sigma_floor=args.sigma_floor,
        min_dlogc=args.min_dlogc,
        sputter_only_refs=not args.no_sputter_only_refs,
        unlabeled_only=args.unlabeled_only, max_shift=args.max_shift,
        suffix=args.suffix)


if __name__ == "__main__":
    main()
