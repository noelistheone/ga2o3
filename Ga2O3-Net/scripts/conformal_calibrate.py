"""
conformal_calibrate.py — split-conformal calibration of prediction intervals
on the frontier OOF predictions, then re-evaluate the lab target regime
using the calibrated quantile.

Method (split conformal regression, Lei & Wasserman 2014):
  - Calibration set: OOF rows with valid label.
  - Normalized conformity score:  s_i = |y_true_i - y_pred_platt_i| / std_i
  - Finite-sample conformal quantile at level α (default 0.80):
        k = ⌈ (n + 1) * α ⌉   (clipped to n)
        q_α = k-th smallest s_i
  - Calibrated 80% PI:  pred_platt ± q_0.80 · std

Three stratification modes are produced:
  * GLOBAL    : single q over full OOF
  * TIER      : per relaxation tier (0..4) using lab-regime tags
  * ELEM_TIER : per (element × tier) for tier 0 only (lab-relevant cell)

For each mode we report empirical coverage on the lab tier-0 slice using
the calibrated q, alongside the naive Gaussian z=1.282 baseline. Conformal
guarantees coverage ≥ α on exchangeable data; the empirical gap quantifies
distribution shift between full OOF and the lab tier.

Usage:
    python scripts/conformal_calibrate.py
    python scripts/conformal_calibrate.py --level 0.80 --target VC

Output:
    results/lab_regime_eval/conformal_quantiles.json
    results/lab_regime_eval/conformal_coverage.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from scripts.eval_lab_target_regime import (
    DEFAULT_BUNDLES, LAB_DOPANTS, PI80_Z, TARGET_COL,
    _build_dataset, _load_bundle, tag_proximity,
)


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Split-conformal finite-sample quantile.

    Returns a multiplier such that, on exchangeable test data,
        P(|y - μ| ≤ q · σ) ≥ level.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * level))
    k = min(k, n)
    return float(np.sort(s)[k - 1])


def empirical_coverage(residuals: np.ndarray, half_width: np.ndarray) -> float:
    r = np.asarray(residuals, dtype=float)
    h = np.asarray(half_width, dtype=float)
    mask = np.isfinite(r) & np.isfinite(h)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(r[mask] <= h[mask]))


def calibrate_target(
    target: str,
    bundle_dir: Path,
    level: float,
) -> dict:
    target_col = TARGET_COL[target]
    print(f"\n=== {target}: {bundle_dir.name} ===")

    oof, fu = _load_bundle(bundle_dir)
    ds = _build_dataset(fu)
    proc = ds.get_process_array()

    src_cols = [c for c in
                ("element", "method", "atmosphere", "temperature_C")
                if c in ds.df.columns]
    src = ds.df[src_cols].copy().reset_index(drop=True)
    src["sample_idx"] = np.arange(len(src), dtype=int)

    merged = oof.merge(src, on="sample_idx", how="left")
    tagged = tag_proximity(merged, proc[merged["sample_idx"].to_numpy()])

    # Drop rows without valid label or std
    pred_col = f"{target_col}_pred_platt"
    std_col  = f"{target_col}_std"
    true_col = f"{target_col}_true"
    keep = tagged.dropna(subset=[true_col, pred_col, std_col]).copy()

    keep["abs_resid"] = np.abs(keep[true_col] - keep[pred_col])
    keep["std_clip"]  = np.maximum(keep[std_col].to_numpy(), 1e-9)
    keep["s_norm"]    = keep["abs_resid"] / keep["std_clip"]

    # ── GLOBAL conformal ─────────────────────────────────────────────────
    q_global_norm = conformal_quantile(keep["s_norm"].to_numpy(), level)
    q_global_abs  = conformal_quantile(keep["abs_resid"].to_numpy(), level)

    # ── PER-TIER conformal ───────────────────────────────────────────────
    per_tier_q: dict = {}
    for tier in sorted(keep["tier"].unique()):
        sub = keep[keep["tier"] == tier]
        if len(sub) >= 5:
            per_tier_q[int(tier)] = {
                "n_calib": int(len(sub)),
                "q_norm":  conformal_quantile(sub["s_norm"].to_numpy(),  level),
                "q_abs":   conformal_quantile(sub["abs_resid"].to_numpy(), level),
            }

    # ── PER-ELEMENT (tier-0 only) ────────────────────────────────────────
    per_elem_q: dict = {}
    for elem in LAB_DOPANTS:
        sub = keep[(keep["tier"] == 0) & (keep["element"] == elem)]
        if len(sub) >= 5:
            per_elem_q[elem] = {
                "n_calib": int(len(sub)),
                "q_norm":  conformal_quantile(sub["s_norm"].to_numpy(),  level),
                "q_abs":   conformal_quantile(sub["abs_resid"].to_numpy(), level),
            }
        else:
            per_elem_q[elem] = {"n_calib": int(len(sub)), "q_norm": None, "q_abs": None}

    # ── Empirical coverage on tier 0, per element ───────────────────────
    tier0 = keep[keep["tier"] == 0]
    rows = []

    def cov_block(sub: pd.DataFrame, q_n: float | None, q_a: float | None) -> dict:
        if len(sub) == 0:
            return dict(n=0, naive_z=float("nan"),
                        cal_norm_global=float("nan"),
                        cal_abs_global=float("nan"))
        r = sub["abs_resid"].to_numpy()
        s = sub["std_clip"].to_numpy()
        return dict(
            n=int(len(sub)),
            naive_z=empirical_coverage(r, PI80_Z * s),
            cal_norm_global=empirical_coverage(r, q_global_norm * s),
            cal_abs_global =empirical_coverage(r, np.full_like(s, q_global_abs)),
            cal_norm_local =(empirical_coverage(r, q_n * s) if q_n is not None else float("nan")),
            cal_abs_local  =(empirical_coverage(r, np.full_like(s, q_a)) if q_a is not None else float("nan")),
        )

    rows.append({"slice": "tier_0_all", **cov_block(tier0, q_global_norm, q_global_abs)})
    for elem in LAB_DOPANTS:
        sub = tier0[tier0["element"] == elem]
        q_pair = per_elem_q.get(elem, {})
        rows.append({
            "slice": f"tier_0_{elem}",
            **cov_block(sub, q_pair.get("q_norm"), q_pair.get("q_abs"))
        })

    # Width comparison (mean PI80 width before / after global conformal)
    width_naive = float(np.mean(2.0 * PI80_Z * tier0["std_clip"]))
    width_cal_norm = float(np.mean(2.0 * q_global_norm * tier0["std_clip"]))

    print(f"  n_calib (full OOF) = {len(keep)}")
    print(f"  q_global_norm @{level:.2f} = {q_global_norm:.4f}  "
          f"(naive z = {PI80_Z:.4f})")
    print(f"  q_global_abs  @{level:.2f} = {q_global_abs:.4f}")
    print(f"  tier-0 PI80 width: naive={width_naive:.3f}  "
          f"calibrated_norm={width_cal_norm:.3f}")
    for r in rows:
        print(f"  {r['slice']:<16s} N={r['n']:3d}  "
              f"cov_naive={r['naive_z']:.2f}  "
              f"cov_cal_norm_global={r['cal_norm_global']:.2f}  "
              f"cov_cal_norm_local={r['cal_norm_local']:.2f}")

    return dict(
        target=target,
        bundle=str(bundle_dir.relative_to(PROJ)),
        level=level,
        n_calib=int(len(keep)),
        q_global_norm=q_global_norm,
        q_global_abs=q_global_abs,
        per_tier_q=per_tier_q,
        per_elem_tier0_q=per_elem_q,
        coverage=rows,
        widths=dict(
            naive_z=PI80_Z,
            tier0_width_naive=width_naive,
            tier0_width_cal_norm_global=width_cal_norm,
        ),
    )


def write_markdown(report: dict, path: Path) -> None:
    L = []
    L.append("# Conformal PI Calibration — Frontier Snapshot")
    L.append("")
    L.append(f"Coverage level: **{report['level']:.2f}** (split conformal, normalized score)")
    L.append("")
    L.append("Naive `z = 1.2816` is the Gaussian 80% half-width multiplier. "
             "Conformal `q_global_norm` replaces it after fitting on the full OOF "
             "calibration set. Coverage gap on tier 0 quantifies how much the lab "
             "regime distribution shifts vs. literature pool.")
    L.append("")

    for blk in report["targets"]:
        L.append(f"## {blk['target']} — `{blk['bundle']}`")
        L.append("")
        L.append(f"Calibration set: **{blk['n_calib']} rows** (full OOF with valid labels).")
        L.append(f"")
        L.append(f"- `naive z`           = {PI80_Z:.4f}")
        L.append(f"- `q_global_norm`     = {blk['q_global_norm']:.4f}  "
                 f"(multiplier on σ, like z)")
        L.append(f"- `q_global_abs`      = {blk['q_global_abs']:.4f}  "
                 f"(absolute half-width in target units)")
        L.append(f"- tier-0 PI80 width: naive = {blk['widths']['tier0_width_naive']:.3f}, "
                 f"calibrated normalized = {blk['widths']['tier0_width_cal_norm_global']:.3f}")
        L.append("")
        L.append("### Empirical coverage on tier 0 (lab regime)")
        L.append("")
        L.append("| slice | N | naive z | cal norm global | cal abs global | cal norm local | cal abs local |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in blk["coverage"]:
            def fmt(x):
                return "—" if x != x else f"{x:.2f}"  # NaN check
            L.append(
                f"| {r['slice']} | {r['n']} | {fmt(r['naive_z'])} | "
                f"{fmt(r['cal_norm_global'])} | {fmt(r['cal_abs_global'])} | "
                f"{fmt(r.get('cal_norm_local', float('nan')))} | "
                f"{fmt(r.get('cal_abs_local',  float('nan')))} |"
            )
        L.append("")
        L.append("### Per-tier calibration quantiles (n ≥ 5)")
        L.append("")
        L.append("| tier | n_calib | q_norm | q_abs |")
        L.append("|---:|---:|---:|---:|")
        for tier_str, q in sorted(blk["per_tier_q"].items()):
            L.append(f"| {tier_str} | {q['n_calib']} | "
                     f"{q['q_norm']:.4f} | {q['q_abs']:.4f} |")
        L.append("")
        L.append("### Tier-0 per-element calibration (lab dopants)")
        L.append("")
        L.append("| element | n_calib | q_norm | q_abs |")
        L.append("|---|---:|---:|---:|")
        for elem in LAB_DOPANTS:
            qd = blk["per_elem_tier0_q"].get(elem, {})
            n = qd.get("n_calib", 0)
            qn = qd.get("q_norm")
            qa = qd.get("q_abs")
            qns = "—" if qn is None else f"{qn:.4f}"
            qas = "—" if qa is None else f"{qa:.4f}"
            L.append(f"| {elem} | {n} | {qns} | {qas} |")
        L.append("")

    path.write_text("\n".join(L))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="both", choices=["both", "VC", "PDR"])
    parser.add_argument("--vc-bundle",  type=Path, default=DEFAULT_BUNDLES["VC"])
    parser.add_argument("--pdr-bundle", type=Path, default=DEFAULT_BUNDLES["PDR"])
    parser.add_argument("--level", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJ / "results" / "lab_regime_eval")
    args = parser.parse_args()

    targets = ["VC", "PDR"] if args.target == "both" else [args.target]
    bundles = {"VC": args.vc_bundle, "PDR": args.pdr_bundle}

    blocks = [calibrate_target(t, bundles[t], args.level) for t in targets]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"level": args.level, "naive_z": PI80_Z, "targets": blocks}
    (args.output_dir / "conformal_quantiles.json").write_text(
        json.dumps(report, indent=2))
    write_markdown(report, args.output_dir / "conformal_coverage.md")
    print(f"\nWrote {args.output_dir}/conformal_coverage.md")


if __name__ == "__main__":
    main()
