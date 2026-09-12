"""
eval_lab_target_regime.py — evaluate frontier OOF predictions against the
specific lab target regime so reported numbers reflect lab-relevant accuracy
instead of literature-pool R².

Lab target regime (default; tunable via flags):
  - method:     RF/DC/magnetron sputtering            (process bit 12 == 1)
  - atmosphere: "Ar" or "vacuum" (NaN treated as Ar)  (strict Ar-only)
  - substrate:  native β-Ga2O3 bulk                   (process bit 8 == 0)
  - anneal:     no post-anneal                        (process bit 6 == 0)
  - dopant:     Mg / Sn / Fe (single-element)         (excludes undoped)

For each OOF row we count how many of the four constraints it violates and
assign it a "relaxation tier" t ∈ {0, 1, 2, 3, 4}. Tier 0 = perfect lab match.

Per tier we report:
  - N (rows; rows with a true label)
  - R²_Platt   (from `_pred_platt` column already in OOF CSVs)
  - Pearson r
  - PI80 coverage (Gaussian: |true - pred_platt| ≤ 1.282 · _std)
  - PI80 mean width

PI coverage is uncalibrated MC-dropout Gaussian. Task F (conformal) will
post-hoc fix the gap reported here.

Usage:
    python scripts/eval_lab_target_regime.py
    python scripts/eval_lab_target_regime.py --target VC
    python scripts/eval_lab_target_regime.py --output-dir results/lab_eval

Output:
    results/lab_regime_eval/<target>_per_tier.csv
    results/lab_regime_eval/<target>_per_row_tagged.csv
    results/lab_regime_eval/lab_regime_eval_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.experimental_dataset import Ga2O3ExpDataset

# Constraint names — must match the order used in the bitmask below.
CONSTRAINTS = ("sputter", "ar_only", "native_bulk", "no_anneal")

LAB_DOPANTS = ("Mg", "Sn", "Fe")
PI80_Z = 1.2815515655446004  # 80% Gaussian central interval

DEPLOYMENT_DIR = PROJ / "results" / "deployment"

DEFAULT_BUNDLES = {
    "VC":  DEPLOYMENT_DIR / "vc_alt_33_sputter_brouwer_atmo_class",
    "PDR": DEPLOYMENT_DIR / "pdr_alt_27_sputter_cmixup",
}
TARGET_COL = {
    "VC":  "vacancy_concentration",
    "PDR": "photo_dark_ratio",
}


# ── Regime tagging ───────────────────────────────────────────────────────────

def _atm_is_ar_only(atm) -> bool:
    """Strict Ar-only: 'Ar', 'vacuum', or NaN (default = Ar in the dataset)."""
    if pd.isna(atm) or str(atm).strip() == "":
        return True
    return str(atm).strip().lower() in {"ar", "vacuum"}


def tag_proximity(
    df: pd.DataFrame,
    proc: np.ndarray,
    *,
    require_dopant: bool = True,
) -> pd.DataFrame:
    """Tag each row with per-constraint satisfaction + tier (n_violations).

    Returns a copy with new columns:
      sputter_ok, ar_only_ok, native_bulk_ok, no_anneal_ok, dopant_in_lab,
      tier (0..4), n_constraints_met (0..4)
    """
    out = df.copy().reset_index(drop=True)
    out["sputter_ok"]     = (proc[:, 12] > 0.5).astype(int)
    out["ar_only_ok"]     = out["atmosphere"].apply(_atm_is_ar_only).astype(int)
    out["native_bulk_ok"] = (proc[:, 8]  < 0.5).astype(int)
    out["no_anneal_ok"]   = (proc[:, 6]  < 0.5).astype(int)

    if "element" in out.columns and require_dopant:
        out["dopant_in_lab"] = out["element"].isin(LAB_DOPANTS).astype(int)
    else:
        out["dopant_in_lab"] = 1

    n_met = (
        out[["sputter_ok", "ar_only_ok", "native_bulk_ok", "no_anneal_ok"]]
        .sum(axis=1)
    )
    out["n_constraints_met"] = n_met
    out["tier"] = 4 - n_met  # tier 0 = all met
    return out


# ── Metrics ──────────────────────────────────────────────────────────────────

def _r2_platt(y_true, y_pred_platt):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_platt, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _metrics_block(sub: pd.DataFrame, target_col: str) -> dict:
    pred_col       = f"{target_col}_pred"
    pred_platt_col = f"{target_col}_pred_platt"
    std_col        = f"{target_col}_std"
    true_col       = f"{target_col}_true"

    have = sub.dropna(subset=[true_col, pred_platt_col, std_col])
    n = len(have)
    if n < 2:
        return dict(n=n, r2_platt=float("nan"), pearson_r=float("nan"),
                    pi80_coverage=float("nan"), pi80_mean_width=float("nan"))

    y_true = have[true_col].to_numpy()
    y_pred = have[pred_platt_col].to_numpy()
    y_std  = have[std_col].to_numpy()

    r2 = _r2_platt(y_true, y_pred)
    if np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        r = float(stats.pearsonr(y_true, y_pred).statistic)
    else:
        r = float("nan")

    half_width = PI80_Z * y_std
    coverage = float(np.mean(np.abs(y_true - y_pred) <= half_width))
    mean_width = float(np.mean(2.0 * half_width))

    return dict(
        n=n,
        r2_platt=float(r2),
        pearson_r=r,
        pi80_coverage=coverage,
        pi80_mean_width=mean_width,
    )


# ── IO helpers ───────────────────────────────────────────────────────────────

def _build_dataset(fusion_cfg: dict) -> Ga2O3ExpDataset:
    return Ga2O3ExpDataset(
        csv_path=str(PROJ / fusion_cfg["paths"]["experimental_csv"]),
        structures_dir=str(PROJ / fusion_cfg["paths"].get(
            "structures_dir", "data/structures")),
        target_cols=fusion_cfg["targets"],
        log_transform_targets=fusion_cfg["data"].get("log_transform_targets", True),
        mask_noconc_labels=fusion_cfg["data"].get("mask_noconc_labels", True),
        mask_undoped_labels=fusion_cfg["data"].get("mask_undoped_labels", True),
        mask_carrier_vc=fusion_cfg["data"].get("mask_carrier_vc", True),
        exclude_elements=fusion_cfg["data"].get("exclude_elements", []),
        include_elements=fusion_cfg["data"].get("include_elements", None),
        include_keep_undoped=fusion_cfg["data"].get("include_keep_undoped", True),
    )


def _load_bundle(bundle_dir: Path) -> tuple[pd.DataFrame, dict]:
    oof = pd.read_csv(bundle_dir / "oof_predictions.csv")
    fusion_path = bundle_dir / "fusion_head128.yaml"
    if not fusion_path.exists():
        raise FileNotFoundError(f"missing fusion config in {bundle_dir}")
    with open(fusion_path) as f:
        fu = yaml.safe_load(f)
    return oof, fu


# ── Main per-target evaluation ───────────────────────────────────────────────

def evaluate_target(target: str, bundle_dir: Path, out_dir: Path) -> dict:
    target_col = TARGET_COL[target]
    print(f"\n=== {target}: {bundle_dir.name} ===")

    oof, fu = _load_bundle(bundle_dir)
    ds = _build_dataset(fu)
    proc = ds.get_process_array()

    # OOF.sample_idx → ds.df.iloc[idx] (post-filter index).
    src_cols = [c for c in
                ("element", "method", "atmosphere", "temperature_C",
                 "time_min", "concentration_at%", "doi", "notes")
                if c in ds.df.columns]
    src = ds.df[src_cols].copy().reset_index(drop=True)
    src["sample_idx"] = np.arange(len(src), dtype=int)

    proc_df = pd.DataFrame(proc, columns=[f"proc_{i}" for i in range(proc.shape[1])])
    proc_df["sample_idx"] = np.arange(len(proc_df), dtype=int)

    merged = oof.merge(src, on="sample_idx", how="left")
    merged = merged.merge(proc_df, on="sample_idx", how="left")

    tagged = tag_proximity(merged, proc[merged["sample_idx"].to_numpy()])

    per_tier = []
    for tier in range(5):
        sub = tagged[tagged["tier"] == tier]
        m = _metrics_block(sub, target_col)
        per_tier.append({"tier": tier, **m})
        if m["n"] > 0:
            print(f"  tier {tier} (constraints met={4-tier}): "
                  f"N={m['n']:3d}  R²_Platt={m['r2_platt']:+.3f}  "
                  f"r={m['pearson_r']:+.3f}  PI80_cov={m['pi80_coverage']:.2f}  "
                  f"width={m['pi80_mean_width']:.2f}")

    cum = []
    for max_tier in range(5):
        sub = tagged[tagged["tier"] <= max_tier]
        m = _metrics_block(sub, target_col)
        cum.append({"max_tier": max_tier, **m})

    elem_rows = []
    for elem in LAB_DOPANTS:
        sub = tagged[(tagged["tier"] == 0) & (tagged["element"] == elem)]
        m = _metrics_block(sub, target_col)
        elem_rows.append({"element": elem, **m})

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_tier).to_csv(out_dir / f"{target.lower()}_per_tier.csv", index=False)
    pd.DataFrame(cum).to_csv(out_dir / f"{target.lower()}_cumulative.csv", index=False)
    pd.DataFrame(elem_rows).to_csv(
        out_dir / f"{target.lower()}_tier0_per_element.csv", index=False)
    keep_cols = (
        ["sample_idx", "tier", "n_constraints_met",
         "sputter_ok", "ar_only_ok", "native_bulk_ok", "no_anneal_ok",
         "dopant_in_lab", "element", "atmosphere", "doi"]
        + [c for c in tagged.columns
           if c.startswith(target_col) or c == "dopant_label"]
    )
    keep_cols = [c for c in keep_cols if c in tagged.columns]
    tagged[keep_cols].to_csv(
        out_dir / f"{target.lower()}_per_row_tagged.csv", index=False)

    return {
        "target": target,
        "bundle": str(bundle_dir.relative_to(PROJ)),
        "n_oof": int(len(tagged)),
        "per_tier": per_tier,
        "cumulative": cum,
        "tier0_per_element": elem_rows,
    }


def write_markdown_summary(report: dict, out_path: Path) -> None:
    lines = []
    lines.append("# Lab-Target-Regime Evaluation")
    lines.append("")
    lines.append(f"Generated from snapshot frontier · {report['regime']}")
    lines.append("")
    lines.append("Tier 0 = all 4 constraints met (sputter ∧ Ar-only ∧ "
                 "native bulk ∧ no anneal). PI80 coverage uses uncalibrated "
                 "MC-dropout Gaussian (PI80 = pred_platt ± 1.282·std). "
                 "Conformal calibration (Task F) will post-hoc tighten this.")
    lines.append("")

    for blk in report["targets"]:
        target = blk["target"]
        lines.append(f"## {target} — `{blk['bundle']}`")
        lines.append("")
        lines.append(f"OOF rows total: **{blk['n_oof']}**")
        lines.append("")
        lines.append("### Per-tier metrics")
        lines.append("")
        lines.append("| Tier | Constraints met | N | R²_Platt | Pearson r | "
                     "PI80 coverage | PI80 width |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in blk["per_tier"]:
            t = row["tier"]
            lines.append(
                f"| {t} | {4 - t} | {row['n']} | "
                f"{row['r2_platt']:+.3f} | {row['pearson_r']:+.3f} | "
                f"{row['pi80_coverage']:.2f} | {row['pi80_mean_width']:.2f} |"
            )
        lines.append("")
        lines.append("### Cumulative (rows with ≤ tier_max violations)")
        lines.append("")
        lines.append("| max tier | N | R²_Platt | Pearson r | PI80 coverage | PI80 width |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in blk["cumulative"]:
            lines.append(
                f"| ≤{row['max_tier']} | {row['n']} | "
                f"{row['r2_platt']:+.3f} | {row['pearson_r']:+.3f} | "
                f"{row['pi80_coverage']:.2f} | {row['pi80_mean_width']:.2f} |"
            )
        lines.append("")
        lines.append("### Tier-0 per-element breakdown")
        lines.append("")
        lines.append("| element | N | R²_Platt | Pearson r | PI80 coverage | PI80 width |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in blk["tier0_per_element"]:
            lines.append(
                f"| {row['element']} | {row['n']} | "
                f"{row['r2_platt']:+.3f} | {row['pearson_r']:+.3f} | "
                f"{row['pi80_coverage']:.2f} | {row['pi80_mean_width']:.2f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="both",
                        choices=["both", "VC", "PDR"])
    parser.add_argument("--vc-bundle", type=Path, default=DEFAULT_BUNDLES["VC"])
    parser.add_argument("--pdr-bundle", type=Path, default=DEFAULT_BUNDLES["PDR"])
    parser.add_argument("--output-dir", type=Path,
                        default=PROJ / "results" / "lab_regime_eval")
    args = parser.parse_args()

    targets = ["VC", "PDR"] if args.target == "both" else [args.target]
    bundles = {"VC": args.vc_bundle, "PDR": args.pdr_bundle}

    target_blocks = []
    for t in targets:
        target_blocks.append(evaluate_target(t, bundles[t], args.output_dir))

    report = {
        "regime": ("Ar-only sputter × native β-Ga2O3 bulk × "
                   "no anneal × Mg/Sn/Fe single-element doping"),
        "constraints_order": list(CONSTRAINTS),
        "lab_dopants": list(LAB_DOPANTS),
        "targets": target_blocks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lab_regime_eval_summary.json").write_text(
        json.dumps(report, indent=2))
    write_markdown_summary(report, args.output_dir / "lab_regime_eval_summary.md")
    print(f"\nWrote {args.output_dir}/lab_regime_eval_summary.md")


if __name__ == "__main__":
    main()
