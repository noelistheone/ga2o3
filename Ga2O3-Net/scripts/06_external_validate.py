"""
Script 06: External validation of archived best models on literature data.

Loads the PDR and VC deployment bundles and evaluates them on a held-out
CSV of experimental rows scraped from papers that were NOT in the training
set. Computes raw / Platt / LOO-nested R², MAE, RMSE, Pearson r, slope with
bootstrap 95% CI, plus per-tier (A/B/C dopant support level) and per-dopant
breakdowns.

Usage:
    conda activate ga2o3
    python scripts/06_external_validate.py \\
        --external-csv data/raw/experimental/ga2o3_exp_external.csv \\
        --pdr-model results/deployment/pdr_best/stage3_model.pt \\
        --pdr-config results/deployment/pdr_best/multimodal.yaml \\
        --vc-model  results/deployment/vc_best/stage3_model.pt \\
        --vc-config results/deployment/vc_best/multimodal.yaml \\
        --out-dir   results/external_validation \\
        --gpu 1 --mc-passes 20

The external CSV must have the same schema as
``data/raw/experimental/ga2o3_exp.csv`` (see the project docs Dopant Specification
Format). Two extra columns are consumed if present:
  source_doi  — provenance (free-form string, typically a DOI).
  dopant_tier — one of {"A","B","C"} for tiered breakdown.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.experimental_dataset import Ga2O3ExpDataset, collate_fn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("external_validate")

TARGETS = ["photo_dark_ratio", "vacancy_concentration"]


# ────────────────────────────────────────────────────────────────────────────
#  Model loading (delegate to 04_finetune_predict.load_model)
# ────────────────────────────────────────────────────────────────────────────

def _load_script04_module():
    spec = importlib.util.spec_from_file_location(
        "script04", Path(__file__).parent / "04_finetune_predict.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_single_target_model(model_path: str, config_path: str, device):
    mod04 = _load_script04_module()
    model = mod04.load_model(model_path, config_path).to(device)
    model.eval()
    tgt = model.target_cols
    if len(tgt) != 1:
        raise ValueError(
            f"Expected single-target deployment model, got target_cols={tgt}"
        )
    return model, tgt[0]


# ────────────────────────────────────────────────────────────────────────────
#  Inference
# ────────────────────────────────────────────────────────────────────────────

def _infer_on_external(
    model,
    target_col: str,
    csv_path: str,
    structures_dir: str,
    device,
    batch_size: int,
    mc_passes: int,
) -> pd.DataFrame:
    """
    Run model on an external CSV and return one row per sample with columns:
      sample_idx, dopant_spec, dopant_label,
      {target}_true, {target}_pred, {target}_pred_platt, {target}_mc_std.

    Raw predictions are de-standardized with target_mean/target_std attached
    to the model (trained with target_standardize=True). Platt calibration
    (a, b) from the bundle is applied on top of de-standardized preds.
    """
    dataset = Ga2O3ExpDataset(
        csv_path=csv_path,
        structures_dir=structures_dir,
        target_cols=[target_col],
        augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    # Target standardization stats may be stored as array-of-1 or scalar.
    t_mean = getattr(model, "_target_mean", None)
    t_std = getattr(model, "_target_std", None)
    if t_mean is not None and not np.isscalar(t_mean):
        t_mean = float(np.asarray(t_mean).reshape(-1)[0])
    if t_std is not None and not np.isscalar(t_std):
        t_std = float(np.asarray(t_std).reshape(-1)[0])

    platt = getattr(model, "_platt_params", None) or {}
    a, b = platt.get(target_col, (1.0, 0.0))

    all_rows: list[dict] = []
    idx = 0
    for batch in loader:
        graph = batch["graph"].to(device)
        process = batch["process"].to(device)
        target = batch["target"]  # [B, 1]
        specs = batch["dopant_spec"]
        labels = batch["dopant_label"]

        with torch.no_grad():
            mean, std = model.mc_predict(graph, specs, process, n_passes=mc_passes)
        mean = mean.cpu().numpy()       # [B, 1] (z-score space)
        std = std.cpu().numpy()
        target = target.numpy()         # [B, 1]

        for i in range(mean.shape[0]):
            pred_z = float(mean[i, 0])
            if t_mean is not None and t_std is not None:
                pred_destd = pred_z * t_std + t_mean
            else:
                pred_destd = pred_z
            pred_platt = float(a) * pred_destd + float(b)

            all_rows.append({
                "sample_idx": idx,
                "dopant_spec": specs[i],
                "dopant_label": labels[i],
                f"{target_col}_true": float(target[i, 0]),
                f"{target_col}_pred": float(pred_destd),
                f"{target_col}_pred_platt": float(pred_platt),
                f"{target_col}_mc_std": float(std[i, 0]),
            })
            idx += 1

    return pd.DataFrame(all_rows)


# ────────────────────────────────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────────────────────────────────

def _load_analyze_oof_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_oof", Path(__file__).parent / "analyze_oof.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics_block(y_pred: np.ndarray, y_true: np.ndarray, label: str) -> str:
    """Return a single-line metrics string (match analyze_oof formatting)."""
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    aoof = _load_analyze_oof_module()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if np.var(y_pred) > 1e-12 and np.var(y_true) > 1e-12:
        r, _ = pearsonr(y_pred, y_true)
        slope = float(np.polyfit(y_pred, y_true, 1)[0])
    else:
        r, slope = float("nan"), float("nan")
    lo, hi = aoof._bootstrap_r2_ci(y_pred, y_true)
    return (
        f"  {label:8s}  N={len(y_true):3d}  R²={r2:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]  "
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  r={r:+.3f}  slope={slope:.3f}"
    )


def _eval_external(
    df: pd.DataFrame,
    target: str,
    external_meta: pd.DataFrame,
    internal_reference: dict | None = None,
) -> list[str]:
    """Produce a multi-line metrics report for one target."""
    aoof = _load_analyze_oof_module()

    t_col, p_col, pl_col = f"{target}_true", f"{target}_pred", f"{target}_pred_platt"
    if t_col not in df.columns or p_col not in df.columns:
        return [f"[skip] {target}: missing columns in predictions"]

    m = df[p_col].notna() & df[t_col].notna()
    # drop rows where the external CSV had no ground-truth value
    m &= df[t_col].notna()

    if m.sum() < 3:
        return [f"[skip] {target}: <3 rows with valid ground-truth"]

    y_pred = df.loc[m, p_col].to_numpy()
    y_true = df.loc[m, t_col].to_numpy()
    y_platt = df.loc[m, pl_col].to_numpy() if pl_col in df.columns else None

    lines = [f"\n### {target}"]
    lines.append(_metrics_block(y_pred, y_true, "raw"))
    if y_platt is not None:
        lines.append(_metrics_block(y_platt, y_true, "Platt"))
    if len(y_pred) >= 5:
        y_nested = aoof._loo_platt(y_pred, y_true)
        lines.append(_metrics_block(y_nested, y_true, "nested"))

    if internal_reference and target in internal_reference:
        ref = internal_reference[target]
        lines.append(
            f"  (internal 3J/3K reference: R²={ref['r2']:+.3f}, slope={ref['slope']:.3f}, "
            f"MAE={ref['mae']:.3f})"
        )

    # Per-tier breakdown
    if "dopant_tier" in df.columns:
        tier_lines = []
        for tier, tier_sub in df.groupby("dopant_tier"):
            mm = tier_sub[p_col].notna() & tier_sub[t_col].notna()
            if mm.sum() < 3:
                continue
            yp = tier_sub.loc[mm, p_col].to_numpy()
            yt = tier_sub.loc[mm, t_col].to_numpy()
            tier_lines.append(_metrics_block(yp, yt, f"tier-{tier}"))
        if tier_lines:
            lines.append("\n  — by tier —")
            lines.extend(tier_lines)

    # Per-dopant (same format as analyze_oof)
    if "dopant_label" in df.columns:
        from scipy.stats import pearsonr
        per_dopant = []
        for lab, sub in df.groupby("dopant_label"):
            mm = sub[p_col].notna() & sub[t_col].notna()
            if mm.sum() < 3:
                continue
            yp = sub.loc[mm, p_col].to_numpy()
            yt = sub.loc[mm, t_col].to_numpy()
            if np.var(yp) < 1e-12 or np.var(yt) < 1e-12:
                continue
            r, _ = pearsonr(yp, yt)
            per_dopant.append(f"    {lab:20s}  N={int(mm.sum()):3d}  r={r:+.3f}")
        if per_dopant:
            lines.append(f"\n  — per-dopant r (N≥3) —")
            lines.extend(per_dopant)

    return lines


def _internal_reference() -> dict:
    """Hard-coded numbers for 3J (PDR) and 3K (VC) nested CV for quick comparison."""
    return {
        "photo_dark_ratio":       {"r2": 0.511, "slope": 0.969, "mae": 1.023},
        "vacancy_concentration":  {"r2": 0.431, "slope": 0.927, "mae": 1.177},
    }


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="External validation of archived best models."
    )
    parser.add_argument("--external-csv", required=True)
    parser.add_argument("--structures-dir", default="data/structures")
    parser.add_argument("--pdr-model", default="results/deployment/pdr_best/stage3_model.pt")
    parser.add_argument("--pdr-config", default="results/deployment/pdr_best/multimodal.yaml")
    parser.add_argument("--vc-model",  default="results/deployment/vc_best/stage3_model.pt")
    parser.add_argument("--vc-config", default="results/deployment/vc_best/multimodal.yaml")
    parser.add_argument("--out-dir",   default="results/external_validation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; -1 for CPU.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mc-passes", type=int, default=20)
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run PDR model ──────────────────────────────────────────────────────
    logger.info(f"Loading PDR model: {args.pdr_model}")
    pdr_model, pdr_tgt = _load_single_target_model(
        args.pdr_model, args.pdr_config, device,
    )
    logger.info(f"  target_col = {pdr_tgt}")
    pdr_df = _infer_on_external(
        pdr_model, pdr_tgt, args.external_csv, args.structures_dir,
        device, args.batch_size, args.mc_passes,
    )

    # ── Run VC model ───────────────────────────────────────────────────────
    logger.info(f"Loading VC model: {args.vc_model}")
    vc_model, vc_tgt = _load_single_target_model(
        args.vc_model, args.vc_config, device,
    )
    logger.info(f"  target_col = {vc_tgt}")
    vc_df = _infer_on_external(
        vc_model, vc_tgt, args.external_csv, args.structures_dir,
        device, args.batch_size, args.mc_passes,
    )

    # ── Merge PDR+VC by sample_idx ─────────────────────────────────────────
    merge_keys = ["sample_idx", "dopant_spec", "dopant_label"]
    merged = pdr_df.merge(vc_df, on=merge_keys, how="outer")

    # Attach external metadata columns for provenance / tier break-down
    ext_df = pd.read_csv(args.external_csv).reset_index().rename(
        columns={"index": "sample_idx"},
    )
    keep = ["sample_idx", "source_doi", "dopant_tier",
            "concentration_at%", "element", "method"]
    keep = [c for c in keep if c in ext_df.columns]
    merged = merged.merge(ext_df[keep], on="sample_idx", how="left")

    # Save predictions
    out_csv = out_dir / "external_predictions.csv"
    merged.to_csv(out_csv, index=False)
    logger.info(f"Saved predictions: {out_csv} ({len(merged)} rows)")

    # ── Metrics report ─────────────────────────────────────────────────────
    ref = _internal_reference()
    lines: list[str] = [
        f"External validation results — {out_csv}",
        "=" * 80,
        f"N rows: {len(merged)}",
    ]
    if "dopant_tier" in ext_df.columns:
        tier_counts = ext_df["dopant_tier"].value_counts().sort_index()
        lines.append(f"Tier distribution: " + ", ".join(
            f"{t}={n}" for t, n in tier_counts.items()
        ))

    for tgt in TARGETS:
        lines.extend(_eval_external(merged, tgt, ext_df, internal_reference=ref))

    report = "\n".join(lines)
    print(report)

    out_txt = out_dir / "external_metrics.txt"
    out_txt.write_text(report + "\n")
    logger.info(f"Saved metrics: {out_txt}")


if __name__ == "__main__":
    main()
