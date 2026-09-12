"""
Script 05: Comprehensive evaluation of the trained Ga2O3Net.

Loads:
  results/oof_predictions.csv   — honest out-of-fold predictions from Script 04
  results/stage3_model.pt       — final model (for MC Dropout uncertainty + embeddings)
  data/raw/experimental/ga2o3_exp.csv — ground-truth metadata
  results/cv_results.csv        — per-fold CV metrics from Script 04

Outputs (results/eval/):
  ── Readable by AI assistant ──────────────────────────────────────────────────
  eval_report.json              — ALL metrics in structured JSON (primary analysis file)
  eval_summary.txt              — formatted text report with tables, residual analysis,
                                  process-condition error correlations, warnings

  ── Visualisation (human) ─────────────────────────────────────────────────────
  parity_plots.pdf              — predicted vs actual, per target, annotated R²/MAE
  error_distribution.pdf        — per-dopant violin + strip plot of signed errors
  concentration_response.pdf    — Mg & Zn concentration series vs MC predictions
  uncertainty_calibration.pdf   — MC Dropout reliability diagram
  embedding_pca.pdf             — 2-D PCA of 160-dim embeddings, coloured by dopant

  ── Tables (LaTeX / CSV) ──────────────────────────────────────────────────────
  per_dopant_metrics.csv        — per-element MAE / RMSE / R² / N
  per_dopant_metrics.tex        — LaTeX booktabs table
  summary_metrics.tex           — overall CV metrics table
  oof_full.csv                  — OOF joined with all metadata

Usage:
    conda activate ga2o3
    python scripts/05_evaluate.py --no-show          # headless server
    python scripts/05_evaluate.py                    # interactive
    python scripts/05_evaluate.py --skip-plots       # metrics only, no matplotlib

Reading the results for model optimisation:
    After running, read results/eval/eval_summary.txt for a complete text report,
    or results/eval/eval_report.json for structured data.  Both files contain
    every number computed in this script in human/AI-readable form.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Human-readable axis labels ────────────────────────────────────────────────
TARGET_LABELS = {
    "photo_dark_ratio":      "log₁₀(Photo/Dark Ratio)",
    "vacancy_concentration": "log₁₀(Vacancy Concentration / cm⁻³)",
}

# ── Dopant colour palette (consistent across all plots) ───────────────────────
_DOPANT_COLORS: dict[str, str] = {}

def _dopant_color(label: str) -> str:
    try:
        import matplotlib.pyplot as plt
        if label not in _DOPANT_COLORS:
            cmap = plt.get_cmap("tab10")
            _DOPANT_COLORS[label] = cmap(len(_DOPANT_COLORS) % 10)
        return _DOPANT_COLORS[label]
    except Exception:
        return "steelblue"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Script 05: Model evaluation")
    parser.add_argument("--oof-csv",        default="results/oof_predictions.csv")
    parser.add_argument("--model-path",     default="results/stage3_model.pt")
    parser.add_argument("--cv-csv",         default="results/cv_results.csv")
    parser.add_argument("--exp-csv",        default="data/raw/experimental/ga2o3_exp.csv")
    parser.add_argument("--multimodal-cfg", default="config/multimodal.yaml")
    parser.add_argument("--fusion-cfg",     default="config/fusion.yaml")
    parser.add_argument("--eval-dir",       default="results/eval")
    parser.add_argument("--no-show",        action="store_true",
                        help="Do not call plt.show() — useful for headless runs.")
    parser.add_argument("--skip-plots",     action="store_true",
                        help="Skip all matplotlib output (metrics + text reports only).")
    parser.add_argument("--gpu",            type=int, default=0)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import yaml
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    with open(args.fusion_cfg) as f:
        fu_cfg = yaml.safe_load(f)
    target_cols = fu_cfg["targets"]

    # Master report dict — everything goes here, saved as JSON at the end
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_cols": target_cols,
        "dataset": {},
        "overall_metrics": {},
        "per_dopant_metrics": {},
        "per_method_metrics": {},
        "per_atmosphere_metrics": {},
        "residual_correlations": {},
        "uncertainty_calibration": {},
        "embedding_quality": {},
        "cv_fold_metrics": {},
        "warnings": [],
    }

    # ── 1. Load OOF predictions ───────────────────────────────────────────────
    logger.info("Loading OOF predictions...")
    oof_df = _load_oof(args.oof_csv, args.exp_csv, target_cols)
    oof_path = eval_dir / "oof_full.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"  OOF joined → {oof_path}  ({len(oof_df)} rows)")

    # Dataset-level stats
    report["dataset"] = _dataset_stats(oof_df, target_cols)

    # ── 2. Load CV fold metrics ───────────────────────────────────────────────
    report["cv_fold_metrics"] = _load_cv_metrics(args.cv_csv)

    # ── 3. Load final model + MC predictions ─────────────────────────────────
    logger.info("Loading final model for MC Dropout predictions...")
    try:
        model = _load_model(args.model_path, args.multimodal_cfg, device)
        mc_df = _run_mc_predictions(model, fu_cfg, device, target_cols)
        logger.info(f"  MC predictions complete: {len(mc_df)} samples")
    except Exception as exc:
        logger.warning(f"  Could not load model or run MC: {exc}")
        report["warnings"].append(f"MC predictions skipped: {exc}")
        model, mc_df = None, pd.DataFrame()

    # ── 4. Core accuracy metrics ──────────────────────────────────────────────
    logger.info("Computing accuracy metrics...")
    report["overall_metrics"]       = _overall_metrics(oof_df, target_cols)
    report["per_dopant_metrics"]    = _per_group_metrics(oof_df, "dopant_label", target_cols)
    report["per_method_metrics"]    = _per_group_metrics(oof_df, "method",      target_cols)
    report["per_atmosphere_metrics"]= _per_group_metrics(oof_df, "atmosphere",  target_cols)
    report["residual_correlations"] = _residual_correlations(oof_df, target_cols)

    # ── 5. Uncertainty calibration ────────────────────────────────────────────
    if not mc_df.empty:
        report["uncertainty_calibration"] = _calibration_stats(mc_df, target_cols)

    # ── 6. Embedding quality ──────────────────────────────────────────────────
    if model is not None and not args.skip_plots:
        try:
            report["embedding_quality"] = _embedding_quality(model, fu_cfg, device)
        except Exception as exc:
            logger.warning(f"  Embedding quality skipped: {exc}")
            report["warnings"].append(f"Embedding quality skipped: {exc}")

    # ── 7. Plots (can be skipped for headless metric-only runs) ───────────────
    if not args.skip_plots:
        _setup_matplotlib()
        show = not args.no_show

        logger.info("Generating parity plots...")
        plot_parity_plots(oof_df, target_cols, eval_dir, show=show)

        logger.info("Generating error distribution plots...")
        plot_error_distribution(oof_df, target_cols, eval_dir, show=show)

        if not mc_df.empty:
            logger.info("Generating concentration-response plots...")
            plot_concentration_response(mc_df, target_cols, eval_dir, show=show)

            logger.info("Generating uncertainty calibration plot...")
            plot_uncertainty_calibration(mc_df, target_cols, eval_dir, show=show)

        if model is not None:
            logger.info("Generating embedding PCA plot...")
            try:
                plot_embedding_pca(model, fu_cfg, device, eval_dir, show=show)
            except Exception as exc:
                logger.warning(f"  Embedding PCA plot failed: {exc}")

    # ── 8. LaTeX tables ───────────────────────────────────────────────────────
    logger.info("Saving LaTeX tables and CSV...")
    per_dop = _per_dopant_df(oof_df, target_cols)
    per_dop.to_csv(eval_dir / "per_dopant_metrics.csv", index=False)
    save_per_dopant_latex(per_dop, target_cols, eval_dir / "per_dopant_metrics.tex")
    save_summary_latex(oof_df, target_cols, eval_dir / "summary_metrics.tex")

    # ── 9. Save readable reports ──────────────────────────────────────────────
    logger.info("Saving eval_report.json and eval_summary.txt...")
    _save_json_report(report, eval_dir / "eval_report.json")
    _save_text_report(report, oof_df, mc_df, target_cols, eval_dir / "eval_summary.txt")

    logger.info(f"\nAll outputs written to {eval_dir}/")
    _print_summary(report, target_cols)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_oof(oof_csv: str, exp_csv: str, target_cols: list[str]) -> pd.DataFrame:
    oof = pd.read_csv(oof_csv)
    exp = pd.read_csv(exp_csv).reset_index().rename(columns={"index": "sample_idx"})
    meta_cols = ["sample_idx", "dopant_spec", "element", "concentration_at%",
                 "atmosphere", "method", "temperature_C", "time_min"]
    meta_cols = [c for c in meta_cols if c in exp.columns]
    return oof.merge(exp[meta_cols], on="sample_idx", how="left")


def _load_model(model_path: str, multimodal_cfg: str, device):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "script04", Path(__file__).parent / "04_finetune_predict.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_model(model_path, multimodal_cfg).to(device)


def _load_cv_metrics(cv_csv: str) -> dict:
    """Load per-fold CV metrics saved by script 04."""
    if not Path(cv_csv).exists():
        return {}
    try:
        df = pd.read_csv(cv_csv)
        return {row["metric"]: float(row["value"]) for _, row in df.iterrows()}
    except Exception:
        return {}


def _run_mc_predictions(model, fu_cfg: dict, device, target_cols: list[str]) -> pd.DataFrame:
    import torch
    from src.data.experimental_dataset import Ga2O3ExpDataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = Ga2O3ExpDataset(
        csv_path=fu_cfg["paths"]["experimental_csv"],
        structures_dir=fu_cfg["paths"]["structures_dir"],
        target_cols=target_cols,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    mc_passes = fu_cfg.get("training", {}).get("mc_dropout_passes", 20) or 20

    all_means, all_stds, all_targets, all_specs, all_labels = [], [], [], [], []
    for batch in loader:
        graph        = batch["graph"].to(device)
        process      = batch["process"].to(device)
        target       = batch["target"]
        dopant_specs = batch["dopant_spec"]
        mean, std = model.mc_predict(graph, dopant_specs, process, mc_passes)
        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())
        all_targets.append(target.numpy())
        all_specs.extend(dopant_specs)
        all_labels.extend(batch["dopant_label"])

    means   = np.concatenate(all_means,   axis=0)
    stds    = np.concatenate(all_stds,    axis=0)
    targets = np.concatenate(all_targets, axis=0)

    exp_df = pd.read_csv(fu_cfg["paths"]["experimental_csv"]).reset_index()
    rows = []
    for i in range(len(all_specs)):
        row = {"sample_idx": i, "dopant_spec": all_specs[i], "dopant_label": all_labels[i]}
        if i < len(exp_df):
            row["concentration_at%"] = exp_df.iloc[i].get("concentration_at%", np.nan)
            row["element"]           = exp_df.iloc[i].get("element", "")
        for t, col in enumerate(target_cols):
            row[f"{col}_pred"] = float(means[i, t])
            row[f"{col}_true"] = float(targets[i, t])
            row[f"{col}_std"]  = float(stds[i, t])
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _regress(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute MAE, RMSE, R², slope, intercept for one target."""
    if len(pred) < 2:
        return {"MAE": None, "RMSE": None, "R2": None, "N": len(pred),
                "slope": None, "intercept": None, "pearson_r": None}
    mae  = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    if np.std(true) < 1e-12 or np.std(pred) < 1e-12:
        slope = intercept = r_val = p_val = None
    else:
        slope, intercept, r_val, p_val, _ = stats.linregress(true, pred)
        slope, intercept, r_val, p_val = (
            round(float(slope), 4), round(float(intercept), 4),
            round(float(r_val), 4), round(float(p_val), 6),
        )
    return {
        "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4),
        "N": int(len(pred)), "slope": slope,
        "intercept": intercept, "pearson_r": r_val, "p_value": p_val,
    }


def _dataset_stats(oof_df: pd.DataFrame, target_cols: list[str]) -> dict:
    stats_dict: dict = {"n_oof_rows": len(oof_df)}
    for col in target_cols:
        true_col = f"{col}_true"
        if true_col in oof_df.columns:
            valid = oof_df[true_col].dropna()
            stats_dict[f"{col}_labeled_N"] = int(len(valid))
            if len(valid):
                stats_dict[f"{col}_true_mean"]  = round(float(valid.mean()), 4)
                stats_dict[f"{col}_true_std"]   = round(float(valid.std()),  4)
                stats_dict[f"{col}_true_min"]   = round(float(valid.min()),  4)
                stats_dict[f"{col}_true_max"]   = round(float(valid.max()),  4)
    if "dopant_label" in oof_df.columns:
        vc = oof_df["dopant_label"].value_counts().to_dict()
        stats_dict["dopant_counts"] = {str(k): int(v) for k, v in vc.items()}
    if "method" in oof_df.columns:
        mc = oof_df["method"].fillna("unknown").value_counts().to_dict()
        stats_dict["method_counts"] = {str(k): int(v) for k, v in mc.items()}
    return stats_dict


def _overall_metrics(oof_df: pd.DataFrame, target_cols: list[str]) -> dict:
    result = {}
    for col in target_cols:
        pred_col, true_col = f"{col}_pred", f"{col}_true"
        valid = oof_df[[pred_col, true_col]].dropna(subset=[true_col])
        if valid.empty:
            result[col] = {"N": 0}
            continue
        result[col] = _regress(valid[pred_col].values, valid[true_col].values)
        # Signed error stats
        errs = valid[pred_col].values - valid[true_col].values
        result[col]["mean_signed_error"] = round(float(errs.mean()), 4)
        result[col]["median_signed_error"] = round(float(np.median(errs)), 4)
        result[col]["error_std"] = round(float(errs.std()), 4)
        result[col]["within_0.5"] = round(float((np.abs(errs) < 0.5).mean()), 4)
        result[col]["within_1.0"] = round(float((np.abs(errs) < 1.0).mean()), 4)
    return result


def _per_group_metrics(oof_df: pd.DataFrame, group_col: str,
                       target_cols: list[str]) -> dict:
    """Compute metrics grouped by dopant / method / atmosphere."""
    if group_col not in oof_df.columns:
        return {}
    result = {}
    groups = oof_df[group_col].fillna("unknown").unique()
    for grp in sorted(groups, key=str):
        sub = oof_df[oof_df[group_col].fillna("unknown") == grp]
        result[str(grp)] = {}
        for col in target_cols:
            pred_col, true_col = f"{col}_pred", f"{col}_true"
            valid = sub[[pred_col, true_col]].dropna(subset=[true_col])
            if valid.empty:
                result[str(grp)][col] = {"N": 0}
            else:
                result[str(grp)][col] = _regress(
                    valid[pred_col].values, valid[true_col].values
                )
    return result


def _residual_correlations(oof_df: pd.DataFrame, target_cols: list[str]) -> dict:
    """
    Pearson correlation between |prediction error| and process conditions.
    High correlation indicates a systematic relationship the model is not capturing.
    """
    continuous_cols = ["temperature_C", "time_min", "concentration_at%"]
    result = {}
    for col in target_cols:
        pred_col, true_col = f"{col}_pred", f"{col}_true"
        valid = oof_df[[pred_col, true_col] + continuous_cols].dropna(subset=[true_col])
        if len(valid) < 5:
            result[col] = {}
            continue
        abs_err = (valid[pred_col] - valid[true_col]).abs().values
        signed_err = (valid[pred_col] - valid[true_col]).values
        col_results = {}
        for pc in continuous_cols:
            if pc not in valid.columns:
                continue
            pcvals = valid[pc].dropna()
            if len(pcvals) < 5:
                continue
            # align on index
            aligned = valid[[pc]].join(pd.Series(abs_err, index=valid.index, name="abs_err")).dropna()
            if len(aligned) < 5:
                continue
            r_abs, p_abs = stats.pearsonr(aligned[pc].values, aligned["abs_err"].values)
            aligned2 = valid[[pc]].join(pd.Series(signed_err, index=valid.index, name="signed_err")).dropna()
            r_sgn, p_sgn = stats.pearsonr(aligned2[pc].values, aligned2["signed_err"].values)
            col_results[pc] = {
                "r_with_abs_error":    round(float(r_abs), 4),
                "p_abs":               round(float(p_abs), 4),
                "r_with_signed_error": round(float(r_sgn), 4),
                "p_signed":            round(float(p_sgn), 4),
                "interpretation": (
                    "significant positive bias trend" if r_sgn > 0.35 and p_sgn < 0.05
                    else "significant negative bias trend" if r_sgn < -0.35 and p_sgn < 0.05
                    else "systematic abs-error trend (heteroscedastic)" if abs(r_abs) > 0.35 and p_abs < 0.05
                    else "no significant trend"
                ),
            }
        result[col] = col_results
    return result


def _calibration_stats(mc_df: pd.DataFrame, target_cols: list[str]) -> dict:
    """
    Uncertainty calibration: for each confidence level α, compute empirical coverage.
    Reports mean calibration error (MCE) and whether model is over/under-confident.
    """
    from scipy.stats import norm as spnorm
    result = {}
    alphas = np.linspace(0.1, 0.9, 17)
    for col in target_cols:
        true_col, pred_col, std_col = f"{col}_true", f"{col}_pred", f"{col}_std"
        if std_col not in mc_df.columns:
            continue
        sub = mc_df[[pred_col, true_col, std_col]].dropna(subset=[true_col])
        if sub.empty or (sub[std_col] == 0).all():
            continue
        pred  = sub[pred_col].values
        true  = sub[true_col].values
        std   = sub[std_col].values.clip(min=1e-6)
        empirical = []
        for alpha in alphas:
            z = spnorm.ppf((1 + alpha) / 2)
            empirical.append(float((np.abs(pred - true) <= z * std).mean()))
        empirical = np.array(empirical)
        mce = float(np.mean(np.abs(empirical - alphas)))
        avg_std = float(std.mean())
        avg_abs_err = float(np.abs(pred - true).mean())
        # sharpness: how tight are the uncertainty estimates?
        sharpness = float((std < avg_abs_err).mean())
        result[col] = {
            "mean_calibration_error": round(mce, 4),
            "avg_predicted_std": round(avg_std, 4),
            "avg_abs_error": round(avg_abs_err, 4),
            "ratio_std_to_mae": round(avg_std / (avg_abs_err + 1e-9), 4),
            "fraction_std_below_mae": round(sharpness, 4),
            "calibration_direction": (
                "over-confident (intervals too narrow)" if mce > 0.05 and avg_std < avg_abs_err
                else "under-confident (intervals too wide)" if mce > 0.05 and avg_std > avg_abs_err * 1.5
                else "well-calibrated"
            ),
            "coverage_at_68pct": round(float(empirical[np.argmin(np.abs(alphas - 0.68))]), 4),
            "coverage_at_95pct": round(float(empirical[np.argmin(np.abs(alphas - 0.95))]), 4),
        }
    return result


def _embedding_quality(model, fu_cfg: dict, device) -> dict:
    """
    Quantify embedding space quality:
    - PCA explained variance
    - Silhouette score (how well dopant classes cluster)
    """
    import torch
    from src.data.experimental_dataset import Ga2O3ExpDataset, collate_fn
    from torch.utils.data import DataLoader
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    dataset = Ga2O3ExpDataset(
        csv_path=fu_cfg["paths"]["experimental_csv"],
        structures_dir=fu_cfg["paths"]["structures_dir"],
        target_cols=fu_cfg["targets"], augment=False,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_emb, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = model.get_embedding(
                batch["graph"].to(device),
                batch["dopant_spec"],
                batch["process"].to(device),
            )
            all_emb.append(emb.cpu().numpy())
            all_labels.extend(batch["dopant_label"])

    embeddings = np.concatenate(all_emb, axis=0)
    pca = PCA(n_components=min(10, embeddings.shape[1]))
    pca.fit(embeddings)
    ev = pca.explained_variance_ratio_

    le = LabelEncoder()
    label_ids = le.fit_transform(all_labels)
    sil = float(silhouette_score(embeddings, label_ids, metric="cosine")) if len(set(all_labels)) > 1 else None

    return {
        "n_samples": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "pca_explained_variance_top10": [round(float(v), 4) for v in ev],
        "pca_cumulative_80pct_components": int(np.argmax(np.cumsum(ev) >= 0.80) + 1),
        "silhouette_score_cosine": round(sil, 4) if sil is not None else None,
        "silhouette_interpretation": (
            "good separation (> 0.5)" if sil and sil > 0.5
            else "moderate separation (0.2–0.5)" if sil and sil > 0.2
            else "poor separation (< 0.2)" if sil is not None
            else "n/a"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  READABLE REPORT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _save_json_report(report: dict, path: Path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  JSON report → {path}")


def _save_text_report(
    report: dict,
    oof_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    target_cols: list[str],
    path: Path,
):
    lines = []
    W = 72  # column width

    def hdr(title: str, ch: str = "═"):
        lines.append("\n" + ch * W)
        lines.append(f"  {title}")
        lines.append(ch * W)

    def sub(title: str):
        lines.append(f"\n── {title} {'─'*(W-5-len(title))}")

    lines.append("=" * W)
    lines.append("  Ga2O3-Net  —  Evaluation Report")
    lines.append(f"  Generated: {report['generated_at']}")
    lines.append("=" * W)

    # ── Dataset ────────────────────────────────────────────────────────────────
    hdr("1. DATASET OVERVIEW")
    ds = report["dataset"]
    lines.append(f"  OOF rows total : {ds.get('n_oof_rows', '?')}")
    for col in target_cols:
        n = ds.get(f"{col}_labeled_N", "?")
        mu  = ds.get(f"{col}_true_mean", "?")
        sig = ds.get(f"{col}_true_std",  "?")
        lo  = ds.get(f"{col}_true_min",  "?")
        hi  = ds.get(f"{col}_true_max",  "?")
        lines.append(f"  {col:<30} N={n}  mean={mu}  std={sig}  [{lo}, {hi}]")
    if "dopant_counts" in ds:
        lines.append("\n  Samples per dopant:")
        for k, v in sorted(ds["dopant_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"    {k:<20} {v:>4}")
    if "method_counts" in ds:
        lines.append("\n  Samples per fabrication method:")
        for k, v in sorted(ds["method_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"    {str(k):<35} {v:>4}")

    # ── Overall accuracy ───────────────────────────────────────────────────────
    hdr("2. OVERALL OOF ACCURACY  (primary metric for model optimisation)")
    om = report["overall_metrics"]
    for col in target_cols:
        m = om.get(col, {})
        lines.append(f"\n  Target: {col}")
        lines.append(f"    N           = {m.get('N', '?')}")
        lines.append(f"    MAE         = {m.get('MAE', '?')}   (log₁₀ units)")
        lines.append(f"    RMSE        = {m.get('RMSE', '?')}")
        lines.append(f"    R²          = {m.get('R2', '?')}")
        lines.append(f"    Pearson r   = {m.get('pearson_r', '?')}  (p={m.get('p_value','?')})")
        lines.append(f"    Slope       = {m.get('slope', '?')}  (ideal=1.0)")
        lines.append(f"    Intercept   = {m.get('intercept', '?')}  (ideal=0.0)")
        lines.append(f"    Mean signed error  = {m.get('mean_signed_error','?')}  (+ = overpredict)")
        lines.append(f"    Median signed error= {m.get('median_signed_error','?')}")
        lines.append(f"    Within ±0.5 log₁₀  = {m.get('within_0.5','?'):.0%}")
        lines.append(f"    Within ±1.0 log₁₀  = {m.get('within_1.0','?'):.0%}")

    # ── CV fold metrics ────────────────────────────────────────────────────────
    hdr("3. CV FOLD METRICS  (from cv_results.csv)")
    cv = report["cv_fold_metrics"]
    if cv:
        for k, v in sorted(cv.items()):
            lines.append(f"  {k:<45} {v:.4f}")
    else:
        lines.append("  (cv_results.csv not found or empty)")

    # ── Per-dopant ─────────────────────────────────────────────────────────────
    hdr("4. PER-DOPANT BREAKDOWN")
    pdm = report["per_dopant_metrics"]
    for col in target_cols:
        lines.append(f"\n  {col}")
        lines.append(f"  {'Dopant':<20} {'N':>4} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
        lines.append(f"  {'-'*20} {'-'*4} {'-'*7} {'-'*7} {'-'*7}")
        rows_sorted = sorted(
            ((grp, m.get(col, {})) for grp, m in pdm.items()),
            key=lambda x: -(x[1].get("N", 0))
        )
        for grp, m in rows_sorted:
            if not m or m.get("N", 0) == 0:
                continue
            mae  = f"{m['MAE']:.4f}"  if m.get("MAE")  is not None else "  --  "
            rmse = f"{m['RMSE']:.4f}" if m.get("RMSE") is not None else "  --  "
            r2   = f"{m['R2']:.4f}"   if m.get("R2")   is not None else "  --  "
            lines.append(f"  {str(grp):<20} {m['N']:>4} {mae:>7} {rmse:>7} {r2:>7}")

    # ── Per-method ─────────────────────────────────────────────────────────────
    hdr("5. PER-FABRICATION-METHOD BREAKDOWN")
    pmm = report["per_method_metrics"]
    if pmm:
        for col in target_cols:
            lines.append(f"\n  {col}")
            lines.append(f"  {'Method':<35} {'N':>4} {'MAE':>7} {'RMSE':>7}")
            lines.append(f"  {'-'*35} {'-'*4} {'-'*7} {'-'*7}")
            for grp, m in sorted(pmm.items(), key=lambda x: -(x[1].get(col, {}).get("N", 0))):
                mc = m.get(col, {})
                if not mc or mc.get("N", 0) == 0:
                    continue
                mae  = f"{mc['MAE']:.4f}"  if mc.get("MAE")  is not None else "  --  "
                rmse = f"{mc['RMSE']:.4f}" if mc.get("RMSE") is not None else "  --  "
                lines.append(f"  {str(grp)[:35]:<35} {mc['N']:>4} {mae:>7} {rmse:>7}")
    else:
        lines.append("  (method column not found in OOF data)")

    # ── Per-atmosphere ─────────────────────────────────────────────────────────
    hdr("6. PER-ATMOSPHERE BREAKDOWN")
    pam = report["per_atmosphere_metrics"]
    if pam:
        for col in target_cols:
            lines.append(f"\n  {col}")
            lines.append(f"  {'Atmosphere':<30} {'N':>4} {'MAE':>7} {'RMSE':>7} {'Mean±err':>10}")
            lines.append(f"  {'-'*30} {'-'*4} {'-'*7} {'-'*7} {'-'*10}")
            for grp, m in sorted(pam.items(), key=lambda x: -(x[1].get(col, {}).get("N", 0))):
                mc = m.get(col, {})
                if not mc or mc.get("N", 0) == 0:
                    continue
                mae  = f"{mc['MAE']:.4f}"  if mc.get("MAE")  is not None else "  --  "
                rmse = f"{mc['RMSE']:.4f}" if mc.get("RMSE") is not None else "  --  "
                se   = f"{mc.get('mean_signed_error',0):+.3f}" if mc.get("mean_signed_error") is not None else "    --"
                lines.append(f"  {str(grp)[:30]:<30} {mc['N']:>4} {mae:>7} {rmse:>7} {se:>10}")
    else:
        lines.append("  (atmosphere column not found in OOF data)")

    # ── Residual correlations ──────────────────────────────────────────────────
    hdr("7. RESIDUAL CORRELATIONS WITH PROCESS CONDITIONS")
    lines.append("  Pearson r between |error| and each process variable.")
    lines.append("  Large |r| (>0.35, p<0.05) = model is not capturing this effect.")
    rc = report["residual_correlations"]
    for col in target_cols:
        lines.append(f"\n  {col}")
        col_rc = rc.get(col, {})
        if not col_rc:
            lines.append("    (insufficient data)")
            continue
        for pc, d in col_rc.items():
            lines.append(
                f"    {pc:<25}  r_abs={d['r_with_abs_error']:+.3f} (p={d['p_abs']:.3f})"
                f"  r_signed={d['r_with_signed_error']:+.3f} (p={d['p_signed']:.3f})"
                f"  → {d['interpretation']}"
            )

    # ── Uncertainty calibration ────────────────────────────────────────────────
    hdr("8. MC DROPOUT UNCERTAINTY CALIBRATION")
    uc = report["uncertainty_calibration"]
    if uc:
        for col in target_cols:
            m = uc.get(col, {})
            if not m:
                lines.append(f"  {col}: no uncertainty data")
                continue
            lines.append(f"\n  {col}")
            lines.append(f"    Mean calibration error (MCE) = {m.get('mean_calibration_error','?')}")
            lines.append(f"    Avg predicted std            = {m.get('avg_predicted_std','?')}")
            lines.append(f"    Avg abs error (MAE)          = {m.get('avg_abs_error','?')}")
            lines.append(f"    std / MAE ratio              = {m.get('ratio_std_to_mae','?')}  (ideal ≈ 1.0)")
            lines.append(f"    Coverage @ 68%               = {m.get('coverage_at_68pct','?'):.0%}  (ideal 68%)")
            lines.append(f"    Coverage @ 95%               = {m.get('coverage_at_95pct','?'):.0%}  (ideal 95%)")
            lines.append(f"    Assessment                   : {m.get('calibration_direction','?')}")
    else:
        lines.append("  (MC predictions not available)")

    # ── Embedding quality ──────────────────────────────────────────────────────
    hdr("9. EMBEDDING SPACE QUALITY")
    eq = report["embedding_quality"]
    if eq:
        lines.append(f"  Embedding dim    : {eq.get('embedding_dim','?')}")
        lines.append(f"  N samples        : {eq.get('n_samples','?')}")
        ev = eq.get("pca_explained_variance_top10", [])
        if ev:
            lines.append(f"  PCA var (top 10) : {[f'{v:.3f}' for v in ev]}")
            lines.append(f"  Components for 80% variance: {eq.get('pca_cumulative_80pct_components','?')}")
        sil = eq.get("silhouette_score_cosine")
        lines.append(f"  Silhouette score (cosine, by dopant): {sil}  — {eq.get('silhouette_interpretation','?')}")
    else:
        lines.append("  (embedding quality not computed)")

    # ── Warnings ───────────────────────────────────────────────────────────────
    if report["warnings"]:
        hdr("10. WARNINGS")
        for w in report["warnings"]:
            lines.append(f"  ⚠  {w}")

    # ── Optimisation hints ─────────────────────────────────────────────────────
    hdr("11. OPTIMISATION HINTS  (auto-generated from metrics above)", ch="─")
    hints = _generate_hints(report, target_cols)
    for h in hints:
        lines.append(f"  • {h}")

    lines.append("\n" + "=" * W + "\n")

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    logger.info(f"  Text report  → {path}")


def _generate_hints(report: dict, target_cols: list[str]) -> list[str]:
    """Auto-generate optimisation hints based on computed metrics."""
    hints = []
    om = report.get("overall_metrics", {})
    uc = report.get("uncertainty_calibration", {})
    rc = report.get("residual_correlations", {})
    eq = report.get("embedding_quality", {})

    for col in target_cols:
        m = om.get(col, {})
        if not m or m.get("N", 0) == 0:
            continue
        r2  = m.get("R2")
        mae = m.get("MAE")
        slp = m.get("slope")
        mse = m.get("mean_signed_error")

        if r2 is not None and r2 < 0.5:
            hints.append(f"[{col}] R²={r2:.3f} < 0.5 — model has low explanatory power. "
                         "Consider: more CV folds, stronger regularisation, or feature engineering.")
        if slp is not None and slp < 0.6:
            hints.append(f"[{col}] Slope={slp:.3f} — model is over-regularised (predictions shrunken toward mean). "
                         "Try reducing dropout or weight decay.")
        if slp is not None and slp > 1.3:
            hints.append(f"[{col}] Slope={slp:.3f} > 1.3 — model is extrapolating. Check for leakage.")
        if mse is not None and abs(mse) > 0.3:
            direction = "overpredicting" if mse > 0 else "underpredicting"
            hints.append(f"[{col}] Systematic bias: mean signed error={mse:+.3f} ({direction}). "
                         "Check if a dopant subgroup dominates with large positive/negative residuals.")

        col_uc = uc.get(col, {})
        if col_uc:
            if col_uc.get("calibration_direction", "").startswith("over-confident"):
                hints.append(f"[{col}] MC Dropout is over-confident (intervals too narrow). "
                             "Try increasing n_passes or dropout rate in prediction heads.")
            elif col_uc.get("calibration_direction", "").startswith("under-confident"):
                hints.append(f"[{col}] MC Dropout is under-confident. "
                             "Consider reducing dropout rate or using temperature scaling.")

        col_rc = rc.get(col, {})
        for pc, d in col_rc.items():
            if abs(d.get("r_with_abs_error", 0)) > 0.35 and d.get("p_abs", 1) < 0.05:
                hints.append(f"[{col}] |error| correlates with {pc} (r={d['r_with_abs_error']:+.3f}). "
                             f"Model is not fully capturing the {pc} effect — consider adding "
                             f"interaction features or a richer process encoding.")

    sil = eq.get("silhouette_score_cosine")
    if sil is not None and sil < 0.2:
        hints.append("Embedding silhouette score < 0.2: dopant classes not well-separated in 160-dim space. "
                     "This may limit the model's ability to generalise across dopants. "
                     "Consider stronger contrastive loss weight (infonce_weight).")

    if not hints:
        hints.append("No critical issues detected from automatic analysis. "
                     "Review per-dopant breakdown for individual outliers.")
    return hints


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_matplotlib():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_parity_plots(oof_df, target_cols, eval_dir, show=True):
    import matplotlib.pyplot as plt
    n = len(target_cols)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        pred_col, true_col, std_col = f"{col}_pred", f"{col}_true", f"{col}_std"
        sub = oof_df[[pred_col, true_col, "dopant_label"]].dropna(subset=[true_col])
        if sub.empty:
            ax.set_title(f"{col}\n(no data)"); continue

        pred, true, labels = sub[pred_col].values, sub[true_col].values, sub["dopant_label"].values
        sizes = 60
        if std_col in oof_df.columns:
            stds = oof_df.loc[sub.index, std_col].fillna(0).values
            sizes = 30 + 80 * (stds - stds.min()) / (stds.max() - stds.min() + 1e-9)

        for lbl in sorted(set(labels)):
            mask = labels == lbl
            ax.scatter(true[mask], pred[mask],
                       s=sizes[mask] if isinstance(sizes, np.ndarray) else sizes,
                       color=_dopant_color(lbl), label=lbl,
                       edgecolors="k", linewidths=0.4, alpha=0.85, zorder=3)

        lo = min(true.min(), pred.min()) - 0.2
        hi = max(true.max(), pred.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Ideal (y=x)", zorder=2)
        slope, intercept, r_val, *_ = stats.linregress(true, pred)
        x_fit = np.linspace(lo, hi, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "b-", lw=1, alpha=0.6,
                label=f"Fit (slope={slope:.2f})", zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"True  {TARGET_LABELS.get(col, col)}")
        ax.set_ylabel(f"Predicted  {TARGET_LABELS.get(col, col)}")
        ax.set_title(col.replace("_", " ").title())

        mae  = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        r2   = float(r_val ** 2)
        ax.text(0.04, 0.96,
                f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\n$R^2$ = {r2:.3f}\nN = {len(true)}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.7, markerscale=0.8)
        ax.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle("Out-of-Fold Parity Plots — Ga2O3-Net", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, eval_dir / "parity_plots.pdf", show)


def plot_error_distribution(oof_df, target_cols, eval_dir, show=True):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed — skipping error distribution plot.")
        return

    n = len(target_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(8, 2.5 * n), 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        sub = oof_df[[f"{col}_pred", f"{col}_true", "dopant_label"]].dropna(subset=[f"{col}_true"]).copy()
        if sub.empty:
            continue
        sub["error"] = sub[f"{col}_pred"] - sub[f"{col}_true"]
        order = sub.groupby("dopant_label")["error"].median().sort_values().index.tolist()
        palette = {lbl: _dopant_color(lbl) for lbl in order}
        sns.violinplot(data=sub, x="dopant_label", y="error", order=order,
                       palette=palette, inner=None, cut=0, ax=ax, alpha=0.6, linewidth=0.8)
        sns.stripplot(data=sub, x="dopant_label", y="error", order=order,
                      palette=palette, jitter=True, size=4, ax=ax,
                      edgecolor="k", linewidth=0.3, alpha=0.85)
        ax.axhline(0, color="red", linestyle="--", lw=1, alpha=0.7)
        ax.set_xlabel("Dopant")
        ax.set_ylabel("Prediction Error (log₁₀)")
        ax.set_title(col.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Prediction Error Distribution by Dopant", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, eval_dir / "error_distribution.pdf", show)


def plot_concentration_response(mc_df, target_cols, eval_dir, show=True, min_points=3):
    import matplotlib.pyplot as plt
    selected = sorted([
        elem for elem, grp in mc_df.groupby("element")
        if len(grp["concentration_at%"].dropna().unique()) >= min_points
    ])
    if not selected:
        logger.warning("No dopants with >= %d concentration points.", min_points)
        return

    n_cols, n_rows = len(selected), len(target_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)

    for row_i, col in enumerate(target_cols):
        for col_i, elem in enumerate(selected):
            ax = axes[row_i][col_i]
            sub = mc_df[mc_df["element"] == elem].sort_values("concentration_at%")
            conc = sub["concentration_at%"].values
            true = sub[f"{col}_true"].values
            pred = sub[f"{col}_pred"].values
            std  = sub.get(f"{col}_std", pd.Series(np.zeros_like(pred))).values

            valid_t = ~np.isnan(true)
            valid_p = ~np.isnan(pred)
            if valid_t.any():
                ax.plot(conc[valid_t] * 100, true[valid_t], "o-", color="steelblue",
                        lw=1.5, ms=5, label="Measured", zorder=3)
            if valid_p.any():
                ax.plot(conc[valid_p] * 100, pred[valid_p], "s--", color="darkorange",
                        lw=1.5, ms=4, label="Predicted (MC mean)", zorder=3)
                ax.fill_between(conc[valid_p] * 100,
                                pred[valid_p] - 2 * std[valid_p],
                                pred[valid_p] + 2 * std[valid_p],
                                color="darkorange", alpha=0.2, label="±2σ MC", zorder=2)
            ax.set_xlabel("Concentration (at%)")
            ax.set_ylabel(TARGET_LABELS.get(col, col))
            ax.set_title(f"{elem} — {col.replace('_', ' ').title()}")
            ax.legend(fontsize=8, framealpha=0.7)
            ax.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle("Concentration–Response: Measured vs. Predicted", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, eval_dir / "concentration_response.pdf", show)


def plot_uncertainty_calibration(mc_df, target_cols, eval_dir, show=True, n_bins=20):
    import matplotlib.pyplot as plt
    from scipy.stats import norm as spnorm

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    alphas = np.linspace(0.05, 0.95, n_bins)
    colors = ["steelblue", "darkorange", "green", "crimson"]

    for i, col in enumerate(target_cols):
        sub = mc_df[[f"{col}_pred", f"{col}_true", f"{col}_std"]].dropna(subset=[f"{col}_true"])
        if sub.empty or (sub[f"{col}_std"] == 0).all():
            continue
        pred = sub[f"{col}_pred"].values
        true = sub[f"{col}_true"].values
        std  = sub[f"{col}_std"].values.clip(min=1e-6)
        empirical = [float((np.abs(pred - true) <= spnorm.ppf((1+a)/2) * std).mean()) for a in alphas]
        ax.plot(alphas, empirical, color=colors[i % len(colors)], lw=2,
                label=col.replace("_", " ").title(), marker="o", ms=3)
        ax.fill_between(alphas, alphas, empirical, color=colors[i % len(colors)], alpha=0.10)

    ax.set_xlabel("Expected Coverage (α)")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("MC Dropout Uncertainty Calibration")
    ax.legend(framealpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()
    _save_fig(fig, eval_dir / "uncertainty_calibration.pdf", show)


def plot_embedding_pca(model, fu_cfg, device, eval_dir, show=True):
    import torch
    import matplotlib.pyplot as plt
    from src.data.experimental_dataset import Ga2O3ExpDataset, collate_fn
    from torch.utils.data import DataLoader
    from sklearn.decomposition import PCA

    dataset = Ga2O3ExpDataset(
        csv_path=fu_cfg["paths"]["experimental_csv"],
        structures_dir=fu_cfg["paths"]["structures_dir"],
        target_cols=fu_cfg["targets"], augment=False,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_emb, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = model.get_embedding(batch["graph"].to(device),
                                      batch["dopant_spec"],
                                      batch["process"].to(device))
            all_emb.append(emb.cpu().numpy())
            all_labels.extend(batch["dopant_label"])

    embeddings = np.concatenate(all_emb, axis=0)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    unique_labels = sorted(set(all_labels))
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    color_map = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in all_labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], label=lbl, color=color_map[lbl],
                   s=70, edgecolors="k", linewidths=0.4, alpha=0.85)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Multi-Modal Embedding Space (PCA)")
    ax.legend(loc="best", fontsize=9, framealpha=0.7, markerscale=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()
    _save_fig(fig, eval_dir / "embedding_pca.pdf", show)


# ═══════════════════════════════════════════════════════════════════════════════
#  LATEX / CSV TABLE WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _per_dopant_df(oof_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for dopant in sorted(oof_df["dopant_label"].unique()):
        sub = oof_df[oof_df["dopant_label"] == dopant]
        row = {"dopant": dopant, "N": len(sub)}
        for col in target_cols:
            valid = sub[[f"{col}_pred", f"{col}_true"]].dropna(subset=[f"{col}_true"])
            n_v = len(valid)
            row[f"{col}_N"] = n_v
            if n_v >= 2:
                p, t = valid[f"{col}_pred"].values, valid[f"{col}_true"].values
                row[f"{col}_MAE"]  = float(np.mean(np.abs(p - t)))
                row[f"{col}_RMSE"] = float(np.sqrt(np.mean((p - t) ** 2)))
                ss_res = np.sum((t - p) ** 2)
                ss_tot = np.sum((t - t.mean()) ** 2)
                row[f"{col}_R2"] = float(1 - ss_res / (ss_tot + 1e-12))
            else:
                row[f"{col}_MAE"] = row[f"{col}_RMSE"] = row[f"{col}_R2"] = float("nan")
        rows.append(row)
    # Add Overall
    row = {"dopant": "Overall", "N": len(oof_df)}
    for col in target_cols:
        valid = oof_df[[f"{col}_pred", f"{col}_true"]].dropna(subset=[f"{col}_true"])
        n_v = len(valid)
        row[f"{col}_N"] = n_v
        if n_v >= 2:
            p, t = valid[f"{col}_pred"].values, valid[f"{col}_true"].values
            row[f"{col}_MAE"]  = float(np.mean(np.abs(p - t)))
            row[f"{col}_RMSE"] = float(np.sqrt(np.mean((p - t) ** 2)))
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - t.mean()) ** 2)
            row[f"{col}_R2"] = float(1 - ss_res / (ss_tot + 1e-12))
        else:
            row[f"{col}_MAE"] = row[f"{col}_RMSE"] = row[f"{col}_R2"] = float("nan")
    rows.append(row)
    return pd.DataFrame(rows)


def save_per_dopant_latex(per_dop: pd.DataFrame, target_cols: list[str], out_path: Path):
    col_headers = [
        col.replace("photo_dark_ratio", "P/D Ratio")
           .replace("vacancy_concentration", "Vac. Conc.")
        for col in target_cols
    ]
    n_metric_cols = len(target_cols) * 2
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Per-dopant out-of-fold evaluation metrics (log$_{10}$ scale).}",
        r"\label{tab:per_dopant_metrics}",
        r"\begin{tabular}{l" + "r" * (1 + n_metric_cols) + r"}",
        r"\toprule",
        "Dopant & $N$" + "".join(f" & \\multicolumn{{2}}{{c}}{{{h}}}" for h in col_headers) + r" \\",
        r"\cmidrule(lr){3-" + str(2 + n_metric_cols) + r"}",
        " & " + "".join(" & MAE & RMSE" for _ in col_headers) + r" \\",
        r"\midrule",
    ]
    for _, row in per_dop.iterrows():
        line = f"{row['dopant']} & {int(row['N'])}"
        for col in target_cols:
            mae, rmse = row.get(f"{col}_MAE", float("nan")), row.get(f"{col}_RMSE", float("nan"))
            line += (" & -- & --" if np.isnan(mae) else f" & {mae:.3f} & {rmse:.3f}")
        line += r" \\"
        if row["dopant"] == "Overall":
            lines.append(r"\midrule")
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"  Per-dopant LaTeX → {out_path}")


def save_summary_latex(oof_df: pd.DataFrame, target_cols: list[str], out_path: Path):
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Overall 5-fold CV metrics (log$_{10}$ scale).}",
        r"\label{tab:cv_metrics}",
        r"\begin{tabular}{lrrr}", r"\toprule", r"Property & MAE & RMSE & $R^2$ \\", r"\midrule",
    ]
    for col in target_cols:
        valid = oof_df[[f"{col}_pred", f"{col}_true"]].dropna(subset=[f"{col}_true"])
        if valid.empty:
            continue
        p, t = valid[f"{col}_pred"].values, valid[f"{col}_true"].values
        mae  = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        r2   = float(1 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-12))
        label = col.replace("photo_dark_ratio", "Photo/Dark Ratio") \
                   .replace("vacancy_concentration", "Vacancy Concentration") \
                   .replace("_", " ").title()
        lines.append(f"{label} & {mae:.3f} & {rmse:.3f} & {r2:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"  Summary LaTeX → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, path: Path, show: bool):
    import matplotlib.pyplot as plt
    fig.savefig(str(path))
    logger.info(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


def _print_summary(report: dict, target_cols: list[str]):
    print("\n" + "=" * 60)
    print(" Ga2O3-Net — Evaluation Summary")
    print("=" * 60)
    om = report.get("overall_metrics", {})
    for col in target_cols:
        m = om.get(col, {})
        if not m or m.get("N", 0) == 0:
            continue
        print(f"\n  {col}  (N={m['N']})")
        print(f"    MAE  = {m.get('MAE','?')}")
        print(f"    RMSE = {m.get('RMSE','?')}")
        print(f"    R²   = {m.get('R2','?')}")
        print(f"    Slope= {m.get('slope','?')}  (ideal 1.0)")
    print("\n  → Full report: results/eval/eval_summary.txt")
    print("  → All metrics: results/eval/eval_report.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
