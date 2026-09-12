"""Phase 51 — Fine-tune driver for V51 (Frozen Multi-Expert Fusion).

Substitutes Ga2O3Net's CGCNN encoder with a MultiEncoderFusion of N
frozen pretrained encoders + small self-attention.

Self-contained: directly drives FinetuneTrainer.run_cv per seed, then
aggregates OOF rows and saves `oof_predictions.csv` + `cv_results.csv`
+ `stage3_model.pt` (single seed) — output is compatible with the
existing eval scripts (`eval_phase42_vs_baselines.py`,
`eval_physics_diag_table.py`).

This is a NEW file. Does not modify the V5 fine-tune driver.

Usage:
    conda activate ga2o3
    python scripts/04b_finetune_v51.py \\
        --multimodal-config config/multimodal_v51.yaml \\
        --fusion-config     config/fusion_v51_vc.yaml \\
        --gpu 0 --ensemble-seeds 5 --cv-folds 10 --cv-group-by doi \\
        --single-target vacancy_concentration \\
        --session phase51v51_multiexpert_vc_5seed
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

# Import the V5 driver as a module (it's import-safe — main is gated by
# `if __name__ == "__main__":`). We reuse `_physics_kwargs` to translate
# the YAML physics: section into Ga2O3Net constructor kwargs.
import importlib.util
_v5_path = str(PROJ / "scripts" / "04_finetune_predict.py")
_spec = importlib.util.spec_from_file_location("v5_driver", _v5_path)
_v5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v5)
_physics_kwargs = _v5._physics_kwargs

from src.data.experimental_dataset import Ga2O3ExpDataset
from src.models.ga2o3_net_v51 import build_v51_model
from src.training.finetune_trainer import FinetuneTrainer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_v51_from_configs(mm_cfg: dict, fu_cfg: dict, device,
                             encoder_specs_override=None):
    hc = fu_cfg["head"]
    dopant_cfg = mm_cfg.get("dopant_stream")
    dopant_kwargs = None
    if dopant_cfg:
        dopant_kwargs = {
            "n_dopants":       dopant_cfg.get("n_dopants", 13),
            "embed_dim":       dopant_cfg.get("embed_dim", 8),
            "hidden_dim":      dopant_cfg.get("hidden_dim", 32),
            "out_dim":         dopant_cfg.get("out_dim", 16),
            "dropout":         dopant_cfg.get("dropout", 0.2),
            "use_descriptors": dopant_cfg.get("use_descriptors", True),
        }

    me_cfg = mm_cfg.get("multi_encoder", {})
    encoder_specs = encoder_specs_override or me_cfg.get("encoders", [])
    if not encoder_specs:
        raise ValueError("multi_encoder.encoders is empty in V51 config")

    resolved_specs = []
    for s in encoder_specs:
        s2 = dict(s)
        ckpt = s2["ckpt_path"]
        if not Path(ckpt).is_absolute():
            ckpt = str((PROJ / ckpt).resolve())
        s2["ckpt_path"] = ckpt
        resolved_specs.append(s2)

    fusion_dim = me_cfg.get("attn_dim", 64)
    fusion_pool = me_cfg.get("pool", "cls")
    n_attn_heads = me_cfg.get("n_heads", 4)
    n_attn_layers = me_cfg.get("n_layers", 1)
    attn_dropout = me_cfg.get("attn_dropout", 0.1)
    ffn_hidden_dim = me_cfg.get("ffn_hidden_dim", 128)

    model = build_v51_model(
        encoder_specs=resolved_specs,
        target_cols=fu_cfg["targets"],
        fusion_dim=fusion_dim,
        fusion_pool=fusion_pool,
        n_attn_heads=n_attn_heads,
        n_attn_layers=n_attn_layers,
        attn_dropout=attn_dropout,
        ffn_hidden_dim=ffn_hidden_dim,
        fusion_mode=mm_cfg["fusion"]["mode"],
        composition_kwargs={
            "hidden_dim": mm_cfg["composition_stream"]["hidden_dim"],
            "out_dim": mm_cfg["composition_stream"]["out_dim"],
            "dropout": mm_cfg["composition_stream"]["dropout"],
        },
        process_kwargs={
            "in_dim":              mm_cfg["process_stream"]["in_dim"],
            "continuous_dim":      mm_cfg["process_stream"].get("continuous_dim", 5),
            "method_embed_dim":    mm_cfg["process_stream"].get("method_embed_dim", 3),
            "n_methods":           mm_cfg["process_stream"].get("n_methods", 6),
            "substrate_embed_dim": mm_cfg["process_stream"].get("substrate_embed_dim", 3),
            "n_substrates":        mm_cfg["process_stream"].get("n_substrates", 7),
            "hidden_dim":          64,
            "out_dim":             mm_cfg["process_stream"]["out_dim"],
            "dropout":             mm_cfg["process_stream"]["dropout"],
        },
        dopant_kwargs=dopant_kwargs,
        head_hidden_dims=hc["hidden_dims"],
        head_dropout=hc["dropout"],
        head_type=hc.get("type", "single"),
        num_experts=hc.get("num_experts", 4),
        moe_routing=hc.get("routing", "soft"),
        **_physics_kwargs(mm_cfg),
    ).to(device)
    return model, dopant_kwargs


def _aggregate_seed_oof(seed_oof_list: list[list[dict]],
                        target_cols: list[str]) -> pd.DataFrame:
    """Average OOF predictions across seeds. Returns DataFrame with
    sample_idx + per-target {pred, std, true}."""
    rows_by_seed = {}
    for seed_idx, rows in enumerate(seed_oof_list):
        df = pd.DataFrame(rows)
        if "sample_idx" not in df.columns:
            raise ValueError(f"OOF rows missing sample_idx (seed {seed_idx})")
        rows_by_seed[seed_idx] = df

    base = list(rows_by_seed.values())[0].sort_values("sample_idx").reset_index(drop=True)
    out = base[["sample_idx"]].copy()
    if "dopant_label" in base.columns:
        out["dopant_label"] = base["dopant_label"]
    elif "cation_label" in base.columns:
        out["dopant_label"] = base["cation_label"]

    for tcol in target_cols:
        pred_col = f"{tcol}_pred"
        true_col = f"{tcol}_true"
        std_col = f"{tcol}_std"
        # Stack per-seed predictions
        pred_stack, std_stack, true_vals = [], [], None
        for s_idx, df in rows_by_seed.items():
            df_s = df.sort_values("sample_idx").reset_index(drop=True)
            if pred_col in df_s.columns:
                pred_stack.append(df_s[pred_col].to_numpy(dtype=np.float32))
            if std_col in df_s.columns:
                std_stack.append(df_s[std_col].to_numpy(dtype=np.float32))
            if true_vals is None and true_col in df_s.columns:
                true_vals = df_s[true_col].to_numpy(dtype=np.float32)
        if pred_stack:
            arr = np.stack(pred_stack, axis=0)
            out[pred_col] = arr.mean(axis=0)
            # Inter-seed std as additional uncertainty signal
            if len(pred_stack) > 1:
                inter_std = arr.std(axis=0)
                if std_stack:
                    intra = np.stack(std_stack, axis=0).mean(axis=0)
                    out[std_col] = np.sqrt(intra ** 2 + inter_std ** 2)
                else:
                    out[std_col] = inter_std
            elif std_stack:
                out[std_col] = std_stack[0]
        if true_vals is not None:
            out[true_col] = true_vals
    return out


def _fit_platt_sputter(df_oof: pd.DataFrame, dataset_df: pd.DataFrame,
                        target_cols: list[str]) -> dict:
    """Fit y = a*pred + b on sputter labeled subset."""
    is_sputter_per_idx = (
        dataset_df["method"].fillna("").astype(str).str.lower()
        .str.contains("sputter").to_numpy()
    )
    platt = {}
    for tcol in target_cols:
        pred_col = f"{tcol}_pred"
        true_col = f"{tcol}_true"
        if pred_col not in df_oof.columns or true_col not in df_oof.columns:
            continue
        m = df_oof[true_col].notna() & df_oof[pred_col].notna()
        sample_idx = df_oof["sample_idx"].astype(int).to_numpy()
        sp_mask = m.to_numpy() & is_sputter_per_idx[sample_idx]
        if sp_mask.sum() < 2:
            slope, intercept = 1.0, 0.0
        else:
            y_raw = df_oof.loc[sp_mask, pred_col].to_numpy()
            y_true = df_oof.loc[sp_mask, true_col].to_numpy()
            slope, intercept = np.polyfit(y_raw, y_true, 1)
        platt[tcol] = (float(slope), float(intercept))
        df_oof[f"{tcol}_pred_platt"] = (
            df_oof[pred_col].astype(float) * slope + intercept
        )
    return platt


def main():
    parser = argparse.ArgumentParser(
        description="Phase 51 V51 fine-tune driver (frozen multi-expert fusion)")
    parser.add_argument("--multimodal-config", default="config/multimodal_v51.yaml")
    parser.add_argument("--fusion-config", default="config/fusion_v51_vc.yaml")
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--ensemble-seeds", type=int, default=1)
    parser.add_argument("--single-target", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--cv-group-by", default=None, choices=[None, "doi"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.multimodal_config) as f:
        mm_cfg = yaml.safe_load(f)
    with open(args.fusion_config) as f:
        fu_cfg = yaml.safe_load(f)

    if args.single_target:
        fu_cfg["targets"] = [args.single_target]
    if args.cv_folds is not None:
        fu_cfg.setdefault("training", {})["cv_folds"] = args.cv_folds
    if args.session:
        fu_cfg["paths"]["results_dir"] = f"results/{args.session}"

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU {args.gpu}")
    else:
        device = torch.device("cpu")

    # ---- Dataset (same as V5) ----
    process_layout = mm_cfg.get("process_stream", {}).get("layout", "v2")
    prefer_ordered = mm_cfg.get("structure_stream", {}).get(
        "prefer_ordered", False)
    dataset = Ga2O3ExpDataset(
        csv_path=fu_cfg["paths"]["experimental_csv"],
        structures_dir=fu_cfg["paths"]["structures_dir"],
        target_cols=fu_cfg["targets"],
        augment=False,
        mask_noconc_labels=fu_cfg.get("data", {}).get("mask_noconc_labels", False),
        mask_undoped_labels=fu_cfg.get("data", {}).get("mask_undoped_labels", False),
        min_conc_threshold=fu_cfg.get("data", {}).get("min_conc_threshold", 0.0),
        mask_carrier_vc=fu_cfg.get("data", {}).get("mask_carrier_vc", False),
        exclude_elements=fu_cfg.get("data", {}).get("exclude_elements", None),
        include_elements=fu_cfg.get("data", {}).get("include_elements", None),
        include_keep_undoped=fu_cfg.get("data", {}).get("include_keep_undoped", True),
        process_layout=process_layout,
        prefer_ordered=prefer_ordered,
    )
    logger.info(f"Samples: {len(dataset)}")

    # ---- Per-seed CV ----
    seeds_arg = int(args.ensemble_seeds)
    if seeds_arg <= 1:
        seeds = [42]
    else:
        seeds = [42, 123, 456, 789, 1024][:seeds_arg]

    all_seed_oof = []
    all_seed_results = []
    last_model = None

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'='*60}\n  V51 SEED {seed_idx+1}/{len(seeds)} (seed={seed})\n{'='*60}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_seed, _ = _build_v51_from_configs(mm_cfg, fu_cfg, device)
        trainer = FinetuneTrainer(
            model=model_seed, dataset=dataset, config=fu_cfg, device=device,
        )
        cv_groups = None
        if args.cv_group_by == "doi":
            cv_groups = dataset.df["doi"].fillna("__NA__").values

        results = trainer.run_cv(random_state=seed, groups=cv_groups)
        oof_rows = trainer.last_oof_rows if hasattr(trainer, "last_oof_rows") else []
        if not oof_rows and isinstance(results, tuple) and len(results) == 2:
            results, oof_rows = results
        all_seed_oof.append(oof_rows)
        all_seed_results.append(results)
        last_model = model_seed

    # ---- Aggregate OOF + Platt + save ----
    results_dir = Path(fu_cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    oof_df = _aggregate_seed_oof(all_seed_oof, fu_cfg["targets"])
    platt = _fit_platt_sputter(oof_df, dataset.df, fu_cfg["targets"])
    logger.info(f"Platt fits: {platt}")

    oof_path = results_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"Saved OOF: {oof_path}")

    # ---- Save model state + config ----
    bundle_path = results_dir / "stage3_model.pt"
    torch.save({
        "model_state_dict": last_model.state_dict(),
        "config": dict(head=fu_cfg["head"], physics=mm_cfg.get("physics", {})),
        "target_cols": fu_cfg["targets"],
    }, str(bundle_path))
    logger.info(f"Saved model bundle: {bundle_path}")

    # ---- Save fusion config copy alongside bundle for eval scripts ----
    import shutil
    fu_copy = results_dir / "fusion_head128.yaml"
    shutil.copy(args.fusion_config, str(fu_copy))
    mm_copy = results_dir / "multimodal.yaml"
    shutil.copy(args.multimodal_config, str(mm_copy))

    # ---- Save cv_results.csv ----
    cv_csv = results_dir / "cv_results.csv"
    with open(cv_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in (all_seed_results[0] if all_seed_results else {}).items():
            try:
                writer.writerow([k, float(v)])
            except (TypeError, ValueError):
                writer.writerow([k, str(v)])
    logger.info(f"Saved cv results: {cv_csv}")
    logger.info("V51 fine-tune complete.")


if __name__ == "__main__":
    main()
