"""
Script 04: Stage 3 — few-shot fine-tuning + evaluation.

Runs stratified k-fold CV with:
  - MLP head: trained end-to-end per fold, scalers fitted on train fold only
  - XGBoost:  trained on fold-specific embeddings (from stage2_fold*/ or
              extracted on-the-fly in the same fold loop)

All CV is leakage-free: scalers, LabelEncoder, and XGBoost are fitted only
on training indices within each fold.

Usage:
    conda activate ga2o3
    python scripts/04_finetune_predict.py
    python scripts/04_finetune_predict.py --xgboost
    python scripts/04_finetune_predict.py --xgboost --gpu 1
    python scripts/04_finetune_predict.py --xgboost --use-cached-embeddings
"""

from __future__ import annotations

import os
import sys
import csv
import pickle
import argparse
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from src.models.ga2o3_net import Ga2O3Net
from src.data.experimental_dataset import Ga2O3ExpDataset
from src.training.finetune_trainer import FinetuneTrainer
from src.utils.metrics import print_metrics_table
from src.utils.visualization import plot_parity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _physics_kwargs(mm_cfg: dict) -> dict:
    """Phase 24 / 25 / 30B — extract physics-stream / envelope / Brouwer flags."""
    phys_cfg = mm_cfg.get("physics") or {}
    return {
        "physics_features":         phys_cfg.get("features", False),
        "physics_features_hidden":  phys_cfg.get("features_hidden", 32),
        "physics_features_out_dim": phys_cfg.get("features_out_dim", 16),
        "physics_features_dropout": phys_cfg.get("features_dropout", 0.1),
        "physics_envelope_pdr":     phys_cfg.get("envelope_pdr", False),
        "physics_envelope_vc":      phys_cfg.get("envelope_vc", False),
        # Phase 25-soft: hinge mode keeps envelope params but doesn't transform output.
        "physics_envelope_apply":   phys_cfg.get("envelope_apply", True),
        # Phase 30B / Path D: anchor BrouwerHeadVC's E_f baseline to the
        # empirical V_O formation-energy proxy (donor/acceptor chemistry).
        "brouwer_anchor_to_proxy":  phys_cfg.get("brouwer_anchor_to_proxy", False),
        # Phase 41 V2: per-method scalar offset on log[V_O] in BrouwerHeadVC.
        "brouwer_use_method_offset": phys_cfg.get("brouwer_use_method_offset", False),
        "brouwer_n_methods":        phys_cfg.get("brouwer_n_methods", 6),
        # Phase 45 V10: per-valence-class scalar offset on log[V_O].
        "brouwer_use_class_offset": phys_cfg.get("brouwer_use_class_offset", False),
        "brouwer_n_classes":        phys_cfg.get("brouwer_n_classes", 4),
        # Phase 46 V11: element-descriptor hypernetwork on log[V_O].
        "brouwer_use_hypernet":     phys_cfg.get("brouwer_use_hypernet", False),
        "hypernet_hidden":          phys_cfg.get("hypernet_hidden", 16),
        "hypernet_mid":             phys_cfg.get("hypernet_mid", 8),
        # Phase 47 V13: small_random init for HN final layer (V11 used zeros, collapsed).
        "hypernet_final_init":      phys_cfg.get("hypernet_final_init", "zeros"),
        # Phase 47 V14: multiplicative form on dopant_term, vs V11/V13 additive on log_VO.
        "hypernet_form":            phys_cfg.get("hypernet_form", "additive"),
        # Phase 47 V15: direct descriptor injection (no HN gating layer).
        "inject_descriptor":        phys_cfg.get("inject_descriptor", False),
        # Phase 43 V5: distil pretrained encoder pretrain_head (bandgap +
        # formation_energy_per_atom) into aux_distill_head over fused emb.
        "pretrain_distill_aux":     phys_cfg.get("pretrain_distill_aux", False),
        "pretrain_distill_dim":     phys_cfg.get("pretrain_distill_dim", 2),
        # Phase 53 V53-γ1: concentration-dependent dopant_term in BrouwerHeadVC.
        "brouwer_use_conc_dep_dopant": phys_cfg.get("brouwer_use_conc_dep_dopant", False),
        "brouwer_conc_ref_at_frac":    phys_cfg.get("brouwer_conc_ref_at_frac", 0.01),
        # Phase 53 V53-ζ: PDR↔V_O latent InfoNCE projection head (trainer adds the loss).
        "contrastive_enabled":  phys_cfg.get("contrastive_enabled", False),
        "contrastive_dim":      phys_cfg.get("contrastive_dim", 32),
        "contrastive_hidden":   phys_cfg.get("contrastive_hidden", 64),
        # Phase 53 V53-α: KROGER frozen expert + aux distill loss.
        "kroger_expert_enabled": phys_cfg.get("kroger_expert_enabled", False),
        "kroger_expert_path":    phys_cfg.get("kroger_expert_path", None),
        # Phase 53 V53-η: KKT-Hardnet Brouwer (full Kröger-Vink charge neutrality)
        "brouwer_use_kkt":       phys_cfg.get("brouwer_use_kkt", False),
        "kkt_ef_swing":          phys_cfg.get("kkt_ef_swing", 0.5),
        "kkt_n_newton":          phys_cfg.get("kkt_n_newton", 8),
        # Phase 54 V54-A1: self-distill aux head consuming V53-ζ pseudo-targets.
        "self_distill_enabled":  phys_cfg.get("self_distill_enabled", False),
        # Phase 54 V54-A2: SNGP sibling head (spectral norm + RFF GP).
        "sngp_enabled":          phys_cfg.get("sngp_enabled", False),
        "sngp_hidden_dim":       phys_cfg.get("sngp_hidden_dim", 256),
        "sngp_rff_dim":          phys_cfg.get("sngp_rff_dim", 512),
        "sngp_ridge":            phys_cfg.get("sngp_ridge", 1e-4),
        "sngp_ema":              phys_cfg.get("sngp_ema", 0.999),
        # Phase 54 V54-B1: charge-state V_O / V_Ga frozen experts (soft distill).
        "charge_vo_expert_path": phys_cfg.get("charge_vo_expert_path", None),
        "vga_expert_path":       phys_cfg.get("vga_expert_path", None),
        # V57-B1-Quartet+1 — 4th DFT-scan frozen expert (synthetic KROGER).
        "dft_scan_expert_path":  phys_cfg.get("dft_scan_expert_path", None),
        # V57-MACE-Latent — 17×256-d MACE-MP-0 → 64-d projection.
        "mace_latent_proj_path": phys_cfg.get("mace_latent_proj_path", None),
        "mace_latent_dim":       phys_cfg.get("mace_latent_dim", 64),
        # V58 — DFT-anchored stream (M4) + FiLM-208 (M8) + DFTAnchoredBrouwerHead (M9).
        "dft_stream_enabled":      phys_cfg.get("dft_stream_enabled", False),
        "dft_features_cache_path": phys_cfg.get("dft_features_cache_path", None),
        "llm_cache_path":          phys_cfg.get("llm_cache_path", None),
        "dft_max_residual":        phys_cfg.get("dft_max_residual", 0.3),
        # Phase 59 — V59-CALM zero-gated Δ-residual (doc §3.4). Extends the
        # V55-Ext film/concat fusion path; NOT the V58 dft_stream concat.
        "calm_enabled":            phys_cfg.get("calm_enabled", False),
        "calm_expert_cache_path":  phys_cfg.get("calm_expert_cache_path", None),
        "calm_dft_cache_path":     phys_cfg.get("calm_dft_cache_path", None),
        "calm_mace_proj_path":     phys_cfg.get("calm_mace_proj_path", None),
        "calm_fusion_rank":        phys_cfg.get("calm_fusion_rank", 4),
        "calm_hidden":             phys_cfg.get("calm_hidden", 16),
        "calm_modality_dropout":   phys_cfg.get("calm_modality_dropout", 0.3),
        "calm_gate_init":          phys_cfg.get("calm_gate_init", 0.0),
        "calm_use_expert":         phys_cfg.get("calm_use_expert", True),
        "calm_use_dft":            phys_cfg.get("calm_use_dft", True),
        "calm_use_mace":           phys_cfg.get("calm_use_mace", True),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: fine-tune + evaluate (leakage-free CV)"
    )
    parser.add_argument("--multimodal-config", default="config/multimodal.yaml")
    parser.add_argument("--fusion-config", default="config/fusion.yaml")
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (default 0). Use -1 for CPU.",
    )
    parser.add_argument(
        "--xgboost", action="store_true",
        help="Also run XGBoost comparison in the same CV fold loop.",
    )
    parser.add_argument(
        "--gp", action="store_true",
        help="Run Gaussian Process regression on raw XenonPy+process features "
             "(no CGCNN) as an additional baseline. Useful to diagnose whether "
             "the structure stream helps or hurts.",
    )
    parser.add_argument(
        "--use-cached-embeddings", action="store_true",
        help="Load fold embeddings pre-extracted by script 03 instead of "
             "re-running the encoder. Faster if you already ran script 03.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=None,
        help="Override number of CV folds (default: fusion.yaml training.cv_folds).",
    )
    parser.add_argument(
        "--ensemble-seeds", type=int, default=1,
        help="Number of random seeds for ensemble averaging (default 1 = no ensemble). "
             "Seeds used: 42, 123, 456, 789, 1024.",
    )
    parser.add_argument(
        "--single-target", default=None,
        choices=["photo_dark_ratio", "vacancy_concentration"],
        help="Override fusion.yaml targets list with a single target. "
             "Useful for Phase 3 comparison: shared-head vs independent-head.",
    )
    parser.add_argument(
        "--session", default=None,
        help="Override results_dir to results/<session> for this run.",
    )
    parser.add_argument(
        "--cv-group-by", default=None, choices=[None, "doi"],
        help="Use GroupKFold on this column instead of StratifiedKFold. "
             "Primary use: --cv-group-by doi guarantees no paper spans a "
             "train/val split (catches paper-level leakage).",
    )
    parser.add_argument(
        "--init-streams-from", default=None,
        help="Phase 13C: path to SSL-pretrained checkpoint (composition_stream + "
             "process_stream + fusion state dicts). Loaded into the initial model "
             "before CV; each fold seed gets a fresh copy via deepcopy.",
    )
    args = parser.parse_args()

    with open(args.multimodal_config) as f:
        mm_cfg = yaml.safe_load(f)
    with open(args.fusion_config) as f:
        fu_cfg = yaml.safe_load(f)

    if args.cv_folds is not None:
        fu_cfg["training"]["cv_folds"] = args.cv_folds

    if args.single_target is not None:
        fu_cfg["targets"] = [args.single_target]
        logger.info(f"--single-target: restricting targets to {fu_cfg['targets']}")

    if args.session is not None:
        fu_cfg["paths"]["results_dir"] = f"results/{args.session}"
        logger.info(f"--session: results_dir = {fu_cfg['paths']['results_dir']}")

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading experimental dataset...")
    process_layout = mm_cfg.get("process_stream", {}).get("layout", "v2")
    # Phase 49 V17: optional CHGNet pre-extracted embedding cache
    encoder_backbone = mm_cfg.get("structure_stream", {}).get(
        "encoder_backbone", "cgcnn")
    chgnet_emb_cache = mm_cfg.get("structure_stream", {}).get(
        "chgnet_emb_cache_path")
    # Phase 50.A: when True, dataset loads SQS+CHGNet-relaxed ordered CIFs
    # from data/structures/ordered/<key>.relaxed.cif instead of disordered
    # partial-occupancy structures. Concentration enters geometry via
    # supercell shape + dopant placement + relaxed local distortion.
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
        chgnet_emb_cache_path=chgnet_emb_cache,
        prefer_ordered=prefer_ordered,
    )
    logger.info(f"Samples: {len(dataset)}")
    logger.info(f"Dopant types: {sorted(set(dataset.get_dopant_labels()))}")

    # ── Build initial model ───────────────────────────────────────────────────
    logger.info("Building model from pretrained encoder...")
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
        logger.info(f"DopantStream enabled: {dopant_kwargs}")
    model = Ga2O3Net.from_pretrained(
        encoder_ckpt=mm_cfg["structure_stream"]["pretrained_ckpt"],
        target_cols=fu_cfg["targets"],
        fusion_mode=mm_cfg["fusion"]["mode"],
        encoder_backbone=encoder_backbone,
        encoder_kwargs=({"chgnet_emb_cache_path": chgnet_emb_cache}
                        if encoder_backbone == "chgnet" else None),
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

    # Apply structure_scale from multimodal.yaml (default 1.0 = full structure stream)
    structure_scale = mm_cfg["fusion"].get("structure_scale", 1.0)
    model.structure_scale = structure_scale
    if structure_scale != 1.0:
        logger.info(f"Structure stream scale: {structure_scale} "
                    f"({'disabled' if structure_scale == 0.0 else 'reduced'})")

    # ── XGBoost with cached embeddings (fast path) ────────────────────────────
    if args.xgboost and args.use_cached_embeddings:
        logger.info("Using pre-cached fold embeddings from script 03...")
        xgb_results = _run_xgb_from_cache(fu_cfg, dataset)
        print_metrics_table(xgb_results, title="XGBoost — CV Results (cached embeddings)")

    # ── MLP + optional integrated XGBoost ────────────────────────────────────
    cv_folds = fu_cfg["training"].get("cv_folds", 3)

    # Inject GP flag into config so FinetuneTrainer can read it
    if args.gp:
        fu_cfg.setdefault("gp", {})["enabled"] = True

    # ── Ensemble seed loop ───────────────────────────────────────────────────
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024]
    n_seeds = max(1, min(args.ensemble_seeds, len(ENSEMBLE_SEEDS)))
    seeds = ENSEMBLE_SEEDS[:n_seeds]

    logger.info(
        f"\nStarting Stage 3 {cv_folds}-fold CV "
        f"({'MLP + XGBoost' if args.xgboost and not args.use_cached_embeddings else 'MLP only'}) "
        f"× {n_seeds} seed{'s' if n_seeds > 1 else ''}..."
    )

    all_seed_oof = []
    all_seed_results = []
    all_trainers: list = []
    last_trainer = None

    for seed_idx, seed in enumerate(seeds):
        if n_seeds > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"  SEED {seed_idx+1}/{n_seeds} (random_state={seed})")
            logger.info(f"{'='*60}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Fresh model per seed (head weights re-initialised)
        model_seed = Ga2O3Net.from_pretrained(
            encoder_ckpt=mm_cfg["structure_stream"]["pretrained_ckpt"],
            target_cols=fu_cfg["targets"],
            fusion_mode=mm_cfg["fusion"]["mode"],
            encoder_backbone=encoder_backbone,
            encoder_kwargs=({"chgnet_emb_cache_path": chgnet_emb_cache}
                            if encoder_backbone == "chgnet" else None),
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
            head_hidden_dims=fu_cfg["head"]["hidden_dims"],
            head_dropout=fu_cfg["head"]["dropout"],
            head_type=fu_cfg["head"].get("type", "single"),
            num_experts=fu_cfg["head"].get("num_experts", 4),
            moe_routing=fu_cfg["head"].get("routing", "soft"),
            **_physics_kwargs(mm_cfg),
        ).to(device)
        model_seed.structure_scale = structure_scale

        # Phase 13C: warm-start streams + fusion from SSL pretraining checkpoint.
        # Heads stay randomly initialised — they are target-specific and SSL had
        # no labels, so SSL weights for the head do not exist. Encoder is also
        # untouched (stays at MP-pretrained weights).
        if args.init_streams_from is not None:
            ssl_ckpt = torch.load(args.init_streams_from, map_location=device, weights_only=False)
            model_seed.composition_stream.load_state_dict(ssl_ckpt["composition_stream"])
            model_seed.process_stream.load_state_dict(ssl_ckpt["process_stream"])
            # Fusion: only load if the SSL fusion mode matches the current model
            if ssl_ckpt.get("fusion_mode") == mm_cfg["fusion"]["mode"]:
                model_seed.fusion.load_state_dict(ssl_ckpt["fusion"])
                logger.info(f"  [seed {seed}] Warm-started streams + fusion from SSL: {args.init_streams_from}")
            else:
                logger.warning(
                    f"  [seed {seed}] SSL ckpt fusion_mode={ssl_ckpt.get('fusion_mode')!r} "
                    f"!= current {mm_cfg['fusion']['mode']!r} — fusion NOT loaded.")

        trainer = FinetuneTrainer(
            model=model_seed,
            dataset=dataset,
            config=fu_cfg,
            device=device,
            xgboost=args.xgboost and not args.use_cached_embeddings,
        )
        cv_groups = None
        if args.cv_group_by == "doi":
            cv_groups = dataset.df["doi"].fillna("__NA__").values
            n_unique_groups = len(set(cv_groups))
            if n_unique_groups < fu_cfg["training"]["cv_folds"]:
                raise ValueError(
                    f"GroupKFold needs unique groups ({n_unique_groups}) "
                    f">= cv_folds ({fu_cfg['training']['cv_folds']})."
                )
            logger.info(
                f"CV groups: --cv-group-by doi → "
                f"{n_unique_groups} unique papers across {len(cv_groups)} rows"
            )
        cv_results, oof_rows = trainer.run_cv(random_state=seed, groups=cv_groups)
        all_seed_results.append(cv_results)
        all_seed_oof.append(oof_rows)
        all_trainers.append(trainer)
        last_trainer = trainer

    # ── Ensemble averaging ───────────────────────────────────────────────────
    if n_seeds > 1:
        logger.info(f"\nAveraging OOF predictions across {n_seeds} seeds...")
        oof_dfs = [pd.DataFrame(rows) for rows in all_seed_oof]
        merged = pd.concat(oof_dfs)
        pred_cols = [c for c in merged.columns if c.endswith("_pred") or c.endswith("_pred_platt")]
        std_cols  = [c for c in merged.columns if c.endswith("_std")]
        true_cols = [c for c in merged.columns if c.endswith("_true")]

        agg_dict = {c: "mean" for c in pred_cols + std_cols}
        agg_dict.update({c: "first" for c in true_cols})
        agg_dict["dopant_label"] = "first"

        oof_df = merged.groupby("sample_idx").agg(agg_dict).reset_index()
        oof_rows = oof_df.to_dict("records")

        # Average CV metrics across seeds
        cv_results = {}
        for key in all_seed_results[0]:
            vals = [r[key] for r in all_seed_results if key in r]
            cv_results[key] = float(np.mean(vals))
    else:
        oof_rows = all_seed_oof[0]
        cv_results = all_seed_results[0]

    trainer = last_trainer

    os.makedirs("results", exist_ok=True)
    print_metrics_table(cv_results, title=f"Stage 3 — {cv_folds}-Fold CV Results")

    # Save CV metrics to CSV
    results_path = Path(fu_cfg["paths"]["results_dir"]) / "cv_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _save_results_csv(cv_results, str(results_path))
    logger.info(f"CV results saved → {results_path}")

    # Save OOF predictions for evaluation script
    if oof_rows:
        oof_df = pd.DataFrame(oof_rows).sort_values("sample_idx").reset_index(drop=True)
        oof_path = results_path.parent / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        logger.info(f"OOF predictions saved → {oof_path}")

    # ── Train final ensemble (one full-data model per CV seed) ────────────────
    # Single-seed final models tend to collapse to the training mean under the
    # strong regularization stack (uncertainty-weighted loss + InfoNCE + mixup
    # + pseudo-labeling). The 5-seed CV preserves diversity in OOF metrics; we
    # mirror that diversity at deployment by saving N final models and
    # averaging at inference. See docs/experiment_log.md §13.22.
    logger.info(
        f"\nTraining {n_seeds} final ensemble member(s) on full dataset..."
    )
    final_state_dicts: list[dict] = []
    final_target_means: list = []
    final_target_stds: list = []
    final_seeds_used: list[int] = []
    for seed_idx, (seed, member_trainer) in enumerate(zip(seeds, all_trainers)):
        logger.info(f"  Final member {seed_idx+1}/{n_seeds} (seed={seed})")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        member_model = member_trainer.train_final_model()
        final_state_dicts.append(
            {k: v.detach().cpu() for k, v in member_model.state_dict().items()}
        )
        final_target_means.append(getattr(member_trainer, "_final_target_mean", None))
        final_target_stds.append(getattr(member_trainer, "_final_target_std", None))
        final_seeds_used.append(seed)

    # Re-fit Platt on the ensemble-averaged OOF predictions: the deployed
    # ensemble averages members, so the calibration must be fit on the same
    # quantity (averaging member-specific Platts is incorrect — they were each
    # fit to their own seed's OOF distribution, not to the ensemble mean).
    ensemble_platt: dict[str, tuple[float, float]] = {}
    if oof_rows and n_seeds > 1:
        from sklearn.linear_model import LinearRegression
        for col in fu_cfg["targets"]:
            preds, trues = [], []
            for r in oof_rows:
                p = r.get(f"{col}_pred", float("nan"))
                t = r.get(f"{col}_true", float("nan"))
                if not (np.isnan(p) or np.isnan(t)):
                    preds.append(p)
                    trues.append(t)
            if len(preds) >= 5:
                lr = LinearRegression().fit(
                    np.array(preds).reshape(-1, 1), np.array(trues)
                )
                a, b = float(lr.coef_[0]), float(lr.intercept_)
                ensemble_platt[col] = (a, b)
                logger.info(
                    f"  Ensemble-OOF Platt [{col}]: y_cal = {a:.3f} * y + {b:.3f}"
                )
            else:
                ensemble_platt[col] = (1.0, 0.0)
    else:
        ensemble_platt = getattr(last_trainer, "_platt_params", None) or {}

    # Use first seed's process / composition scalers — all members fitted on
    # the same all-data view, so scalers are essentially identical.
    template_model = member_model  # last one trained, scalers fitted on all data
    _save_ensemble(
        template_model=template_model,
        state_dicts=final_state_dicts,
        config=fu_cfg,
        path=results_path.parent / "stage3_model.pt",
        target_mean=final_target_means[0] if final_target_means else None,
        target_std=final_target_stds[0] if final_target_stds else None,
        platt_params=ensemble_platt,
        seeds_used=final_seeds_used,
    )
    logger.info(
        f"Ensemble bundle ({n_seeds} members) saved → "
        f"{results_path.parent / 'stage3_model.pt'}"
    )


def _save_ensemble(
    template_model,
    state_dicts: list,
    config: dict,
    path,
    target_mean=None,
    target_std=None,
    platt_params=None,
    seeds_used=None,
):
    """
    Save N-member ensemble bundle. Format:
      model_state_dicts: list[dict] — one state_dict per member
      ensemble_seeds:    list[int]  — random_state used for each member
      process_scaler / composition_scaler: shared (members fit on identical data)
      target_mean / target_std: shared (computed on full dataset)
      platt_params: re-fit on ensemble-averaged OOF (not member-averaged)
      config: full fusion.yaml snapshot
    """
    bundle = {
        "model_state_dicts": state_dicts,
        "ensemble_seeds": seeds_used or [],
        "process_scaler": template_model.process_stream._scaler,
        "composition_scaler": template_model.composition_stream._scaler,
        "composition_raw_dim": template_model.composition_stream.raw_dim,
        "target_cols": config["targets"],
        "config": config,
        "target_mean": target_mean,
        "target_std": target_std,
        "platt_params": platt_params or {},
    }
    torch.save(bundle, path)


def _save_model(model, config: dict, path, trainer=None):
    """
    Save trained Ga2O3Net + scalers for deployment.

    Saved bundle contains:
      model_state_dict  — torch state dict (all head/stream weights)
      process_scaler    — sklearn StandardScaler fitted on all training data
      composition_scaler— sklearn StandardScaler fitted on all training data
      composition_raw_dim — XenonPy feature dimension
      target_cols       — list of target property names
      config            — full fusion.yaml config snapshot
      target_mean/std   — target standardization stats (if enabled)
      platt_params      — post-hoc slope correction (if fitted)
    """
    bundle = {
        "model_state_dict": model.state_dict(),
        "process_scaler": model.process_stream._scaler,
        "composition_scaler": model.composition_stream._scaler,
        "composition_raw_dim": model.composition_stream.raw_dim,
        "target_cols": config["targets"],
        "config": config,
    }
    if trainer is not None:
        bundle["target_mean"] = getattr(trainer, "_final_target_mean", None)
        bundle["target_std"] = getattr(trainer, "_final_target_std", None)
        bundle["platt_params"] = getattr(trainer, "_platt_params", None)
    torch.save(bundle, path)


def load_model(model_path: str, multimodal_config_path: str) -> "Ga2O3Net":
    """
    Load a saved Ga2O3Net for inference.

    Usage:
        model = load_model("results/stage3_model.pt", "config/multimodal.yaml")
        model.eval()
        # Single prediction
        mean, std = model.mc_predict(graph, ["Fe:0.0265"], process_tensor)

    Args:
        model_path: Path to stage3_model.pt saved by script 04.
        multimodal_config_path: Path to config/multimodal.yaml.

    Returns:
        Ga2O3Net with weights and scalers restored.
    """
    import yaml
    from src.models.ga2o3_net import Ga2O3Net, EnsembleGa2O3Net

    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    with open(multimodal_config_path) as f:
        mm_cfg = yaml.safe_load(f)

    hc = bundle["config"]["head"]
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

    _backbone_load = mm_cfg.get("structure_stream", {}).get(
        "encoder_backbone", "cgcnn")
    _chgnet_cache_load = mm_cfg.get("structure_stream", {}).get(
        "chgnet_emb_cache_path")

    def _build_one() -> Ga2O3Net:
        m = Ga2O3Net.from_pretrained(
            encoder_ckpt=mm_cfg["structure_stream"]["pretrained_ckpt"],
            target_cols=bundle["target_cols"],
            fusion_mode=mm_cfg["fusion"]["mode"],
            encoder_backbone=_backbone_load,
            encoder_kwargs=({"chgnet_emb_cache_path": _chgnet_cache_load}
                            if _backbone_load == "chgnet" else None),
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
            **_physics_kwargs(mm_cfg),
        )
        m.process_stream._scaler = bundle["process_scaler"]
        m.composition_stream._scaler = bundle["composition_scaler"]
        if bundle.get("composition_raw_dim"):
            raw_dim = bundle["composition_raw_dim"]
            if raw_dim != m.composition_stream.raw_dim:
                import torch.nn as nn
                m.composition_stream.raw_dim = raw_dim
                m.composition_stream.mlp[0] = nn.Linear(
                    raw_dim, m.composition_stream.mlp[0].out_features
                )
        return m

    def _assert_calm_match(res, where):
        # Phase 59 (v59-impl-verify bug #2): a strict=False load can silently leave the
        # zero-gated CALM residual at random init if the rebuilt architecture (from the
        # multimodal config) disagrees with the saved state_dict — which would evaluate a
        # partly-random model and make the §9 gate verdict meaningless. Fail loud instead.
        bad = [k for k in (list(res.missing_keys) + list(res.unexpected_keys))
               if ("calm_residual" in k) or k.startswith("_calm")]
        if bad:
            raise RuntimeError(
                f"CALM state_dict mismatch in {where}: model would run partly-random. "
                f"Likely wrong multimodal config for this bundle. Offending keys: {bad}")

    state_dicts = bundle.get("model_state_dicts")
    if state_dicts:
        members = []
        for sd in state_dicts:
            mem = _build_one()
            res = mem.load_state_dict(sd, strict=False)
            _assert_calm_match(res, "ensemble member")
            mem.eval()
            members.append(mem)
        model = EnsembleGa2O3Net(members)
        logger.info(
            f"Loaded ensemble bundle: {len(members)} members "
            f"(seeds={bundle.get('ensemble_seeds', '?')})"
        )
    else:
        # Legacy single-member bundle
        model = _build_one()
        res = model.load_state_dict(bundle["model_state_dict"], strict=False)
        _assert_calm_match(res, "single-member bundle")

    # Attach inference-time calibration. Raw model outputs are in z-score
    # space (if target_standardize was used during training). Inference
    # pipeline: pred_raw = model.forward(...);
    #          pred = pred_raw * target_std + target_mean;
    #          pred_cal = a * pred + b   (if platt_params present)
    model._target_mean = bundle.get("target_mean")
    model._target_std = bundle.get("target_std")
    model._platt_params = bundle.get("platt_params")

    model.eval()
    return model


def _run_xgb_from_cache(fu_cfg: dict, dataset: Ga2O3ExpDataset) -> dict:
    """
    Run XGBoost using pre-extracted fold embeddings from script 03.
    Each fold uses its own train/val embeddings and targets — leakage-free.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.error("pip install xgboost")
        return {}
    from src.utils.metrics import regression_metrics
    from collections import defaultdict

    cv_folds = fu_cfg["training"].get("cv_folds", 3)
    xgb_cfg = fu_cfg.get("xgboost", {})
    target_cols = fu_cfg["targets"]

    fold_metrics: dict[str, list] = defaultdict(list)

    for fold in range(1, cv_folds + 1):
        fold_dir = Path(f"data/processed/stage2_fold{fold}")
        required = [
            fold_dir / "embeddings_train.npy",
            fold_dir / "embeddings_val.npy",
            fold_dir / "targets_train.npy",
            fold_dir / "targets_val.npy",
        ]
        if not all(p.exists() for p in required):
            logger.warning(
                f"Fold {fold} embedding files missing. "
                "Run script 03 first or remove --use-cached-embeddings."
            )
            continue

        train_emb = np.load(fold_dir / "embeddings_train.npy")
        val_emb = np.load(fold_dir / "embeddings_val.npy")
        train_tgt = np.load(fold_dir / "targets_train.npy")
        val_tgt = np.load(fold_dir / "targets_val.npy")

        logger.info(
            f"  Fold {fold}: train_emb={train_emb.shape}, val_emb={val_emb.shape}"
        )

        for i, col in enumerate(target_cols):
            y_train, y_val = train_tgt[:, i], val_tgt[:, i]
            valid_tr = ~np.isnan(y_train)
            valid_v = ~np.isnan(y_val)
            if valid_tr.sum() < 2 or valid_v.sum() < 1:
                continue

            xgb = XGBRegressor(
                n_estimators=xgb_cfg.get("n_estimators", 200),
                max_depth=xgb_cfg.get("max_depth", 5),
                learning_rate=xgb_cfg.get("learning_rate", 0.05),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                random_state=42,
                verbosity=0,
            )
            xgb.fit(train_emb[valid_tr], y_train[valid_tr])
            pred = xgb.predict(val_emb[valid_v])
            for k, v in regression_metrics(pred, y_val[valid_v]).items():
                fold_metrics[f"{col}_{k}"].append(v)

    # Aggregate
    summary = {}
    for k, vals in fold_metrics.items():
        summary[f"xgb_{k}_mean"] = float(np.mean(vals))
        summary[f"xgb_{k}_std"] = float(np.std(vals))
    return summary


def _save_results_csv(results: dict, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.6f}"])


if __name__ == "__main__":
    main()
