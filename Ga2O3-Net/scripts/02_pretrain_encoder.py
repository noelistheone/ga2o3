"""
Script 02: Pre-train CGCNNEncoder on Materials Project oxide data (Stage 1).

This script can be run BEFORE any experimental data is available.
Only requires:
  - data/raw/mp_oxides/*.cif          (from script 01)
  - data/processed/mp_labels.csv      (from script 01)

Usage:
    conda activate ga2o3

    # Standard run (GPU 0, 8 parallel CIF workers):
    python scripts/02_pretrain_encoder.py

    # Second GPU:
    python scripts/02_pretrain_encoder.py --gpu 1

    # Large dataset (35k+ structures): use more workers:
    python scripts/02_pretrain_encoder.py --process-workers 16

    # CPU fallback:
    python scripts/02_pretrain_encoder.py --gpu -1 --process-workers 8

Output: checkpoints/pretrained_encoder.pt
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from src.models.cgcnn_encoder import CGCNNEncoder
from src.data.mp_dataset import MPOxideDataset
from src.training.pretrain_trainer import PretrainTrainer
from src.utils.visualization import plot_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-train CGCNN encoder on MP oxides")
    parser.add_argument("--config", default="config/pretrain.yaml")
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (default 0). Use -1 for CPU.",
    )
    parser.add_argument(
        "--process-workers", type=int, default=None,
        metavar="N",
        help="Parallel workers for CIF→graph conversion "
             "(default: value from config, typically 8). "
             "Use 0 for serial processing. -1 = all CPU cores.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=None,
        metavar="K",
        help="K-fold CV for Stage 1 (default: from config, typically 1). "
             "K=1 uses a single train/val split (fast). "
             "K>1 trains K independent models and saves the best checkpoint.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Experiment name for TensorBoard and log files "
             "(default: auto-generated timestamp). "
             "Use descriptive names for easy comparison, e.g. 'lr1e-3_batch256'.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        logger.info(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (training will be slow for large datasets)")

    mc = config["model"]
    dc = config["data"]

    # ── Dataset ──────────────────────────────────────────────────────────────
    n_process_workers = args.process_workers
    if n_process_workers is None:
        n_process_workers = dc.get("process_workers", 0)

    logger.info("Building dataset...")
    logger.info(f"  CIF directory: {config['paths']['raw_data']}")
    logger.info(f"  Labels CSV:    {config['paths']['labels_csv']}")
    logger.info(f"  CIF→graph workers: {n_process_workers} "
                f"({'parallel' if n_process_workers > 1 else 'serial'})")

    dataset = MPOxideDataset(
        raw_dir=config["paths"]["raw_data"],
        processed_dir=config["paths"]["processed"],
        labels_csv=config["paths"]["labels_csv"],
        cutoff=dc["cutoff_radius"],
        num_rbf=mc["edge_rbf_dim"],
        num_process_workers=n_process_workers,
    )

    if args.cv_folds is not None:
        config["data"]["cv_folds"] = args.cv_folds
    cv_folds = config["data"].get("cv_folds", 1)
    logger.info(
        f"Stage 1 CV mode: {'%d-fold' % cv_folds if cv_folds > 1 else 'single split'}"
    )

    stats = dataset.dataset_stats
    logger.info(f"Dataset size: {stats['total']} structures")
    logger.info(f"  Sources: {stats['sources']}")
    logger.info(f"  Band gap: mean={stats['band_gap_mean']:.2f} eV, "
                f"max={stats['band_gap_max']:.2f} eV")
    logger.info(f"  Formation energy: mean={stats['formation_energy_mean']:.3f} eV/atom")

    if stats["total"] == 0:
        logger.error("Dataset is empty! Run script 01 first to download MP structures.")
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building CGCNN encoder...")
    model = CGCNNEncoder(
        atom_embedding_dim=mc["atom_embedding_dim"],
        edge_rbf_dim=mc["edge_rbf_dim"],
        num_conv_layers=mc["num_conv_layers"],
        residual=mc.get("residual", False),
        pool=mc.get("pool", "mean"),
        activation=mc.get("activation", "softplus"),
        pre_pool_dim=mc.get("pre_pool_dim", 0),
        pretrain=True,
        num_targets=mc["output_dim"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ── Training ─────────────────────────────────────────────────────────────
    trainer = PretrainTrainer(
        model, dataset, config, device=device, run_name=args.run_name
    )
    logger.info("Starting Stage 1 pre-training...")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Epochs (max per fold): {config['training']['epochs']}")
    logger.info(f"  CV folds: {cv_folds}")
    logger.info(f"  Checkpoint: {config['paths']['checkpoint']}")
    history = trainer.train()

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    # For k-fold CV, plot curves from each fold; for single split, plot directly
    if cv_folds > 1 and "train_curves" in history:
        for fold_i, (tc, vc) in enumerate(
            zip(history["train_curves"], history["val_curves"])
        ):
            plot_training_curves(
                {"train_loss": tc, "val_loss": vc},
                save_path=f"results/pretrain_loss_fold{fold_i+1}.png",
                title=f"Stage 1 Fold {fold_i+1}: Pre-training Loss",
            )
        logger.info(
            f"CV summary: val_loss={history['cv_val_loss_mean']:.4f} "
            f"± {history['cv_val_loss_std']:.4f} | "
            f"val_mae={history['cv_val_mae_mean']:.4f} "
            f"± {history['cv_val_mae_std']:.4f} eV"
        )
    else:
        plot_training_curves(
            history,
            save_path="results/pretrain_loss.png",
            title="Stage 1: Pre-training Loss (band_gap + formation_energy)",
        )
    logger.info(f"Loss curve saved: results/pretrain_loss.png")
    logger.info(f"Checkpoint saved: {config['paths']['checkpoint']}")
    logger.info("")
    logger.info("Pre-training complete. When you have experimental data, run:")
    logger.info("  python scripts/03_extract_features.py")


if __name__ == "__main__":
    main()
