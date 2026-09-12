"""
Script 03: Extract and cache Stage 2 multi-modal embeddings (160-dim).

Scalers are fitted on the training split of a k-fold CV partition, not on
the full dataset, to prevent data leakage into the validation set.

For each fold, the following are saved:
  data/processed/stage2_fold{k}/embeddings_train.npy
  data/processed/stage2_fold{k}/embeddings_val.npy
  data/processed/stage2_fold{k}/targets_train.npy
  data/processed/stage2_fold{k}/targets_val.npy
  data/processed/stage2_fold{k}/process_scaler.pkl
  data/processed/stage2_fold{k}/composition_scaler.pkl

A global embedding (all samples, scaler fit on all data) is also saved for
visualisation purposes only — NOT for training.

Usage:
    conda activate ga2o3
    python scripts/03_extract_features.py
    python scripts/03_extract_features.py --gpu 1 --cv-folds 3
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from src.models.ga2o3_net import Ga2O3Net
from src.data.experimental_dataset import Ga2O3ExpDataset, collate_fn
from src.utils.visualization import plot_embedding_scatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(mm_cfg: dict, fu_cfg: dict, device: torch.device) -> Ga2O3Net:
    return Ga2O3Net.from_pretrained(
        encoder_ckpt=mm_cfg["structure_stream"]["pretrained_ckpt"],
        target_cols=fu_cfg["targets"],
        fusion_mode=mm_cfg["fusion"]["mode"],
        composition_kwargs={
            "hidden_dim": mm_cfg["composition_stream"]["hidden_dim"],
            "out_dim": mm_cfg["composition_stream"]["out_dim"],
            "dropout": mm_cfg["composition_stream"]["dropout"],
        },
        process_kwargs={
            "hidden_dim": 64,
            "out_dim": mm_cfg["process_stream"]["out_dim"],
            "dropout": mm_cfg["process_stream"]["dropout"],
        },
    ).to(device)


def extract_embeddings(
    model: Ga2O3Net,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run full forward pass and collect embeddings, targets, labels.
    Returns: (embeddings [N, 160], targets [N, num_targets], labels [N])
    """
    model.eval()
    all_emb, all_targets, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            process = batch["process"].to(device)
            dopant_specs = batch["dopant_spec"]
            target = batch["target"]
            emb = model.get_embedding(graph, dopant_specs, process)
            all_emb.append(emb.cpu().numpy())
            all_targets.append(target.numpy())
            all_labels.extend(batch["dopant_label"])
    return (
        np.concatenate(all_emb, axis=0),
        np.concatenate(all_targets, axis=0),
        all_labels,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract Stage 2 embeddings (with CV)")
    parser.add_argument("--multimodal-config", default="config/multimodal.yaml")
    parser.add_argument("--fusion-config", default="config/fusion.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--cv-folds", type=int, default=None,
        help="Number of CV folds for scaler fitting "
             "(default: taken from fusion.yaml training.cv_folds).",
    )
    args = parser.parse_args()

    with open(args.multimodal_config) as f:
        mm_cfg = yaml.safe_load(f)
    with open(args.fusion_config) as f:
        fu_cfg = yaml.safe_load(f)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    cv_folds = args.cv_folds or fu_cfg["training"].get("cv_folds", 3)

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading experimental dataset...")
    dataset = Ga2O3ExpDataset(
        csv_path=fu_cfg["paths"]["experimental_csv"],
        structures_dir=fu_cfg["paths"]["structures_dir"],
        target_cols=fu_cfg["targets"],
        augment=False,
    )
    logger.info(f"Samples: {len(dataset)}, CV folds: {cv_folds}")

    all_specs = dataset.get_dopant_specs()
    all_process = dataset.get_process_array()
    all_labels = np.array(dataset.get_dopant_labels())
    indices = np.arange(len(dataset))

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ── K-fold: per-fold scaler fitting + embedding extraction ────────────────
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, all_labels)):
        logger.info(
            f"=== Stage 2 Fold {fold+1}/{cv_folds} "
            f"(train={len(train_idx)}, val={len(val_idx)}) ==="
        )

        fold_dir = Path(f"data/processed/stage2_fold{fold+1}")
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Build a fresh model for each fold (scalers are part of model state)
        model = build_model(mm_cfg, fu_cfg, device)

        # Fit scalers on training indices ONLY
        train_specs = [all_specs[i] for i in train_idx]
        train_process = all_process[train_idx]

        model.process_stream.fit_scaler(train_process)
        model.composition_stream.fit_scaler(dopant_specs=train_specs)
        logger.info(f"  Scalers fitted on {len(train_idx)} training samples.")

        # Save fold scalers
        model.process_stream.save_scaler(str(fold_dir / "process_scaler.pkl"))
        model.composition_stream.save_scaler(str(fold_dir / "composition_scaler.pkl"))

        # Extract embeddings
        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=len(train_idx), shuffle=False, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=len(val_idx), shuffle=False, collate_fn=collate_fn,
        )

        train_emb, train_tgt, train_lbl = extract_embeddings(model, train_loader, device)
        val_emb, val_tgt, val_lbl = extract_embeddings(model, val_loader, device)

        np.save(fold_dir / "embeddings_train.npy", train_emb)
        np.save(fold_dir / "embeddings_val.npy", val_emb)
        np.save(fold_dir / "targets_train.npy", train_tgt)
        np.save(fold_dir / "targets_val.npy", val_tgt)

        # Save index mapping for traceability
        fold_meta = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "train_labels": train_lbl,
            "val_labels": val_lbl,
            "target_cols": fu_cfg["targets"],
            "fold": fold + 1,
            "cv_folds": cv_folds,
        }
        with open(fold_dir / "fold_meta.pkl", "wb") as f:
            pickle.dump(fold_meta, f)

        logger.info(
            f"  Fold {fold+1} saved: train_emb={train_emb.shape}, "
            f"val_emb={val_emb.shape}"
        )

    # ── Global embeddings for visualisation (ALL data, scaler on all) ─────────
    logger.info("Extracting global embeddings for visualisation (scaler on all data)...")
    model_vis = build_model(mm_cfg, fu_cfg, device)
    model_vis.process_stream.fit_scaler(all_process)
    model_vis.composition_stream.fit_scaler(dopant_specs=all_specs)

    full_loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn
    )
    all_emb, all_tgt, all_lbl = extract_embeddings(model_vis, full_loader, device)

    vis_cache = {
        "embeddings": all_emb,
        "targets": all_tgt,
        "labels": all_lbl,
        "target_cols": fu_cfg["targets"],
        "note": "Global embeddings for visualisation only — NOT for CV training.",
    }
    cache_path = fu_cfg["paths"]["embeddings_cache"]
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(vis_cache, f)
    logger.info(f"Global embeddings saved → {cache_path}")

    plot_embedding_scatter(
        all_emb, all_lbl,
        save_path="results/stage2_embeddings_pca.png",
        title="Stage 2 Multi-Modal Embeddings (PCA) — visualisation only",
    )

    logger.info(
        f"\nStage 2 complete. Fold embeddings saved in data/processed/stage2_fold*/\n"
        f"Next: python scripts/04_finetune_predict.py [--xgboost]"
    )


if __name__ == "__main__":
    main()
