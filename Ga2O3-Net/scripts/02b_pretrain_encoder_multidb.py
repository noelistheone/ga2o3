"""Phase 50.B — Multi-DB encoder pretrain (incremental over v2 checkpoint).

Strategy: rather than retraining the encoder from scratch on a combined
corpus, we LOAD the existing `pretrained_encoder_v2.pt` (trained on 82k
MP oxides predicting bandgap+E_form) and ADD a new "vacancy" head on top
of the same encoder. Then continue training both heads jointly:

  • MP batches (head="mp"): predict (bandgap, E_form) — keeps existing
    chemistry signal, low weight (0.3) so we don't overshoot.
  • Witman batches (head="vacancy"): predict V_O formation enthalpy
    (Witman/Goyal Zenodo 8087871 — 795 V_O entries on 199 oxides),
    weight 1.0. New signal: encoder must encode defect-chemistry-relevant
    features.

Wall time: ~1-2 hours (50 epochs × small Witman corpus + occasional MP
batch). Output: `checkpoints/pretrained_encoder_phase50.pt` containing
encoder weights + both heads.

Usage:
    python scripts/02b_pretrain_encoder_multidb.py --gpu 0
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.mp_dataset import MPOxideDataset  # noqa: E402
from src.data.witman_vacancy_dataset import WitmanVacancyDataset  # noqa: E402
from src.models.cgcnn_encoder import CGCNNEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collate_pyg(batch):
    return Batch.from_data_list(batch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mp-encoder-ckpt",
                   default="checkpoints/pretrained_encoder_v2.pt")
    p.add_argument("--mp-processed-dir",
                   default="data/processed/mp_oxides")
    p.add_argument("--mp-labels-csv",
                   default="data/processed/mp_labels.csv")
    p.add_argument("--witman-master-csv",
                   default="data/processed/witman_v_o_master.csv")
    p.add_argument("--witman-cache-dir",
                   default="data/processed/witman_graphs")
    p.add_argument("--out", default="checkpoints/pretrained_encoder_phase50.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mp-loss-weight", type=float, default=0.3)
    p.add_argument("--vacancy-loss-weight", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--mp-fraction", type=float, default=0.3,
                   help="Fraction of MP corpus to subsample per epoch (0-1)")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # --- Datasets ---
    logger.info("Loading MP oxide dataset...")
    mp_ds = MPOxideDataset(
        raw_dir=str(PROJ / "data" / "raw" / "mp_oxides"),
        processed_dir=str(PROJ / args.mp_processed_dir),
        labels_csv=str(PROJ / args.mp_labels_csv),
    )
    logger.info(f"MP dataset size: {len(mp_ds)}")

    logger.info("Loading Witman V_O dataset...")
    wit_ds = WitmanVacancyDataset(
        master_csv=str(PROJ / args.witman_master_csv),
        cache_dir=str(PROJ / args.witman_cache_dir),
        normalize=True,
    )
    wit_target_mean, wit_target_std = wit_ds.target_stats()
    logger.info(f"Witman dataset size: {len(wit_ds)}")

    # --- Encoder + multi-head ---
    logger.info(f"Loading encoder from {args.mp_encoder_ckpt}...")
    encoder = CGCNNEncoder(
        atom_embedding_dim=64, edge_rbf_dim=40, num_conv_layers=4,
        residual=False, pool="mean", activation="softplus",
        pretrain=True,
        multi_head_targets={"mp": 2, "vacancy": 1},
    )
    state = torch.load(str(PROJ / args.mp_encoder_ckpt), map_location="cpu",
                       weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    # Migrate `pretrain_head.*` → `pretrain_heads.mp.*` (new multi-head naming)
    migrated = {}
    for k, v in state.items():
        if k.startswith("pretrain_head."):
            migrated["pretrain_heads.mp." + k.split(".", 1)[1]] = v
        else:
            migrated[k] = v
    missing, unexpected = encoder.load_state_dict(migrated, strict=False)
    logger.info(f"Loaded MP weights: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        logger.info(f"  missing (will train from scratch): {list(missing)[:5]}")
    encoder = encoder.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(encoder.parameters(),
                                  lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # --- Data loaders ---
    mp_loader = DataLoader(mp_ds, batch_size=args.batch_size,
                           shuffle=True, collate_fn=collate_pyg,
                           num_workers=2)
    wit_loader = DataLoader(wit_ds, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_pyg,
                            num_workers=2)

    logger.info(f"Starting {args.epochs} epochs of multi-DB pretrain")
    logger.info(f"  MP: {len(mp_loader)} batches/epoch (subsampled to "
                f"~{int(len(mp_loader) * args.mp_fraction)} per epoch)")
    logger.info(f"  Witman: {len(wit_loader)} batches/epoch")
    logger.info(f"  loss_weights: mp={args.mp_loss_weight} "
                f"vacancy={args.vacancy_loss_weight}")

    history = []
    t0 = time.time()
    best_loss = float("inf")
    for epoch in range(args.epochs):
        encoder.train()
        # Build randomly-mixed sequence of "MP" / "Witman" batches per epoch
        mp_iter = iter(mp_loader)
        wit_iter = iter(wit_loader)
        n_mp = max(1, int(len(mp_loader) * args.mp_fraction))
        n_wit = len(wit_loader)
        sequence = ["mp"] * n_mp + ["vacancy"] * n_wit
        np.random.shuffle(sequence)

        ep_mp_loss, ep_wit_loss, n_mp_seen, n_wit_seen = 0.0, 0.0, 0, 0
        for src in sequence:
            try:
                if src == "mp":
                    batch = next(mp_iter)
                    batch = batch.to(device)
                    pred = encoder(batch, head="mp")
                    target = batch.y.view(-1, 2) if batch.y.dim() == 2 else batch.y.view(pred.shape)
                    loss = F.mse_loss(pred, target) * args.mp_loss_weight
                    ep_mp_loss += loss.item()
                    n_mp_seen += 1
                else:
                    batch = next(wit_iter)
                    batch = batch.to(device)
                    pred = encoder(batch, head="vacancy")
                    target = batch.y.view(-1, 1)
                    loss = F.mse_loss(pred, target) * args.vacancy_loss_weight
                    ep_wit_loss += loss.item()
                    n_wit_seen += 1
            except StopIteration:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        avg_mp = ep_mp_loss / max(n_mp_seen, 1)
        avg_wit = ep_wit_loss / max(n_wit_seen, 1)
        total = avg_mp + avg_wit
        history.append({
            "epoch": epoch + 1, "mp_loss": avg_mp, "wit_loss": avg_wit,
            "lr": optimizer.param_groups[0]["lr"],
        })
        logger.info(f"epoch {epoch+1:3d}/{args.epochs}  "
                    f"mp_loss={avg_mp:.4f} ({n_mp_seen} batches)  "
                    f"vacancy_loss={avg_wit:.4f} ({n_wit_seen} batches)  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                    f"elapsed={(time.time()-t0)/60:.1f}min")
        if total < best_loss:
            best_loss = total

    # Save final
    out_path = PROJ / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = encoder.state_dict()
    # Also save MP-only legacy alias for backward compat with V5 distill aux
    if "pretrain_heads.mp.weight" in state_dict:
        state_dict["pretrain_head.weight"] = state_dict["pretrain_heads.mp.weight"]
        state_dict["pretrain_head.bias"] = state_dict["pretrain_heads.mp.bias"]
    torch.save({
        "encoder_state": state_dict,
        "multi_head_targets": {"mp": 2, "vacancy": 1},
        "wit_target_mean": wit_target_mean,
        "wit_target_std": wit_target_std,
        "history": history,
        "args": vars(args),
    }, str(out_path))
    # Save a v2-compatible flat state too (for legacy CGCNNEncoder loading)
    torch.save(state_dict, str(out_path).replace(".pt", "_flat.pt"))
    logger.info(f"Saved {out_path}")
    logger.info(f"Wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
