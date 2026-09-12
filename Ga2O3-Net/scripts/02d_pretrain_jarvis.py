"""Phase 51 — Single-task encoder pretrain on JARVIS-DFT.

Trains a separate CGCNN encoder to predict TBmBJ bandgap +
orientation-averaged dielectric on ~50k oxide entries from JARVIS-DFT
(Choudhary 2020 npj CM).

Output: `checkpoints/pretrained_encoder_jarvis.pt` — frozen feature
extractor for Phase 51 MultiEncoderFusion.

This is a NEW file — does not touch existing pretrain code.

Usage:
    conda activate ga2o3
    python scripts/02d_pretrain_jarvis.py --gpu 0 --epochs 60
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.jarvis_dataset import JarvisDFTDataset  # noqa: E402
from src.models.cgcnn_encoder import CGCNNEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collate(batch):
    return Batch.from_data_list(batch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="data/raw/jarvis/dft_3d.json")
    p.add_argument("--cache-dir",
                   default="data/processed/jarvis_graphs")
    p.add_argument("--targets", nargs="+",
                   default=["mbj_bandgap", "epsx", "epsy", "epsz"])
    p.add_argument("--max-atoms", type=int, default=80)
    p.add_argument("--warm-start",
                   default="checkpoints/pretrained_encoder_v2.pt")
    p.add_argument("--out", default="checkpoints/pretrained_encoder_jarvis.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    ds = JarvisDFTDataset(
        json_path=str(PROJ / args.json),
        cache_dir=str(PROJ / args.cache_dir),
        targets=args.targets,
        require_all_targets=True,
        oxide_only=True,
        max_atoms=args.max_atoms,
        normalize=True,
    )
    n_total = len(ds)
    if n_total < 100:
        raise RuntimeError(f"JARVIS dataset too small after filtering: {n_total}")

    n_val = max(50, int(n_total * args.val_frac))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"JARVIS: {n_total} → {n_train} train / {n_val} val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)

    encoder = CGCNNEncoder(
        atom_embedding_dim=64, edge_rbf_dim=40, num_conv_layers=4,
        residual=False, pool="mean", activation="softplus",
        pretrain=True, num_targets=len(args.targets),
    )
    if args.warm_start and (PROJ / args.warm_start).exists():
        state = torch.load(str(PROJ / args.warm_start), map_location="cpu",
                           weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k: v for k, v in state.items() if not k.startswith("pretrain_head")}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        logger.info(f"Warm-started from {args.warm_start}: "
                    f"missing={len(missing)} unexpected={len(unexpected)}")
    encoder = encoder.to(device)

    optimizer = torch.optim.AdamW(encoder.parameters(),
                                  lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_state = None
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        encoder.train()
        ep_train_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = encoder(batch)                    # [B, n_targets]
            target = batch.y.view(pred.shape)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            ep_train_loss += loss.item() * batch.num_graphs
            n_seen += batch.num_graphs
        ep_train_loss /= max(n_seen, 1)

        encoder.eval()
        ep_val_loss = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = encoder(batch)
                target = batch.y.view(pred.shape)
                ep_val_loss += F.mse_loss(pred, target).item() * batch.num_graphs
                n_val_seen += batch.num_graphs
        ep_val_loss /= max(n_val_seen, 1)
        scheduler.step()

        history.append({"epoch": epoch + 1,
                        "train_loss": ep_train_loss,
                        "val_loss": ep_val_loss,
                        "lr": optimizer.param_groups[0]["lr"]})

        if ep_val_loss < best_val:
            best_val = ep_val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in encoder.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            logger.info(f"epoch {epoch+1:3d}/{args.epochs}  "
                        f"train={ep_train_loss:.4f}  val={ep_val_loss:.4f}  "
                        f"best={best_val:.4f}  "
                        f"elapsed={(time.time()-t0)/60:.1f}min")

    out_path = PROJ / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(out_path))
    target_mean, target_std = ds.target_stats()
    meta_path = out_path.with_suffix(".meta.pt")
    torch.save({
        "best_val_loss_normalized": best_val,
        "targets": args.targets,
        "target_mean": target_mean,
        "target_std": target_std,
        "history": history,
        "args": vars(args),
    }, str(meta_path))
    logger.info(f"Saved best encoder → {out_path}")
    logger.info(f"Saved meta        → {meta_path}")
    logger.info(f"Wall time: {(time.time()-t0)/60:.1f} min, "
                f"best val MSE (normalized): {best_val:.4f}")


if __name__ == "__main__":
    main()
