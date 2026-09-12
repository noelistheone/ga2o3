"""
Stage 1: Pre-training loop for CGCNNEncoder on Materials Project oxide data.

Trains the encoder to predict band_gap and formation_energy_per_atom.

Supports:
  - Single train/val split  (cv_folds=1, fast)
  - K-fold cross-validation (cv_folds > 1, recommended for robustness)

When cv_folds > 1:
  - Trains k independent models from scratch on k different train/val partitions
  - Each fold's model starts from the same random initialisation
  - The checkpoint saved is from the fold with lowest val MAE (best generalisation)
  - Aggregate MAE mean ± std across folds is reported
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.models.cgcnn_encoder import CGCNNEncoder
from src.training.losses import MSEWithL2Loss

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    Trainer for Stage 1: encoder pre-training on MP oxide data.

    Args:
        model  : CGCNNEncoder with pretrain=True. Weights are treated as the
                 initial state — a deep copy is made per fold so the original
                 is never mutated.
        dataset: MPOxideDataset.
        config : dict from pretrain.yaml.
        device : torch device.
    """

    def __init__(
        self,
        model: CGCNNEncoder,
        dataset,
        config: dict,
        device: torch.device | str = "cpu",
        run_name: str | None = None,
        log_dir: str = "logs",
    ):
        self.initial_model = model      # kept as init state; never trained directly
        self.dataset = dataset
        self.device = device
        self.config = config
        tc = config["training"]
        dc = config["data"]

        self.epochs = tc["epochs"]
        self.patience = tc["patience"]
        self.batch_size = tc["batch_size"]
        self.lr = tc["learning_rate"]
        self.weight_decay = tc["weight_decay"]
        self.min_lr = tc.get("min_lr", 1e-5)
        self.val_fraction = dc["val_fraction"]
        self.num_workers = dc.get("num_workers", 0)
        self.cv_folds = dc.get("cv_folds", 1)
        self.ckpt_path = config["paths"]["checkpoint"]
        Path(self.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        # ── Experiment logging ─────────────────────────────────────────────
        self.run_name = run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(log_dir)
        self.tb_dir = self.log_dir / "tensorboard" / self.run_name
        self.csv_path = self.log_dir / f"{self.run_name}_metrics.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot for reproducibility
        config_snap = self.log_dir / f"{self.run_name}_config.json"
        with open(config_snap, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.tb_dir))
        except ImportError:
            self.tb = None
            logger.warning("TensorBoard not available; install with: pip install tensorboard")

        # CSV metric log
        self._csv_file = open(self.csv_path, "w", newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["fold", "epoch", "train_loss", "val_loss", "val_mae", "lr", "elapsed_s"]
        )

        logger.info(f"Run name: {self.run_name}")
        logger.info(f"TensorBoard logs: {self.tb_dir}")
        logger.info(f"Metric CSV: {self.csv_path}")

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Run pre-training. Returns history dict with loss curves and CV metrics.

        If cv_folds == 1: standard single train/val split.
        If cv_folds > 1 : k-fold CV; saves checkpoint from best-val-loss fold.
        """
        try:
            if self.cv_folds > 1:
                return self._train_cv()
            else:
                return self._train_single()
        finally:
            # Always flush and close CSV + TB, even if training crashes
            self._csv_file.flush()
            self._csv_file.close()
            if self.tb is not None:
                self.tb.close()

    # ── Single split ──────────────────────────────────────────────────────────

    def _train_single(self) -> dict:
        n = len(self.dataset)
        n_val = max(1, int(n * self.val_fraction))
        n_train = n - n_val
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = self._make_loader(train_set, shuffle=True)
        val_loader = self._make_loader(val_set, shuffle=False)

        model = copy.deepcopy(self.initial_model).to(self.device)
        optimizer, scheduler = self._make_optim(model)
        loss_fn = MSEWithL2Loss(l2_lambda=0.0)

        history = {"train_loss": [], "val_loss": [], "val_mae": []}
        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(model, train_loader, optimizer, loss_fn, train=True)
            val_loss, val_mae = self._run_epoch_eval(model, val_loader, loss_fn)
            scheduler.step()
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)

            # Write to CSV (buffering=1 → line-buffered, survives disconnect)
            self._csv_writer.writerow(
                [1, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{val_mae:.6f}", f"{lr:.2e}", f"{elapsed:.1f}"]
            )

            # TensorBoard
            if self.tb is not None:
                self.tb.add_scalar("Loss/train", train_loss, epoch)
                self.tb.add_scalar("Loss/val", val_loss, epoch)
                self.tb.add_scalar("MAE/val_band_gap_eV", val_mae, epoch)
                self.tb.add_scalar("LR", lr, epoch)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:4d}/{self.epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | "
                    f"val_mae={val_mae:.4f} eV | lr={lr:.2e} | {elapsed:.1f}s"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), self.ckpt_path)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        history["best_val_loss"] = best_val_loss
        logger.info(f"Pre-training done. Best val loss: {best_val_loss:.4f}")
        return history

    # ── K-fold CV ─────────────────────────────────────────────────────────────

    def _train_cv(self) -> dict:
        from sklearn.model_selection import KFold

        n = len(self.dataset)
        indices = np.arange(n)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        fold_metrics: list[dict] = []
        best_global_val_loss = float("inf")
        all_train_curves = []
        all_val_curves = []

        logger.info(f"Starting {self.cv_folds}-fold cross-validation "
                    f"on {n} MP structures...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            logger.info(
                f"=== Stage 1 Fold {fold+1}/{self.cv_folds} "
                f"(train={len(train_idx)}, val={len(val_idx)}) ==="
            )

            train_set = torch.utils.data.Subset(self.dataset, train_idx.tolist())
            val_set = torch.utils.data.Subset(self.dataset, val_idx.tolist())
            train_loader = self._make_loader(train_set, shuffle=True)
            val_loader = self._make_loader(val_set, shuffle=False)

            # Fresh copy of initial weights for each fold
            model = copy.deepcopy(self.initial_model).to(self.device)
            optimizer, scheduler = self._make_optim(model)
            loss_fn = MSEWithL2Loss(l2_lambda=0.0)

            train_curve, val_curve = [], []
            best_val_loss = float("inf")
            best_val_mae = float("inf")
            no_improve = 0
            # TensorBoard global step offset so all folds appear on one timeline
            global_step_offset = fold * self.epochs

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()
                train_loss = self._run_epoch(
                    model, train_loader, optimizer, loss_fn, train=True
                )
                val_loss, val_mae = self._run_epoch_eval(model, val_loader, loss_fn)
                scheduler.step()
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]
                global_step = global_step_offset + epoch

                train_curve.append(train_loss)
                val_curve.append(val_loss)

                # CSV — line-buffered, persists through disconnect
                self._csv_writer.writerow(
                    [fold + 1, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                     f"{val_mae:.6f}", f"{lr:.2e}", f"{elapsed:.1f}"]
                )

                # TensorBoard: fold-specific tags + combined tag
                if self.tb is not None:
                    self.tb.add_scalar(f"Loss_fold/train_f{fold+1}", train_loss, epoch)
                    self.tb.add_scalar(f"Loss_fold/val_f{fold+1}", val_loss, epoch)
                    self.tb.add_scalar(f"MAE_fold/val_f{fold+1}", val_mae, epoch)
                    # Combined view across folds on one global timeline
                    self.tb.add_scalar("Loss/train_all_folds", train_loss, global_step)
                    self.tb.add_scalar("Loss/val_all_folds", val_loss, global_step)

                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        f"  [F{fold+1}] Epoch {epoch:4d}/{self.epochs} | "
                        f"train={train_loss:.4f} | val={val_loss:.4f} | "
                        f"val_mae={val_mae:.4f} eV | lr={lr:.2e} | {elapsed:.1f}s"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_mae = val_mae
                    no_improve = 0
                    if val_loss < best_global_val_loss:
                        best_global_val_loss = val_loss
                        torch.save(model.state_dict(), self.ckpt_path)
                        logger.info(
                            f"  [F{fold+1}] New best checkpoint "
                            f"(val_loss={val_loss:.4f})"
                        )
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        logger.info(f"  [F{fold+1}] Early stop at epoch {epoch}.")
                        break

            fold_metrics.append({
                "fold": fold + 1,
                "best_val_loss": best_val_loss,
                "best_val_mae_eV": best_val_mae,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            })
            all_train_curves.append(train_curve)
            all_val_curves.append(val_curve)
            logger.info(
                f"  [F{fold+1}] Done — best val_loss={best_val_loss:.4f}, "
                f"val_mae={best_val_mae:.4f} eV"
            )

        # Summary
        all_mae = [m["best_val_mae_eV"] for m in fold_metrics]
        all_loss = [m["best_val_loss"] for m in fold_metrics]
        logger.info(
            f"\n{'='*60}\n"
            f"Stage 1 {self.cv_folds}-Fold CV Summary\n"
            f"{'='*60}\n"
            f"  Val loss:  {np.mean(all_loss):.4f} ± {np.std(all_loss):.4f}\n"
            f"  Val MAE:   {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f} eV\n"
            f"  Best fold: {int(np.argmin(all_loss))+1}\n"
            f"  Checkpoint saved from best fold → {self.ckpt_path}\n"
            f"{'='*60}"
        )

        return {
            "fold_metrics": fold_metrics,
            "train_curves": all_train_curves,
            "val_curves": all_val_curves,
            "cv_val_loss_mean": float(np.mean(all_loss)),
            "cv_val_loss_std": float(np.std(all_loss)),
            "cv_val_mae_mean": float(np.mean(all_mae)),
            "cv_val_mae_std": float(np.std(all_mae)),
            "best_val_loss": best_global_val_loss,
        }

    # ── Training helpers ──────────────────────────────────────────────────────

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def _make_optim(self, model: nn.Module):
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)
        return optimizer, scheduler

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: Adam,
        loss_fn: MSEWithL2Loss,
        train: bool,
    ) -> float:
        model.train() if train else model.eval()
        total_loss, n_batches = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                batch = batch.to(self.device)
                if train:
                    optimizer.zero_grad()
                pred = model(batch)          # [B, 2]
                target = batch.y.view(-1, 2) # [B, 2]
                loss, _ = loss_fn(pred, target)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _run_epoch_eval(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: MSEWithL2Loss,
    ) -> tuple[float, float]:
        """Returns (val_loss, val_mae_band_gap_eV)."""
        model.eval()
        total_loss, n_batches = 0.0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = model(batch)
                target = batch.y.view(-1, 2)
                loss, _ = loss_fn(pred, target)
                total_loss += loss.item()
                n_batches += 1
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        # MAE on band gap (column 0), de-normalised approximately via std
        val_mae = float(np.mean(np.abs(preds[:, 0] - targets[:, 0])))
        return total_loss / max(n_batches, 1), val_mae
