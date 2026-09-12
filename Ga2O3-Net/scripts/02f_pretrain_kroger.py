"""Phase 53 Stage 2 — Pretrain encoder_kroger MLP on synthetic KROGER dataset.

The encoder_kroger is a frozen "process expert" that maps (T, log_pO2, dopant_one_hot, log_c)
→ predicted log10[V_O] for β-Ga2O3 with given dopant. Trained on 10k synthetic tuples
from kroger_synthetic.py.

Architecture: 4-layer MLP (input 22 = 1+1+19+1 → 64 → 32 → 16 → 1), ~50k params.
Training: MSE on log_VO, 200 epochs, Adam lr=1e-3.

Output: checkpoints/encoder_kroger.pt
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

PROJ = Path(__file__).resolve().parents[1]
SYN_CSV = PROJ / "data" / "processed" / "kroger_synthetic_q0.csv"
CKPT = PROJ / "checkpoints" / "encoder_kroger.pt"

# Element vocabulary (must match kroger_synthetic.DOPANT_OFFSET keys)
ELEMENTS = [
    "Mg", "Zn", "Cu", "Ni",
    "Al", "Fe", "B", "V", "Er", "Eu", "Cr",
    "Sb", "Bi",
    "Si", "Sn", "Ti", "Ge",
    "Ta", "W",
]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}


class KrogerSyntheticDataset(Dataset):
    """Loads synthetic (dopant_idx, log_c, log_T, log_pO2) → log_VO tuples."""

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        df = df[df["elem"].isin(ELEMENTS)].reset_index(drop=True)
        self.elem_idx = torch.tensor([ELEM_TO_IDX[e] for e in df["elem"]], dtype=torch.long)
        # Standardize features
        log_c = np.log10(np.clip(df["c_total"].values, 1e-6, 1.0))
        log_T = np.log10(df["T_K"].values)
        log_pO2 = df["log_pO2"].values  # already log10
        self.cont_features = torch.tensor(
            np.stack([log_c, log_T, log_pO2], axis=1), dtype=torch.float32
        )
        # Standardize each column
        self.cont_mean = self.cont_features.mean(0)
        self.cont_std = self.cont_features.std(0).clamp(min=1e-3)
        self.cont_features = (self.cont_features - self.cont_mean) / self.cont_std

        # Targets (log_VO, also standardized for stable training)
        self.target = torch.tensor(df["log_VO"].values, dtype=torch.float32)
        self.target_mean = self.target.mean()
        self.target_std = self.target.std().clamp(min=1e-3)
        self.target_norm = (self.target - self.target_mean) / self.target_std

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.elem_idx[idx], self.cont_features[idx], self.target_norm[idx]


class EncoderKroger(nn.Module):
    """Process-expert MLP: (dopant_one_hot, log_c, log_T, log_pO2) → log_VO."""

    def __init__(self, n_elements: int = 19, hidden: int = 64):
        super().__init__()
        in_dim = n_elements + 3  # one-hot dopant + 3 continuous
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4), nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )
        self.n_elements = n_elements

    def forward(self, elem_idx: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(elem_idx, self.n_elements).float()
        x = torch.cat([one_hot, cont], dim=-1)
        return self.net(x).squeeze(-1)


def main():
    print(f"Loading {SYN_CSV} ...")
    ds = KrogerSyntheticDataset(SYN_CSV)
    n_train = int(len(ds) * 0.9)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = EncoderKroger(n_elements=len(ELEMENTS), hidden=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"encoder_kroger params: {n_params:,}")

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    best_val = float("inf")
    for epoch in range(200):
        model.train()
        train_loss = 0.0
        for elem_idx, cont, y in train_dl:
            opt.zero_grad()
            pred = model(elem_idx, cont)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            opt.step()
            train_loss += float(loss) * len(y)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for elem_idx, cont, y in val_dl:
                pred = model(elem_idx, cont)
                # un-standardize for interpretable MAE
                pred_real = pred * ds.target_std + ds.target_mean
                y_real = y * ds.target_std + ds.target_mean
                val_loss += float(((pred - y) ** 2).mean()) * len(y)
                val_mae += float((pred_real - y_real).abs().mean()) * len(y)
        val_loss /= n_val
        val_mae /= n_val

        if val_loss < best_val:
            best_val = val_loss
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "elements": ELEMENTS,
                "cont_mean": ds.cont_mean.tolist(),
                "cont_std": ds.cont_std.tolist(),
                "target_mean": float(ds.target_mean),
                "target_std": float(ds.target_std),
                "epoch": epoch,
            }, CKPT)

        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}: train MSE={train_loss:.4f} val MSE={val_loss:.4f} "
                  f"val MAE (log10)={val_mae:.3f}")

    print(f"\nDone. Best val MSE={best_val:.4f}. Checkpoint: {CKPT}")


if __name__ == "__main__":
    main()
