"""Phase 54 V54-B1 — V_Ga charge-state encoder pretrain.

Trains a 3-head MLP predicting (E_f^-1, E_f^-2, E_f^-3) for Ga vacancies in
β-Ga2O3, using the 33 HSE06 V_Ga calcs from V53-η v2 as the sole training
source. With N=33 valid entries (Bi/Sb/Sn/Si/Mg/Ge × 4 charges), this
encoder is necessarily small and serves as a soft distill signal for V54-B1.

Output: `checkpoints/encoder_vga.pt`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

VGA_CSV = PROJ / "dft" / "qe_hse06" / "results" / "v_ga_ef.csv"
OUT_CKPT = PROJ / "checkpoints" / "encoder_vga.pt"


class VGaEncoderHead(nn.Module):
    """3-head MLP on element descriptor input → (E_f^-1, E_f^-2, E_f^-3)."""

    def __init__(self, in_dim: int = 11, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])

    def forward(self, elem_emb: torch.Tensor) -> torch.Tensor:
        h = self.trunk(elem_emb)
        return torch.cat([head(h) for head in self.heads], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="V54-B1 V_Ga charge encoder")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--standardize-targets", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not VGA_CSV.exists():
        print(f"V_Ga HSE06 CSV missing: {VGA_CSV} — skipping pretrain")
        return

    df = pd.read_csv(VGA_CSV).dropna(subset=["E_f_eV"]).reset_index(drop=True)
    print(f"Loaded {len(df)} valid V_Ga^q entries")
    print(df.groupby("charge").size())

    from src.data.element_descriptors import _standardized_descriptor_table
    table = _standardized_descriptor_table()
    table_dim = len(next(iter(table.values())))

    # Map dopant → standardized 11-d emb
    rows = []
    for _, r in df.iterrows():
        d = r["dopant"]
        if d not in table:
            continue
        rows.append({
            "elem_emb": np.array(table[d], dtype=np.float32),
            "charge": int(r["charge"]),
            "E_f": float(r["E_f_eV"]),
        })
    if not rows:
        print("No matching dopants — skipping")
        return

    elem_embs = torch.tensor(np.stack([r["elem_emb"] for r in rows]),
                              dtype=torch.float32).to(device)
    charges = torch.tensor([r["charge"] for r in rows], dtype=torch.long, device=device)
    E_f = torch.tensor([r["E_f"] for r in rows], dtype=torch.float32, device=device)
    print(f"elem_embs: {tuple(elem_embs.shape)}  unique charges: {sorted(set(charges.cpu().tolist()))}")

    # Standardize targets per-charge (raw E_f values have huge ref offset)
    if args.standardize_targets:
        target_mean = torch.zeros(4, device=device)  # charges -3..0
        target_std = torch.ones(4, device=device)
        for q in (-3, -2, -1, 0):
            mask = charges == q
            if mask.sum() > 1:
                target_mean[-q] = E_f[mask].mean()
                target_std[-q] = E_f[mask].std().clamp(min=1e-3)
        E_f_norm = (E_f - target_mean[-charges]) / target_std[-charges]
        print(f"Per-charge target stats: mean={target_mean.cpu().tolist()}, std={target_std.cpu().tolist()}")
    else:
        target_mean = torch.zeros(4)
        target_std = torch.ones(4)
        E_f_norm = E_f

    head = VGaEncoderHead(in_dim=table_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    # Train: each charge maps to head index (q=-1→0, q=-2→1, q=-3→2; q=0 ignored)
    charge_to_head = {-1: 0, -2: 1, -3: 2, 0: 0}  # q=0 also routed to head 0 if present

    for ep in range(args.epochs):
        head.train()
        pred_all = head(elem_embs)                    # [N, 3]
        # Index per-row by charge mapping
        head_idx = torch.tensor([charge_to_head[int(c)] for c in charges.cpu().tolist()],
                                  dtype=torch.long, device=device)
        pred_per_row = pred_all.gather(1, head_idx.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(pred_per_row, E_f_norm)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            head.eval()
            with torch.no_grad():
                mae = (pred_per_row - E_f_norm).abs().mean().item()
            print(f"  Epoch {ep + 1:>4d}  MSE={loss.item():.4f}  MAE(norm)={mae:.4f}")

    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "table_dim": table_dim,
        "hidden_dim": args.hidden_dim,
        "target_mean": target_mean.cpu().tolist(),
        "target_std": target_std.cpu().tolist(),
        "n_train": len(rows),
        "charges_seen": sorted(set(charges.cpu().tolist())),
        "trained_on": "HSE06 β-Ga2O3 V_Ga^q (V53-η v2 dataset)",
    }, OUT_CKPT)
    print(f"\nSaved {OUT_CKPT}")


if __name__ == "__main__":
    main()
