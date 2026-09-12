"""V57 Stage 5 — 4th DFT-scan frozen expert pretrain (V57-B1-Quartet+1).

Builds a *fourth* frozen expert to extend V54-B1's (KROGER + ChargeStateV_O + V_Ga)
triplet into a quartet. Uses kroger_synthetic to generate ~10⁴ scanned
(element, c, T, log_pO2, q=2) → log10[V_O^+2] tuples in the sputter-relevant
window, fits a small MLP, and saves to `checkpoints/encoder_dft_scan.pt`.

The expert is loaded later by [src/models/ga2o3_net.py] following the
existing V54-B1 frozen-expert pattern (requires_grad_(False) + .eval() +
detach() in forward; Rule 1 compliant — no gradient back to MACE/MLP weights).

Loss in the V57-A1 retrain: scaled MSE between V57-A1's fused-embedding
prediction and this expert's frozen prediction (auxiliary distillation,
weight λ_aux=0.05 first run, gridable to {0.10, 0.15} only if no Mg flip).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.data.kroger_synthetic import (
    DOPANT_OFFSET, ELEMENTS, EG_GA2O3, generate_synthetic_dataset, kroger_predict,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_b1_quartet_plus")

OUT_CKPT = PROJ / "checkpoints" / "encoder_dft_scan.pt"
OUT_DIR  = PROJ / "results" / "phase57v57b1_quartet_plus_pretrain"


class FrozenEncoderDFTScan(nn.Module):
    """4th frozen expert: predicts log10[V_O^q] from (element one-hot,
    log10[c], log10[T], log10[p_O2], q-onehot).

    Input layout matches FrozenEncoderKroger but adds explicit charge-state
    one-hot so the expert can distill all 3 charge predictions, not just q=0.
    """

    def __init__(
        self,
        n_elements: int,
        n_charges: int = 3,
        hidden: int = 96,
        cont_mean: list[float] | None = None,
        cont_std: list[float] | None = None,
        target_mean: float = 16.0,
        target_std: float = 2.0,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_charges = n_charges
        in_dim = n_elements + 3 + n_charges
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        cm = torch.tensor(cont_mean or [0.0, 0.0, 0.0], dtype=torch.float32)
        cs = torch.tensor(cont_std  or [1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer("cont_mean", cm)
        self.register_buffer("cont_std",  cs)
        self.target_mean = float(target_mean)
        self.target_std  = float(target_std)

    def forward(
        self,
        elem_onehot: torch.Tensor,  # [B, n_elements]
        log_c: torch.Tensor,        # [B]
        log_T: torch.Tensor,        # [B]
        log_pO2: torch.Tensor,      # [B]
        q_onehot: torch.Tensor,     # [B, n_charges]
    ) -> torch.Tensor:
        cont = torch.stack([log_c, log_T, log_pO2], dim=-1)
        cont_norm = (cont - self.cont_mean) / self.cont_std
        x = torch.cat([elem_onehot.float(), cont_norm, q_onehot.float()], dim=-1)
        pred_norm = self.net(x).squeeze(-1)
        return pred_norm * self.target_std + self.target_mean


def df_to_tensors(df, elem_to_idx, n_elements, n_charges=3):
    """Convert synthetic dataset DataFrame → input tensors."""
    elem_idx = np.array([elem_to_idx[e] for e in df["elem"]], dtype=np.int64)
    log_c    = np.log10(np.clip(df["c_total"].to_numpy(), 1e-6, None)).astype(np.float32)
    log_T    = np.log10(df["T_K"].to_numpy().astype(np.float32))
    log_pO2  = df["log_pO2"].to_numpy().astype(np.float32)
    q        = df["q"].to_numpy().astype(np.int64)
    y        = df["log_VO"].to_numpy().astype(np.float32)

    elem_oh = np.eye(n_elements, dtype=np.float32)[elem_idx]
    q_oh    = np.eye(n_charges,  dtype=np.float32)[q]
    return (
        torch.from_numpy(elem_oh),
        torch.from_numpy(log_c),
        torch.from_numpy(log_T),
        torch.from_numpy(log_pO2),
        torch.from_numpy(q_oh),
        torch.from_numpy(y),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=10000,
                    help="number of synthetic (T,p_O2,c,elem) points; ×3 charges")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 1. Generate synthetic dataset using closed-form KROGER predictor
    log.info(f"Generating {args.n_samples} KROGER scan points × 3 charges...")
    df = generate_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)
    log.info(f"Generated {len(df)} (elem, c, T, p_O2, q) rows; "
             f"log_VO range [{df['log_VO'].min():.2f}, {df['log_VO'].max():.2f}]")

    elem_to_idx = {e: i for i, e in enumerate(ELEMENTS)}
    elem_oh, log_c, log_T, log_pO2, q_oh, y = df_to_tensors(
        df, elem_to_idx, len(ELEMENTS), n_charges=3
    )

    # Continuous stats
    cont_all = torch.stack([log_c, log_T, log_pO2], dim=-1)
    cont_mean = cont_all.mean(0).tolist()
    cont_std  = (cont_all.std(0) + 1e-6).tolist()
    y_mean = float(y.mean())
    y_std  = float(y.std() + 1e-6)
    log.info(f"y_mean={y_mean:.2f}, y_std={y_std:.2f}, "
             f"cont_mean={[round(x,2) for x in cont_mean]}")

    # Train/val split
    n = len(y)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    def gather(idx):
        return (elem_oh[idx], log_c[idx], log_T[idx], log_pO2[idx], q_oh[idx], y[idx])

    tr_tens = gather(tr_idx)
    val_tens = gather(val_idx)

    tr_ds  = TensorDataset(*tr_tens)
    tr_dl  = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 2. Build encoder
    model = FrozenEncoderDFTScan(
        n_elements=len(ELEMENTS), n_charges=3, hidden=args.hidden,
        cont_mean=cont_mean, cont_std=cont_std,
        target_mean=y_mean, target_std=y_std,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # 3. Train
    val_inputs = [t.to(device) for t in val_tens[:5]]
    val_y = val_tens[5].to(device)

    best_val_mae = float("inf")
    history = []
    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        n_seen = 0
        for batch in tr_dl:
            batch = [b.to(device) for b in batch]
            elem_b, lc_b, lT_b, lp_b, qoh_b, y_b = batch
            pred = model(elem_b, lc_b, lT_b, lp_b, qoh_b)
            loss = F.mse_loss(pred, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * y_b.size(0)
            n_seen += y_b.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(*val_inputs)
            val_mae = F.l1_loss(val_pred, val_y).item()
            val_rmse = torch.sqrt(F.mse_loss(val_pred, val_y)).item()
            # R^2
            ss_res = ((val_pred - val_y) ** 2).sum().item()
            ss_tot = ((val_y - val_y.mean()) ** 2).sum().item()
            val_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        history.append({"epoch": ep, "tr_mse": tr_loss/max(n_seen,1),
                        "val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2,
                        "lr": opt.param_groups[0]["lr"]})
        if val_mae < best_val_mae:
            best_val_mae = val_mae

        if (ep + 1) % 25 == 0:
            log.info(f"epoch {ep+1:3d}/{args.epochs}: "
                     f"tr_mse={history[-1]['tr_mse']:.4f}  "
                     f"val_mae={val_mae:.3f}  val_rmse={val_rmse:.3f}  val_R²={val_r2:.3f}")

    log.info(f"Best val_mae={best_val_mae:.4f}")

    # 4. Save state with metadata
    state = {
        "model_state": {f"net.{k}": v for k, v in model.net.state_dict().items()},
        "cont_mean": cont_mean,
        "cont_std":  cont_std,
        "target_mean": y_mean,
        "target_std":  y_std,
        "elements": ELEMENTS,
        "n_charges": 3,
        "hidden": args.hidden,
        "trained_with": {
            "n_samples": args.n_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        "validation": {
            "n_val": int(n_val), "best_val_mae": best_val_mae,
            "final_val_r2": history[-1]["val_r2"],
        },
    }
    torch.save(state, OUT_CKPT)
    log.info(f"Wrote {OUT_CKPT}")

    (OUT_DIR / "training_log.json").write_text(json.dumps({
        "history": history,
        "best_val_mae": best_val_mae,
        "final_val_r2": history[-1]["val_r2"],
        "n_train": int(n - n_val),
        "n_val": int(n_val),
        "n_elements": len(ELEMENTS),
    }, indent=2))
    log.info(f"Wrote {OUT_DIR/'training_log.json'}")


if __name__ == "__main__":
    main()
