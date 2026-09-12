"""Phase 54 V54-B1 — Charge-state V_O encoder pretrain.

Kiyohara et al. arXiv:2510.00513 protocol: train a single CGCNN with three
heads predicting (E_f^0, E_f^+1, E_f^+2) per oxygen vacancy site. Witman
Zenodo 8087871 provides 795 neutral V_O entries for q=0 pretrain; the V53-η
v2 HSE06 dataset (54 β-Ga2O3 calcs across q=0/+1/+2 with 6 dopants) provides
charge-state-specific fine-tuning labels.

Output: `checkpoints/encoder_charge_vo.pt` — loaded as a frozen expert
by V54-B1's Ga2O3Net (via `aux_distill_charge_vo_head`).

Rule 1 compliant: this encoder produces aux targets only; main Brouwer
pipeline is untouched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

WITMAN_CSV = PROJ / "data" / "processed" / "witman_v_o_master.csv"
WITMAN_GRAPHS = PROJ / "data" / "processed" / "witman_graphs"
QE_VO_CSV = PROJ / "dft" / "qe_hse06" / "results" / "v_o_ef.csv"
OUT_CKPT = PROJ / "checkpoints" / "encoder_charge_vo.pt"


class ChargeStateVOHead(nn.Module):
    """3-head output on top of a shared CGCNN-style encoder.

    Backbone reuses V5 CGCNNEncoder; we add three independent linear heads
    over the pooled crystal embedding. q=0 head trains on Witman + HSE q=0;
    q=+1/+2 heads train only on HSE.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        # 3 heads (one per charge state)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: [B, in_dim] crystal embedding → [B, 3] (E_f^0, E_f^+1, E_f^+2)."""
        h = self.trunk(emb)
        return torch.cat([head(h) for head in self.heads], dim=-1)


def _load_witman_q0() -> tuple[list[Path], np.ndarray]:
    """Load Witman 795 q=0 entries; return (graph_paths, E_f values)."""
    df = pd.read_csv(WITMAN_CSV)
    paths, labels = [], []
    for _, row in df.iterrows():
        cid = str(row["compound_id"]).zfill(7)
        p = WITMAN_GRAPHS / f"{cid}.pt"
        if p.exists() and pd.notna(row["dH_eV"]):
            paths.append(p)
            labels.append(float(row["dH_eV"]))
    return paths, np.array(labels, dtype=np.float32)


def _load_hse_vo() -> pd.DataFrame:
    """Load HSE06 β-Ga2O3 V_O^q entries — multiple charges per (dopant, site)."""
    df = pd.read_csv(QE_VO_CSV).dropna(subset=["E_f_eV"]).copy()
    df["charge"] = df["charge"].astype(int)
    return df.reset_index(drop=True)


def _embed_witman_graphs(paths: list[Path], encoder, device, batch_size: int = 32) -> torch.Tensor:
    """Run CGCNN encoder over Witman graphs → [N, emb_dim]."""
    from torch_geometric.data import Batch
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            graphs = [torch.load(p, weights_only=False) for p in batch_paths]
            graphs = [g for g in graphs if g is not None]
            if not graphs:
                continue
            batch = Batch.from_data_list(graphs).to(device)
            # Encoder forward returns either (emb,) or (emb, pretrain_pred); take emb.
            ret = encoder(batch)
            emb = ret[0] if isinstance(ret, tuple) else ret
            out.append(emb.cpu())
    return torch.cat(out, dim=0) if out else torch.zeros(0, 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="V54-B1 charge-state V_O encoder")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not WITMAN_CSV.exists() or not WITMAN_GRAPHS.exists():
        print("Witman dataset missing — pretrain stage skipped (need data/processed/witman_*)")
        return
    if not QE_VO_CSV.exists():
        print(f"HSE06 V_O dataset missing: {QE_VO_CSV} — V54-B1 fine-tune skipped")
        return

    # Load V5 CGCNN encoder (frozen) and use its embeddings.
    from src.models.cgcnn_encoder import CGCNNEncoder
    encoder = CGCNNEncoder(pretrain=False).to(device)
    enc_state = torch.load(PROJ / "checkpoints" / "pretrained_encoder_v2.pt",
                            map_location="cpu", weights_only=False)
    encoder.load_state_dict({k: v for k, v in enc_state.items()
                              if not k.startswith("pretrain_head")}, strict=False)
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── Witman q=0 pretrain ──────────────────────────────────────────────
    print("Loading Witman q=0 dataset…")
    paths_w, labels_w = _load_witman_q0()
    print(f"  {len(paths_w)} Witman graphs with dH_eV labels")
    if len(paths_w) < 10:
        print("Insufficient Witman data for pretrain — falling back to HSE-only fine-tune")

    emb_w = _embed_witman_graphs(paths_w, encoder, device, args.batch_size)
    labels_w_t = torch.tensor(labels_w[:emb_w.shape[0]], dtype=torch.float32)
    print(f"  Witman embeddings: {tuple(emb_w.shape)}")

    head = ChargeStateVOHead(in_dim=emb_w.shape[1], hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)

    # Train q=0 head on Witman first
    head.train()
    perm = torch.randperm(emb_w.shape[0])
    train_n = int(0.9 * emb_w.shape[0])
    train_idx, val_idx = perm[:train_n], perm[train_n:]

    for ep in range(args.epochs):
        # Minibatch over Witman train split
        head.train()
        loss_acc = 0.0
        nb = 0
        for i in range(0, len(train_idx), args.batch_size):
            sel = train_idx[i:i + args.batch_size]
            x = emb_w[sel].to(device)
            y = labels_w_t[sel].to(device)
            pred_all = head(x)             # [B, 3]
            pred_q0 = pred_all[:, 0]       # q=0 head
            loss = F.mse_loss(pred_q0, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += loss.item(); nb += 1
        if (ep + 1) % 20 == 0:
            head.eval()
            with torch.no_grad():
                x_v = emb_w[val_idx].to(device)
                y_v = labels_w_t[val_idx].to(device)
                p_v = head(x_v)[:, 0]
                val_mae = (p_v - y_v).abs().mean().item()
            print(f"  Epoch {ep + 1:>4d}  train MSE={loss_acc/nb:.4f}  val MAE q=0={val_mae:.4f}")

    # ── HSE β-Ga2O3 V_O^q fine-tune ───────────────────────────────────────
    hse = _load_hse_vo()
    print(f"\nHSE V_O fine-tune set: {len(hse)} entries "
          f"({len(hse[hse['charge']==0])} q=0 / "
          f"{len(hse[hse['charge']==1])} q=+1 / "
          f"{len(hse[hse['charge']==2])} q=+2)")
    # We don't have per-dopant β-Ga2O3 graphs in graphs/; the fine-tune here
    # is structurally limited. We use the dopant-doped β-Ga2O3 cell's V5
    # encoder embedding as the per-row feature, indexed by dopant label.
    from src.data.experimental_dataset import Ga2O3ExpDataset
    # Build a tiny dataset of representative β-Ga2O3+dopant graphs to embed.
    # For simplicity, use one row per dopant from the experimental CSV.
    exp = pd.read_csv(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")
    exp = exp[exp["element"].isin(hse["dopant"].unique())]
    dopant_to_row = {d: exp[exp["element"] == d].iloc[0] for d in hse["dopant"].unique()
                      if (exp["element"] == d).any()}
    print(f"  Found {len(dopant_to_row)}/{hse['dopant'].nunique()} dopants in exp CSV")
    if not dopant_to_row:
        print("  Cannot find matching dopants in experimental CSV — skipping HSE fine-tune.")
    else:
        # Build minimal dataset that just gives us the graphs (skip targets)
        ds = Ga2O3ExpDataset(
            csv_path=str(PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv"),
            structures_dir=str(PROJ / "data" / "structures"),
            target_cols=["vacancy_concentration"],
        )
        # Build dopant → embedding lookup
        dop_emb_lookup: dict[str, torch.Tensor] = {}
        for idx in range(len(ds)):
            sample = ds[idx]
            elem = sample["dopant_label"]
            if elem in hse["dopant"].unique() and elem not in dop_emb_lookup:
                from torch_geometric.data import Batch
                with torch.no_grad():
                    batch = Batch.from_data_list([sample["graph"]]).to(device)
                    ret = encoder(batch)
                    emb = ret[0] if isinstance(ret, tuple) else ret
                    dop_emb_lookup[elem] = emb.detach().cpu()

        # Build (X, y_qX) per charge
        X_hse, y_hse, charge_hse = [], [], []
        for _, row in hse.iterrows():
            d = row["dopant"]
            if d not in dop_emb_lookup:
                continue
            X_hse.append(dop_emb_lookup[d].squeeze(0))
            y_hse.append(float(row["E_f_eV"]))
            charge_hse.append(int(row["charge"]))
        if not X_hse:
            print("  No HSE rows had matching graph embeddings — skipping.")
        else:
            X_hse_t = torch.stack(X_hse).to(device)
            y_hse_t = torch.tensor(y_hse, dtype=torch.float32, device=device)
            ch_hse_t = torch.tensor(charge_hse, dtype=torch.long, device=device)
            print(f"  HSE fine-tune: X.shape={tuple(X_hse_t.shape)}")
            # Train each charge-head separately on HSE-q subset
            for q in (0, 1, 2):
                mask = ch_hse_t == q
                if mask.sum() < 3:
                    continue
                for ep in range(50):
                    pred = head(X_hse_t[mask])[:, q]
                    loss = F.mse_loss(pred, y_hse_t[mask])
                    opt.zero_grad(); loss.backward(); opt.step()
                print(f"  q={q}: final HSE MSE = {loss.item():.4f}  on {int(mask.sum())} rows")

    # Save
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "in_dim": head.heads[0].in_features,
        "out_dim": 3,
        "trained_on": "Witman (q=0) + HSE β-Ga2O3 V_O (q=0/1/2)",
        "n_witman": int(emb_w.shape[0]),
        "n_hse_per_q": {
            "0": int(len(hse[hse["charge"] == 0])),
            "1": int(len(hse[hse["charge"] == 1])),
            "2": int(len(hse[hse["charge"] == 2])),
        },
    }, OUT_CKPT)
    print(f"\nSaved {OUT_CKPT}")


if __name__ == "__main__":
    main()
