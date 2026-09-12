"""
Phase 13C: SSL pretraining via masked-process reconstruction.

Trains CompositionStream + ProcessStream + FiLM fusion on ALL rows
(labeled + unlabeled) of the experimental CSV using a self-supervised
process-reconstruction objective:

    fused = FiLM(struct_emb, comp_emb, proc_emb)
    pred_proc = MLP(fused)               # decoder (discarded after SSL)
    loss = MSE(pred_proc, proc[:, :12])  # 12 continuous process dims

The decoder is thrown away. The point is to train CompositionStream,
ProcessStream, and FiLM fusion on the full data distribution before
the supervised regression head sees only the ~170 labeled rows.

Output:
    checkpoints/phase13c_ssl_streams.pt
        - composition_stream state_dict
        - process_stream state_dict
        - fusion (FiLM) state_dict

Usage:
    python scripts/13c_ssl_pretrain.py --gpu 1 \\
      --multimodal-config config/multimodal_v2_film.yaml \\
      --csv data/raw/experimental/ga2o3_exp_aug3x_interp.csv \\
      --epochs 200
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.models.ga2o3_net import Ga2O3Net
from src.data.experimental_dataset import Ga2O3ExpDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ssl_pretrain")


class ProcessDecoder(nn.Module):
    """Reads fused embedding, reconstructs 12 continuous process dims."""
    def __init__(self, fused_dim: int, hidden: int = 64, out_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--multimodal-config", default="config/multimodal_v2_film.yaml")
    ap.add_argument("--csv", default="data/raw/experimental/ga2o3_exp_aug3x_interp.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="checkpoints/phase13c_ssl_streams.pt")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    with open(args.multimodal_config) as f:
        mm = yaml.safe_load(f)

    # Build model with FiLM fusion (same architecture as 13A fine-tune)
    model = Ga2O3Net.from_pretrained(
        encoder_ckpt=mm["structure_stream"]["pretrained_ckpt"],
        target_cols=["photo_dark_ratio"],   # heads are unused in SSL but constructor needs them
        fusion_mode=mm["fusion"]["mode"],
        composition_kwargs={
            "hidden_dim": mm["composition_stream"]["hidden_dim"],
            "out_dim":    mm["composition_stream"]["out_dim"],
            "dropout":    mm["composition_stream"]["dropout"],
        },
        process_kwargs={
            "in_dim":              mm["process_stream"]["in_dim"],
            "continuous_dim":      mm["process_stream"].get("continuous_dim", 12),
            "method_embed_dim":    mm["process_stream"].get("method_embed_dim", 3),
            "n_methods":           mm["process_stream"].get("n_methods", 6),
            "substrate_embed_dim": mm["process_stream"].get("substrate_embed_dim", 3),
            "n_substrates":        mm["process_stream"].get("n_substrates", 7),
            "hidden_dim":          64,
            "out_dim":             mm["process_stream"]["out_dim"],
            "dropout":             mm["process_stream"]["dropout"],
        },
        head_hidden_dims=[64, 32],
        head_dropout=0.3,
        head_type="mlp",
    ).to(device)

    decoder = ProcessDecoder(fused_dim=model.fused_dim, hidden=64, out_dim=12).to(device)

    # Dataset — uses ALL rows (no label requirement).
    ds = Ga2O3ExpDataset(
        csv_path=args.csv,
        target_cols=["photo_dark_ratio", "vacancy_concentration"],
        structures_dir="data/structures",
        augment=False, conc_jitter=0.0,
        log_transform_targets=True,
        mask_undoped_labels=False, mask_noconc_labels=False,
    )
    log.info(f"SSL dataset size: {len(ds)} rows (uses ALL, labels not required)")

    # Fit process scaler on continuous dims only — same as fine-tune does
    all_proc = torch.stack([ds[i]["process"] for i in range(len(ds))], dim=0)
    cont = all_proc[:, :12]
    proc_mean = cont.mean(dim=0)
    proc_std = cont.std(dim=0).clamp(min=1e-3)
    log.info(f"Process continuous mean (T, time, ...): {proc_mean[:4].tolist()}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Freeze encoder (already pretrained on MP); train comp + proc + fusion + decoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    opt_params = (
        list(model.composition_stream.parameters())
        + list(model.process_stream.parameters())
        + list(model.fusion.parameters())
        + list(decoder.parameters())
    )
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    proc_mean_d = proc_mean.to(device)
    proc_std_d = proc_std.to(device)

    log.info(f"Starting SSL pretraining: {args.epochs} epochs, lr={args.lr}")
    for ep in range(args.epochs):
        model.train(); decoder.train()
        epoch_loss = 0.0
        n_seen = 0
        for batch in loader:
            graph = batch["graph"].to(device)
            proc = batch["process"].to(device)
            specs = batch["dopant_spec"]

            # Standardize process targets
            target = (proc[:, :12] - proc_mean_d) / proc_std_d

            opt.zero_grad()
            fused = model.get_embedding(graph, specs, proc)
            pred = decoder(fused)
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()

            epoch_loss += loss.item() * proc.shape[0]
            n_seen += proc.shape[0]

        sched.step()
        avg_loss = epoch_loss / max(n_seen, 1)
        if (ep + 1) % 10 == 0 or ep == 0:
            log.info(f"Epoch {ep+1:>3d}/{args.epochs}  loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "composition_stream": model.composition_stream.state_dict(),
        "process_stream":     model.process_stream.state_dict(),
        "fusion":             model.fusion.state_dict(),
        "fusion_mode":        mm["fusion"]["mode"],
        "fused_dim":          model.fused_dim,
        "ssl_csv":            args.csv,
        "ssl_n_rows":         len(ds),
        "ssl_epochs":         args.epochs,
        "final_loss":         avg_loss,
    }, out_path)
    log.info(f"Saved SSL checkpoint: {out_path}  (final loss={avg_loss:.4f})")


if __name__ == "__main__":
    main()
