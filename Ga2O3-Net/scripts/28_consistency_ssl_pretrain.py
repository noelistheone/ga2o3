"""
Phase 28: Multi-view consistency SSL pretrain.

Builds on Phase 13C (masked-process reconstruction) by adding a
**consistency-regularization** term across two augmented views per sample:

    View A (clean): (graph, spec, process_clean)
    View B (noisy): (graph, spec, process_clean + 𝒩(0, σ²))

  L_consistency = MSE(emb_A, emb_B)
  L_recon       = MSE(decode(emb_A), standardize(process[:, :12]))
  L_total       = 0.5·L_consistency + 0.5·L_recon

The encoder is frozen (MP-pretrained); we train composition_stream +
process_stream + FiLM fusion + a small process-recon decoder.

Output: a single checkpoint with the trained streams, drop-in compatible
with `scripts/04_finetune_predict.py --init-streams-from`.

Usage:
    python scripts/28_consistency_ssl_pretrain.py --gpu 0 \\
      --multimodal-config config/multimodal_v2_film.yaml \\
      --csv data/raw/experimental/ga2o3_exp_aug3x_interp.csv \\
      --epochs 200 --noise-sigma 0.5 --consistency-weight 0.5
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
log = logging.getLogger("phase28_ssl")


class ProcessDecoder(nn.Module):
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
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--multimodal-config", default="config/multimodal_v2_film.yaml")
    ap.add_argument("--csv", default="data/raw/experimental/ga2o3_exp_aug3x_interp.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--noise-sigma", type=float, default=0.5,
                    help="Gaussian noise std on standardized process continuous dims")
    ap.add_argument("--consistency-weight", type=float, default=0.5,
                    help="Weight for consistency loss; (1 - this) goes to recon")
    ap.add_argument("--out", default="checkpoints/phase28_consistency_streams.pt")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    with open(args.multimodal_config) as f:
        mm = yaml.safe_load(f)

    model = Ga2O3Net.from_pretrained(
        encoder_ckpt=mm["structure_stream"]["pretrained_ckpt"],
        target_cols=["photo_dark_ratio"],
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

    ds = Ga2O3ExpDataset(
        csv_path=args.csv,
        target_cols=["photo_dark_ratio", "vacancy_concentration"],
        structures_dir="data/structures",
        augment=False, conc_jitter=0.0,
        log_transform_targets=True,
        mask_undoped_labels=False, mask_noconc_labels=False,
    )
    log.info(f"SSL dataset size: {len(ds)} rows")

    # Standardization stats (continuous dims 0..11 only)
    all_proc = torch.stack([ds[i]["process"] for i in range(len(ds))], dim=0)
    cont = all_proc[:, :12]
    proc_mean = cont.mean(dim=0)
    proc_std = cont.std(dim=0).clamp(min=1e-3)
    log.info(f"Process continuous mean/std fitted on {len(ds)} rows; "
             f"first 4 means: {proc_mean[:4].tolist()}")

    # Fit ProcessStream's internal scaler on the same data so noisy injection
    # operates in standardized units.
    model.process_stream.fit_scaler(cont.numpy())

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

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

    cw = float(args.consistency_weight)
    rw = 1.0 - cw
    sigma = float(args.noise_sigma)

    log.info(
        f"Phase 28 SSL: {args.epochs} epochs, lr={args.lr}, "
        f"noise σ={sigma} (standardized units), "
        f"consistency_w={cw:.2f}, recon_w={rw:.2f}"
    )

    for ep in range(args.epochs):
        model.train(); decoder.train()
        n_seen = 0
        sum_loss = sum_cons = sum_recon = 0.0
        for batch in loader:
            graph = batch["graph"].to(device)
            proc = batch["process"].to(device)               # [B, 18]
            specs = batch["dopant_spec"]
            B = proc.shape[0]

            # Build the noisy view: standardize, add noise, de-standardize.
            cont_clean = proc[:, :12]
            cont_std = (cont_clean - proc_mean_d) / proc_std_d
            cont_noisy_std = cont_std + sigma * torch.randn_like(cont_std)
            cont_noisy = cont_noisy_std * proc_std_d + proc_mean_d
            proc_noisy = proc.clone()
            proc_noisy[:, :12] = cont_noisy
            # Categorical/embedding dims (12..17) left untouched

            opt.zero_grad()
            emb_clean = model.get_embedding(graph, specs, proc)
            emb_noisy = model.get_embedding(graph, specs, proc_noisy)

            # Recon loss: predict standardized continuous from clean view
            recon_pred = decoder(emb_clean)
            recon_target = cont_std
            l_recon = ((recon_pred - recon_target) ** 2).mean()

            # Consistency: clean and noisy views should map to same emb
            l_cons = ((emb_clean - emb_noisy) ** 2).mean()

            loss = cw * l_cons + rw * l_recon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()

            sum_loss += loss.item() * B
            sum_cons += l_cons.item() * B
            sum_recon += l_recon.item() * B
            n_seen += B

        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            log.info(
                f"Epoch {ep+1:>3d}/{args.epochs}  "
                f"loss={sum_loss/n_seen:.4f}  "
                f"cons={sum_cons/n_seen:.4f}  "
                f"recon={sum_recon/n_seen:.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )

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
        "ssl_noise_sigma":    sigma,
        "ssl_consistency_w":  cw,
        "ssl_recon_w":        rw,
        "final_loss":         sum_loss / n_seen,
        "final_cons":         sum_cons / n_seen,
        "final_recon":        sum_recon / n_seen,
    }, out_path)
    log.info(
        f"Saved Phase 28 SSL checkpoint: {out_path}  "
        f"(loss={sum_loss/n_seen:.4f}, cons={sum_cons/n_seen:.4f}, recon={sum_recon/n_seen:.4f})"
    )


if __name__ == "__main__":
    main()
