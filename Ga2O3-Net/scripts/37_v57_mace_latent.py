"""V57 Stage 7 — MACE-MP-0 latent integration via SINCERE positive pair.

Loads the 17×256-d cached MACE-MP-0 features at
`data/processed/mace_mp_features.npz` (Phase 55 V55-MM dangling product),
trains a frozen 256→64 linear projection, and packages it for use as a
SINCERE (SupervisedInfoNCELoss) positive pair in V57-A1.

Rule 1 compliance: the MACE projection runs OUTSIDE the gradient path of
V54-A1's encoder. The projection itself trains here on a small contrastive
loss (pull same-element MACE close in projected space; push others apart),
but downstream V54-A1 retrain calls forward with `requires_grad_(False)` +
`.eval()` and `detach()` before adding the auxiliary InfoNCE term.

Outputs:
  checkpoints/mace_proj_64d.pt — state_dict {proj_weight, proj_bias,
                                              elements, mace_mean_dop, dim_in, dim_out}
  results/phase57v57mace_latent/training_log.json — losses + per-epoch metrics
  results/phase57v57mace_latent/cosine_matrix.png — per-element cosine similarity
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

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.training.losses import SupervisedInfoNCELoss  # SINCERE / supervised InfoNCE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_mace_latent")

MACE_NPZ = PROJ / "data" / "processed" / "mace_mp_features.npz"
OUT_DIR = PROJ / "results" / "phase57v57mace_latent"
CKPT = PROJ / "checkpoints" / "mace_proj_64d.pt"


class MACELatentProj(nn.Module):
    """256-d MACE → 64-d projection. Frozen at inference; trains here only."""

    def __init__(self, dim_in: int = 256, dim_out: int = 64):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        # Linear without bias keeps norms predictable for cosine similarity
        self.proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return F.normalize(z, dim=-1)


def synth_dopant_batch(
    mace_dop: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    noise_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a contrastive batch by sampling with replacement + small Gaussian
    noise — gives the model multiple positives per element so SINCERE has work
    to do (the bare 17 features alone give a degenerate contrast).

    Returns (batch_features, batch_labels).
    """
    rng = rng or np.random.default_rng(0)
    n_elems = mace_dop.shape[0]
    idx = rng.integers(0, n_elems, size=batch_size)
    feats = mace_dop[idx]
    noise = torch.randn_like(feats) * noise_std
    return feats + noise, labels[idx]


def cosine_similarity_matrix(
    model: MACELatentProj, mace_dop: torch.Tensor
) -> np.ndarray:
    """Per-element cosine matrix in projected space."""
    with torch.no_grad():
        z = model(mace_dop)
    mat = (z @ z.T).cpu().numpy()
    return mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-key", default="mean_dop",
                    choices=["mean_all", "mean_ga", "mean_dop", "mean_o"],
                    help="Which MACE aggregation to use as positive-pair feature")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load MACE features
    d = np.load(MACE_NPZ, allow_pickle=True)
    elements = [str(e) for e in d["elements"]]
    feats = torch.tensor(d[args.feature_key], dtype=torch.float32, device=device)
    n_elems, dim_in = feats.shape
    labels = torch.arange(n_elems, dtype=torch.long, device=device)
    log.info(f"MACE features ({args.feature_key}) shape={tuple(feats.shape)} on {device}")
    log.info(f"Elements: {elements}")

    # Build projection
    model = MACELatentProj(dim_in=dim_in, dim_out=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sinc_loss = SupervisedInfoNCELoss(temperature=args.temperature).to(device)

    losses = []
    for ep in range(args.epochs):
        x, y = synth_dopant_batch(feats, labels, args.batch_size,
                                   noise_std=args.noise_std, rng=rng)
        z = model(x)
        loss = sinc_loss(z, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if (ep + 1) % 50 == 0:
            log.info(f"epoch {ep+1:4d}/{args.epochs}: loss={loss.item():.4f}")

    # Final cosine matrix
    cos = cosine_similarity_matrix(model, feats)
    diag = float(np.mean(np.diag(cos)))
    off = float((np.sum(cos) - np.trace(cos)) / (n_elems * (n_elems - 1)))
    sep = diag - off
    log.info(f"Cosine: mean_diag={diag:.3f}, mean_off-diag={off:.3f}, separation={sep:.3f}")

    # Save checkpoint
    ckpt = {
        "proj_weight": model.proj.weight.detach().cpu(),
        "elements": elements,
        "mace_mean_dop": feats.cpu(),
        "dim_in": dim_in,
        "dim_out": 64,
        "feature_key": args.feature_key,
        "trained_with": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "noise_std": args.noise_std,
            "temperature": args.temperature,
            "seed": args.seed,
        },
    }
    torch.save(ckpt, CKPT)
    log.info(f"Wrote {CKPT}")

    # Training log + cosine matrix
    (OUT_DIR / "training_log.json").write_text(json.dumps({
        "final_loss": losses[-1],
        "loss_curve": losses,
        "mean_diag_cos": diag,
        "mean_offdiag_cos": off,
        "separation": sep,
        "elements": elements,
        "cosine_matrix": cos.tolist(),
    }, indent=2))
    log.info(f"Wrote {OUT_DIR/'training_log.json'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_elems)); ax.set_xticklabels(elements, rotation=45, ha="right")
        ax.set_yticks(range(n_elems)); ax.set_yticklabels(elements)
        ax.set_title(
            f"V57-MACE-Latent — projected cosine similarity\n"
            f"diag={diag:.2f}, off-diag={off:.2f}, sep={sep:.2f}",
        )
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "cosine_matrix.png", dpi=120)
        plt.close()
        log.info(f"Wrote {OUT_DIR/'cosine_matrix.png'}")
    except Exception as e:
        log.warning(f"plot failed: {e}")


if __name__ == "__main__":
    main()
