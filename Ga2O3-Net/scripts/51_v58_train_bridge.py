"""
Phase-58 LLMBridge refinement — Step 2/3: train the 25k LLMBridge.

The bridge (src/models/llm_bridge.py) was NOT trained in Stage 2 (lambda_bridge
was forced to 0 because the Stage-1 z_phys teacher didn't exist yet). Now BOTH
LLMs are trained, so we train ONLY the 25k bridge on cached (z_proto, z_phys)
pairs — the 7B and 1.5B stay frozen and are never loaded here.

Data join (per task spec)
-------------------------
A bridge-training sample is a tuple (paper_id, dopant, atm, T) that has BOTH:
  - a z_proto  : cached per paper_id          (script 50)
  - a z_phys   : cached per "{dopant}|{atm}|{T}" key  (script 43,
                 data/processed/z_phys_cache_v58.npz)
The (paper -> dopant) link comes from the v56 markdown manifest's source_pdf
path (`.../doping/<Element>/...`); a paper doped with element E is paired with
every "{E}|{atm}|{T}" key that actually occurs (non-NONE dft_features_idx) in
the v58 CSV and has a z_phys entry.

Losses (design doc §3.7.2 / §5.2.3)
-----------------------------------
  L_recon  = MSE(z_LLM, z_phys.detach())
             ("z_LLM approximates z_phys shifted by a process-conditional
              offset" — task: plain MSE acceptable as the recon anchor)
  L_nce    = InfoNCE(tau=0.07) over the in-batch z_LLM, where positives are
             tuples sharing the same paper_id (protocol structure) and
             negatives are other papers in the batch.
  L_total  = L_recon + lambda_nce * L_nce

Reports the loss curve + final InfoNCE recall@5 (target >= 0.7, §5.2.5) and
saves the trained bridge weights to checkpoints/qwen25_v58_bridge/bridge.pt
plus a metrics JSON.

This is tiny (25k params on cached 32-d vectors); runs in seconds either on
CPU or GPU. Default GPU is 1 (task: GPU 1 ONLY); GPU 0 is never touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _norm_el(el: str) -> str:
    return el[0].upper() + el[1:].lower() if el else el


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--z-proto-cache", type=str,
                   default="data/processed/z_proto_cache_v58.npz")
    p.add_argument("--z-phys-cache", type=str,
                   default="data/processed/z_phys_cache_v58.npz")
    p.add_argument("--v58-csv", type=str,
                   default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v58.csv")
    p.add_argument("--md-manifest", type=str,
                   default="extra-paper/v56_markdown/manifest.csv",
                   help="maps paper hash (doi_hash) -> source_pdf path -> dopant element")
    p.add_argument("--out-dir", type=str, default="checkpoints/qwen25_v58_bridge")
    p.add_argument("--metrics", type=str,
                   default="results/v58_bridge_train_metrics.json")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-nce", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=1, help="GPU 1 ONLY (task). -1=CPU.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from src.models.llm_bridge import LLMBridge

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("51_bridge")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dev = torch.device("cuda:0" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")
    log.info("device: %s (physical GPU %s)", dev, args.gpu)

    # ── 1. Load caches ────────────────────────────────────────────────────────
    zp_proto = np.load(HERE / args.z_proto_cache, allow_pickle=True)
    proto_keys = [str(k) for k in zp_proto["keys"]]
    proto_feats = zp_proto["features"].astype(np.float32)   # [Np, 32]
    proto_map = {k: proto_feats[i] for i, k in enumerate(proto_keys)}
    log.info("z_proto cache: %d papers", len(proto_map))

    zp_phys = np.load(HERE / args.z_phys_cache, allow_pickle=True)
    # z_phys cache is keyed-by-name (each "{dopant}|{atm}|{T}" is its own array)
    phys_map = {k: np.asarray(zp_phys[k], dtype=np.float32)
                for k in zp_phys.files if "|" in k}
    log.info("z_phys cache: %d keys", len(phys_map))

    # ── 2. paper hash -> dopant element (from markdown manifest source_pdf) ────
    mdman = pd.read_csv(HERE / args.md_manifest)
    dopant_re = re.compile(r"/doping/([A-Za-z]{1,3})/")
    hash2dop: dict[str, str] = {}
    for _, r in mdman.iterrows():
        m = dopant_re.search(str(r["source_pdf"]))
        if m:
            hash2dop[str(r["doi_hash"]).strip()] = _norm_el(m.group(1))
    log.info("paper->dopant tags: %d", len(hash2dop))

    # ── 3. downstream-needed keys = unique non-NONE dft_features_idx in CSV ────
    df = pd.read_csv(HERE / args.v58_csv)
    csv_keys = df["dft_features_idx"].astype(str)
    needed_keys = sorted(set(csv_keys[(csv_keys != "NONE")
                                      & (csv_keys.str.contains(r"\|"))]))
    keys_by_cation: dict[str, list[str]] = defaultdict(list)
    for k in needed_keys:
        keys_by_cation[k.split("|")[0]].append(k)
    log.info("downstream-needed z_llm keys: %d (cations: %d)",
             len(needed_keys), len(keys_by_cation))

    # ── 4. Build (paper_id, key) tuples with BOTH z_proto and z_phys ──────────
    tuples: list[tuple[str, str]] = []   # (paper_id, key)
    for pid, dop in hash2dop.items():
        if pid not in proto_map:
            continue
        for k in keys_by_cation.get(dop, []):
            if k in phys_map:
                tuples.append((pid, k))
    if not tuples:
        raise RuntimeError("no (paper, key) tuples with both z_proto and z_phys")

    paper_ids = sorted({t[0] for t in tuples})
    pid2grp = {p: i for i, p in enumerate(paper_ids)}    # InfoNCE group = paper_id
    Z_proto = np.stack([proto_map[t[0]] for t in tuples], 0).astype(np.float32)
    Z_phys = np.stack([phys_map[t[1]] for t in tuples], 0).astype(np.float32)
    grp = np.array([pid2grp[t[0]] for t in tuples], dtype=np.int64)
    N = len(tuples)
    log.info("bridge-training tuples: %d  (papers=%d, keys=%d)",
             N, len(paper_ids), len({t[1] for t in tuples}))

    Z_proto_t = torch.tensor(Z_proto, device=dev)
    Z_phys_t = torch.tensor(Z_phys, device=dev)
    grp_t = torch.tensor(grp, device=dev)

    # ── 5. Bridge + optimizer ─────────────────────────────────────────────────
    bridge = LLMBridge(dim=32, n_heads=4, n_layers=2, dropout=0.1).to(dev)
    n_params = sum(p.numel() for p in bridge.parameters())
    log.info("LLMBridge params: %d", n_params)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=1e-4)

    def info_nce(z: torch.Tensor, groups: torch.Tensor, tau: float) -> torch.Tensor:
        """InfoNCE: positives = same group (paper_id), negatives = rest of batch."""
        zc = F.normalize(z, dim=-1)
        sim = zc @ zc.t() / tau                       # [B, B]
        B = z.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = (groups[:, None] == groups[None, :]) & (~eye)
        # rows with at least one positive contribute
        valid = pos_mask.any(dim=1)
        if valid.sum() == 0:
            return torch.zeros((), device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))     # exclude self
        logZ = torch.logsumexp(sim, dim=1)            # denom over all non-self
        # log-prob mass on positives (mean over positives per anchor)
        exp_sim = torch.exp(sim - logZ[:, None])
        pos_prob = (exp_sim * pos_mask).sum(dim=1).clamp_min(1e-12)
        loss = -torch.log(pos_prob)[valid].mean()
        return loss

    @torch.no_grad()
    def recall_at_k(k: int = 5) -> float:
        """For each tuple, retrieve top-k nearest other tuples by z_LLM cosine;
        hit if any retrieved shares the same paper_id. Mean over anchors that
        have >=1 same-paper partner."""
        bridge.eval()
        z = bridge(Z_proto_t, Z_phys_t)
        zc = F.normalize(z, dim=-1)
        sim = zc @ zc.t()
        B = z.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))
        pos_mask = (grp_t[:, None] == grp_t[None, :]) & (~eye)
        valid = pos_mask.any(dim=1)
        kk = min(k, B - 1)
        topk = sim.topk(kk, dim=1).indices                 # [B, k]
        hit = torch.zeros(B, dtype=torch.bool, device=z.device)
        for r in range(B):
            hit[r] = pos_mask[r, topk[r]].any()
        bridge.train()
        return float(hit[valid].float().mean().item())

    # ── 6. Train ──────────────────────────────────────────────────────────────
    bridge.train()
    bsz = min(args.batch_size, N)
    curve = []
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.choice(N, size=bsz, replace=(bsz > N)),
                           device=dev, dtype=torch.long)
        zp = Z_proto_t[idx]
        zy = Z_phys_t[idx]
        g = grp_t[idx]
        z_llm = bridge(zp, zy)
        l_recon = F.mse_loss(z_llm, zy.detach())
        l_nce = info_nce(z_llm, g, args.tau)
        loss = l_recon + args.lambda_nce * l_nce
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1 or step == args.steps:
            r5 = recall_at_k(5)
            curve.append({
                "step": step,
                "loss": float(loss.item()),
                "l_recon": float(l_recon.item()),
                "l_nce": float(l_nce.item()),
                "recall@5": r5,
            })
            log.info("step %4d | loss %.4f (recon %.4f nce %.4f) | recall@5 %.3f",
                     step, loss.item(), l_recon.item(), l_nce.item(), r5)

    final_r5 = recall_at_k(5)
    final_r1 = recall_at_k(1)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out_dir = HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = out_dir / "bridge.pt"
    torch.save({"bridge": bridge.state_dict(),
                "dim": 32, "n_heads": 4, "n_layers": 2}, bridge_path)
    log.info("saved bridge -> %s", bridge_path)

    metrics = {
        "ok": True,
        "n_tuples": int(N),
        "n_papers": int(len(paper_ids)),
        "n_keys": int(len({t[1] for t in tuples})),
        "n_bridge_params": int(n_params),
        "steps": args.steps,
        "batch_size": bsz,
        "lr": args.lr,
        "lambda_nce": args.lambda_nce,
        "tau": args.tau,
        "final_recall@5": final_r5,
        "final_recall@1": final_r1,
        "final_loss": curve[-1]["loss"],
        "final_l_recon": curve[-1]["l_recon"],
        "final_l_nce": curve[-1]["l_nce"],
        "loss_curve": curve,
        "bridge_path": str(bridge_path),
        "recall_target_met": bool(final_r5 >= 0.7),
    }
    metrics_path = HERE / args.metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote metrics -> %s", metrics_path)

    print(json.dumps({
        "ok": True,
        "n_tuples": N,
        "n_papers": len(paper_ids),
        "n_keys": len({t[1] for t in tuples}),
        "final_recall@5": final_r5,
        "final_loss": curve[-1]["loss"],
        "recall_target_met": final_r5 >= 0.7,
        "bridge_path": str(bridge_path),
        "metrics_path": str(metrics_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
