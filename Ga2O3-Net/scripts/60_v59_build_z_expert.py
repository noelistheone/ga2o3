"""
Phase 59 V59-CALM — build the frozen, cached, MI-distilled multi-expert LLM
embedding ``z_expert`` (16-d).  Implements §3.2(a) + §6 of
docs/phase59_llm_multiexpert_physics_law_design.md.

Recipe (task-agnostic multi-teacher MI distillation, arXiv:2510.18680):
  * Teachers = the ALREADY-EXTRACTED local-LLM embeddings (zero external API):
      - Physics-LLM  (Qwen2.5-1.5B QLoRA) → data/processed/z_phys_cache_v58.npz
        (320 element|atm|T combos, 32-d mean-pooled hidden state, MAE 0.110 dex)
      - Fused multi-expert z_llm          → data/processed/z_llm_cache_v58.npz
        (40 combos, 32-d; built by scripts/52 from the Physics+Process experts)
    For every (element|atm|T) combo we concat the available teachers (zero-pad a
    missing teacher) → a per-combo teacher vector.
  * Student = a tiny MLP  s: teacher_concat → R^16, trained on the UNLABELED set
    of all (element × process) combos (NOT the 14 labeled rows — distillation
    never sees the scarce labels, so it cannot over-fit them, §3.2).
  * Objective = InfoNCE MI-maximization between the L2-normalized student
    embedding and each teacher (a per-teacher linear projection head); positives
    are the same combo, negatives are the other combos in the batch.  This
    maximizes a lower bound on MI(student; teacher_i) for each teacher.
  * After training the student + projections are FROZEN; we cache the 16-d
    student embedding per combo to data/processed/z_expert_v59.npz in the
    {keys, features, meta_json} layout that Ga2O3Net._load_calm_caches expects.

The student/projection weights add ZERO trainable params to the regressor —
z_expert is read-only at fine-tune time (Rule 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _load_z_phys(path: Path) -> dict[str, np.ndarray]:
    """z_phys is saved as savez(**{key: vec}) — every key is its own array."""
    d = np.load(path, allow_pickle=True)
    return {str(k): np.asarray(d[k], dtype=np.float32) for k in d.files}


def _load_keyed(path: Path) -> dict[str, np.ndarray]:
    """z_llm is saved as {keys, features, meta_json}."""
    d = np.load(path, allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    feats = np.asarray(d["features"], dtype=np.float32)
    return {k: feats[i] for i, k in enumerate(keys)}


class Student(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def info_nce(z_student: torch.Tensor, z_teacher: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE between student and one teacher projection (both L2-norm)."""
    logits = (z_student @ z_teacher.t()) / temp        # [N, N]
    labels = torch.arange(z_student.shape[0], device=z_student.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-phys", default="data/processed/z_phys_cache_v58.npz")
    ap.add_argument("--z-llm", default="data/processed/z_llm_cache_v58.npz")
    ap.add_argument("--out", default="data/processed/z_expert_v59.npz")
    ap.add_argument("--out-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")

    phys = _load_z_phys(Path(args.z_phys))
    llm = _load_keyed(Path(args.z_llm)) if Path(args.z_llm).exists() else {}
    print(f"[z_expert] teachers: z_phys={len(phys)} combos (dim={next(iter(phys.values())).shape[0]}), "
          f"z_llm={len(llm)} combos")

    # Union of combos (the UNLABELED element×process set).
    keys = sorted(set(phys) | set(llm))
    d_phys = next(iter(phys.values())).shape[0] if phys else 0
    d_llm = next(iter(llm.values())).shape[0] if llm else 0

    # Per-teacher matrices + presence masks (zero-pad missing teacher).
    T_phys = np.zeros((len(keys), d_phys), np.float32)
    T_llm = np.zeros((len(keys), d_llm), np.float32)
    m_phys = np.zeros(len(keys), np.float32)
    m_llm = np.zeros(len(keys), np.float32)
    for i, k in enumerate(keys):
        if k in phys:
            T_phys[i] = phys[k]; m_phys[i] = 1.0
        if k in llm:
            T_llm[i] = llm[k]; m_llm[i] = 1.0

    # Standardize each teacher (per-dim z-score over present rows) for stable InfoNCE.
    def _zscore(M, mask):
        present = mask > 0
        if present.sum() < 2:
            return M
        mu = M[present].mean(0, keepdims=True)
        sd = M[present].std(0, keepdims=True) + 1e-6
        return (M - mu) / sd

    T_phys = _zscore(T_phys, m_phys)
    T_llm = _zscore(T_llm, m_llm)

    teacher_concat = np.concatenate([T_phys, T_llm], axis=1)   # [N, d_phys+d_llm]
    tc = torch.tensor(teacher_concat, device=device)
    tp = torch.tensor(T_phys, device=device)
    tl = torch.tensor(T_llm, device=device)
    mp = torch.tensor(m_phys, device=device)
    ml = torch.tensor(m_llm, device=device)

    student = Student(in_dim=teacher_concat.shape[1], out_dim=args.out_dim).to(device)
    proj_phys = nn.Linear(d_phys, args.out_dim).to(device) if d_phys else None
    proj_llm = nn.Linear(d_llm, args.out_dim).to(device) if d_llm else None
    params = list(student.parameters())
    if proj_phys is not None: params += list(proj_phys.parameters())
    if proj_llm is not None: params += list(proj_llm.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    for ep in range(1, args.epochs + 1):
        student.train()
        opt.zero_grad()
        zs = student(tc)                                   # [N, 16] L2-norm
        loss = torch.zeros((), device=device)
        # Teacher 1 (physics) — restrict InfoNCE to rows where physics teacher present.
        if proj_phys is not None and mp.sum() > 1:
            idx = (mp > 0).nonzero(as_tuple=True)[0]
            zt = F.normalize(proj_phys(tp[idx]), dim=-1)
            loss = loss + info_nce(zs[idx], zt, args.temp)
        if proj_llm is not None and ml.sum() > 1:
            idx = (ml > 0).nonzero(as_tuple=True)[0]
            zt = F.normalize(proj_llm(tl[idx]), dim=-1)
            loss = loss + info_nce(zs[idx], zt, args.temp)
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == 1:
            print(f"[z_expert] epoch {ep:4d}  MI-InfoNCE loss={loss.item():.4f}")

    # Freeze + emit the 16-d student embedding per combo.
    student.eval()
    with torch.no_grad():
        feats = student(tc).cpu().numpy().astype(np.float32)   # L2-normalized 16-d

    meta = {
        "source": "MI-distilled (InfoNCE) from z_phys_cache_v58 (Physics-LLM Qwen2.5-1.5B) "
                  "+ z_llm_cache_v58 (fused multi-expert); arXiv:2510.18680",
        "out_dim": args.out_dim, "n_combos": len(keys),
        "teacher_dims": {"phys": d_phys, "llm": d_llm},
        "frozen": True, "phase": "59 V59-CALM z_expert",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, keys=np.array(keys, dtype="U32"), features=feats,
             meta_json=json.dumps(meta))
    print(f"[z_expert] wrote {args.out}  keys={len(keys)} dim={args.out_dim}")
    # Sanity: report a couple of cosine sims (donor vs acceptor should differ).
    fk = {k: feats[i] for i, k in enumerate(keys)}
    def cos(a, b):
        if a not in fk or b not in fk: return float("nan")
        x, y = fk[a], fk[b]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    print(f"[z_expert] cos(Sn|Ar|700, Si|Ar|700)={cos('Sn|Ar|700','Si|Ar|700'):.3f} (donor-donor); "
          f"cos(Sn|Ar|700, Mg|Ar|700)={cos('Sn|Ar|700','Mg|Ar|700'):.3f} (donor-acceptor)")


if __name__ == "__main__":
    main()
