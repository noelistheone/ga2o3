"""Phase 63 — leave-one-ELEMENT-out: how well does the surrogate predict a BRAND-NEW dopant element's trend?

For each held element E: train a sum world-model on configs that contain NO E at all, then test on configs
that DO contain E (single-E and E co-doped with a random partner). Reports trend rho + R^2 (vs the V63 solver)
on log[V_O] and E_F. This isolates the MODEL's contribution to new-dopant error within the solver-world
(the token carries E's physical descriptor + carrier sign + transition level, so this measures whether the
phys-encoding lets the model place a never-seen element). Pairs with the Phase-62 leave-one-element-out on
REAL experimental data (cross_element_trend.json). GPU, deterministic. -> results/phase63/new_element_test.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.data.codoping_oracle import (sample_recipes, build_batch, label_with_oracle, VOCAB, DONORS,  # noqa: E402
                                       ACCEPTORS, ISOVALENT)
from src.models.v63_world_model import V63WorldModel                                                  # noqa: E402
from scipy.stats import spearmanr                                                                      # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TGT = ["E_F", "log10_VO", "bound_frac"]
RNG = np.random.default_rng(7)


def train_sum(rec, lab, n_epoch=60, bs=4096):
    Y = torch.stack([lab[t] for t in TGT], 1); mean = Y.mean(0, keepdim=True); std = Y.std(0, keepdim=True).clamp_min(1e-6)
    Ys = ((Y - mean) / std).to(DEV)
    tok, msk, T, lp = rec["tokens"].to(DEV), rec["mask"].to(DEV), rec["T"].to(DEV), rec["lpo2"].to(DEV)
    m = V63WorldModel(agg="sum").to(DEV); opt = torch.optim.Adam(m.parameters(), 2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epoch)
    n = tok.shape[0]; g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(n_epoch):
        perm = torch.randperm(n, generator=g, device=DEV)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]; o = m(tok[idx], msk[idx], T[idx], lp[idx])
            pred = torch.stack([o["E_F"], o["log10_VO"], o["bound_frac"]], 1)
            loss = nn.functional.mse_loss(pred, Ys[idx]); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return m, mean, std


@torch.no_grad()
def pred(m, recipes, T, lp, mean, std):
    rc = build_batch(recipes, T, lp)
    o = m(rc["tokens"].to(DEV), rc["mask"].to(DEV), rc["T"].to(DEV), rc["lpo2"].to(DEV))
    return (torch.stack([o["E_F"], o["log10_VO"], o["bound_frac"]], 1).cpu() * std + mean)


def r2(y, p):
    y = np.asarray(y); p = np.asarray(p); tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / tot) if tot > 0 else float("nan")


def main():
    g = torch.Generator().manual_seed(1)
    pool = sample_recipes(140000, g, d_choices=(1, 2, 3), force_da=0.4)
    eids = pool["elem_ids"]
    rep = {"device": DEV, "note": "leave-one-ELEMENT-out on the surrogate (vs V63 solver). Token carries the "
           "held element's physical descriptor + carrier sign + transition level; measures whether phys-encoding "
           "lets the model place a never-seen element. Compare to REAL-data cross_element_trend.json (Phase 62).",
           "held": {}}
    held = ["Sn", "Mg", "Ge", "Zn", "Ti", "In"]
    print(f"{'held':5s} {'carrier':9s} | {'logVO_R2':>8s} {'logVO_rho':>9s} {'EF_R2':>6s} {'EF_rho':>6s}  (single + co-doped, vs solver)")
    for E in held:
        ei = VOCAB.index(E)
        contains = (eids == ei).any(dim=1)
        keep = ~contains
        rec_tr = {k: (v[keep] if torch.is_tensor(v) and v.shape[0] == keep.shape[0] else v) for k, v in pool.items()}
        lab_tr = label_with_oracle(rec_tr, device=DEV, chunk=5000)
        m, mean, std = train_sum(rec_tr, lab_tr)
        # test: 1500 single-E + 1500 E co-doped with a random partner
        others = [e for e in VOCAB if e != E]
        recs, T, lp = [], [], []
        for _ in range(1500):
            recs.append([(E, float(10 ** RNG.uniform(-3.5, -1.5)))]); T.append(float(RNG.uniform(700, 1200))); lp.append(float(RNG.uniform(-7, 0)))
        for _ in range(1500):
            o = str(RNG.choice(others))
            recs.append([(E, float(10 ** RNG.uniform(-3.5, -1.5))), (o, float(10 ** RNG.uniform(-3.5, -1.5)))])
            T.append(float(RNG.uniform(700, 1200))); lp.append(float(RNG.uniform(-7, 0)))
        lab = label_with_oracle(build_batch(recs, T, lp), device=DEV, chunk=5000)
        p = pred(m, recs, T, lp, mean, std)
        sign = "donor" if E in DONORS else ("acceptor" if E in ACCEPTORS else "isovalent")
        res = dict(carrier=sign, n_train=int(keep.sum()),
                   logVO_R2=round(r2(lab["log10_VO"].numpy(), p[:, 1].numpy()), 3),
                   logVO_rho=round(float(spearmanr(lab["log10_VO"].numpy(), p[:, 1].numpy()).correlation), 3),
                   EF_R2=round(r2(lab["E_F"].numpy(), p[:, 0].numpy()), 3),
                   EF_rho=round(float(spearmanr(lab["E_F"].numpy(), p[:, 0].numpy()).correlation), 3))
        rep["held"][E] = res
        print(f"{E:5s} {sign:9s} | {res['logVO_R2']:8.3f} {res['logVO_rho']:9.3f} {res['EF_R2']:6.3f} {res['EF_rho']:6.3f}")
    # aggregate
    rep["summary"] = dict(
        mean_logVO_rho=round(float(np.mean([v["logVO_rho"] for v in rep["held"].values()])), 3),
        mean_logVO_R2=round(float(np.mean([v["logVO_R2"] for v in rep["held"].values()])), 3),
        mean_EF_rho=round(float(np.mean([v["EF_rho"] for v in rep["held"].values()])), 3))
    out = PROJ / "results/phase63/new_element_test.json"; out.write_text(json.dumps(rep, indent=2))
    print(f"\nMEAN over held elements: logVO rho={rep['summary']['mean_logVO_rho']} R2={rep['summary']['mean_logVO_R2']} "
          f"EF rho={rep['summary']['mean_EF_rho']}\nsaved -> {out}")


if __name__ == "__main__":
    main()
