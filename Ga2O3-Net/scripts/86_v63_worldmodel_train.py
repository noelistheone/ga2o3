"""Phase 63 — train the doping-type-general world-model surrogate by distilling the V63 solver oracle.

Trains 4 architectures on the SAME data to isolate the inductive bias:
  sum   : Deep Sets sum-pool          (hypothesis: extrapolates the additive charge law across cardinality)
  mean  : Deep Sets mean-pool         (ablation: normalizes cardinality -> should fail D-extrapolation)
  attn  : masked self-attention       (can capture the PAIRWISE DAP-pairing/bound_frac mechanism)
  slot  : fixed-slot flattened MLP     (control: order-sensitive, cardinality-bound)
plus an RF baseline on sum-pooled raw tokens (the required tree gate).

Train set: D in {1,2} (single + binary), force_da=0.5 so pairing has signal.
Held-out:  test_highD = D in {3,4}  -> CARDINALITY EXTRAPOLATION (the headline law-transfer test).
           test_idD   = D in {1,2}  -> in-distribution sanity.
Targets standardized on train. Metrics (R2/Spearman/MAE) reported in ORIGINAL units. A frozen linear probe
decodes E_F from the latent (latent=state diagnostic; NOT the mechanism claim -- see law tests). GPU.
Writes results/phase63/worldmodel/{metrics.json, *.pt, standardize.json} + saves test tensors for reuse.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.data.codoping_oracle import sample_recipes, label_with_oracle, TOKEN_DIM, D_MAX  # noqa: E402
from src.models.v63_world_model import V63WorldModel, SlotMLP                              # noqa: E402
from scipy.stats import spearmanr                                                          # noqa: E402
from sklearn.ensemble import RandomForestRegressor                                         # noqa: E402
from sklearn.linear_model import Ridge                                                     # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TARGETS = ["E_F", "log10_VO", "bound_frac"]


def gen_split(n, seed, d_choices, force_da=0.0):
    g = torch.Generator().manual_seed(seed)
    rec = sample_recipes(n, g, d_choices=d_choices, force_da=force_da)
    lab = label_with_oracle(rec, device=DEV, chunk=5000)
    return rec, lab


def _stack_targets(lab):
    return torch.stack([lab[t] for t in TARGETS], dim=1)  # [n,3]


def metrics(pred, true):
    out = {}
    for i, t in enumerate(TARGETS):
        p = pred[:, i].numpy(); y = true[:, i].numpy()
        ss_res = float(((y - p) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rho = float(spearmanr(y, p).correlation)
        out[t] = dict(R2=round(r2, 4), spearman=round(rho, 4), MAE=round(float(np.abs(y - p).mean()), 4))
    return out


def train_model(model, Xtr, Ytr_std, n_epoch=60, bs=4096, lr=2e-3):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epoch)
    tok, msk, T, lp = [x.to(DEV) for x in Xtr]
    Y = Ytr_std.to(DEV)
    n = tok.shape[0]
    g = torch.Generator(device=DEV).manual_seed(0)
    for ep in range(n_epoch):
        perm = torch.randperm(n, generator=g, device=DEV)
        model.train()
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            out = model(tok[idx], msk[idx], T[idx], lp[idx])
            pred = torch.stack([out["E_F"], out["log10_VO"], out["bound_frac"]], dim=1)
            loss = nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, X, mean, std):
    model.eval()
    tok, msk, T, lp = [x.to(DEV) for x in X]
    out = model(tok, msk, T, lp)
    pred = torch.stack([out["E_F"], out["log10_VO"], out["bound_frac"]], dim=1).cpu()
    return pred * std + mean


def to_X(rec):
    return (rec["tokens"], rec["mask"], rec["T"], rec["lpo2"])


def rf_features(rec):
    """Sum-pooled raw tokens + cardinality + knobs (fair tree baseline = tree on hand-sum-pooled features)."""
    tok = rec["tokens"] * rec["mask"].unsqueeze(-1)
    S = tok.sum(dim=1)                                   # [n,TOKEN_DIM]
    D = rec["mask"].sum(dim=1, keepdim=True)
    return torch.cat([S, D, rec["T"].unsqueeze(1), rec["lpo2"].unsqueeze(1)], dim=1).numpy()


def main():
    t0 = time.time()
    outdir = PROJ / "results/phase63/worldmodel"; outdir.mkdir(parents=True, exist_ok=True)
    datadir = PROJ / "data/processed/phase63_oracle"; datadir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print(f"[{time.time()-t0:.0f}s] generating data on {DEV} ...")
    rec_tr, lab_tr = gen_split(120000, 1, d_choices=(1, 2), force_da=0.5)
    rec_idD, lab_idD = gen_split(15000, 2, d_choices=(1, 2), force_da=0.25)
    rec_hiD, lab_hiD = gen_split(30000, 3, d_choices=(3, 4), force_da=0.0)   # natural high-cardinality OOD
    print(f"[{time.time()-t0:.0f}s] train={rec_tr['tokens'].shape[0]} idD={rec_idD['tokens'].shape[0]} "
          f"hiD={rec_hiD['tokens'].shape[0]}")

    Ytr = _stack_targets(lab_tr); mean = Ytr.mean(0, keepdim=True); std = Ytr.std(0, keepdim=True).clamp_min(1e-6)
    Ytr_std = (Ytr - mean) / std
    Xtr, X_idD, X_hiD = to_X(rec_tr), to_X(rec_idD), to_X(rec_hiD)
    Y_idD, Y_hiD = _stack_targets(lab_idD), _stack_targets(lab_hiD)

    results = {}
    builders = {
        "sum": lambda: V63WorldModel(agg="sum"), "mean": lambda: V63WorldModel(agg="mean"),
        "attn": lambda: V63WorldModel(agg="attn"), "slot": lambda: SlotMLP(d_max=D_MAX),
        "sum_gated": lambda: V63WorldModel(agg="sum", conc_gate=True),
    }
    for name, build in builders.items():
        tm = time.time()
        model = train_model(build(), Xtr, Ytr_std)
        pred_id = predict(model, X_idD, mean, std)
        pred_hi = predict(model, X_hiD, mean, std)
        results[name] = dict(in_distribution_D12=metrics(pred_id, Y_idD),
                             extrapolation_D34=metrics(pred_hi, Y_hiD),
                             train_s=round(time.time() - tm, 1))
        torch.save(model.state_dict(), outdir / f"wm_{name}.pt")
        print(f"[{time.time()-t0:.0f}s] {name:5s} trained {results[name]['train_s']}s  "
              f"D34 E_F R2={results[name]['extrapolation_D34']['E_F']['R2']:.3f} "
              f"logVO R2={results[name]['extrapolation_D34']['log10_VO']['R2']:.3f} "
              f"bf R2={results[name]['extrapolation_D34']['bound_frac']['R2']:.3f}")

    # RF baseline (tree gate) on sum-pooled features
    Ftr, F_id, F_hi = rf_features(rec_tr), rf_features(rec_idD), rf_features(rec_hiD)
    rf_res = {}
    for i, t in enumerate(TARGETS):
        rf = RandomForestRegressor(n_estimators=300, max_depth=14, n_jobs=-1, random_state=0)
        rf.fit(Ftr, Ytr[:, i].numpy())
        for split, F, Y in [("in_distribution_D12", F_id, Y_idD), ("extrapolation_D34", F_hi, Y_hiD)]:
            p = rf.predict(F); y = Y[:, i].numpy()
            ss_res = float(((y - p) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
            rf_res.setdefault(split, {})[t] = dict(
                R2=round(1 - ss_res / ss_tot, 4), spearman=round(float(spearmanr(y, p).correlation), 4),
                MAE=round(float(np.abs(y - p).mean()), 4))
    results["rf"] = rf_res
    print(f"[{time.time()-t0:.0f}s] rf    D34 E_F R2={rf_res['extrapolation_D34']['E_F']['R2']:.3f} "
          f"logVO R2={rf_res['extrapolation_D34']['log10_VO']['R2']:.3f} "
          f"bf R2={rf_res['extrapolation_D34']['bound_frac']['R2']:.3f}")

    # latent = state: frozen linear probe of E_F from the SUM model's latent (fit on idD, eval on hiD)
    sum_model = V63WorldModel(agg="sum").to(DEV); sum_model.load_state_dict(torch.load(outdir / "wm_sum.pt"))
    sum_model.eval()
    with torch.no_grad():
        z_id = sum_model.encode(*[x.to(DEV) for x in X_idD]).cpu().numpy()
        z_hi = sum_model.encode(*[x.to(DEV) for x in X_hiD]).cpu().numpy()
    probe = Ridge(alpha=1.0).fit(z_id, Y_idD[:, 0].numpy())
    pe = probe.predict(z_hi); ye = Y_hiD[:, 0].numpy()
    probe_r2 = 1 - float(((ye - pe) ** 2).sum()) / float(((ye - ye.mean()) ** 2).sum())
    results["latent_probe_EF_R2_on_D34"] = round(probe_r2, 4)

    rep = dict(device=DEV, n_train=120000, targets=TARGETS, results=results,
               note="OOD = cardinality extrapolation (train D in {1,2} -> test D in {3,4}). Labels are the "
                    "V63 self-consistent complex solver oracle; this tests LAW learning, not experimental "
                    "magnitude (research: oracle uncorrelated with cross-paper absolute [V_O]).")
    (outdir / "metrics.json").write_text(json.dumps(rep, indent=2))
    json.dump(dict(mean=mean.squeeze().tolist(), std=std.squeeze().tolist(), targets=TARGETS),
              open(outdir / "standardize.json", "w"), indent=2)
    torch.save(dict(rec_hiD=rec_hiD, lab_hiD=lab_hiD), datadir / "test_highD.pt")
    print(f"[{time.time()-t0:.0f}s] DONE -> {outdir}/metrics.json   latent-probe E_F R2(D34)={probe_r2:.3f}")


if __name__ == "__main__":
    main()
