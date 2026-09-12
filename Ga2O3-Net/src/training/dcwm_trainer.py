"""Phase 60 V60-DCWM — Stage-A (world-model pretrain) + Stage-B (Δ-readout fit) trainers.

Stage-A: end-to-end LeWM objective on cached physics transitions (no EMA / no stop-grad):
    ℒ = ‖g_φ(f_θ(x_s),a) − f_θ(x_t)‖²  +  λ·SIGReg(f_θ(x_s))  +  γ·readout-probe(meas+bulk, s&t)
    (+ optional μ_cyc cycle-consistency on continuous-only actions).
The readout probe co-trains r_ψ on the SIMULATED V_O (free physics labels) so the latent is
V_O-decodable; the counterfactual law then flows through g_φ. NO experimental labels in Stage-A.

Stage-B: freeze f_θ,g_φ,r_ψ; fit the zero-init Δ-correction `delta_meas(z)` on the experimental VC
labels, GroupKFold-by-DOI (leakage-free), collect OOF. Δ-learning over the WM-readout floor (Rule 2
spirit): the law (slope) is owned by the frozen WM; experiment only corrects magnitude.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[2]
ACTION_CONT = slice(0, 3)          # continuous action dims (log c, T, pO2)
ACTION_FLAGS = [9, 21, 25]         # method/element/device changed-flags


# ──────────────────────────────────────────────────────────────────────────────
# Stage-A
# ──────────────────────────────────────────────────────────────────────────────
def train_stageA(model, cache: dict, device, *, steps=40000, batch=4096, lr=1e-3,
                 lam_sig=0.05, gam_ro=1.0, mu_cyc=0.0, weight_decay=1e-5,
                 log_every=500, log_path=None, seed=0, exp_X=None, lam_sig_exp=0.5):
    """Train the world model on cached transitions. cache holds numpy arrays:
    x_s,x_t [N,D]; a [N,A]; ym_s,yb_s,ym_t,yb_t [N]. Returns the loss log (list of dicts).

    exp_X: optional [M,D] experimental features (all rows, NO labels) — an extra SIGReg term keeps
    experimental z ~ N(0,I) too, fixing the OOD-z collapse (the deployed model is evaluated on these).
    """
    torch.manual_seed(seed)
    model.to(device).train()
    Xexp = torch.tensor(exp_X, device=device) if exp_X is not None else None
    # move cache to GPU as tensors (standardize targets once)
    X_s = torch.tensor(cache["x_s"], device=device)
    X_t = torch.tensor(cache["x_t"], device=device)
    A = torch.tensor(cache["a"], device=device)
    ym_s = (torch.tensor(cache["ym_s"], device=device) - model.y_meas_mean) / model.y_meas_std
    ym_t = (torch.tensor(cache["ym_t"], device=device) - model.y_meas_mean) / model.y_meas_std
    yb_s = (torch.tensor(cache["yb_s"], device=device) - model.y_bulk_mean) / model.y_bulk_std
    yb_t = (torch.tensor(cache["yb_t"], device=device) - model.y_bulk_mean) / model.y_bulk_std
    N = X_s.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device=device).manual_seed(seed)
    log = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g, device=device)
        xs, xt, a = X_s[idx], X_t[idx], A[idx]
        o = model.forward_stageA(xs, a, xt)
        L_pred = F.mse_loss(o["z_pred"], o["z_t"])
        L_sig = model.sigreg(o["z_s"])
        L_ro = (F.mse_loss(o["m_s"], ym_s[idx]) + F.mse_loss(o["m_t"], ym_t[idx])
                + F.mse_loss(o["b_s"], yb_s[idx]) + F.mse_loss(o["b_t"], yb_t[idx]))
        loss = L_pred + lam_sig * L_sig + gam_ro * L_ro
        if Xexp is not None and lam_sig_exp > 0:
            # regularize experimental-recipe embeddings to N(0,I) too (fix OOD-z collapse)
            z_exp = model.encode(Xexp)
            loss = loss + lam_sig_exp * lam_sig * model.sigreg(z_exp)
        if mu_cyc > 0:
            # cycle-consistency on continuous-only actions (no categorical change)
            cont_only = (a[:, ACTION_FLAGS].abs().sum(dim=1) == 0)
            if cont_only.any():
                a_inv = a[cont_only].clone(); a_inv[:, ACTION_CONT] *= -1.0
                z_back = model.predict(o["z_pred"][cont_only], a_inv)
                loss = loss + mu_cyc * F.mse_loss(z_back, o["z_s"][cont_only])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps - 1:
            rec = dict(step=step, loss=float(loss.item()), pred=float(L_pred.item()),
                       sig=float(L_sig.item()), ro=float(L_ro.item()),
                       lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1))
            log.append(rec)
            print(f"  [A] step {step:6d}/{steps}  loss={rec['loss']:.4f} "
                  f"pred={rec['pred']:.4f} sig={rec['sig']:.4f} ro={rec['ro']:.4f} "
                  f"lr={rec['lr']:.2e}  {rec['sec']}s", flush=True)
    if log_path:
        import csv
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
    return log


# ──────────────────────────────────────────────────────────────────────────────
# Stage-B  (GroupKFold-by-DOI Δ-readout fit; WM frozen)
# ──────────────────────────────────────────────────────────────────────────────
def build_readout(d, hidden=64, dropout=0.2):
    return torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.SiLU(),
                               torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))


def _fit_readout(Z, y, sw, device, epochs, lr, wd, hidden, dropout, seed,
                 law_z=None, law_dz=None, law_sign=None, mu_law=0.0):
    """Fit a readout MLP r(z)->standardized real y on (Z,y). Optional law-preservation reg:
    on synthetic counterfactual pairs (law_z -> law_z+law_dz with expected sign law_sign), penalize
    wrong-signed predicted change (one-sided hinge) so the deployed readout keeps the WM's law."""
    torch.manual_seed(seed)
    r = build_readout(Z.shape[1], hidden, dropout).to(device)
    ymean, ystd = y.mean(), y.std().clamp_min(1e-6)
    yt = (y - ymean) / ystd
    w = (sw / sw.mean().clamp_min(1e-6))
    opt = torch.optim.Adam(r.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        r.train()
        pred = r(Z).squeeze(-1)
        loss = (w * (pred - yt) ** 2).mean()
        if mu_law > 0 and law_z is not None:
            p0 = r(law_z).squeeze(-1); p1 = r(law_z + law_dz).squeeze(-1)
            # want sign(p1-p0)==law_sign -> penalize -law_sign*(p1-p0) when negative
            loss = loss + mu_law * torch.relu(-law_sign * (p1 - p0)).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    r.eval()
    return r, float(ymean), float(ystd)


def _fit_delta(Z, resid, sw, device, epochs, lr, wd, seed, z_perts=None, rho_lip=0.0):
    """Fit a Δ-head Linear(d,32)-SiLU-Linear(32,1) (matches model.delta_meas) to the residual
    (y - r_ψ(z)) in REAL log10[V_O] units. Zero-init last layer; weight decay bounds it.

    z_perts: optional list of concentration-perturbed latents (via the WM) for the Δ-LIPSCHITZ prior
    (rho_lip): penalize Δ varying under a concentration intervention so Δ is a pure magnitude OFFSET
    and CANNOT inject within-DOI concentration slope (the law stays in r_ψ). This kills the Δ-driven
    within-DOI FLIPs (MOCVD-Si / H / N / Mg@apsusc) seen in the GNG."""
    torch.manual_seed(seed)
    d = Z.shape[1]
    delta = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1)).to(device)
    torch.nn.init.zeros_(delta[-1].weight); torch.nn.init.zeros_(delta[-1].bias)
    opt = torch.optim.Adam(delta.parameters(), lr=lr, weight_decay=wd)
    w = (sw / sw.mean().clamp_min(1e-6))
    for _ in range(epochs):
        pred = delta(Z).squeeze(-1)
        loss = (w * (pred - resid) ** 2).mean()
        if rho_lip > 0 and z_perts:
            d0 = delta(Z).squeeze(-1)
            lip = sum(((delta(zp).squeeze(-1) - d0) ** 2).mean() for zp in z_perts) / len(z_perts)
            loss = loss + rho_lip * lip
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(delta.parameters(), 5.0)
        opt.step()
    delta.eval()
    return delta


def fit_stageB(model, exp: dict, device, *, n_folds=10, n_seeds=5, epochs=400, lr=3e-3,
               weight_decay=3e-3, rho_lip=0.0, **_ignore):
    """Δ-learning over the FROZEN WM readout (the design-faithful path): ŷ = r_ψ(z) + Δ(z).
    The WM readout r_ψ carries the physics LAW (counterfactual 9/9); experiment fits only a magnitude
    Δ-offset. GroupKFold-by-DOI, 5-seed ensemble. Returns OOF + a global Δ for the deployed model
    (baked into model.delta_meas; the Stage-C counterfactual holds Δ at the base recipe so the law is
    preserved exactly)."""
    from sklearn.model_selection import GroupKFold
    model.to(device).eval()
    X = torch.tensor(exp["X"], device=device)
    y = torch.tensor(exp["y"], device=device)
    sw = torch.tensor(exp["sample_weight"], device=device)
    groups = exp["doi"]
    N = len(y)
    with torch.no_grad():
        Z = model.encode(X).detach()
        base_meas, _ = model.readout_real(Z, apply_delta=False)     # r_ψ(z) real units (carries law)
        # concentration-perturbed latents via the WM, for the Δ-Lipschitz prior
        z_perts = None
        if rho_lip > 0:
            adim = model.action_enc.net[0].in_features
            z_perts = []
            for dlc in (+0.5, -0.5):
                a = torch.zeros(N, adim, device=device); a[:, 0] = dlc
                z_perts.append(model.predict(Z, a).detach())
    resid = (y - base_meas).detach()
    d = Z.shape[1]
    folds = min(n_folds, len(set(groups.tolist())) if hasattr(groups, "tolist") else len(set(groups)))
    gkf = GroupKFold(n_splits=folds)

    oof_seedavg = np.zeros(N, dtype=np.float64)
    for seed in range(n_seeds):
        seed_oof = np.full(N, np.nan, dtype=np.float64)
        for tr, va in gkf.split(np.arange(N), groups=groups):
            tr_t = torch.tensor(tr, device=device); va_t = torch.tensor(va, device=device)
            zp_tr = [zp[tr_t] for zp in z_perts] if z_perts else None
            delta = _fit_delta(Z[tr_t], resid[tr_t], sw[tr_t], device, epochs, lr, weight_decay,
                               seed=1000 + seed, z_perts=zp_tr, rho_lip=rho_lip)
            with torch.no_grad():
                seed_oof[va] = (base_meas[va_t] + delta(Z[va_t]).squeeze(-1)).cpu().numpy()
        oof_seedavg += seed_oof
    oof = oof_seedavg / n_seeds

    gdelta = _fit_delta(Z, resid, sw, device, epochs, lr, weight_decay, seed=2024,
                        z_perts=z_perts, rho_lip=rho_lip)   # deployed Δ
    return dict(oof_pred=oof, y=exp["y"].astype(np.float64), element=exp["element"],
                doi=exp["doi"], conc=exp["conc"], base_meas=base_meas.detach().cpu().numpy(),
                global_delta_state={k: v.detach().cpu() for k, v in gdelta.state_dict().items()})
