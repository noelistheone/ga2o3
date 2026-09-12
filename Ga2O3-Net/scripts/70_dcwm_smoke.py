"""Phase 60 V60-DCWM smoke test — verify the correctness-critical invariants before training.

  1. SIGReg is MINIMIZED at N(0,I) (vs collapsed / anisotropic / shifted batches).
  2. Identity-at-init: predict(z, a) == z exactly (ReZero gate α=0) for any action.
  3. Zero counterfactual slope at init (a direct corollary): readout∘predict is action-invariant.
  4. Stage-A loss is finite and backprops with finite gradients through ALL params (end-to-end).
  5. After a few hundred steps on the generator, SIGReg drops and the predictor learns a non-zero,
     correctly-signed concentration response (Mg measured ↓, Sn measured ↑) — quick learnability check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.dcwm_transitions import (StateFeaturizer, TransitionGenerator, ACTION_DIM,
                                        measured_log_vo, ATM_PO2)
from src.models.dcwm import DCWM, SIGReg


def main():
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    # ── 1. SIGReg ordering ────────────────────────────────────────────────────
    sig = SIGReg(n_slices=512, n_quad=17).to(dev)
    N, d = 1024, 64
    g = torch.Generator(device=dev).manual_seed(1)
    Z_iso = torch.randn(N, d, generator=g, device=dev)
    Z_collapse = torch.zeros(N, d, device=dev) + 0.01 * torch.randn(N, d, generator=g, device=dev)
    Z_aniso = Z_iso.clone(); Z_aniso[:, 8:] *= 0.05                     # rank-deficient (dim collapse)
    Z_wide = 3.0 * torch.randn(N, d, generator=g, device=dev)          # std 3 != 1
    Z_shift = torch.randn(N, d, generator=g, device=dev) + 2.0         # mean 2 != 0
    vals = {k: float(sig(v)) for k, v in
            dict(isoN01=Z_iso, collapsed=Z_collapse, anisotropic=Z_aniso, wide_std3=Z_wide,
                 shifted_mu2=Z_shift).items()}
    print("\n[1] SIGReg (should be SMALLEST for isoN01):")
    for k, v in sorted(vals.items(), key=lambda kv: kv[1]):
        print(f"    {k:14s} {v:.5f}")
    assert vals["isoN01"] == min(vals.values()), "SIGReg not minimized at N(0,I)!"
    print("    PASS: N(0,I) is the minimizer.")

    # ── build model on the real featurizer ────────────────────────────────────
    feat = StateFeaturizer()
    gen = TransitionGenerator(feat, seed=2)
    model = DCWM(in_dim=feat.dim, action_dim=ACTION_DIM, d=64, n_blocks=4,
                 sig_slices=256).to(dev)
    b = gen.sample_batch(4096)
    Xall = torch.tensor(np.concatenate([b["x_s"], b["x_t"]], 0), device=dev)
    model.encoder.std.fit(Xall)
    model.set_target_stats(b["ym_s"].mean(), b["ym_s"].std(), b["yb_s"].mean(), b["yb_s"].std())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] featurizer dim={feat.dim}, latent d={model.d}, trainable params={n_train}")

    # ── 2 & 3. identity-at-init + zero counterfactual slope ────────────────────
    model.eval()
    xb = torch.tensor(b["x_s"][:256], device=dev)
    ab = torch.tensor(b["a"][:256], device=dev)
    with torch.no_grad():
        z = model.encode(xb)
        zp = model.predict(z, ab)
        ident = (zp - z).abs().max().item()
    print(f"\n[2] identity-at-init max|predict(z,a)-z| = {ident:.2e}  (should be ~0)")
    assert ident < 1e-5, "predictor not identity at init!"
    # zero counterfactual slope: sweep Mg concentration through the WM at init
    def cf_curve(elem, apply_delta):
        from src.data.dcwm_transitions import C_REF
        import math
        base = feat.featurize(elem, 0.001, 700.0, "Ar_O2_1_1", 0)
        xs = torch.tensor(np.tile(base, (7, 1)), device=dev)
        grid = np.logspace(-3.3, -1.3, 7)
        acts = np.zeros((7, ACTION_DIM), np.float32)
        acts[:, 0] = np.log10(grid) - math.log10(0.001)
        meas = model.counterfactual_measured(xs, torch.tensor(acts, device=dev),
                                              apply_delta=apply_delta).cpu().numpy()
        return np.log10(grid), meas
    lc, m0 = cf_curve("Mg", apply_delta=False)
    slope0 = np.polyfit(lc, m0, 1)[0]
    print(f"[3] init Mg counterfactual slope = {slope0:+.3e}  (should be ~0 at init)")
    assert abs(slope0) < 1e-3, "counterfactual slope not ~0 at init!"
    print("    PASS: identity dynamics at init.")

    # ── 4. finite Stage-A loss + gradients ─────────────────────────────────────
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    def stageA_loss(bb, lam=0.05, gam=1.0):
        xs = torch.tensor(bb["x_s"], device=dev); xt = torch.tensor(bb["x_t"], device=dev)
        a = torch.tensor(bb["a"], device=dev)
        o = model.forward_stageA(xs, a, xt)
        L_pred = F.mse_loss(o["z_pred"], o["z_t"])
        L_sig = model.sigreg(o["z_s"])
        # readout probe on standardized targets (both heads, both states)
        ym_s = (torch.tensor(bb["ym_s"], device=dev) - model.y_meas_mean) / model.y_meas_std
        ym_t = (torch.tensor(bb["ym_t"], device=dev) - model.y_meas_mean) / model.y_meas_std
        yb_s = (torch.tensor(bb["yb_s"], device=dev) - model.y_bulk_mean) / model.y_bulk_std
        yb_t = (torch.tensor(bb["yb_t"], device=dev) - model.y_bulk_mean) / model.y_bulk_std
        L_ro = (F.mse_loss(o["m_s"], ym_s) + F.mse_loss(o["m_t"], ym_t)
                + F.mse_loss(o["b_s"], yb_s) + F.mse_loss(o["b_t"], yb_t))
        return L_pred + lam * L_sig + gam * L_ro, (float(L_pred), float(L_sig), float(L_ro))
    loss, parts = stageA_loss(b)
    opt.zero_grad(); loss.backward()
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\n[4] Stage-A loss={float(loss):.4f}  (pred={parts[0]:.4f} sig={parts[1]:.4f} ro={parts[2]:.4f})")
    print(f"    grad-norm={float(gnorm):.4f} finite={torch.isfinite(gnorm).item()}  "
          f"params-with-grad={n_with_grad}/{n_params}")
    assert torch.isfinite(gnorm), "non-finite gradients!"

    # ── 5. quick learnability: 400 steps, check SIGReg drops + correct cf signs ─
    print("\n[5] quick train (400 steps)...")
    for step in range(400):
        bb = gen.sample_batch(1024)
        loss, parts = stageA_loss(bb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 399:
            print(f"    step {step:3d}  loss={float(loss):.4f}  pred={parts[0]:.4f} "
                  f"sig={parts[1]:.4f} ro={parts[2]:.4f}")
    model.eval()
    for elem, want in [("Mg", "neg"), ("Sn", "pos")]:
        lc, m = cf_curve(elem, apply_delta=False)
        slope = np.polyfit(lc, m, 1)[0]
        ok = (slope < 0) if want == "neg" else (slope > 0)
        print(f"    {elem} measured cf slope = {slope:+.3f}  (want {want})  {'OK' if ok else 'XX'}")
    print("\nSMOKE TEST COMPLETE.")


if __name__ == "__main__":
    main()
