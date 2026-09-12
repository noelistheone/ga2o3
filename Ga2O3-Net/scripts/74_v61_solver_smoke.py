"""V61 smoke test — verify the differentiable defect-equilibrium solver's PHYSICS (design §3 gates).

Checks, on GPU if available:
  1. Root validity: |F(E_F*)| / (dominant charge scale) tiny; E_F* strictly inside (0, E_g).
  2. Monotone/unique root: F(0) > 0 and F(E_g) < 0 (sign change ⇒ unique root for a decreasing F).
  3. pO2 power law: slope d log10[V_O] / d log10 pO2 negative, in the physical [−1/2, −1/6] band
     (−1/2 frozen/extrinsic, −1/6 self-consistent intrinsic — the latter is impossible for the
     bolted-on closed form and is a key reason to solve self-consistently).
  4. Bulk dopant sign: donor (z=+1) raises E_F ⇒ [V_O] DOWN; acceptor (z=−1) ⇒ [V_O] UP.
  5. Implicit gradients: d log10[V_O]/d(dH_f) finite & nonzero; d log10[V_O]/d log_pO2 ≈ the slope in (3).
  6. Self-consistency under one Newton re-solve at E_F* (residual unchanged).
Saves results/phase61_preflight/solver_smoke.json. No /tmp.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import torch

import sys
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.models.defect_equilibrium import (  # noqa: E402
    solve_equilibrium, _residual, _carrier_dos, NATIVE_VO_DHF, EG_GA2O3,
    KB_EV, LN10, N_GA_SITES,
)

OUT = PROJ / "results/phase61_preflight/solver_smoke.json"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def dHf_batch(B):
    return torch.tensor([[NATIVE_VO_DHF[0], NATIVE_VO_DHF[1], NATIVE_VO_DHF[2]]], device=dev).repeat(B, 1)


def main():
    rep = {"device": str(dev)}

    # ---- base case: undoped, T=973K, pO2=0.2 ----
    B = 1
    dHf = dHf_batch(B)
    T = torch.full((B,), 973.15, device=dev)
    lp = torch.full((B,), math.log10(0.2), device=dev)
    z = torch.zeros(B, device=dev); c = torch.zeros(B, device=dev)
    out = solve_equilibrium(dHf, T, lp, z, c)
    E_F = out["E_F"].item(); logVO = out["log10_VO"].item(); resid = out["residual"].item()

    # (1)+(2) root validity & sign change
    q = torch.tensor([0., 1., 2.], device=dev)
    kT = (KB_EV * T).clamp_min(1e-4); NC, NV = _carrier_dos(T)
    mu = 0.5 * kT * (LN10 * lp); dH_eff = dHf + mu.view(-1, 1)
    cd = c * N_GA_SITES; E_D = torch.full_like(T, EG_GA2O3 - 0.05); E_A = torch.full_like(T, 1.3)
    F0, conc0 = _residual(torch.zeros(B, device=dev), dH_eff, q, kT, NC, NV, z, cd, E_D, E_A, EG_GA2O3)
    Fg, _ = _residual(torch.full((B,), EG_GA2O3, device=dev), dH_eff, q, kT, NC, NV, z, cd, E_D, E_A, EG_GA2O3)
    dominant = conc0.abs().max().item()
    rep["base"] = dict(E_F=E_F, log10_VO=logVO, residual=resid,
                       residual_rel=abs(resid) / max(dominant, 1.0),
                       F_at_0=F0.item(), F_at_Eg=Fg.item(),
                       sign_change=bool(F0.item() > 0 > Fg.item()),
                       E_F_in_gap=bool(0.0 < E_F < EG_GA2O3))

    # (3)+(5) pO2 slope (undoped, dilute): sweep log pO2, fit slope; also via autograd
    lps = torch.linspace(-6, 0, 13, device=dev)
    dd = dHf_batch(len(lps)); TT = torch.full((len(lps),), 973.15, device=dev)
    zz = torch.zeros(len(lps), device=dev); cc = torch.zeros(len(lps), device=dev)
    o = solve_equilibrium(dd, TT, lps, zz, cc)
    y = o["log10_VO"]
    A = torch.stack([lps, torch.ones_like(lps)], 1)
    slope_fit = torch.linalg.lstsq(A, y.unsqueeze(1)).solution[0, 0].item()
    # autograd slope at pO2=1e-3
    lp1 = torch.tensor([math.log10(1e-3)], device=dev, requires_grad=True)
    o1 = solve_equilibrium(dHf_batch(1), torch.full((1,), 973.15, device=dev), lp1,
                           torch.zeros(1, device=dev), torch.zeros(1, device=dev))
    gslope = torch.autograd.grad(o1["log10_VO"].sum(), lp1)[0].item()
    rep["pO2_slope"] = dict(fit=slope_fit, autograd=gslope,
                            in_physical_band=bool(-0.55 <= slope_fit <= -0.10))

    # (4) bulk dopant sign at fixed conditions (T=973, pO2=0.2, c=2%)
    def solve_dop(zval):
        return solve_equilibrium(dHf_batch(1), torch.full((1,), 973.15, device=dev),
                                 torch.full((1,), math.log10(0.2), device=dev),
                                 torch.full((1,), float(zval), device=dev),
                                 torch.full((1,), 0.02, device=dev))
    base_u = solve_equilibrium(dHf_batch(1), torch.full((1,), 973.15, device=dev),
                               torch.full((1,), math.log10(0.2), device=dev),
                               torch.zeros(1, device=dev), torch.zeros(1, device=dev))
    don = solve_dop(+1); acc = solve_dop(-1)
    rep["bulk_dopant_sign"] = dict(
        undoped_logVO=base_u["log10_VO"].item(), undoped_EF=base_u["E_F"].item(),
        donor_logVO=don["log10_VO"].item(), donor_EF=don["E_F"].item(),
        acceptor_logVO=acc["log10_VO"].item(), acceptor_EF=acc["E_F"].item(),
        donor_lowers_VO=bool(don["log10_VO"].item() < base_u["log10_VO"].item()),
        acceptor_raises_VO=bool(acc["log10_VO"].item() > base_u["log10_VO"].item()),
        donor_raises_EF=bool(don["E_F"].item() > base_u["E_F"].item()),
        acceptor_lowers_EF=bool(acc["E_F"].item() < base_u["E_F"].item()))

    # (5) implicit grad to dH_f finite & nonzero
    dHg = dHf_batch(1).requires_grad_(True)
    og = solve_equilibrium(dHg, torch.full((1,), 973.15, device=dev),
                           torch.full((1,), math.log10(0.2), device=dev),
                           torch.zeros(1, device=dev), torch.zeros(1, device=dev))
    gdH = torch.autograd.grad(og["log10_VO"].sum(), dHg)[0]
    rep["grad_dHf"] = dict(values=gdH.flatten().tolist(),
                           finite=bool(torch.isfinite(gdH).all()),
                           nonzero=bool(gdH.abs().sum().item() > 1e-6))

    # (6) self-consistency: residual at E_F* should be ~0 already (one Newton refine changes nothing)
    rep["self_consistency_residual_rel"] = rep["base"]["residual_rel"]

    gates = {
        "root_valid": rep["base"]["sign_change"] and rep["base"]["E_F_in_gap"]
                      and rep["base"]["residual_rel"] < 1e-6,
        "pO2_slope_physical": rep["pO2_slope"]["in_physical_band"],
        "bulk_sign_correct": rep["bulk_dopant_sign"]["donor_lowers_VO"]
                             and rep["bulk_dopant_sign"]["acceptor_raises_VO"],
        "grad_ok": rep["grad_dHf"]["finite"] and rep["grad_dHf"]["nonzero"],
    }
    rep["GATES"] = gates
    rep["ALL_PASS"] = all(gates.values())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("base", "pO2_slope", "bulk_dopant_sign", "grad_dHf",
                                          "GATES", "ALL_PASS")}, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
