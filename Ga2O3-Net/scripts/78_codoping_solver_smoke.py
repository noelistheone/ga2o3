"""V62 physics gates for the MULTI-DOPANT defect-equilibrium solver (design docs/phase62 sec 5, gates 1-2).

Verifies, by construction, that solve_equilibrium_codoping is correct co-doping physics:
  G1 monotonicity   - F(E_F) strictly decreasing over the gap for a donor+acceptor co-doping config
  G2 neutrality     - charge-balance residual ~ 0 at the solved E_F (undoped / donor / acceptor / co-doped)
  G3 D=1 == V61     - single-dopant reduction reproduces the V61 solver to machine precision
  G4 compensation   - adding acceptor at fixed donor pulls E_F DOWN monotonically (Fermi tug-of-war)
                      and V_O^2+ RISES (self-compensation) -- the textbook frustrated-p-type mechanism
  G5 exact gradient - autograd d log10[V_O]/d(dH_f) matches central finite difference
  G6 pO2 slope      - d log10[V_O] / d log10(pO2) ~ -0.5 (frozen-E_F Brouwer term emerges from mu_O)

Writes results/phase62_codoping/solver_smoke.json. CPU, deterministic.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.models.defect_equilibrium import solve_equilibrium, EG_GA2O3, KB_EV, LN10, _carrier_dos, N_GA_SITES  # noqa: E402
from src.models.defect_equilibrium_codoping import (  # noqa: E402
    solve_equilibrium_codoping, _residual_codoping, elem_transition_level,
)

torch.set_default_dtype(torch.float64)
EG = EG_GA2O3


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


def gate_monotonic():
    """G1: F(E_F) strictly decreasing for Sn(1%) donor + Mg(1%) acceptor co-doping at 973 K, 1 atm."""
    T_K = _t([973.15]); log_pO2 = _t([0.0])
    dH = _t([[3.5, 1.8, 0.3]])
    q_vals = _t([0.0, 1.0, 2.0])
    kT = (KB_EV * T_K).clamp_min(1e-4)
    N_C, N_V = _carrier_dos(T_K)
    z = _t([[+1.0, -1.0]]); cfrac = _t([[0.01, 0.01]])
    c_dop = cfrac * N_GA_SITES
    elev = _t([[EG - 0.03, 1.3]])
    mask = _t([[1.0, 1.0]])
    mu = 0.5 * kT * (LN10 * log_pO2)
    dH_eff = dH + mu.view(-1, 1)
    grid = torch.linspace(0.0, EG, 200)
    Fs = []
    for ef in grid:
        F, _, _ = _residual_codoping(ef.view(1), dH_eff, q_vals, kT, N_C, N_V, z, c_dop, elev, mask, EG)
        Fs.append(float(F.item()))
    diffs = [Fs[i + 1] - Fs[i] for i in range(len(Fs) - 1)]
    strictly_decreasing = all(d < 0 for d in diffs)
    return dict(strictly_decreasing=bool(strictly_decreasing),
                max_diff=max(diffs), F_at_0=Fs[0], F_at_Eg=Fs[-1],
                sign_change=bool(Fs[0] > 0 > Fs[-1]))


def gate_neutrality():
    """G2: charge-balance residual ~ 0 at the solved E_F.

    Non-degenerate (dilute) regime MUST be machine-tiny. Heavy single-donor configs intentionally PIN at
    E_F=E_g (donor density >> non-degenerate N_C -> degenerate; Boltzmann statistics saturate, same
    documented V61 limitation) -- those are reported separately and do NOT fail the gate. The co-doping
    physics regime (compensated -> interior root) is the dilute one and is exactly where accuracy matters.
    """
    # --- dilute / compensated regime (must be tiny) ---
    T_K = _t([973.15] * 4); log_pO2 = _t([0.0] * 4); dH = _t([[3.5, 1.8, 0.3]] * 4)
    z = _t([[0.0, 0.0], [+1.0, 0.0], [-1.0, 0.0], [+1.0, -1.0]])
    cfrac = _t([[0.0, 0.0], [1e-4, 0.0], [1e-4, 0.0], [5e-3, 5e-3]])  # dilute; co-doped balanced
    elev = _t([[0.5 * EG, 0.5 * EG], [EG - 0.03, 0.5 * EG], [1.3, 0.5 * EG], [EG - 0.03, 1.3]])
    mask = _t([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    out = solve_equilibrium_codoping(dH, T_K, log_pO2, z, cfrac, elev, dop_mask=mask)
    resid = out["residual"].abs()
    scale = (cfrac * N_GA_SITES).clamp_min(1.0).max(dim=1).values
    rel = (resid / scale).tolist()
    pinned = [bool(abs(ef - EG) < 1e-6 or abs(ef) < 1e-6) for ef in out["E_F"].tolist()]
    # --- heavy single-donor (expected pin / degenerate, reported only) ---
    heavy = solve_equilibrium_codoping(_t([[3.5, 1.8, 0.3]]), _t([973.15]), _t([0.0]),
                                       _t([[+1.0]]), _t([[0.01]]), _t([[EG - 0.03]]))
    return dict(dilute_configs=["undoped", "Sn0.01%", "Mg0.01%", "Sn0.5%+Mg0.5%"],
                residual_abs=resid.tolist(), rel_residual=rel, E_F=out["E_F"].tolist(),
                pinned=pinned,
                all_tiny=bool(max(rel) < 1e-3),
                heavy_Sn1pct_E_F=heavy["E_F"].item(),
                heavy_Sn1pct_degenerate_pin=bool(abs(heavy["E_F"].item() - EG) < 1e-6))


def gate_reduction():
    """G3: D=1 co-doping solver == V61 single-dopant solver (donor and acceptor), default class levels."""
    T_K = _t([700.0, 1100.0]); log_pO2 = _t([-3.0, 0.0]); dH = _t([[3.5, 1.8, 0.3]] * 2)
    out = {}
    for name, sign, cfr, ebind_d, ebind_a, lvl in [
        ("donor", +1.0, 0.02, 0.05, 1.3, EG - 0.05),
        ("acceptor", -1.0, 0.02, 0.05, 1.3, 1.3),
    ]:
        v61 = solve_equilibrium(dH, T_K, log_pO2, _t([sign, sign]), _t([cfr, cfr]),
                                e_bind_donor=ebind_d, e_bind_acceptor=ebind_a)
        v62 = solve_equilibrium_codoping(dH, T_K, log_pO2, _t([[sign], [sign]]),
                                         _t([[cfr], [cfr]]), _t([[lvl], [lvl]]))
        d_vo = (v61["log10_VO"] - v62["log10_VO"]).abs().max().item()
        d_ef = (v61["E_F"] - v62["E_F"]).abs().max().item()
        out[name] = dict(max_dlogVO=d_vo, max_dEF=d_ef, match=bool(d_vo < 1e-6 and d_ef < 1e-6))
    out["all_match"] = bool(all(v["match"] for k, v in out.items() if isinstance(v, dict)))
    return out


def gate_compensation():
    """G4: fixed Sn donor 1%, sweep Mg acceptor 0..2% -> E_F monotonically DOWN, [V_O^2+] monotonically UP."""
    n = 11
    mg = torch.linspace(0.0, 0.02, n)
    T_K = _t([973.15] * n); log_pO2 = _t([0.0] * n); dH = _t([[3.5, 1.8, 0.3]] * n)
    z = _t([[+1.0, -1.0]] * n)
    cfrac = torch.stack([torch.full((n,), 0.01), mg], dim=1)
    # mask Mg off where conc==0 so the isovalent/zero slot is harmless
    mask = torch.stack([torch.ones(n), (mg > 0).double()], dim=1)
    elev = _t([[EG - 0.03, 1.3]] * n)
    out = solve_equilibrium_codoping(dH, T_K, log_pO2, z, cfrac, elev, dop_mask=mask)
    EF = out["E_F"].tolist()
    vo2 = out["conc"][:, 2].clamp_min(1.0).log10().tolist()   # log10 [V_O^2+]
    logvo = out["log10_VO"].tolist()
    ef_down = all(EF[i + 1] <= EF[i] + 1e-9 for i in range(n - 1))
    vo2_up = all(vo2[i + 1] >= vo2[i] - 1e-9 for i in range(n - 1))
    return dict(mg_fraction=mg.tolist(), E_F=EF, log10_VO2plus=vo2, log10_VO_total=logvo,
                E_F_monotone_down=bool(ef_down), VO2plus_monotone_up=bool(vo2_up),
                E_F_drop_eV=float(EF[0] - EF[-1]))


def gate_gradient():
    """G5: autograd d log10[V_O]/d(dH_f[q=0]) vs central finite difference for a compensated co-doping config.

    Gradient is taken w.r.t. the NEUTRAL-V_O enthalpy dH_f[0], which always contributes to the V_O total
    (E_F-independent occupancy) -> a non-vacuous test of the implicit-function-theorem path. Uses a dilute,
    compensated config (interior E_F) so the Newton re-attach (not the boundary pin) is exercised.
    """
    base = [3.5, 1.8, 0.3]; T_K = _t([973.15]); log_pO2 = _t([0.0])
    z = _t([[+1.0, -1.0]]); cfrac = _t([[5e-3, 5e-3]]); elev = _t([[EG - 0.03, 1.3]]); mask = _t([[1.0, 1.0]])
    dH = _t([base]).clone().requires_grad_(True)
    out = solve_equilibrium_codoping(dH, T_K, log_pO2, z, cfrac, elev, dop_mask=mask)
    out["log10_VO"].sum().backward()
    g_auto = float(dH.grad[0, 0].item())
    eps = 1e-4
    dHp = _t([[base[0] + eps, base[1], base[2]]]); dHm = _t([[base[0] - eps, base[1], base[2]]])
    yp = solve_equilibrium_codoping(dHp, T_K, log_pO2, z, cfrac, elev, dop_mask=mask)["log10_VO"].item()
    ym = solve_equilibrium_codoping(dHm, T_K, log_pO2, z, cfrac, elev, dop_mask=mask)["log10_VO"].item()
    g_fd = (yp - ym) / (2 * eps)
    rel = abs(g_auto - g_fd) / (abs(g_fd) + 1e-12)
    return dict(grad_autograd=g_auto, grad_finite_diff=g_fd, rel_err=rel,
                match=bool(rel < 1e-3 and abs(g_fd) > 1e-9))


def gate_po2_slope():
    """G6: d log10[V_O]/d log10(pO2) ~ -0.5 (frozen Brouwer term from mu_O), for a lightly-doped cell."""
    po2 = torch.linspace(-6.0, 0.0, 7)
    n = po2.numel()
    T_K = _t([973.15] * n); dH = _t([[3.5, 1.8, 0.3]] * n)
    z = _t([[+1.0, -1.0]] * n); cfrac = _t([[0.001, 0.001]] * n)
    elev = _t([[EG - 0.03, 1.3]] * n); mask = _t([[1.0, 1.0]] * n)
    out = solve_equilibrium_codoping(dH, T_K, po2, z, cfrac, elev, dop_mask=mask)
    y = out["log10_VO"]
    # least-squares slope
    x = po2
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return dict(log_pO2=po2.tolist(), log10_VO=y.tolist(), slope=slope,
                near_minus_half=bool(-0.7 < slope < -0.3))


def main():
    rep = {
        "G1_monotonic": gate_monotonic(),
        "G2_neutrality": gate_neutrality(),
        "G3_reduction_eq_V61": gate_reduction(),
        "G4_compensation": gate_compensation(),
        "G5_exact_gradient": gate_gradient(),
        "G6_pO2_slope": gate_po2_slope(),
    }
    rep["ALL_PASS"] = bool(
        rep["G1_monotonic"]["strictly_decreasing"] and
        rep["G2_neutrality"]["all_tiny"] and
        rep["G3_reduction_eq_V61"]["all_match"] and
        rep["G4_compensation"]["E_F_monotone_down"] and rep["G4_compensation"]["VO2plus_monotone_up"] and
        rep["G5_exact_gradient"]["match"] and
        rep["G6_pO2_slope"]["near_minus_half"]
    )
    out = PROJ / "results/phase62_codoping/solver_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if isinstance(vv, (bool, float, int, str))})
                      for k, v in rep.items()}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
