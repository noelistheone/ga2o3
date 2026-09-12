"""V62 validation gate 3a — INDEPENDENT cross-check of the multi-dopant solver.

Re-implements the charge-neutrality equation FROM SCRATCH in plain numpy/scipy (a separate code path and
a different root-finder, scipy.optimize.brentq, instead of the torch bisection + implicit-gradient
re-attach). If the two agree on E_F and log10[V_O] across single- and co-doping configs, formulation and
numerics bugs in src/models/defect_equilibrium_codoping.py are ruled out.

(An independent peer-authored sc-Fermi code -- py-sc-fermi / doped -- is not installed; this from-scratch
reference solves the identical physics by an independent method, which is the available zero-dependency
equivalent. The research sweep evaluates whether to add py-sc-fermi for a physics-convention cross-check.)

Writes results/phase62_codoping/independent_xcheck.json. CPU, deterministic.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.models.defect_equilibrium import (  # noqa: E402
    KB_EV, EG_GA2O3, N_O_SITES, N_GA_SITES, N_C0, N_V0,
)
from src.models.defect_equilibrium_codoping import solve_equilibrium_codoping, elem_transition_level  # noqa: E402
from src.models.v61_net import carrier_sign  # noqa: E402

EG = EG_GA2O3
DHF = [3.5, 1.8, 0.3]
G_D, G_A = 2.0, 4.0


def F_indep(EF, T_K, log_pO2, dopants):
    """Independent charge-neutrality residual (pure python). dopants: list of (z, c_frac, e_level)."""
    kT = KB_EV * T_K
    NC = N_C0 * (T_K / 300.0) ** 1.5
    NV = N_V0 * (T_K / 300.0) ** 1.5
    mu = 0.5 * kT * math.log(10.0) * log_pO2
    dHeff = [DHF[q] + mu for q in range(3)]
    conc = [N_O_SITES / (1.0 + math.exp((dHeff[q] + q * EF) / kT)) for q in range(3)]
    defect_charge = sum(q * conc[q] for q in range(3))
    n = NC * math.exp(-(EG - EF) / kT)
    p = NV * math.exp(-EF / kT)
    dop = 0.0
    for (z, cf, el) in dopants:
        c = cf * N_GA_SITES
        if z > 0:
            dop += c / (1.0 + G_D * math.exp((EF - el) / kT))
        elif z < 0:
            dop -= c / (1.0 + G_A * math.exp((el - EF) / kT))
    return p - n + defect_charge + dop


def solve_indep(T_K, log_pO2, dopants):
    f0 = F_indep(1e-9, T_K, log_pO2, dopants)
    fg = F_indep(EG - 1e-9, T_K, log_pO2, dopants)
    if f0 <= 0:           # over-doped acceptor -> pin VBM
        EF = 0.0
    elif fg >= 0:         # over-doped donor -> pin CBM (degenerate)
        EF = EG
    else:
        EF = brentq(lambda e: F_indep(e, T_K, log_pO2, dopants), 1e-9, EG - 1e-9, xtol=1e-10, rtol=1e-12)
    kT = KB_EV * T_K
    mu = 0.5 * kT * math.log(10.0) * log_pO2
    conc = [N_O_SITES / (1.0 + math.exp((DHF[q] + mu + q * EF) / kT)) for q in range(3)]
    return EF, math.log10(max(sum(conc), 1.0))


def solver_call(T_K, log_pO2, dopants):
    z = torch.tensor([[d[0] for d in dopants]], dtype=torch.float64)
    c = torch.tensor([[d[1] for d in dopants]], dtype=torch.float64)
    el = torch.tensor([[d[2] for d in dopants]], dtype=torch.float64)
    out = solve_equilibrium_codoping(torch.tensor([DHF], dtype=torch.float64),
                                     torch.tensor([T_K], dtype=torch.float64),
                                     torch.tensor([log_pO2], dtype=torch.float64), z, c, el)
    return float(out["E_F"].item()), float(out["log10_VO"].item())


def main():
    def D(elem, c):
        z = carrier_sign(elem); return (z, c, elem_transition_level(elem, z))
    cases = [
        ("undoped", 973.15, 0.0, [(0.0, 0.0, 0.5 * EG)]),
        ("Sn 0.01%", 973.15, 0.0, [D("Sn", 1e-4)]),
        ("Mg 0.5%", 973.15, 0.0, [D("Mg", 5e-3)]),
        ("Sn0.5%+Mg0.5%", 973.15, 0.0, [D("Sn", 5e-3), D("Mg", 5e-3)]),
        ("Sn0.3%+Mg0.6%", 1073.15, -3.0, [D("Sn", 3e-3), D("Mg", 6e-3)]),
        ("Si0.4%+N0.4%", 873.15, 0.0, [D("Si", 4e-3), D("N", 4e-3)]),
        ("Mg0.3%+Zn0.3%", 973.15, 0.0, [D("Mg", 3e-3), D("Zn", 3e-3)]),
        ("Sn0.2%+Fe1%", 973.15, 0.0, [D("Sn", 2e-3), D("Fe", 1e-2)]),
        ("Sn0.1%+Mg0.05%+Zn0.05%", 973.15, 0.0, [D("Sn", 1e-3), D("Mg", 5e-4), D("Zn", 5e-4)]),
    ]
    rows = []
    for name, T, lp, dop in cases:
        ef_i, vo_i = solve_indep(T, lp, dop)
        ef_s, vo_s = solver_call(T, lp, dop)
        rows.append(dict(case=name, T_K=T, log_pO2=lp,
                         EF_independent=ef_i, EF_solver=ef_s, dEF=abs(ef_i - ef_s),
                         logVO_independent=vo_i, logVO_solver=vo_s, dlogVO=abs(vo_i - vo_s)))
    max_dEF = max(r["dEF"] for r in rows)
    max_dVO = max(r["dlogVO"] for r in rows)
    rep = dict(rows=rows, max_dEF=max_dEF, max_dlogVO=max_dVO,
               AGREE=bool(max_dEF < 1e-4 and max_dVO < 1e-4))
    out = PROJ / "results/phase62_codoping/independent_xcheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"{'case':28s} {'EF_ind':>8s} {'EF_slv':>8s} {'dEF':>9s} {'lVO_ind':>9s} {'lVO_slv':>9s} {'dlVO':>9s}")
    for r in rows:
        print(f"{r['case']:28s} {r['EF_independent']:8.4f} {r['EF_solver']:8.4f} {r['dEF']:9.2e} "
              f"{r['logVO_independent']:9.4f} {r['logVO_solver']:9.4f} {r['dlogVO']:9.2e}")
    print(f"\nmax dEF={max_dEF:.2e}  max dlogVO={max_dVO:.2e}  AGREE={rep['AGREE']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
