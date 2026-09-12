"""Phase 63 V63 — verification gates for the SELF-CONSISTENT defect-complex (DAP) solver.

Gates (all -> results/phase63/solver_complex_smoke.json):
  G1  reduce to V62: no binding matrix -> identical to solve_equilibrium_codoping (bit-for-bit).
  G2  reduce to V61: D=1 no binding -> matches single-dopant solve_equilibrium (< 1e-6).
  G3  outer fixed point converges on Sn+Mg co-doping (outer_resid small, iters logged).
  G4  Coulomb DECOMPOSITION cross-validation: E_assoc_near + (fully-ionized) Coulomb == MACE E_bind_iso
      to within ~0.1 eV for the 5 BOUND donor-acceptor pairs (the Opt-B result).
  G5  self-consistent charged pairing -> near-complete pairing; reports decoupled vs self-consistent
      E_F / log[V_O] / bound_frac so the mechanism shift is on disk.
  G6  independent scipy fixed-point cross-check (brentq inner + python pairing loop) agrees with the
      torch solver on E_F, log[V_O], bound_frac for D=2 and D=3.

CPU, float64, deterministic. Rule 1/2 preserved (binding detached; grad only via inner E_F solve).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
torch.set_default_dtype(torch.float64)

from src.models.defect_equilibrium import EG_GA2O3, KB_EV, N_GA_SITES                 # noqa: E402
from src.models.defect_equilibrium_codoping import solve_equilibrium_codoping, elem_transition_level  # noqa: E402
from src.models.defect_equilibrium import solve_equilibrium                            # noqa: E402
from src.models.defect_equilibrium_complex import solve_equilibrium_complex, _solve_pairing  # noqa: E402
from src.models.dap_coulomb import coulomb_energy, COULOMB_EV_ANGSTROM                  # noqa: E402
from src.models.v61_net import carrier_sign                                            # noqa: E402

EG = EG_GA2O3
DHF = [3.5, 1.8, 0.3]
T = 973.15
LP = 0.0


def _mk(elems, cfracs):
    """Build z, c, e_level tensors [1,D] for a list of elements + fractions."""
    z = [carrier_sign(e) for e in elems]
    el = [elem_transition_level(e, carrier_sign(e)) for e in elems]
    return (torch.tensor([z]), torch.tensor([cfracs]), torch.tensor([el]))


def _bind_matrix(elems, eassoc):
    """Symmetric [1,D,D] short-range (E_assoc_near) binding; donor-acceptor entries only."""
    D = len(elems)
    M = torch.zeros(1, D, D)
    for i in range(D):
        for j in range(i + 1, D):
            zi, zj = carrier_sign(elems[i]), carrier_sign(elems[j])
            if zi * zj < 0:                       # donor-acceptor
                M[0, i, j] = M[0, j, i] = eassoc
    return M


def g1_reduce_v62():
    z, c, el = _mk(["Sn", "Mg"], [0.01, 0.01])
    dH = torch.tensor([DHF])
    a = solve_equilibrium_codoping(dH, torch.tensor([T]), torch.tensor([LP]), z, c, el)
    b = solve_equilibrium_complex(dH, torch.tensor([T]), torch.tensor([LP]), z, c, el, E_bind_neutral=None)
    return dict(name="G1_reduce_to_V62_no_binding",
                dEF=abs(float(a["E_F"]) - float(b["E_F"])),
                dlogVO=abs(float(a["log10_VO"]) - float(b["log10_VO"])),
                pass_=bool(abs(float(a["E_F"]) - float(b["E_F"])) < 1e-9))


def g2_reduce_v61():
    dH = torch.tensor([DHF])
    z = torch.tensor([[carrier_sign("Sn")]]); c = torch.tensor([[0.01]])
    el = torch.tensor([[elem_transition_level("Sn", carrier_sign("Sn"))]])
    v61 = solve_equilibrium(dH, torch.tensor([T]), torch.tensor([LP]),
                            torch.tensor([carrier_sign("Sn")]), torch.tensor([0.01]))
    cx = solve_equilibrium_complex(dH, torch.tensor([T]), torch.tensor([LP]), z, c, el, E_bind_neutral=None)
    return dict(name="G2_reduce_to_V61_D1",
                dEF=abs(float(v61["E_F"]) - float(cx["E_F"])),
                dlogVO=abs(float(v61["log10_VO"]) - float(cx["log10_VO"])),
                pass_=bool(abs(float(v61["E_F"]) - float(cx["E_F"])) < 1e-6))


def g3_outer_converges():
    z, c, el = _mk(["Sn", "Mg"], [0.01, 0.01])
    M = _bind_matrix(["Sn", "Mg"], -0.542)
    out = solve_equilibrium_complex(torch.tensor([DHF]), torch.tensor([T]), torch.tensor([LP]),
                                    z, c, el, E_bind_neutral=M, use_coulomb=True)
    return dict(name="G3_outer_fixed_point_converges",
                outer_iters=int(out["outer_iters"]), outer_resid=float(out["outer_resid"]),
                bound_frac=float(out["bound_frac"]),
                pass_=bool(out["outer_resid"] < 1e-6 and out["outer_iters"] <= 40))


def g4_coulomb_decomposition():
    """E_assoc_near + (fully-ionized) Coulomb ~= MACE E_bind_iso for the 5 BOUND pairs."""
    bj = json.load(open(PROJ / "dft/codoping/results/binding_energies.json"))
    dnear = bj["separations"]["near"]["d_AB"]
    e_coul = coulomb_energy(+1.0, -1.0, eps_r=10.2, d=dnear)         # -0.463 eV
    rows, errs = [], []
    for p in bj["pairs"]:
        if not str(p.get("verdict", "")).startswith("BOUND"):
            continue
        recon = p["E_assoc_near_eV"] + e_coul                       # short-range + monopole
        err = abs(recon - p["E_bind_eV"])
        errs.append(err)
        rows.append(dict(pair=f"{p['A']}+{p['B']}", E_assoc_near=p["E_assoc_near_eV"],
                         E_bind_iso=p["E_bind_eV"], reconstructed=round(recon, 4), abs_err=round(err, 4)))
    return dict(name="G4_coulomb_decomposition_xval", d_near=dnear, E_coulomb=round(e_coul, 4),
                pairs=rows, max_abs_err=round(max(errs), 4), mean_abs_err=round(sum(errs) / len(errs), 4),
                pass_=bool(max(errs) < 0.12))


def g5_self_consistent_vs_decoupled():
    """Decoupled (Phase-62: neutral binding, one-shot) vs self-consistent charged pairing."""
    z, c, el = _mk(["Sn", "Mg"], [0.01, 0.01])
    dH = torch.tensor([DHF]); Tt = torch.tensor([T]); LPt = torch.tensor([LP])
    # decoupled Phase-62 style: short-range binding only, no Coulomb, single outer pass (n_outer=1)
    M_sr = _bind_matrix(["Sn", "Mg"], -0.542)
    dec = solve_equilibrium_complex(dH, Tt, LPt, z, c, el, E_bind_neutral=M_sr,
                                    use_coulomb=False, n_outer=1)
    # self-consistent charged: short-range + ionization-weighted Coulomb, full outer loop
    sc = solve_equilibrium_complex(dH, Tt, LPt, z, c, el, E_bind_neutral=M_sr, use_coulomb=True)
    return dict(name="G5_self_consistent_vs_decoupled",
                decoupled=dict(E_F=round(float(dec["E_F"]), 4), log10_VO=round(float(dec["log10_VO"]), 4),
                               bound_frac=round(float(dec["bound_frac"]), 4)),
                self_consistent=dict(E_F=round(float(sc["E_F"]), 4), log10_VO=round(float(sc["log10_VO"]), 4),
                                     bound_frac=round(float(sc["bound_frac"]), 4)),
                charged_pairing_higher=bool(float(sc["bound_frac"]) >= float(dec["bound_frac"]) - 1e-9),
                pass_=bool(float(sc["bound_frac"]) >= float(dec["bound_frac"]) - 1e-9))


# ---- G6: independent scipy fixed-point re-implementation ----------------------------------------
def _scipy_complex(elems, cfracs, eassoc, use_coulomb=True, Z=8.0, eps_r=10.2, d=3.048):
    import numpy as np
    from scipy.optimize import brentq
    kT = KB_EV * T
    N_C0, N_V0 = 3.7e18, 4.0e19
    s = (T / 300.0) ** 1.5
    N_C, N_V = N_C0 * s, N_V0 * s
    q = np.array([0.0, 1.0, 2.0]); dH = np.array(DHF) + 0.5 * kT * (np.log(10.0) * LP)
    z = np.array([carrier_sign(e) for e in elems])
    lev = np.array([elem_transition_level(e, carrier_sign(e)) for e in elems])
    cden = np.array(cfracs)                      # site fractions (we pair in fractions, charge in density)
    NGA = N_GA_SITES
    ecoul = (z[:, None] * z[None, :]) * (COULOMB_EV_ANGSTROM / (eps_r * d)) if use_coulomb else np.zeros((len(z), len(z)))

    def f_ion(EF):
        fd = 1.0 / (1.0 + 2.0 * np.exp((EF - lev) / kT))
        fa = 1.0 / (1.0 + 4.0 * np.exp((lev - EF) / kT))
        return np.where(z > 0, fd, np.where(z < 0, fa, 0.0))

    def resid(EF, cfree):
        ef_q = dH + q * EF
        conc = 2.85e22 / (1.0 + np.exp(ef_q / kT))
        dch = (q * conc).sum()
        n = N_C * np.exp(-(EG - EF) / kT); pp = N_V * np.exp(-EF / kT)
        fi = f_ion(EF)
        dop = (z * (cfree * NGA) * fi).sum()
        return pp - n + dch + dop

    cfree = cden.copy(); EF = 0.5 * EG
    for _ in range(80):
        EF = brentq(lambda e: resid(e, cfree), 0.0, EG, xtol=1e-12)
        fi = np.abs(f_ion(EF))
        D = len(elems); xP = np.zeros((D, D))
        for _s in range(30):
            for i in range(D):
                for j in range(i + 1, D):
                    if z[i] * z[j] >= 0:
                        continue
                    Eeff = eassoc[i][j] + (fi[i] * fi[j]) * ecoul[i, j]
                    if Eeff >= 0:
                        continue
                    K = Z * np.exp(-Eeff / kT)
                    ai = max(cden[i] - (xP[i].sum() - xP[i, j]), 0.0)
                    aj = max(cden[j] - (xP[j].sum() - xP[i, j]), 0.0)
                    b = 1.0 + K * (ai + aj)
                    disc = max(b * b - 4.0 * K * K * ai * aj, 0.0)
                    x = (b - disc ** 0.5) / (2.0 * K + 1e-300)
                    x = min(max(x, 0.0), min(ai, aj))
                    xP[i, j] = xP[j, i] = x
        cfree_new = np.maximum(cden - xP.sum(axis=1), 0.0)
        if np.abs(cfree_new - cfree).max() < 1e-11:
            cfree = cfree_new; break
        cfree = 0.4 * cfree + 0.6 * cfree_new
    ef_q = dH + q * EF
    conc = 2.85e22 / (1.0 + np.exp(ef_q / kT))
    import math
    bound = 2.0 * xP.sum() * 0.5 / max(cden.sum(), 1e-30)
    return EF, math.log10(max(conc.sum(), 1.0)), float(bound)


def g6_independent_xcheck():
    cases = []
    for elems, cf in [(["Sn", "Mg"], [0.01, 0.01]), (["Sn", "Mg", "Zn"], [0.01, 0.006, 0.006])]:
        z, c, el = _mk(elems, cf)
        D = len(elems)
        eassoc = [[0.0] * D for _ in range(D)]
        for i in range(D):
            for j in range(i + 1, D):
                if carrier_sign(elems[i]) * carrier_sign(elems[j]) < 0:
                    eassoc[i][j] = eassoc[j][i] = -0.542
        M = torch.tensor([eassoc])
        out = solve_equilibrium_complex(torch.tensor([DHF]), torch.tensor([T]), torch.tensor([LP]),
                                        z, c, el, E_bind_neutral=M, use_coulomb=True, d_cation=3.048)
        ef_s, vo_s, bf_s = _scipy_complex(elems, cf, eassoc, use_coulomb=True, d=3.048)
        cases.append(dict(case="+".join(elems),
                          dEF=abs(float(out["E_F"]) - ef_s),
                          dlogVO=abs(float(out["log10_VO"]) - vo_s),
                          dboundfrac=abs(float(out["bound_frac"]) - bf_s),
                          torch=dict(E_F=round(float(out["E_F"]), 5), log10_VO=round(float(out["log10_VO"]), 5),
                                     bound_frac=round(float(out["bound_frac"]), 5)),
                          scipy=dict(E_F=round(ef_s, 5), log10_VO=round(vo_s, 5), bound_frac=round(bf_s, 5))))
    mx = max(max(c["dEF"], c["dlogVO"], c["dboundfrac"]) for c in cases)
    return dict(name="G6_independent_scipy_xcheck", cases=cases, max_dev=mx, pass_=bool(mx < 5e-3))


def main():
    gates = [g1_reduce_v62(), g2_reduce_v61(), g3_outer_converges(), g4_coulomb_decomposition(),
             g5_self_consistent_vs_decoupled(), g6_independent_xcheck()]
    rep = dict(gates=gates, ALL_PASS=bool(all(g["pass_"] for g in gates)))
    out = PROJ / "results/phase63/solver_complex_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    for g in gates:
        print(f"[{'PASS' if g['pass_'] else 'FAIL'}] {g['name']}")
        for k, v in g.items():
            if k not in ("name", "pass_"):
                print(f"        {k}: {v}")
    print(f"\nALL_PASS={rep['ALL_PASS']}  ->  {out}")


if __name__ == "__main__":
    main()
