"""Phase S0 — solver-port verification + native-V_O Brouwer sandbox demo.

Three parts, all gated (results/phase0/phase0_gates.json):

  A. PORT PARITY: the ported src/sandbox/equilibrium.py must reproduce Ga2O3-Net's
     src/models/defect_equilibrium.py bit-for-bit on identical random inputs.
  B. ANCHOR CONSISTENCY: data/energetics/native_VO_hse06_anchor.json == module default.
  C. PHYSICS SANITY on a (T, pO2) x {undoped, Sn, Si, Mg} grid with full Shomate mu_O(T,p):
       G1 undoped Brouwer slope d log10[V_O]/d log10 pO2 in [-0.55, -0.01]
       G2 donors (Sn/Si) raise n and E_F vs undoped at same conditions
       G3 acceptor (Mg) lowers E_F and raises [V_O] vs undoped (compensation)
       G4 undoped n rises with T
       G5 charge-balance residual relatively small at every unpinned grid point

Grid CSV -> results/phase0/brouwer_grid.csv.  Run:
    conda run -n ga2o3 python scripts/phase0_brouwer_demo.py --gpu 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from sandbox import equilibrium as eq  # noqa: E402
from sandbox import mu_o as muo        # noqa: E402

GA2O3NET_SOLVER = PROJ.parent / "Ga2O3-Net" / "src" / "models" / "defect_equilibrium.py"
OUT = PROJ / "results" / "phase0"


def load_original_solver():
    spec = importlib.util.spec_from_file_location("v61_orig", GA2O3NET_SOLVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v61_orig"] = mod  # @dataclass needs the module resolvable in sys.modules
    spec.loader.exec_module(mod)
    return mod


def part_a_port_parity(dev: str) -> dict:
    """Ported solver must agree with the Ga2O3-Net original to float64 round-off."""
    orig = load_original_solver()
    g = torch.Generator().manual_seed(0)
    B = 512
    dH = torch.stack([
        3.5 + 0.6 * torch.randn(B, generator=g),
        1.8 + 0.6 * torch.randn(B, generator=g),
        0.3 + 0.6 * torch.randn(B, generator=g),
    ], dim=1).to(dev)
    T = (700.0 + 1200.0 * torch.rand(B, generator=g)).to(dev)
    lp = (-20.0 + 20.0 * torch.rand(B, generator=g)).to(dev)
    z = torch.tensor([1.0, -1.0, 0.0])[torch.randint(0, 3, (B,), generator=g)].to(dev)
    c = (10.0 ** (-4.0 + 2.5 * torch.rand(B, generator=g))).to(dev)

    r_new = eq.solve_equilibrium(dH, T, lp, z, c)
    r_old = orig.solve_equilibrium(dH, T, lp, z, c)
    d_vo = (r_new["log10_VO"] - r_old["log10_VO"]).abs().max().item()
    d_ef = (r_new["E_F"] - r_old["E_F"]).abs().max().item()
    return {"max_abs_diff_log10_VO": d_vo, "max_abs_diff_E_F": d_ef,
            "pass": bool(d_vo < 1e-9 and d_ef < 1e-9)}


def part_b_anchor_consistency() -> dict:
    j = json.loads((PROJ / "data" / "energetics" / "native_VO_hse06_anchor.json").read_text())
    db = {e["charge"]: e["E_f_at_VBM_eV"] for e in j["entries"] if e["defect"] == "V_O"}
    ok = all(abs(db[q] - eq.NATIVE_VO_DHF[q]) < 1e-12 for q in (0, 1, 2))
    return {"db_values": db, "module_default": eq.NATIVE_VO_DHF, "pass": bool(ok)}


def part_c_grid(dev: str) -> tuple[dict, "np.ndarray", list[str]]:
    T_grid = np.linspace(700.0, 1900.0, 61)
    lp_grid = np.linspace(-20.0, 0.0, 41)
    dopants = [("undoped", 0.0, 0.0), ("Sn", 1.0, 0.01), ("Si", 1.0, 0.01), ("Mg", -1.0, 0.01)]

    rows, cols = [], ["dopant", "z_dop", "c_frac", "T_K", "log10_pO2",
                      "log10_VO", "E_F_eV", "log10_n", "log10_p", "residual", "pinned",
                      "half_dmu_O2_T_eV"]
    anchor = torch.tensor([eq.NATIVE_VO_DHF[0], eq.NATIVE_VO_DHF[1], eq.NATIVE_VO_DHF[2]],
                          dtype=torch.float64, device=dev)
    res_by = {}
    for name, z, cf in dopants:
        TT, LL = np.meshgrid(T_grid, lp_grid, indexing="ij")
        Tt = torch.tensor(TT.ravel(), dtype=torch.float64, device=dev)
        Lt = torch.tensor(LL.ravel(), dtype=torch.float64, device=dev)
        # Full Shomate mu_O(T): E_f(T) = anchor + 1/2 dmu_O2(T); solver adds the 1/2 kT ln p term.
        half_dmu = 0.5 * muo._shomate_delta_mu_o2(Tt)
        dH = anchor.view(1, 3) + half_dmu.view(-1, 1)
        r = eq.solve_equilibrium(dH, Tt, Lt,
                                 torch.full_like(Tt, z), torch.full_like(Tt, cf))
        res_by[name] = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                        for k, v in r.items()}
        for i in range(Tt.shape[0]):
            rows.append([name, z, cf, float(Tt[i]), float(Lt[i]),
                         float(r["log10_VO"][i]), float(r["E_F"][i]),
                         float(r["log10_n"][i]), float(r["log10_p"][i]),
                         float(r["residual"][i]), bool(r["pinned"][i]),
                         float(half_dmu[i])])

    arr = np.array([[str(x) for x in row] for row in rows])

    # ── gates ──────────────────────────────────────────────────────────────────
    nT, nP = len(T_grid), len(lp_grid)
    iT_1073 = int(np.argmin(np.abs(T_grid - 1073.0)))          # the lab's 800C anneal
    sel_p = (lp_grid >= -15.0) & (lp_grid <= -5.0)

    vo_u = res_by["undoped"]["log10_VO"].reshape(nT, nP)
    slope = np.polyfit(lp_grid[sel_p], vo_u[iT_1073, sel_p], 1)[0]
    g1 = bool(-0.55 <= slope <= -0.01)

    def at(name, key):
        return res_by[name][key].reshape(nT, nP)[iT_1073, int(np.argmin(np.abs(lp_grid + 4.0)))]

    g2 = bool(at("Sn", "log10_n") > at("undoped", "log10_n") and
              at("Si", "log10_n") > at("undoped", "log10_n") and
              at("Sn", "E_F_eV" if "E_F_eV" in res_by["Sn"] else "E_F") > at("undoped", "E_F"))
    g3 = bool(at("Mg", "E_F") < at("undoped", "E_F") and
              at("Mg", "log10_VO") > at("undoped", "log10_VO"))
    n_u_T = res_by["undoped"]["log10_n"].reshape(nT, nP)[:, int(np.argmin(np.abs(lp_grid + 4.0)))]
    g4 = bool(np.all(np.diff(n_u_T) > -1e-9))

    # G5: residual (cm^-3 units of net charge) small vs the dominant charged species
    rel, n_pinned = [], 0
    for name in res_by:
        r = res_by[name]
        unpinned = ~r["pinned"].astype(bool)
        n_pinned += int((~unpinned).sum())
        dom = np.maximum.reduce([10.0 ** r["log10_n"], 10.0 ** r["log10_p"],
                                 2.0 * r["conc"][:, 2], np.full_like(r["residual"], 1e10)])
        rel.append(np.abs(r["residual"][unpinned]) / dom[unpinned])
    rel = np.concatenate(rel)
    g5 = bool(np.nanmax(rel) < 1e-3)

    gates = {
        "G1_undoped_brouwer_slope": {"slope_dlogVO_dlogpO2_at_1073K": float(slope), "pass": g1},
        "G2_donors_raise_n_and_EF": {"pass": g2},
        "G3_acceptor_lowers_EF_raises_VO": {"pass": g3},
        "G4_undoped_n_rises_with_T": {"pass": g4},
        "G5_charge_balance_residual_unpinned": {"max_relative_residual": float(np.nanmax(rel)),
                                                "n_pinned_points_excluded": n_pinned, "pass": g5},
    }
    return gates, arr, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    a = part_a_port_parity(dev)
    b = part_b_anchor_consistency()
    c_gates, arr, cols = part_c_grid(dev)

    header = ",".join(cols)
    np.savetxt(OUT / "brouwer_grid.csv", arr, fmt="%s", delimiter=",",
               header=header, comments="")

    report = {"A_port_parity": a, "B_anchor_consistency": b, "C_physics_gates": c_gates,
              "device": dev,
              "all_pass": bool(a["pass"] and b["pass"] and
                               all(v["pass"] for v in c_gates.values()))}
    (OUT / "phase0_gates.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
