"""Tier-3 — MLIP defect-energetics benchmark (the decisive accuracy gate), v2.

v1 finding: MACE-MP-0 and ORB overestimate the neutral V_O formation energy by
~1.7-1.9 eV vs QE-PBE (3.09 eV) when using the MLIP's OWN 1/2 E(O2) reference.
That is the well-known O2-molecule reference error of MP-trained MLIPs, NOT
necessarily an error in the solid-state energetics.

v2 isolates it: compute the vacancy formation energy at BOTH chemical-potential
limits, referencing O TWO ways:
  (A) O-rich : mu_O = 1/2 E(O2)               [contaminated by the O2 error]
  (B) Ga-rich: mu_O = [E(Ga2O3 fu) - 2 mu_Ga]/3, mu_Ga = E(alpha-Ga)/8
      -> pure-SOLIDS reference (bulk Ga metal + bulk Ga2O3), O2-free.
QE ground truth: E_f(V_O,q0) = 3.09 (O-rich) ... 0.14 (Ga-rich)  [PBE].
Also: MLIP Ga2O3 formation enthalpy per f.u. vs experiment (-11.29 eV) -> shows the
O2 error magnitude directly.

If (B) Ga-rich MATCHES QE while (A) O-rich does not, the MLIP solid energetics are
good and only a single mu_O offset needs calibrating -> MLIPs CAN drive the sandbox
defect thermodynamics for all dopants. That is the architecture-deciding result.

Writes results/tier3/mlip_defect_benchmark.json.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
OUT = PROJ / "results/tier3"
OUT.mkdir(parents=True, exist_ok=True)

QE = {"O_rich": 3.09, "mid": 1.61, "Ga_rich": 0.14, "HSE_O_rich": 3.67, "KROGER_HSE": 3.5}
EXP_DHF_GA2O3 = -11.29   # eV / f.u. (approx -1089 kJ/mol) experimental formation enthalpy
FMAX = 0.03
STEPS = 400


def alpha_Ga():
    """alpha-Ga (Cmce, oS8) built from spacegroup + Wyckoff 8f — pure-solids Ga reference."""
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    lat = Lattice.orthorhombic(4.5192, 7.6586, 4.5258)
    s = Structure.from_spacegroup("Cmce", lat, ["Ga"], [[0.0, 0.15263, 0.08128]])
    return AseAtomsAdaptor.get_atoms(s)


def load_structs():
    perfect = ase.io.read(str(CC / "gate2_perfect.in"), format="espresso-in")   # 80 atoms = 16 f.u.
    vo = ase.io.read(str(CC / "cc_VO_q0_relax.in"), format="espresso-in")        # 79 atoms
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]], cell=[15, 15, 15], pbc=True)
    ga = alpha_Ga()
    return perfect, vo, o2, ga


# ---- calculators -----------------------------------------------------------
def get_calc(name):
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="/home/lawrence/.cache/mace/macempa0mediummodel",
                       device="cuda", default_dtype="float64"), None
    if name == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        for attr in ["orb_v3_conservative_inf_omat", "orb_v3_conservative_20_omat",
                     "orb_v2", "orb_d3_v2", "orb_mptraj_only_v2"]:
            if hasattr(pretrained, attr):
                m = getattr(pretrained, attr)(device="cuda")
                return ORBCalculator(m, device="cuda"), attr
        raise RuntimeError("no orb attr")
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet
        return CHGNetCalculator(CHGNet.load(), use_device="cuda"), None
    raise ValueError(name)


def relax(atoms, name, calc, cell=False):
    """Relax; CHGNet cell-relax via its own StructOptimizer to dodge the float32/64 clash."""
    a = atoms.copy()
    t0 = time.time()
    if name == "chgnet" and cell:
        from chgnet.model import StructOptimizer
        from pymatgen.io.ase import AseAtomsAdaptor
        opt = StructOptimizer(use_device="cuda")
        r = opt.relax(AseAtomsAdaptor.get_structure(a), relax_cell=True,
                      fmax=FMAX, steps=STEPS, verbose=False)
        e = float(r["trajectory"].energies[-1])
        vol = float(r["final_structure"].volume)
        return {"E": e, "t": round(time.time() - t0, 1), "vol": vol, "converged": True}
    a.calc = calc
    dyn = FIRE(FrechetCellFilter(a) if cell else a, logfile=None)
    dyn.run(fmax=FMAX, steps=STEPS)
    e = float(a.get_potential_energy())
    fmax = float(np.abs(a.get_forces()).max())
    return {"E": e, "t": round(time.time() - t0, 1), "vol": float(a.get_volume()),
            "converged": fmax <= FMAX * 1.5}


def run_method(name, perfect, vo, o2, ga):
    res = {"method": name}
    try:
        calc, variant = get_calc(name)
        res["variant"] = variant
        rp = relax(perfect, name, calc, cell=True)
        rv = relax(vo, name, calc, cell=False)
        ro = relax(o2, name, calc, cell=False)
        rg = relax(ga, name, calc, cell=True)

        E_perfect, E_VO, E_O2, E_Ga = rp["E"], rv["E"], ro["E"], rg["E"]
        E_fu = E_perfect / 16.0                       # Ga2O3 formula unit (80 atoms/5)
        mu_Ga_bulk = E_Ga / len(ga)                   # per Ga atom
        mu_O_Orich = 0.5 * E_O2
        mu_O_Garich = (E_fu - 2.0 * mu_Ga_bulk) / 3.0

        ef_Orich = E_VO - E_perfect + mu_O_Orich
        ef_Garich = E_VO - E_perfect + mu_O_Garich
        dHf_Ga2O3 = E_fu - 2.0 * mu_Ga_bulk - 3.0 * mu_O_Orich   # vs exp -11.29

        res.update({
            "E_perfect": round(E_perfect, 4), "E_VO": round(E_VO, 4),
            "E_O2": round(E_O2, 4), "E_alphaGa_cell": round(E_Ga, 4),
            "mu_Ga_bulk": round(mu_Ga_bulk, 4),
            "mu_O_Orich_halfO2": round(mu_O_Orich, 4),
            "mu_O_Garich_solids": round(mu_O_Garich, 4),
            "Ef_VO_Orich": round(ef_Orich, 4),
            "Ef_VO_Garich": round(ef_Garich, 4),
            "d_Orich_vs_QE": round(ef_Orich - QE["O_rich"], 3),
            "d_Garich_vs_QE": round(ef_Garich - QE["Ga_rich"], 3),
            "dHf_Ga2O3_fu": round(dHf_Ga2O3, 3),
            "dHf_error_vs_exp": round(dHf_Ga2O3 - EXP_DHF_GA2O3, 3),
            "perfect_vol": round(rp["vol"], 2), "perfect_vol_QE": 845.9,
            "conv": {k: v["converged"] for k, v in
                     [("perfect", rp), ("vo", rv), ("o2", ro), ("ga", rg)]},
            "t_sec": {k: v["t"] for k, v in
                      [("perfect", rp), ("vo", rv), ("o2", ro), ("ga", rg)]},
        })
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-1800:]
    return res


def main():
    perfect, vo, o2, ga = load_structs()
    methods = sys.argv[1:] or ["mace", "orb", "chgnet"]
    out = {"qe_reference": QE, "exp_dHf_Ga2O3_fu": EXP_DHF_GA2O3, "methods": {}}
    for m in methods:
        print(f"[{m}] running...", flush=True)
        r = run_method(m, perfect, vo, o2, ga)
        out["methods"][m] = r
        if "error" in r:
            print(f"[{m}] ERROR: {r['error']}", flush=True)
        else:
            print(f"[{m}] Ef(V_O) O-rich={r['Ef_VO_Orich']} (d{r['d_Orich_vs_QE']:+})  "
                  f"Ga-rich={r['Ef_VO_Garich']} (d{r['d_Garich_vs_QE']:+})  "
                  f"dHf(Ga2O3)={r['dHf_Ga2O3_fu']} (exp -11.29, err {r['dHf_error_vs_exp']:+})",
                  flush=True)
        (OUT / "mlip_defect_benchmark.json").write_text(json.dumps(out, indent=2))
    print("WROTE", OUT / "mlip_defect_benchmark.json", flush=True)


if __name__ == "__main__":
    main()
