"""Sputtered-film structure campaign — compute the actual as-deposited atomic arrangement of the
doped film (user directive), not the idealised bulk crystal.

Sputter deposition is highly non-equilibrium: as-deposited films are amorphous/nanocrystalline,
with a disordered cation/anion arrangement and dopant local environments that differ from the
bulk crystal — and this disorder is one source of the bulk-equilibrium model's absolute error.
We emulate it with a MACE-MP-0 melt-quench (heat a doped supercell above the melt, then quench),
producing a physically-motivated as-deposited structure whose local order (RDF, coordination,
dopant-O bonds) can be compared to the crystal and (later) fed to defect-energetics in the
disordered matrix.

MLIP-MD (MACE) makes this tractable: thousands of MD steps in minutes on one GPU. Run per dopant.
Usage: python sputter_film_structure.py --gpu 0 --dopant Si --conc 0.0234
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

SANDBOX = Path(__file__).resolve().parents[1]
NET = SANDBOX.parent / "Ga2O3-Net"
OUT = SANDBOX / "dft/sputter_films"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dopant", type=str, default="Si")
    ap.add_argument("--conc", type=float, default=0.0234)   # cation fraction
    ap.add_argument("--melt_K", type=float, default=3000.0)
    ap.add_argument("--melt_steps", type=int, default=3000)
    ap.add_argument("--quench_steps", type=int, default=3000)
    ap.add_argument("--dt_fs", type=float, default=2.0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import mace_mp
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=dev)

    # build a doped 2x2x2 supercell (undoped + substitute the dopant fraction on Ga sites)
    prim = Structure.from_file(NET / "data/structures/ordered/undoped.cif")
    sc = prim * (2, 2, 2)
    ga_idx = [i for i, s in enumerate(sc) if s.specie.symbol == "Ga"]
    n_sub = max(1, round(args.conc * len(ga_idx)))
    rng = np.random.default_rng(0)
    for i in rng.choice(ga_idx, n_sub, replace=False):
        sc.replace(i, args.dopant)
    atoms = AseAtomsAdaptor.get_atoms(sc)
    atoms.calc = calc
    name = f"{args.dopant}_{args.conc:.4f}_sputter"
    log = {"dopant": args.dopant, "conc": args.conc, "n_atoms": len(atoms),
           "n_dopant": n_sub, "melt_K": args.melt_K, "device": dev, "stages": []}

    def rdf_first_peak_and_coord(a, center_sym, neighbor_sym="O", rcut=2.6):
        from ase.neighborlist import neighbor_list
        idx_c = [i for i, s in enumerate(a.get_chemical_symbols()) if s == center_sym]
        ii, jj, dd = neighbor_list("ijd", a, rcut)
        bonds, coords = [], []
        for c in idx_c:
            nb = [dd[k] for k in range(len(ii)) if ii[k] == c
                  and a.get_chemical_symbols()[jj[k]] == neighbor_sym]
            coords.append(len(nb))
            bonds += nb
        return (float(np.mean(bonds)) if bonds else None,
                float(np.mean(coords)) if coords else None)

    # crystalline reference geometry
    b0, c0 = rdf_first_peak_and_coord(atoms, args.dopant)
    log["crystal_dopant_O_bond"] = round(b0, 3) if b0 else None
    log["crystal_dopant_coord"] = round(c0, 2) if c0 else None

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.melt_K)
    # melt
    dyn = Langevin(atoms, args.dt_fs * units.fs, temperature_K=args.melt_K, friction=0.02)
    dyn.run(args.melt_steps)
    e_melt = float(atoms.get_potential_energy())
    log["stages"].append({"stage": "melt", "T": args.melt_K, "E": e_melt})
    # quench to 300 K in stages
    for T in [2000.0, 1000.0, 300.0]:
        dyn.set_temperature(temperature_K=T)
        dyn.run(args.quench_steps // 3)
        log["stages"].append({"stage": "quench", "T": T,
                              "E": float(atoms.get_potential_energy())})
    # final relaxation
    from ase.optimize import FIRE
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=200)

    bq, cq = rdf_first_peak_and_coord(atoms, args.dopant)
    log["asdeposited_dopant_O_bond"] = round(bq, 3) if bq else None
    log["asdeposited_dopant_coord"] = round(cq, 2) if cq else None
    log["final_E"] = float(atoms.get_potential_energy())
    log["final_E_per_atom"] = log["final_E"] / len(atoms)
    # overall disorder metric: Ga-O coordination spread vs crystal
    bga, cga = rdf_first_peak_and_coord(atoms, "Ga")
    log["asdeposited_GaO_coord"] = round(cga, 2) if cga else None
    struct = AseAtomsAdaptor.get_structure(atoms)
    (OUT / f"{name}.json").write_text(json.dumps(
        {**log, "structure": struct.as_dict()}, indent=2))
    (OUT / f"{name}.summary.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
