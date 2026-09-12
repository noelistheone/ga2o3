"""Disorder-dEg ensemble generator (research recipe: 15-30 independent melt-quench cells,
NOT one). Produces N amorphous a-Ga2O3 cells (80 atoms, MACE-MPA-0 melt-quench, different
random seeds) for the crystal->amorphous gap-narrowing / Urbach study. Each cell + a summary
(density, RDF-ish bond stats) saved; the crystalline reference is gate2_perfect.

Usage: tier3_disorder_ensemble.py [N] [start_seed]
Writes dft/disorder_ensemble/cell_<seed>.xyz + ensemble_summary.json
"""
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import ase.io
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

PROJ = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
CC = PROJ / "dft/qe_cc"
OUT = PROJ / "dft/disorder_ensemble"
OUT.mkdir(parents=True, exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MELT_K, MELT_STEPS, QUENCH_STEPS, DT = 3000.0, 3000, 3000, 2.0


def calc():
    from mace.calculators import mace_mp
    # float32 for the MD sampling (fast); precision not needed for amorphous-ensemble structures
    return mace_mp(model="/home/lawrence/.cache/mace/macempa0mediummodel", device="cuda",
                   default_dtype="float32")


def bond_stats(atoms):
    """Ga-O bond mean/std (the structural-disorder metric feeding the narrowing)."""
    pos = atoms.get_positions(); sym = np.array(atoms.get_chemical_symbols())
    cell = np.array(atoms.cell)
    ga = np.where(sym == "Ga")[0]; o = np.where(sym == "O")[0]
    bonds = []
    for g in ga:
        d = pos[o] - pos[g]
        frac = np.linalg.solve(cell.T, d.T).T; frac -= np.round(frac)
        dist = np.linalg.norm(frac @ cell, axis=1)
        bonds += list(dist[dist < 2.4])
    bonds = np.array(bonds)
    return float(bonds.mean()), float(bonds.std()), len(bonds)


def melt_quench(seed, c):
    atoms = ase.io.read(str(CC / "gate2_perfect.in"), format="espresso-in")
    atoms.calc = c
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=MELT_K, rng=rng)
    dyn = Langevin(atoms, DT * units.fs, temperature_K=MELT_K, friction=0.02, rng=rng)
    dyn.run(MELT_STEPS)
    for T in (2200.0, 1400.0, 700.0, 300.0):
        dyn.set_temperature(temperature_K=T)
        dyn.run(QUENCH_STEPS // 4)
    # short final relax at 0 K
    from ase.optimize import FIRE
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=200)
    return atoms


def main():
    c = calc()
    summary = {"melt_K": MELT_K, "n_cells": N, "cells": []}
    for i in range(N):
        seed = START + i
        atoms = melt_quench(seed, c)
        mean, std, nb = bond_stats(atoms)
        ase.io.write(str(OUT / f"cell_{seed}.xyz"), atoms)
        rec = {"seed": seed, "E_per_atom": round(float(atoms.get_potential_energy()) / len(atoms), 4),
               "GaO_bond_mean_A": round(mean, 3), "GaO_bond_std_A": round(std, 3),
               "n_GaO_bonds": nb, "volume_A3": round(float(atoms.get_volume()), 1)}
        summary["cells"].append(rec)
        print(f"cell {seed}: GaO std={std:.3f} A  E/at={rec['E_per_atom']}  V={rec['volume_A3']}", flush=True)
        (OUT / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    stds = [c["GaO_bond_std_A"] for c in summary["cells"]]
    summary["GaO_std_ensemble_mean"] = round(float(np.mean(stds)), 4)
    summary["GaO_std_ensemble_sem"] = round(float(np.std(stds) / np.sqrt(len(stds))), 4)
    (OUT / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"ENSEMBLE DONE: {N} cells, GaO_std = {summary['GaO_std_ensemble_mean']} "
          f"+/- {summary['GaO_std_ensemble_sem']} A", flush=True)


if __name__ == "__main__":
    main()
