"""R6 reviewer #2 item 9: LOW-DENSITY amorphous cells to test the 'conservative bound' claim.

The paper argues the fixed near-crystalline-density ensembles (5.95 g/cc) give a conservative
(lower-bound) disorder narrowing because real sputtered films are LESS dense and would disorder
further. This script measures that monotonicity directly: MatterSim melt-quench at 5.4 and
5.0 g/cc (one cell each, protocol identical to tier3_mattersim_disorder.py), for QE-PBE DOS
extraction with the identical occupation-based gap. Runs in the isolated `mattersim` env.
Writes dft/disorder_mattersim/amorph_lowrho_{54,50}.xyz + lowrho_summary.json."""
import sys, json
from pathlib import Path
import numpy as np
import ase.io
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.nvtberendsen import NVTBerendsen
from mattersim.forcefield import MatterSimCalculator

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "dft/disorder_mattersim"
MELT_K, QUENCH_STEPS, DT = 3000.0, 4000, 2.0
TARGETS = [("54", 5.4, 10), ("50", 5.0, 11)]      # (tag, g/cc, seed)


def density_gcc(atoms):
    m = atoms.get_masses().sum() * 1.66053906660e-24
    v = atoms.get_volume() * 1e-24
    return m / v


def bond_stats(atoms, rmax=2.6):
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, rmax)
    sym = atoms.get_chemical_symbols()
    gao = [d[k] for k in range(len(d)) if {sym[i[k]], sym[j[k]]} == {"Ga", "O"}]
    return (float(np.mean(gao)), float(np.std(gao))) if gao else (0.0, 0.0)


def melt_quench(rho_target, seed, calc):
    atoms = ase.io.read(str(PROJ / "dft/qe_cc/gate2_perfect.in"), format="espresso-in")
    scale = (density_gcc(atoms) / rho_target) ** (1.0 / 3.0)
    atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
    atoms.calc = calc
    atoms.set_pbc(True)
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=MELT_K, rng=rng)
    Stationary(atoms)
    nvt = NVTBerendsen(atoms, timestep=DT * units.fs, temperature_K=MELT_K, taut=50 * units.fs)
    nvt.run(2000)
    for i in range(QUENCH_STEPS):
        T = MELT_K - (MELT_K - 300.0) * (i / QUENCH_STEPS)
        nvt.temperature = T * units.kB
        nvt.run(1)
    return atoms


def main():
    calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")
    cells = []
    for tag, rho, seed in TARGETS:
        a = melt_quench(rho, seed, calc)
        ase.io.write(str(OUT / f"amorph_lowrho_{tag}.xyz"), a)
        bm, bs = bond_stats(a)
        cells.append({"tag": tag, "target_gcc": rho, "density_gcc": round(density_gcc(a), 3),
                      "bond_mean_A": round(bm, 3), "bond_std_A": round(bs, 3), "seed": seed})
        print(f"lowrho_{tag}: rho={rho} g/cc, GaO bond {bm:.3f}+-{bs:.3f} A", flush=True)
    out = {"protocol": "identical to tier3_mattersim_disorder.py (NVT melt 3000 K + staged quench "
                       "to 300 K, 2 fs), cell isotropically scaled to the target density",
           "reference_595": "3 cells at 5.95 g/cc: gaps 1.671/1.844/1.751 (mean 1.755), "
                            "bond_std ~0.14 A",
           "cells": cells}
    (OUT / "lowrho_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
