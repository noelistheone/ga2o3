"""GPU structure layer — MACE-MP-0 relaxation of doped beta-Ga2O3 supercells.

Produces the atomic/molecular structure of each doped material (relaxed geometry + dopant-local
distortion) and a substitution energy, for the full dopant set including species OUTSIDE
KROGER's 19 (Bi, Cu, Er, Al, B, La, Nd, ...). Feeds: (a) atomic-structure deliverable;
(b) energetics cross-check vs KROGER; (c) new-dopant energetics.

Reuses the doped supercells already built in Ga2O3-Net/data/structures/ordered/. One dopant at a
time on ONE GPU (per the serial-per-card rule); launch two instances with --gpu 0 and --gpu 1
and complementary --shard to use both cards.

Outputs (this project):
  dft/mace_relax/relaxed/<name>.json   — relaxed structure + energy + geometry summary
  dft/mace_relax/mace_relax_gpu<N>.csv — running table
Usage: python scripts/mace_relax_doped.py --gpu 0 --shard 0/2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
ORDERED = NET / "data/structures/ordered"
UNDOPED = NET / "data/structures/ordered/undoped.cif"
OUT = PROJ / "dft/mace_relax"
RELAXED = OUT / "relaxed"


def enumerate_targets(shard):
    import re
    cifs = sorted(c for c in glob.glob(str(ORDERED / "*.cif")) if "relaxed" not in c)
    # accept only single-dopant "<Elem>-<float>.cif" (skip co-doping / odd names)
    pat = re.compile(r"^([A-Z][a-z]?)-([0-9]*\.?[0-9]+)\.cif$")
    by_dop = {}
    for c in cifs:
        m = pat.match(os.path.basename(c))
        if not m:
            continue
        dop, conc = m.group(1), float(m.group(2))
        by_dop.setdefault(dop, []).append((conc, c))
    targets = []
    for dop, lst in sorted(by_dop.items()):
        lst.sort()
        # take the two lowest concentrations per dopant (dilute + a check point)
        for conc, c in lst[:2]:
            targets.append((dop, conc, c))
    # also the undoped reference
    targets = [("undoped", 0.0, str(UNDOPED))] + targets
    i, n = [int(x) for x in shard.split("/")]
    return [t for k, t in enumerate(targets) if k % n == i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from mace.calculators import mace_mp
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    RELAXED.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=dev)

    targets = enumerate_targets(args.shard)
    csv = OUT / f"mace_relax_gpu{args.gpu}.csv"
    rows = []
    print(f"[gpu{args.gpu}] {len(targets)} targets, device={dev}", flush=True)

    for dop, conc, cif in targets:
        name = "undoped" if dop == "undoped" else f"{dop}_{conc:.4f}"
        outj = RELAXED / f"{name}.json"
        if outj.exists():
            print(f"[gpu{args.gpu}] skip {name} (done)", flush=True)
            continue
        t0 = time.time()
        try:
            struct = Structure.from_file(cif)
            atoms = AseAtomsAdaptor.get_atoms(struct)
            atoms.calc = calc
            pos0 = atoms.get_positions().copy()
            # relax atoms + cell (FrechetCellFilter) — full structural relaxation
            opt = FIRE(FrechetCellFilter(atoms), logfile=None)
            opt.run(fmax=args.fmax, steps=args.steps)
            E = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmax_final = float(np.linalg.norm(forces, axis=1).max())
            disp = float(np.linalg.norm(atoms.get_positions() - pos0, axis=1).max())
            relaxed = AseAtomsAdaptor.get_structure(atoms)
            # dopant-local geometry: nearest-neighbour bond lengths around the dopant
            nn = None
            if dop != "undoped":
                idxs = [i for i, s in enumerate(relaxed) if s.specie.symbol == dop]
                if idxs:
                    di = idxs[0]
                    dists = sorted(relaxed.get_neighbors(relaxed[di], 3.0),
                                   key=lambda x: x[1])[:6]
                    nn = [round(d[1], 3) for d in dists]
            payload = {
                "name": name, "dopant": dop, "concentration": conc,
                "n_atoms": len(relaxed), "energy_eV": E, "energy_per_atom": E / len(relaxed),
                "fmax_final": fmax_final, "n_steps": opt.nsteps,
                "max_disp_during_relax": disp,
                "dopant_nn_bond_lengths": nn,
                "lattice_abc": [round(x, 4) for x in relaxed.lattice.abc],
                "structure": relaxed.as_dict(),
                "mace_model": "mace-mp-0 medium", "relaxation": "atoms+cell (FrechetCellFilter)",
            }
            outj.write_text(json.dumps(payload))
            dt = time.time() - t0
            rows.append({"name": name, "dopant": dop, "conc": conc, "E": E,
                         "E_per_atom": E / len(relaxed), "fmax": fmax_final,
                         "nn": nn, "sec": round(dt, 1)})
            print(f"[gpu{args.gpu}] {name}: E={E:.3f} eV E/at={E/len(relaxed):.4f} "
                  f"fmax={fmax_final:.3f} nn={nn} ({dt:.0f}s)", flush=True)
        except Exception as e:
            print(f"[gpu{args.gpu}] FAIL {name}: {e}", flush=True)
        # append CSV each iter (crash-safe)
        import csv as _csv
        with open(csv, "w", newline="") as f:
            if rows:
                w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"[gpu{args.gpu}] DONE {len(rows)} relaxed", flush=True)


if __name__ == "__main__":
    main()
