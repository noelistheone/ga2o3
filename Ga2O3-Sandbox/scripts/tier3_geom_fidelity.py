"""Tier-3 — does an MLIP-relaxed defect GEOMETRY match the QE-relaxed one?

Zero-shot MLIP ENERGETICS are off by ~1 eV (tier3_mlip_defect_benchmark) -> can't
replace QE for the formation energy. But the EXPENSIVE part of a QE defect calc is
the ionic RELAXATION (dozens of SCF cycles). If the MLIP geometry ~ the QE geometry,
then MLIP-relax + ONE QE/HSE single-point reproduces the QE/HSE energetics at ~10x
lower cost, and scales to all dopants. This tests that.

Compares, for V_O (q0), the MLIP-relaxed local structure vs the QE-relaxed one:
  - per-atom displacement RMSD (same start, same fixed cell, same atom order)
  - displacement of the Ga neighbors of the vacancy (the physically important bit)
Also writes a QE scf input at the MACE geometry for an optional single-point check.

Writes results/tier3/geom_fidelity.json.
"""
import json
from pathlib import Path

import numpy as np
import ase.io
from ase.optimize import FIRE

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
OUT = PROJ / "results/tier3"
OUT.mkdir(parents=True, exist_ok=True)
FMAX, STEPS = 0.02, 500


def mace_calc():
    from mace.calculators import mace_mp
    return mace_mp(model="/home/lawrence/.cache/mace/macempa0mediummodel",
                   device="cuda", default_dtype="float64")


def min_image_disp(a_pos, b_pos, cell):
    """b - a with minimum-image convention (fixed identical cell)."""
    d = b_pos - a_pos
    frac = np.linalg.solve(cell.T, d.T).T
    frac -= np.round(frac)
    return frac @ cell


def main():
    init = ase.io.read(str(CC / "cc_VO_q0_relax.in"), format="espresso-in")
    qe_final = ase.io.read(str(CC / "cc_VO_q0_relax.out"), format="espresso-out")
    assert len(init) == len(qe_final) == 79
    cell = np.array(init.cell)

    a = init.copy()
    a.calc = mace_calc()
    FIRE(a, logfile=None).run(fmax=FMAX, steps=STEPS)   # fixed cell (matches QE relax)
    mace_final = a

    # per-atom displacement MACE-final vs QE-final (min image)
    disp = min_image_disp(np.array(qe_final.positions), np.array(mace_final.positions), cell)
    dnorm = np.linalg.norm(disp, axis=1)
    rmsd = float(np.sqrt((dnorm ** 2).mean()))
    dmax = float(dnorm.max())

    # displacement each cell underwent from the UNRELAXED start (magnitude of relaxation)
    d_qe = np.linalg.norm(min_image_disp(np.array(init.positions), np.array(qe_final.positions), cell), axis=1)
    d_mace = np.linalg.norm(min_image_disp(np.array(init.positions), np.array(mace_final.positions), cell), axis=1)

    # identify the vacancy neighbors: Ga atoms nearest the removed-O site.
    # removed O = the O present in perfect but absent in V_O; approximate its site as the
    # centroid of the atoms that relaxed most in QE (they surround the vacancy).
    top = np.argsort(d_qe)[-6:]
    syms = np.array(init.get_chemical_symbols())
    neigh = {
        "top6_relaxed_atoms_QE": [{"idx": int(i), "elem": syms[i],
                                   "qe_disp": round(float(d_qe[i]), 3),
                                   "mace_disp": round(float(d_mace[i]), 3)} for i in top[::-1]],
    }

    result = {
        "system": "V_O q0 (79 atoms), fixed QE cell, same unrelaxed start",
        "rmsd_MACE_vs_QE_final_A": round(rmsd, 4),
        "max_atom_disp_MACE_vs_QE_A": round(dmax, 4),
        "mean_relax_magnitude_QE_A": round(float(d_qe.mean()), 4),
        "mean_relax_magnitude_MACE_A": round(float(d_mace.mean()), 4),
        "max_relax_QE_A": round(float(d_qe.max()), 4),
        "max_relax_MACE_A": round(float(d_mace.max()), 4),
        "neighbors": neigh,
        "interpretation": (
            "If rmsd << mean_relax_magnitude, MACE captures the QE relaxation pattern -> "
            "MLIP-relax + QE/HSE single-point is a valid fast surrogate for the relaxation step."),
    }
    (OUT / "geom_fidelity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # write MACE-relaxed geometry as xyz for a possible QE single-point
    ase.io.write(str(CC / "vo_q0_mace_relaxed.xyz"), mace_final)
    print("wrote", CC / "vo_q0_mace_relaxed.xyz")


if __name__ == "__main__":
    main()
