"""tau GPU campaign — configuration-coordinate (CC) diagram for the native oxygen vacancy V_O.

The seconds-scale device tau / PPC our corpus measures is governed by nonradiative multiphonon
(NMP) carrier capture at deep traps; V_O is THE deep donor implicated in Ga2O3 persistent
photoconductivity. The NMP ingredients — mass-weighted displacement ΔQ between charge states,
Franck-Condon relaxation energies, classical capture barrier — come from the CC diagram: relax
V_O in charge q=0 AND q=+2, then evaluate both charge states along the connecting path.

Existing Ga2O3-Net QE calcs are SCF-only on one (neutral) geometry → no CC data. This computes
the missing per-charge-state relaxations (the τ novelty rung). PBE first (robust/fast); HSE
single-point refinement of the CC path is a later accuracy tier.

Step 1 (this script): build native V_O (2x2x2 undoped supercell − 1 O) and write QE `relax`
inputs for q=0 and q=+2. Launch both on the two GPUs. Step 2: CC-path single-points → NMP → τ.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from pymatgen.core import Structure

SANDBOX = Path(__file__).resolve().parents[1]
NET = SANDBOX.parent / "Ga2O3-Net"
UNDOPED = NET / "data/structures/ordered/undoped.cif"
PSEUDO_DIR = NET / "dft/qe_hse06/pseudos"
CC_DIR = SANDBOX / "dft/qe_cc"
CC_DIR.mkdir(parents=True, exist_ok=True)

MASSES = {"Ga": 69.7230, "O": 15.9994}
UPF = {"Ga": "Ga.upf", "O": "O.upf"}

RELAX_TEMPLATE = """\
&CONTROL
    calculation = 'relax'
    restart_mode = 'from_scratch'
    prefix = '{prefix}'
    pseudo_dir = '{pseudo_dir}'
    outdir = '{outdir}'
    tprnfor = .true.
    tstress = .false.
    verbosity = 'low'
    nstep = 150
    forc_conv_thr = 1.0d-3
/
&SYSTEM
    ibrav = 0
    nat = {nat}
    ntyp = {ntyp}
    ecutwfc = 80.0
    ecutrho = 480.0
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.005
    nspin = 2
    starting_magnetization(1) = 0.1
    starting_magnetization(2) = 0.0
    tot_charge = {q}
/
&ELECTRONS
    electron_maxstep = 200
    conv_thr = 1.0d-7
    mixing_beta = 0.3
    mixing_mode = 'local-TF'
    diagonalization = 'david'
/
&IONS
    ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
{species}
ATOMIC_POSITIONS angstrom
{positions}
K_POINTS automatic
1 1 1 0 0 0
CELL_PARAMETERS angstrom
{cell}
"""


def build_vo_supercell():
    prim = Structure.from_file(UNDOPED)
    sc = prim * (2, 2, 2)
    # remove one O near the cell centre (a bulk-like site, away from image interactions)
    o_idx = [i for i, s in enumerate(sc) if s.specie.symbol == "O"]
    center = sc.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    dists = [np.linalg.norm(sc[i].coords - center) for i in o_idx]
    remove = o_idx[int(np.argmin(dists))]
    sc.remove_sites([remove])
    return sc


def write_relax(sc, q):
    species = sorted({s.specie.symbol for s in sc})
    species_block = "\n".join(f"  {e:4s} {MASSES[e]:10.4f}  {UPF[e]}" for e in species)
    pos = "\n".join(f"  {s.specie.symbol:4s} {s.coords[0]:16.8f} {s.coords[1]:16.8f} "
                    f"{s.coords[2]:16.8f}" for s in sc)
    cell = "\n".join(f"  {v[0]:16.8f} {v[1]:16.8f} {v[2]:16.8f}" for v in sc.lattice.matrix)
    prefix = f"cc_VO_q{q}_relax"
    txt = RELAX_TEMPLATE.format(prefix=prefix, pseudo_dir=str(PSEUDO_DIR),
                                outdir=str(CC_DIR / "outdir" / prefix),
                                nat=len(sc), ntyp=len(species), q=q,
                                species=species_block, positions=pos, cell=cell)
    (CC_DIR / f"{prefix}.in").write_text(txt)
    return prefix, len(sc)


def main():
    sc = build_vo_supercell()
    comp = {}
    for s in sc:
        comp[s.specie.symbol] = comp.get(s.specie.symbol, 0) + 1
    print(f"native V_O supercell: {len(sc)} atoms, composition {comp}")
    for q in (0, 2):
        prefix, nat = write_relax(sc, q)
        print(f"wrote {prefix}.in (nat={nat}, tot_charge={q})")
    print(f"CC inputs in {CC_DIR}")


if __name__ == "__main__":
    main()
