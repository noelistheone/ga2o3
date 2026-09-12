"""Elemental metal references (Ga, Sb, Bi) for the ABSOLUTE new-dopant formation-energy
diagram + solubility. mu_M^0 = E(metal)/atom (QE-PBE, same pseudos/ecut as the defect cells).
Writes QE scf inputs with a k-mesh + smearing (metals). alpha-Ga (Cmce), Sb/Bi A7 (R-3m).
"""
import re
from pathlib import Path
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

CC = Path("/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc")
MASS = {"Ga": 69.723, "Sb": 121.760, "Bi": 208.980}
PSEUDO_DIR = "/home/lawrence/Physics/Ga2O3-Net/dft/qe_hse06/pseudos"


def build(elem):
    if elem == "Ga":                       # alpha-Ga, Cmce (oS8)
        lat = Lattice.orthorhombic(4.5192, 7.6586, 4.5258)
        return Structure.from_spacegroup("Cmce", lat, ["Ga"], [[0.0, 0.15263, 0.08128]])
    if elem == "Sb":                       # A7 rhombohedral, R-3m (hex setting)
        lat = Lattice.hexagonal(4.307, 11.273)
        return Structure.from_spacegroup("R-3m", lat, ["Sb"], [[0.0, 0.0, 0.2336]])
    if elem == "Bi":
        lat = Lattice.hexagonal(4.546, 11.862)
        return Structure.from_spacegroup("R-3m", lat, ["Bi"], [[0.0, 0.0, 0.2341]])


HEADER = """&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = '{prefix}'
    pseudo_dir = '{pd}'
    outdir = '{outdir}'
    tprnfor = .true.
    verbosity = 'low'
/
&SYSTEM
    ibrav = 0
    nat = {nat}
    ntyp = 1
    ecutwfc = 80.0
    ecutrho = 480.0
    occupations = 'smearing'
    smearing = 'mv'
    degauss = 0.01
/
&ELECTRONS
    conv_thr = 1.0d-7
    mixing_beta = 0.3
/
ATOMIC_SPECIES
  {elem} {mass:9.3f}  {elem}.upf
"""


def write_input(elem):
    s = build(elem)
    prefix = f"metal_{elem}"
    lat = np.array(s.lattice.matrix)
    txt = HEADER.format(prefix=prefix, pd=PSEUDO_DIR, outdir=str(CC / "outdir" / prefix),
                        nat=len(s), elem=elem, mass=MASS[elem])
    txt += "CELL_PARAMETERS angstrom\n" + "".join(
        f"  {lat[i,0]:14.9f} {lat[i,1]:14.9f} {lat[i,2]:14.9f}\n" for i in range(3))
    txt += "ATOMIC_POSITIONS angstrom\n" + "".join(
        f"  {elem:<4s} {c[0]:16.10f} {c[1]:16.10f} {c[2]:16.10f}\n" for c in s.cart_coords)
    txt += "K_POINTS automatic\n  6 6 6 0 0 0\n"
    (CC / f"{prefix}.in").write_text(txt)
    print(f"wrote metal_{elem}.in  nat={len(s)}")


if __name__ == "__main__":
    for e in ["Ga", "Sb", "Bi"]:
        write_input(e)
