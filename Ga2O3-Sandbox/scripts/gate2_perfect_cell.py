"""Gate-2 seed: build the perfect (undoped) 80-atom β-Ga2O3 supercell PBE SCF input — the
reference energy needed to compute proper defect formation energies from the relaxed charged-
defect calcs (V_O, V_Ga) already done. This is the first piece of rebuilt formation-energy
post-processing (the flagged-broken Phase-53 step) toward reducing the ±0.5 eV energetics error.
"""
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from pymatgen.core import Structure

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
CC = PROJ / "dft/qe_cc"
MASS = {"Ga": 69.723, "O": 15.999}

prim = Structure.from_file(NET / "data/structures/ordered/undoped.cif")
sc = prim * (2, 2, 2)
comp = {}
for s in sc:
    comp[s.specie.symbol] = comp.get(s.specie.symbol, 0) + 1
print(f"perfect supercell: {len(sc)} atoms {comp}")

tmpl = (CC / "cc_VO_q0_relax.in").read_text()
head = tmpl.split("ATOMIC_SPECIES")[0].replace("calculation = 'relax'", "calculation = 'scf'")
head = re.sub(r"\n\s*nstep.*", "", head)
head = re.sub(r"\n\s*forc_conv_thr.*", "", head)
head = head.replace("cc_VO_q0_relax", "gate2_perfect")
head = re.sub(r"nat = \d+", f"nat = {len(sc)}", head)
head = re.sub(r"ntyp = \d+", "ntyp = 2", head)
head = re.sub(r"tot_charge = -?\d+", "tot_charge = 0", head)

species = "ATOMIC_SPECIES\n  Ga    69.7230  Ga.upf\n  O     15.9994  O.upf"
pos = "ATOMIC_POSITIONS angstrom\n" + "\n".join(
    f"  {s.specie.symbol:4s} {s.coords[0]:16.8f} {s.coords[1]:16.8f} {s.coords[2]:16.8f}" for s in sc)
cell = "CELL_PARAMETERS angstrom\n" + "\n".join(
    f"  {v[0]:16.8f} {v[1]:16.8f} {v[2]:16.8f}" for v in sc.lattice.matrix)
(CC / "gate2_perfect.in").write_text(head + species + "\n" + pos + "\nK_POINTS automatic\n1 1 1 0 0 0\n" + cell + "\n")
print("wrote gate2_perfect.in")
