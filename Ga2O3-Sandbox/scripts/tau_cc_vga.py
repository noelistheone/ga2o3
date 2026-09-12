"""Build native V_Ga configuration-coordinate relax inputs (q=0 and q=-3) for the τ campaign.
V_Ga is the E1/E2-related deep acceptor trap, the second key defect for Ga2O3 PPC/τ."""
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from pymatgen.core import Structure

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
CC = PROJ / "dft/qe_cc"
CC.mkdir(parents=True, exist_ok=True)
MASS = {"Ga": 69.723, "O": 15.999}
UPF = {"Ga": "Ga.upf", "O": "O.upf"}

prim = Structure.from_file(NET / "data/structures/ordered/undoped.cif")
sc = prim * (2, 2, 2)
ga = [i for i, s in enumerate(sc) if s.specie.symbol == "Ga"]
center = sc.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
rm = ga[int(np.argmin([np.linalg.norm(sc[i].coords - center) for i in ga]))]
sc.remove_sites([rm])
comp = {}
for s in sc:
    comp[s.specie.symbol] = comp.get(s.specie.symbol, 0) + 1
print(f"V_Ga supercell: {len(sc)} atoms {comp}")

tmpl = (CC / "cc_VO_q0_relax.in").read_text()
head = tmpl.split("ATOMIC_SPECIES")[0]
species = sorted({s.specie.symbol for s in sc})
sp = "ATOMIC_SPECIES\n" + "\n".join(f"  {e:4s} {MASS[e]:10.4f}  {UPF[e]}" for e in species)
pos = "ATOMIC_POSITIONS angstrom\n" + "\n".join(
    f"  {s.specie.symbol:4s} {s.coords[0]:16.8f} {s.coords[1]:16.8f} {s.coords[2]:16.8f}" for s in sc)
cell = "CELL_PARAMETERS angstrom\n" + "\n".join(
    f"  {v[0]:16.8f} {v[1]:16.8f} {v[2]:16.8f}" for v in sc.lattice.matrix)

for q in (0, -3):
    pfx = f"cc_VGa_q{q}_relax"
    h = head.replace("cc_VO_q0_relax", pfx)
    h = re.sub(r"nat = \d+", f"nat = {len(sc)}", h)
    h = re.sub(r"ntyp = \d+", f"ntyp = {len(species)}", h)
    h = re.sub(r"tot_charge = -?\d+", f"tot_charge = {q}", h)
    h = h.replace(f"outdir/cc_VO_q0_relax", f"outdir/{pfx}")
    (CC / f"{pfx}.in").write_text(h + sp + "\n" + pos + "\nK_POINTS automatic\n1 1 1 0 0 0\n" + cell + "\n")
    print(f"wrote {pfx}.in (nat={len(sc)}, ntyp={len(species)}, q={q})")
