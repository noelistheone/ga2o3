"""M1 response: supercell-scaling check of the charged-defect finite-size treatment.
Build a 160-atom cell (2x1x1 of the 80-atom cell), MACE-relax the neutral V_O, then PBE
single-points: perfect(160), V_O q0(160), V_O q2(160, at the neutral geometry -> isolates the
electrostatic monopole scaling from relaxation). Compare E_f(q2)+MP(L) between 80- and 160-atom
cells: if the MP-corrected values agree, the monopole treatment is converged at the stated level.
Writes results/tier3/finite_size_160.json (energies; assembly step computes the comparison)."""
import sys
from pathlib import Path
import numpy as np, ase.io
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"src"))
from ase.optimize import BFGS
from mace.calculators import mace_mp
CC=PROJ/"dft/qe_cc"
a80=ase.io.read(str(CC/"gate2_perfect.in"),format="espresso-in")
a160=a80.repeat((2,1,1))
calc=mace_mp(model="/home/lawrence/.cache/mace/macempa0mediummodel",device="cuda",default_dtype="float64")
# V_O: remove the O corresponding to the SAME site as the 80-atom study (first O in cell)
o_idx=[i for i,s in enumerate(a160.get_chemical_symbols()) if s=="O"][0]
vo=a160.copy(); del vo[o_idx]
vo.calc=calc
BFGS(vo,logfile=None).run(fmax=0.02,steps=300)
ase.io.write(str(CC/"fs160_VO_mace.xyz"),vo)
ase.io.write(str(CC/"fs160_perfect.xyz"),a160)
print("MACE 160-atom relax done",flush=True)
# emit QE inputs by cloning gate2 settings
base=(CC/"gate2_perfect.in").read_text()
import re as _re
def qe_input(atoms,prefix,tot_charge=0):
    t=base
    t=_re.sub(r"prefix = '[^']+'",f"prefix = '{prefix}'",t)
    t=_re.sub(r"outdir = '[^']+'",f"outdir = '{CC/'outdir'/prefix}'",t)
    t=_re.sub(r"nat = \d+",f"nat = {len(atoms)}",t)
    t=_re.sub(r"tot_charge = \d+",f"tot_charge = {tot_charge}",t)
    if "tot_charge" not in t: t=t.replace("&SYSTEM",f"&SYSTEM\n    tot_charge = {tot_charge}")
    cell=atoms.cell[:]
    cs="CELL_PARAMETERS angstrom\n"+"".join(f"  {c[0]:.9f} {c[1]:.9f} {c[2]:.9f}\n" for c in cell)
    ps="ATOMIC_POSITIONS angstrom\n"+"".join(f"  {s:<3s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n"
        for s,p in zip(atoms.get_chemical_symbols(),atoms.get_positions()))
    t=_re.sub(r"CELL_PARAMETERS[^\n]*\n(?:\s*[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*\n)+",cs,t)
    t=_re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.eE]+\s+[-\d.eE]+\s*[-\d.eE]+\s*\n)+",ps,t)
    (CC/f"{prefix}.in").write_text(t)
qe_input(a160,"fs160_perfect",0)
qe_input(vo,"fs160_VO_q0",0)
qe_input(vo,"fs160_VO_q2",2)
print("QE inputs written: fs160_{perfect,VO_q0,VO_q2}.in",flush=True)
