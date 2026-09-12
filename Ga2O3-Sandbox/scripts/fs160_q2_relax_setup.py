"""Set up the 160-atom q=2+ PBE relaxation. Start = 160-atom MACE-neutral cell with the 80-atom
(R2-R0) displacement pattern applied around the vacancy (halves the ionic steps)."""
import sys, re
from pathlib import Path
import numpy as np, ase.io
PROJ=Path(__file__).resolve().parents[1]; CC=PROJ/"dft/qe_cc"
v160=ase.io.read(str(CC/"fs160_VO_mace.xyz"))
# 80-atom R0(neutral relaxed) and R2(charged relaxed) from the cc campaign
r0=ase.io.read(str(CC/"cc_VO_q0_relax.out"),format="espresso-out")
r2=ase.io.read(str(CC/"cc_VO_q2_relax.out"),format="espresso-out")
d=r2.get_positions()-r0.get_positions()          # displacement field (80-atom, minus the vacancy)
# map onto the first-half copy of the 160 cell (same ordering: repeat((2,1,1)) tiles copy0 first)
p=v160.get_positions()
n80=len(r0)                                       # 79 atoms
p[:n80]+=d
v160.set_positions(p)
base=(CC/"fs160_VO_q2.in").read_text()
t=base.replace("calculation = 'scf'","calculation = 'relax'")
t=re.sub(r"prefix = '[^']+'","prefix = 'fs160_VO_q2_relax'",t)
t=re.sub(r"outdir = '[^']+'",f"outdir = '{CC/'outdir'/'fs160_VO_q2_relax'}'",t)
if "&IONS" not in t: t=t.replace("&ELECTRONS","&IONS\n  ion_dynamics='bfgs'\n/\n&ELECTRONS")
if "nstep" not in t: t=t.replace("&CONTROL","&CONTROL\n    nstep = 120")
if "forc_conv_thr" not in t: t=t.replace("&CONTROL","&CONTROL\n    forc_conv_thr = 1.0d-3")
ps="ATOMIC_POSITIONS angstrom\n"+"".join(f"  {s:<3s} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f}\n"
    for s,q in zip(v160.get_chemical_symbols(),v160.get_positions()))
t=re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*\n)+",ps,t)
(CC/"fs160_VO_q2_relax.in").write_text(t)
print("fs160_VO_q2_relax.in written (seeded with 80-atom R2 displacement)")
