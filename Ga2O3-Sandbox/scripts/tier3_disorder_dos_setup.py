"""Set up QE scf(+DOS) inputs on the MACE melt-quench SPUTTER-FILM disordered cell and a
crystalline reference, to compute the disorder-induced band-edge / Urbach narrowing that the
dEg FOM is dominated by (rigorous closure of the disorder-dEg scoping). The band-edge shift
disordered-vs-crystal = the disorder gap narrowing; combined with Burstein-Moss it gives the
net dEg from first principles instead of the deformation-potential order-of-magnitude proxy.

PBE first (trend); the disordered gap is a lower bound but the disordered-vs-crystal SHIFT
is what matters and is more robust to the PBE gap error. Writes disorder_<M>_film.in.
"""
import json
import re
import sys
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

CC = Path("/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc")
SF = Path("/home/lawrence/Physics/Ga2O3-Sandbox/dft/sputter_films")
MASS = {"Ga": 69.723, "O": 15.999, "Si": 28.085, "Sn": 118.71, "Mg": 24.305, "Zn": 65.38}
FILM = sys.argv[1] if len(sys.argv) > 1 else "Si_0.0234"

d = json.load(open(SF / f"{FILM}_sputter.json"))
struct = Structure.from_dict(d["structure"])
syms = [str(s) for s in struct.species]
uniq = []
for s in syms:
    if s not in uniq:
        uniq.append(s)

# base header from the defect template (same pseudos/ecut), scf + DOS-friendly (more bands)
base = (CC / "cc_VO_q0_relax.in").read_text()
t = base
t = t.replace("calculation = 'relax'", "calculation = 'scf'")
t = re.sub(r"\n\s*nstep\s*=.*", "", t)
t = re.sub(r"\n\s*forc_conv_thr\s*=.*", "", t)
t = re.sub(r"&IONS.*?/\n", "", t, flags=re.S)
prefix = f"disorder_{FILM}_film"
t = re.sub(r"prefix = '[^']+'", f"prefix = '{prefix}'", t)
t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / prefix}'", t)
t = re.sub(r"nat = \d+", f"nat = {len(struct)}", t)
t = re.sub(r"ntyp = \d+", f"ntyp = {len(uniq)}", t)
# add extra empty bands for a clean conduction-edge DOS
if "nbnd" not in t:
    t = t.replace("tot_charge = 0", "tot_charge = 0\n    nbnd = 400")
# species
spec = "ATOMIC_SPECIES\n" + "".join(f"  {s:<4s} {MASS[s]:9.3f}  {s}.upf\n" for s in uniq)
t = re.sub(r"ATOMIC_SPECIES\n(?:\s*[A-Z][a-z]?\s+[\d.]+\s+\S+\.upf\s*\n)+", spec, t)
# cell + positions (angstrom, cartesian)
lat = np.array(struct.lattice.matrix)
cell = "CELL_PARAMETERS angstrom\n" + "".join(
    f"  {lat[i,0]:14.9f} {lat[i,1]:14.9f} {lat[i,2]:14.9f}\n" for i in range(3))
pos = "ATOMIC_POSITIONS angstrom\n" + "".join(
    f"  {str(s):<4s} {c[0]:16.10f} {c[1]:16.10f} {c[2]:16.10f}\n"
    for s, c in zip(syms, struct.cart_coords))
t = re.sub(r"CELL_PARAMETERS[^\n]*\n(?:\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+", cell, t)
t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+", pos, t)

(CC / f"{prefix}.in").write_text(t)
print(f"wrote {prefix}.in  nat={len(struct)} ntyp={len(uniq)} species={uniq}")
print("READY: run when GPUs free -> scf on disordered film -> DOS band-edge vs crystal = disorder narrowing")
