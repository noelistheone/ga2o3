"""Set up QE-PBE SHORT relaxations for new dopants, starting from the MACE-MPA-0 geometry
(Tier B: MACE pre-relax -> short QE final relax removes the ~150 meV geometry error at a
fraction of a from-scratch relaxation). Neutral substitutional Sb_Ga / Bi_Ga at the
MACE-preferred octahedral site. Writes dft/qe_cc/dop_<M>_relax.in.
"""
import json
import re
from pathlib import Path
import ase.io

CC = Path("/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc")
MASS = {"Ga": 69.723, "O": 15.999, "Sb": 121.760, "Bi": 208.980}

for elem in ["Sb", "Bi"]:
    res = json.load(open(f"/home/lawrence/Physics/Ga2O3-Sandbox/results/tier3/new_dopant_{elem}.json"))
    site = res["preferred_site"]
    atoms = ase.io.read(str(CC / f"dop_{elem}_{site}_mace.xyz"))

    src = (CC / "cc_VO_q0_relax.in").read_text()
    t = src
    t = re.sub(r"prefix = '[^']+'", f"prefix = 'dop_{elem}_relax'", t)
    t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / ('dop_' + elem + '_relax')}'", t)
    t = re.sub(r"nstep\s*=.*", "nstep = 40", t)
    t = re.sub(r"nat = \d+", f"nat = {len(atoms)}", t)      # 80 (substitution, not vacancy)
    t = re.sub(r"ntyp = \d+", "ntyp = 3", t)
    t = re.sub(r"tot_charge = 0", "tot_charge = 0", t)
    # species block: Ga, O, M
    spec = "ATOMIC_SPECIES\n"
    for s in ["Ga", "O", elem]:
        spec += f"  {s:<4s} {MASS[s]:9.3f}  {s}.upf\n"
    t = re.sub(r"ATOMIC_SPECIES\n(?:\s*[A-Z][a-z]?\s+[\d.]+\s+\S+\.upf\s*\n)+", spec, t)
    # positions
    pos = "ATOMIC_POSITIONS angstrom\n"
    for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        pos += f"  {s:<4s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}\n"
    t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+",
               pos, t)
    (CC / f"dop_{elem}_relax.in").write_text(t)
    print(f"wrote dop_{elem}_relax.in  (nat={len(atoms)}, site={site}, ntyp=3, {elem}.upf)")
