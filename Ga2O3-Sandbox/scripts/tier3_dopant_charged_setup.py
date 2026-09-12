"""After the neutral relax, set up SHORT charged relaxations (q=+1, +2) for a new dopant,
starting from the relaxed NEUTRAL geometry. Sb/Bi on the Ga site can be donors (Sb5+/Bi5+
-> +2 double donor) or isovalent (Sb3+/Bi3+ -> neutral). The charged states give the
transition levels (mu-INDEPENDENT) -> donor/acceptor classification. Substitutional
donors have small charge-geometry coupling (unlike negative-U V_O), so a short relax from
the neutral geometry converges fast.

Usage: tier3_dopant_charged_setup.py Sb   (reads dop_Sb_relax.out)
"""
import re
import sys
from pathlib import Path
import ase.io

CC = Path("/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc")
elem = sys.argv[1]

base = (CC / f"dop_{elem}_relax.in").read_text()
relaxed = ase.io.read(str(CC / f"dop_{elem}_relax.out"), format="espresso-out")   # final geom

pos = "ATOMIC_POSITIONS angstrom\n"
for s, p in zip(relaxed.get_chemical_symbols(), relaxed.get_positions()):
    pos += f"  {s:<4s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}\n"

for q in [1, 2]:
    t = base
    t = re.sub(r"prefix = '[^']+'", f"prefix = 'dop_{elem}_q{q}'", t)
    t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / ('dop_' + elem + '_q' + str(q))}'", t)
    t = re.sub(r"tot_charge = [-\d.]+", f"tot_charge = {q}", t)
    t = re.sub(r"nstep\s*=.*", "nstep = 25", t)
    t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+",
               pos, t)
    (CC / f"dop_{elem}_q{q}.in").write_text(t)
    print(f"wrote dop_{elem}_q{q}.in (tot_charge={q}, short relax from neutral geom)")
