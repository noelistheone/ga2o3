"""Build a QE-PBE scf single-point at the MACE-relaxed V_O geometry (fixed).
Energy vs the full QE-relaxed V_O isolates the ENERGY cost of the MACE-vs-QE
geometry difference -> validates MLIP-relax + QE/HSE single-point (Tier B).
Preserves the CELL_PARAMETERS card; only swaps the ATOMIC_POSITIONS block and
switches to scf.
"""
import re
from pathlib import Path
import ase.io

CC = Path("/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc")
src = (CC / "cc_VO_q0_relax.in").read_text()
mace = ase.io.read(str(CC / "vo_q0_mace_relaxed.xyz"))

t = src
t = t.replace("calculation = 'relax'", "calculation = 'scf'")
t = re.sub(r"\n\s*nstep\s*=.*", "", t)
t = re.sub(r"\n\s*forc_conv_thr\s*=.*", "", t)
t = re.sub(r"prefix = '[^']+'", "prefix = 'vo_q0_scf_at_mace'", t)
t = re.sub(r"outdir = '[^']+'",
           "outdir = '/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc/outdir/vo_q0_scf_at_mace'", t)
t = re.sub(r"&IONS.*?/\n", "", t, flags=re.S)   # drop ions namelist

# swap ONLY the ATOMIC_POSITIONS card (keep CELL_PARAMETERS etc.)
pos = "ATOMIC_POSITIONS angstrom\n"
for s, p in zip(mace.get_chemical_symbols(), mace.positions):
    pos += f"  {s:<4s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}\n"
t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+",
           pos, t)

(CC / "vo_q0_scf_at_mace.in").write_text(t)
print("wrote", CC / "vo_q0_scf_at_mace.in")
print("has CELL_PARAMETERS:", "CELL_PARAMETERS" in t, " nat lines:", pos.count("\n") - 1)
