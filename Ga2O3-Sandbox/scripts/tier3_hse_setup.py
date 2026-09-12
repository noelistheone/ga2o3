"""Set up the HSE transition-level campaign for the new dopants (Sb, Bi) — the rigorous
accuracy tier that fixes the PBE-gap-shift making the dopants look artificially deep.

HSE single-points (input_dft=hse, HSE@PBE-geom) at the PBE-relaxed charge-state geometries:
  hse_dop_<M>_q0  (neutral geom, tot_charge=0)
  hse_dop_<M>_q2  (q2 geom, tot_charge=2)
+ hse_perfect_fixocc (HSE perfect, fixed occupations -> clean VBM for the transition level).
Charged cells keep smearing (total energy); only the perfect cell needs fixed-occ for the VBM.
All dual-GPU (run_cc_qe_dual.sh).
"""
import re
from pathlib import Path
import ase.io

CC = Path("/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc")
HSE = ("    input_dft = 'hse'\n    nqx1 = 1\n    nqx2 = 1\n    nqx3 = 1\n"
       "    exxdiv_treatment = 'gygi-baldereschi'\n")


def relaxed_positions(prefix):
    a = ase.io.read(str(CC / f"{prefix}.out"), format="espresso-out")
    pos = "ATOMIC_POSITIONS angstrom\n"
    for s, p in zip(a.get_chemical_symbols(), a.get_positions()):
        pos += f"  {s:<4s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}\n"
    return pos


def make_hse_defect(src_prefix, dst_prefix, q):
    t = (CC / f"{src_prefix}.in").read_text()
    t = t.replace("calculation = 'relax'", "calculation = 'scf'")
    t = t.replace("calculation = 'vc-relax'", "calculation = 'scf'")
    t = re.sub(r"\n\s*nstep\s*=.*", "", t)
    t = re.sub(r"\n\s*forc_conv_thr\s*=.*", "", t)
    t = re.sub(r"&IONS.*?/\n", "", t, flags=re.S)
    t = re.sub(r"prefix = '[^']+'", f"prefix = '{dst_prefix}'", t)
    t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / dst_prefix}'", t)
    t = re.sub(r"tot_charge = [-\d.]+", f"tot_charge = {q}", t)
    t = re.sub(r"(tot_charge = [-\d.]+\n)", r"\1" + HSE, t)
    t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+",
               relaxed_positions(src_prefix), t)
    (CC / f"{dst_prefix}.in").write_text(t)
    print(f"wrote {dst_prefix}.in (HSE @ {src_prefix} geom, tot_charge={q})")


# defect single-points
for M in ["Sb", "Bi"]:
    make_hse_defect(f"dop_{M}_relax", f"hse_dop_{M}_q0", 0)
    make_hse_defect(f"dop_{M}_q2", f"hse_dop_{M}_q2", 2)

# HSE perfect with fixed occupations -> clean VBM. Perfect Ga2O3 is a non-magnetic closed-shell
# insulator -> nspin=1 (fixed-occ + nspin=2 needs tot_magnetization and fails here).
t = (CC / "gate2_perfect_hse.in").read_text()
t = re.sub(r"prefix = '[^']+'", "prefix = 'hse_perfect_fixocc'", t)
t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / 'hse_perfect_fixocc'}'", t)
t = re.sub(r"occupations = '[^']+'", "occupations = 'fixed'", t)
t = re.sub(r"\n\s*smearing\s*=.*", "", t)
t = re.sub(r"\n\s*degauss\s*=.*", "", t)
t = re.sub(r"nspin = 2", "nspin = 1", t)
t = re.sub(r"\n\s*starting_magnetization\(\d+\)\s*=.*", "", t)
if "input_dft" not in t:                       # ensure HSE present
    t = re.sub(r"(tot_charge = [-\d.]+\n)", r"\1" + HSE, t)
(CC / "hse_perfect_fixocc.in").write_text(t)
print("wrote hse_perfect_fixocc.in (HSE perfect, fixed occ, nspin=1 -> VBM)")
