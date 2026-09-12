"""Regenerate HSE single-point inputs on the PBE-RELAXED geometries (HSE@PBE-geom, standard
approx) for V_O q0/q2 + O2, to complete the HSE formation energy (better energetics tier).
"""
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
MASS = {"Ga": 69.723, "O": 15.999}


def relaxed_geom(prefix):
    txt = (CC / f"{prefix}.out").read_text()
    blocks = re.findall(r"ATOMIC_POSITIONS.*?\n((?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)", txt)
    return blocks[-1].strip()


HSE = "    input_dft = 'hse'\n    nqx1 = 1\n    nqx2 = 1\n    nqx3 = 1\n    exxdiv_treatment = 'gygi-baldereschi'\n"

for src, dst, q in [("cc_VO_q0_relax", "hse_VO_q0", 0), ("cc_VO_q2_relax", "hse_VO_q2", 2)]:
    t = (CC / f"{src}.in").read_text()
    t = t.replace("calculation = 'relax'", "calculation = 'scf'")
    t = re.sub(r"\n\s*nstep.*", "", t)
    t = re.sub(r"\n\s*forc_conv_thr.*", "", t)
    t = re.sub(r"prefix = '[^']+'", f"prefix = '{dst}'", t)
    t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / dst}'", t)
    t = re.sub(r"(tot_charge = -?\d+\n)", r"\1" + HSE, t)
    # replace positions with relaxed
    geom = relaxed_geom(src)
    t = re.sub(r"ATOMIC_POSITIONS angstrom\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+",
              "ATOMIC_POSITIONS angstrom\n" + geom + "\n", t)
    (CC / f"{dst}.in").write_text(t)
    print(f"wrote {dst}.in (HSE single-point on PBE-relaxed geometry, q={q})")

# HSE O2 (for mu_O at HSE) — reuse gate2_O2 geom with HSE
o2 = (CC / "gate2_O2.in").read_text()
o2 = o2.replace("gate2_O2", "hse_O2")
o2 = re.sub(r"(nspin = 2\n)", r"\1" + HSE, o2)
(CC / "hse_O2.in").write_text(o2)
print("wrote hse_O2.in (HSE O2 for μ_O at HSE)")
