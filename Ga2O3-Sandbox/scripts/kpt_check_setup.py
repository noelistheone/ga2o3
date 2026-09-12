"""R6 reviewer #2 item 5: k-point sampling check for the defect transition level.
Sets up 4 PBE single points on the relaxed 80-atom geometries: V_O q0@R0 and q2@R2, each at
Gamma and at a 2x2x2 Monkhorst-Pack mesh (identical settings otherwise). The reported check is
the k-shift of the (2+/0) level: [ (E_q0 - E_q2)_222 - (E_q0 - E_q2)_Gamma ] / 2."""
import re
from pathlib import Path
import ase.io

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"

base = (CC / "cc_VO_q2_relax.in").read_text()

for q, geomfile, charge in [("q0", "cc_VO_q0_relax.out", 0), ("q2", "cc_VO_q2_relax.out", 2)]:
    atoms = ase.io.read(str(CC / geomfile), format="espresso-out")
    ps = "ATOMIC_POSITIONS angstrom\n" + "".join(
        f"  {s:<3s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n"
        for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()))
    for ktag, kblock in [("gamma", "K_POINTS gamma\n"),
                         ("222", "K_POINTS automatic\n  2 2 2 0 0 0\n")]:
        t = base
        t = t.replace("calculation = 'relax'", "calculation = 'scf'")
        t = re.sub(r"\n\s*nstep = \d+", "", t)
        t = re.sub(r"\n\s*forc_conv_thr = [\d.d-]+", "", t)
        t = re.sub(r"&IONS.*?/\n", "", t, flags=re.S)
        t = t.replace("prefix = 'cc_VO_q2_relax'", f"prefix = 'kpt_VO_{q}_{ktag}'")
        t = t.replace("outdir/cc_VO_q2_relax", f"outdir/kpt_VO_{q}_{ktag}")
        t = t.replace("tot_charge = 2", f"tot_charge = {charge}")
        t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*\n)+", ps, t)
        t = re.sub(r"K_POINTS.*?(?=\n[A-Z&]|\Z)", kblock, t, flags=re.S)
        assert f"kpt_VO_{q}_{ktag}" in t and kblock.splitlines()[0] in t
        (CC / f"kpt_VO_{q}_{ktag}.in").write_text(t)
        print(f"wrote kpt_VO_{q}_{ktag}.in (charge {charge})")
