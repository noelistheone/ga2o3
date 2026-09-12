"""R6 reviewer #2 item 4: does the PBE geometry bias the HSE level for the large-relaxation
V_O 2+ state? 1-D HSE scan along the PBE relaxation coordinate: geometries R(lam) = R0 +
lam*(R2-R0) for lam in {0.85, 0.95, 1.00, 1.05, 1.15}; HSE (alpha=0.25, production settings)
single points; parabola fit gives the HSE-level minimum lam* and the energy correction
E_HSE(lam*) - E_HSE(1.0) to the charged relaxed state (and hence to the (2+/0) level, /2)."""
import re
from pathlib import Path
import ase.io

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"

r0 = ase.io.read(str(CC / "cc_VO_q0_relax.out"), format="espresso-out")
r2 = ase.io.read(str(CC / "cc_VO_q2_relax.out"), format="espresso-out")
d = r2.get_positions() - r0.get_positions()
base = (CC / "hse_VO_q2.in").read_text()   # production HSE charged single-point settings

LAMS = [0.85, 0.95, 1.00, 1.05, 1.15]
for lam in LAMS:
    tag = f"l{int(round(lam*100)):03d}"
    a = r0.copy()
    a.set_positions(r0.get_positions() + lam * d)
    ps = "ATOMIC_POSITIONS angstrom\n" + "".join(
        f"  {s:<3s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n"
        for s, p in zip(a.get_chemical_symbols(), a.get_positions()))
    t = base
    t = t.replace("prefix = 'hse_VO_q2'", f"prefix = 'hse_VO_q2_scan_{tag}'")
    t = t.replace("outdir/hse_VO_q2", f"outdir/hse_VO_q2_scan_{tag}")
    t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*\n)+", ps, t)
    assert f"hse_VO_q2_scan_{tag}" in t
    (CC / f"hse_VO_q2_scan_{tag}.in").write_text(t)
    print(f"wrote hse_VO_q2_scan_{tag}.in (lam={lam})")
print("note: lam=1.00 duplicates hse_VO_q2 geometry (consistency check, cheap)")
