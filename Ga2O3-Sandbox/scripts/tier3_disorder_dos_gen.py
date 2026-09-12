"""Generate QE-PBE scf inputs (2x2x2 k, extra empty bands, 50 meV smearing — research recipe)
for the disorder ensemble cells + the crystalline reference, for band-edge / Urbach extraction.
Identical settings for crystal and amorphous so PBE systematic error cancels in the SHIFT.

Usage: tier3_disorder_dos_gen.py            (all ensemble cells + crystal ref)
Writes dft/qe_cc/dos_<name>.in
"""
import re
from pathlib import Path
import ase.io

PROJ = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
CC = PROJ / "dft/qe_cc"
ENS = PROJ / "dft/disorder_ensemble"
MASS = {"Ga": 69.723, "O": 15.999}


def write_dos(atoms, name):
    uniq = []
    for s in atoms.get_chemical_symbols():
        if s not in uniq:
            uniq.append(s)
    base = (CC / "cc_VO_q0_relax.in").read_text()
    t = base
    t = t.replace("calculation = 'relax'", "calculation = 'scf'")
    t = re.sub(r"\n\s*nstep\s*=.*", "", t)
    t = re.sub(r"\n\s*forc_conv_thr\s*=.*", "", t)
    t = re.sub(r"&IONS.*?/\n", "", t, flags=re.S)
    t = re.sub(r"prefix = '[^']+'", f"prefix = 'dos_{name}'", t)
    t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / ('dos_' + name)}'", t)
    t = re.sub(r"nat = \d+", f"nat = {len(atoms)}", t)
    t = re.sub(r"ntyp = \d+", f"ntyp = {len(uniq)}", t)
    t = re.sub(r"degauss = [\d.]+", "degauss = 0.0037", t)      # ~50 meV
    t = re.sub(r"nspin = 2", "nspin = 1", t)                    # non-magnetic -> clean gap
    t = re.sub(r"\n\s*starting_magnetization\(\d+\)\s*=.*", "", t)
    t = re.sub(r"verbosity = '[^']+'", "verbosity = 'high'", t)  # print all eigenvalues
    if "verbosity" not in t:
        t = t.replace("tot_charge = 0", "tot_charge = 0\n    verbosity = 'high'")
    # disk_io='none' -> no multi-GB wavefunction dump (we only need the eigenvalues)
    t = re.sub(r"prefix = '([^']+)'", r"prefix = '\1'\n    disk_io = 'none'", t)
    if "nbnd" not in t:
        # nelec=704 -> 352 occupied; 500 gives ~150 conduction bands for the DOS above CBM
        t = t.replace("tot_charge = 0", "tot_charge = 0\n    nbnd = 500")
    spec = "ATOMIC_SPECIES\n" + "".join(f"  {s:<4s} {MASS[s]:9.3f}  {s}.upf\n" for s in uniq)
    t = re.sub(r"ATOMIC_SPECIES\n(?:\s*[A-Z][a-z]?\s+[\d.]+\s+\S+\.upf\s*\n)+", spec, t)
    lat = atoms.cell[:]
    cell = "CELL_PARAMETERS angstrom\n" + "".join(
        f"  {lat[i][0]:14.9f} {lat[i][1]:14.9f} {lat[i][2]:14.9f}\n" for i in range(3))
    pos = "ATOMIC_POSITIONS angstrom\n" + "".join(
        f"  {s:<4s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}\n"
        for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()))
    t = re.sub(r"CELL_PARAMETERS[^\n]*\n(?:\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+", cell, t)
    t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+", pos, t)
    # K_POINTS 2x2x2 (append; remove any existing gamma card first)
    t = re.sub(r"K_POINTS[^\n]*\n(?:.*\n)?", "", t)
    if not t.endswith("\n"):
        t += "\n"
    t += "K_POINTS automatic\n  2 2 2 0 0 0\n"
    (CC / f"dos_{name}.in").write_text(t)


def main():
    # crystalline reference
    write_dos(ase.io.read(str(CC / "gate2_perfect.in"), format="espresso-in"), "crystal")
    n = 1
    for xyz in sorted(ENS.glob("cell_*.xyz")):
        name = xyz.stem                       # cell_0, cell_1, ...
        write_dos(ase.io.read(str(xyz)), name)
        n += 1
    print(f"wrote {n} DOS inputs (dos_crystal + {n-1} amorphous cells)")


if __name__ == "__main__":
    main()
