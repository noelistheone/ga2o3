"""General CC-diagram processor for any defect: parse the two relaxed charge-state geometries,
compute ΔQ, generate the cross single-point inputs. Usage:
    python cc_process_general.py <defect> <qA> <qB>   e.g. VGa 0 -3   or   FeGa 0 -1
Relax outputs expected: cc_<defect>_q<qA>_relax.out, cc_<defect>_q<qB>_relax.out
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
MASS = {"Ga": 69.723, "O": 15.999, "Fe": 55.845, "Si": 28.086, "Sn": 118.71}
RY = 13.605693


def parse_relaxed(prefix):
    p = CC / f"{prefix}.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    txt = p.read_text()
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", txt, re.M)
    blocks = re.findall(r"ATOMIC_POSITIONS.*?\n((?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)", txt)
    if not blocks or not e:
        return None
    syms, coords = [], []
    for line in blocks[-1].strip().split("\n"):
        pr = line.split()
        syms.append(pr[0]); coords.append([float(pr[1]), float(pr[2]), float(pr[3])])
    return {"E_eV": float(e[-1]) * RY, "symbols": syms, "coords": np.array(coords)}


def main():
    defect, qA, qB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    a = parse_relaxed(f"cc_{defect}_q{qA}_relax")
    b = parse_relaxed(f"cc_{defect}_q{qB}_relax")
    if not (a and b):
        print(json.dumps({"waiting": {"qA": a is not None, "qB": b is not None}}))
        return
    dr = b["coords"] - a["coords"]
    m = np.array([MASS[s] for s in a["symbols"]])
    dQ = float(np.sqrt(np.sum(m[:, None] * dr**2)))
    res = {"defect": defect, "qA": qA, "qB": qB, "dQ_amu_half_A": round(dQ, 3),
           "max_disp_A": round(float(np.linalg.norm(dr, axis=1).max()), 3),
           f"E_q{qA}_at_R{qA}": round(a["E_eV"], 3), f"E_q{qB}_at_R{qB}": round(b["E_eV"], 3)}

    # generate cross single-points from a relax template of this defect
    tmpl = (CC / f"cc_{defect}_q{qA}_relax.in").read_text()
    header = tmpl.split("ATOMIC_SPECIES")[0].replace("calculation = 'relax'", "calculation = 'scf'")
    header = re.sub(r"\n\s*nstep.*", "", header)
    header = re.sub(r"\n\s*forc_conv_thr.*", "", header)
    species = "ATOMIC_SPECIES" + tmpl.split("ATOMIC_SPECIES")[1].split("ATOMIC_POSITIONS")[0]
    cell = "CELL_PARAMETERS" + tmpl.split("CELL_PARAMETERS")[1]

    def write_sp(prefix, geom, q):
        pos = "ATOMIC_POSITIONS angstrom\n" + "\n".join(
            f"  {s:4s} {c[0]:16.8f} {c[1]:16.8f} {c[2]:16.8f}" for s, c in zip(geom["symbols"], geom["coords"]))
        h = re.sub(r"prefix = '[^']+'", f"prefix = '{prefix}'", header)
        h = re.sub(r"tot_charge = -?\d+", f"tot_charge = {q}", h)
        h = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / prefix}'", h)
        (CC / f"{prefix}.in").write_text(h + species + pos + "\nK_POINTS automatic\n1 1 1 0 0 0\n" + cell)
        return prefix

    p1 = write_sp(f"cc_{defect}_q{qA}_at_R{qB}", b, qA)   # qA electrons at R_qB geometry
    p2 = write_sp(f"cc_{defect}_q{qB}_at_R{qA}", a, qB)   # qB electrons at R_qA geometry
    res["cross_singlepoints"] = [p1, p2]
    (PROJ / f"results/tier2/cc_{defect}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nlaunch: bash scripts/run_cc_qe.sh {p1} 1 ; bash scripts/run_cc_qe.sh {p2} 1")


if __name__ == "__main__":
    main()
