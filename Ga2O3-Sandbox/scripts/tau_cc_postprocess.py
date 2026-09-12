"""tau CC-diagram post-processing — compute the configuration-coordinate diagram + NMP capture
from the relaxed V_O q=0 and q=+2 geometries, and generate the cross single-point inputs needed
to complete it. Run after both cc_VO_q{0,2}_relax.out reach JOB DONE.

Steps:
  1. Parse final relaxed geometries + energies from the two relax outputs.
  2. Compute the configuration coordinate ΔQ = sqrt(Σ m_i |Δr_i|²) (amu^½·Å) between them.
  3. Generate cross single-point QE inputs: q=0 electronic state AT the q=2 geometry, and q=2 AT
     the q=0 geometry → the four corners of the CC diagram (E_q(R_q'), harmonic parabolas).
  4. Once those run, compute Franck-Condon relaxation energies, the classical capture barrier
     ΔE_b (Marcus/NMP), and an order-of-magnitude capture coefficient / τ estimate.
This script does steps 1-3 now; step 4 runs after the cross single-points complete.
"""
import re
import sys
from pathlib import Path

import numpy as np

SANDBOX = Path(__file__).resolve().parents[1]
CC = SANDBOX / "dft/qe_cc"
MASS = {"Ga": 69.723, "O": 15.999}


def parse_relaxed(out_file):
    txt = Path(out_file).read_text()
    if "JOB DONE" not in txt:
        return None
    # final energy (last '!' line)
    e = [float(m) for m in re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", txt, re.M)]
    energy_Ry = e[-1] if e else None
    # final coordinates: last "ATOMIC_POSITIONS" block (from 'Begin final coordinates' if present)
    blocks = re.findall(r"ATOMIC_POSITIONS.*?\n((?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)", txt)
    if not blocks:
        return None
    syms, coords = [], []
    for line in blocks[-1].strip().split("\n"):
        p = line.split()
        syms.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return {"energy_Ry": energy_Ry, "energy_eV": energy_Ry * 13.605693 if energy_Ry else None,
            "symbols": syms, "coords": np.array(coords)}


def main():
    q0 = parse_relaxed(CC / "cc_VO_q0_relax.out")
    q2 = parse_relaxed(CC / "cc_VO_q2_relax.out")
    status = {"q0_done": q0 is not None, "q2_done": q2 is not None}
    if not (q0 and q2):
        print("Waiting for relaxations:", status)
        if q0:
            print(f"  q0 energy = {q0['energy_eV']:.3f} eV ({len(q0['symbols'])} atoms)")
        return status

    dr = q2["coords"] - q0["coords"]
    masses = np.array([MASS[s] for s in q0["symbols"]])
    dQ = float(np.sqrt(np.sum(masses[:, None] * dr**2)))  # amu^½ Å
    max_disp = float(np.linalg.norm(dr, axis=1).max())
    result = {**status, "dQ_amu_half_A": round(dQ, 3),
              "max_atomic_disp_A": round(max_disp, 3),
              "E_q0_at_R0_eV": round(q0["energy_eV"], 3),
              "E_q2_at_R2_eV": round(q2["energy_eV"], 3),
              "note": "cross single-points (q0@R2, q2@R0) generate the CC parabolas; barrier + NMP "
                      "capture follow"}

    # generate cross single-point inputs (q=0 electrons at R_q2 geometry, and q=2 at R_q0)
    tmpl = (CC / "cc_VO_q0_relax.in").read_text()
    header = tmpl.split("ATOMIC_SPECIES")[0].replace("calculation = 'relax'", "calculation = 'scf'")
    header = re.sub(r"\n\s*nstep.*", "", header)
    header = re.sub(r"\n\s*forc_conv_thr.*", "", header)
    species = "ATOMIC_SPECIES" + tmpl.split("ATOMIC_SPECIES")[1].split("ATOMIC_POSITIONS")[0]
    cell = "CELL_PARAMETERS" + tmpl.split("CELL_PARAMETERS")[1]

    def write_sp(prefix, geom, q):
        pos = "ATOMIC_POSITIONS angstrom\n" + "\n".join(
            f"  {s:4s} {c[0]:16.8f} {c[1]:16.8f} {c[2]:16.8f}" for s, c in zip(geom["symbols"], geom["coords"]))
        h = header.replace(re.search(r"prefix = '([^']+)'", header).group(1), prefix)
        h = re.sub(r"tot_charge = \d+", f"tot_charge = {q}", h)
        h = h.replace("outdir = '" + re.search(r"outdir = '([^']+)'", h).group(1) + "'",
                      f"outdir = '{CC / 'outdir' / prefix}'")
        (CC / f"{prefix}.in").write_text(h + species + pos + "\nK_POINTS automatic\n1 1 1 0 0 0\n" + cell)
        return prefix

    p1 = write_sp("cc_VO_q0_at_R2", q2, 0)   # neutral electrons, distorted (q2) geometry
    p2 = write_sp("cc_VO_q2_at_R0", q0, 2)   # charged electrons, neutral (q0) geometry
    result["cross_singlepoints_generated"] = [p1, p2]
    import json
    (SANDBOX / "results/tier2/tau_cc.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nLaunch cross single-points:\n  bash scripts/run_cc_qe.sh {p1} 0\n  bash scripts/run_cc_qe.sh {p2} 1")
    return result


if __name__ == "__main__":
    main()
