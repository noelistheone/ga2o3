"""R8 TAU-08 (blocking): rebuild the capture chain as SEQUENTIAL single-carrier steps.
Per step (qA -> qB, one electron): dQ from the two relaxed geometries (mass-weighted),
per-state relaxation energies lam_A = E(qA@R_B)-E(qA@R_A), lam_B = E(qB@R_A)-E(qB@R_B),
effective lam = mean, Marcus barrier E_b = (lam+dE)^2/(4 lam) with the driving force dE from
the database transition level below the CB (electron capture; identical convention to the
concerted-chain files of record). Steps:
  V_O:  (2+/+) then (+/0)         [the negative-U rate question]
  V_Ga: (0/-), (-/2-), (2-/3-)    [the acceptor chain]
Writes results/tier3/r8_sequential_capture.json.
"""
import json, re, sys
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
RY = 13.605693
MASS = {"Ga": 69.723, "O": 15.999}
sys.path.insert(0, str(PROJ / "src"))
from sandbox import kroger_db

db = kroger_db.load()


def efinal(name):
    m = re.findall(r"^!!?\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry",
                   (CC / f"{name}.out").read_text(), re.M)
    return float(m[-1]) * RY


def geom(name):
    txt = (CC / f"{name}.out").read_text()
    blocks = re.findall(r"ATOMIC_POSITIONS \(?angstrom\)?\n"
                        r"((?:\s*\S+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)", txt)
    syms, xyz = [], []
    for ln in blocks[-1].strip().split("\n"):
        p = ln.split()
        syms.append(p[0]); xyz.append([float(x) for x in p[1:4]])
    return syms, np.array(xyz)


def dQ_of(nameA, nameB):
    sA, xA = geom(nameA)
    sB, xB = geom(nameB)
    m = np.array([MASS[s] for s in sA])
    return float(np.sqrt(np.sum(m[:, None] * (xA - xB) ** 2)))


def db_level(defect_tag, qA, qB):
    """Transition level eps(qA/qB) above VBM from the KROGER database (dHo per charge state)."""
    if defect_tag == "VO":
        o = db.col("O")
        mask = (db.dm[:, o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
    else:
        g = db.col("Ga")
        mask = (db.dm[:, g] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
    idx = np.where(mask)[0]
    dho = {int(db.charge[i]): float(db.dHo[i]) for i in idx}
    return (dho[qB] - dho[qA]) / (qA - qB)


EG = 4.9   # database gap convention for the CB reference (as in the concerted files)
STEPS = [
    ("VO", 2, 1, "electron"), ("VO", 1, 0, "electron"),
    ("VGa", 0, -1, "electron"), ("VGa", -1, -2, "electron"), ("VGa", -2, -3, "electron"),
]
out = {"protocol": "sequential single-carrier Marcus steps; dE = database level below CB "
                   "(electron capture), lam = mean per-state relaxation energy from the cross "
                   "single points, barrier = (lam+dE)^2/(4 lam) with dE<0 for downhill capture "
                   "convention matched to the concerted files",
       "steps": []}
for tag, qA, qB, kind in STEPS:
    rA = f"cc_{tag}_q{qA}_relax"
    rB = f"cc_{tag}_q{qB}_relax"
    xAB = f"cc_{tag}_q{qA}_at_R{qB}"
    xBA = f"cc_{tag}_q{qB}_at_R{qA}"
    lamA = efinal(xAB) - efinal(rA)
    lamB = efinal(xBA) - efinal(rB)
    lam = 0.5 * (lamA + lamB)
    dq = dQ_of(rA, rB)
    eps = db_level(tag, qA, qB)
    dE = -(EG - eps)                     # electron capture from the CB: downhill by (Ec - eps)
    Eb = (lam + dE) ** 2 / (4 * lam) if lam > 0.05 else None
    rec = {"defect": tag, "step": f"({qA:+d}/{qB:+d})", "dQ_amu_half_A": round(dq, 2),
           "lam_A_eV": round(lamA, 3), "lam_B_eV": round(lamB, 3), "lam_eff_eV": round(lam, 3),
           "eps_above_VBM_dbconv_eV": round(eps, 3), "dE_capture_eV": round(dE, 3),
           "barrier_eV": None if Eb is None else round(Eb, 3),
           "regime": "strong-coupling" if lam >= 0.2 else "weak-coupling (excluded)"}
    out["steps"].append(rec)
    print(rec, flush=True)

vo = [s for s in out["steps"] if s["defect"] == "VO"]
vga = [s for s in out["steps"] if s["defect"] == "VGa"]
out["rate_limiting"] = {
    "VO": max((s for s in vo if s["barrier_eV"] is not None), key=lambda s: s["barrier_eV"])["step"],
    "VGa": max((s for s in vga if s["barrier_eV"] is not None), key=lambda s: s["barrier_eV"])["step"],
}
out["class_ranking_preserved"] = (
    max(s["barrier_eV"] or 0 for s in vo) > max(s["barrier_eV"] or 0 for s in vga))
(PROJ / "results/tier3/r8_sequential_capture.json").write_text(json.dumps(out, indent=1))
print("rate-limiting:", out["rate_limiting"], "| VO slower than VGa:",
      out["class_ranking_preserved"])
