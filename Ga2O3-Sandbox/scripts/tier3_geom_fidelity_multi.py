"""Tier-3 geometry-fidelity gate across ALL existing QE-relaxed defects.
For each defect (neutral + charged), MACE-MPA-0 relaxes from the same unrelaxed start
(fixed QE cell); compare to the QE-relaxed geometry. Small RMSD -> MLIP can replace the
QE ionic relaxation for that defect (Tier B). Charged defects test the charge-geometry
coupling the research flagged (MACE is charge-blind -> a large RMSD on a charged state
means that defect needs an HSE final relax, not MLIP).

Writes results/tier3/geom_fidelity_multi.json.
"""
import json
from pathlib import Path
import numpy as np
import ase.io
from ase.optimize import FIRE

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
OUT = PROJ / "results/tier3"
OUT.mkdir(parents=True, exist_ok=True)
FMAX, STEPS = 0.02, 500

DEFECTS = [
    ("V_O",   0,  "cc_VO_q0_relax"),
    ("V_O",  +2,  "cc_VO_q2_relax"),
    ("V_Ga",  0,  "cc_VGa_q0_relax"),
    ("V_Ga", -3,  "cc_VGa_q-3_relax"),
    ("Fe_Ga", 0,  "cc_FeGa_q0_relax"),
    ("Fe_Ga",-1,  "cc_FeGa_q-1_relax"),
]


def mace_calc():
    from mace.calculators import mace_mp
    return mace_mp(model="/home/lawrence/.cache/mace/macempa0mediummodel",
                   device="cuda", default_dtype="float64")


def min_image_disp(a, b, cell):
    d = b - a
    frac = np.linalg.solve(cell.T, d.T).T
    frac -= np.round(frac)
    return frac @ cell


def main():
    calc = mace_calc()
    rows = []
    for name, q, prefix in DEFECTS:
        inf, outf = CC / f"{prefix}.in", CC / f"{prefix}.out"
        if not outf.exists() or "JOB DONE" not in outf.read_text():
            rows.append({"defect": name, "q": q, "skip": "no QE relax"}); continue
        init = ase.io.read(str(inf), format="espresso-in")
        qe = ase.io.read(str(outf), format="espresso-out")
        cell = np.array(init.cell)
        a = init.copy(); a.calc = calc
        FIRE(a, logfile=None).run(fmax=FMAX, steps=STEPS)
        disp = np.linalg.norm(min_image_disp(np.array(qe.positions), np.array(a.positions), cell), axis=1)
        d_qe = np.linalg.norm(min_image_disp(np.array(init.positions), np.array(qe.positions), cell), axis=1)
        d_mace = np.linalg.norm(min_image_disp(np.array(init.positions), np.array(a.positions), cell), axis=1)
        rmsd = float(np.sqrt((disp ** 2).mean()))
        rows.append({
            "defect": name, "q": q,
            "rmsd_A": round(rmsd, 4),
            "max_disp_A": round(float(disp.max()), 4),
            "max_relax_QE_A": round(float(d_qe.max()), 4),
            "max_relax_MACE_A": round(float(d_mace.max()), 4),
            "pass_0.1A": bool(rmsd < 0.1),
        })
        print(f"{name:6s} q={q:+d}: RMSD={rmsd:.4f} A  max={disp.max():.3f}  "
              f"QE-relax-max={d_qe.max():.3f}  {'PASS' if rmsd<0.1 else 'FAIL(>0.1)'}", flush=True)
    result = {"fmax": FMAX, "model": "MACE-MPA-0 (medium 9.06M)", "defects": rows,
              "note": "RMSD < 0.1 A => MLIP-relax can replace QE relaxation for that defect "
                      "(Tier B). Larger RMSD on a charged state => charge-geometry coupling => "
                      "needs HSE final relax."}
    (OUT / "geom_fidelity_multi.json").write_text(json.dumps(result, indent=2))
    print("WROTE", OUT / "geom_fidelity_multi.json")


if __name__ == "__main__":
    main()
