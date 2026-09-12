"""Assemble new-dopant (Sb/Bi) charge-transition levels + donor/acceptor classification
from the QE charged-state total energies. Transition levels are chemical-potential
INDEPENDENT (mu cancels between charge states of the same defect) -> rigorous, QE-consistent
classification without needing mu_Sb.

  epsilon(q1/q2) above VBM = [A(q1) - A(q2)] / (q2 - q1),  A(q)=E_tot(D^q)+q*E_VBM+E_corr(q)
  E_corr(q) = Makov-Payne (∝ q^2), E_corr(1)=MP2/4, E_corr(2)=MP2.

Donor if positive charge states are favorable at low E_F and the (+/0) or (2+/0) level sits
in the UPPER gap (near CBM = shallow donor). Writes results/tier3/new_dopant_levels.json.
"""
import json
from pathlib import Path
import re

PROJ = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
CC = PROJ / "dft/qe_cc"
RY = 13.605693

gate2 = json.load(open(PROJ / "results/tier2/gate2_formation_energy.json"))
E_VBM = gate2["E_VBM_eV"]            # 8.9497 eV
MP2 = gate2["makov_payne_q2_eV"]     # 0.847 eV (q=2 monopole)
E_CORR = {0: 0.0, 1: MP2 / 4.0, 2: MP2}
EG = 4.9                             # gate2 gap used


def etot_eV(prefix):
    p = CC / f"{prefix}.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", p.read_text(), re.M)
    return float(e[-1]) * RY if e else None


def analyze(elem):
    E = {0: etot_eV(f"dop_{elem}_relax"), 1: etot_eV(f"dop_{elem}_q1"), 2: etot_eV(f"dop_{elem}_q2")}
    have = {q: v for q, v in E.items() if v is not None}
    if 0 not in have:
        return {"dopant": elem, "status": "neutral not done"}
    A = {q: have[q] + q * E_VBM + E_CORR[q] for q in have}
    levels = {}
    pairs = [(2, 1), (1, 0), (2, 0)]
    for q1, q2 in pairs:
        if q1 in A and q2 in A:
            eps = (A[q1] - A[q2]) / (q2 - q1)      # above VBM
            levels[f"{q1}+/{q2 if q2 else '0'}"] = {
                "above_VBM_eV": round(eps, 3),
                "below_CBM_eV": round(EG - eps, 3),
            }
    # classification: the transition into neutral from the highest available + state
    cls = "undetermined"
    key = "2+/0" if "2+/0" in levels else ("1+/0" if "1+/0" in levels else None)
    if key:
        below_cbm = levels[key]["below_CBM_eV"]
        if below_cbm < 0.3:
            cls = f"SHALLOW DONOR ({key} at CBM-{below_cbm:.2f} eV)"
        elif below_cbm < EG / 2:
            cls = f"deep donor ({key} at CBM-{below_cbm:.2f} eV)"
        else:
            cls = f"deep level / weak donor ({key} at CBM-{below_cbm:.2f} eV)"
    return {"dopant": elem, "E_tot_eV": {str(k): round(v, 3) for k, v in have.items()},
            "transition_levels": levels, "classification": cls,
            "note": "mu-independent (rigorous). Absolute conc needs a QE mu_M metal ref "
                    "(queued separately). Potential-alignment term for charged cells not "
                    "yet applied (refinement) — levels are first-pass."}


def main():
    out = {"E_VBM_eV": E_VBM, "makov_payne_q2_eV": MP2, "Eg_eV": EG, "dopants": {}}
    for elem in ["Sb", "Bi"]:
        r = analyze(elem)
        out["dopants"][elem] = r
        print(f"[{elem}] {r.get('classification', r.get('status'))}")
        for k, v in r.get("transition_levels", {}).items():
            print(f"    e({k}) = VBM+{v['above_VBM_eV']} = CBM-{v['below_CBM_eV']} eV")
    (PROJ / "results/tier3/new_dopant_levels.json").write_text(json.dumps(out, indent=2))
    print("WROTE results/tier3/new_dopant_levels.json")


if __name__ == "__main__":
    main()
