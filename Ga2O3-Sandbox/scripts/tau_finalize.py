"""Finalize the V_O τ: assemble the CC diagram from the 4 total energies (2 relax minima + 2
cross single-points), extract the Franck-Condon reorganization energies, and run the NMP→τ chain.

CC-diagram energies (all from dft/qe_cc/, eV):
  E00 = E_q0(R0)  neutral at neutral minimum      (cc_VO_q0_relax)
  E22 = E_q2(R2)  charged at charged minimum       (cc_VO_q2_relax)
  E02 = E_q0(R2)  neutral electrons, charged geom   (cc_VO_q0_at_R2)   Franck-Condon
  E20 = E_q2(R0)  charged electrons, neutral geom   (cc_VO_q2_at_R0)   Franck-Condon

Reorganization energies:
  λ0 = E02 − E00  (neutral relaxes from R2→R0)
  λ2 = E20 − E22  (charged relaxes from R0→R2)
  λ  = ½(λ0 + λ2) effective one-mode reorganization energy for the 2+/0 capture.
Driving force ΔE: the (2+/0) thermodynamic transition level below the conduction band — taken
from the KROGER V_O energetics (deep, negative-U); the CC barrier is (λ+ΔE)²/4λ.
"""
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import tau_nmp  # noqa: E402

CC = PROJ / "dft/qe_cc"
RY = 13.605693


def scf_energy(out):
    p = Path(out)
    if not p.exists():
        return None
    txt = p.read_text()
    if "JOB DONE" not in txt:
        return None
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", txt, re.M)
    return float(e[-1]) * RY if e else None


E00 = scf_energy(CC / "cc_VO_q0_relax.out")
E22 = scf_energy(CC / "cc_VO_q2_relax.out")
E02 = scf_energy(CC / "cc_VO_q0_at_R2.out")
E20 = scf_energy(CC / "cc_VO_q2_at_R0.out")

status = {"E00": E00, "E22": E22, "E02": E02, "E20": E20}
missing = [k for k, v in status.items() if v is None]
if missing:
    print(json.dumps({"waiting_for": missing, "have": status}, indent=2))
    sys.exit(0)

lam0 = E02 - E00      # neutral relaxation energy
lam2 = E20 - E22      # charged relaxation energy
lam = 0.5 * (lam0 + lam2)

# ΔQ from the post-process
ccj = json.loads((PROJ / "results/tier2/tau_cc.json").read_text())
dQ = ccj["dQ_amu_half_A"]

# V_O (2+/0) transition level below Ec — KROGER/Varley: deep negative-U donor, ~1.0-2.0 eV below
# Ec. For the capture driving force we use the level below the CB (electron capture from CB tail).
dE_level_below_Ec = 1.0   # eV (representative; from the KROGER V_O (2+/0) negative-U level)

# N_trap: representative V_O density in a reduced film (~1e18)
N_trap = 1e18

tau = tau_nmp.tau_from_cc(dQ=dQ, lam=abs(lam), dE=dE_level_below_Ec, N_trap_cm3=N_trap)

result = {
    "cc_diagram_eV": {"E00": round(E00, 3), "E22": round(E22, 3), "E02": round(E02, 3),
                      "E20": round(E20, 3)},
    "reorg_energy_neutral_lam0_eV": round(lam0, 3),
    "reorg_energy_charged_lam2_eV": round(lam2, 3),
    "effective_reorg_lambda_eV": round(lam, 3),
    "dQ_amu_half_A": dQ,
    "driving_force_dE_eV": dE_level_below_Ec,
    "N_trap_cm3": N_trap,
    **tau,
    "defect": "V_O (2+/0) negative-U — dominant PPC/τ deep trap",
    "note": "first first-principles CC→NMP→τ for a native Ga2O3 defect in THIS sandbox; "
            "order-of-magnitude / class-ranking fidelity (τ was the property with no prior "
            "forward prediction). PBE CC; HSE refinement + full Alkauskas matrix element = next tier.",
}
(PROJ / "results/tier2/tau_VO_result.json").write_text(json.dumps(result, indent=2, default=str))
print(json.dumps(result, indent=2, default=str))
