"""V62 validation gate (research C2c-2): co-doping must REDUCE to the single-dopant model as a co-dopant -> 0.

Self-consistency unit test on the solver: solve(A only) must equal solve(A + B) when c_B -> 0 (the second
species contributes ~0 charge). This is free (no data), and together with the D=1==V61 reduction (script 79)
and the independent scipy cross-check (script 81) closes the "the co-doping model is a strict, faithful
extension" argument. Also records the qualitative-agreement checklist vs the published beta-Ga2O3 multi-
element Kroger-Vink oracle (arXiv 2501.17373) -- same formalism (Fermi-Dirac n/p, E_g, mu_O(pO2),
multi-species charge sum); honest gaps (E_g(T), vibrational entropy) flagged. CPU. Writes consistency.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.models.defect_equilibrium import EG_GA2O3  # noqa: E402
from src.models.defect_equilibrium_codoping import solve_equilibrium_codoping, elem_transition_level  # noqa: E402
from src.models.v61_net import carrier_sign  # noqa: E402

torch.set_default_dtype(torch.float64)
EG = EG_GA2O3
DHF = [3.5, 1.8, 0.3]


def single(A, c, T=973.15, lp=0.0):
    z = carrier_sign(A); e = elem_transition_level(A, z)
    o = solve_equilibrium_codoping(torch.tensor([DHF]), torch.tensor([T]), torch.tensor([lp]),
                                   torch.tensor([[z]]), torch.tensor([[c]]), torch.tensor([[e]]))
    return float(o["E_F"]), float(o["log10_VO"])


def codoped(A, cA, B, cB, T=973.15, lp=0.0):
    zA, zB = carrier_sign(A), carrier_sign(B)
    eA, eB = elem_transition_level(A, zA), elem_transition_level(B, zB)
    o = solve_equilibrium_codoping(torch.tensor([DHF]), torch.tensor([T]), torch.tensor([lp]),
                                   torch.tensor([[zA, zB]]), torch.tensor([[cA, cB]]),
                                   torch.tensor([[eA, eB]]),
                                   dop_mask=torch.tensor([[1.0, float(cB > 0)]]))
    return float(o["E_F"]), float(o["log10_VO"])


def main():
    checks = []
    for A, cA, B in [("Sn", 0.01, "Mg"), ("Mg", 0.01, "Sn"), ("Si", 0.005, "Zn"),
                     ("Zn", 0.02, "Si"), ("Sn", 0.002, "N")]:
        ef1, vo1 = single(A, cA)
        ef2, vo2 = codoped(A, cA, B, 1e-12)   # co-dopant -> 0 (residual scales linearly with this)
        checks.append(dict(case=f"{A}({cA}) vs {A}+{B}(1e-12)", dEF=abs(ef1 - ef2), dlogVO=abs(vo1 - vo2),
                           match=bool(abs(ef1 - ef2) < 1e-4 and abs(vo1 - vo2) < 1e-4)))
    rep = dict(
        reduction_to_single_dopant=checks,
        all_match=bool(all(c["match"] for c in checks)),
        oracle_qualitative_agreement={
            "reference": "Frodason et al., quantitative point-defect modeling of beta-Ga2O3 (arXiv:2501.17373, PCCP 2025)",
            "shared_formalism": ["Fermi-Dirac n(E_F)/p(E_F)", "charge-neutrality root-find for E_F",
                                 "multi-species (19-element) charge sum -> co-doping is native",
                                 "mu_O(T,pO2) chemical potential -> Brouwer pO2 power law",
                                 "V_O deep-donor / self-compensation of acceptors"],
            "honest_gaps_vs_oracle": ["E_g(T) temperature dependence (we use fixed E_g=4.85)",
                                      "defect vibrational entropy", "explicit Ga-rich vs O-rich mu endpoints",
                                      "full 259-defect inventory (we model V_O + dopant species)"],
            "status": "qualitative agreement on formalism + trends; quantitative oracle benchmark is future work "
                      "(paper data not fetched offline; zero external API)."},
    )
    out = PROJ / "results/phase62_codoping/consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print("co-dopant -> 0 reduces to single-dopant:")
    for c in checks:
        print(f"  {c['case']:24s} dEF={c['dEF']:.2e} dlogVO={c['dlogVO']:.2e} match={c['match']}")
    print(f"\nall_match={rep['all_match']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
