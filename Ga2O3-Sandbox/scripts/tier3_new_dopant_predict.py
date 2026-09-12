"""Close the new-dopant loop: QE charge-state energies -> dHo -> extend KROGER DB ->
full 9-property forward prediction for a dopant OUTSIDE KROGER's 19 (Sb, Bi).

Reads E_tot(q) from dop_<M>_relax.out (q0), dop_<M>_q1.out, dop_<M>_q2.out; converts to
dHo (KROGER convention) with the Gate-2 references (E_perfect, E_VBM, Makov-Payne); injects
via src/sandbox/new_dopant; runs src/sandbox/full_forward at a representative sputter freeze-in.
Writes results/tier3/new_dopant_prediction.json.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from sandbox import kroger_db, new_dopant, full_forward

PROJ = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
CC = PROJ / "dft/qe_cc"
RY = 13.605693

g2 = json.load(open(PROJ / "results/tier2/gate2_formation_energy.json"))
E_PERFECT = g2["E_perfect_eV"]
E_VBM = g2["E_VBM_eV"]
MP2 = g2["makov_payne_q2_eV"]

# representative sputter film process (donor freeze-in regime; matches absolute_accuracy_path)
T_ANNEAL, PO2, CONC = 1350.0, 1e-4, 0.01
METAL_NAT = {"Ga": 8, "Sb": 6, "Bi": 6}


def etot_eV(prefix):
    p = CC / f"{prefix}.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", p.read_text(), re.M)
    return float(e[-1]) * RY if e else None


def mu_metal(elem):
    """mu_M^0 = E(metal)/atom [eV] (absolute, same pseudo/ecut). None if not done."""
    E = etot_eV(f"metal_{elem}")
    return None if E is None else E / METAL_NAT[elem]


def predict(elem):
    E_tot = {0: etot_eV(f"dop_{elem}_relax"), 1: etot_eV(f"dop_{elem}_q1"),
             2: etot_eV(f"dop_{elem}_q2")}
    E_tot = {q: v for q, v in E_tot.items() if v is not None}
    if 0 not in E_tot:
        return {"dopant": elem, "status": "neutral not done", "have_charges": list(E_tot)}
    raw = new_dopant.dHo_from_qe(E_tot, E_PERFECT, E_VBM, MP2)   # absolute (gate2 convention)
    cls = new_dopant.classify(raw)                                # offset-independent (differences)
    # KROGER-convention reference offset = mu_Ga^0 - mu_M^0 (elemental metals). cs_dHo(q0) is then
    # the ABSOLUTE formation energy at the Ga-metal/M-metal corner. Doping n is offset-invariant.
    mu_Ga0, mu_M0 = mu_metal("Ga"), mu_metal(elem)
    if mu_Ga0 is not None and mu_M0 is not None:
        offset = mu_Ga0 - mu_M0
        referenced = True
    else:
        offset = 3.0 - raw[0]        # nominal fallback (n is offset-invariant; E_f^0 not meaningful)
        referenced = False
    dHo = {q: raw[q] + offset for q in raw}
    Ef0_neutral = dHo.get(0)         # absolute formation energy at Ga/M-rich corner (if referenced)
    db = kroger_db.load()
    db2 = new_dopant.extend_with_dopant(db, elem, dHo)
    fwd = full_forward.forward(elem, CONC, T_anneal=T_ANNEAL, pO2=PO2, db=db2, Nd_bg=1e17)
    keep = ["hall_n_cm3", "hall_p_cm3", "hall_mu_cm2Vs", "sigma_S_cm", "Eg_optical_eV",
            "dEg_eV", "V_O_cm3_quench", "dark_activation_eV", "PDR_score", "tau_score",
            "EF_300K_below_Ec", "solubility_limited"]
    return {"dopant": elem,
            "referenced_to_metals": referenced,
            "Ef0_neutral_GaM_rich_eV": None if not referenced else round(Ef0_neutral, 3),
            "cs_dHo_KROGER_conv_eV": {str(k): round(v, 3) for k, v in dHo.items()},
            "E_tot_charges_done": list(E_tot),
            "classification": cls["classification"], "transition_levels": cls["transition_levels"],
            "process": {"T_anneal": T_ANNEAL, "pO2": PO2, "conc_frac": CONC},
            "nine_properties": {k: fwd[k] for k in keep}}


def main():
    out = {"references": {"E_perfect_eV": E_PERFECT, "E_VBM_eV": E_VBM, "makov_payne_q2_eV": MP2},
           "note": "New dopants beyond KROGER-19 via MACE-MPA-0 structure + QE energetics (Tier B). "
                   "dHo(q) drives donor/acceptor + compensation; absolute n is exp-sensitivity-"
                   "limited (project rankings/trends scope). Charged geoms are short-relax from "
                   "the neutral MACE geom; potential-alignment for charged cells is a refinement.",
           "dopants": {}}
    for elem in ["Sb", "Bi"]:
        r = predict(elem)
        out["dopants"][elem] = r
        if "nine_properties" in r:
            np_ = r["nine_properties"]
            print(f"[{elem}] {r['classification']}")
            print(f"    n={np_['hall_n_cm3']:.2e} sigma={np_['sigma_S_cm']:.2e} "
                  f"mu={np_['hall_mu_cm2Vs']:.1f} Eg={np_['Eg_optical_eV']:.3f} "
                  f"dEg={np_['dEg_eV']:+.3f} EF(CBM-)={np_['EF_300K_below_Ec']:.2f}")
        else:
            print(f"[{elem}] {r.get('status')}")
    (PROJ / "results/tier3/new_dopant_prediction.json").write_text(json.dumps(out, indent=2))
    print("WROTE results/tier3/new_dopant_prediction.json")


if __name__ == "__main__":
    main()
