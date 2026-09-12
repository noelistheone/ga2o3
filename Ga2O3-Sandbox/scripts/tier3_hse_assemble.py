"""Assemble the HSE transition levels for the new dopants (Sb, Bi) and re-run the 9-property
prediction with HSE energetics — the accuracy tier that removes the PBE-gap-shift.

  e(2+/0) above VBM = [A(2)-A(0)]/(0-2),  A(q) = E_HSE(D^q) + q*E_VBM_HSE + E_corr(q)
HSE VBM from hse_perfect_fixocc ('highest occupied level'). Re-predicts with HSE dHo (n is
offset-invariant, so the activation reflects the HSE transition level). Writes
results/tier3/new_dopant_hse.json.
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
MP2 = g2["makov_payne_q2_eV"]
EG = 4.9
T_ANNEAL, PO2, CONC = 1350.0, 1e-4, 0.01


def etot(prefix):
    p = CC / f"{prefix}.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", p.read_text(), re.M)
    return float(e[-1]) * RY if e else None


def hse_vbm():
    p = CC / "hse_perfect_fixocc.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    m = re.findall(r"highest occupied.*?([-\d.]+)\s*(?:ev)?\s*$", p.read_text(), re.M | re.I)
    if not m:
        m = re.findall(r"highest occupied,\s*lowest unoccupied level.*?:\s*([-\d.]+)", p.read_text(), re.I)
    return float(m[-1]) if m else None


def analyze(elem, E_VBM_HSE, E_perf_hse):
    E = {0: etot(f"hse_dop_{elem}_q0"), 2: etot(f"hse_dop_{elem}_q2")}
    if E[0] is None or E[2] is None or E_VBM_HSE is None:
        return {"dopant": elem, "status": "HSE not done", "have": {k: v is not None for k, v in E.items()}}
    A = {q: E[q] + q * E_VBM_HSE + MP2 * (q / 2.0) ** 2 for q in E}
    eps = (A[2] - A[0]) / (0 - 2)              # e(2+/0) above VBM
    # dHo (HSE) for the prediction (offset-invariant n)
    dHo = {q: E[q] - E_perf_hse + q * E_VBM_HSE + MP2 * (q / 2.0) ** 2 for q in E}
    off = 3.0 - dHo[0]
    cs = {q: dHo[q] + off for q in dHo}
    db2 = new_dopant.extend_with_dopant(kroger_db.load(), elem, cs)
    fwd = full_forward.forward(elem, CONC, T_anneal=T_ANNEAL, pO2=PO2, db=db2, Nd_bg=1e17)
    keep = ["hall_n_cm3", "sigma_S_cm", "hall_mu_cm2Vs", "Eg_optical_eV", "dEg_eV",
            "dark_activation_eV", "EF_300K_below_Ec", "solubility_limited"]
    return {"dopant": elem,
            "e_2plus_0_above_VBM_eV": round(eps, 3),
            "e_2plus_0_below_CBM_eV": round(EG - eps, 3),
            "classification_HSE": ("SHALLOW DONOR" if EG - eps < 0.3 else
                                   "deep donor" if EG - eps < EG / 2 else "deep/inactive"),
            "nine_properties_HSE": {k: fwd[k] for k in keep}}


def main():
    vbm = hse_vbm()
    E_perf_hse = etot("gate2_perfect_hse")
    out = {"E_VBM_HSE_eV": vbm, "E_perfect_HSE_eV": E_perf_hse, "Eg_eV": EG,
           "note": "HSE@PBE-geom transition levels — removes the PBE-gap-shift. Compare to the "
                   "PBE result (new_dopant_prediction.json) where both looked artificially deep.",
           "dopants": {}}
    for elem in ["Sb", "Bi"]:
        r = analyze(elem, vbm, E_perf_hse)
        out["dopants"][elem] = r
        if "nine_properties_HSE" in r:
            p = r["nine_properties_HSE"]
            print(f"[{elem}] HSE e(2+/0)=CBM-{r['e_2plus_0_below_CBM_eV']} -> {r['classification_HSE']} "
                  f"| n={p['hall_n_cm3']:.2e} sigma={p['sigma_S_cm']:.2e} EF(CBM-)={p['EF_300K_below_Ec']:.3f}")
        else:
            print(f"[{elem}] {r.get('status')}")
    (PROJ / "results/tier3/new_dopant_hse.json").write_text(json.dumps(out, indent=2))
    print("WROTE results/tier3/new_dopant_hse.json")


if __name__ == "__main__":
    main()
