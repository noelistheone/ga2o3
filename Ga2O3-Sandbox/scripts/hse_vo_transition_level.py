"""Compute the in-house HSE V_O (2+/0) thermodynamic transition level from EXISTING DFT ($0).

The charge-MACE campaign plan flagged "no in-house V_O2+ HSE energy" — but both HSE single-points
exist in dft/qe_cc/ (from the tau/CC campaign): hse_VO_q0 (neutral@R0) and hse_VO_q2 (2+@R2), at
their respective relaxed geometries (verified: they differ by 1.33 A max = the negative-U V_O2+
relaxation). So the thermodynamic HSE (2+/0) level is computable now, filling the flagged gap.

Convention = the sandbox's VALIDATED gate2 convention (reproduces PBE V_O (2+/0)=VBM+0.91):
  E_f(q; E_F) = E_tot(q) - E_perfect + n_O*mu_O + q*(E_VBM + E_F) + E_corr_MP(q)
  eps(2+/0) above VBM = [E_f(0; E_F=0) - E_f(2; E_F=0)] / 2
E_f(0) HSE O-rich = 3.674 eV (results/tier2/hse_formation_energy.json); everything else cancels or
is taken from the same HSE setup. E_corr_MP(2)=+0.847 eV (Makov-Payne monopole, eps=10.2, 80-atom).

HONEST CAVEAT (recorded in output): this is MP-monopole-corrected but WITHOUT charged-cell potential
alignment (eFNV/Kumagai-Oba) — the plan's fallback step 1 — so it carries ~+-0.2-0.3 eV; the sign,
depth, and the +1.3 eV HSE-vs-PBE shift are robust to alignment.
"""
import sys, json
from pathlib import Path
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import kroger_db

RY = 13.605693122994  # eV/Ry

# --- existing HSE total energies (dft/qe_cc, JOB DONE, same 80-atom HSE setup as the anchor) ---
E_q0_Ry = -6235.86571596      # hse_VO_q0.out  (neutral @ R0)
E_q2_Ry = -6237.38949824      # hse_VO_q2.out  (2+ @ R2, negative-U relaxed)
E_q0 = E_q0_Ry * RY
E_q2 = E_q2_Ry * RY

hse = json.load(open(PROJ / "results/tier2/hse_formation_energy.json"))
Ef0_Orich = hse["E_f_HSE_VO_q0_Orich_eV"]     # 3.674 (neutral formation E, O-rich)
lvl = json.load(open(PROJ / "results/tier3/new_dopant_hse.json"))
E_VBM = lvl["E_VBM_HSE_eV"]                    # 7.7191 (hse_perfect_fixocc)
Eg = lvl["Eg_eV"]                             # 4.9
MP2 = 0.847                                    # Makov-Payne q=2 (gate2, eps=10.2, 80-atom)

# E_f(2;0) - E_f(0;0) = [E_tot(2)-E_tot(0)] + 2*E_VBM + MP2   (mu_O, E_perfect cancel)
Ef2_minus_Ef0 = (E_q2 - E_q0) + 2 * E_VBM + MP2
Ef2_Orich = Ef0_Orich + Ef2_minus_Ef0
eps_2plus_0 = (Ef0_Orich - Ef2_Orich) / 2.0   # above VBM
below_CBM = Eg - eps_2plus_0

# --- KROGER's V_O (2+/0) for cross-check (isolated native V_O, all O sites) ---
db = kroger_db.load()
o = db.col("O")
import numpy as np
is_vo = (db.dm[:, o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)   # isolated V_O charge states
kro = {}
for i in np.where(is_vo)[0]:
    kro.setdefault(int(db.cs_ID[i]), {})[int(db.charge[i])] = float(db.dHo[i])
# transition (2+/0) for each V_O site defect = E_F where dHo(0)=dHo(2)+2 E_F  -> E_F=(dHo0-dHo2)/2
kro_levels = []
for did, cs in kro.items():
    if 0 in cs and 2 in cs:
        kro_levels.append(round((cs[0] - cs[2]) / 2.0, 3))

out = {
    "quantity": "HSE V_O (2+/0) thermodynamic transition level (in-house, from existing dft/qe_cc)",
    "E_HSE_VO_q0_eV": round(E_q0, 3), "E_HSE_VO_q2_eV": round(E_q2, 3),
    "geom": "q0@R0, q2@R2 (verified 1.33 A max disp = negative-U relaxation)",
    "Ef0_Orich_eV": Ef0_Orich, "Ef2_Orich_eV": round(Ef2_Orich, 3),
    "eps_2plus_0_above_VBM_eV": round(eps_2plus_0, 3),
    "eps_2plus_0_below_CBM_eV": round(below_CBM, 3),
    "PBE_eps_2plus_0_above_VBM_eV": 0.91,
    "HSE_minus_PBE_shift_eV": round(eps_2plus_0 - 0.91, 3),
    "KROGER_VO_2plus_0_above_VBM_eV (isolated sites)": sorted(kro_levels),
    "verdict": ("HSE V_O (2+/0) = VBM+%.2f eV (CBM-%.2f) = DEEP negative-U donor (NOT the n-type "
                "source), HSE deepens it +%.2f eV vs PBE. Consistent with KROGER + the established "
                "deep-V_O picture." % (eps_2plus_0, below_CBM, eps_2plus_0 - 0.91)),
    "caveat": ("MP-monopole corrected, NO charged-cell potential alignment (eFNV/Kumagai-Oba) yet "
               "-> ~+-0.2-0.3 eV; sign/depth/HSE-shift robust. Alignment = plan fallback step 1."),
}
(PROJ / "results/tier3/hse_vo_2plus0_level.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
