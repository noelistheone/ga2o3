"""R6 consolidated charged-defect level assembly (file of record for the revised paper numbers).

Collects every ingredient with provenance:
  - HSE total energies: V_O/Sb/Bi x charge {0,1,2} x mixing {0.25, 0.33} (dft/qe_cc/*.out)
  - VBM references: hse_perfect_fixocc (7.7191) / _a033 (7.3249)
  - point-charge corrections: production MP (0.847 for q=2) and exact anisotropic Ewald
    (0.668, results/tier3/efnv_aniso_check.json), scaled by (q/2)^2
  - elastic finite-size (V_O only): -0.288 on E(q2,relaxed) => level -0.144 (fs160_level_shift)
  - k-point check: (E_q0 - E_q2) at Gamma vs 2x2x2, PBE 80-atom (kpt_VO_*)
  - HSE 1-D relaxation scan for V_O q2 (parabola in lam; if all 5 points done)
Writes results/tier3/levels_v2.json."""
import json, re
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
RY = 13.605693122994


def etot(prefix):
    p = CC / f"{prefix}.out"
    if not p.exists():
        return None
    t = p.read_text()
    if "JOB DONE" not in t:
        return None
    m = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", t, re.M)
    return float(m[-1]) * RY if m else None


VBM = {"a025": 7.7191, "a033": 7.3249}
PC = {"MP_prod": 0.847, "aniso_exact": 0.668}          # q=2; scale (q/2)^2
ELASTIC_LEVEL = -0.144                                  # V_O only (2+/0)
KRO = [2.625, 3.353, 3.49]

PREFIX = {
    ("VO", "a025"): {0: "hse_VO_q0", 2: "hse_VO_q2"},
    ("VO", "a033"): {0: "hse_VO_q0_a033", 2: "hse_VO_q2_a033"},
    ("Sb", "a025"): {0: "hse_dop_Sb_q0", 1: "hse_dop_Sb_q1", 2: "hse_dop_Sb_q2"},
    ("Sb", "a033"): {0: "hse_dop_Sb_q0_a033", 1: "hse_dop_Sb_q1_a033", 2: "hse_dop_Sb_q2_a033"},
    ("Bi", "a025"): {0: "hse_dop_Bi_q0", 1: "hse_dop_Bi_q1", 2: "hse_dop_Bi_q2"},
    ("Bi", "a033"): {0: "hse_dop_Bi_q0_a033", 1: "hse_dop_Bi_q1_a033", 2: "hse_dop_Bi_q2_a033"},
}

out = {"provenance": "levels_v2: unified alpha x point-charge x elastic chain (R6)"}
for (defect, mix), pref in PREFIX.items():
    E = {q: etot(p) for q, p in pref.items()}
    if any(v is None for v in E.values()):
        out[f"{defect}_{mix}"] = {"status": "missing", "have": {q: v is not None for q, v in E.items()}}
        continue
    entry = {}
    for pc_name, pc2 in PC.items():
        A = {q: E[q] + q * VBM[mix] + pc2 * (q / 2.0) ** 2 for q in E}
        lv = {"eps_2+/0": round((A[0] - A[2]) / 2.0, 3)}
        if 1 in A:
            lv["eps_2+/+"] = round(A[1] - A[2], 3)
            lv["eps_+/0"] = round(A[0] - A[1], 3)
            lv["U_eV"] = round(lv["eps_+/0"] - lv["eps_2+/+"], 3)
        if defect == "VO":
            lv["eps_2+/0_elastic"] = round(lv["eps_2+/0"] + ELASTIC_LEVEL, 3)
        entry[pc_name] = lv
    out[f"{defect}_{mix}"] = entry

# best-estimate V_O chain vs database
vo = out.get("VO_a033", {}).get("aniso_exact", {})
if "eps_2+/0_elastic" in vo:
    out["VO_best_estimate_vs_DB"] = {
        "chain": "alpha=0.33 + exact anisotropic point charge + 160-atom elastic",
        "eps_eV": vo["eps_2+/0_elastic"],
        "DB_shallowest": KRO[0],
        "residual_eV": round(KRO[0] - vo["eps_2+/0_elastic"], 3),
    }

# k-point check (PBE, 80-atom): shift of the raw (2+/0) level = d[(E_q0-E_q2)]/2
kg0, kg2 = etot("kpt_VO_q0_gamma"), etot("kpt_VO_q2_gamma")
k20, k22 = etot("kpt_VO_q0_222"), etot("kpt_VO_q2_222")
if None not in (kg0, kg2, k20, k22):
    dG, d2 = kg0 - kg2, k20 - k22
    out["kpoint_check_PBE"] = {
        "Eq0_minus_Eq2_gamma_eV": round(dG, 4), "Eq0_minus_Eq2_222_eV": round(d2, 4),
        "level_shift_eV": round((d2 - dG) / 2.0, 4),
        "protocol": "PBE single points on the relaxed 80-atom geometries; identical settings, "
                    "K_POINTS gamma vs 2x2x2 MP; shift applies to the raw (2+/0) level",
    }

# HSE 1-D relaxation scan (V_O q2 along R0->R2 coordinate)
lams, Es = [], []
for lam in (0.85, 0.95, 1.00, 1.05, 1.15):
    e = etot(f"hse_VO_q2_scan_l{int(round(lam*100)):03d}")
    if e is not None:
        lams.append(lam); Es.append(e)
if len(lams) >= 4:
    c = np.polyfit(lams, Es, 2)
    lam_min = float(-c[1] / (2 * c[0]))
    E_min = float(np.polyval(c, lam_min))
    E_100 = float(np.polyval(c, 1.0))
    out["hse_q2_relaxation_scan"] = {
        "lams": lams, "E_eV": [round(e, 4) for e in Es],
        "lam_min": round(lam_min, 3),
        "E_min_minus_E_pbe_geom_eV": round(E_min - E_100, 4),
        "level_shift_eV": round((E_100 - E_min) / 2.0, 4),
        "curvature_ok": bool(c[0] > 0),
        "protocol": "HSE(alpha=0.25) single points along R(lam)=R0+lam(R2-R0); parabola fit; "
                    "the HSE minimum along the PBE relaxation coordinate bounds the "
                    "PBE-geometry bias of the charged state",
    }
else:
    out["hse_q2_relaxation_scan"] = {"status": f"only {len(lams)} points done", "lams": lams}

(PROJ / "results/tier3/levels_v2.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
