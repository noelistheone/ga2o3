"""File-of-record for the CORRECTED elastic finite-size accounting on the V_O (2+/0) level,
plus the alpha=0.33 V_O level (if the a033 HSE single-points are done).

Accounting chain (R5 review fix — the prior narrative had the DIRECTION of the elastic shift
reversed):
  Production eps(2+/0)=2.223 eV uses E(q2@R2) relaxed in the 80-atom cell.
  The 160-atom re-relaxation gives Erelax_q2 = 2.049 vs 2.337 eV at 80 atoms: the small cell
  OVER-relaxes by 0.287 eV (elastic image assistance of the 1.3-A distortion). The vertical
  reference E(q2@R0) is converged across cells (0.019 eV, finite_size_160.json). Therefore the
  dilute-limit E(q2, relaxed) is HIGHER by 0.287 eV -> E_f(2+) rises by 0.287 -> the level
  eps = [E_f(0) - E_f(2+; E_F=0)]/2 DROPS by 0.287/2 = 0.144 eV: 2.223 -> 2.080, moving AWAY
  from the database band (2.625/3.353/3.49), residual to the nearest anchor 0.545 eV.
CC-diagram consistency: substituting Erelax_160 for the charged-state reorganization energy at
fixed driving force gives lambda_eff (1.241+2.049)/2 = 1.645 eV and capture barrier
(lambda+dE)^2/(4 lambda) = 1.063 eV (was 1.789 / 1.087) — class ranking vs V_Ga unchanged.
alpha=0.33 (database-style gap-tuned mixing): same geometries/protocol, exx_fraction=0.33,
VBM from hse_perfect_fixocc_a033 (7.3249 eV), same MP monopole (+0.847 eV, same dielectric).
"""
import json, re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
RY = 13.605693122994

lvl = json.load(open(PROJ / "results/tier3/hse_vo_2plus0_level.json"))
r4 = json.load(open(PROJ / "results/tier3/r4_supplements.json"))
tau = json.load(open(PROJ / "results/tier2/tau_VO_result.json"))
kro = sorted(lvl["KROGER_VO_2plus_0_above_VBM_eV (isolated sites)"])

Er80, Er160 = r4["Erelax_q2_80"], r4["Erelax_q2_160"]
d_relax = Er160 - Er80                       # -0.287: dilute E(q2,relaxed) higher by 0.287
eps_raw = lvl["eps_2plus_0_above_VBM_eV"]    # 2.223 (80-atom q2@R2)
shift = d_relax / 2.0                        # -0.144: level moves DOWN
eps_corr = round(eps_raw + shift, 3)         # 2.080
residual = round(kro[0] - eps_corr, 3)       # 0.545 to the shallowest database site

lam0 = tau["reorg_energy_neutral_lam0_eV"]   # 1.241
dE = tau["driving_force_dE_eV"]              # 1.0
lam160 = round((lam0 + Er160) / 2.0, 3)      # 1.645
barrier160 = round((lam160 + dE) ** 2 / (4 * lam160), 3)   # 1.063 (was 1.087)

out = {
    "quantity": "corrected elastic finite-size accounting for V_O (2+/0); alpha=0.33 level",
    "Erelax_q2_80_eV": Er80, "Erelax_q2_160_eV": Er160, "elastic_component_eV": round(d_relax, 3),
    "direction": "80-atom cell OVER-relaxes (image-assisted); dilute E(q2,relaxed) HIGHER; "
                 "E_f(2+) rises; level DROPS — away from the database band",
    "eps_raw_a025_eV": eps_raw,
    "level_shift_eV": round(shift, 3),
    "eps_elastic_corrected_a025_eV": eps_corr,
    "database_sites_eV": kro,
    "residual_to_shallowest_site_eV": residual,
    "cc_consistency": {
        "lambda_eff_80_eV": tau["effective_reorg_lambda_eV"],
        "capture_barrier_80_eV": tau["capture_barrier_eV"],
        "lambda_eff_160_eV": lam160, "capture_barrier_160_fixed_dE_eV": barrier160,
        "note": "barrier formula (lambda+dE)^2/(4 lambda), driving force held at the 80-atom "
                "value; class ranking vs V_Ga (barrier 0.89, dQ 3.4) unchanged",
    },
}

# ---- alpha=0.33 level (if the dual-GPU single-points are done) ----
def last_total_energy_eV(path):
    t = path.read_text()
    if "JOB DONE" not in t:
        return None
    m = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", t, re.M)
    return float(m[-1]) * RY if m else None

q0f, q2f = CC / "hse_VO_q0_a033.out", CC / "hse_VO_q2_a033.out"
E_q0 = last_total_energy_eV(q0f) if q0f.exists() else None
E_q2 = last_total_energy_eV(q2f) if q2f.exists() else None
if E_q0 is not None and E_q2 is not None:
    VBM033 = r4["VBM_alpha033"]              # 7.3249 (hse_perfect_fixocc_a033)
    MP2 = 0.847                              # same monopole correction (same dielectric)
    eps033 = round(((E_q0 - E_q2) - 2 * VBM033 - MP2) / 2.0, 3)
    out["alpha033"] = {
        "E_HSE_VO_q0_a033_eV": round(E_q0, 3), "E_HSE_VO_q2_a033_eV": round(E_q2, 3),
        "VBM_alpha033_eV": VBM033, "MP2_eV": MP2,
        "eps_2plus_0_above_VBM_a033_eV": eps033,
        "eps_elastic_corrected_a033_eV": round(eps033 + shift, 3),
        "residual_to_shallowest_site_a033_eV": round(kro[0] - (eps033 + shift), 3),
        "alpha_response_eV": round(eps033 - eps_raw, 3),
        "Sb_alpha_response_eV": round(r4["eps_Sb_alpha033_above_VBM"] - r4["eps_Sb_alpha025_check"], 3),
        "protocol": "same 80-atom geometries (q0@R0, q2@R2), exx_fraction=0.33, "
                    "gap-matched-mixing VBM reference, same MP monopole",
    }
else:
    out["alpha033"] = "PENDING (hse_VO_q0_a033 / hse_VO_q2_a033 not finished)"

(PROJ / "results/tier3/fs160_level_shift.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
