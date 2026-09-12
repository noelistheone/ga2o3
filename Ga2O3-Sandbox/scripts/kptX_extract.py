"""R7 plan B extraction: zone-boundary (1/2,1/2,1/2) single-k dispersion of the Sb (q0)-(q2)
energy difference at PBE and HSE tiers. The HSE/PBE ratio measures how much the hybrid's
localization shrinks the defect-band image dispersion; applied to the full PBE 2x2x2 shift
(+0.6357 eV) it gives the measured hybrid-tier k-axis estimate. Writes
results/tier3/kptX_hse_check.json."""
import json, re
from pathlib import Path

P = Path(__file__).resolve().parents[1]
CC = P / "dft/qe_cc"
RY = 13.605693


def efinal(name):
    t = (CC / f"{name}.out").read_text()
    m = re.findall(r"^!!?\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", t, re.M)
    assert m, f"{name}: no converged energy"
    return float(m[-1]) * RY


pbe_G = efinal("kpt_Sb_q0_gamma") - efinal("kpt_Sb_q2_gamma")
pbe_X = efinal("kptX_pbe_Sb_q0") - efinal("kptX_pbe_Sb_q2")
hse_G = efinal("hse_dop_Sb_q0") - efinal("hse_dop_Sb_q2")
hse_X = efinal("kptX_hse_Sb_q0") - efinal("kptX_hse_Sb_q2")

shift_pbe = pbe_X - pbe_G
shift_hse = hse_X - hse_G
ratio = shift_hse / shift_pbe if abs(shift_pbe) > 1e-6 else None
full_pbe_222 = 0.6357   # levels_v2.json kpoint_check_PBE.dopants.Sb.level_shift
out = {
    "k_point": "(0.5, 0.5, 0.5) crystal",
    "protocol": "single-k SCF at the zone-boundary point vs Gamma, identical settings per tier "
                "(PBE: 80/480 Ry; HSE alpha=0.25, Fock q at Gamma-equivalent single q); "
                "dispersion of the (q0)-(q2) total-energy difference",
    "PBE_D_gamma_eV": round(pbe_G, 4), "PBE_D_X_eV": round(pbe_X, 4),
    "HSE_D_gamma_eV": round(hse_G, 4), "HSE_D_X_eV": round(hse_X, 4),
    "shift_X_PBE_eV": round(shift_pbe, 4), "shift_X_HSE_eV": round(shift_hse, 4),
    "HSE_over_PBE_ratio": round(ratio, 3) if ratio is not None else None,
    "PBE_full_222_shift_eV": full_pbe_222,
    "implied_HSE_222_shift_eV": round(ratio * full_pbe_222, 3) if ratio is not None else None,
}
(P / "results/tier3/kptX_hse_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
