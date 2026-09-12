"""Compute the HSE neutral V_O formation energy (better-energetics tier) and compare to PBE +
KROGER-HSE. E_f_HSE(V_O,q0) = E_HSE(V_O,q0) − E_HSE(perfect) + μ_O_HSE, μ_O_HSE = ½E_HSE(O2)
(O-rich). Runs when the HSE calcs are done."""
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
RY = 13.605693


def E(prefix):
    p = CC / f"{prefix}.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", p.read_text(), re.M)
    return float(e[-1]) * RY if e else None


E_perf = E("gate2_perfect_hse")
E_vo0 = E("hse_VO_q0")
E_o2 = E("hse_O2")
miss = [n for n, v in [("perfect_hse", E_perf), ("VO_q0_hse", E_vo0), ("O2_hse", E_o2)] if v is None]
if miss:
    print(json.dumps({"waiting_for": miss}))
    raise SystemExit

mu_O_hse = E_o2 / 2.0                       # O-rich μ_O at HSE (elemental O2 ref)
ef_hse_q0 = E_vo0 - E_perf + mu_O_hse       # E_f at μ_O = O-rich elemental (μ=0 ref → +½E(O2))
# NOTE: this uses the O2-elemental (O-rich) reference; the KROGER dHo is at that same O-rich limit
result = {
    "E_HSE_perfect_eV": round(E_perf, 3),
    "E_HSE_VO_q0_eV": round(E_vo0, 3),
    "mu_O_HSE_eV(half_E_O2)": round(mu_O_hse, 3),
    "E_f_HSE_VO_q0_Orich_eV": round(ef_hse_q0, 3),
    "comparison": {
        "PBE_VO_q0_Orich": 3.09,
        "HSE_this_work_Orich": round(ef_hse_q0, 3),
        "KROGER_HSE_dHo_q0": "~3.5 (O-rich-consistent anchor)",
    },
    "note": "HSE neutral V_O formation energy on the PBE-relaxed geometry (HSE@PBE-geom). If it "
            "lands nearer the KROGER HSE ~3.5 eV than PBE's 3.09, the HSE tier tightens the "
            "energetics as expected. Charged HSE E_f needs an HSE VBM (fixed-occ HSE perfect) — "
            "next refinement. Even tightened, absolutes stay order-of-magnitude (ceiling).",
}
(PROJ / "results/tier2/hse_formation_energy.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
