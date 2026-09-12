"""R6 reviewer confrontation: run the solver at the EXPERIMENTAL Ta-doped beta-Ga2O3 conditions
(OFZ single crystals, mol%-level Ta, air-like ambients) and locate the compensation cliff in the
(concentration, pO2) plane.

Experimental record (to be quoted with verified refs): OFZ Ta-doped crystals with n controllably
tuned ~3.6e16 -> 3e19 cm^-3 via Ta content; shallow-donor behavior (C-V to 20 K); 0.05 mol% Ta
substrate mu=138 cm2/Vs @ 7e17 cm^-3. The paper's cliff panel used 0.5 at.% (cation fraction
5e-3 ~ 1.9e20 cm^-3 nominal) at pO2 1e-4..1e-5 atm, film mode. Question: does the solver
reproduce controlled shallow-donor behavior at the experimental (10-1000x lower) loadings and
air-like pO2, with the cliff confined to the (high-loading, O-poor) corner?
"""
import sys, json
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db
from sandbox.kroger_db import N_SITE

db = kroger_db.load()
N_cat = N_SITE * 2.0 / 3.0

out = {"N_cat_cm3": N_cat, "protocol": {
    "bulk_crystal": "film=False (no grain-boundary term), Nd_bg=1e16 (UID melt crystal), "
                    "Tf=1500 K (certified engine global; 1350 K sensitivity)",
    "cliff_map": "grid over cation fraction x pO2, film=True as in the paper's Fig. mech c",
}}

# ---- 1) n(Ta) at OFZ-like conditions across the experimental loading range ----
fracs = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3]
for tag, Tf, pO2, Nd_bg in [("OFZ_air_Tf1500", 1500.0, 0.2, 1e16),
                            ("OFZ_air_Tf1350", 1350.0, 0.2, 1e16),
                            ("OFZ_O2poor_Tf1500", 1500.0, 1e-4, 1e16)]:
    rows = []
    for f in fracs:
        r = ff.forward("Ta", f, T_anneal=Tf, pO2=pO2, film=False, Nd_bg=Nd_bg,
                       db=db, undoped_cache={})
        NTa = f * N_cat
        rows.append({"frac": f, "Ta_cm3": float(f"{NTa:.3g}"),
                     "n_cm3": float(f"{r['hall_n_cm3']:.3g}"),
                     "mu_cm2Vs": round(r["hall_mu_cm2Vs"], 1),
                     "activation": float(f"{r['hall_n_cm3']/NTa:.3g}")})
        print(tag, f, rows[-1]["n_cm3"], rows[-1]["activation"], flush=True)
    out[tag] = rows

# ---- 2) cliff map: where does the runaway compensation live? ----
grid_f = [1e-5, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3]
grid_p = [-1, -2, -3, -4, -5, -6, -7, -8]
cliff = {"fracs": grid_f, "log_pO2": grid_p, "log_n": []}
for f in grid_f:
    row = []
    for lp in grid_p:
        r = ff.forward("Ta", f, T_anneal=1350.0, pO2=10.0 ** lp, film=True, Nd_bg=1e17,
                       db=db, undoped_cache={})
        row.append(round(float(np.log10(max(r["hall_n_cm3"], 1.0))), 2))
    cliff["log_n"].append(row)
    print("map frac", f, row, flush=True)
out["cliff_map_Tf1350_film"] = cliff

# cliff location per loading: max |dlogn/dlogp| and the pO2 where it occurs
loc = []
for f, row in zip(grid_f, cliff["log_n"]):
    d = np.abs(np.diff(row))
    i = int(np.argmax(d))
    loc.append({"frac": f, "max_jump_decades": float(d[i]),
                "between_log_pO2": [grid_p[i], grid_p[i + 1]]})
out["cliff_location"] = loc

# ---- 3) control: Si at the same states (is nominal-air collapse Ta-specific or generic?) ----
si = {}
for tag, pO2 in [("Si_air_Tf1500", 0.2), ("Si_O2poor_Tf1500", 1e-4)]:
    rows = []
    for f in [1e-4, 1e-3, 5e-3]:
        r = ff.forward("Si", f, T_anneal=1500.0, pO2=pO2, film=False, Nd_bg=1e16,
                       db=db, undoped_cache={})
        rows.append({"frac": f, "n_cm3": float(f"{r['hall_n_cm3']:.3g}"),
                     "activation": float(f"{r['hall_n_cm3']/(f*N_cat):.3g}")})
    si[tag] = rows
    print(tag, rows, flush=True)
out["Si_control"] = si

(PROJ / "results/tier2/ta_experimental_conditions.json").write_text(json.dumps(out, indent=1))
print(json.dumps(loc, indent=1))
print("wrote results/tier2/ta_experimental_conditions.json")
