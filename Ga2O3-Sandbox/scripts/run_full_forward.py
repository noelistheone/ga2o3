"""Run the integrated 9-property forward simulator over the full dopant set + save the table."""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff  # noqa: E402
from sandbox import kroger_db  # noqa: E402

db = kroger_db.load()
dopants = [e for e in db.elements if e not in ("Ga", "O")]
CONC = 0.005   # 0.5 at% cation fraction (dilute, in Lany's optimal window)

# undoped reference once
und = ff.forward("undoped", 0.0, db=db)
cache = {"n_und": und["hall_n_cm3"]}

rows = [("undoped", und)]
for X in dopants:
    try:
        r = ff.forward(X, CONC, db=db, undoped_cache=cache)
        rows.append((X, r))
    except Exception as e:
        rows.append((X, {"error": str(e)[:120]}))

table = {"conditions": {"conc_frac": CONC, "T_anneal": 1073.0, "pO2": 1e-4, "film": True,
                        "E_B_eV": 0.02, "note": "pO2 & E_B are placeholder nuisance params; "
                        "Tier-2 validation fits them on lab paired intrinsics"},
         "fidelity": und["fidelity"],
         "rows": {X: (r if "error" in r else {
             "hall_n_cm3": f"{r['hall_n_cm3']:.2e}",
             "V_O_cm3": f"{r['V_O_cm3_quench']:.2e}",
             "hall_mu_cm2Vs": round(r["hall_mu_cm2Vs"], 1),
             "sigma_S_cm": f"{r['sigma_S_cm']:.2e}",
             "Eg_eV": round(r["Eg_optical_eV"], 3),
             "dEg_eV": round(r["dEg_eV"], 4),
             "dark_act_eV": round(r["dark_activation_eV"], 3),
             "PDR_score": round(r["PDR_score"], 2),
             "tau_score": round(r["tau_score"], 2),
             "EF_belowEc": round(r["EF_300K_below_Ec"], 3),
             "solub_lim": r["solubility_limited"],
         }) for X, r in rows}}

(PROJ / "results/tier2").mkdir(parents=True, exist_ok=True)
(PROJ / "results/tier2/full_forward_table.json").write_text(json.dumps(table, indent=2))

# also save the full detailed result for undoped + Si + Sn + Mg (lab dopants) as examples
detail = {X: r for X, r in rows if X in ("undoped", "Si", "Sn", "Mg", "Zn")}
(PROJ / "results/tier2/full_forward_detail_labdopants.json").write_text(json.dumps(detail, indent=2, default=str))

# print table
hdr = f"{'dopant':8s} {'n':>9s} {'mu':>6s} {'sigma':>9s} {'Eg':>6s} {'dEg':>7s} {'dark_Ea':>7s} {'PDR':>6s} {'tau':>5s}"
print(hdr)
for X, r in rows:
    if "error" in r:
        print(f"{X:8s} ERROR {r['error']}"); continue
    print(f"{X:8s} {r['hall_n_cm3']:9.1e} {r['hall_mu_cm2Vs']:6.1f} {r['sigma_S_cm']:9.1e} "
          f"{r['Eg_optical_eV']:6.3f} {r['dEg_eV']:+7.4f} {r['dark_activation_eV']:7.3f} "
          f"{r['PDR_score']:6.2f} {r['tau_score']:5.2f}")
