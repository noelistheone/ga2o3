import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
import numpy as np
from sandbox import full_forward as ff, kroger_db

db = kroger_db.load()
PROC = dict(T_anneal=1350.0, pO2=1e-5, EcT_fraction=0.40, film=True, E_B_eV=0.03, Nd_bg=1e17)
und = ff.forward("undoped", 0.0, db=db, **PROC)
cache = {"n_und": und["hall_n_cm3"]}

print("=== CALIBRATED forward (freeze-in T=1350 K, donor-optimal) ===")
print(f"{'dopant':7s} {'log n':>7s} {'mu':>6s} {'logSig':>7s} {'Eg':>6s} {'dEg':>8s} {'darkEa':>7s}")
rows = {}
for X in ["undoped"] + [e for e in db.elements if e not in ("Ga", "O")]:
    c = 0.0 if X == "undoped" else 0.01
    try:
        r = und if X == "undoped" else ff.forward(X, c, db=db, undoped_cache=cache, **PROC)
        ln = np.log10(max(r["hall_n_cm3"], 1))
        ls = np.log10(max(r["sigma_S_cm"], 1e-30))
        print(f"{X:7s} {ln:7.2f} {r['hall_mu_cm2Vs']:6.1f} {ls:7.2f} {r['Eg_optical_eV']:6.3f} "
              f"{r['dEg_eV']:+8.4f} {r['dark_activation_eV']:7.3f}")
        rows[X] = {"log_n": round(ln, 2), "mu": round(r["hall_mu_cm2Vs"], 1),
                   "log_sigma": round(ls, 2), "Eg": round(r["Eg_optical_eV"], 3),
                   "dEg": round(r["dEg_eV"], 4)}
    except Exception as e:
        print(f"{X:7s} ERR {str(e)[:40]}")
(PROJ / "results/tier2/calibrated_forward_table.json").write_text(
    json.dumps({"freeze_in_T": 1350.0, "rows": rows}, indent=2))
