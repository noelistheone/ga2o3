"""Smoke test for src/sandbox/design.py — theta_fit + design_battery + uncertainty propagation."""
import sys, time, json
from pathlib import Path
import numpy as np
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import design, kroger_db

db = kroger_db.load()
t0 = time.time()

# 1) generate synthetic 'lab' data at a KNOWN theta, then recover it (identifiability check)
true = {"log10_Tf": np.log10(1600.0), "E_B": 0.05, "log10_Nd_bg": np.log10(3e17)}
pts = []
for cond in [{"dopant": "undoped", "conc": 0.0, "film": True, "pO2": 1e-5, "obs_mask": ["log10_n"]},
             {"dopant": "Si", "conc": 0.005, "film": True, "pO2": 1e-5},
             {"dopant": "Si", "conc": 0.02, "film": True, "pO2": 1e-5},
             {"dopant": "Sn", "conc": 0.005, "film": False, "pO2": 1e-5}]:
    y = design.observe(cond, true, db)
    meas = {"log10_n": float(y[0])}
    if cond.get("dopant") != "undoped":
        meas["log10_mu"] = float(y[1]); meas["log10_sigma"] = float(y[2])
    pts.append({**cond, "meas": meas})

th, cov = design.theta_fit(pts, db=db)
print("=== theta recovery (true vs fit) ===")
for k in design.THETA_KEYS:
    print(f"  {k:12s} true={true[k]:.3f}  fit={th[k]:.3f}  poststd={np.sqrt(cov[design.THETA_KEYS.index(k),design.THETA_KEYS.index(k)]):.3f}")

# 2) design the next battery from scratch (no base data): which experiments does it pick?
print("\n=== design_battery (K=4, D-optimal, no prior data) ===")
res = design.design_battery(K=4, db=db)
for c, g in zip(res["battery"], res["info_gain_slogdet"]):
    print(f"  +{g:6.2f}  {c['dopant']:7s} {c.get('conc',0)*100:4.1f}at% {'film' if c['film'] else 'xtal'} pO2={c['pO2']:.0e}")
print("theta_std before:", res["theta_std_before"])
print("theta_std after :", res["theta_std_after"])
print(f"abs_n_dex: {res['abs_n_dex_before']} -> {res['abs_n_dex_after']}")

# 3) propagate the fitted posterior to the 9 properties for a target prediction
print("\n=== predict_with_uncertainty (Si 1at% film, using fitted posterior) ===")
pu = design.predict_with_uncertainty({"dopant": "Si", "conc": 0.01, "film": True, "pO2": 1e-5}, th, cov, db)
for k, v in pu.items():
    print(f"  {k:20s} {v}")

print(f"\n[elapsed {time.time()-t0:.1f}s]")
