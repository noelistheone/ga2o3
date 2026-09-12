"""R7 reviewer #2 item 3 (error bands on solver outputs): re-run the solver with the paper's
+-0.25 eV level/formation-energy budget applied, giving MEASURED (not sketched) uncertainty
bands for Fig. 3a (Brouwer, rigid V_O formation-energy shift, undoped 1273 K) and Fig. 6c
(Ta boundary, donor-level shift via dHo(q) + q*delta). Writes paper/npj/figures/budget_bands.json."""
import sys, json, dataclasses
from pathlib import Path
import numpy as np
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import engine, kroger_db, full_forward as ff

db = kroger_db.load()
DELTA = 0.25
o = db.col("O")
is_vo = (db.dm[:, o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)

out = {"delta_eV": DELTA}

grid = list(np.logspace(-16, 0, 33))
out["brouwer_1273_undoped"] = {"pO2_grid": grid}
for tag, dE in [("lo", +DELTA), ("hi", -DELTA)]:  # +dE raises E_f -> fewer V_O
    dbp = dataclasses.replace(db, dHo=db.dHo + is_vo.astype(float) * dE)
    curve = []
    for p in grid:
        eq = engine.solve_single(dbp, 1273.0, pO2=p, fixed_conc={}, fd=True, Nd_bg=1e17)
        curve.append(round(float(np.log10(max(eq.N_cs[is_vo].sum(), 1.0))), 3))
    out["brouwer_1273_undoped"][tag] = curve
    print("brouwer", tag, "done", flush=True)

ta = db.col("Ta")
qmask = np.where(db.dm[:, ta] != 0, db.charge.astype(float), 0.0)
lps = [-3, -4, -5, -6, -7, -8, -9, -10]
out["cliff_Ta"] = {"log_pO2": lps}
for tag, sgn in [("deeper", +1.0), ("shallower", -1.0)]:
    dbp = dataclasses.replace(db, dHo=db.dHo + qmask * sgn * DELTA)
    curve = []
    for lp in lps:
        r = ff.forward("Ta", 0.005, T_anneal=1350.0, pO2=10.0 ** lp, film=True,
                       Nd_bg=1e17, db=dbp, undoped_cache={})
        curve.append(round(float(np.log10(max(r["hall_n_cm3"], 1.0))), 2))
    out["cliff_Ta"][tag] = curve
    print("Ta", tag, curve, flush=True)

(PROJ / "paper/npj/figures/budget_bands.json").write_text(json.dumps(out, indent=1))
print("wrote paper/npj/figures/budget_bands.json")
