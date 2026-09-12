"""Brouwer-diagram data for the paper physics figure: [V_O], n vs pO2 at three anneal T,
undoped + Sn-doped, from the sandbox's own solver. Writes paper/npj/figures/brouwer_data.json."""
import sys, json
from pathlib import Path
import numpy as np
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import engine, kroger_db, full_forward as ff

db = kroger_db.load()
out = {"pO2_grid": list(np.logspace(-16, 0, 33)), "curves": {}}
N_cat = kroger_db.N_SITE * 2 / 3
for T in [1073.0, 1273.0, 1473.0]:
    for dop, conc in [("undoped", 0.0), ("Sn", 0.005), ("Mg", 0.005)]:
        key = f"T{int(T)}_{dop}"
        vo, n = [], []
        for p in out["pO2_grid"]:
            fixed = {} if dop == "undoped" else {dop: conc * N_cat}
            eq = engine.solve_single(db, T, pO2=p, fixed_conc=fixed, fd=True, Nd_bg=1e17)
            o = db.col("O")
            is_vo = (db.dm[:, o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
            vo.append(float(eq.N_cs[is_vo].sum()))
            n.append(float(eq.n))
        out["curves"][key] = {"log10_VO": [round(np.log10(max(v,1.0)),3) for v in vo],
                              "log10_n": [round(np.log10(max(x,1.0)),3) for x in n]}
        print(key, "done", flush=True)
d = PROJ / "paper/npj/figures"; d.mkdir(parents=True, exist_ok=True)
(d / "brouwer_data.json").write_text(json.dumps(out))
print("wrote paper/npj/figures/brouwer_data.json")
