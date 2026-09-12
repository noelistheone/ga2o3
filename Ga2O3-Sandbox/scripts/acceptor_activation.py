"""Test the acceptor-activation hypothesis for the Mg/Zn conductivity gap.

Chain predicts Mg/Zn fully insulating (n~1) but corpus shows conductive (n~1e15-1e18). Physical
hypothesis: in sputtered films only a fraction f_active of the nominal deep-acceptor dopant sits
on active substitutional (acceptor) sites; the rest is neutral (interstitial / clustered / on the
wrong sublattice), so it under-compensates the donor background → residual n survives. Sweep
f_active and find what reproduces the corpus n. (The true f_active will come from the Mg/Zn
sputter-film structure — running — but this tests whether the mechanism is quantitatively viable.)
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import engine, kroger_db  # noqa: E402

db = kroger_db.load()
N_cat = kroger_db.N_SITE * 2 / 3
CORPUS_N = {"Mg": 10 ** 14.76, "Zn": 10 ** 18.0}   # transport-corpus median n

report = {"hypothesis": "incomplete acceptor activation retains donor-background n", "dopants": {}}
for X in ["Mg", "Zn"]:
    nominal = 0.01 * N_cat        # 1 at%
    rows = []
    for f_active in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
        active = f_active * nominal
        eq = engine.solve_single(db, 1350.0, pO2=1e-5, EcT_fraction=0.40,
                                 fixed_conc={X: active}, fd=True, Nd_bg=1e17)
        q = engine.quench(db, eq, T_quench=300.0, EcT_fraction=0.40, fd=True, Nd_bg=1e17)
        rows.append({"f_active": f_active, "log_n": round(np.log10(max(q.n, 1)), 2)})
    # find f_active matching corpus n
    target = np.log10(CORPUS_N[X])
    best = min(rows, key=lambda r: abs(r["log_n"] - target))
    report["dopants"][X] = {"corpus_log_n": round(target, 2), "sweep": rows,
                            "f_active_matching_corpus": best["f_active"],
                            "matched_log_n": best["log_n"]}

report["interpretation"] = ("if a physically-plausible f_active (~0.01-0.1, i.e. 1-10% active "
                            "acceptor, consistent with deep-acceptor low activation in sputtered "
                            "films) reproduces the corpus n, the acceptor gap is an ACTIVATION "
                            "effect (fixable, informed by the film structure) not a chain error")
(PROJ / "results/tier2/acceptor_activation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
