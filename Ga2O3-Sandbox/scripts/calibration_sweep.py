"""Process-bridge calibration sweep — can tuning the 4 nuisance params rescue discriminating
skill (the T1 cross-dopant carrier-density ranking)? If the ranking is energetics-limited (set
by donor/acceptor classification, not process), calibration won't move it — which would confirm
the exponential-sensitivity ceiling manifesting in the FORWARD physics, not just the ML.

Sweeps (freeze-in T_anneal, effective pO2, Nd_bg) and reports the best T1 Spearman ρ vs the
transport-corpus median n, with the permutation floor. Disciplined: only 3 knobs swept, coarse
grid, and we report whether ANY setting beats the floor (not cherry-pick a single lucky cell).
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db  # noqa: E402

db = kroger_db.load()
KROGER = set(db.elements)
rng = np.random.default_rng(0)

tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
tp = tp[tp["carrier_cm3"].notna()]
corpus_n = {}
for el, g in tp.groupby("dopant_element"):
    el = str(el)
    if el in KROGER and el not in ("Ga", "O"):
        corpus_n[el] = float(np.median(np.log10(g["carrier_cm3"].astype(float).clip(lower=1))))

T_grid = [1073.0, 1400.0, 1700.0]
pO2_grid = [1e-8, 1e-5, 1e-2]
Nd_grid = [1e16, 1e17, 1e18]

results = []
best = None
for T in T_grid:
    for pO2 in pO2_grid:
        for Nd in Nd_grid:
            proc = dict(T_anneal=T, pO2=pO2, EcT_fraction=0.40, film=True, E_B_eV=0.03, Nd_bg=Nd)
            und = ff.forward("undoped", 0.0, db=db, **proc)
            cache = {"n_und": und["hall_n_cm3"]}
            preds = {}
            for X in corpus_n:
                if X in ("Ga", "O"):
                    continue
                try:
                    r = ff.forward(X, 0.01, db=db, undoped_cache=cache, **proc)
                    preds[X] = np.log10(max(r["hall_n_cm3"], 1.0))
                except Exception:
                    pass
            common = [e for e in preds if e in corpus_n]
            if len(common) < 6:
                continue
            x = np.array([preds[e] for e in common])
            y = np.array([corpus_n[e] for e in common])
            rho = float(spearmanr(x, y).correlation)
            null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(2000)])
            p = float((np.abs(null) >= abs(rho)).mean())
            row = {"T": T, "pO2": pO2, "Nd_bg": Nd, "n_dopants": len(common),
                   "rho": round(rho, 3), "perm_p": round(p, 4), "beats_floor": bool(p < 0.05)}
            results.append(row)
            if best is None or rho > best["rho"]:
                best = row

any_beats = any(r["beats_floor"] for r in results)
report = {
    "grid": {"T": T_grid, "pO2": pO2_grid, "Nd_bg": Nd_grid},
    "n_settings": len(results),
    "best_setting": best,
    "any_setting_beats_floor_p05": any_beats,
    "all_results": sorted(results, key=lambda r: -r["rho"])[:10],
    "interpretation": ("calibration CAN rescue the cross-dopant n ranking (some setting beats "
                       "the floor) → process bridge matters" if any_beats else
                       "NO process setting makes the cross-dopant n ranking floor-robust → the "
                       "ranking is ENERGETICS-limited (donor/acceptor classification), not "
                       "process-limited: the exponential-sensitivity ceiling manifests in the "
                       "FORWARD physics too, confirming the thesis from the physics side"),
}
(PROJ / "results/tier2/calibration_sweep.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
