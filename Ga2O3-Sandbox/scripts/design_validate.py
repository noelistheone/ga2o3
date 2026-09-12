"""HONEST real-data validation of the experiment-designer (anti-p-hacking) — FAST version.

Claim under test: choosing per-lab ANCHOR experiments by the D-optimal design criterion reduces
held-out absolute-n prediction error more than a RANDOM anchor choice, on REAL measured lab data,
and approaches the within-lab floor with fewer experiments.

Method: train/test holdout per multi-point lab (doi). Fit per-lab theta=(log10 Tf, log10 Nd_bg) by
GRID SEARCH minimising median |Δlog10 n| on a k-point anchor set, predict the REMAINING doped
points' absolute n. Strategies: global (no anchor), designed-k (D-optimal), random-k, floor (fit on
all). Designer earns its keep iff des_k < rand_k at the same k and des_k -> floor at small k.

Speed: a module-level cache memoises ff.forward over (dopant, conc, film, Tf, Nd_bg, pO2). Since the
grid is fixed, the same forward recurs across floor/designed/random fits -> ~10x fewer solves.
Every predicted n is the sandbox forward pass; theta is a physical per-lab calibration (Tf freeze-in,
Nd_bg background donor), not a table lookup.
"""
import sys, json, warnings, itertools
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db

db = kroger_db.load()
DOP = set(db.elements) - {"Ga", "O"}
NET = PROJ.parent / "Ga2O3-Net"

TF_GRID = np.log10(np.linspace(1100.0, 1950.0, 9))       # effective freeze-in T candidates
ND_GRID_UND = [np.log10(x) for x in (1e16, 1e17)]        # only if lab has an undoped point
ND_FIX = np.log10(1e17)
THETA0 = {"log10_Tf": np.log10(1400.0), "log10_Nd_bg": ND_FIX}
SIG_N = 0.15                                              # log10 n noise (dex)
PRIOR2 = np.diag([1.0 / 0.18 ** 2, 1.0 / 1.0 ** 2])       # (Tf, Nd_bg) prior precision
STEP = {"log10_Tf": 0.03, "log10_Nd_bg": 0.2}
_CACHE = {}


def logn(cond, lTf, lNd):
    key = (cond["dopant"], round(cond["conc"], 5), cond["film"], round(float(lTf), 3),
           round(float(lNd), 2), cond["pO2"])
    if key not in _CACHE:
        r = ff.forward(cond["dopant"], cond["conc"], T_anneal=10 ** lTf, pO2=cond["pO2"],
                       film=cond["film"], E_B_eV=0.03, Nd_bg=10 ** lNd, db=db,
                       undoped_cache=({} if cond["dopant"] != "undoped" else None))
        _CACHE[key] = np.log10(max(r["hall_n_cm3"], 1.0))
    return _CACHE[key]


def conc_frac(v, u):
    if pd.isna(v):
        return 0.005
    v = float(v); u = str(u).lower()
    if "cm" in u:
        return min(v / 3.83e22, 0.1)
    if "at" in u or "%" in u:
        return v / 100.0
    if "wt" in u:
        return v / 100.0 * 0.5
    return 0.005


def load_labs():
    tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
    labs = {}
    for _, r in tp.iterrows():
        el = str(r["dopant_element"])
        if pd.isna(r.get("carrier_cm3")):
            continue
        n = float(r["carrier_cm3"])
        if n <= 0:
            continue
        und = el.lower() in ("undoped", "nan", "none", "")
        if not und and el not in DOP:
            continue
        film = "sputter" in str(r.get("growth_method", "")).lower()
        cf = 0.0 if und else conc_frac(r.get("concentration_value"), r.get("concentration_unit"))
        labs.setdefault(str(r["doi"]), []).append(
            {"dopant": "undoped" if und else el, "conc": cf, "film": film, "pO2": 1e-5,
             "meas": float(np.log10(n)), "und": und})
    return {k: v for k, v in labs.items() if len(v) >= 5}


def fit_grid(anchor, use_nd):
    nd_opts = ND_GRID_UND if use_nd else [ND_FIX]
    best_err, best = np.inf, THETA0
    for lTf in TF_GRID:
        for lNd in nd_opts:
            err = np.median([abs(logn(p, lTf, lNd) - p["meas"]) for p in anchor])
            if err < best_err:
                best_err, best = err, {"log10_Tf": float(lTf), "log10_Nd_bg": float(lNd)}
    return best


def score(theta, test):
    return float(np.median([abs(logn(p, theta["log10_Tf"], theta["log10_Nd_bg"]) - p["meas"])
                            for p in test]))


def fisher2(cond, theta):
    j = np.zeros(2)
    for i, key in enumerate(["log10_Tf", "log10_Nd_bg"]):
        tp = dict(theta); tm = dict(theta); tp[key] += STEP[key]; tm[key] -= STEP[key]
        j[i] = (logn(cond, tp["log10_Tf"], tp["log10_Nd_bg"])
                - logn(cond, tm["log10_Tf"], tm["log10_Nd_bg"])) / (2 * STEP[key])
    return np.outer(j, j) / SIG_N ** 2


def designed(points, k):
    M = PRIOR2.copy()
    fish = [fisher2(p, THETA0) for p in points]
    chosen, rem = [], set(range(len(points)))
    for _ in range(min(k, len(points))):
        best, bv = None, -np.inf
        for i in rem:
            v = np.linalg.slogdet(M + fish[i])[1]
            if v > bv:
                bv, best = v, i
        M = M + fish[best]; chosen.append(best); rem.discard(best)
    return chosen


def main():
    labs = load_labs()
    print(f"labs with >=5 n points: {len(labs)}", flush=True)
    rng = np.random.RandomState(0)
    rows, agg = [], {k: [] for k in ["global", "rand2", "des2", "rand3", "des3", "floor"]}
    for doi, pts in labs.items():
        doped = [i for i, p in enumerate(pts) if not p["und"]]
        if len(doped) < 4:
            continue
        use_nd = any(p["und"] for p in pts)
        res = {"doi": doi[:34], "n": len(pts), "und": use_nd,
               "dopants": sorted(set(p["dopant"] for p in pts))}
        res["global"] = round(score(THETA0, [pts[i] for i in doped]), 2)
        res["floor"] = round(score(fit_grid(pts, use_nd), [pts[i] for i in doped]), 2)
        for k in (2, 3):
            di = designed(pts, k)
            test = [pts[i] for i in doped if i not in di] or [pts[i] for i in doped]
            res[f"des{k}"] = round(score(fit_grid([pts[j] for j in di], use_nd), test), 2)
            combos = list(itertools.combinations(range(len(pts)), k)); rng.shuffle(combos)
            rr = []
            for c in combos[:5]:
                tr = [pts[i] for i in doped if i not in c]
                if tr:
                    rr.append(score(fit_grid([pts[j] for j in c], use_nd), tr))
            res[f"rand{k}"] = round(float(np.mean(rr)), 2) if rr else None
        rows.append(res)
        for key in agg:
            if res.get(key) is not None:
                agg[key].append(res[key])
        print(f"{res['doi']:34s} n={res['n']:2d} und={str(res['und'])[0]} {res['dopants']} | "
              f"glob {res['global']} | des2 {res['des2']} rand2 {res['rand2']} | "
              f"des3 {res['des3']} rand3 {res['rand3']} | floor {res['floor']}", flush=True)

    summ = {k: round(float(np.median(v)), 3) for k, v in agg.items() if v}
    out = {"n_labs": len(rows), "per_lab": rows, "pooled_median_dex": summ,
           "cache_size": len(_CACHE),
           "reads": ["global=Mode-A baseline(no anchor); floor=within-lab best fit.",
                     "designer adds value iff des_k<rand_k at same k and des_k->floor at small k.",
                     "n scored in log10 cm^-3; every prediction is the sandbox forward pass."]}
    (PROJ / "results/hybrid/design_validation.json").write_text(json.dumps(out, indent=2))
    print("\nPOOLED median |Δlog10 n| (dex):", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
