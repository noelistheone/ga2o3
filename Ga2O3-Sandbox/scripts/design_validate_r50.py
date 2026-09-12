"""R6 reviewer #1 item 3: harden the designed-vs-random comparison by expanding the random arm
from 5 to 50 draws per laboratory (same 13 dedup labs, same fit/score machinery reused from
design_validate.py). Reports pooled medians, per-lab pairs, sign-test p, and the paired
bootstrap CI of the median difference. Writes results/hybrid/design_validation_r50.json."""
import sys, json, itertools
from pathlib import Path
import numpy as np
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("dv", PROJ / "scripts/design_validate.py")
dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dv)

DUPS = {'10.48550/arxiv.2311.00821', '10.48550/arxiv.2407.17089', '10.48550/arxiv.2001.11187'}
labs = {k: v for k, v in dv.load_labs().items() if k != 'nan' and not any(k.startswith(x) for x in DUPS)}
rng = np.random.RandomState(0)
N_DRAWS = 50

out = {"n_draws": N_DRAWS, "per_K": {}}
for K in (2, 3):
    per_lab = []
    for doi, pts in labs.items():
        doped = [i for i, p in enumerate(pts) if not p["und"]]
        if len(doped) < 4:
            continue
        use_nd = any(p["und"] for p in pts)
        di = dv.designed(pts, K)
        test = [pts[i] for i in doped if i not in di] or [pts[i] for i in doped]
        des = dv.score(dv.fit_grid([pts[j] for j in di], use_nd), test)
        combos = list(itertools.combinations(range(len(pts)), K))
        rng.shuffle(combos)
        rr = []
        for c in combos[:N_DRAWS]:
            tr = [pts[i] for i in doped if i not in c]
            if tr:
                rr.append(dv.score(dv.fit_grid([pts[j] for j in c], use_nd), tr))
        if rr:
            per_lab.append({"doi": doi[:34], "designed": round(des, 3),
                            "random_mean": round(float(np.mean(rr)), 3),
                            "n_random_draws": len(rr)})
            print(f"K={K} {doi[:30]:30s} des {des:.2f} rand {np.mean(rr):.2f} ({len(rr)} draws)",
                  flush=True)
    d = np.array([r["designed"] for r in per_lab])
    r = np.array([r["random_mean"] for r in per_lab])
    wins = int(np.sum(d < r))
    p_sign = float(stats.binomtest(wins, len(d), 0.5, alternative="greater").pvalue)
    boots = []
    brng = np.random.default_rng(1)
    for _ in range(4000):
        i = brng.integers(0, len(d), len(d))
        boots.append(np.median(d[i] - r[i]))
    out["per_K"][K] = {"n_labs": len(per_lab),
                       "designed_median": round(float(np.median(d)), 3),
                       "random_median": round(float(np.median(r)), 3),
                       "wins_designed": wins, "sign_test_p": round(p_sign, 3),
                       "median_diff_bootstrap_CI95": [round(float(np.percentile(boots, 2.5)), 3),
                                                      round(float(np.percentile(boots, 97.5)), 3)],
                       "frac_bootstrap_mass_favoring_designed": round(
                           float(np.mean(np.array(boots) < 0)), 3),
                       "per_lab": per_lab}
    print(f"K={K}: designed {np.median(d):.3f} vs random {np.median(r):.3f}, "
          f"wins {wins}/{len(d)}, p={p_sign:.3f}", flush=True)

(PROJ / "results/hybrid/design_validation_r50.json").write_text(json.dumps(out, indent=1))
print("wrote results/hybrid/design_validation_r50.json")
