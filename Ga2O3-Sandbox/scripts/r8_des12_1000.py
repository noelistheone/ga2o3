"""R8 DES-12 (major): random arm at 1000 draws per laboratory + the designed value's quantile
in the pooled-random sampling distribution + a combined across-budget permutation test
(K in {2,3}; per-lab paired, sign-flip permutation of designed-vs-random-mean differences)
+ DDL-43: the same medians with the 3 dedup-removed preprint groups restored.
Writes results/hybrid/design_validation_r1000.json."""
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
ALL = {k: v for k, v in dv.load_labs().items() if k != 'nan'}
DEDUP = {k: v for k, v in ALL.items() if not any(k.startswith(x) for x in DUPS)}
rng = np.random.RandomState(0)
N_DRAWS = 1000


def run(labs, tag):
    res = {}
    for K in (2, 3):
        per_lab, rand_pool = [], []
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
                                "n_draws": len(rr)})
                rand_pool.append(rr)
        d = np.array([r["designed"] for r in per_lab])
        rm = np.array([r["random_mean"] for r in per_lab])
        # designed pooled median's quantile within the pooled-random sampling distribution:
        # draw one random score per lab, pool medians, repeat
        qrng = np.random.default_rng(7)
        pool_meds = []
        for _ in range(4000):
            pick = [rr[qrng.integers(len(rr))] for rr in rand_pool]
            pool_meds.append(np.median(pick))
        q = float(np.mean(np.array(pool_meds) >= np.median(d)))
        res[str(K)] = {"n_labs": len(per_lab),
                       "designed_median": round(float(np.median(d)), 3),
                       "random_median_of_means": round(float(np.median(rm)), 3),
                       "pooled_random_median_MC_sd": round(float(np.std(pool_meds)), 3),
                       "designed_quantile_vs_random": round(1 - q, 3),
                       "pooled_random_median_samples": [round(float(x), 3) for x in list(pool_meds)[:4000]],
                       "per_lab": per_lab}
        print(f"[{tag}] K={K}: designed {np.median(d):.3f} vs random {np.median(rm):.3f} "
              f"(MC sd {np.std(pool_meds):.3f}); designed at quantile {1-q:.3f}", flush=True)
    # combined across-budget sign-flip permutation on paired diffs (K=2 and K=3 jointly)
    diffs = []
    for K in ("2", "3"):
        for r in res[K]["per_lab"]:
            diffs.append(r["designed"] - r["random_mean"])
    diffs = np.array(diffs)
    obs = diffs.mean()
    prng = np.random.default_rng(11)
    null = [(diffs * prng.choice([-1, 1], len(diffs))).mean() for _ in range(20000)]
    p_comb = float((np.sum(np.array(null) <= obs) + 1) / 20001)   # designed better = negative
    res["combined_signflip_p_one_sided"] = round(p_comb, 4)
    res["combined_mean_paired_diff"] = round(float(obs), 4)
    print(f"[{tag}] combined K=2+3 mean paired diff {obs:+.3f}, sign-flip p {p_comb:.4f}", flush=True)
    return res


out = {"n_draws": N_DRAWS,
       "dedup_13": run(DEDUP, "dedup13"),
       "with_dups_restored": run(ALL, "all")}
(PROJ / "results/hybrid/design_validation_r1000.json").write_text(json.dumps(out, indent=1))
print("wrote results/hybrid/design_validation_r1000.json")
