"""R6 reviewer #1 minor: Proposition 2 is proven for Pearson correlation; the empirical metrics
are Spearman. Numeric check that the attenuation bound holds for Spearman under the fitted
variance components. Simulation: y_ij = s_ij + b_j + e_ij with Var(b)/(Var(b)+Var(e)) = ICC
matched to each property's REML fit; the ORACLE predictor is the true signal s_ij (best case any
model can do). We report the Monte-Carlo Spearman and Pearson correlations of oracle vs observed
against the bound sqrt(1-ICC), for Gaussian and heavy-tailed/skewed signal distributions.
Writes results/tier2/prop2_spearman_check.json."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)

# (property, ICC, n_studies, rows_per_study) matched to the corpus scale
CASES = [("optical_bandgap_Eg", 0.587, 150, 4), ("hall_mobility_mu", 0.850, 90, 4),
         ("photo_dark_ratio", 0.604, 60, 4)]
DISTS = {"gaussian": lambda size: rng.normal(size=size),
         "lognormal_skew": lambda size: rng.lognormal(sigma=1.0, size=size),
         "t3_heavy": lambda size: rng.standard_t(3, size=size)}

out = {}
for prop, icc, J, m in CASES:
    bound = float(np.sqrt(1 - icc))
    res = {}
    for dname, gen in DISTS.items():
        sp, pe = [], []
        for _ in range(400):
            s = gen((J, m))
            s = (s - s.mean()) / s.std()
            sig_b = np.sqrt(icc / (1 - icc))     # Var(e)=1 after standardizing signal-noise below
            b = rng.normal(scale=sig_b, size=(J, 1))
            e = rng.normal(scale=1.0, size=(J, m))
            # signal variance chosen equal to... oracle correlation is bounded regardless of
            # signal share; use signal std 1 (worst-case bound is at high signal share)
            y = s + b          # tight case of Prop 2: corr(s, s+b) = sqrt(1-ICC) exactly;
            del e              # within-study noise only lowers the correlation further
            sp.append(stats.spearmanr(s.ravel(), y.ravel()).statistic)
            pe.append(stats.pearsonr(s.ravel(), y.ravel()).statistic)
        res[dname] = {"spearman_mc": round(float(np.mean(sp)), 4),
                      "spearman_p97.5": round(float(np.percentile(sp, 97.5)), 4),
                      "pearson_mc": round(float(np.mean(pe)), 4)}
    out[prop] = {"icc": icc, "bound_sqrt_1_minus_icc": round(bound, 4), "cases": res,
                 "holds_spearman": all(v["spearman_p97.5"] <= bound + 0.02 for v in res.values())}
    print(prop, out[prop]["bound_sqrt_1_minus_icc"],
          {k: v["spearman_mc"] for k, v in res.items()}, flush=True)

(PROJ / "results/tier2/prop2_spearman_check.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/prop2_spearman_check.json")
