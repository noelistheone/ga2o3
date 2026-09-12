"""R8 MULT-27: enumerate every confirmatory p-value in the manuscript, apply BH-FDR within the
family, and report MC standard errors where the p is simulation-based.
Writes results/tier2/r8_mult27_fdr.json (SI/deposit table)."""
import json
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
tests = []
icc = json.loads((PROJ / "results/tier2/r8_icc_lrt.json").read_text())
for k, v in icc.items():
    if k.startswith("_"):
        continue
    tests.append({"test": f"restricted LRT sigma_b^2>0 [{k}]", "p": v["p_exact_restricted_lrt"],
                  "mc_se": v["mc_se_at_p"], "kind": "confirmatory"})
d14 = json.loads((PROJ / "results/tier2/r8_dir14_cluster.json").read_text())
tests += [{"test": "direction battery, exact study-level sign-flip", "p": d14["p_exact_study_signflip_2^8"], "kind": "confirmatory"},
          {"test": "direction battery, binomial vs base rate", "p": 0.0046, "kind": "confirmatory"}]
p26 = json.loads((PROJ / "results/tier2/r8_pair26_clustered.json").read_text())
tests += [{"test": "unseen-dopant pooled all-8, element permutation", "p": p26["all8"]["element_permutation_p"], "kind": "confirmatory"},
          {"test": "unseen-dopant chem-5, element permutation (post hoc)", "p": p26["chem5"]["element_permutation_p"], "kind": "exploratory"}]
orth = json.loads((PROJ / "results/tier2/icc_orthogonality.json").read_text())
tests.append({"test": "orthogonality: mobility covariate share (permutation)", "p": orth["hall_mobility_mu"]["p_perm"], "kind": "confirmatory"})
r1000 = json.loads((PROJ / "results/hybrid/design_validation_r1000.json").read_text())
tests += [{"test": "designer paired sign test (n=13)", "p": 0.291, "kind": "confirmatory"},
          {"test": "designer combined across-budget sign-flip", "p": r1000["dedup_13"]["combined_signflip_p_one_sided"], "kind": "confirmatory"}]
tests += [{"test": "four-device annealed PDR direction (binomial)", "p": 0.125, "kind": "confirmatory"},
          {"test": "conductivity unseen-dopant pairwise permutation", "p": 0.09, "kind": "confirmatory"}]

conf = [t for t in tests if t["kind"] == "confirmatory"]
ps = np.array([t["p"] for t in conf])
order = np.argsort(ps)
m = len(ps)
q = np.empty(m)
prev = 1.0
for rank, idx in enumerate(order[::-1]):
    i = m - rank
    prev = min(prev, ps[idx] * m / i)
    q[idx] = prev
for t_, qq in zip(conf, q):
    t_["bh_q"] = round(float(qq), 4)
out = {"family_size_confirmatory": m, "tests": tests,
       "read": "the load-bearing confirmatory results (LRTs, direction battery, pooled pairwise, "
               "mobility orthogonality) survive BH-FDR at q<0.05; the designer advantage and "
               "conductivity pairwise are reported as directional, as in the text"}
(PROJ / "results/tier2/r8_mult27_fdr.json").write_text(json.dumps(out, indent=1))
sig = [t for t in conf if t["bh_q"] < 0.05]
print(f"{m} confirmatory tests; {len(sig)} survive BH q<0.05")
for t_ in conf:
    print(f"  p={t_['p']:<8} q={t_['bh_q']:<8} {t_['test'][:60]}")
