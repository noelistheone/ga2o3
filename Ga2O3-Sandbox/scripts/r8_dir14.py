"""R8 DIR-14 (major): cluster-aware statistics for the 37/37 direction battery.
Rebuilds the true within-study doped-vs-undoped contrasts (identical construction to
trend_t2_rebuilt.py), then reports: (a) the within-study permutation p at 2x10^4 draws
(permute predicted signs WITHIN each study); (b) the exact study-level sign-flip test
(enumerate all 2^8 study sign configurations); (c) the per-study SI table (contrasts,
directions, dopants, downward-contrast studies). Writes results/tier2/r8_dir14_cluster.json."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
src = (PROJ / "scripts/trend_t2_rebuilt.py").read_text()
# reuse the contrast construction verbatim: execute up to the DataFrame build
head = src.split("c = pd.DataFrame(contrasts)")[0]
ns = {"__file__": str(PROJ / "scripts/trend_t2_rebuilt.py")}
exec(head, ns)
c = pd.DataFrame(ns["contrasts"])
rng = np.random.default_rng(3)

hits, tot = int((c.pred == c.obs).sum()), len(c)
# (a) within-study permutation: permute pred within each doi
accs = []
for _ in range(20000):
    perm = c.groupby("doi")["pred"].transform(lambda s: rng.permutation(s.values))
    accs.append(float((perm == c.obs).mean()))
p_within = float((np.array(accs) >= hits / tot).mean())
# (b) exact study-level sign-flip: flip all of a study's predictions together, enumerate 2^n
studies = c.doi.unique()
n_s = len(studies)
correct_frac = []
for mask in range(2 ** n_s):
    acc = 0
    for i, s in enumerate(studies):
        sub = c[c.doi == s]
        flip = (mask >> i) & 1
        pred = -sub.pred.values if flip else sub.pred.values
        acc += int((pred == sub.obs.values).sum())
    correct_frac.append(acc / tot)
p_exact_study_flip = float(np.mean(np.array(correct_frac) >= hits / tot))
# (c) per-study table
tab = []
for s, g in c.groupby("doi"):
    tab.append({"doi": s, "n_contrasts": int(len(g)),
                "n_correct": int((g.pred == g.obs).sum()),
                "dopants": sorted(set(g.el)),
                "n_downward_obs": int((g.obs < 0).sum())})
down_studies = sum(1 for t in tab if t["n_downward_obs"] > 0)
out = {"n_contrasts": tot, "correct": hits, "n_studies": n_s,
       "p_within_study_permutation_20000": round(p_within, 5),
       "p_exact_study_signflip_2^8": round(p_exact_study_flip, 5),
       "downward_contrast_studies": down_studies,
       "per_study": tab}
print(json.dumps({k: v for k, v in out.items() if k != "per_study"}, indent=1))
(PROJ / "results/tier2/r8_dir14_cluster.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_dir14_cluster.json")
