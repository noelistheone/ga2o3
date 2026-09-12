"""R9: per-property unseen-dopant (leave-one-element-out) pairwise accuracy with the SAME
element-clustered inference used for the pooled numbers in results/tier2/r8_pair26_clustered.json.
Reuses Ga2O3-Net phase105h element_loeo READ-ONLY. Writes results/revision_r9/unseen_dopant_perfom.json.
Statistics are element-clustered throughout: the naive binomial over pairs is NOT reported, because
pairs share elements and are not independent."""
import sys, json, itertools, warnings, importlib.util
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parents[1]; NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(NET / "scripts"))
spec = importlib.util.spec_from_file_location("p105h", NET / "scripts/phase105h_newdopant_supp.py")
p105h = importlib.util.module_from_spec(spec); spec.loader.exec_module(p105h)

FOMS8 = ["hall_mobility_mu", "hall_carrier_n", "optical_bandgap_Eg", "vacancy_concentration",
         "dark_current_pA", "photo_dark_ratio", "tau_decay", "bandgap_shift_dEg"]
CHEM5 = {"hall_mobility_mu", "hall_carrier_n", "optical_bandgap_Eg", "vacancy_concentration",
         "dark_current_pA"}

def pair_acc(els):
    c = [int(np.sign(a[2]-b[2]) == np.sign(a[1]-b[1]))
         for a, b in itertools.combinations(els, 2) if abs(a[2]-b[2]) > 1e-6]
    return (float(np.mean(c)), len(c)) if c else (float('nan'), 0)

def median_split_acc(els):
    if len(els) < 3: return float('nan')
    pm, tm = np.median([e[1] for e in els]), np.median([e[2] for e in els])
    return float(np.mean([(e[1] > pm) == (e[2] > tm) for e in els]))

rng = np.random.default_rng(5)
out = {"_method": "leave-one-element-out per FOM; element-clustered bootstrap CI (4000) and "
                  "element-permutation null (10000); naive per-pair binomial deliberately not reported",
       "_reuses": "Ga2O3-Net/scripts/phase105h_newdopant_supp.py::element_loeo (read-only)",
       "per_fom": {}}
for fom in FOMS8:
    els, _ = p105h.element_loeo(fom)
    acc, npair = pair_acc(els)
    boot = []
    for _ in range(4000):
        pick = rng.choice(len(els), len(els), replace=True)
        a, _n = pair_acc([els[i] for i in pick])
        if not np.isnan(a): boot.append(a)
    null = []
    for _ in range(10000):
        perm = rng.permutation(len(els))
        a, _n = pair_acc([(els[i][0], els[perm[k]][1], els[i][2]) for k, i in enumerate(range(len(els)))])
        if not np.isnan(a): null.append(a)
    p = float((np.sum(np.array(null) >= acc) + 1) / (len(null) + 1))
    out["per_fom"][fom] = {"pairwise_acc": round(acc, 3), "n_pairs": npair, "n_elements": len(els),
                           "element_bootstrap_ci95": [round(float(np.percentile(boot, 2.5)), 3),
                                                      round(float(np.percentile(boot, 97.5)), 3)],
                           "element_permutation_p": round(p, 4),
                           "median_split_acc": round(median_split_acc(els), 3),
                           "chem5": fom in CHEM5}
    print(f"{fom:22s} acc={acc:.3f} n_pairs={npair:3d} n_el={len(els):2d} p_perm={p:.4f}", flush=True)
o = PROJ / "results/revision_r9/unseen_dopant_perfom.json"
o.parent.mkdir(parents=True, exist_ok=True); o.write_text(json.dumps(out, indent=1))
print("\nwrote", o)
