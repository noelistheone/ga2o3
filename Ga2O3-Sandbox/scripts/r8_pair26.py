"""R8 PAIR-26 (major): element-clustered inference for the pooled unseen-dopant pairwise
accuracies (59.0% all-8 / 62.3% chem-5). Reuses the Net phase105h element_loeo machinery
READ-ONLY to regenerate per-element (pred, true) medians per FOM, then:
  (a) element-cluster bootstrap CI (resample element NAMES with replacement, jointly across
      FOMs, recompute pooled pairwise accuracy);
  (b) element-permutation null (shuffle pred across elements within FOM, 10^4).
Writes results/tier2/r8_pair26_clustered.json."""
import sys, json, itertools, warnings
from pathlib import Path
import numpy as np
import importlib.util

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(NET / "scripts"))
spec = importlib.util.spec_from_file_location("p105h", NET / "scripts/phase105h_newdopant_supp.py")
p105h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p105h)

CHEM5 = None  # resolved from the module's grouping if available
FOMS8 = ["photo_dark_ratio", "dark_current_pA", "vacancy_concentration", "hall_carrier_n",
         "hall_mobility_mu", "optical_bandgap_Eg", "bandgap_shift_dEg", "tau_decay"]
CHEM = {"hall_mobility_mu", "hall_carrier_n", "optical_bandgap_Eg", "dark_current_pA",
        "vacancy_concentration"}

per_fom = {}
for fom in FOMS8:
    els, hib = p105h.element_loeo(fom)
    per_fom[fom] = els   # list of (elem, pred_median, true_median, n_rows)
    print(fom, "elements:", len(els), flush=True)


def pooled_acc(fomset, elems_by_fom):
    corr = []
    for fom in fomset:
        els = elems_by_fom[fom]
        for a, b in itertools.combinations(els, 2):
            if abs(a[2] - b[2]) > 1e-6:
                corr.append(int(np.sign(a[2] - b[2]) == np.sign(a[1] - b[1])))
    return (float(np.mean(corr)), len(corr)) if corr else (np.nan, 0)


out = {}
rng = np.random.default_rng(5)
for tag, fomset in (("all8", FOMS8), ("chem5", sorted(CHEM))):
    acc, npairs = pooled_acc(fomset, per_fom)
    # (a) element-name cluster bootstrap
    names = sorted({e[0] for fom in fomset for e in per_fom[fom]})
    baccs = []
    for _ in range(4000):
        pick = rng.choice(names, len(names), replace=True)
        counts = {}
        for nm in pick:
            counts[nm] = counts.get(nm, 0) + 1
        eb = {}
        for fom in fomset:
            lst = []
            for e in per_fom[fom]:
                lst.extend([e] * counts.get(e[0], 0))
            eb[fom] = lst
        a, _ = pooled_acc(fomset, eb)
        if np.isfinite(a):
            baccs.append(a)
    ci = [round(float(np.percentile(baccs, 2.5)), 3), round(float(np.percentile(baccs, 97.5)), 3)]
    # (b) element-permutation null
    null = []
    for _ in range(10000):
        eb = {}
        for fom in fomset:
            els = per_fom[fom]
            perm = rng.permutation([e[1] for e in els])
            eb[fom] = [(e[0], perm[i], e[2], e[3]) for i, e in enumerate(els)]
        a, _ = pooled_acc(fomset, eb)
        null.append(a)
    p_perm = float((np.sum(np.array(null) >= acc) + 1) / 10001)
    out[tag] = {"pooled_pairwise_acc": round(acc, 3), "n_pairs": npairs,
                "element_bootstrap_ci95": ci, "element_permutation_p": round(p_perm, 5),
                "n_element_names": len(names)}
    print(tag, out[tag], flush=True)

(PROJ / "results/tier2/r8_pair26_clustered.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_pair26_clustered.json")
