"""C1 rebuild: within-paper doped-vs-UNDOPED n direction test (true reference, base-rate null).
Only papers containing a genuine undoped/intrinsic row are used; each doped row's observed
direction = sign(log n_doped - log n_undoped); predicted direction from the solver chain the
same way (doped chain-n vs undoped chain-n at matched process). Nulls: (a) always-majority
baseline, (b) permutation of predicted signs within papers. Writes results/tier2/trend_t2_rebuilt.json."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
PROJ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJ/"src"))
from sandbox import full_forward as ff, kroger_db
db = kroger_db.load()
KRO = set(db.elements)-{"Ga","O"}
tp = pd.read_csv(PROJ.parent/"Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
tp = tp[tp["carrier_cm3"].notna()]
PROC = dict(T_anneal=1073.0, pO2=1e-5, film=True, Nd_bg=1e17, db=db)
und = ff.forward("undoped", 0.0, **PROC); n_und_chain = und["hall_n_cm3"]
chain_cache = {}
def chain_n(el):
    if el not in chain_cache:
        chain_cache[el] = ff.forward(el, 0.005, **PROC)["hall_n_cm3"]
    return chain_cache[el]
contrasts = []
for doi, g in tp.groupby("doi"):
    els = [str(e) for e in g["dopant_element"]]
    und_rows = g[[str(e).lower() in ("undoped","nan","none","") for e in g["dopant_element"]]]
    if len(und_rows)==0: continue          # true reference required
    ref = float(np.log10(max(und_rows["carrier_cm3"].astype(float).max(),1)))
    for _, r in g.iterrows():
        el = str(r["dopant_element"])
        if el.lower() in ("undoped","nan","none","") or el not in KRO: continue
        obs = np.sign(np.log10(max(float(r["carrier_cm3"]),1)) - ref)
        pred = np.sign(np.log10(max(chain_n(el),1)) - np.log10(max(n_und_chain,1)))
        if obs==0 or pred==0: continue
        contrasts.append({"doi":str(doi)[:30],"el":el,"obs":int(obs),"pred":int(pred)})
c = pd.DataFrame(contrasts)
hits = int((c.obs==c.pred).sum()); tot=len(c)
base_up = float((c.obs>0).mean()); majority = max(base_up,1-base_up)
rng = np.random.RandomState(0)
null=[ (rng.permutation(c.pred.values)==c.obs.values).mean() for _ in range(20000)]
perm_p = float((np.array(null)>=hits/tot).mean())
from scipy.stats import binomtest
p_vs_majority = float(binomtest(hits, tot, majority, alternative="greater").pvalue)
out = {"n_contrasts":tot,"correct":hits,"frac":round(hits/tot,3),
       "base_rate_up":round(base_up,3),"majority_baseline":round(majority,3),
       "perm_p":round(perm_p,4),"binom_p_vs_majority":round(p_vs_majority,4),
       "n_papers":int(c.doi.nunique()),
       "note":"true doped-vs-undoped within-paper contrasts only (C1 rebuild)"}
(PROJ/"results/tier2/trend_t2_rebuilt.json").write_text(json.dumps(out,indent=1))
print(json.dumps(out,indent=1))
