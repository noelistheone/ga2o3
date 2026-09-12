"""Protocol-upgraded K-sweep: per lab, a FIXED held-out test set (every other doped row, seeded)
never enters any anchor set; anchors (designed/random) drawn only from the remainder.
Eliminates the shrinking-test artifact. Writes results/hybrid/design_ksweep_fixedtest.json."""
import sys, json, itertools
from pathlib import Path
import numpy as np
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"scripts"))
import importlib.util
spec=importlib.util.spec_from_file_location("dv", PROJ/"scripts/design_validate.py")
dv=importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
DUPS={'10.48550/arxiv.2311.00821','10.48550/arxiv.2407.17089','10.48550/arxiv.2001.11187'}
labs={k:v for k,v in dv.load_labs().items() if k!='nan' and not any(k.startswith(x) for x in DUPS)}
rng=np.random.RandomState(0)
out={"Ks":[1,2,3,4,5],"per_K":{},"protocol":"fixed held-out test (alternating doped rows); anchors from remainder only"}
prep={}
for doi,pts in labs.items():
    doped=[i for i,p in enumerate(pts) if not p["und"]]
    if len(doped)<4: continue
    test_idx=doped[::2]; pool_idx=[i for i in range(len(pts)) if i not in test_idx]
    prep[doi]=(pts,[pts[i] for i in test_idx],pool_idx)
for K in out["Ks"]:
    des,rnd=[],[]
    for doi,(pts,test,pool_idx) in prep.items():
        if len(pool_idx)<K: continue
        use_nd=any(p["und"] for p in pts)
        pool=[pts[i] for i in pool_idx]
        di=dv.designed(pool,K)
        des.append(dv.score(dv.fit_grid([pool[j] for j in di],use_nd),test))
        combos=list(itertools.combinations(range(len(pool)),K)); rng.shuffle(combos)
        rr=[dv.score(dv.fit_grid([pool[j] for j in c],use_nd),test) for c in combos[:5]]
        if rr: rnd.append(float(np.mean(rr)))
    out["per_K"][K]={"designed":round(float(np.median(des)),3),"random":round(float(np.median(rnd)),3),"n_labs":len(des)}
    print(K,out["per_K"][K],flush=True)
(PROJ/"results/hybrid/design_ksweep_fixedtest.json").write_text(json.dumps(out,indent=1))
print("done")
