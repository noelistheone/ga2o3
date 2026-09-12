"""K-sweep validation: designed vs random anchors at K=1..5 across the 13 dedup labs.
Converts the '8-12 experiments' Laplace projection into a MEASURED saturation curve.
Writes results/hybrid/design_ksweep.json."""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"scripts"))
import importlib.util
spec=importlib.util.spec_from_file_location("dv", PROJ/"scripts/design_validate.py")
dv=importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)   # reuse cache/fit/score/designed
DUPS={'10.48550/arxiv.2311.00821','10.48550/arxiv.2407.17089','10.48550/arxiv.2001.11187'}
labs={k:v for k,v in dv.load_labs().items() if k!='nan' and not any(k.startswith(x) for x in DUPS)}
rng=np.random.RandomState(0)
out={"Ks":[1,2,3,4,5],"per_K":{}}
for K in out["Ks"]:
    des,rnd=[],[]
    for doi,pts in labs.items():
        doped=[i for i,p in enumerate(pts) if not p["und"]]
        if len(doped)<4: continue
        di=dv.designed(pts,K)
        test=[pts[i] for i in doped if i not in di] or [pts[i] for i in doped]
        use_nd=any(p["und"] for p in pts)
        des.append(dv.score(dv.fit_grid([pts[j] for j in di],use_nd),test))
        combos=list(itertools.combinations(range(len(pts)),K)); rng.shuffle(combos)
        rr=[]
        for c in combos[:5]:
            tr=[pts[i] for i in doped if i not in c]
            if tr: rr.append(dv.score(dv.fit_grid([pts[j] for j in c],use_nd),tr))
        if rr: rnd.append(float(np.mean(rr)))
    out["per_K"][K]={"designed":round(float(np.median(des)),3),"random":round(float(np.median(rnd)),3),
                     "n_labs":len(des)}
    print(f"K={K}: designed {out['per_K'][K]['designed']} random {out['per_K'][K]['random']}",flush=True)
out["floor"]=0.47; out["blind"]=1.28
(PROJ/"results/hybrid/design_ksweep.json").write_text(json.dumps(out,indent=1))
print("done")
