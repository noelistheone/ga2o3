import sys, json, warnings
warnings.filterwarnings("ignore")
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location(
    "hc", "/home/lawrence/Physics/Ga2O3-Sandbox/scripts/hybrid_certify.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

out = {}
for prop in ("mu", "n"):
    df = hc.build(prop)
    full = df["resid"].median()
    deltas = []
    for lab, g in df.groupby("doi"):
        off = df[df.doi != lab]["resid"].median()
        deltas.append(abs(off - full))
    out[prop] = {"n_labs": int(df.doi.nunique()),
                 "max_single_lab_offset_influence_dex": round(float(max(deltas)), 4),
                 "median_influence_dex": round(float(np.median(deltas)), 4)}
    print(prop, out[prop])
json.dump(out, open("/home/lawrence/Physics/Ga2O3-Sandbox/results/hybrid/lolo_leakage_bound.json", "w"), indent=1)
print("wrote results/hybrid/lolo_leakage_bound.json")
