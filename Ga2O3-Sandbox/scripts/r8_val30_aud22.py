"""R8 VAL-30 + AUD-22.
VAL-30: split the 13 designer-validation labs by whether their row set contains an undoped
reference row; compare calibration quality (designed K=2 score) between the two groups -- the
designer's prescription (undoped film + ambient ladder) predicts labs WITH a reference identify
theta better.
AUD-22: quantify the transport audit's 'partial' class with the |value - quote| deviation
distribution per subclass, from the per-row verdicts of the 100-row audit.
Writes results/tier2/r8_val30_aud22.json."""
import sys, json
from pathlib import Path
import numpy as np
import importlib.util

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
spec = importlib.util.spec_from_file_location("dv", PROJ / "scripts/design_validate.py")
dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dv)

out = {}
DUPS = {'10.48550/arxiv.2311.00821', '10.48550/arxiv.2407.17089', '10.48550/arxiv.2001.11187'}
labs = {k: v for k, v in dv.load_labs().items() if k != 'nan' and not any(k.startswith(x) for x in DUPS)}
import itertools
withref, noref = [], []
for doi, pts in labs.items():
    doped = [i for i, p in enumerate(pts) if not p["und"]]
    if len(doped) < 4:
        continue
    use_nd = any(p["und"] for p in pts)
    di = dv.designed(pts, 2)
    test = [pts[i] for i in doped if i not in di] or [pts[i] for i in doped]
    des = dv.score(dv.fit_grid([pts[j] for j in di], use_nd), test)
    (withref if use_nd else noref).append(float(des))
out["val30"] = {"labs_with_undoped_ref": len(withref), "labs_without": len(noref),
                "designed_K2_median_withref": round(float(np.median(withref)), 3) if withref else None,
                "designed_K2_median_noref": round(float(np.median(noref)), 3) if noref else None}
print("VAL-30:", out["val30"], flush=True)

# AUD-22: partial-class deviation distribution
try:
    aud = json.load(open(PROJ / "results/tier2/quote_audit_100.json"))
except Exception:
    aud = None
if aud is None:
    import glob
    cand = glob.glob(str(PROJ / "results/tier2/*audit*100*.json")) + \
           glob.glob(str(PROJ / "results/tier2/quote_audit*.json"))
    out["aud22"] = {"note": "per-row audit file candidates", "files": cand}
else:
    out["aud22_keys"] = list(aud.keys())[:8]
print(json.dumps(out.get("aud22", out.get("aud22_keys", "loaded")), indent=1)[:300], flush=True)
(PROJ / "results/tier2/r8_val30_aud22.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_val30_aud22.json")
