"""R7 reviewer #1 item 3: add the paired median-difference point estimates (designed - random_mean
per lab, median over 13 labs) to design_validation_r50.json so the manuscript's updated numbers
have a single file of record."""
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1] / "results/hybrid/design_validation_r50.json"
d = json.loads(P.read_text())
for K, blk in d["per_K"].items():
    diffs = [r["designed"] - r["random_mean"] for r in blk["per_lab"]]
    blk["median_paired_diff"] = round(float(np.median(diffs)), 3)
    print(f"K={K}: designed_median={blk['designed_median']} random_median={blk['random_median']} "
          f"median_paired_diff={blk['median_paired_diff']} CI={blk['median_diff_bootstrap_CI95']} "
          f"mass={blk['frac_bootstrap_mass_favoring_designed']} p={blk['sign_test_p']}")
P.write_text(json.dumps(d, indent=1))
