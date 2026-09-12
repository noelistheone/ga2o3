"""R8 SIG-36 + DDL-43 (minor): (1) quantify the sigma=e*n*mu vs 1/rho cross-check tolerance on
the 15 dual-route rows; (2) show the designer 17->13 laboratory dedup is outcome-irrelevant by
re-running the K=2/3 pooled medians with the 4 removed groups restored.
Writes results/tier2/r8_sig36_ddl43.json.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"

out = {}
# ---- SIG-36: dual-route sigma tolerance ----
tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
cols = {c.lower(): c for c in tp.columns}
rho_col = next((c for c in tp.columns if ("resist" in c.lower() or c.lower().startswith("rho"))), None)
if rho_col is not None:
    m = tp.carrier_cm3.notna() & tp.mu_cm2Vs.notna() & tp[rho_col].notna() \
        & (tp.carrier_cm3 > 0) & (tp.mu_cm2Vs > 0) & (tp[rho_col] > 0)
    e = 1.602176634e-19
    s_enmu = np.log10(e * tp.carrier_cm3[m] * tp.mu_cm2Vs[m])
    s_rho = np.log10(1.0 / tp[rho_col][m])
    dev = np.abs(s_enmu - s_rho)
    out["sig36_dual_route"] = {"n_rows": int(m.sum()),
                               "max_abs_dex": round(float(dev.max()), 3),
                               "median_abs_dex": round(float(dev.median()), 3)}
else:
    out["sig36_dual_route"] = {"error": "no resistivity column found",
                               "columns": list(tp.columns)}
print("SIG-36:", out["sig36_dual_route"], flush=True)
(PROJ / "results/tier2/r8_sig36_ddl43.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_sig36_ddl43.json (ddl43 handled by r8_des12 rerun set)")
