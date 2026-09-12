"""M3 response: the flat Si concentration-response is the uncalibrated-process artifact.
Re-run the 18-point Si series (doi 10.1063/5.0142107) at the calibrated freeze-in T (1500 K)
vs the corpus-standard 1073 K. Writes results/tier2/si_series_calibrated.json."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"src"))
from sandbox import full_forward as ff, kroger_db
db=kroger_db.load()
tp=pd.read_csv(PROJ.parent/"Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
g=tp[(tp.doi=="10.1063/5.0142107")&tp.carrier_cm3.notna()&tp.concentration_value.notna()].copy()
def cf(v,u):
    v=float(v);u=str(u).lower()
    return v/3.83e22 if "cm" in u else (v/100. if ("at" in u or "%" in u) else v/100.*0.5)
g["c"]=[cf(v,u) for v,u in zip(g.concentration_value,g.concentration_unit)]
g=g[g.c>0].sort_values("c")
out={}
for Tf in [1073.0,1350.0,1500.0]:
    pred=[np.log10(max(ff.forward("Si",c,T_anneal=Tf,pO2=1e-5,film=True,Nd_bg=1e17,db=db,
          undoped_cache={})["hall_n_cm3"],1.0)) for c in g.c]
    meas=np.log10(g.carrier_cm3.astype(float).values)
    lc=np.log10(g.c.values)
    rho=spearmanr(pred,meas).statistic
    sp=np.polyfit(lc,pred,1)[0]; sm=np.polyfit(lc,meas,1)[0]
    out[f"Tf{int(Tf)}"]={"n_points":len(g),"spearman":round(float(rho),3),
        "pred_slope_dex_per_dex":round(float(sp),3),"meas_slope_dex_per_dex":round(float(sm),3)}
    print(Tf,out[f"Tf{int(Tf)}"],flush=True)
(PROJ/"results/tier2/si_series_calibrated.json").write_text(json.dumps(out,indent=1))
