"""Can the solver reproduce the Ge downturn? Re-run the 6-point Ge series (doi 10.1063/5.0058059)
across freeze-in T; check predicted slope sign & solubility flags."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"src"))
from sandbox import full_forward as ff, kroger_db
db=kroger_db.load()
tp=pd.read_csv(PROJ.parent/"Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
g=tp[(tp.doi=="10.1063/5.0058059")&tp.carrier_cm3.notna()&tp.concentration_value.notna()].copy()
def cf(v,u):
    v=float(v);u=str(u).lower()
    return v/3.83e22 if "cm" in u else (v/100. if ("at" in u or "%" in u) else v/100.*0.5)
g["c"]=[cf(v,u) for v,u in zip(g.concentration_value,g.concentration_unit)]
g=g[g.c>0].sort_values("c")
meas=np.log10(g.carrier_cm3.astype(float).values); lc=np.log10(g.c.values)
out={"n_points":len(g),"meas_slope":round(float(np.polyfit(lc,meas,1)[0]),3),"runs":{}}
for Tf in [1073.,1350.,1500.,1650.]:
    pred=[]; sol=[]
    for c in g.c:
        r=ff.forward("Ge",c,T_anneal=Tf,pO2=1e-5,film=True,Nd_bg=1e17,db=db,undoped_cache={})
        pred.append(np.log10(max(r["hall_n_cm3"],1.0))); sol.append(bool(r["solubility_limited"]))
    rho=spearmanr(pred,meas).statistic
    out["runs"][f"Tf{int(Tf)}"]={"spearman":round(float(rho),3),
        "pred_slope":round(float(np.polyfit(lc,pred,1)[0]),3),"any_solubility_flag":any(sol)}
    print(Tf,out["runs"][f"Tf{int(Tf)}"],flush=True)
(PROJ/"results/tier2/ge_series_calibrated.json").write_text(json.dumps(out,indent=1))
