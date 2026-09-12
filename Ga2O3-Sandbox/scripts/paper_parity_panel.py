"""Per-dopant parity data: simulator (certified global process, Tf=1500) vs corpus median,
for mu and n. Writes paper/npj/figures/parity_data.json."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"src"))
from sandbox import full_forward as ff, kroger_db
db=kroger_db.load(); DOP=set(db.elements)-{"Ga","O"}
tp=pd.read_csv(PROJ.parent/"Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
out={"process":"Tf=1500K pO2=1e-5 film c=0.5at%","dopants":{}}
for el in sorted(DOP):
    g=tp[tp.dopant_element==el]
    mu_m=g[g.mu_cm2Vs>0]["mu_cm2Vs"].astype(float)
    n_m=g[g.carrier_cm3>0]["carrier_cm3"].astype(float)
    if len(mu_m)==0 and len(n_m)==0: continue
    r=ff.forward(el,0.005,T_anneal=1500.0,pO2=1e-5,film=True,Nd_bg=1e17,db=db,undoped_cache={})
    out["dopants"][el]={
        "pred_log_mu":round(float(np.log10(max(r["hall_mu_cm2Vs"],1e-3))),3),
        "pred_log_n":round(float(np.log10(max(r["hall_n_cm3"],1.0))),3),
        "meas_log_mu":round(float(np.log10(mu_m.median())),3) if len(mu_m) else None,
        "meas_log_n":round(float(np.log10(n_m.median())),3) if len(n_m) else None,
        "n_rows_mu":int(len(mu_m)),"n_rows_n":int(len(n_m))}
    print(el,out["dopants"][el],flush=True)
d=PROJ/"paper/npj/figures"; (d/"parity_data.json").write_text(json.dumps(out,indent=1))
print("wrote parity_data.json")
