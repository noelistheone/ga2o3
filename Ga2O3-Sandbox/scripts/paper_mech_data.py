import sys, json
from pathlib import Path
import numpy as np
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"src"))
from sandbox import full_forward as ff, kroger_db
db=kroger_db.load()
out={}
out["freezein_Si"]={"Tf":[],"log_n":[]}
for Tf in [1073,1200,1350,1500,1650,1800,1950]:
    r=ff.forward("Si",0.01,T_anneal=float(Tf),pO2=1e-5,film=True,Nd_bg=1e17,db=db,undoped_cache={})
    out["freezein_Si"]["Tf"].append(Tf); out["freezein_Si"]["log_n"].append(round(float(np.log10(max(r["hall_n_cm3"],1.0))),2))
    print("Tf",Tf,out["freezein_Si"]["log_n"][-1],flush=True)
out["cliff_Ta"]={"log_pO2":[],"log_n":[]}
for lp in [-3,-4,-5,-6,-7,-8,-9,-10]:
    r=ff.forward("Ta",0.005,T_anneal=1350.0,pO2=10.0**lp,film=True,Nd_bg=1e17,db=db,undoped_cache={})
    out["cliff_Ta"]["log_pO2"].append(lp); out["cliff_Ta"]["log_n"].append(round(float(np.log10(max(r["hall_n_cm3"],1.0))),2))
    print("pO2 1e%d"%lp,out["cliff_Ta"]["log_n"][-1],flush=True)
(PROJ/"paper/npj/figures/mech_data.json").write_text(json.dumps(out,indent=1))
print("wrote mech_data.json")
