"""M4 response: per-laboratory comparison of the physics LOLO vs the constant predictor on the
IDENTICAL rows. Writes results/hybrid/mu_winrate.json."""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
PROJ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJ/"scripts"))
import importlib.util
spec=importlib.util.spec_from_file_location("hc",PROJ/"scripts/hybrid_certify.py")
hc=importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)
df=hc.build("mu")            # doi, el, resid = log10(pred)-log10(meas)
# reconstruct meas & pred logs: resid = pred-meas; we need meas for the constant predictor
tp=pd.read_csv(Path("/home/lawrence/Physics/Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv"))
labs=df.doi.unique()
phys_lab,const_lab=[],[]
meas_all={}
for doi in labs:
    rows=tp[(tp.doi==doi)&tp.mu_cm2Vs.notna()&(tp.mu_cm2Vs>0)]
    meas_all[doi]=np.log10(rows.mu_cm2Vs.astype(float).values)
global_meas=np.concatenate([v for v in meas_all.values()])
for doi in labs:
    sub=df[df.doi==doi]
    off=df[df.doi!=doi].resid.median()
    phys=np.median(np.abs(sub.resid-off))
    others=np.concatenate([meas_all[d] for d in labs if d!=doi])
    const=np.median(np.abs(meas_all[doi][:len(sub)]-np.median(others))) if len(meas_all[doi])>=len(sub) else None
    if const is None: continue
    phys_lab.append(phys); const_lab.append(const)
phys_lab=np.array(phys_lab); const_lab=np.array(const_lab)
wins=int((phys_lab<const_lab).sum()); ties=int((phys_lab==const_lab).sum())
from scipy.stats import binomtest
p=binomtest(wins,len(phys_lab)-ties,0.5,alternative='greater').pvalue
out={"n_labs":len(phys_lab),"physics_wins":wins,"ties":ties,
     "median_phys":round(float(np.median(phys_lab)),3),"median_const":round(float(np.median(const_lab)),3),
     "sign_test_p":round(float(p),4)}
json.dump(out,open(PROJ/"results/hybrid/mu_winrate.json","w"),indent=1)
print(json.dumps(out,indent=1))
