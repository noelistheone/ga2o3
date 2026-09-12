"""M7 response: is the between-study ICC just chemistry choice? Refit REML per property with
DOPANT FIXED EFFECTS (y ~ C(dopant) + (1|study)) and compare to intercept-only ICC.
Uses the Net corpus via its own loader (read-only). Writes results/tier2/icc_dopant_adjusted.json."""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/lawrence/Physics/Ga2O3-Net/scripts")
import importlib.util
spec=importlib.util.spec_from_file_location("c72","/home/lawrence/Physics/Ga2O3-Net/scripts/_phase72_common.py")
c72=importlib.util.module_from_spec(spec); spec.loader.exec_module(c72)
import statsmodels.formula.api as smf
out={}
for fom in c72.FOMS:
    try:
        d=c72.load_fom(fom)
        if d is None: continue
        y=np.asarray(d["y"],float); g=np.asarray(d["groups"])
        # dopant labels: first column(s)? use labels/sub if present
        lab=d.get("labels")
        el=[str(x) for x in (lab if lab is not None else ["?"]*len(y))]
        df=pd.DataFrame({"y":y,"study":g,"dop":el})
        df=df[np.isfinite(df.y)]
        if df.study.nunique()<8: continue
        m0=smf.mixedlm("y~1",df,groups=df["study"]).fit(reml=True)
        icc0=float(m0.cov_re.iloc[0,0]/(m0.cov_re.iloc[0,0]+m0.scale))
        # dopant fixed effects (drop dopants with <2 rows to stabilize)
        keep=df.dop.map(df.dop.value_counts())>=2
        d2=df[keep]
        m1=smf.mixedlm("y~C(dop)",d2,groups=d2["study"]).fit(reml=True)
        icc1=float(m1.cov_re.iloc[0,0]/(m1.cov_re.iloc[0,0]+m1.scale))
        out[fom]={"icc_intercept_only":round(icc0,3),"icc_dopant_adjusted":round(icc1,3),
                  "n_rows":int(len(df)),"n_studies":int(df.study.nunique()),"n_dopants":int(df.dop.nunique())}
        print(fom,out[fom],flush=True)
    except Exception as e:
        print(fom,"ERR",str(e)[:120],flush=True)
(Path("results/tier2/icc_dopant_adjusted.json")).write_text(json.dumps(out,indent=1))
print("done")
