"""Corpus-wide detection of the paired-row extraction defect found by the 100-row audit:
a junk sibling row whose concentration_value equals the carrier mantissa of a good row sharing
doi+quote, with its own carrier_cm3 empty. Writes results/hybrid/flagged_rows.json and a
sensitivity re-run of the mobility LOLO certification excluding flagged rows."""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
PROJ=Path(__file__).resolve().parents[1]
tp=pd.read_csv(PROJ.parent/"Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
tp["_idx"]=tp.index
flag=set()
for (doi,), g in tp.groupby(["doi"]):
    good=g[g.carrier_cm3.notna()&(g.carrier_cm3>0)]
    mant={round(float(n)/10**np.floor(np.log10(float(n))),2) for n in good.carrier_cm3}
    cand=g[g.carrier_cm3.isna()&g.concentration_value.notna()&(g.concentration_unit.astype(str).str.contains("at|%",case=False,na=False))]
    for _,r in cand.iterrows():
        if round(float(r.concentration_value),2) in mant:
            flag.add(int(r._idx))
# plus the audit's specific misc bugs
for i in (6,101,515,520): flag.add(i)
# plus the audit's 3 UNSUPPORTED rows (quote present but contains none of the recorded values)
for i in (68,75,109): flag.add(i)
# plus the R8 inter-route consistency probe's >2-dex outlier (row 5: resistivity and Hall
# values from different treatment states of the same film; doi-less legacy row)
flag.add(5)
# plus the two Ge-series rows whose extracted concentrations are physically impossible
# (125 and 138 'at%'; caught by the R8 complete-data series check)
for i in (420, 427): flag.add(i)
tp_flag=tp.loc[sorted(flag)]
out={"n_flagged":len(flag),"n_with_mu":int(tp_flag.mu_cm2Vs.notna().sum()),
     "n_with_n":int(tp_flag.carrier_cm3.notna().sum()),
     "by_doi":tp_flag.groupby("doi").size().sort_values(ascending=False).head(8).to_dict(),
     "signature":"conc(at%) == carrier mantissa of doi-sibling & own carrier empty; + 4 audit-specific + 3 audit-unsupported (68,75,109) + 1 inter-route outlier (5) + 2 impossible-concentration series rows (420,427)"}
json.dump(out,open(PROJ/"results/hybrid/flagged_rows.json","w"),indent=1)
print(json.dumps(out,indent=1))
# sensitivity: mobility LOLO excluding flagged
sys.path.insert(0,str(PROJ/"scripts"))
import importlib.util
spec=importlib.util.spec_from_file_location("hc",PROJ/"scripts/hybrid_certify.py")
hc=importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)
orig_read=pd.read_csv
def patched(path,*a,**k):
    df=orig_read(path,*a,**k)
    if "transport_llm_extracted_v2" in str(path):
        df=df.drop(index=[i for i in flag if i in df.index])
    return df
pd.read_csv=patched
res=hc.certify("mu")
json.dump(res,open(PROJ/"results/hybrid/certification_mu_flagged_excluded.json","w"),indent=1)
print("mu LOLO excl flagged:",json.dumps(res,indent=1))
