"""Dopant-input predictions followed by a source-excluded learned transport head.

Conditional 52-pair transport validation and complete dose-to-mu validation are
separate tasks. The latter currently has five eligible paired labels/two sources.
No measured target nH or mu enters the complete-chain prediction.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from sandbox import transport
OUT=ROOT/"results/trainable_sandbox_20261017"

def free_from_Hall_log(lognH):
    def f(logn):
        return logn-np.log10(float(transport.hall_factor(300.,10.**logn)))-lognH
    return 10.**brentq(f,lognH-1.,lognH+1.,xtol=1e-12)
def logCT(lognH):
    return np.log10(float(transport.mu_caughey_thomas(free_from_Hall_log(lognH),300.)))
def median(x,w):
    i=np.argsort(x,kind="stable");x,w=np.asarray(x)[i],np.asarray(w)[i]
    return float(x[np.searchsorted(np.cumsum(w),.5*np.sum(w),side="left")])

def main():
    d=pd.read_csv(ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv")
    assert len(d)==52 and d.family.nunique()==11
    d["logmu"]=np.log10(d.mu_cm2Vs);d["baseline"]=[logCT(x) for x in np.log10(d.carrier_cm3)]
    cache={}
    def fit(held):
        if held not in cache:
            tr=d[d.family!=held]
            w=1/tr.family.map(tr.family.value_counts()).to_numpy(float)
            cache[held]=dict(offset_dex=median((tr.logmu-tr.baseline).to_numpy(),w),
                train_records=len(tr),train_families=sorted(tr.family.unique()))
        return cache[held]
    conditional=[]
    for row in d.itertuples():
        f=fit(row.family)
        conditional.append(dict(family=row.family,sample_id=row.sample_id,observed_log_mu=row.logmu,
            predicted_log_mu=row.baseline+f["offset_dex"],scope="given measured nH"))
    conditional=pd.DataFrame(conditional)
    conditional.to_csv(OUT/"conditional_transport_lofo.csv",index=False)
    error=abs(conditional.predicted_log_mu-conditional.observed_log_mu)
    score=float(error.groupby(conditional.family).mean().mean())
    assert abs(score-.3666089)<2e-6,(score,"Conditional transport reference mismatch")
    primary=pd.read_csv(OUT/"predictions_v1.csv")
    dataset=pd.read_csv(OUT/"dataset_v1.csv").set_index("sample_id")
    valid=set(dataset[dataset.mobility_eligible.fillna(False).astype(bool)&dataset.mu_cm2Vs.gt(0)].index)
    assert valid.issubset(set(d.sample_id)),"Paired labels need an explicit join"
    records=[]
    for row in primary[primary.role.eq("query")&primary.sample_id.isin(valid)].itertuples():
        f=fit(row.family)
        assert row.family not in f["train_families"]
        pred_mu=logCT(row.predicted_log_nH)+f["offset_dex"]
        observed_mu=np.log10(dataset.loc[row.sample_id,"mu_cm2Vs"])
        records.append(dict(method=row.method,budget=row.budget,family=row.family,sample_id=row.sample_id,
            predicted_log_nH=row.predicted_log_nH,observed_log_nH=row.observed_log_nH,
            predicted_log_muH=pred_mu,observed_log_muH=observed_mu,
            predicted_log_sigma=np.log10(transport.E)+row.predicted_log_nH+pred_mu,
            observed_log_sigma=np.log10(transport.E)+row.observed_log_nH+observed_mu,
            transport_training_records=f["train_records"],transport_training_families="|".join(f["train_families"]),
            scope="dose/element/process to predicted nH to learned empirical mu to derived sigma",
            no_target_measured_nH_in_transport=True))
    result=pd.DataFrame(records);result.to_csv(OUT/"full_property_chain_predictions_v1b.csv",index=False)
    by=[]
    for keys,g in result.groupby(["method","budget","family"]):
        by.append(dict(method=keys[0],budget=keys[1],family=keys[2],n=len(g),
            carrier_mae=float(abs(g.predicted_log_nH-g.observed_log_nH).mean()),
            mobility_mae=float(abs(g.predicted_log_muH-g.observed_log_muH).mean()),
            conductivity_mae=float(abs(g.predicted_log_sigma-g.observed_log_sigma).mean())))
    pd.DataFrame(by).to_csv(OUT/"full_property_chain_by_family_v1b.csv",index=False)
    payload=dict(conditional_transport_records=52,conditional_transport_families=11,
        conditional_macro_mae_dex=score,conditional_reference_reproduced=True,
        eligible_full_chain_records=len(valid),eligible_full_chain_families=dataset.loc[list(valid)].family.nunique(),
        transport_held_family_labels_excluded=True,target_measured_nH_not_used=True,
        carrier_predictions_unchanged=True,scope="prototype: empirical nH-to-mu head, not a validated q-squared microscopic scattering model",
        conductivity_is_derived=True,fit_parameters=cache)
    (OUT/"transport_chain_audit_v1b.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in payload.items() if k!="fit_parameters"},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
