"""Common-anchor empirical/hierarchical/direct-ML controls, frozen before fits."""
from pathlib import Path
from itertools import product
import os,sys,json,time,pickle,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as evaluator

OUT=energy.BASE/"phase56_direct_controls"
DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
GRIDS={"identity":[{}],"constant":[{}],"unit_dose":[{}],"dose_line":[{}],
       "direct_ridge":[dict(alpha=a) for a in (.01,.1,1.,10.,100.,1e4)],
       "direct_trees":[dict(max_depth=d,min_samples_leaf=l) for d,l in product((2,5,None),(2,5))]}
FITS={}

def features(d):
    z=d.log_dose.to_numpy(float)-18.
    return np.c_[z,z*z,(d.growth_T_C.to_numpy(float)+273.15-1100.)/300.,d.element.eq("Sn").to_numpy(float)]

def freeze():
    p=OUT/"protocol.json"
    payload=dict(created_utc=energy.now(),models=GRIDS,
        code_sha256=energy.sha(Path(__file__)),
        evaluation_driver_sha256=energy.sha(Path(evaluator.__file__)),
        inputs={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv")},
        features=["logC-18","(logC-18)^2","(growthK-1100)/300","Snindicator"],
        feature_scope="same observed material/process proxies as phase1; no measured querynH or mu",
        task="29records/5source outer LOFO, nested remaining4source; exact0/1/2anchor and guarded/common18 policy",
        adaptation="same delta=sum(residual)/(m+lambda), lambda[.1,1,10,inf], budget0inf",
        hierarchy="shared empirical function plus shrinkage of the new-source intercept",
        tuning="at most6model candidates; lambdas chosen jointly inside training sources, no outer selection",
        actual_energy_usage="no energetic labels used; may ignore the same allowed physical-input budget; gains combine physical prior and architecture, not equal-information ablation",
        computation="deterministic closed-form/64trees, n_jobs1; actual fit cost reported, not equal-runtime claim",
        split_grade="same retrospective cohort; no new independent material validation")
    if p.exists():
        old=json.loads(p.read_text())
        assert old["code_sha256"]==payload["code_sha256"] and old["inputs"]==payload["inputs"]
    else:energy.write(p,payload)

def fit(families,method,config):
    key=evaluator.digest(dict(families=sorted(families),method=method,config=config,
                              protocol=energy.sha(OUT/"protocol.json")))
    if key in FITS:return FITS[key]
    p=OUT/"fits"/f"{key}.json"
    if p.exists():FITS[key]=json.loads(p.read_text());return FITS[key]
    start=time.perf_counter();tr=DATA[DATA.family.isin(families)]
    w=1/tr.family.map(tr.family.value_counts()).to_numpy(float);w/=w.mean()
    y=tr.observed_log_nH.to_numpy(float);x=tr.log_dose.to_numpy(float)
    z=DATA.log_dose.to_numpy(float);metadata={}
    if method=="identity":prediction=z
    elif method=="constant":prediction=np.full(len(DATA),np.average(y,weights=w))
    elif method=="unit_dose":prediction=z+np.average(y-x,weights=w)
    elif method=="dose_line":
        X=np.c_[np.ones(len(tr)),x-18.]
        b=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        prediction=b[0]+b[1]*(z-18.);metadata["coefficients"]=b
    else:
        X=features(tr);center=np.average(X,axis=0,weights=w)
        scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
        if method=="direct_ridge":m=Ridge(**config)
        else:m=ExtraTreesRegressor(n_estimators=64,random_state=1729,n_jobs=1,**config)
        m.fit((X-center)/scale,y,sample_weight=w);prediction=m.predict((features(DATA)-center)/scale)
        modelpath=OUT/"models"/f"{key}.pkl";modelpath.parent.mkdir(parents=True,exist_ok=True)
        with modelpath.open("wb") as f:pickle.dump(dict(model=m,center=center,scale=scale),f)
        metadata.update(center=center,scale=scale)
    f=dict(key=key,method=method,config=config,train_families=sorted(families),train_records=len(tr),
        seconds=time.perf_counter()-start,preprocessing=metadata,
        predictions=[dict(sample_id=r.sample_id,predicted_log_nH=float(p),numeric=True,
                          physical_valid=True,physical_validity="not_applicable_empirical")
                     for r,p in zip(DATA.itertuples(),prediction)])
    energy.write(p,f);FITS[key]=energy.clean(f);return FITS[key]

def main():
    freeze();evaluator.DATA=DATA;evaluator.ROLES=ROLES
    families=sorted(DATA.family.unique());allrows=[];selections=[]
    for held in families:
        training=[f for f in families if f!=held]
        for method,grid in GRIDS.items():
            for budget in (0,1,2):
                candidates=[]
                for i,config in enumerate(grid):
                    for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                        errors=[]
                        for inner in training:
                            f=fit([x for x in training if x!=inner],method,config)
                            errors.append(evaluator.evaluate(f,inner,budget,lam)[0])
                        candidates.append(dict(index=i,config=config,lam=lam,score=float(np.mean(errors))))
                best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"],c["index"]))
                f=fit(training,method,best["config"])
                score,rows=evaluator.evaluate(f,held,budget,best["lam"])
                for r in rows:r.update(method=method,fit_key=f["key"],physical_validity="not_applicable_empirical")
                allrows.extend(rows);selections.append(dict(held=held,method=method,budget=budget,
                    candidates=candidates,selected=best,MAE=score))
    d=pd.DataFrame(allrows);d.to_csv(OUT/"predictions.csv",index=False)
    energy.write(OUT/"selections.json",selections)
    common=set(ROLES[ROLES.budget.eq(2)&ROLES.role.eq("query")].sample_id)
    rows=[]
    for (m,b,f),g in d[d.role.eq("query")].groupby(["method","budget","family"]):
        for scope in ("all_query","common18"):
            h=g if scope=="all_query" else g[g.sample_id.isin(common)]
            rows.append(dict(method=m,budget=int(b),family=f,scope=scope,n=len(h),
                MAE=float(abs(h.predicted_log_nH-h.observed_log_nH).mean())))
    by=pd.DataFrame(rows);by.to_csv(OUT/"by_family.csv",index=False)
    summary=by.groupby(["method","budget","scope"]).agg(families=("family","nunique"),
        queries=("n","sum"),MAE=("MAE","mean"),worst=("MAE","max")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    print(summary[summary.scope.eq("all_query")].to_string(index=False),flush=True)

if __name__=="__main__":main()
