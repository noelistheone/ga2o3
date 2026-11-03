"""Predictive preparation: independent-source temperature-curve controls.

These reduced controls do not replace the full energetic Sandbox or the
29-record primary task. No fitted source donor density is a chemical input.
"""
from pathlib import Path
import os,sys,json,time
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sandbox.donor_ionization import DonorIonizationSandbox,solve_ionization
from sandbox.implicit_defects import ImplicitDomainError
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as old

DATA_DIR=energy.BASE/"phase8_process_temperature/extraction_v2"
OUT=energy.BASE/"phase8_process_temperature/temperature_baselines"
SPECS=read_result_csv(DATA_DIR/"paired_specimens.csv")
OBS=read_result_csv(DATA_DIR/"temperature_observations.csv")
OBS=OBS[OBS.endpoint.eq("log10_nH_cm3")].copy()
anchors=pd.DataFrame([dict(read_id="tableRT::"+r.sample_id,sample_id=r.sample_id,
    source=r.source,family=r.family,T_K=300.,observed=np.log10(r.reported_RT_nH_cm3),
    extraction="published room-temperature text/table",endpoint="log10_nH_cm3",
    low_temperature_warning=False) for r in SPECS.itertuples()])
PRED=pd.concat([OBS,anchors],ignore_index=True)
PRED["incorporated_cm3"]=PRED.sample_id.map(SPECS.set_index("sample_id").incorporated_cm3)
TRAIN=PRED[(~PRED.read_id.str.startswith("tableRT::")&PRED.T_K.le(275))|
           PRED.read_id.str.startswith("tableRT::")].copy()
GRIDS={"chemical_constant":[{}],"hydrogenic_FD":[{}],"learned_binding":[{}],
       "learned_screened_binding":[{}],
       "direct_ridge":[dict(alpha=x) for x in (.01,.1,1.,10.,100.,1e4)]}
FITS={}

def freeze():
    files=[Path(__file__),ROOT/"src/sandbox/donor_ionization.py",
           ROOT/"src/sandbox/implicit_defects.py",ROOT/"src/sandbox/thermo.py",
           ROOT/"scripts/trainable_sandbox_20261017/csv_io_v2.py"]
    payload=dict(version=1,created_utc=energy.now(),
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            DATA_DIR/"temperature_observations.csv",DATA_DIR/"paired_specimens.csv",DATA_DIR/"extraction_protocol.json")},
        goal="prepare predictive temperature challenge for full ML+Sandbox; not substitute reduced controls for primary objective",
        methods=GRIDS,
        physical_bounds=dict(logB=[10,18],logit_compensation=[-8,6],log10E0=[-3,float(np.log10(.2))],screening=[0,5]),
        physics="FD approximation reused from existing layer; ionization C/(1+2exp(eta+E/kBT)), effective fully ionized background; frozen compensators",
        observation="unit Hall factor, uniform one-channel band transport assumed, microscopic identification not claimed",
        physical_optimization="two starts,max_nfev80,analytic IFT,three/four parameters,all failures retained",
        direct_features=["logC-17","(logC-17)^2","log(T/300)","300/T-1","(logC-17)*(300/T-1)"],
        training="sample-equal loss; measured trace<=275K plus publishedRT per train specimen, to avoid double counting nearbyRT",
        outer="leave entire DOI out,2source groups; other source has3or4specimens",
        inner="leave one specimen of only available training DOI out; does not estimate inner across-DOI shift",
        primary_scope="fixed40<=T<=125K future points, equal specimen then equal DOI; all-query/warm/cold also retained",
        adaptation="0/1/2TEMPERATURE anchors PER SPECIMEN; not old0/1/2specimens per DOI",
        anchor1="publishedRT,300K; plot points within25K are guards",
        anchor2="nearestTto150K in125..175; selected byT only, plus10K guards",
        lambda_grid=[.1,1.,10.,"infinity"],
        tuning="inside train specimens only; minimize fixed common40..125 MAE; ties stronger shrinkage,then lowercandidate index",
        budgets="same C,T,train curves,anchors; <=6model candidates, actual compute varies",
        prior_energy="hydrogenic36.6meV from specified mass/dielectric, no source-fitted donor label supplied",
        lowT="all visible points retained; lowT model assumptions potentially fail, no quality-based cherry picking",
        grade="retrospective2source pilot on7previous specimens, not independent new materials or broad cross-source accuracy",
        new_DFT=0,new_material_measurements=0)
    p=OUT/"protocol.json"
    if p.exists():
        d=json.loads(p.read_text())
        for k in ("code_hashes","input_hashes","methods"):assert d[k]==energy.clean(payload[k]),k
    else:energy.write(p,payload)

def gates():
    rows=[]
    for screened in (False,True):
        layer=DonorIonizationSandbox(screened)
        for offset in (0.,.2):
            w=np.array([14.,-3.,np.log10(.0366)]+([.6] if screened else []));w[0]+=offset
            c=np.array([1.93e16,1.4e17,1.3e18]);t=np.array([40.,150.,300.])
            p=layer.predict(c,t,w)
            for j in range(len(w)):
                h=1e-5;plus=w.copy();minus=w.copy();plus[j]+=h;minus[j]-=h
                fd=(layer.predict(c,t,plus).log_n-layer.predict(c,t,minus).log_n)/(2*h)
                err=float(np.max(abs(fd-p.jacobian[:,j])/(1+abs(fd))))
                rows.append(dict(screened=screened,offset=offset,parameter=j,scaled_error=err,passed=err<3e-6))
    assert all(x["passed"] for x in rows)
    energy.write(OUT/"gradient_gates.json",dict(passed=True,checks=rows,
        interpretation="first derivatives of specified FD approximation and effective model, not physical accuracy"))

def features(d):
    c=np.log10(d.incorporated_cm3.to_numpy(float))-17;t=d.T_K.to_numpy(float)
    return np.c_[c,c*c,np.log(t/300),300/t-1,c*(300/t-1)]

def fit(ids,method,config):
    ids=sorted(ids);key=old.digest(dict(ids=ids,method=method,config=config,protocol=energy.sha(OUT/"protocol.json")))
    path=OUT/"fits"/f"{key}.json"
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    tr=TRAIN[TRAIN.sample_id.isin(ids)];y=tr.observed.to_numpy(float)
    weight=1/tr.sample_id.map(tr.sample_id.value_counts()).to_numpy(float);weight/=weight.mean()
    start=time.perf_counter();trials=[];diag={};parameters=None;state_failures=[]
    C=PRED.incorporated_cm3.to_numpy(float);T=PRED.T_K.to_numpy(float)
    if method=="chemical_constant":pred=np.log10(C)
    elif method=="hydrogenic_FD":
        q=solve_ionization(C,T,background=0,compensation=0,energies_eV=.0366);pred=np.log10(q["n"])
    elif method=="direct_ridge":
        x=features(tr);center=np.average(x,axis=0,weights=weight)
        scale=np.sqrt(np.average((x-center)**2,axis=0,weights=weight));scale[scale<1e-12]=1
        m=Ridge(**config).fit((x-center)/scale,y,sample_weight=weight)
        pred=m.predict((features(PRED)-center)/scale)
        diag=dict(center=center,scale=scale,coefficients=m.coef_,intercept=m.intercept_)
    else:
        screened=method=="learned_screened_binding";layer=DonorIonizationSandbox(screened)
        lo=[10.,-8.,-3.]+([0.] if screened else [])
        hi=[18.,6.,float(np.log10(.2))]+([5.] if screened else [])
        Ct=tr.incorporated_cm3.to_numpy(float);Tt=tr.T_K.to_numpy(float);cache={}
        def calc(w):
            key=tuple(w)
            if key not in cache:
                try:p=layer.predict(Ct,Tt,w)
                except (ImplicitDomainError,ValueError,FloatingPointError) as exc:
                    state_failures.append(dict(parameters=w,error=repr(exc)));raise
                cache.clear();cache[key]=p
            return cache[key]
        for B,rho,Ed,s in ((14.,-3.,.0366,.5),(15.5,-1.,.06,2.)):
            initial=np.array([B,rho,np.log10(Ed)]+([s] if screened else []))
            try:
                op=least_squares(lambda w:(calc(w).log_n-y)*np.sqrt(weight),initial,
                    jac=lambda w:calc(w).jacobian*np.sqrt(weight)[:,None],
                    bounds=(lo,hi),max_nfev=80,xtol=1e-8,ftol=1e-8,gtol=1e-8)
                trials.append(dict(parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                    nfev=int(op.nfev),active_mask=op.active_mask,message=op.message))
            except (ImplicitDomainError,ValueError,FloatingPointError) as exc:
                trials.append(dict(parameters=initial,loss=None,success=False,error=repr(exc)))
        good=[t for t in trials if t.get("loss") is not None]
        if good:
            parameters=min(good,key=lambda x:x["loss"])["parameters"]
            p=layer.predict(C,T,parameters);pred=p.log_n;diag=p.diagnostics
        else:pred=np.full(len(PRED),np.nan)
    f=dict(key=key,method=method,config=config,training_specimens=ids,
        training_families=sorted(tr.family.unique()),training_observations=len(tr),parameters=parameters,
        trials=trials,state_failures=state_failures,diagnostics=diag,seconds=time.perf_counter()-start,
        predictions=[dict(read_id=r.read_id,predicted=float(v) if np.isfinite(v) else None)
                     for r,v in zip(PRED.itertuples(),pred)])
    energy.write(path,f);FITS[key]=energy.clean(f);return FITS[key]

def evaluate(f,specimen,budget,lam):
    d=PRED[PRED.sample_id.eq(specimen)].copy()
    values=dict((r["read_id"],r["predicted"]) for r in f["predictions"])
    isRT=d.read_id.str.startswith("tableRT::")
    ar=[d[isRT].iloc[0]] if budget else []
    if budget==2:
        candidates=d[~isRT&d.T_K.between(125,175)].copy()
        assert len(candidates)
        ix=(abs(candidates.T_K-150)).sort_values(kind="stable").index[0];ar.append(d.loc[ix])
    anchor_ids={r.read_id for r in ar}
    residuals=[r.observed-values[r.read_id] for r in ar if values[r.read_id] is not None]
    delta=float(sum(residuals)/(len(ar)+lam)) if ar and np.isfinite(lam) and len(residuals)==len(ar) else 0.
    rows=[]
    for r in d.itertuples():
        role="unavailable_reference" if r.read_id.startswith("tableRT::") else "query"
        if r.read_id in anchor_ids:role="anchor"
        elif budget and abs(r.T_K-300)<=25:role="guard"
        if budget==2 and abs(r.T_K-ar[1].T_K)<=10 and r.read_id not in anchor_ids:role="guard"
        p=values[r.read_id]
        rows.append(dict(read_id=r.read_id,sample_id=r.sample_id,family=r.family,source=r.source,T_K=r.T_K,
            observed=r.observed,predicted=None if p is None else p+delta,raw_prediction=p,
            role=role,budget=budget,offset=delta,lambda_shrinkage=lam if np.isfinite(lam) else "infinity",
            common40_125=bool(40<=r.T_K<=125),numeric=p is not None))
    q=[r for r in rows if r["role"]=="query" and r["common40_125"]]
    assert q
    score=float(np.mean([abs(r["predicted"]-r["observed"]) for r in q])) if all(r["numeric"] for r in q) else np.inf
    return score,rows

def main():
    freeze();gates();families=sorted(SPECS.family.unique());output=[];selections=[]
    for held in families:
        tr_ids=sorted(SPECS[SPECS.family.ne(held)].sample_id);te_ids=sorted(SPECS[SPECS.family.eq(held)].sample_id)
        for method,grid in GRIDS.items():
            for budget in (0,1,2):
                candidates=[]
                for i,config in enumerate(grid):
                    for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                        scores=[]
                        for inner in tr_ids:
                            f=fit([x for x in tr_ids if x!=inner],method,config)
                            scores.append(evaluate(f,inner,budget,lam)[0])
                        candidates.append(dict(index=i,config=config,lam=lam,score=float(np.mean(scores))))
                best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"],c["index"]))
                f=fit(tr_ids,method,best["config"]);scores=[]
                for specimen in te_ids:
                    score,rows=evaluate(f,specimen,budget,best["lam"]);scores.append(score)
                    for r in rows:r.update(method=method,fit_key=f["key"])
                    output+=rows
                selections.append(dict(held=held,method=method,budget=budget,training_specimens=tr_ids,
                    candidates=candidates,selected=best,common40_125_MAE=float(np.mean(scores))))
                print(held,method,budget,float(np.mean(scores)),flush=True)
    energy.csvwrite(OUT/"predictions.csv",output);energy.write(OUT/"selections.json",selections)
    d=pd.DataFrame(output);by=[]
    for (m,b,f,s),g in d[d.role.eq("query")].groupby(["method","budget","family","sample_id"]):
        for scope in ("all_query","common40_125","warm100plus","coldBelow100"):
            h=g if scope=="all_query" else (g[g.common40_125] if scope=="common40_125"
                 else (g[g.T_K.ge(100)] if scope=="warm100plus" else g[g.T_K.lt(100)]))
            if not len(h):continue
            by.append(dict(method=m,budget=b,family=f,sample_id=s,scope=scope,n=len(h),
                numeric=int(h.predicted.notna().sum()),
                MAE=float(abs(h.predicted-h.observed).mean()) if h.predicted.notna().all() else None))
    a=pd.DataFrame(by);a.to_csv(OUT/"by_specimen.csv",index=False)
    f=a.groupby(["method","budget","family","scope"]).agg(specimens=("sample_id","nunique"),
        reads=("n","sum"),numeric=("numeric","sum"),MAE=("MAE","mean"),scored=("MAE","count")).reset_index()
    f.loc[f.scored.ne(f.specimens),"MAE"]=np.nan;f.to_csv(OUT/"by_family.csv",index=False)
    summary=f.groupby(["method","budget","scope"]).agg(source_groups=("family","nunique"),
        specimens=("specimens","sum"),reads=("reads","sum"),numeric=("numeric","sum"),
        MAE=("MAE","mean"),scored_sources=("MAE","count"),worst_source=("MAE","max")).reset_index()
    summary.loc[summary.scored_sources.ne(summary.source_groups),"MAE"]=np.nan
    summary.to_csv(OUT/"summary.csv",index=False)
    print(summary[summary.scope.eq("common40_125")].to_string(index=False),flush=True)

if __name__=="__main__":main()
