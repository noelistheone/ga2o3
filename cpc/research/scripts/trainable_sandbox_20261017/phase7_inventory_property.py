"""Fixed-protocol source-held-out prediction with learned internal inventory."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,time,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import least_squares
from sandbox.kroger_db import load
from sandbox.inventory_closure import InventoryClosureLayer
from sandbox.implicit_defects import ImplicitDomainError
from sandbox import transport
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as evaluator
import phase7_inventory_gates as protocol

OUT=protocol.OUT;MODELS=protocol.MODELS
DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
LAYERS={};FITS={};L0,L1=np.log10(600.),np.log10(1950.)

def layer(element):
    if element not in LAYERS:LAYERS[element]=InventoryClosureLayer(load(),element)
    return LAYERS[element]

def point(row,w,method):
    l=layer(row.element);xT=(float(row.growth_T_C)+273.15-1100.)/300.
    tlogit=w[0]
    if method in ("temperature_relation","temperature_element_relation"):tlogit+=w[2]*xT
    if method=="temperature_element_relation":tlogit+=w[3]*(row.element=="Sn")
    st=expit(tlogit);logT=L0+(L1-L0)*st
    if method=="fill_endpoint":eta=1.
    elif method=="fill_const":eta=expit(w[2])
    elif method=="fill_temperature":eta=expit(w[2]+w[3]*xT)
    else:eta=0.
    p=l.inventory_parameters(T=10**logT,Nd=10**w[1],doses=(row.incorporated_cm3,),eta=eta,pO2=1e-5)
    pred=l.predict(p)
    chain=np.zeros((l.base_np+1,len(w)))
    chain[0,0]=(L1-L0)*st*(1-st);chain[1,1]=1.
    if method in ("temperature_relation","temperature_element_relation"):chain[0,2]=chain[0,0]*xT
    if method=="temperature_element_relation":chain[0,3]=chain[0,0]*(row.element=="Sn")
    if method in ("fill_const","fill_temperature"):chain[-1,2]=eta*(1-eta)
    if method=="fill_temperature":chain[-1,3]=eta*(1-eta)*xT
    return pred,pred.jacobian@chain,p

def fit(families,method):
    key=evaluator.digest(dict(families=sorted(families),method=method,
                             protocol=energy.sha(OUT/"protocol.json")))
    path=OUT/"fits"/f"{key}.json"
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    tr=DATA[DATA.family.isin(families)];obs=tr.observed_log_nH.to_numpy(float)
    weights=1/tr.family.map(tr.family.value_counts()).to_numpy(float);weights/=weights.mean()
    sw=np.sqrt(weights);cache={};start=time.perf_counter();failures=[]
    def calc(w):
        k=tuple(float(x) for x in w)
        if k not in cache:
            v=[];j=[]
            for row in tr.itertuples():
                try:p,a,_=point(row,w,method)
                except (ImplicitDomainError,np.linalg.LinAlgError,ValueError) as exc:
                    failures.append(dict(sample_id=row.sample_id,parameters=w.copy(),error=repr(exc)))
                    raise
                v.append(p.values[0]);j.append(a[0])
            cache.clear();cache[k]=(np.asarray(v),np.asarray(j))
        return cache[k]
    def residual(w):return (calc(w)[0]-obs)*sw
    def jacobian(w):return calc(w)[1]*sw[:,None]
    def tinit(T):return logit((np.log10(T)-L0)/(L1-L0))
    n=MODELS[method];lo=[-6.,14.];hi=[6.,18.]
    if n>2:
        lo+=[-6. if method.startswith("fill") else -4.];hi+=[6. if method.startswith("fill") else 4.]
    if n>3:lo+=[-4.];hi+=[4.]
    trials=[]
    for T,nd,eta in ((850.,15.5,.05),(1550.,16.5,.75)):
        initial=[tinit(T),nd]
        if n>2:initial+=[float(logit(eta)) if method.startswith("fill") else 0.]
        if n>3:initial+=[0.]
        try:
            opt=least_squares(residual,initial,jac=jacobian,bounds=(lo,hi),max_nfev=60,
                              ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(parameters=opt.x,loss=float(opt.fun@opt.fun),
                success=bool(opt.success),nfev=int(opt.nfev),optimality=float(opt.optimality),
                active_mask=opt.active_mask,message=opt.message))
        except (ImplicitDomainError,np.linalg.LinAlgError,ValueError) as exc:
            trials.append(dict(parameters=initial,loss=None,success=False,error=repr(exc)))
    good=[t for t in trials if t.get("loss") is not None]
    best=min(good,key=lambda t:t["loss"]) if good else None
    predrows=[]
    for row in DATA.itertuples():
        try:
            if best is None:raise ImplicitDomainError("all_training_starts_failed")
            pred,_,p=point(row,best["parameters"],method);dg=pred.diagnostics
            # Diagnostic only: compare the saturation algebra at the SAME fitted
            # parameters, without another material-label optimization.
            if dg["eta"]>0:
                pp=p.copy();pp[-1]=0.;zero=layer(row.element).predict(pp)
                na=zero.diagnostics["free_n_cm3"]+3*dg["eta"]*dg["vacancy_equivalent_hot_cm3"]
                alg=float(np.log10(na/transport.hall_factor(300.,na)))
            else:alg=float(pred.values[0])
            valid=dg["gradient_valid"] and dg["single_electron_Hall_valid"] and dg["above_carrier_detection_floor"]
            predrows.append(dict(sample_id=row.sample_id,predicted_log_nH=float(pred.values[0]),
                numeric=True,physical_valid=bool(valid),diagnostics=dg,
                saturation_algebraic_log_nH=alg,saturation_algebraic_difference=float(pred.values[0]-alg)))
        except (ImplicitDomainError,np.linalg.LinAlgError,ValueError) as exc:
            predrows.append(dict(sample_id=row.sample_id,predicted_log_nH=None,numeric=False,
                                 physical_valid=False,error=repr(exc)))
    result=dict(key=key,method=method,train_families=sorted(families),train_records=len(tr),
        parameters=best["parameters"] if best else None,trials=trials,
        training_state_failures=failures,predictions=predrows,seconds=time.perf_counter()-start,
        code_protocol_sha256=energy.sha(OUT/"protocol.json"))
    energy.write(path,result);FITS[key]=energy.clean(result)
    print("inventory fit",method,len(families),"loss",best["loss"] if best else None,
          "seconds",round(result["seconds"],2),flush=True)
    return FITS[key]

def worker(method):
    protocol.freeze()
    gate=json.loads((OUT/"numerical_gates.json").read_text());assert gate["passed"]
    evaluator.DATA=DATA;evaluator.ROLES=ROLES
    for held in sorted(DATA.family.unique()):
        destination=OUT/"outer"/f"{evaluator.digest(held)}__{method}.csv"
        if destination.exists():continue
        training=sorted(set(DATA.family)-{held});rows=[];selected=[]
        for budget in (0,1,2):
            candidates=[]
            for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                errors=[]
                for inner in training:
                    f=fit([x for x in training if x!=inner],method)
                    errors.append(evaluator.evaluate(f,inner,budget,lam)[0])
                candidates.append(dict(lam=lam,score=float(np.mean(errors))))
            best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"]))
            f=fit(training,method);score,rs=evaluator.evaluate(f,held,budget,best["lam"])
            lookup={p["sample_id"]:p for p in f["predictions"]}
            for r in rs:
                dp=lookup[r["sample_id"]];dg=dp.get("diagnostics",{})
                r.update(method=method,fit_key=f["key"],eta=dg.get("eta"),Tf_K=dg.get("Tf_K"),
                    saturation_algebraic_log_nH=dp.get("saturation_algebraic_log_nH"),
                    saturation_algebraic_difference=dp.get("saturation_algebraic_difference"))
            rows.extend(rs)
            selected.append(dict(held=held,budget=budget,candidates=candidates,selected=best,
                                 MAE=score,selection_failure=not np.isfinite(best["score"])))
            print("inventory outer",held,method,budget,score,flush=True)
        energy.csvwrite(destination,rows)
        energy.write(OUT/"selections"/f"{evaluator.digest(held)}__{method}.json",selected)
    return method

def summarize():
    paths=sorted((OUT/"outer").glob("*.csv"))
    if not paths:return
    d=pd.concat([read_result_csv(p) for p in paths],ignore_index=True)
    d.to_csv(OUT/"predictions.csv",index=False)
    common=set(ROLES[ROLES.budget.eq(2)&ROLES.role.eq("query")].sample_id);rows=[]
    for (m,b,f),g in d[d.role.eq("query")].groupby(["method","budget","family"]):
        for scope in ("all_query","common18"):
            h=g if scope=="all_query" else g[g.sample_id.isin(common)]
            err=abs(h.predicted_log_nH-h.observed_log_nH)
            rows.append(dict(method=m,budget=int(b),family=f,scope=scope,n=len(h),
                numeric=int(err.notna().sum()),physical=int(h.physical_valid.sum()),
                MAE=float(err.mean()) if err.notna().all() else np.nan,
                conditional_MAE=float(err.mean()),eta_mean=float(h.eta.mean()),
                eta_min=float(h.eta.min()),eta_max=float(h.eta.max()),
                max_abs_saturation_difference=float(abs(h.saturation_algebraic_difference).max())))
    by=pd.DataFrame(rows);by.to_csv(OUT/"by_family.csv",index=False)
    s=by.groupby(["method","budget","scope"]).agg(families=("family","nunique"),
        queries=("n","sum"),numeric=("numeric","sum"),physical=("physical","sum"),
        scored_families=("MAE","count"),MAE=("MAE","mean"),worst=("MAE","max"),
        max_abs_saturation_difference=("max_abs_saturation_difference","max")).reset_index()
    s["full_numeric_coverage"]=s.queries.eq(s.numeric)&s.families.eq(5)
    s.to_csv(OUT/"summary.csv",index=False)
    print(s[s.scope.eq("all_query")].to_string(index=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--method",choices=MODELS);ap.add_argument("--summarize",action="store_true")
    a=ap.parse_args()
    if a.summarize:summarize();return
    protocol.freeze()
    if a.method:worker(a.method)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for method in pool.map(worker,MODELS):print("inventory method complete",method,flush=True)
    summarize()

if __name__=="__main__":main()
