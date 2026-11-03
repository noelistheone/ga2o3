"""Equal-label-budget energy reconstruction -> source-held-out nH prediction.

Only the target element's declared mono-single nonreference energies are masked.
All remaining database energies, charge catalogues and prefactors are supplied.
This is conditional energy completion, not prediction of an unknown dopant.
"""
from pathlib import Path
from itertools import combinations
import sys, os, json, csv, hashlib, argparse, time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
from sandbox.implicit_defects import ImplicitDefectLayer, ImplicitDomainError
from sandbox.energy_learning import reconstruct
from csv_io import read_result_csv
import phase2_energy as energy

BASE=energy.BASE
OUT=energy.OUT/"property"
METHODS=("oracle_full","global_mean","charge_pair_mean","ridge","rbf","extra_trees")
L0,L1=np.log10(600.),np.log10(1950.)
FITS={}
DBS={}
LAYERS={}
DATA=read_result_csv(BASE/"dataset_v1.csv")
ROLES=read_result_csv(BASE/"episodes_v1.csv")

def write(p,x):energy.write(p,x)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:20]
def freeze():
    energy.frozen()
    path=OUT/"protocol.json"
    payload=dict(version=1,created_utc=energy.now(),
        parent_protocol_sha256=energy.sha(energy.OUT/"protocol.json"),
        code_sha256=energy.sha(Path(__file__)),methods=METHODS,
        root_validity="same ImplicitDefectLayer regularity/site gates as phase1; record every failure",
        process_fit="family-equal squared nH log loss; bounds logTf600..1950,logNd14..18; two starts; max_nfev60",
        adaptation_lambdas=[.1,1.,10.,"infinity"],
        adaptation_formula="delta=sum(anchor residual)/(m+lambda); infinity -> zero",
        hyper_selection="inner leave training source out, separately per anchor count; ties prefer stronger shrinkage",
        family_selection="same five DOI families and 0/1/2 anchor roles as phase1",
        energy_label_budget="Si844/873 supplied, Sn850/873 supplied; exact A_ref/topology/all unmasked energies given to every method",
        oracle="873/873 labels, separate full-information reference, not same-budget competitor",
        known_energy_references="kept out of energy learner features, used only to reconstruct A",
        missing_protocols="all pO2=1e-5 atm common unmeasured scenario; source T metadata proxies retained",
        incomplete_coverage="report refusals separately; conditional MAE cannot support full-coverage superiority",
        uncertainty="anchor offset is a point adaptation; no claim of calibrated posterior intervals",
        grade="retrospective, not blind or independent materials confirmation")
    if path.exists():
        old=json.loads(path.read_text())
        assert old["code_sha256"]==payload["code_sha256"]
        assert old["parent_protocol_sha256"]==payload["parent_protocol_sha256"]
    else:write(path,payload)

def database(element,method):
    key=(element,method)
    if key in DBS:return DBS[key]
    db,rr,_=energy.records()
    if method!="oracle_full":
        p=read_result_csv(energy.OUT/"energy_predictions.csv")
        p=p[p.element.eq(element)&p.method.eq(method)]
        target=[r for r in rr if r["element"]==element]
        lookup=dict(zip(p.label_id,p.predicted_crossing_eV))
        if set(lookup)!={r["label_id"] for r in target}:
            raise RuntimeError("Energy prediction catalogue incomplete")
        db=reconstruct(db,target,[lookup[r["label_id"]] for r in target])
    DBS[key]=db
    return db

def point(row,w,target_element,method):
    key=(target_element,method,row.element)
    if key not in LAYERS:
        LAYERS[key]=ImplicitDefectLayer(database(target_element,method),(row.element,))
    layer=LAYERS[key]
    s=expit(w[0]);logT=L0+(L1-L0)*s
    pars=layer.parameters(T=10**logT,Nd=10**w[1],doses=(row.incorporated_cm3,),pO2=1e-5)
    pred=layer.predict(pars)
    chain=np.zeros((layer.np,2))
    chain[0,0]=(L1-L0)*s*(1-s);chain[1,1]=1.
    return pred,pred.jacobian@chain

def fit(families,element,method):
    # The oracle database is independent of which element is designated missing.
    target="Si" if method=="oracle_full" else element
    key=digest(dict(families=sorted(families),element=target,method=method,
                    protocol=energy.sha(OUT/"protocol.json")))
    if key in FITS:return FITS[key]
    path=OUT/"fits"/f"{key}.json"
    if path.exists():
        f=json.loads(path.read_text());FITS[key]=f;return f
    tr=DATA[DATA.family.isin(families)]
    weights=1/tr.family.map(tr.family.value_counts()).to_numpy(float)
    weights/=weights.mean();sw=np.sqrt(weights)
    obs=tr.observed_log_nH.to_numpy(float)
    def transform_T(T):return float(np.log((np.log10(T)-L0)/(L1-np.log10(T))))
    trials=[];start=time.perf_counter();cache={}
    def calc(w):
        k=tuple(float(x) for x in w)
        if k not in cache:
            v=[];j=[]
            for row in tr.itertuples():
                p,a=point(row,w,target,method)
                v.append(p.values[0]);j.append(a[0])
            cache.clear();cache[k]=(np.array(v),np.array(j))
        return cache[k]
    def residual(w):return (calc(w)[0]-obs)*sw
    def jacobian(w):return calc(w)[1]*sw[:,None]
    for initial in ([transform_T(850.),15.5],[transform_T(1550.),16.5]):
        try:
            opt=least_squares(residual,initial,jac=jacobian,bounds=([-6.,14.],[6.,18.]),
                max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(parameters=opt.x,loss=float(opt.fun@opt.fun),
                success=bool(opt.success),nfev=int(opt.nfev),message=opt.message))
        except (ImplicitDomainError,np.linalg.LinAlgError,ValueError) as exc:
            trials.append(dict(parameters=initial,loss=None,success=False,error=repr(exc)))
    good=[t for t in trials if t.get("loss") is not None]
    best=min(good,key=lambda t:t["loss"]) if good else None
    predrows=[]
    if best is not None:
        for row in DATA.itertuples():
            try:
                pred,j=point(row,best["parameters"],target,method);dg=pred.diagnostics
                valid=(dg["gradient_valid"] and dg["single_electron_Hall_valid"]
                       and dg["above_carrier_detection_floor"])
                predrows.append(dict(sample_id=row.sample_id,predicted_log_nH=pred.values[0],
                    physical_valid=bool(valid),numeric=True,diagnostics=dg))
            except (ImplicitDomainError,np.linalg.LinAlgError,ValueError) as exc:
                predrows.append(dict(sample_id=row.sample_id,predicted_log_nH=None,
                    physical_valid=False,numeric=False,error=repr(exc)))
    else:
        predrows=[dict(sample_id=row.sample_id,predicted_log_nH=None,
            physical_valid=False,numeric=False,error="all_training_starts_failed") for row in DATA.itertuples()]
    f=dict(key=key,method=method,masked_element=target,train_families=sorted(families),
        parameters=best["parameters"] if best is not None else None,trials=trials,
        predictions=predrows,seconds=time.perf_counter()-start)
    write(path,f);FITS[key]=energy.clean(f)
    print("property fit",method,target,len(families),"valid",best is not None,
          "sec",round(f["seconds"],2),flush=True)
    return FITS[key]

def evaluate(f,family,budget,lam):
    d=DATA[DATA.family.eq(family)].copy()
    roles=ROLES[ROLES.family.eq(family)&ROLES.budget.eq(budget)]
    role=dict(zip(roles.sample_id,roles.role))
    lookup={r["sample_id"]:r for r in f["predictions"]}
    values=np.array([lookup[r.sample_id]["predicted_log_nH"]
                     if lookup[r.sample_id]["predicted_log_nH"] is not None else np.nan
                     for r in d.itertuples()])
    anchor=np.array([role[s]=="anchor" for s in d.sample_id])
    query=np.array([role[s]=="query" for s in d.sample_id])
    delta=0.
    if anchor.any() and np.isfinite(lam):
        if np.all(np.isfinite(values[anchor])):
            delta=float((d.observed_log_nH.to_numpy(float)[anchor]-values[anchor]).sum()/(anchor.sum()+lam))
        else:
            values[:]=np.nan
    values+=delta
    complete=bool(np.isfinite(values[query]).all())
    mae=float(abs(values[query]-d.observed_log_nH.to_numpy(float)[query]).mean()) if complete else np.inf
    out=[]
    for row,pred in zip(d.itertuples(),values):
        out.append(dict(sample_id=row.sample_id,family=family,element=row.element,
            role=role[row.sample_id],budget=budget,
            observed_log_nH=row.observed_log_nH,
            predicted_log_nH=float(pred) if np.isfinite(pred) else None,
            physical_valid=lookup[row.sample_id]["physical_valid"],
            offset_dex=delta,shrinkage_lambda=lam if np.isfinite(lam) else "infinity"))
    return mae,out

def run(only=None):
    freeze()
    families=sorted(DATA.family.unique())
    methods=METHODS if only is None else (only,)
    for held in families:
        element=DATA.loc[DATA.family.eq(held),"element"].iloc[0]
        training=[f for f in families if f!=held]
        for method in methods:
            target=OUT/"outer"/f"{digest(held)}__{method}.csv"
            if target.exists():continue
            result=[];selection=[]
            for budget in (0,1,2):
                candidates=[]
                for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                    errors=[]
                    for inner in training:
                        f=fit([x for x in training if x!=inner],element,method)
                        errors.append(evaluate(f,inner,budget,lam)[0])
                    score=float(np.mean(errors))
                    candidates.append(dict(lam=lam,score=score))
                best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"]))
                f=fit(training,element,method)
                score,rows=evaluate(f,held,budget,best["lam"])
                for row in rows:row.update(method=method,masked_element=element,fit_key=f["key"])
                result.extend(rows)
                selection.append(dict(budget=budget,candidates=candidates,selected=best,
                    target_families=[held],query_MAE=score,
                    selection_failure=not np.isfinite(best["score"])))
                print("property outer",held,method,budget,"MAE",score,flush=True)
            energy.csvwrite(target,result)
            write(OUT/"selections"/f"{digest(held)}__{method}.json",selection)
    summarize()

def summarize():
    paths=sorted((OUT/"outer").glob("*.csv"))
    if not paths:return
    d=pd.concat([read_result_csv(p) for p in paths],ignore_index=True)
    d.to_csv(OUT/"predictions.csv",index=False)
    rows=[]
    common=set(ROLES[ROLES.budget.eq(2)&ROLES.role.eq("query")].sample_id)
    oracle=d[d.method.eq("oracle_full")].set_index(["sample_id","budget"]).predicted_log_nH.to_dict()
    for (m,b,f),g in d[d.role.eq("query")].groupby(["method","budget","family"]):
        for scope in ("all_query","common18"):
            h=g if scope=="all_query" else g[g.sample_id.isin(common)]
            err=h.predicted_log_nH-h.observed_log_nH
            diff=np.array([r.predicted_log_nH-oracle.get((r.sample_id,r.budget),np.nan)
                           for r in h.itertuples()])
            rows.append(dict(method=m,budget=int(b),family=f,scope=scope,n=len(h),
                numeric=int(h.predicted_log_nH.notna().sum()),
                physical=int(h.physical_valid.sum()),
                MAE=float(abs(err).mean()) if err.notna().all() else None,
                conditional_numeric_MAE=float(abs(err).mean()),
                oracle_prediction_MAE=float(np.abs(diff).mean()) if np.isfinite(diff).all() else None))
    by=pd.DataFrame(rows)
    by.to_csv(OUT/"by_family.csv",index=False)
    summary=by.groupby(["method","budget","scope"]).agg(
        families=("family","nunique"),query_records=("n","sum"),numeric=("numeric","sum"),
        physical=("physical","sum"),scored_families=("MAE","count"),
        family_equal_MAE=("MAE","mean"),worst_family_MAE=("MAE","max"),
        oracle_prediction_MAE=("oracle_prediction_MAE","mean")).reset_index()
    summary["full_coverage"]=summary.numeric.eq(summary.query_records)&summary.families.eq(5)
    # Scores conditional on missing predictions must be visibly marked.
    summary.to_csv(OUT/"summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--initialize",action="store_true")
    ap.add_argument("--method",choices=METHODS)
    ap.add_argument("--summarize",action="store_true")
    a=ap.parse_args()
    if a.initialize:freeze()
    elif a.summarize:summarize()
    else:run(a.method)
