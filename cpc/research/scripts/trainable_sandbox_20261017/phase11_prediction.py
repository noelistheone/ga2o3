"""Train shared parent allocation fields and matched controls on real labels."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os,sys,json,time,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit,logit
from sklearn.linear_model import Ridge
from sandbox.parent_allocation import ParentAllocationLayer
from sandbox.kroger_db import load
import phase9_joint_prediction as prior
import phase2_energy as energy
import phase2_property as old
import phase11_allocation_gates as gates

OUT=gates.OUT;PRED=prior.PRED;DATA=prior.DATA;ROLES=prior.ROLES
FAMILIES=tuple(sorted(DATA.family.unique()))
METHODS={
    "original_RT":dict(field="none",training="RT",thermal=False),
    "thermal_RT":dict(field="none",training="RT",thermal=True),
    "joint_baseline":dict(field="none",training="joint",thermal=True),
    "parent_constant_joint":dict(field="parent",features="constant",training="joint",thermal=True),
    "parent_linear_joint":dict(field="parent",features="linear",training="joint",thermal=True),
    "parent_rbf_joint":dict(field="parent",features="rbf",training="joint",thermal=True),
    "tf_rbf_joint":dict(field="Tf",features="rbf",training="joint",thermal=True),
    "output_rbf_joint":dict(field="output",features="rbf",training="joint",thermal=True),
    "parent_rbf_RT":dict(field="parent",features="rbf",training="RT",thermal=True),
    "output_rbf_RT":dict(field="output",features="rbf",training="RT",thermal=True),
    "dose_activation_RT":dict(field="activation",training="RT"),
    "dose_line_RT":dict(field="dose_line",training="RT"),
    "direct_ridge_RT":dict(field="ridge",training="RT"),
    "direct_ridge_joint":dict(field="ridge",training="joint"),
}
LAYERS={};FITS={}


def grids(method):
    d=METHODS[method]
    if d["field"]=="ridge":return [dict(alpha=a) for a in (.01,.1,1.,10.,100.,1e4)]
    if d["field"] in ("parent","Tf","output"):
        return [dict(reg=0.)] if d["features"]=="constant" else [dict(reg=a) for a in (0.,1e-4,.1,np.inf)]
    return [{}]


def freeze():
    payload=dict(version=1,created_utc=energy.now(),methods={m:dict(**d,configs=grids(m)) for m,d in METHODS.items()},
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),
            ROOT/"src/sandbox/parent_allocation.py",ROOT/"src/sandbox/temperature_defects.py",ROOT/"src/sandbox/material_profile.py",
            ROOT/"docs/ML_SANDBOX_PARENT_LEARNING_20261017.md",ROOT/"scripts/trainable_sandbox_20261017/phase11_learning_gates.py",
            ROOT/"scripts/trainable_sandbox_20261017/phase9_joint_prediction.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (OUT/"numerical_gates.json",OUT/"zero_replays.json",OUT/"learning_gates.json",
            energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",prior.TEMPDIR/"paired_specimens.csv",
            prior.TEMPDIR/"temperature_observations.csv",energy.BASE/"artifact_manifest_phase10.json")},
        loss="equalDOI RT +.1 equalDOI/specimen/readcurve <=275K + reg||psi||²;pureRT onlyRT",
        physical_scope="same supplied873energyinputs,actualSi/Sn+reportedH,unknownH conditionalzero,Cnotmodeled,pO2scenario1e-5,publishedprofile,rH1",
        field="24tanh(phi.psi/24),constant/linear/RBF;parentprocessbias notnewHSE labels",
        RBF="1,x,exp[-(x+1)^2/2],exp[-x²/2],exp[-(x-1)^2/2],Snindicator,(growthTC-800)/200",
        feature_scaling="mean/stdlogC trainRTunique specimens sourceequal only,no label or outer preprocessing",
        controls="samefield modifies logitTf byb/4 or output by-b/ln10;nonmonotone empiricaldosecurve;directRidge",
        optimizer="two starts,max_nfev60,analyticIFT,ftol/xtol/gtol1e-7;all failures retained",
        starts="Tf1400/logB16/E.03/s.2/psi0;Tf1750/logB15.5/E.06/s1/psiintercept8 where present",
        bounds="baseTf logit[-6,6],logB14..18,E0 .003...2,s0..5;psi-24..24;Tf field extra b/4 before sigmoid",
        primary="original29/5/sourceexcluded0-1-2sampleanchors/guard/common18;innerwholeDOI",
        anchor_lambda=[.1,1.,10.,np.inf],temperature="same7oldspecimens;transfer RTselected model/lambda;do not equate to phase10temperaturetuning",
        scope="retrospective,no newphysicalspecimen/DFT/MD/GPUtraining;currentv4andphase10runtimeunchanged")
    p=OUT/"prediction_protocol.json"
    if p.exists():
        x=json.loads(p.read_text())
        for key in ("methods","code_hashes","input_hashes"):assert x[key]==energy.clean(payload[key]),key
    else:energy.write(p,payload)


def training_data(families,method):
    d=PRED[PRED.family.isin(families)]
    rt=d[d.kind.eq("RT")].copy()
    w=1/(rt.groupby("family").sample_id.transform("nunique").to_numpy(float)*rt.family.nunique())
    rt["loss_weight"]=w
    if METHODS[method]["training"]=="RT":return rt
    curve=d[d.kind.eq("curve")&d.Tobs_K.le(275)].copy()
    if len(curve):
        count=curve.groupby("sample_id").obs_id.transform("count").to_numpy(float)
        ns=curve.groupby("family").sample_id.transform("nunique").to_numpy(float)
        curve["loss_weight"]=.1/(count*ns*curve.family.nunique())
    return pd.concat([rt,curve],ignore_index=True).sort_values(["family","sample_id","Tobs_K","kind"])


def feature_metadata(families):
    rt=PRED[PRED.family.isin(families)&PRED.kind.eq("RT")]
    w=1/rt.groupby("family").sample_id.transform("count").to_numpy(float);w/=w.sum()
    c=np.log10(rt.incorporated_cm3.to_numpy(float));center=float(w@c)
    scale=float(np.sqrt(w@((c-center)**2)))
    return dict(logC_center=center,logC_scale=max(scale,1e-12),training_specimens=rt.sample_id.tolist())


def features(row,kind,metadata):
    x=(np.log10(row.incorporated_cm3)-metadata["logC_center"])/metadata["logC_scale"]
    g=(row.growth_T_C-800)/200;sn=float(row.element=="Sn")
    if kind=="constant":return np.ones(1)
    if kind=="linear":return np.array([1.,x,sn,g])
    return np.array([1.,x,np.exp(-.5*(x+1)**2),np.exp(-.5*x*x),np.exp(-.5*(x-1)**2),sn,g])


def layer_for(row,extension):
    elements=(row.element,"H") if row.H_known else (row.element,)
    key=(elements,extension)
    if key not in LAYERS:LAYERS[key]=ParentAllocationLayer(load(),elements,extension=extension)
    return LAYERS[key]


def point(row,w,method,metadata):
    d=METHODS[method];nbase=4 if d["training"]=="joint" else 2
    field=d["field"];hasfield=field in ("parent","Tf","output")
    phi=features(row,d["features"],metadata) if hasfield else np.empty(0)
    v=float(phi@w[nbase:]) if hasfield else 0.
    tanh=np.tanh(v/24);b=24*tanh;db=(1-tanh*tanh)*phi
    st=expit(w[0]+(b/4 if field=="Tf" else 0.));logT=prior.L0+(prior.L1-prior.L0)*st
    T=10**logT;dlogT=(prior.L1-prior.L0)*st*(1-st)
    E=10**w[2] if nbase==4 else .0366;s=w[3] if nbase==4 else 0.
    l=layer_for(row,d["thermal"]);doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,)
    a,J,_=l.point(Tf_K=T,background_cm3=10**w[1],doses=doses,Tobs_K=row.Tobs_K,
        E0_eV=E,s=s,bias=b if field=="parent" else 0.)
    jac=np.zeros((3,len(w)));jac[:,0]=J[:,0]*dlogT;jac[:,1]=J[:,1]
    if nbase==4:jac[:,2]=J[:,2];jac[:,3]=J[:,3]
    values=a.values.copy()
    if field=="parent":jac[:,nbase:]=J[:,4,None]*db[None,:]
    elif field=="Tf":jac[:,nbase:]=J[:,0,None]*dlogT*db[None,:]/4
    elif field=="output":
        values[0]-=b/np.log(10);jac[0,nbase:]=-db/np.log(10)
    return values,jac,a.diagnostics,dict(process_bias=b,field=field,Tf_K=T)


def counters():
    return {k:sum(getattr(l.core,k) for l in LAYERS.values()) for k in
            ("root_calls","hot_solve_calls","hot_cache_hits","root_evaluations")}


def activation_curve(p,x):
    x=np.asarray(x,float);z=p[3]*(x-p[2]);s=expit(np.log(10)*z)
    vals=p[0]+p[1]*x-np.logaddexp(0,np.log(10)*z)/np.log(10)
    return vals,np.c_[np.ones(len(x)),x,p[3]*s,-(x-p[2])*s]


def canonical(method,config):
    if config.get("reg") is not None and np.isinf(config["reg"]):
        return ("joint_baseline" if METHODS[method]["training"]=="joint" else "thermal_RT"),{}
    return method,config


def fit(families,method,config):
    method,config=canonical(method,config);families=tuple(sorted(families))
    key=old.digest(dict(protocol=energy.sha(OUT/"prediction_protocol.json"),families=families,method=method,config=energy.clean(config)))
    path=OUT/"fits"/f"{key}.json"
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    d=METHODS[method];tr=training_data(families,method);weights=tr.loss_weight.to_numpy(float)
    y=tr.target.to_numpy(float);sw=np.sqrt(weights);meta=feature_metadata(families)
    clock=time.perf_counter();before=counters();trials=[];failures=[];parameters=None;origin=None
    field=d["field"];nbase=4 if d["training"]=="joint" else 2
    if field=="ridge":
        X=prior.features(tr);center=np.average(X,axis=0,weights=weights)
        scale=np.sqrt(np.average((X-center)**2,axis=0,weights=weights));scale[scale<1e-12]=1.
        m=Ridge(**config).fit((X-center)/scale,y,sample_weight=weights*len(tr)/weights.sum())
        empirical=m.predict((prior.features(PRED)-center)/scale)
        meta.update(center=center,scale=scale,coefficients=m.coef_,intercept=m.intercept_)
    elif field=="dose_line":
        X=np.c_[np.ones(len(tr)),np.log10(tr.incorporated_cm3)-18]
        p=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
        empirical=p[0]+p[1]*(np.log10(PRED.incorporated_cm3)-18);meta["coefficients"]=p
    elif field=="activation":
        x=np.log10(tr.incorporated_cm3.to_numpy(float))-18
        for initial in ([18.,1.,1.,2.],[18.,1.,1.,10.]):
            op=least_squares(lambda p:(activation_curve(p,x)[0]-y)*sw,initial,jac=lambda p:activation_curve(p,x)[1]*sw[:,None],
                bounds=([12.,0.,-3.,0.],[22.,2.,4.,30.]),max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trials.append(dict(parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),nfev=int(op.nfev),message=op.message))
        parameters=min(trials,key=lambda t:t["loss"])["parameters"]
        empirical=activation_curve(parameters,np.log10(PRED.incorporated_cm3.to_numpy(float))-18)[0]
    else:
        k=0 if field=="none" else (1 if d["features"]=="constant" else (4 if d["features"]=="linear" else 7))
        lo=[-6.,14.]+([np.log10(.003),0.] if nbase==4 else [])+[-24.]*k
        hi=[6.,18.]+([np.log10(.2),5.] if nbase==4 else [])+[24.]*k
        reg=config.get("reg",0.)
        # Exact oldRT objective differs only by a positive constant scale.
        if method=="original_RT":
            for source in (prior.OUT/"fits").glob("*.json"):
                f=json.loads(source.read_text())
                if f["method"]=="no_extension_RT" and set(f["train_families"])==set(families):
                    parameters=f["parameters"]
                    trials=[{**t,"loss":None if t.get("loss") is None else t["loss"]/len(tr),"source_loss":t.get("loss")} for t in f["trials"]]
                    origin=dict(path=str(source.relative_to(ROOT)),sha256=energy.sha(source),objective_equivalence="positive constant scale only")
                    break
        if parameters is None:
            cache={}
            def calc(p):
                key=tuple(p)
                if key not in cache:
                    vals=[];js=[]
                    for row in tr.itertuples():
                        try:v,j,_,_=point(row,p,method,meta)
                        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                            failures.append(dict(obs_id=row.obs_id,parameters=p,error=repr(exc)));raise
                        vals.append(v[0]);js.append(j[0])
                    cache.clear();cache[key]=(np.asarray(vals),np.asarray(js))
                return cache[key]
            for T,B,E,s,intercept in ((1400.,16.,.03,.2,0.),(1750.,15.5,.06,1.,8.)):
                p=[logit((np.log10(T)-prior.L0)/(prior.L1-prior.L0)),B]
                if nbase==4:p+=[np.log10(E),s]
                p+=([intercept]+[0.]*(k-1)) if k else []
                def residual(p):return np.r_[(calc(p)[0]-y)*sw,np.sqrt(reg)*np.asarray(p[nbase:])]
                def jacobian(p):
                    penalty=np.zeros((k,nbase+k));penalty[:,nbase:]=np.sqrt(reg)*np.eye(k)
                    return np.vstack([calc(p)[1]*sw[:,None],penalty])
                try:
                    op=least_squares(residual,p,jac=jacobian,bounds=(lo,hi),max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                    trials.append(dict(parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                        nfev=int(op.nfev),active_mask=op.active_mask,message=op.message))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    trials.append(dict(parameters=p,loss=None,success=False,error=repr(exc)))
            finite=[t for t in trials if t.get("loss") is not None]
            parameters=min(finite,key=lambda t:t["loss"])["parameters"] if finite else None
    predictions=[]
    for i,row in enumerate(PRED.itertuples()):
        if field in ("ridge","dose_line","activation"):
            predictions.append(dict(obs_id=row.obs_id,predicted=float(empirical[i]),numeric=True,physical_valid=None))
        else:
            try:
                if parameters is None:raise RuntimeError("all_fit_starts_failed")
                v,j,dg,extra=point(row,parameters,method,meta)
                predictions.append(dict(obs_id=row.obs_id,predicted=float(v[0]),numeric=True,
                    physical_valid=None if field=="output" else bool(dg["gradient_valid"] and dg["single_electron_Hall_valid"]),
                    diagnostics=dg,extra=extra,charge_outputs=v[1:]))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                predictions.append(dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc)))
    after=counters()
    f=dict(key=key,method=method,config=config,train_families=families,parameters=parameters,trials=trials,
        state_failures=failures,training_obs_ids=tr.obs_id.tolist(),weight_per_observation=weights,
        training_specimens=sorted(tr.sample_id.unique()),feature_metadata=meta,origin=origin,
        predictions=predictions,seconds=time.perf_counter()-clock,counters={k:after[k]-before[k] for k in before})
    energy.write(path,f);FITS[key]=energy.clean(f)
    print("parentfit",method,config,len(families),"seconds",round(f["seconds"],2),"finite",parameters is not None or field in ("ridge","dose_line"),flush=True)
    return FITS[key]


def evaluate(f,held,budget,lam):
    return prior.primary(f,held,budget,lam)


def selected(method):
    rows=[];temp=[];choices=[]
    for held in FAMILIES:
        train=[f for f in FAMILIES if f!=held]
        for budget in (0,1,2):
            candidates=[]
            for i,c in enumerate(grids(method)):
                for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                    scores=[];keys=[]
                    for inner in train:
                        f=fit([x for x in train if x!=inner],method,c)
                        scores.append(evaluate(f,inner,budget,lam)[0]);keys.append(f["key"])
                    candidates.append(dict(index=i,config=c,lam=lam,score=float(np.mean(scores)),inner_keys=keys))
            best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"],c["index"]))
            f=fit(train,method,best["config"]);score,rs=evaluate(f,held,budget,best["lam"])
            for r in rs:r.update(method=method,fit_key=f["key"])
            rows+=rs
            for sample in prior.SPECS[prior.SPECS.family.eq(held)].index:
                rs=prior.temperature(f,sample,budget,best["lam"])
                for r in rs:r.update(method=method,fit_key=f["key"])
                temp+=rs
            choices.append(dict(method=method,held=held,budget=budget,candidates=candidates,selected=best,fit_key=f["key"],score=score))
            print("parentouter",method,held,budget,round(score,6),flush=True)
    folder=OUT/"methods"/method
    energy.csvwrite(folder/"primary_predictions.csv",rows);energy.csvwrite(folder/"temperature_predictions.csv",temp)
    energy.write(folder/"selections.json",choices)


def one(args):
    families,method,config=args;return fit(families,method,config)["key"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4);a=ap.parse_args();freeze()
    assert json.loads((OUT/"numerical_gates.json").read_text())["passed"]
    assert json.loads((OUT/"zero_replays.json").read_text())["passed"]
    assert json.loads((OUT/"learning_gates.json").read_text())["passed"]
    jobs={}
    for n in (3,4):
        for families in combinations(FAMILIES,n):
            for method in METHODS:
                for config in grids(method):
                    m,c=canonical(method,config);key=old.digest(dict(families=families,method=m,config=energy.clean(c)))
                    jobs[key]=(families,m,c)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(one,jobs.values()):pass
    print("parent fits completed",len(jobs),flush=True)
    for method in METHODS:selected(method)


if __name__=="__main__":main()
