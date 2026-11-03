"""Same-label-budget local physical adaptation, retaining the original RT task."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os,sys,json,time,argparse
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(name,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit,logit
from sklearn.linear_model import Ridge
from sandbox.parameter_adaptation import regularized_adapt,local_information
from sandbox.implicit_defects import ImplicitDomainError
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as old
import phase9_joint_prediction as prior

OUT=energy.BASE/"phase10_internal_adaptation"
LAMBDAS=(.001,.01,.1,1.,10.,np.inf)
ALPHAS=(.01,.1,1.,10.,100.,1e4)
ESCAL=(.3,.1,.6,.05,.2,1.2)
FAMILIES=tuple(sorted(prior.DATA.family.unique()))
PRED=prior.PRED.copy();ROLES=prior.ROLES.copy()
METHODS={
    "original_RT_offset":("no_extension_RT","offset"),
    "original_RT_Tf":("no_extension_RT","Tf"),
    "fixed_joint_offset":("fixed_binding_joint","offset"),
    "fixed_joint_Tf_E":("fixed_binding_joint","Tf_E"),
    "shared_joint_offset":("learned_screen_joint","offset"),
    "shared_joint_Tf":("learned_screen_joint","Tf"),
    "shared_joint_E":("learned_screen_joint","E"),
    "shared_joint_Tf_E":("learned_screen_joint","Tf_E"),
    "direct_joint_offset":("direct_ridge_joint","offset"),
    "direct_joint_affine":("direct_ridge_joint","affine"),
    "dose_line_offset":("dose_line_RT","offset"),
}
BASES={};CALIB={};OLD_INDEX=None


def grids(method):
    base,mode=METHODS[method]
    if base.startswith("direct"):return [dict(alpha=a) for a in ALPHAS]
    if mode=="Tf_E":return [dict(E_scale=e) for e in ESCAL]
    return [{}]


def freeze():
    path=OUT/"protocol.json"
    sources=[Path(__file__),ROOT/"scripts/trainable_sandbox_20261017/phase10_adaptation_gates.py",
        ROOT/"src/sandbox/parameter_adaptation.py",
        ROOT/"src/sandbox/temperature_defects.py",ROOT/"src/sandbox/material_profile.py",
        ROOT/"scripts/trainable_sandbox_20261017/phase9_joint_prediction.py",
        ROOT/"docs/ML_SANDBOX_PARAMETER_ADAPTATION_20261017.md"]
    p=dict(version=1,created_utc=energy.now(),methods={m:dict(base=b,mode=s,configs=grids(m)) for m,(b,s) in METHODS.items()},
        lambdas=LAMBDAS,code_hashes={str(f.relative_to(ROOT)):energy.sha(f) for f in sources},
        input_hashes={str(f.relative_to(ROOT)):energy.sha(f) for f in (
            prior.OUT/"prediction_protocol.json",energy.BASE/"artifact_manifest_phase9.json",
            energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
            prior.TEMPDIR/"temperature_observations.csv",prior.TEMPDIR/"paired_specimens.csv")},
        base_training="reuse exact phase9 full-family fits where available;new partial-specimen fits use same loss/two starts/60nfev",
        state="same published profile/rH1/full supplied catalogue/H/C/pressure scenarios as phase9;no changed physics branch",
        local_coordinates="log10 effective Tf and log10 effective E0;B,s and energetic labels fixed to base fit",
        local_scales=dict(log10_Tf=.05,log10_E0=ESCAL,single_E=.3),
        bounds=dict(logit_Tf=[-6,6],Tf_base_limits=[600,1950],E0_eV=[.003,.2]),
        local_loss="sum anchor log10_nH residual squared + lambda*standardized_shift squared;lambda is predictive regularization,not measured noise",
        starts="zero shift and clipped linearMAP;each max_nfev30;finite lowest loss;all failures/nonconverged retained",
        fallback="all local starts fail:explicit base_fallback status and unadapted predictions;query root failures remain missing,not dropped",
        direct_affine="baseRidge plus regularized intercept and train-standardized logdose(primary) or inverseT(temperature) residual slope",
        primary="unchanged29/5,0-1-2specimen anchors per heldDOI,query29/23/18,common18;innerwholeDOI exclusion",
        temperature="same7oldSi specimens/2DOI,RT+nearest150K per-specimen anchors and guards;common80 and all142/138/131",
        temperature_inner="leave one specimen of available training temperature DOI out;remove ALL its RT/curve labels from base fit;all outerDOI labels excluded",
        tuning="primary inner-source MAE;temperature inner-specimen common40..125MAE;tuning schemes differ from stage9RTlambda transfer",
        selection="ties at1e-10 prefer infinite/larger lambda then earlier config;max36candidate pairs,actualcounts/costs reported",
        limitations="retrospective previously exposed sources;inner temperature has onlyoneDOI;MAP effective coordinates are not microscopic truth or new ML algorithm",
        main_endpoint_preserved=True,GPU_policy="CPU small root/calibration tasks;only GPU1 authorized if needed")
    if path.exists():
        before=json.loads(path.read_text())
        for k in ("code_hashes","input_hashes","methods"):assert before[k]==energy.clean(p[k]),k
    else:energy.write(path,p)


def old_index():
    global OLD_INDEX
    if OLD_INDEX is None:
        OLD_INDEX={}
        for p in (prior.OUT/"fits").glob("*.json"):
            f=json.loads(p.read_text())
            OLD_INDEX[(f["method"],tuple(f["train_families"]),json.dumps(f["config"],sort_keys=True))]=(p,f)
    return OLD_INDEX


def base_fit(families,method,config=None,dropped=()):
    families=tuple(sorted(families));dropped=tuple(sorted(dropped));config=config or {}
    config={"alpha":config["alpha"]} if method.startswith("direct") else {}
    key=old.digest(dict(parent=energy.sha(OUT/"protocol.json"),families=families,
        method=method,config=config,dropped=dropped))
    if key in BASES:return BASES[key]
    oldkey=(method,families,json.dumps(config,sort_keys=True))
    if not dropped and oldkey in old_index():
        path,f=old_index()[oldkey];BASES[key]=(path,f);return path,f
    path=OUT/"base_fits"/f"{key}.json"
    if path.exists():
        f=json.loads(path.read_text());BASES[key]=(path,f);return path,f
    tr,w=prior.training_data(families,method)
    tr=tr[~tr.sample_id.isin(dropped)].copy()
    w=1/(tr.groupby("sample_id").obs_id.transform("count").to_numpy(float)*
         tr.groupby("family").sample_id.transform("nunique").to_numpy(float));w/=w.mean()
    y=tr.target.to_numpy(float);sw=np.sqrt(w);start=time.perf_counter()
    trials=[];failures=[];metadata={};parameters=None
    if method=="dose_line_RT":
        X=np.c_[np.ones(len(tr)),np.log10(tr.incorporated_cm3)-18]
        b=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
        prediction=b[0]+b[1]*(np.log10(PRED.incorporated_cm3)-18)
        metadata["coefficients"]=b
    elif method.startswith("direct"):
        X=prior.features(tr);center=np.average(X,axis=0,weights=w)
        scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
        model=Ridge(**config).fit((X-center)/scale,y,sample_weight=w)
        prediction=model.predict((prior.features(PRED)-center)/scale)
        metadata.update(center=center,scale=scale,coefficients=model.coef_,intercept=model.intercept_)
    else:
        n=4 if method=="learned_screen_joint" else 2
        lo=[-6.,14.]+([np.log10(.003),0.] if n==4 else [])
        hi=[6.,18.]+([np.log10(.2),5.] if n==4 else [])
        cache={}
        def calc(v):
            k=tuple(v)
            if k not in cache:
                vals=[];js=[]
                for r in tr.itertuples():
                    try:a,j=prior.point(r,v,method)
                    except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                        failures.append(dict(obs_id=r.obs_id,parameters=v,error=repr(exc)));raise
                    vals.append(a.values[0]);js.append(j[0])
                cache.clear();cache[k]=(np.asarray(vals),np.asarray(js))
            return cache[k]
        for T,B,E,s in ((1400.,16.,.03,.2),(1750.,15.5,.06,1.)):
            v=[logit((np.log10(T)-prior.L0)/(prior.L1-prior.L0)),B]
            if n==4:v += [np.log10(E),s]
            try:
                opt=least_squares(lambda v:(calc(v)[0]-y)*sw,v,
                    jac=lambda v:calc(v)[1]*sw[:,None],bounds=(lo,hi),max_nfev=60,
                    ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(parameters=opt.x,loss=float(opt.fun@opt.fun),success=bool(opt.success),
                    nfev=int(opt.nfev),active_mask=opt.active_mask,message=opt.message))
            except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                trials.append(dict(parameters=v,loss=None,success=False,error=repr(exc)))
        good=[t for t in trials if t.get("loss") is not None]
        parameters=min(good,key=lambda t:t["loss"])["parameters"] if good else None
        predictions=[]
        for r in PRED.itertuples():
            try:
                if parameters is None:raise ImplicitDomainError("all_base_starts_failed")
                a,j=prior.point(r,parameters,method);dg=a.diagnostics
                predictions.append(dict(obs_id=r.obs_id,predicted=float(a.values[0]),numeric=True,
                    physical_valid=bool(dg["gradient_valid"] and dg["single_electron_Hall_valid"]),
                    above_reference_floor=dg["above_carrier_detection_floor"],diagnostics=dg))
            except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                predictions.append(dict(obs_id=r.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc)))
    if method.startswith(("direct","dose")):
        predictions=[dict(obs_id=r.obs_id,predicted=float(v),numeric=True,physical_valid=None)
                     for r,v in zip(PRED.itertuples(),prediction)]
    f=dict(key=key,method=method,config=config,train_families=families,dropped_specimens=dropped,
        training_obs_ids=tr.obs_id.tolist(),training_specimens=sorted(tr.sample_id.unique()),
        training_observations=len(tr),weight_per_observation=w,parameters=parameters,
        trials=trials,state_failures=failures,metadata=metadata,predictions=predictions,
        seconds=time.perf_counter()-start)
    energy.write(path,f);f=energy.clean(f);BASES[key]=(path,f)
    print("basefit",method,"dropped",len(dropped),"seconds",round(f["seconds"],2),flush=True)
    return path,f


def physical_point(row,base,theta):
    extension=base["method"]!="no_extension_RT"
    layer=prior.layer_for(row,extension);w=base["parameters"]
    s=w[3] if base["method"]=="learned_screen_joint" else 0.
    doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,)
    p=layer.parameters(T=10**theta[0],Nd=10**w[1],doses=doses,Tobs=row.Tobs_K,
        binding_eV=10**theta[1],screen_s=s)
    a=layer.predict(p);J=np.zeros((3,2));J[:,0]=a.jacobian[:,0]
    if extension:J[:,1]=a.jacobian[:,layer.ie]
    return a,J


def initial_coordinates(base):
    w=base["parameters"]
    return np.array([prior.L0+(prior.L1-prior.L0)*expit(w[0]),
        w[2] if base["method"]=="learned_screen_joint" else np.log10(.0366)])


def episode(task,context,budget):
    if task=="primary":
        d=PRED[PRED.family.eq(context)&PRED.kind.eq("RT")].copy()
        r=ROLES[ROLES.family.eq(context)&ROLES.budget.eq(budget)].set_index("sample_id").role
        d["role"]=[r.loc[s] for s in d.sample_id]
    else:
        d=PRED[PRED.sample_id.eq(context)].copy();a=[]
        if budget:a=[d[d.kind.eq("RT")].iloc[0].obs_id]
        if budget==2:
            q=d[d.kind.eq("curve")&d.Tobs_K.between(125,175)]
            a.append(q.loc[abs(q.Tobs_K-150).sort_values(kind="stable").index[0]].obs_id)
        d["role"]=["unavailable_reference" if k=="RT" else "query" for k in d.kind]
        for i,r in d.iterrows():
            if r.obs_id in a:d.loc[i,"role"]="anchor"
            elif budget and abs(r.Tobs_K-300)<=25:d.loc[i,"role"]="guard"
            if budget==2 and r.obs_id not in a and abs(r.Tobs_K-float(d[d.obs_id.eq(a[1])].Tobs_K.iloc[0]))<=10:
                d.loc[i,"role"]="guard"
    return d


def calibrate(task,context,budget,method,config,lam,path,base):
    mode=METHODS[method][1]
    key=old.digest(dict(protocol=energy.sha(OUT/"protocol.json"),task=task,context=context,
        budget=budget,method=method,config=config,lam=energy.clean(lam),base=base["key"]))
    dest=OUT/"calibrations"/method/f"{key}.json"
    if key in CALIB:return CALIB[key]
    if dest.exists():
        v=json.loads(dest.read_text());CALIB[key]=v;return v
    start=time.perf_counter();d=episode(task,context,budget);a=d[d.role.eq("anchor")]
    raw={x["obs_id"]:x for x in base["predictions"]};status="base_no_update"
    diagnostics={};trials=[];params=None;predictions=[]
    if mode in ("offset","affine"):
        train=PRED[PRED.obs_id.isin(base["training_obs_ids"])]
        w=np.asarray(base["weight_per_observation"],float)
        assert train.obs_id.tolist()==base["training_obs_ids"]
        if mode=="affine":
            values=(np.log10(train.incorporated_cm3.to_numpy(float)) if task=="primary"
                    else 300/train.Tobs_K.to_numpy(float)-1)
            center=float(np.average(values,weights=w)) if task=="primary" else 0.
            scale=float(np.sqrt(np.average((values-center)**2,weights=w)))
            scale=max(scale,1e-12)
            v=(np.log10(d.incorporated_cm3.to_numpy(float)) if task=="primary"
               else 300/d.Tobs_K.to_numpy(float)-1)
            X=np.c_[np.ones(len(d)),(v-center)/scale]
            diagnostics.update(feature_center=center,feature_scale=scale)
        else:X=np.ones((len(d),1))
        mask=d.role.eq("anchor").to_numpy();J=X[mask]
        y=np.array([raw[x]["predicted"] for x in a.obs_id],float)
        b=np.linalg.solve(J.T@J+lam*np.eye(X.shape[1]),J.T@(a.target.to_numpy(float)-y)) if budget and np.isfinite(lam) else np.zeros(X.shape[1])
        status="adapted_analytic" if budget and np.isfinite(lam) else "base_no_update"
        diagnostics.update(local_information(J,np.ones(X.shape[1]),lam));params=b
        for r,change in zip(d.itertuples(),X@b):
            v=raw[r.obs_id]["predicted"]
            predictions.append(dict(obs_id=r.obs_id,predicted=None if v is None else v+float(change),
                raw_prediction=v,physical_valid=None,numeric=v is not None))
    else:
        theta0=initial_coordinates(base)
        D=np.array([.05 if mode in ("Tf","Tf_E") else 0.,
                    config.get("E_scale",.3) if mode in ("E","Tf_E") else 0.])
        lo=np.array([prior.L0+(prior.L1-prior.L0)*expit(-6),np.log10(.003)])
        hi=np.array([prior.L0+(prior.L1-prior.L0)*expit(6),np.log10(.2)])
        def forward(theta):
            values=[];J=[]
            for r in a.itertuples():
                p,j=physical_point(r,base,theta);values.append(p.values[0]);J.append(j[0])
            return np.asarray(values),np.asarray(J).reshape(len(a),2)
        fit=regularized_adapt(forward,a.target.to_numpy(float),theta0,D,lo,hi,lam,max_nfev=30)
        status=fit.status;diagnostics=fit.diagnostics;trials=fit.trials;params=fit.parameters
        diagnostics.update(initial_parameters=theta0,scales=D,standardized_shift=fit.standardized_shift)
        for r in d.itertuples():
            try:
                if status in ("base_no_update","base_fallback"):
                    p=raw[r.obs_id].copy();p["raw_prediction"]=p.pop("predicted");p["predicted"]=p["raw_prediction"]
                else:
                    v,j=physical_point(r,base,params);dg=v.diagnostics
                    p=dict(obs_id=r.obs_id,predicted=float(v.values[0]),raw_prediction=raw[r.obs_id]["predicted"],numeric=True,
                        physical_valid=bool(dg["gradient_valid"] and dg["single_electron_Hall_valid"]),diagnostics=dg)
                predictions.append(p)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                predictions.append(dict(obs_id=r.obs_id,predicted=None,raw_prediction=raw[r.obs_id]["predicted"],
                    numeric=False,physical_valid=False,error=repr(exc)))
    lookup={p["obs_id"]:p for p in predictions};rows=[]
    for r in d.itertuples():
        p=lookup[r.obs_id]
        rows.append(dict(task=task,context=context,method=method,budget=budget,family=r.family,
            sample_id=r.sample_id,obs_id=r.obs_id,Tobs_K=r.Tobs_K,role=r.role,observed=r.target,
            predicted=p["predicted"],raw_prediction=p["raw_prediction"],numeric=p["numeric"],
            physical_valid=p.get("physical_valid"),common40_125=bool(40<=r.Tobs_K<=125),
            calibration_status=status,calibration_key=key,base_fit_key=base["key"]))
    q=[r for r in rows if r["role"]=="query" and (task=="primary" or r["common40_125"])]
    score=float(np.mean([abs(r["predicted"]-r["observed"]) for r in q])) if all(r["numeric"] for r in q) else np.inf
    v=dict(key=key,task=task,context=context,budget=budget,method=method,config=config,lam=lam,
        base_fit_key=base["key"],base_path=str(path.relative_to(ROOT)),base_sha256=energy.sha(path),
        anchor_obs_ids=a.obs_id.tolist(),anchor_observations=a.target.tolist(),
        status=status,parameters=params,trials=trials,diagnostics=diagnostics,predictions=predictions,
        rows=rows,score=score,seconds=time.perf_counter()-start)
    energy.write(dest,v);v=energy.clean(v);CALIB[key]=v
    return v


def precompute_one(args):
    families,base,config,dropped=args
    path,f=base_fit(families,base,config,dropped)
    return str(path)


def contexts():
    jobs=[]
    definitions=[("no_extension_RT",{}),("fixed_binding_joint",{}),("learned_screen_joint",{}),
        ("dose_line_RT",{})]+[("direct_ridge_joint",dict(alpha=a)) for a in ALPHAS]
    for n in (3,4):
        for fs in combinations(FAMILIES,n):
            for base,c in definitions:jobs.append((fs,base,c,()))
    for held in sorted(prior.SPECS.family.unique()):
        fs=tuple(f for f in FAMILIES if f!=held)
        for specimen in prior.SPECS[prior.SPECS.family.ne(held)].index:
            for base,c in definitions:jobs.append((fs,base,c,(specimen,)))
    return jobs


def worker(method):
    freeze();base,mode=METHODS[method];rows=[];selections=[]
    for task in ("primary","temperature"):
        outer=FAMILIES if task=="primary" else tuple(sorted(prior.SPECS.family.unique()))
        for held in outer:
            fs=tuple(f for f in FAMILIES if f!=held)
            inners=fs if task=="primary" else tuple(prior.SPECS[prior.SPECS.family.ne(held)].index)
            for budget in (0,1,2):
                candidates=[]
                for i,c in enumerate(grids(method)):
                    for lam in ([np.inf] if budget==0 else LAMBDAS):
                        scores=[];keys=[]
                        for inner in inners:
                            train=tuple(f for f in fs if f!=inner) if task=="primary" else fs
                            drop=() if task=="primary" else (inner,)
                            path,f=base_fit(train,base,c,drop)
                            cal=calibrate(task,inner,budget,method,c,lam,path,f)
                            scores.append(np.inf if cal["score"] is None else cal["score"]);keys.append(cal["key"])
                        candidates.append(dict(index=i,config=c,lam=lam,score=float(np.mean(scores)),inner_calibration_keys=keys))
                best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"],c["index"]))
                path,f=base_fit(fs,base,best["config"])
                queries=[held] if task=="primary" else list(prior.SPECS[prior.SPECS.family.eq(held)].index)
                scores=[];keys=[]
                for query in queries:
                    cal=calibrate(task,query,budget,method,best["config"],best["lam"],path,f)
                    rows+=cal["rows"];scores.append(np.inf if cal["score"] is None else cal["score"]);keys.append(cal["key"])
                selections.append(dict(task=task,held=held,budget=budget,method=method,candidates=candidates,
                    selected=best,outer_calibration_keys=keys,score=float(np.mean(scores))))
                print("adaptation outer",method,task,held,budget,round(float(np.mean(scores)),6),flush=True)
    folder=OUT/"methods"/method
    energy.csvwrite(folder/"predictions.csv",rows);energy.write(folder/"selections.json",selections)
    return method


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--method",choices=METHODS);ap.add_argument("--precompute-only",action="store_true")
    args=ap.parse_args();freeze()
    assert json.loads((OUT/"numerical_gates.json").read_text())["passed"],"Run new numerical gates first"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(precompute_one,contexts()):pass
    print("base contexts ready",len(contexts()),flush=True)
    if args.precompute_only:return
    if args.method:worker(args.method)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for m in pool.map(worker,METHODS):print("adaptation method complete",m,flush=True)


if __name__=="__main__":main()
