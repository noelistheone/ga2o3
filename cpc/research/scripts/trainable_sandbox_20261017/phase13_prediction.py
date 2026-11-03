"""Train and compare shared donor fractions under the unchanged property task."""
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
from scipy.special import logit
from sklearn.linear_model import Ridge
from sandbox.donor_mixture_model import DonorMixtureModel,L0,L1
from sandbox.material_profile import PUBLISHED_PROFILE
from sandbox.kroger_db import load
from csv_io_v2 import read_result_csv
import phase11_prediction as prior
import phase2_energy as energy
import transport_chain as median_utils

OUT=energy.BASE/"phase13_donor_mixture";PRED=prior.PRED;DATA=prior.DATA;ROLES=prior.ROLES
FAMILIES=tuple(sorted(DATA.family.unique()));LAM=(0.,.001,.01,.1,1.,10.,np.inf)
METHODS={
    "original_RT":dict(fraction="zero",training="RT"),
    "original_joint":dict(fraction="zero",training="joint"),
    "localized_fixed_joint":dict(fraction="one",training="joint"),
    "binding_screen_joint":dict(fraction="one",binding="screen",training="joint"),
    "fraction_constant_RT":dict(fraction="constant",training="RT"),
    "fraction_dose_RT":dict(fraction="dose",training="RT"),
    "fraction_constant_joint":dict(fraction="constant",training="joint"),
    "fraction_dose_joint":dict(fraction="dose",training="joint"),
    "fraction_binding_joint":dict(fraction="dose",binding="learned",training="joint"),
    "output_linear_RT":dict(fraction="zero",output="linear",training="RT"),
    "output_linear_joint":dict(fraction="zero",output="linear",training="joint"),
    "direct_ridge_RT":dict(empirical="ridge",training="RT"),
    "direct_ridge_joint":dict(empirical="ridge",training="joint"),
    "dose_line_RT":dict(empirical="dose",training="RT"),
    "activation_prior_RT":dict(empirical="activation",training="RT"),
    "identity_unbounded":dict(empirical="identity",training="RT"),
    "identity_H_unbounded":dict(empirical="identityH",training="RT"),
}
MODELS={};FITS={}


def grids(method):
    return [dict(alpha=a) for a in (.01,.1,1.,10.,100.,1e4)] if METHODS[method].get("empirical")=="ridge" else [{}]


def mapper(method):
    if method not in MODELS:MODELS[method]=DonorMixtureModel(METHODS[method],profile=PUBLISHED_PROFILE,pO2_atm=1e-5)
    return MODELS[method]


def specification(method):
    m=mapper(method);lo=[-6.,14.];hi=[6.,18.];starts=[]
    if m.iE is not None:lo.append(np.log10(.003));hi.append(np.log10(.2))
    if m.iscreen is not None:lo.append(0.);hi.append(5.)
    if m.ia is not None:lo.append(-12.);hi.append(12.)
    if m.ib is not None:lo.append(0.);hi.append(12.)
    if m.iout is not None:lo.extend([-4.,-4.]);hi.extend([4.,4.])
    for T,B,E,s,a,b in ((1400.,16.,.03,.2,2.,2.),(1750.,15.5,.06,1.,0.,4.)):
        w=[logit((np.log10(T)-L0)/(L1-L0)),B]
        if m.iE is not None:w.append(np.log10(E))
        if m.iscreen is not None:w.append(s)
        if m.ia is not None:w.append(a)
        if m.ib is not None:w.append(b)
        if m.iout is not None:w.extend([0.,0.])
        starts.append(w)
    return np.array(lo),np.array(hi),starts


def training_data(families,method):
    d=PRED[PRED.family.isin(families)];rt=d[d.kind.eq("RT")].copy()
    rt["loss_weight"]=1/(rt.groupby("family").sample_id.transform("nunique").to_numpy(float)*rt.family.nunique())
    if METHODS[method]["training"]=="RT":return rt
    c=d[d.kind.eq("curve")&d.Tobs_K.le(275)].copy()
    if len(c):c["loss_weight"]=.1/(c.groupby("sample_id").obs_id.transform("count").to_numpy(float)*
        c.groupby("family").sample_id.transform("nunique").to_numpy(float)*c.family.nunique())
    return pd.concat([rt,c],ignore_index=True).sort_values(["family","sample_id","Tobs_K","kind"])


def freeze():
    for gate in ("numerical_gates.json","model_gates.json"):
        assert json.loads((OUT/gate).read_text())["passed"],gate
    numerical=json.loads((OUT/"numerical_protocol.json").read_text())
    for group in ("code_hashes","input_hashes"):
        for path,sha in numerical[group].items():assert energy.sha(ROOT/path)==sha,path
    for path,sha in json.loads((OUT/"model_gates.json").read_text())["code_hashes"].items():
        assert energy.sha(ROOT/path)==sha,path
    files=(Path(__file__),ROOT/"src/sandbox/donor_mixture_model.py",ROOT/"src/sandbox/donor_mixture.py",
        ROOT/"scripts/trainable_sandbox_20261017/phase13_model_gates.py",ROOT/"src/sandbox/parent_allocation.py",
        ROOT/"docs/ML_SANDBOX_DONOR_MIXTURE_THEORY_20261017.md",ROOT/"scripts/trainable_sandbox_20261017/phase9_joint_prediction.py")
    inputs=(OUT/"numerical_gates.json",OUT/"model_gates.json",energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
        prior.prior.TEMPDIR/"paired_specimens.csv",prior.prior.TEMPDIR/"temperature_observations.csv",energy.BASE/"artifact_manifest_phase12.json")
    payload=dict(created_utc=energy.now(),version=1,methods=METHODS,grids={m:grids(m) for m in METHODS},
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        primary="unchanged29Si/Sn5DOI;0/1/2sampleanchors;29/23/18queries andcommon18",
        training="sourceequalRTsquared +.1source/specimen/readequal <=275Kcurve squared;RT-only controls labelled",
        numerical_model="fixed-input constrained localized/delocalized subgroups;exactoriginal/localized endpoints;not newmicroscopicMottmodel",
        definitions="no direct target labels used by map;no DOS/impurityband transport learning; n>p does not establish singleband",
        physical_inputs="same873energyinputs;actualSi/Sn,reportedH;Cabsent;unknownpO2scenario1e-5;publishedprofile;rH1",
        optimizer="ALLnewphysicalmodels2declaredstarts,max_nfev120,ftol/xtol/gtol1e-7,analyticIFT;all failuresretained",
        per_model_specifications={m:dict(names=mapper(m).parameter_names,bounds=specification(m)[:2],starts=specification(m)[2])
            for m in METHODS if "empirical" not in METHODS[m]},
        directRidge="sameknownfeatures asphase11;trainingsource scalingonly;sourceequal weights normalized toreadcount;6alpha configs",
        anchor_adaptation="closed unbounded outputintercept,lambda0/.001/.01/.1/1/10/inf;identitycontrols fixedlambda0",
        hyperselection="innerwholeDOI;roundscore1e-10,tiespreferlargerlambda;max42candidates,not matchedruntime",
        temperature="same7oldSi;RT-selectedlambda transfers,RT/near150Ksampleanchors andguards;not independent sources/new specimens",
        retrospective=True,new_physical_experiments=0,new_DFT_MD=0,goal_complete=False)
    path=OUT/"prediction_protocol.json"
    if path.exists():
        old=json.loads(path.read_text())
        for key in ("methods","grids","code_hashes","input_hashes"):assert old[key]==energy.clean(payload[key]),key
    else:energy.write(path,payload)


def fit(families,method,config):
    families=tuple(sorted(families));key=prior.old.digest(dict(protocol=energy.sha(OUT/"prediction_protocol.json"),
        train_families=families,method=method,config=config));path=OUT/"fits"/(key+".json")
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    d=METHODS[method];tr=training_data(families,method);weights=tr.loss_weight.to_numpy(float)
    y=tr.target.to_numpy(float);sw=np.sqrt(weights);clock=time.perf_counter();trials=[];failures=[];parameters=None;meta={}
    empirical=None;field=d.get("empirical");before=None
    if field=="ridge":
        X=prior.prior.features(tr);center=np.average(X,axis=0,weights=weights)
        scale=np.sqrt(np.average((X-center)**2,axis=0,weights=weights));scale[scale<1e-12]=1.
        m=Ridge(**config).fit((X-center)/scale,y,sample_weight=weights*len(tr)/weights.sum())
        empirical=m.predict((prior.prior.features(PRED)-center)/scale)
        meta=dict(center=center,scale=scale,coefficients=m.coef_,intercept=m.intercept_)
    elif field=="dose":
        X=np.c_[np.ones(len(tr)),np.log10(tr.incorporated_cm3)-18.]
        coef=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
        empirical=coef[0]+coef[1]*(np.log10(PRED.incorporated_cm3)-18.);meta=dict(coefficients=coef)
    elif field in ("identity","identityH","activation"):
        x=np.log10(PRED.incorporated_cm3.to_numpy(float)+(PRED.H_cm3.to_numpy(float) if field=="identityH" else 0.))
        delta=0. if field!="activation" else median_utils.median(y-np.log10(tr.incorporated_cm3.to_numpy(float)),weights)
        empirical=x+delta;meta=dict(log_activation_ratio=delta)
    else:
        model=mapper(method);lo,hi,starts=specification(method);before=model.counters();cache={}
        def calc(w):
            k=tuple(w)
            if k not in cache:
                vals=[];jac=[]
                for row in tr.itertuples():
                    try:a=model.point(row,w)
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        failures.append(dict(obs_id=row.obs_id,parameters=w,error=repr(exc)));raise
                    vals.append(a.values[0]);jac.append(a.jacobian[0])
                cache.clear();cache[k]=(np.array(vals),np.array(jac))
            return cache[k]
        for initial in starts:
            try:
                op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],bounds=(lo,hi),
                    max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(initial=initial,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                    nfev=int(op.nfev),njev=int(op.njev),optimality=float(op.optimality),active_mask=op.active_mask,message=op.message))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                trials.append(dict(initial=initial,parameters=initial,loss=None,success=False,error=repr(exc)))
        finite=[t for t in trials if t["loss"] is not None]
        if finite:parameters=min(finite,key=lambda t:t["loss"])["parameters"]
    training_counts={} if before is None else {k:mapper(method).counters()[k]-before[k] for k in before}
    predictions=[]
    for i,row in enumerate(PRED.itertuples()):
        if empirical is not None:
            predictions.append(dict(obs_id=row.obs_id,predicted=float(empirical[i]),physical_valid=None,numeric=True))
        else:
            try:
                if parameters is None:raise RuntimeError("allstartsfailed")
                a=mapper(method).point(row,parameters);g=a.diagnostics
                dg={k:g[k] for k in ("root_residual_log","jacobian_condition","free_n_cm3","free_p_cm3","Tf_K",
                    "measurement_T_K","Nd_cm3","profile_sha256","localized_fraction","donor_subgroups","same_physical_state_for_output")}
                predictions.append(dict(obs_id=row.obs_id,predicted=float(a.values[0]),charge_outputs=a.values[1:],numeric=True,
                    physical_valid=True if not d.get("output") else None,diagnostics=dg))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                predictions.append(dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc)))
    counts={} if before is None else {k:mapper(method).counters()[k]-before[k] for k in before}
    f=dict(key=key,method=method,definition=d,config=config,train_families=families,parameters=parameters,
        parameter_names=[] if field else mapper(method).parameter_names,trials=trials,state_failures=failures,
        training_obs_ids=tr.obs_id.tolist(),training_specimens=sorted(tr.sample_id.unique()),weight_per_observation=weights,
        metadata=meta,predictions=predictions,seconds=time.perf_counter()-clock,counters=counts,
        training_counters=training_counts,cached_prediction_counters={k:counts[k]-training_counts[k] for k in counts})
    energy.write(path,f);FITS[key]=energy.clean(f)
    print("mixturefit",method,len(families),"seconds",round(f["seconds"],2),"finite",parameters is not None or empirical is not None,flush=True)
    return FITS[key]


def evaluate(f,held,budget,lam):
    return prior.prior.primary(f,held,budget,lam)


def worker(job):return fit(*job)["key"]


def select(method):
    rows=[];temps=[];choices=[]
    for held in FAMILIES:
        train=[x for x in FAMILIES if x!=held]
        for budget in (0,1,2):
            candidates=[]
            lambdas=[np.inf] if budget==0 else ([0.] if method.startswith("identity") else LAM)
            for i,c in enumerate(grids(method)):
                for lam in lambdas:
                    fits=[fit([x for x in train if x!=inner],method,c) for inner in train]
                    scores=[evaluate(f,inner,budget,lam)[0] for f,inner in zip(fits,train)]
                    candidates.append(dict(index=i,config=c,lam=lam,score=float(np.mean(scores)),inner_keys=[f["key"] for f in fits]))
            best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"],c["index"]))
            f=fit(train,method,best["config"]);score,result=evaluate(f,held,budget,best["lam"])
            for r in result:r.update(method=method,fit_key=f["key"])
            rows+=result
            for sample in prior.prior.SPECS[prior.prior.SPECS.family.eq(held)].index:
                result=prior.prior.temperature(f,sample,budget,best["lam"])
                for r in result:r.update(method=method,fit_key=f["key"])
                temps+=result
            choices.append(dict(method=method,held=held,budget=budget,candidates=candidates,selected=best,fit_key=f["key"],outer_MAE=score))
    folder=OUT/"methods"/method
    energy.csvwrite(folder/"primary_predictions.csv",rows);energy.csvwrite(folder/"temperature_predictions.csv",temps)
    energy.write(folder/"selections.json",choices)
    print("selected",method,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4);ap.add_argument("--fit-only",action="store_true");a=ap.parse_args()
    freeze();jobs=[(families,method,config) for n in (3,4) for families in combinations(FAMILIES,n)
        for method in METHODS for config in grids(method)]
    energy.write(OUT/"training_execution_started.json",dict(created_utc=energy.now(),jobs=len(jobs),workers=a.workers,methods=len(METHODS)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    energy.write(OUT/"training_execution_complete.json",dict(created_utc=energy.now(),jobs=len(jobs),methods=len(METHODS),goal_complete=False))
    if not a.fit_only:
        for method in METHODS:select(method)
        energy.write(OUT/"selection_execution_complete.json",dict(created_utc=energy.now(),methods=len(METHODS),goal_complete=False))


if __name__=="__main__":main()
