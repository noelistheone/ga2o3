"""Real-label training of full profile/stock/temperature Sandbox and controls."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,time,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit,logit
from sklearn.linear_model import Ridge
from sandbox.kroger_db import load
from sandbox.material_profile import PUBLISHED_PROFILE
from sandbox.temperature_defects import TemperatureDefectLayer
from sandbox.implicit_defects import ImplicitDomainError
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as old
import phase9_temperature_gates as gates

OUT=gates.OUT;DATA=read_result_csv(energy.BASE/"dataset_v1.csv")
ROLES=read_result_csv(energy.BASE/"episodes_v1.csv")
TEMPDIR=energy.BASE/"phase8_process_temperature/extraction_v2"
SPECS=read_result_csv(TEMPDIR/"paired_specimens.csv").set_index("sample_id")
CURVES=read_result_csv(TEMPDIR/"temperature_observations.csv")
CURVES=CURVES[CURVES.endpoint.eq("log10_nH_cm3")]
GRIDS={m:[{}] for m in ("no_extension_RT","no_extension_joint","fixed_binding_joint",
                      "learned_binding_joint","learned_screen_joint","dose_line_RT")}
for m in ("direct_ridge_RT","direct_ridge_joint"):
    GRIDS[m]=[dict(alpha=a) for a in (.01,.1,1.,10.,100.,1e4)]
LAYERS={};FITS={};L0,L1=np.log10(600.),np.log10(1950.)

def inputs():
    base=DATA.copy();base["H_cm3"]=0.;base["H_known"]=False
    base["carbon_cm3"]=0.;base["carbon_known"]=False
    for i,r in base.iterrows():
        if r.sample_id in SPECS.index:
            s=SPECS.loc[r.sample_id]
            for field,target,flag in (("H_detected_cm3","H_cm3","H_known"),
                                      ("C_detected_cm3","carbon_cm3","carbon_known")):
                v=s.get(field,np.nan)
                if pd.notna(v) and float(v)>0:base.loc[i,target]=float(v);base.loc[i,flag]=True
    rows=[]
    for r in base.to_dict("records"):
        x=dict(r);x.update(obs_id="RT::"+r["sample_id"],Tobs_K=300.,kind="RT",target=r["observed_log_nH"]);rows.append(x)
    b=base.set_index("sample_id")
    for r in CURVES.itertuples():
        x=b.loc[r.sample_id].to_dict();x.update(sample_id=r.sample_id,obs_id=r.read_id,
            Tobs_K=r.T_K,kind="curve",target=r.observed);rows.append(x)
    return pd.DataFrame(rows).sort_values(["family","sample_id","Tobs_K","kind"]).reset_index(drop=True)

PRED=inputs()

def freeze():
    gates.freeze()
    assert json.loads((OUT/"numerical_gates.json").read_text())["passed"]
    path=OUT/"prediction_protocol.json"
    files=[Path(__file__),ROOT/"src/sandbox/material_profile.py",ROOT/"src/sandbox/temperature_defects.py",
           ROOT/"docs/ML_SANDBOX_FULL_TEMPERATURE_20261017.md"]
    p=dict(version=1,created_utc=energy.now(),models=GRIDS,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
            TEMPDIR/"temperature_observations.csv",TEMPDIR/"paired_specimens.csv",
            OUT/"numerical_gates.json")},
        material_profile=PUBLISHED_PROFILE.to_dict(),profile_sha256=PUBLISHED_PROFILE.fingerprint,
        main="original29specimens/5DOI,sourceexcludedRT0/1/2anchor/guard/common18",
        supplement="7same specimens,142carrierreadings;RT+nearest150Ktemperatureanchors PER SPECIMEN",
        temperature_adaptation="same lambda selected for the originalRTouter fold transferred unchanged;no additional temperature hyperselection",
        temperature_hypergrade="differs from stage8temperature-specific inner specimen tuning;strong controls still reported",
        chemistry="reportedH forC/D supplied;unknownH=0 conditional scenario elsewhere,not measured absence;carbonDnotmodeled bycatalogue",
        direct_inputs="dose,growthproxy,element,T,H/carbon knownflags/knownconcentrations;missing flags retained",
        Hall="rH=1 for ALL new primary methods;densityheuristic only for oldstate replay",
        old_comparison="stage8usesdifferentHall/Hinformation,so not pure architecture contrast",
        joint_loss="mean overreadings<=275KplusRT within specimen,then mean over specimens withinDOI,then equalDOI",
        RT_only="diagnostic controls ignore allowedcurves,not same-training-data causal comparison",
        starting_points=[dict(T=1400,logNd=16,E0=.03,s=.2),dict(T=1750,logNd=15.5,E0=.06,s=1.)],
        bounds=dict(T=[600,1950],logNd=[14,18],logitT=[-6,6],E0=[.003,.2],screen=[0,5]),
        optimizer="two starts,max_nfev60,analyticIFT,ftol/xtol/gtol1e-7;all failures retained",
        hyperselection="RTinnerleaveoneDOIout;candidate<=6;lambda.1/1/10/inf;training-only scaling",
        supplied_energetics="entire873input available pluscatalogue;effective neutral extension not newDFTlabels",
        interpretation="retrospective same exposed cohort,notnewmaterialreplication ornovelMLalgorithmclaim")
    if path.exists():
        before=json.loads(path.read_text())
        for k in ("code_hashes","input_hashes","models"):assert before[k]==energy.clean(p[k]),k
    else:energy.write(path,p)

def layer_for(row,extension):
    elements=(row.element,"H") if row.H_known else (row.element,)
    key=(elements,extension)
    if key not in LAYERS:
        LAYERS[key]=TemperatureDefectLayer(load(),elements,profile=PUBLISHED_PROFILE,
            donor_extension=extension,donor_element=row.element,hall="rH1")
    return LAYERS[key]

def point(row,w,method):
    extension=method in ("fixed_binding_joint","learned_binding_joint","learned_screen_joint")
    l=layer_for(row,extension);st=expit(w[0]);T=10**(L0+(L1-L0)*st)
    E=10**w[2] if method in ("learned_binding_joint","learned_screen_joint") else .0366
    s=w[3] if method=="learned_screen_joint" else 0.
    doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,)
    p=l.parameters(T=T,Nd=10**w[1],doses=doses,Tobs=row.Tobs_K,binding_eV=E,screen_s=s)
    a=l.predict(p);chain=np.zeros((l.np,len(w)));chain[0,0]=(L1-L0)*st*(1-st);chain[1,1]=1
    if len(w)>2:chain[l.ie,2]=1.
    if len(w)>3:chain[l.ie+1,3]=1.
    return a,a.jacobian@chain

def features(d):
    c=np.log10(d.incorporated_cm3.to_numpy(float))-18
    t=d.Tobs_K.to_numpy(float);g=(d.growth_T_C.to_numpy(float)+273.15-1100)/300.
    hk=d.H_known.to_numpy(float);ck=d.carbon_known.to_numpy(float)
    h=np.log10(np.maximum(d.H_cm3.to_numpy(float),1))-17
    carbon=np.log10(np.maximum(d.carbon_cm3.to_numpy(float),1))-17
    return np.c_[c,c*c,g,d.element.eq("Sn").to_numpy(float),np.log(t/300),300/t-1,
                 c*(300/t-1),hk,h*hk,ck,carbon*ck]

def training_data(families,method):
    d=PRED[PRED.family.isin(families)]
    if method.endswith("_RT"):d=d[d.kind.eq("RT")]
    else:d=d[d.kind.eq("RT")|d.Tobs_K.le(275)]
    perSample=d.groupby("sample_id").obs_id.transform("count").to_numpy(float)
    nSample=d.groupby("family").sample_id.transform("nunique").to_numpy(float)
    w=1/(perSample*nSample);w/=w.mean()
    return d,w

def fit(families,method,config):
    families=sorted(families);key=old.digest(dict(families=families,method=method,config=config,
        protocol=energy.sha(OUT/"prediction_protocol.json")))
    path=OUT/"fits"/f"{key}.json"
    if key in FITS:return FITS[key]
    if path.exists():FITS[key]=json.loads(path.read_text());return FITS[key]
    tr,weights=training_data(families,method);y=tr.target.to_numpy(float);sw=np.sqrt(weights)
    start=time.perf_counter();parameters=None;trials=[];failures=[];metadata={}
    if method=="dose_line_RT":
        x=np.c_[np.ones(len(tr)),np.log10(tr.incorporated_cm3)-18.]
        b=np.linalg.lstsq(x*sw[:,None],y*sw,rcond=None)[0]
        prediction=b[0]+b[1]*(np.log10(PRED.incorporated_cm3)-18)
        metadata["coefficients"]=b;preds=None
    elif method.startswith("direct_ridge"):
        x=features(tr);center=np.average(x,axis=0,weights=weights)
        scale=np.sqrt(np.average((x-center)**2,axis=0,weights=weights));scale[scale<1e-12]=1.
        m=Ridge(**config).fit((x-center)/scale,y,sample_weight=weights)
        prediction=m.predict((features(PRED)-center)/scale);preds=None
        metadata.update(center=center,scale=scale,coefficients=m.coef_,intercept=m.intercept_)
    else:
        n=4 if method=="learned_screen_joint" else (3 if method=="learned_binding_joint" else 2)
        low=[-6.,14.]+([np.log10(.003)] if n>2 else [])+([0.] if n>3 else [])
        high=[6.,18.]+([np.log10(.2)] if n>2 else [])+([5.] if n>3 else [])
        cache={}
        def calc(w):
            k=tuple(w)
            if k not in cache:
                v=[];js=[]
                for row in tr.itertuples():
                    try:a,j=point(row,w,method)
                    except (ImplicitDomainError,ValueError,np.linalg.LinAlgError) as exc:
                        failures.append(dict(obs_id=row.obs_id,parameters=w,error=repr(exc)));raise
                    v.append(a.values[0]);js.append(j[0])
                cache.clear();cache[k]=(np.asarray(v),np.asarray(js))
            return cache[k]
        for T,B,E,s in ((1400.,16.,.03,.2),(1750.,15.5,.06,1.)):
            t=logit((np.log10(T)-L0)/(L1-L0))
            initial=np.array([t,B]+([np.log10(E)] if n>2 else [])+([s] if n>3 else []))
            try:
                op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],
                    bounds=(low,high),max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                    nfev=int(op.nfev),active_mask=op.active_mask,message=op.message))
            except (ImplicitDomainError,ValueError,np.linalg.LinAlgError) as exc:
                trials.append(dict(parameters=initial,loss=None,success=False,error=repr(exc)))
        good=[t for t in trials if t.get("loss") is not None]
        parameters=min(good,key=lambda t:t["loss"])["parameters"] if good else None
        preds=[]
        for row in PRED.itertuples():
            try:
                if parameters is None:raise ImplicitDomainError("all_initializations_failed")
                a,j=point(row,parameters,method);dg=a.diagnostics
                preds.append(dict(obs_id=row.obs_id,predicted=float(a.values[0]),numeric=True,
                    physical_valid=bool(dg["gradient_valid"] and dg["single_electron_Hall_valid"]),
                    above_reference_floor=dg["above_carrier_detection_floor"],diagnostics=dg))
            except (ImplicitDomainError,ValueError,np.linalg.LinAlgError) as exc:
                preds.append(dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc)))
    if preds is None:
        preds=[dict(obs_id=r.obs_id,predicted=float(v),numeric=True,physical_valid=None,
                    physical_validity="not_applicable_empirical") for r,v in zip(PRED.itertuples(),prediction)]
    f=dict(key=key,method=method,config=config,train_families=families,
        training_obs_ids=tr.obs_id.tolist(),training_specimens=sorted(tr.sample_id.unique()),
        training_observations=len(tr),weight_per_observation=weights,
        parameters=parameters,trials=trials,state_failures=failures,metadata=metadata,
        predictions=preds,seconds=time.perf_counter()-start)
    energy.write(path,f);FITS[key]=energy.clean(f)
    print("fulltemperature fit",method,len(families),"observations",len(tr),
          "success",parameters is not None or method.startswith(("direct","dose")),"seconds",round(f["seconds"],2),flush=True)
    return FITS[key]

def primary(f,family,budget,lam):
    lookup={p["obs_id"]:p for p in f["predictions"]}
    adapter=dict(predictions=[dict(sample_id=r.sample_id,
        predicted_log_nH=lookup["RT::"+r.sample_id]["predicted"],
        physical_valid=lookup["RT::"+r.sample_id]["physical_valid"]) for r in DATA.itertuples()])
    old.DATA=DATA;old.ROLES=ROLES
    return old.evaluate(adapter,family,budget,lam)

def temperature(f,sample,budget,lam):
    d=PRED[PRED.sample_id.eq(sample)].copy();lookup={r["obs_id"]:r for r in f["predictions"]}
    a=[d[d.kind.eq("RT")].iloc[0]] if budget else []
    if budget==2:
        candidates=d[d.kind.eq("curve")&d.Tobs_K.between(125,175)]
        i=abs(candidates.Tobs_K-150).sort_values(kind="stable").index[0];a.append(d.loc[i])
    residual=[r.target-lookup[r.obs_id]["predicted"] for r in a if lookup[r.obs_id]["predicted"] is not None]
    delta=float(sum(residual)/(len(a)+lam)) if a and len(residual)==len(a) and np.isfinite(lam) else 0.
    valid_anchors=len(residual)==len(a)
    anchorids={r.obs_id for r in a};rows=[]
    for r in d.itertuples():
        role="unavailable_reference" if r.kind=="RT" else "query"
        if r.obs_id in anchorids:role="anchor"
        elif budget and abs(r.Tobs_K-300)<=25:role="guard"
        if budget==2 and abs(r.Tobs_K-a[1].Tobs_K)<=10 and r.obs_id not in anchorids:role="guard"
        val=lookup[r.obs_id]["predicted"]
        rows.append(dict(obs_id=r.obs_id,sample_id=sample,family=r.family,Tobs_K=r.Tobs_K,
            observed=r.target,predicted=None if val is None or not valid_anchors else val+delta,
            raw_prediction=val,role=role,budget=budget,offset_dex=delta,
            common40_125=bool(40<=r.Tobs_K<=125),anchor_complete=valid_anchors))
    return rows

def worker(method):
    freeze();allrows=[];trows=[];selections=[];families=sorted(DATA.family.unique())
    for held in families:
        training=[f for f in families if f!=held]
        for budget in (0,1,2):
            candidates=[]
            for i,config in enumerate(GRIDS[method]):
                for lam in ([np.inf] if budget==0 else [.1,1.,10.,np.inf]):
                    errors=[]
                    for inner in training:
                        f=fit([x for x in training if x!=inner],method,config)
                        errors.append(primary(f,inner,budget,lam)[0])
                    candidates.append(dict(index=i,config=config,lam=lam,score=float(np.mean(errors))))
            best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"],c["index"]))
            f=fit(training,method,best["config"]);score,rows=primary(f,held,budget,best["lam"])
            for r in rows:r.update(method=method,fit_key=f["key"])
            allrows+=rows
            for sample in SPECS[SPECS.family.eq(held)].index:
                rs=temperature(f,sample,budget,best["lam"])
                for r in rs:r.update(method=method,fit_key=f["key"])
                trows+=rs
            selections.append(dict(held=held,budget=budget,method=method,candidates=candidates,
                selected=best,query_MAE=score,selection_failure=not np.isfinite(best["score"])))
            print("fulltemperature outer",held,method,budget,score,flush=True)
    energy.csvwrite(OUT/"methods"/method/"primary_predictions.csv",allrows)
    energy.csvwrite(OUT/"methods"/method/"temperature_predictions.csv",trows)
    energy.write(OUT/"methods"/method/"selections.json",selections)
    return method

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--method",choices=GRIDS);ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args();freeze()
    energy.csvwrite(OUT/"observations_and_inputs.csv",PRED.to_dict("records"))
    if a.method:worker(a.method)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for method in pool.map(worker,GRIDS):print("fulltemperature method complete",method,flush=True)

if __name__=="__main__":main()
