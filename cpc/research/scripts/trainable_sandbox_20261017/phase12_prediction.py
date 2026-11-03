"""Nested source prediction with matched shared fields and internal adaptation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import sys,json,time,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.source_stock_model import SourceStockModel
from sandbox.scalar_profile_adaptation import ScalarProfile
from sandbox.material_profile import PUBLISHED_PROFILE
import phase12_base as base

core=base.core;energy=base.energy;OUT=base.OUT
LAM=(0.,.001,.01,.1,1.,10.,np.inf)
MODELS={
    "original_offset":("original_RT","offset"),
    "original_stock":("original_RT","stock"),
    "original_Tf":("original_RT","Tf"),
    "constant_offset":("stock_const_noext_RT","offset"),
    "constant_stock":("stock_const_noext_RT","stock"),
    "shared_RT_offset":("stock_rbf_noext_RT","offset"),
    "shared_RT_stock":("stock_rbf_noext_RT","stock"),
    "matched_output_offset":("out_rbf_noext_RT","offset"),
    "matched_Tf_offset":("tf_rbf_noext_RT","offset"),
    "joint_offset":("joint_baseline","offset"),
    "joint_stock":("joint_baseline","stock"),
    "shared_joint_offset":("parent_rbf_joint","offset"),
    "shared_joint_stock":("parent_rbf_joint","stock"),
    "direct_RT_offset":("direct_ridge_RT","offset"),
    "dose_offset":("dose_line_RT","offset"),
    "dose_affine":("dose_line_RT","affine"),
    "activation_prior_offset":("median_activation","offset"),
    "identity_offset":("identity","offset"),
    "identity_unbounded":("identity","offset"),
    "identity_H_unbounded":("identity_H","offset"),
}
MODEL_CACHE={};POINTS={};REFERENCES={};PRIORS={}


def grids(method):
    return [{}] if method in ("identity","identity_H","median_activation") else core.grids(method)


def weighted_median(values,weights):
    i=np.argsort(values,kind="stable");c=np.cumsum(weights[i]);return float(values[i[np.searchsorted(c,c[-1]/2)]])


def getfit(families,method,config):
    if method not in ("identity","identity_H","median_activation"):return base.getfit(families,method,config)
    key=core.old.digest(dict(protocol=energy.sha(OUT/"prediction_protocol.json"),families=sorted(families),baseline=method))
    if key in PRIORS:return PRIORS[key]
    path=OUT/"baselines"/(key+".json")
    if path.exists():PRIORS[key]=json.loads(path.read_text());return PRIORS[key]
    tr=core.training_data(families,"original_RT");w=tr.loss_weight.to_numpy(float)
    eta=weighted_median(tr.target.to_numpy(float)-np.log10(tr.incorporated_cm3.to_numpy(float)),w) if method=="median_activation" else 0.
    dose=core.PRED.incorporated_cm3.to_numpy(float)+(core.PRED.H_cm3.to_numpy(float) if method=="identity_H" else 0.)
    f=dict(key=key,method=method,config={},train_families=sorted(families),parameters=[eta],
        feature_metadata=core.feature_metadata(families),training_obs_ids=tr.obs_id.tolist(),
        training_specimens=sorted(tr.sample_id.unique()),uses_training_labels=method=="median_activation",
        predictions=[dict(obs_id=r.obs_id,predicted=float(v),numeric=True,physical_valid=None) for r,v in
            zip(core.PRED.itertuples(),np.log10(dose)+eta)],source="chemicaldoseonly ortrain-source-equal weightedmedianactivation")
    energy.write(path,f);PRIORS[key]=energy.clean(f);return PRIORS[key]


def reference(f):
    if f["key"] not in REFERENCES:
        if f.get("frozen_origin"):r=f["frozen_origin"]
        else:
            p=(OUT/"baselines"/(f["key"]+".json")) if f["method"] in ("identity","identity_H","median_activation") else base.NEW/"fits"/(f["key"]+".json")
            r=dict(path=str(p.relative_to(ROOT)),sha256=energy.sha(p))
        REFERENCES[f["key"]]=r
    return REFERENCES[f["key"]]


def runtime(f):
    if f["key"] not in MODEL_CACHE:
        MODEL_CACHE[f["key"]]=SourceStockModel(f["parameters"],core.METHODS[f["method"]],f["feature_metadata"],
            profile=PUBLISHED_PROFILE,pO2_atm=1e-5)
    return MODEL_CACHE[f["key"]]


def counters():
    names=("inference_calls","root_calls","hot_solve_calls","hot_cache_hits","root_evaluations")
    return {k:sum(m.counters()[k] for m in MODEL_CACHE.values()) for k in names}


def point(f,row,kind,z):
    key=(f["key"],row.obs_id,kind,float(z))
    if key not in POINTS:
        m=runtime(f);p=m.predict(row,source_kind=kind,source_shift=z)
        if len(POINTS)>20000:POINTS.clear()
        POINTS[key]=(p.values,p.source_jacobian,p.diagnostics)
    return POINTS[key]


def episode_key(f,held,kind):
    return core.old.digest(dict(protocol=energy.sha(OUT/"prediction_protocol.json"),base_key=f["key"],
        reference=reference(f),held=held,source_coordinate=kind))


def episode(f,held,kind):
    key=episode_key(f,held,kind);path=OUT/"episodes"/(key+".json")
    if path.exists():return json.loads(path.read_text())
    before=counters();start=time.perf_counter();results=[]
    d=core.PRED[core.PRED.family.eq(held)&core.PRED.kind.eq("RT")]
    lookup={r["obs_id"]:r for r in f["predictions"]}
    raw=np.array([lookup[r.obs_id]["predicted"] for r in d.itertuples()],float)
    observed=d.target.to_numpy(float)
    for budget in (0,1,2):
        roles=core.ROLES[core.ROLES.family.eq(held)&core.ROLES.budget.eq(budget)].set_index("sample_id").role
        anchor=np.array([roles[s]=="anchor" for s in d.sample_id]);query=np.array([roles[s]=="query" for s in d.sample_id])
        inputs=d[anchor];y=observed[anchor];profile=None
        if kind in ("stock","Tf") and budget:
            def forward(z):
                vals=[];jac=[]
                for row in inputs.itertuples():
                    v,J,_=point(f,row,kind,z);vals.append(v[0]);jac.append(J[0])
                return np.array(vals),np.array(jac)
            profile=ScalarProfile(forward)
        for lam in ([np.inf] if budget==0 else LAM):
            z=0.;coefficients=None;diagnostics={};trials=[];candidates=[];status="base_no_update"
            complete=bool(np.isfinite(raw[anchor]).all())
            if kind in ("stock","Tf") and budget and np.isfinite(lam):
                fit=profile.fit(y,lam);z=fit.shift;diagnostics=fit.diagnostics
                trials=fit.trials;candidates=fit.candidates;status=fit.status
                complete=fit.status!="no_feasible_anchor_state"
            elif kind=="offset" and budget and np.isfinite(lam) and complete:
                z=float(np.sum(raw[anchor]-y)/(budget+lam));status="closed_offset"
            elif kind=="affine" and budget and np.isfinite(lam) and complete:
                meta=f["feature_metadata"];x=(np.log10(d.incorporated_cm3.to_numpy(float))-meta["logC_center"])/meta["logC_scale"]
                X=np.c_[np.ones(len(d)),x];A=X[anchor]
                coefficients=(np.linalg.lstsq(A,y-raw[anchor],rcond=None)[0] if lam==0 else
                    np.linalg.solve(A.T@A+lam*np.eye(2),A.T@(y-raw[anchor])))
                status="closed_affine"
            records=[]
            for i,row in enumerate(d.itertuples()):
                value=None;error=None;physical=None;extra={}
                if complete:
                    try:
                        if kind in ("stock","Tf") and z!=0:
                            v,_,dg=point(f,row,kind,z);value=float(v[0]);physical=True
                            extra=dict(log10_Nabs=float(v[1]),log10_Nq2=float(v[2]),process_bias=dg["process_bias"],Tf_K=dg["Tf_K"],
                                root_residual=dg["root_residual_log"],jacobian_condition=dg["jacobian_condition"])
                        elif kind=="offset":value=float(raw[i]-z);physical=None
                        elif kind=="affine" and coefficients is not None:value=float(raw[i]+np.array([1.,x[i]])@coefficients);physical=None
                        else:
                            value=float(raw[i]);physical=lookup[row.obs_id].get("physical_valid")
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:error=repr(exc)
                if value is not None and not np.isfinite(value):error="nonfinite_prediction";value=None
                records.append(dict(obs_id=row.obs_id,sample_id=row.sample_id,family=held,role=roles[row.sample_id],budget=budget,
                    observed=row.target,predicted=value,raw_prediction=float(raw[i]),physical_valid=physical,error=error,**extra))
            vals=np.array([r["predicted"] if r["predicted"] is not None else np.nan for r in records])
            score=float(abs(vals[query]-observed[query]).mean()) if np.isfinite(vals[query]).all() else np.inf
            results.append(dict(budget=budget,lam=lam,shift=z,coefficients=coefficients,status=status,
                anchors=d.sample_id[anchor].tolist(),anchor_observed=y,anchor_complete=complete,
                profile_candidates=candidates,trials=trials,diagnostics=diagnostics,MAE=score,records=records))
    after=counters()
    result=dict(key=key,base_key=f["key"],base_method=f["method"],base_config=f["config"],base_reference=reference(f),
        train_families=f["train_families"],held=held,kind=kind,results=results,seconds=time.perf_counter()-start,
        counters={k:after[k]-before[k] for k in after})
    energy.write(path,result);print("source episode",kind,held,f["method"],round(result["seconds"],2),flush=True)
    return energy.clean(result)


def choose_result(ep,budget,lam):
    return next(r for r in ep["results"] if r["budget"]==budget and ((r["lam"] is None and np.isinf(lam)) or r["lam"]==lam))


def lambdas(model,budget):
    return [np.inf] if budget==0 else ([0.] if model in ("identity_unbounded","identity_H_unbounded") else LAM)


def selected(model):
    method,kind=MODELS[model];selections=[];rows=[]
    for held in core.FAMILIES:
        train=[f for f in core.FAMILIES if f!=held]
        for budget in (0,1,2):
            candidates=[]
            for index,config in enumerate(grids(method)):
                for lam in lambdas(model,budget):
                    scores=[];keys=[]
                    for inner in train:
                        f=getfit([s for s in train if s!=inner],method,config)
                        ep=episode(f,inner,kind);r=choose_result(ep,budget,lam)
                        scores.append(np.inf if r["MAE"] is None else r["MAE"]);keys.append(ep["key"])
                    candidates.append(dict(index=index,config=config,lam=lam,score=float(np.mean(scores)),inner_episode_keys=keys))
            best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"],c["index"]))
            f=getfit(train,method,best["config"]);ep=episode(f,held,kind);r=choose_result(ep,budget,best["lam"])
            for row in r["records"]:
                rows.append({**row,"model":model,"fit_key":f["key"],"episode_key":ep["key"],"source_kind":kind,
                    "source_shift":r["shift"],"adaptation_status":r["status"],"anchor_lambda":best["lam"]})
            selections.append(dict(model=model,held=held,budget=budget,candidates=candidates,selected=best,
                fit_key=f["key"],episode_key=ep["key"],outer_MAE=r["MAE"]))
            print("source outer",model,held,budget,r["MAE"],flush=True)
    folder=OUT/"models"/model;energy.write(folder/"selections.json",selections)
    energy.csvwrite(folder/"predictions.csv",rows)


def freeze():
    for name in ("base_gates.json","adaptation_gates.json"):assert json.loads((OUT/name).read_text())["passed"]
    assert json.loads((OUT/"base_training_execution.json").read_text())["new_records"]==150
    files=[Path(__file__),Path(base.__file__),ROOT/"src/sandbox/source_stock_model.py",ROOT/"src/sandbox/scalar_profile_adaptation.py",
        ROOT/"docs/ML_SANDBOX_SOURCE_STOCK_ADAPTATION_20261017.md"]
    inputs=[OUT/"base_gates.json",OUT/"adaptation_gates.json",base.NEW/"prediction_protocol.json",
        OUT/"base_training_execution.json",energy.BASE/"artifact_manifest_phase11.json",
        energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv"]
    p=dict(created_utc=energy.now(),version=1,models=MODELS,lambda_grid=LAM,
        base_configs={m:grids(m) for m,_ in MODELS.values()},
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        primary="unchanged29SiSn/5DOI/0-1-2anchors/guard/common18;whole-source nested model+lambda selection",
        stock="frozen sharedrelation;inside-logit increment ln10*z,range-6..6,25nodes+allgrid-localmin refinements",
        output="unbounded exact ridge mean offset,same lambda grid;freeidentityoffset reference",
        affine="two-coordinate doseintercept/slope ridge,training-only scaling;lambda0 minimum-norm solution",
        input_scope="same supplied energies and SiSn/H;pO2scenario1e-5;Cnotmodeled,not uniform-supplylabel",
        candidates_max_per_budget=42,tie_break="round MAE10digits,prefer larger lambda,then original configuration index",
        failure_policy="retain missing/invalid queries;scoreinfinite,no query-dependent fallback",
        interpretation="retrospective trained shared field+conditional source adaptation,not unique process labels,not blinded",
        temperature="deferred until RT study audited;original temperatureresults retained,not replaced")
    path=OUT/"prediction_protocol.json"
    if path.exists():
        old=json.loads(path.read_text())
        for key in ("models","lambda_grid","code_hashes","input_hashes","base_configs"):assert old[key]==energy.clean(p[key]),key
    else:energy.write(path,p)


def one(job):
    families,method,config,held,kind=job
    return episode(getfit(families,method,config),held,kind)["key"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4);args=ap.parse_args();freeze()
    jobs={}
    for n in (3,4):
        for families in combinations(core.FAMILIES,n):
            for held in core.FAMILIES:
                if held in families:continue
                for method,kind in MODELS.values():
                    for config in grids(method):
                        f=getfit(families,method,config);key=episode_key(f,held,kind)
                        jobs[key]=(families,method,config,held,kind)
    energy.write(OUT/"episode_design.json",dict(created_utc=energy.now(),unique_episodes=len(jobs),
        jobs=[dict(key=k,families=v[0],method=v[1],config=v[2],held=v[3],kind=v[4]) for k,v in jobs.items()]))
    print("source episode jobs",len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(one,jobs.values()):pass
    for model in MODELS:selected(model)
    energy.write(OUT/"prediction_execution.json",dict(created_utc=energy.now(),models=len(MODELS),
        episodes=len(jobs),goal_complete=False))
    print("phase12 source prediction completed",len(MODELS),len(jobs),flush=True)


if __name__=="__main__":main()
