"""Same-anchor source adaptation on frozen original/E(C)/joint models."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import phase15_prediction as ref
import phase2_energy as energy
from sandbox.scalar_profile_adaptation import ScalarProfile

BASE=ref.OUT;OUT=BASE/"source_adaptation";core=ref.core
LAM=(0.,.001,.01,.1,1.,10.,np.inf)
VARIANTS={method+"_"+label:(method,kind) for method in ref.NEW_METHODS
    for label,kind in (("output","offset"),("stock","stock"),("Tf","Tf"))}
BANK={};POINTS={}


def bank():
    global BANK
    if not BANK:BANK=ref.composite()
    return BANK


def getfit(families,method):
    return next(f for f in bank().values() if f["method"]==method and set(f["train_families"])==set(families))


def freeze():
    assert json.loads((ref.OUT/"validation/contract_audit.json").read_text())["passed"]
    files=(Path(__file__),ROOT/"src/sandbox/coherent_hall_model.py",ROOT/"src/sandbox/coherent_fermi.py",ROOT/"src/sandbox/coherent_transport.py",ROOT/"src/sandbox/scalar_profile_adaptation.py",
        ROOT/"docs/ML_SANDBOX_COHERENT_SOURCE_20261018.md")
    p=dict(created_utc=energy.now(),variants=VARIANTS,code_hashes={str(f.relative_to(ROOT)):energy.sha(f) for f in files},
        input_hashes={str(f.relative_to(ROOT)):energy.sha(f) for f in (ref.OUT/"validation/contract_audit.json",BASE/"prediction_protocol.json",
            ref.OUT/"numerical_gates.json",energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv")},
        fits=[dict(key=f["key"],**f["composite_origin"]) for f in bank().values() if f["method"] in {v[0] for v in VARIANTS.values()}],
        primary="original29/5,0-1-2specimenanchors/common18",lambda_grid=LAM,
        adaptation="frozen sharedparameters;25nodes/localrefinement forinternal;closedoffset;noquerylabels enterfit",
        selection="innerwholeDOI equalMAE/tiespreferlargerlambda",scope="retrospective;no newtemperature/source/mulabels",goal_complete=False)
    pp=OUT/"source_protocol.json"
    if pp.exists():
        old=json.loads(pp.read_text())
        for k in ("variants","code_hashes","input_hashes","fits"):assert old[k]==energy.clean(p[k]),k
    else:energy.write(pp,p)


def point(f,row,kind,z):
    key=(f["key"],row.obs_id,kind,float(z))
    if key not in POINTS:
        p=core.mapper(f["method"]).point(row,f["parameters"],source_kind=kind,source_shift=z)
        POINTS[key]=(p.values,p.source_jacobian,p.diagnostics)
    return POINTS[key]


def episode(f,held,kind):
    key=core.prior.old.digest(dict(protocol=energy.sha(OUT/"source_protocol.json"),fit=f["key"],fit_sha=f["composite_origin"]["sha256"],held=held,kind=kind))
    path=OUT/"episodes"/(key+".json")
    if path.exists():return json.loads(path.read_text())
    d=core.PRED[core.PRED.family.eq(held)&core.PRED.kind.eq("RT")];observed=d.target.to_numpy(float)
    model=core.mapper(f["method"]);before=model.counters();clock=time.perf_counter();results=[]
    for budget in (0,1,2):
        roles=core.ROLES[core.ROLES.family.eq(held)&core.ROLES.budget.eq(budget)].set_index("sample_id").role
        anchor=np.array([roles[s]=="anchor" for s in d.sample_id]);query=np.array([roles[s]=="query" for s in d.sample_id])
        aa=d[anchor];y=observed[anchor];profile=None
        if kind!="offset" and budget:
            def forward(z):
                p=[point(f,r,kind,z) for r in aa.itertuples()]
                return np.array([v[0][0] for v in p]),np.array([v[1][0] for v in p])
            profile=ScalarProfile(forward)
        raw=[]
        for row in d.itertuples():
            try:raw.append(float(point(f,row,"none",0.)[0][0]))
            except (ValueError,RuntimeError,np.linalg.LinAlgError):raw.append(np.nan)
        raw=np.array(raw)
        for lam in ([np.inf] if budget==0 else LAM):
            z=0.;status="base_no_update";dg={};candidates=[];trials=[];complete=True
            if budget and np.isfinite(lam):
                if kind=="offset":
                    complete=bool(np.isfinite(raw[anchor]).all())
                    if complete:z=float(np.sum(raw[anchor]-y)/(budget+lam));status="closed_offset"
                else:
                    fit=profile.fit(y,lam);z=fit.shift;status=fit.status;dg=fit.diagnostics;candidates=fit.candidates;trials=fit.trials
                    complete=status!="no_feasible_anchor_state"
            rows=[]
            for i,row in enumerate(d.itertuples()):
                val=None;error=None;extra={}
                if complete:
                    try:
                        p,J,g=point(f,row,kind,z);val=float(p[0]);extra=dict(logNabs=float(p[1]),logNq2=float(p[2]),
                            localized_fraction=g["localized_fraction"],process_bias=g["process_bias"],Tf_K=g["Tf_K"],
                            same_physical_state_for_output=g["same_physical_state_for_output"],root_residual=g["root_residual_log"],
                            predicted_log_mu_H=g["predicted_log_mu_H"],predicted_log_sigma=g["predicted_log_sigma"],hall_factor=g["hall_factor"])
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:error=repr(exc)
                rows.append(dict(sample_id=row.sample_id,obs_id=row.obs_id,family=held,budget=budget,role=roles[row.sample_id],
                    observed=float(row.target),predicted=val,error=error,raw_prediction=raw[i],**extra))
            vals=np.array([np.nan if r["predicted"] is None else r["predicted"] for r in rows])
            score=float(abs(vals[query]-observed[query]).mean()) if np.isfinite(vals[query]).all() else np.inf
            results.append(dict(budget=budget,lam=lam,shift=z,status=status,anchor_samples=aa.sample_id.tolist(),anchor_observed=y,
                anchor_complete=complete,diagnostics=dg,profile_candidates=candidates,trials=trials,records=rows,MAE=score))
    after=model.counters();result=dict(key=key,fit_key=f["key"],method=f["method"],held=held,kind=kind,
        train_families=f["train_families"],fit_origin=f["composite_origin"],results=results,seconds=time.perf_counter()-clock,
        counters={k:after[k]-before[k] for k in before})
    energy.write(path,result);print("jointsource",f["method"],kind,held,round(result["seconds"],2),flush=True)
    return energy.clean(result)


def choose(ep,budget,lam):
    return next(r for r in ep["results"] if r["budget"]==budget and ((r["lam"] is None and np.isinf(lam)) or r["lam"]==lam))


def select(variant):
    method,kind=VARIANTS[variant];rows=[];choices=[]
    for held in core.FAMILIES:
        train=[x for x in core.FAMILIES if x!=held]
        inner=[episode(getfit([x for x in train if x!=i],method),i,kind) for i in train]
        outer=episode(getfit(train,method),held,kind)
        for budget in (0,1,2):
            candidates=[]
            for lam in ([np.inf] if budget==0 else LAM):
                score=float(np.mean([np.inf if choose(ep,budget,lam)["MAE"] is None else choose(ep,budget,lam)["MAE"] for ep in inner]))
                candidates.append(dict(lam=lam,score=score,inner_episode_keys=[ep["key"] for ep in inner]))
            best=min(candidates,key=lambda c:(round(c["score"],10),-c["lam"]))
            r=choose(outer,budget,best["lam"])
            for row in r["records"]:rows.append(dict(**row,variant=variant,base_method=method,source_kind=kind,shift=r["shift"],fit_key=outer["fit_key"]))
            choices.append(dict(variant=variant,held=held,budget=budget,candidates=candidates,selected=best,outer_episode_key=outer["key"],
                fit_key=outer["fit_key"],shift=r["shift"],status=r["status"],MAE=r["MAE"],diagnostics=r["diagnostics"]))
    folder=OUT/"variants"/variant
    energy.csvwrite(folder/"predictions.csv",rows);energy.write(folder/"selections.json",choices)


def worker(job):
    fitkey,held,kind=job;return episode(bank()[fitkey],held,kind)["key"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=4);a=ap.parse_args();freeze();jobs=[]
    for method,kind in sorted(set(VARIANTS.values())):
        for f in bank().values():
            if f["method"]!=method:continue
            for held in core.FAMILIES:
                if held not in f["train_families"]:jobs.append((f["key"],held,kind))
    assert len(jobs)==450
    energy.write(OUT/"execution_started.json",dict(created_utc=energy.now(),jobs=len(jobs),workers=a.workers))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    for variant in VARIANTS:select(variant)
    energy.write(OUT/"execution_complete.json",dict(created_utc=energy.now(),episodes=len(jobs),variants=18,goal_complete=False))


if __name__=="__main__":main()
