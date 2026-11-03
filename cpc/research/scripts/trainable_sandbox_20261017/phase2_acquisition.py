"""Replay an equal-budget energy acquisition loop through the full Sandbox.

The revealed labels are existing published energies, not newly performed DFT.
Primary outputs compare against a full-energy Sandbox on new virtual inputs.
This isolates model/cost coupling; it is not independent material validation.
"""
from pathlib import Path
from itertools import product
import os,sys,json,pickle,time,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sandbox.energy_learning import features, group_weights, ChargeEnergyLearner, reconstruct
from sandbox.implicit_defects import ImplicitDefectLayer, ImplicitDomainError
from sandbox.goal_acquisition import (
    condition_coordinate,prediction_variance_reduction,validate_covariance)
from csv_io import read_result_csv
import phase2_energy as energy

OUT=energy.OUT/"acquisition"
DESIGN=list(product((850.,1325.,1700.),(1e17,1e18,1e19,1e20)))
EVAL=list(product((1100.,1500.),(3e17,3e18,3e19,3e20)))
METHODS=("rbf","extra_trees")
MAX_BUDGET=8
SEEDS=list(range(32))

def freeze():
    energy.frozen()
    p=OUT/"protocol.json"
    contract=dict(version=1,created_utc=energy.now(),
        code_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in
                     (Path(__file__),ROOT/"src/sandbox/goal_acquisition.py")},
        parent_protocol_sha256=energy.sha(energy.OUT/"protocol.json"),
        goal="learn missing charge energetics, use Sandbox derivatives to select energies that improve doped-carrier predictions",
        prior_models=METHODS,elements=["Si","Sn"],
        covariance="RBF latent posterior or tree empirical covariance, plus diagonal training-LOEO mean-square discrepancy; Gaussian moment approximation",
        calibration="training elements only; not a certified Bayesian posterior or empirical coverage guarantee",
        selection="goal local mean variance reduction; marginal energy variance; random32 seeds",
        design_inputs=DESIGN,evaluation_inputs=EVAL,
        process="fixed declared Tf grid, Nd9.1e15cm-3,pO2=1e-5atm,300K observation",
        active_budget=list(range(MAX_BUDGET+1)),
        observations="exact reveal of previously masked published xi; zero added replay noise, not zero DFT systematic error",
        cost="one previously masked original charge-state energy revealed per step; exact parent references and all unmasked energies supplied",
        primary_metric="MAE of log10 nH relative to full-energy Sandbox at 8 distinct evaluation inputs",
        secondary_metric="remaining energy reconstruction error, physical/numeric coverage, wall time",
        no_material_measurement_labels_used=True,
        validation="retrospective conditional published-label replay; virtual forward targets, not new material data",
        theorem_scope="Gaussian linearized trace reduction exact; nonlinear realized error need not improve monotonically",
        source=dict(url="https://arxiv.org/abs/1802.06517",
            doi="10.1088/1361-6420/aad210",title="Attia et al. goal-oriented OED2018",
            reading="abstract and main PDF sections2-3 pp4-6; not full-paper replication",
            novelty="goal-oriented design has prior art; no claim to invent Gaussian conditioning/A-optimality"))
    if p.exists():
        old=json.loads(p.read_text())
        if old["code_hashes"]!=contract["code_hashes"]:
            raise RuntimeError("Version acquisition code before changing frozen replay")
    else:energy.write(p,contract)

def prior(element,method):
    path=OUT/"priors"/f"{element}__{method}.npz"
    db,rr,_=energy.records()
    targets=[r for r in rr if r["element"]==element]
    if path.exists():
        x=np.load(path);return db,targets,x["mean"],x["cov"]
    tr=[r for r in rr if r["element"]!=element]
    with (energy.OUT/"models"/f"{element}__{method}.pkl").open("rb") as f:model=pickle.load(f)
    errors=[];valrecords=[]
    for held in sorted({r["element"] for r in tr}):
        a=[r for r in tr if r["element"]!=held];b=[r for r in tr if r["element"]==held]
        m=ChargeEnergyLearner(method,model.config).fit(a)
        errors.extend((m.predict(b)-np.array([r["crossing_eV"] for r in b])).tolist())
        valrecords.extend(b)
    w=group_weights(valrecords)
    variance=float(np.average(np.array(errors)**2,weights=w))
    X=(features(targets)-model.center)/model.scale
    if method=="extra_trees":
        predictions=np.array([t.predict(X)+model.mean for t in model.model.estimators_])
        C=np.cov(predictions,rowvar=False,ddof=1)
    else:
        T=(features(tr)-model.center)/model.scale
        gamma=model.model.gamma
        K=rbf_kernel(T,T,gamma=gamma)
        Kq=rbf_kernel(X,T,gamma=gamma)
        Kqq=rbf_kernel(X,X,gamma=gamma)
        weights=group_weights(tr)
        A=K+np.diag(model.config["alpha"]/weights)
        yy=np.array([r["crossing_eV"] for r in tr])
        signal=float(np.average((yy-model.mean)**2,weights=weights))
        C=signal*(Kqq-Kq@np.linalg.solve(A,Kq.T))
    C=.5*(C+C.T)+variance*np.eye(len(targets))
    validate_covariance(C)
    mean=model.predict(targets)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,mean=mean,cov=C,label_ids=np.array([r["label_id"] for r in targets]))
    energy.write(path.with_suffix(".json"),dict(element=element,method=method,
        training_elements=model.training_elements,config=model.config,
        training_only_CV_discrepancy_variance_eV2=variance,
        gaussian_prior_is_assumption=True,independent_material_validation=False))
    return db,targets,mean,C

def state_predictions(db,targets,mean,inputs,derivatives=False):
    altered=reconstruct(db,targets,mean)
    B=np.zeros((db.nCS,len(targets))) if derivatives else None
    if derivatives:
        for j,r in enumerate(targets):B[r["index"],j]=-r["delta_q"]
    element=targets[0]["element"]
    layer=ImplicitDefectLayer(altered,(element,),B)
    values=[];jac=[];valid=[]
    for T,C in inputs:
        pars=layer.parameters(T=T,Nd=9.1e15,doses=(C,),pO2=1e-5)
        p=layer.predict(pars)
        values.append(p.values[0])
        if derivatives:jac.append(p.jacobian[0,2:2+len(targets)])
        valid.append(bool(p.diagnostics["single_electron_Hall_valid"] and
                          p.diagnostics["above_carrier_detection_floor"]))
    return np.array(values),np.array(jac),np.array(valid)

def numerical_gates():
    rng=np.random.default_rng(184)
    X=rng.normal(size=(7,7));C=X@X.T+.2*np.eye(7)
    J=rng.normal(size=(3,7));m=rng.normal(size=7)
    score=prediction_variance_reduction(J,C)
    errors=[]
    for i in range(7):
        a,b=condition_coordinate(m,C,i,float(rng.normal()))
        direct=np.trace(J@(C-b)@J.T)/len(J)
        errors.append(abs(direct-score[i]))
    indices=[1,4,6];observations=rng.normal(size=3)
    sm=m.copy();sc=C.copy()
    for i,y in zip(indices,observations):sm,sc=condition_coordinate(sm,sc,i,y)
    S=C[np.ix_(indices,indices)]
    batch_m=m+C[:,indices]@np.linalg.solve(S,observations-m[indices])
    batch_C=C-C[:,indices]@np.linalg.solve(S,C[indices,:])
    audit=dict(trace_utility_max_error=float(max(errors)),
        sequential_batch_mean_error=float(abs(sm-batch_m).max()),
        sequential_batch_covariance_error=float(abs(sc-batch_C).max()))
    probes=[]
    for e in ("Si","Sn"):
        db,rr,mean,cov=prior(e,"extra_trees")
        pred,J,valid=state_predictions(db,rr,mean,[(1325.,1e19)],True)
        order=np.argsort(-np.abs(J[0]))[:4]
        for j in order:
            step=1e-5;a=mean.copy();b=mean.copy();a[j]+=step;b[j]-=step
            ap=state_predictions(db,rr,a,[(1325.,1e19)])[0][0]
            bp=state_predictions(db,rr,b,[(1325.,1e19)])[0][0]
            fd=(ap-bp)/(2*step)
            error=abs(fd-J[0,j])/max(1.,abs(fd),abs(J[0,j]))
            probes.append(dict(element=e,label=rr[j]["label_id"],analytic=float(J[0,j]),
                finite_difference=float(fd),scaled_error=float(error)))
    audit["energy_to_property_gradient_probes"]=probes
    audit["passed"]=bool(max(audit[k] for k in (
        "trace_utility_max_error","sequential_batch_mean_error","sequential_batch_covariance_error"))<1e-10
        and max(r["scaled_error"] for r in probes)<3e-5)
    energy.write(OUT/"numerical_gates.json",audit)
    assert audit["passed"],audit
    print("acquisition numerical gates",audit["passed"],flush=True)

def run():
    freeze()
    numerical_gates()
    allrows=[]
    for element,method in product(("Si","Sn"),METHODS):
        db,rr,mean,C=prior(element,method)
        truth=np.array([r["crossing_eV"] for r in rr])
        oracle,_,oracle_valid=state_predictions(db,rr,truth,EVAL)
        policies=[("goal",0),("energy_variance",0)]+[("random",s) for s in SEEDS]
        for policy,seed in policies:
            dest=OUT/"replays"/f"{element}__{method}__{policy}__{seed}.csv"
            if dest.exists():
                with dest.open(newline="") as f:
                    import csv
                    allrows.extend(list(csv.DictReader(f)))
                continue
            m=mean.copy();cov=C.copy();acquired=[];rows=[]
            rng=np.random.default_rng(seed+1729)
            started=time.perf_counter()
            for budget in range(MAX_BUDGET+1):
                try:
                    y,_,physical=state_predictions(db,rr,m,EVAL)
                    remaining=[i for i in range(len(rr)) if i not in acquired]
                    energy_mae=float(np.mean([abs(rr[i]["delta_q"]*(m[i]-truth[i])) for i in remaining]))
                    for i,(T,dose) in enumerate(EVAL):
                        rows.append(dict(element=element,model=method,policy=policy,seed=seed,budget=budget,
                            Tf_K=T,dose_cm3=dose,predicted_log_nH=float(y[i]),
                            oracle_log_nH=float(oracle[i]),physical_valid=bool(physical[i]),
                            oracle_physical_valid=bool(oracle_valid[i]),
                            remaining_energy_MAE_eV=energy_mae,
                            acquired_labels="|".join(rr[j]["label_id"] for j in acquired),
                            numeric=True,failure="",elapsed_seconds=time.perf_counter()-started))
                    if budget==MAX_BUDGET:break
                    if policy=="goal":
                        _,jac,_=state_predictions(db,rr,m,DESIGN,True)
                        scores=prediction_variance_reduction(jac,cov)
                    elif policy=="energy_variance":
                        # Compare uncertainty in original A_q, not only xi.
                        dq=np.array([r["delta_q"] for r in rr])
                        scores=np.diag(cov)*dq**2
                    else:
                        scores=np.zeros(len(rr));scores[int(rng.choice(remaining))]=1.
                    scores[acquired]=-np.inf
                    j=int(np.argmax(scores))
                    m,cov=condition_coordinate(m,cov,j,truth[j])
                    acquired.append(j)
                except (ImplicitDomainError,np.linalg.LinAlgError,ValueError) as exc:
                    rows.append(dict(element=element,model=method,policy=policy,seed=seed,budget=budget,
                        numeric=False,failure=repr(exc),
                        acquired_labels="|".join(rr[j]["label_id"] for j in acquired),
                        elapsed_seconds=time.perf_counter()-started))
                    break
            energy.csvwrite(dest,rows);allrows.extend(rows)
            if policy!="random" or seed%8==0:
                print("replay",element,method,policy,seed,"steps",len(acquired),
                      "sec",round(time.perf_counter()-started,2),flush=True)
    energy.csvwrite(OUT/"predictions.csv",allrows)
    summarize()

def summarize():
    files=sorted((OUT/"replays").glob("*.csv"))
    d=pd.concat([read_result_csv(p) for p in files],ignore_index=True)
    d.to_csv(OUT/"predictions.csv",index=False)
    rows=[]
    for keys,g in d.groupby(["element","model","policy","seed","budget"]):
        numeric=g.numeric.eq(True)
        complete=bool(len(g)==len(EVAL) and numeric.all())
        error=abs(g.predicted_log_nH-g.oracle_log_nH) if complete else pd.Series(dtype=float)
        rows.append(dict(element=keys[0],model=keys[1],policy=keys[2],seed=int(keys[3]),budget=int(keys[4]),
            complete=complete,MAE=float(error.mean()) if complete else None,
            max_error=float(error.max()) if complete else None,
            physical=int(g.physical_valid.sum()) if "physical_valid" in g else 0,
            n=len(g),remaining_energy_MAE_eV=float(g.remaining_energy_MAE_eV.mean()) if complete else None))
    by=pd.DataFrame(rows);by.to_csv(OUT/"by_replay.csv",index=False)
    summary=by.groupby(["element","model","policy","budget"]).agg(
        replays=("seed","count"),complete_replays=("complete","sum"),
        MAE_mean=("MAE","mean"),MAE_min=("MAE","min"),MAE_max=("MAE","max"),
        remaining_energy_MAE=("remaining_energy_MAE_eV","mean")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    print(summary[summary.budget.isin([0,1,2,4,8])].to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--initialize",action="store_true")
    ap.add_argument("--summarize",action="store_true")
    a=ap.parse_args()
    if a.initialize:freeze()
    elif a.summarize:summarize()
    else:run()
