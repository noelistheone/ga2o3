"""Uniform solver-risk refinement; no outer scores choose affected fits."""
from pathlib import Path
import sys,json,time,argparse,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import least_squares
import phase13_prediction as study
import phase2_energy as energy

ORIGINAL=study.OUT;OUT=ORIGINAL/"solver_budget_v2"


def load_composite():
    rows={}
    for path in (ORIGINAL/"fits").glob("*.json"):
        f=json.loads(path.read_text());f["composite_origin"]=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path))
        rows[f["key"]]=f
    for path in (OUT/"fits").glob("*.json"):
        f=json.loads(path.read_text());f["composite_origin"]=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path))
        rows[f["key"]]=f
    return rows


def refine():
    study.freeze();oldfiles=sorted((ORIGINAL/"fits").glob("*.json"))
    affected=[p for p in oldfiles if any(not t["success"] for t in json.loads(p.read_text())["trials"])]
    assert len(affected)==1
    protocol=dict(created_utc=energy.now(),code_hashes={str(Path(__file__).relative_to(ROOT)):energy.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in [ORIGINAL/"prediction_protocol.json",*affected]},
        eligibility="ALLrecords with any original failedstart;notouterpredictionranking",
        starts="both saved original initializations plus saved best;max_nfev360,unchangedobjective/bounds/tolerances/IFT",
        original_artifacts="all405unchanged;composite bank overrides onlyrefinedrecord",goal_complete=False)
    path=OUT/"protocol.json"
    if path.exists():
        old=json.loads(path.read_text())
        for key in ("code_hashes","input_hashes"):assert old[key]==protocol[key],key
    else:energy.write(path,protocol)
    executions=[]
    for path in affected:
        old=json.loads(path.read_text());dest=OUT/"fits"/path.name
        if dest.exists():continue
        f=copy.deepcopy(old);method=f["method"];m=study.mapper(method);lo,hi,_=study.specification(method)
        tr=study.training_data(f["train_families"],method);y=tr.target.to_numpy(float);sw=np.sqrt(tr.loss_weight.to_numpy(float))
        cache={};failures=[];before=m.counters();clock=time.perf_counter()
        def calc(w):
            k=tuple(w)
            if k not in cache:
                vals=[];js=[]
                for row in tr.itertuples():
                    try:a=m.point(row,w)
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        failures.append(dict(obs_id=row.obs_id,parameters=w,error=repr(exc)));raise
                    vals.append(a.values[0]);js.append(a.jacobian[0])
                cache.clear();cache[k]=(np.array(vals),np.array(js))
            return cache[k]
        starts=[t["initial"] for t in old["trials"]]+[old["parameters"]];trials=[]
        for w in starts:
            try:
                op=least_squares(lambda w:(calc(w)[0]-y)*sw,w,jac=lambda w:calc(w)[1]*sw[:,None],bounds=(lo,hi),
                    max_nfev=360,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(initial=w,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),nfev=int(op.nfev),
                    njev=int(op.njev),optimality=float(op.optimality),active_mask=op.active_mask,message=op.message))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                trials.append(dict(initial=w,parameters=w,loss=None,success=False,error=repr(exc)))
        best=min((t for t in trials if t["loss"] is not None),key=lambda t:t["loss"])
        oldbest=min(t["loss"] for t in old["trials"] if t["loss"] is not None)
        check=dict(passed=best["loss"]<=oldbest+1e-10,key=f["key"],original_objective=oldbest,new_objective=best["loss"],
            new_starts=len(trials),unconverged=sum(not t["success"] for t in trials),selected_converged=best["success"])
        energy.write(OUT/"refinement_gate.json",check)
        assert check["passed"],"Retain failed gate;do not replace with outer-selected solution"
        parameters=best["parameters"];training_counts={k:m.counters()[k]-before[k] for k in before};predictions=[]
        for row in study.PRED.itertuples():
            try:
                a=m.point(row,parameters);g=a.diagnostics
                dg={k:g[k] for k in ("root_residual_log","jacobian_condition","free_n_cm3","free_p_cm3","Tf_K",
                    "measurement_T_K","Nd_cm3","profile_sha256","localized_fraction","donor_subgroups","same_physical_state_for_output")}
                predictions.append(dict(obs_id=row.obs_id,predicted=float(a.values[0]),charge_outputs=a.values[1:],numeric=True,
                    physical_valid=True if not f["definition"].get("output") else None,diagnostics=dg))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                predictions.append(dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc)))
        counts={k:m.counters()[k]-before[k] for k in before}
        f.update(parameters=parameters,trials=trials,original_trials=old["trials"],state_failures=failures,
            predictions=predictions,counters=counts,training_counters=training_counts,
            cached_prediction_counters={k:counts[k]-training_counts[k] for k in counts},seconds=time.perf_counter()-clock,
            refinement=dict(original_fit_sha256=energy.sha(path),original_fit_path=str(path.relative_to(ROOT)),gate=check))
        energy.write(dest,f);executions.append(dict(key=f["key"],**counts,seconds=f["seconds"],**check))
        print("refined",method,check,flush=True)
    energy.write(OUT/"refinement_execution_complete.json",dict(created_utc=energy.now(),affected_records=len(affected),records=executions,goal_complete=False))


def adapter():
    fits=load_composite();lookup={(f["method"],tuple(sorted(f["train_families"])),json.dumps(f["config"],sort_keys=True)):f for f in fits.values()}
    def getfit(families,method,config):return lookup[method,tuple(sorted(families)),json.dumps(config,sort_keys=True)]
    study.fit=getfit
    study.OUT=OUT
    return fits


def select():
    assert json.loads((OUT/"refinement_gate.json").read_text())["passed"]
    adapter()
    for method in study.METHODS:study.select(method)
    energy.write(OUT/"selection_execution_complete.json",dict(created_utc=energy.now(),models=len(study.METHODS),goal_complete=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--select-only",action="store_true");a=ap.parse_args()
    if not a.select_only:refine()
    select()


if __name__=="__main__":main()
