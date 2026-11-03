"""Resolve the concrete 30-evaluation convergence risk without rewriting V1.

Increase the common cap to120 for every calibration with any unconverged start.
Converged trajectories stop before30 under either cap and are reused. Recompute
training-only selections and any affected outer fits; no outer-error choice.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.parameter_adaptation import regularized_adapt
import phase10_internal_adaptation as study
import phase2_energy as energy
import phase2_property as old
from csv_io_v2 import read_result_csv

OUT=study.OUT/"solver_budget_sensitivity"
PATHS={p.stem:p for p in (study.OUT/"calibrations").glob("*/*.json")}


def freeze(keys):
    path=OUT/"protocol.json"
    payload=dict(created_utc=energy.now(),version=1,cap=120,parent_cap=30,
        trigger="all V1 calibrations with ANY nonconverged start,not selected by outer error",
        triggered_keys=sorted(keys),
        semantics="identical two starts/bounds/regularization;all prior converged calibs reused;training-only model selection rerun",
        timing="afterV1convergenceaudit,retrospective numerical sensitivity;not new blind validation",
        hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),
            ROOT/"src/sandbox/parameter_adaptation.py",study.OUT/"protocol.json",study.OUT/"calibration_trials.csv")})
    if path.exists():
        x=json.loads(path.read_text());assert x["hashes"]==payload["hashes"] and x["triggered_keys"]==payload["triggered_keys"]
    else:energy.write(path,payload)


def refine(c,config=None,lam=None,override=False):
    config=c["config"] if config is None else config
    reg=(np.inf if c["lam"] is None else c["lam"]) if not override else (np.inf if lam is None else lam)
    key=old.digest(dict(parent=c["key"],config=config,lam=energy.clean(reg),protocol=energy.sha(OUT/"protocol.json")))
    path=OUT/"refined"/f"{key}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.perf_counter();base=json.loads((ROOT/c["base_path"]).read_text())
    mode=study.METHODS[c["method"]][1]
    assert mode in ("Tf","E","Tf_E")
    data=study.episode(c["task"],c["context"],c["budget"]);anchors=data[data.role.eq("anchor")]
    theta0=study.initial_coordinates(base)
    D=np.array([.05 if mode in ("Tf","Tf_E") else 0.,config.get("E_scale",.3) if mode in ("E","Tf_E") else 0.])
    lower=[study.prior.L0+(study.prior.L1-study.prior.L0)*expit(-6),np.log10(.003)]
    upper=[study.prior.L0+(study.prior.L1-study.prior.L0)*expit(6),np.log10(.2)]
    def forward(theta):
        values=[];jac=[]
        for r in anchors.itertuples():
            a,J=study.physical_point(r,base,theta);values.append(a.values[0]);jac.append(J[0])
        return np.asarray(values),np.asarray(jac).reshape(len(anchors),2)
    fit=regularized_adapt(forward,anchors.target.to_numpy(float),theta0,D,lower,upper,reg,max_nfev=120)
    raw={p["obs_id"]:p for p in base["predictions"]};rows=[]
    for original in c["rows"]:
        r=next(data[data.obs_id.eq(original["obs_id"])].itertuples())
        out=original.copy();out["calibration_key"]=key;out["calibration_status"]=fit.status
        try:
            if fit.status in ("base_no_update","base_fallback"):
                pred=raw[r.obs_id]["predicted"];valid=raw[r.obs_id].get("physical_valid")
            else:
                a,J=study.physical_point(r,base,fit.parameters);pred=float(a.values[0])
                valid=bool(a.diagnostics["gradient_valid"] and a.diagnostics["single_electron_Hall_valid"])
            out.update(predicted=pred,numeric=pred is not None,physical_valid=valid)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            out.update(predicted=None,numeric=False,physical_valid=False,error=repr(exc))
        rows.append(out)
    q=[r for r in rows if r["role"]=="query" and (c["task"]=="primary" or r["common40_125"])]
    score=float(np.mean([abs(r["predicted"]-r["observed"]) for r in q])) if all(r["numeric"] for r in q) else np.inf
    result=dict(key=key,parent_key=c["key"],method=c["method"],task=c["task"],context=c["context"],
        budget=c["budget"],config=config,lam=reg,status=fit.status,parameters=fit.parameters,
        trials=fit.trials,diagnostics=fit.diagnostics,rows=rows,score=score,seconds=time.perf_counter()-start,
        base_path=c["base_path"],base_sha256=c["base_sha256"],anchor_obs_ids=c["anchor_obs_ids"])
    energy.write(path,result);return energy.clean(result)


def one(key):return refine(json.loads(PATHS[key].read_text()))


def val(x):return np.inf if x is None else float(x)


def main():
    trials=read_result_csv(study.OUT/"calibration_trials.csv")
    keys=sorted(trials[~trials.success].calibration_key.unique());freeze(keys)
    with ProcessPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(one,keys))
    replacements={v["parent_key"]:v for v in values}
    predictions=[];selections=[];changes=[];outer_refinements=[]
    for method in study.METHODS:
        saved=json.loads((study.OUT/"methods"/method/"selections.json").read_text())
        for s in saved:
            candidates=[]
            for c in s["candidates"]:
                c=c.copy();scores=[]
                for key in c["inner_calibration_keys"]:
                    r=replacements[key] if key in replacements else json.loads(PATHS[key].read_text())
                    scores.append(val(r["score"]))
                c["score"]=float(np.mean(scores));candidates.append(c)
            best=min(candidates,key=lambda c:(round(val(c["score"]),10),-val(c["lam"]),c["index"]))
            changed=best["config"]!=s["selected"]["config"] or best["lam"]!=s["selected"]["lam"]
            scores=[]
            for key in s["outer_calibration_keys"]:
                c=json.loads(PATHS[key].read_text())
                if changed:
                    r=refine(c,best["config"],best["lam"],override=True);outer_refinements.append(r)
                else:r=replacements.get(key,c)
                predictions+=r["rows"];scores.append(val(r["score"]))
            selections.append(dict(task=s["task"],held=s["held"],method=method,budget=s["budget"],
                selected=best,outer_score=float(np.mean(scores)),selection_changed=changed))
            changes.append(dict(task=s["task"],held=s["held"],method=method,budget=s["budget"],
                selection_changed=changed,V1_score=s["score"],V2_score=float(np.mean(scores))))
    energy.csvwrite(OUT/"predictions.csv",predictions);energy.write(OUT/"selections.json",selections)
    energy.csvwrite(OUT/"selection_changes.csv",changes)
    remaining=sum(v["status"]=="adapted_nonconverged" for v in values+outer_refinements)
    report=dict(passed=remaining==0,cap=120,triggered_calibrations=len(keys),
        refined_outer_due_to_selection_changes=len(outer_refinements),
        changed_selections=sum(r["selection_changed"] for r in changes),remaining_nonconverged=remaining,
        new_optimizer_starts=sum(len(r["trials"]) for r in values+outer_refinements),
        summed_seconds=sum(r["seconds"] for r in values+outer_refinements),
        failed_query_predictions=sum(not r["numeric"] for r in predictions if r["role"]=="query"),
        protocol="all prior converged trajectories retained;only unconverged trajectories/newly selected outer configs rerun",
        mainV1_unchanged=True,selection_uses_only_inner_scores=True)
    energy.write(OUT/"convergence_audit.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
