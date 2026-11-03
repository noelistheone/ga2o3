"""Targeted bounded scalar-objective solve for the two remaining TRF stalls."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import phase10_internal_adaptation as study
import phase2_energy as energy
import phase2_property as old
from csv_io_v2 import read_result_csv


def main():
    out=study.OUT/"solver_budget_sensitivity";dest=out/"stalled_refinement"
    records=[];replacement={};resolved=[]
    for path in (out/"refined").glob("*.json"):
        c=json.loads(path.read_text())
        replacement[c["parent_key"]]=c
        if c["status"]!="adapted_nonconverged":continue
        base=json.loads((ROOT/c["base_path"]).read_text());data=study.episode(c["task"],c["context"],c["budget"])
        anchors=data[data.role.eq("anchor")];theta0=study.initial_coordinates(base);D=np.array([.05,c["config"]["E_scale"]])
        lo=(np.array([study.prior.L0+(study.prior.L1-study.prior.L0)*expit(-6),np.log10(.003)])-theta0)/D
        hi=(np.array([study.prior.L0+(study.prior.L1-study.prior.L0)*expit(6),np.log10(.2)])-theta0)/D
        cache={}
        def calc(z):
            key=tuple(z)
            if key not in cache:
                vals=[];js=[]
                for r in anchors.itertuples():
                    p,J=study.physical_point(r,base,theta0+D*z);vals.append(p.values[0]);js.append(J[0]*D)
                residual=np.asarray(vals)-anchors.target.to_numpy(float)
                cache.clear();cache[key]=(float(residual@residual+c["lam"]*(z@z)),
                    2*(np.asarray(js).T@residual+c["lam"]*z))
            return cache[key]
        trials=[]
        for z in ((np.asarray(c["parameters"])-theta0)/D,np.zeros(2)):
            op=minimize(lambda z:calc(z),z,jac=True,method="L-BFGS-B",bounds=list(zip(lo,hi)),
                options=dict(maxiter=300,maxfun=600,ftol=1e-14,gtol=1e-8,maxls=40))
            g=op.jac.copy();g[(op.x<=lo+1e-7)&(g>0)]=0;g[(op.x>=hi-1e-7)&(g<0)]=0
            trials.append(dict(standardized_parameters=op.x,objective=float(op.fun),success=bool(op.success),
                projected_gradient_max=float(abs(g).max()),message=str(op.message),nfev=int(op.nfev)))
        best=min(trials,key=lambda x:x["objective"]);theta=theta0+D*best["standardized_parameters"]
        key=old.digest(dict(parent=c["key"],method="L-BFGS-B_bound_refinement"));rows=[]
        for r in c["rows"]:
            inp=next(data[data.obs_id.eq(r["obs_id"])].itertuples());a,J=study.physical_point(inp,base,theta)
            r=r.copy();r.update(predicted=float(a.values[0]),numeric=True,calibration_key=key,
                calibration_status="adapted_bounded_scalar_refinement");rows.append(r)
        query=[r for r in rows if r["role"]=="query" and r["common40_125"]]
        score=float(np.mean([abs(r["predicted"]-r["observed"]) for r in query]))
        v=dict(key=key,parent_key=c["parent_key"],prior_refinement_key=c["key"],method=c["method"],
            task=c["task"],context=c["context"],budget=c["budget"],parameters=theta,trials=trials,
            rows=rows,score=score,original_objective=c["diagnostics"]["objective"],
            projected_gradient_max=best["projected_gradient_max"],
            converged=best["projected_gradient_max"]<1e-6 and best["objective"]<=c["diagnostics"]["objective"]+1e-10)
        energy.write(dest/f"{key}.json",v);replacement[c["parent_key"]]=energy.clean(v);resolved.append(energy.clean(v))
    final=read_result_csv(out/"predictions.csv");maximum_change=0.
    for r in resolved:
        prior_key=r["prior_refinement_key"];mask=final.calibration_key.eq(prior_key)
        lookup={v["obs_id"]:v for v in r["rows"]}
        for index in final[mask].index:
            oldvalue=final.loc[index,"predicted"];new=lookup[final.loc[index,"obs_id"]]
            maximum_change=max(maximum_change,abs(oldvalue-new["predicted"]))
            for field in ("predicted","calibration_key","calibration_status"):final.loc[index,field]=new[field]
    changes=[]
    for method in study.METHODS:
        original=json.loads((study.OUT/"methods"/method/"selections.json").read_text())
        v2=[s for s in json.loads((out/"selections.json").read_text()) if s["method"]==method]
        for s,s2 in zip(original,v2):
            candidates=[]
            for candidate in s["candidates"]:
                candidate=candidate.copy();values=[]
                for key in candidate["inner_calibration_keys"]:
                    r=replacement.get(key)
                    if r is None:r=json.loads(next((study.OUT/"calibrations").glob("*/"+key+".json")).read_text())
                    values.append(np.inf if r["score"] is None else r["score"])
                candidate["score"]=float(np.mean(values));candidates.append(candidate)
            best=min(candidates,key=lambda c:(round(c["score"],10),-(np.inf if c["lam"] is None else c["lam"]),c["index"]))
            changes.append(best["lam"]!=s2["selected"]["lam"] or best["config"]!=s2["selected"]["config"])
    assert not any(changes),"Refinement changed a training-only choice;evaluate new outer settings before finalizing"
    final.to_csv(out/"predictions_resolved.csv",index=False)
    report=dict(passed=all(r["converged"] for r in resolved),stalled_cases=len(resolved),
        cases=[dict(parent_key=r["parent_key"],projected_gradient_max=r["projected_gradient_max"],converged=r["converged"]) for r in resolved],
        training_only_choices_changed=sum(changes),maximum_outer_prediction_change_from120=maximum_change,
        method="L-BFGS-B,analyticIFT gradient,two starts including prior120 solution;projectedKKT check",
        code_sha256=energy.sha(Path(__file__)),
        parent_budget_protocol_sha256=energy.sha(out/"protocol.json"),
        interpretation="targeted convergence resolution,originalV1/V2 failures retained;no outer-error choice")
    energy.write(out/"stalled_refinement_audit.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
