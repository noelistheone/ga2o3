"""Resolve the retained common-cap objective regression without erasing it."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase11_prediction as study
import phase2_energy as energy

PARENT=study.OUT;REF=PARENT/"solver_budget_sensitivity";OUT=REF/"retained_stall_refinement"


def main():
    failed=read_result_csv(REF/"objective_comparison.csv")
    failed=failed[~failed.not_worse]
    assert len(failed)==1
    key=failed.fit_key.iloc[0];v1=json.loads((PARENT/"fits"/(key+".json")).read_text())
    v2=json.loads((REF/"fits"/(key+".json")).read_text())
    assert len(v1["train_families"])==3,"Do not silently change an outer deployment fit"
    p=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        original_sha256=energy.sha(PARENT/"fits"/(key+".json")),
        extended_sha256=energy.sha(REF/"fits"/(key+".json")),
        failed_gate_sha256=energy.sha(REF/"sensitivity_audit.json"),
        reason="one training objective4.79e-8 higher after commoncap;V2firststart underflow;not selected by outererrors",
        solver="same analytic objective/bounds,L-BFGS-B from retained originalbest,maxiter300/maxfun500/ftol1e-13/gtol1e-8",
        preservation="V1/V2,all failures and initial passed=false gate unchanged")
    pp=OUT/"protocol.json";assert not pp.exists(),"Version before repeating"
    energy.write(pp,p)
    tr=study.training_data(v1["train_families"],v1["method"]);meta=v1["feature_metadata"]
    weights=tr.loss_weight.to_numpy(float);y=tr.target.to_numpy(float)
    nbase=4 if study.METHODS[v1["method"]]["training"]=="joint" else 2
    k=len(v1["parameters"])-nbase;reg=v1["config"].get("reg",0.)
    lower=np.array([-6.,14.]+([np.log10(.003),0.] if nbase==4 else [])+[-24.]*k)
    upper=np.array([6.,18.]+([np.log10(.2),5.] if nbase==4 else [])+[24.]*k)
    cache={};failures=[]
    def objective(w):
        key=tuple(w)
        if key not in cache:
            values=[];grads=[]
            for row in tr.itertuples():
                v,J,_,_=study.point(row,w,v1["method"],meta);values.append(v[0]);grads.append(J[0])
            residual=np.asarray(values)-y
            loss=float(weights@(residual**2)+reg*np.sum(w[nbase:]**2))
            grad=2*np.asarray(grads).T@(weights*residual);grad[nbase:]+=2*reg*w[nbase:]
            cache.clear();cache[key]=(loss,grad)
        return cache[key]
    initial=np.array(v1["parameters"],float);start_loss,start_grad=objective(initial)
    checks=[]
    for j in (0,2,4):
        step=1e-5;a=initial.copy();b=initial.copy();a[j]+=step;b[j]-=step
        fd=(objective(a)[0]-objective(b)[0])/(2*step)
        err=abs(fd-start_grad[j])/max(1.,abs(fd),abs(start_grad[j]))
        checks.append(dict(coordinate=j,analytic=start_grad[j],finite_difference=fd,error=err))
    clock=time.perf_counter()
    try:
        op=minimize(objective,initial,jac=True,method="L-BFGS-B",bounds=list(zip(lower,upper)),
            options=dict(maxiter=300,maxfun=500,ftol=1e-13,gtol=1e-8,maxls=40))
        loss,grad=objective(op.x)
        projected=op.x-np.clip(op.x-grad,lower,upper)
        optimizer=dict(parameters=op.x,loss=loss,success=bool(op.success),message=str(op.message),
            nfev=int(op.nfev),nit=int(op.nit),projected_gradient_max=float(abs(projected).max()))
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        energy.write(OUT/"failure.json",dict(error=repr(exc)));raise
    f=dict(v2);f["parameters"]=op.x;f["trials"]=v1["trials"]+v2["trials"]+[optimizer]
    predictions=[]
    for row in study.PRED.itertuples():
        v,_,dg,extra=study.point(row,op.x,f["method"],meta)
        predictions.append(dict(obs_id=row.obs_id,predicted=float(v[0]),numeric=True,
            physical_valid=bool(dg["gradient_valid"] and dg["single_electron_Hall_valid"]),
            diagnostics=dg,extra=extra,charge_outputs=v[1:]))
    f["predictions"]=predictions
    energy.write(OUT/"corrected_fit.json",f)
    lookup={}
    for source in (REF/"fits").glob("*.json"):
        record=json.loads(source.read_text())
        if record["key"]==v1["key"]:record=energy.clean(f)
        lookup[record["method"],tuple(record["train_families"]),json.dumps(record["config"],sort_keys=True)]=record
    def getfit(families,method,config):
        method,config=study.canonical(method,config)
        return lookup[method,tuple(sorted(families)),json.dumps(energy.clean(config),sort_keys=True)]
    study.fit=getfit;study.OUT=OUT;study.selected(v1["method"])
    old=json.loads((REF/"methods"/v1["method"]/"selections.json").read_text())
    new=json.loads((OUT/"methods"/v1["method"]/"selections.json").read_text())
    unchanged=all(a["selected"]["config"]==b["selected"]["config"] and a["selected"]["lam"]==b["selected"]["lam"] and a["fit_key"]==b["fit_key"] for a,b in zip(old,new))
    a=read_result_csv(REF/"methods"/v1["method"]/"primary_predictions.csv")
    b=read_result_csv(OUT/"methods"/v1["method"]/"primary_predictions.csv")
    prediction_error=float(abs(a.predicted_log_nH-b.predicted_log_nH).max())
    passed=(loss<=start_loss+1e-10 and max(x["error"] for x in checks)<2e-6 and unchanged and prediction_error<1e-12)
    audit=dict(passed=passed,created_utc=energy.now(),fit_key=v1["key"],original_objective=start_loss,
        commoncap_objective=min(t["loss"] for t in v2["trials"] if t.get("loss") is not None),
        corrected_objective=loss,optimizer=optimizer,actual_state_gradient_checks=checks,
        all_config_lambda_and_outer_fit_choices_unchanged=unchanged,
        outer_primary_prediction_maximum_change=prediction_error,
        new_source_deployment_unchanged="corrected fit has3trainingfamilies;global deploymentCV uses4family fits only",
        seconds=time.perf_counter()-clock,original_failed_gate_preserved=True,
        scope="targeted numerical correction,not material evidence or global convergence proof")
    energy.write(OUT/"audit.json",audit);print(json.dumps(energy.clean(audit),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
