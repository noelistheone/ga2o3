"""Nested whole-element test of full nonlinear Sandbox-response training."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os,sys,json,pickle,time,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.nonlinear_property_learning import NonlinearPropertyKernel
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase3_metric as metric
import phase4_nonlinear_pilot as pilot

OUT=pilot.OUT
CONFIGS=[dict(length=l,alpha=a,strength=1.,max_steps=6,max_lines=8)
         for a,l in product(metric.ALPHAS,metric.LENGTHS)]
def freeze():
    metric.frozen();pilot.initialize()
    assert json.loads((OUT/"gradient_gates.json").read_text())["passed"]
    p=OUT/"nested_protocol.json"
    payload=dict(version=1,created_utc=energy.now(),
        objective="test a cross-condition shared energy learner trained through full nonlinear Sandbox responses",
        outer_elements=["Si","Sn"],inner="leave each of remaining14 elements out",
        scope="the two elements of the real29-record carrier cohort; not a new15-element accuracy claim",
        same_features_labels_references=True,
        configs=CONFIGS,primary_candidates=6,
        baseline="phase3 ordinary energy kernel, same6 alpha/length candidates and inner nonlinear property criterion",
        strength="fixed1; no external outcome tuning",
        compute_budget="same6 candidate selection grid, same supplied energetic labels; <=54response batches per nonlinear fit, ordinary fits use less. Actual extra root/time cost reported, not equal-runtime claim.",
        design=metric.DESIGN,evaluation=metric.EVAL,
        primary_selection="eligible-element-equal full nonlinear inner property MAE, any numerical refusal -> infinity",
        secondary_gating="12-candidate nonlinear-or-energy choice using inner scores only; separate from6-candidate causal comparison",
        code_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in
            (Path(__file__),Path(pilot.__file__),ROOT/"src/sandbox/nonlinear_property_learning.py")},
        parent_protocol_sha256=energy.sha(metric.OUT/"protocol.json"),
        data_grade="retrospective published-energy forward references, no experimental query labels",
        physical_validity="report all regular numeric responses and electron-Hall coverage separately",
        convergence="finite-budget local optimization, no global optimality claim")
    if p.exists():
        assert json.loads(p.read_text())["code_hashes"]==payload["code_hashes"]
    else:energy.write(p,payload)

def check_descent(model):
    bad=[]
    for i,r in enumerate(model.diagnostics.get("optimization_trace",[])):
        for t in r.get("trials",[]):
            if t.get("accepted") and t["objective"]>r["objective"]+1e-8:
                bad.append(i)
    if bad:raise RuntimeError("Accepted nonlinear training loss increased")
    return True

def inner_job(task):
    held,k=task;freeze();cfg=CONFIGS[k];db,r,oracle=metric.blocks()
    tr=[v for v in r if v["element"]!=held]
    elems=sorted({v["element"] for v in tr})
    rows=[];started=time.perf_counter()
    for inner in elems:
        p=OUT/"nested_inner"/held/f"config{k}"/f"{inner}.json"
        if p.exists():row=json.loads(p.read_text())
        else:
            a=[v for v in tr if v["element"]!=inner];b=[v for v in tr if v["element"]==inner]
            response,counter=pilot.response_callback(db,a,oracle)
            t=time.perf_counter()
            model=NonlinearPropertyKernel(**cfg).fit_nonlinear(a,response)
            fit_seconds=time.perf_counter()-t
            assert held not in model.training_elements and inner not in model.training_elements
            check_descent(model)
            result=metric.evaluate_model(model,b,db,oracle[inner],metric.DESIGN,"design")
            score=result["property_MAE"]
            row=dict(held_element=held,inner_element=inner,config=cfg,eligible=int(result["eligible"].sum()),
                property_MAE=score if score is None or np.isfinite(score) else None,
                numerical_refusal=bool(score is not None and not np.isfinite(score)),
                energy_MAE=result["energy_MAE"],failure=result["failure"],
                train_elements=model.training_elements,fit_seconds=fit_seconds,
                response_counter=counter,diagnostics=model.diagnostics,accepted_descent_checked=True)
            energy.write(p,row)
        rows.append(row)
    active=[(np.inf if v["numerical_refusal"] else v["property_MAE"])
            for v in rows if v["eligible"]>0]
    out=dict(held=held,index=k,config=cfg,
        inner_property_MAE=float(np.mean(active)) if active else None,
        incomplete_selection=bool(any(v["numerical_refusal"] for v in rows)),
        inner_energy_MAE=float(np.mean([v["energy_MAE"] for v in rows])),
        fit_seconds=float(sum(v["fit_seconds"] for v in rows)),
        requested_state_solves=int(sum(v["response_counter"]["requested_state_solves"] for v in rows)),
        details=rows,seconds_this_job=time.perf_counter()-started)
    energy.write(OUT/"nested_candidates"/f"{held}__{k}.json",out)
    print("nonlinear candidate",held,k,"inner",out["inner_property_MAE"],
          "fit seconds",round(out["fit_seconds"],2),flush=True)
    return held,k

def outer(held):
    if (OUT/"nested_outer"/f"{held}.json").exists():return
    db,r,oracle=metric.blocks()
    train=[v for v in r if v["element"]!=held];test=[v for v in r if v["element"]==held]
    candidates=[json.loads((OUT/"nested_candidates"/f"{held}__{k}.json").read_text()) for k in range(len(CONFIGS))]
    def score(v):
        return np.inf if v["incomplete_selection"] or v["inner_property_MAE"] is None else v["inner_property_MAE"]
    best=min(candidates,key=lambda v:(round(score(v),10),-v["config"]["alpha"],-v["config"]["length"]))
    response,counter=pilot.response_callback(db,train,oracle)
    model=NonlinearPropertyKernel(**best["config"]).fit_nonlinear(train,response);check_descent(model)
    old=json.loads((metric.OUT/"outer"/f"{held}.json").read_text())
    ordinary=next(v for v in old["selections"] if v["method"]=="kernel_energy")
    ordinary_score=ordinary["selected"]["inner_property_MAE"]
    choose_nonlinear=bool(score(best)<ordinary_score-1e-10)
    with (metric.OUT/"models"/f"{held}__kernel_energy.pkl").open("rb") as f:energy_model=pickle.load(f)
    models={"kernel_nonlinear_l1":model,
            "kernel_nonlinear_or_energy":model if choose_nonlinear else energy_model}
    erows=[];prows=[];selections=[]
    for name,m in models.items():
        p=OUT/"models"/f"{held}__{name}.pkl"
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("wb") as f:pickle.dump(m,f)
        result=metric.evaluate_model(m,test,db,oracle[held],metric.EVAL,"eval")
        for rec,pred in zip(test,result["predicted_crossings"]):
            erows.append(dict(**rec,method=name,predicted_crossing_eV=float(pred),
                training_elements="|".join(m.training_elements)))
        for j,(T,C) in enumerate(metric.EVAL):
            prows.append(dict(element=held,method=name,Tf_K=T,dose_cm3=C,
                reference_log_nH=result["reference"][j],predicted_log_nH=result["values"][j],
                reference_eligible=bool(result["eligible"][j]),predicted_physical_valid=bool(result["physical"][j])))
        selections.append(dict(method=name,selected_candidate=best,
            inner_nonlinear_MAE=score(best),inner_ordinary_MAE=ordinary_score,
            adaptive_choice="nonlinear" if choose_nonlinear else "ordinary",
            energy_MAE=result["energy_MAE"],virtual_property_MAE=result["property_MAE"],
            diagnostics=getattr(m,"diagnostics",{}),response_counter=counter if m is model else {},
            comparison_grade="6-candidate primary" if name=="kernel_nonlinear_l1" else "12-candidate secondary"))
    energy.csvwrite(OUT/"energy_outer"/f"{held}.csv",erows)
    energy.csvwrite(OUT/"property_outer"/f"{held}.csv",prows)
    energy.write(OUT/"nested_outer"/f"{held}.json",dict(held=held,selections=selections,
        candidates=candidates,training_elements=model.training_elements))
    print("nested nonlinear outer",held,
          [(v["method"],v["virtual_property_MAE"]) for v in selections],flush=True)

def run(workers):
    freeze()
    tasks=list(product(("Si","Sn"),range(len(CONFIGS))))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for held,k in pool.map(inner_job,tasks):print("candidate complete",held,k,flush=True)
    for e in ("Si","Sn"):outer(e)
    p=pd.concat([read_result_csv(OUT/"energy_outer"/f"{e}.csv") for e in ("Si","Sn")],ignore_index=True)
    p.to_csv(OUT/"energy_predictions.csv",index=False)
    q=pd.concat([read_result_csv(OUT/"property_outer"/f"{e}.csv") for e in ("Si","Sn")],ignore_index=True)
    q.to_csv(OUT/"virtual_property_predictions.csv",index=False)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--workers",type=int,default=4)
    x=a.parse_args();run(x.workers)
