"""Fixed-configuration test of full implicit-gradient energy learning.

This is a diagnostic Si/Sn pilot, not nested hyperparameter validation or
independent material evidence. Controls use identical features and configuration.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,json,pickle,time,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.property_metric_learning import PropertyMetricKernel
from sandbox.nonlinear_property_learning import NonlinearPropertyKernel
import phase2_energy as energy
import phase3_metric as metric

OUT=energy.BASE/"phase4_nonlinear"
CONFIG=dict(length=1.,alpha=.01,strength=1.,max_steps=6,max_lines=8)

def initialize():
    metric.frozen()
    p=OUT/"pilot_protocol.json"
    payload=dict(version=1,created_utc=energy.now(),
        stage="fixed-configuration actual nonlinear coupling pilot",
        direction="learn a shared energetic function with the full Sandbox property error and analytic implicit gradient",
        primary_targets=["Si","Sn"],energy_labels="same whole-element mask, same supplied references/catalogue",
        controls=["ordinary energy kernel","fixed true-state full metric","nonlinear full-response refinement"],
        configuration=CONFIG,configuration_status="fixed before pilot results, no target test selection",
        budget="same training energies/features; nonlinear <=54 full-response batches; report actual extra physics work rather than claim equal runtime",
        references="training-element full-energy virtual responses; no measured carrier queries",
        target_weights="same design/reference eligibility as phase3, equal training element/case and .3dex scale",
        objective="energy residual + RKHS penalty + lambda*full nonlinear property residual",
        optimizer="analytic IFT -> shared kernel coefficient gradients, positive damped Gauss-Newton, full-loss Armijo backtracking",
        guarantees="accepted training-loss descent only, no global optimum or test accuracy guarantee",
        invalid_observation_extension="intermediate n/rH outside electron-dominant/detection scope is only a mathematical training response; numeric root regularity is required; final physical coverage reported",
        validation="diagnostic retrospective pilot at one config, not completed nested or independent material validation",
        next="if implementation gates pass, decide next bounded refinement using all control/failure outcomes; do not label pilot as full study",
        code_hashes={str(q.relative_to(ROOT)):energy.sha(q) for q in
            (Path(__file__),ROOT/"src/sandbox/nonlinear_property_learning.py",
             ROOT/"src/sandbox/property_metric_learning.py",Path(metric.__file__))},
        parent_protocol_sha256=energy.sha(metric.OUT/"protocol.json"))
    if p.exists():
        assert json.loads(p.read_text())["code_hashes"]==payload["code_hashes"]
    else:energy.write(p,payload)

def response_callback(db,records,oracle):
    elements=sorted({r["element"] for r in records})
    entries=[]
    for e in elements:
        ids=np.flatnonzero([r["element"]==e for r in records])
        rr=[records[i] for i in ids]
        eligible=oracle[e]["design_eligible"]
        if eligible.any():
            scale=1/(np.sqrt(len(elements)*int(eligible.sum()))*.3)
            entries.append((e,ids,rr,eligible,oracle[e]["design_y"][eligible],scale))
    counter=dict(batch_calls=0,requested_state_solves=0)
    def callback(xi,derivatives):
        counter["batch_calls"]+=1
        residual=[];matrices=[];physical=0;cases=0
        for e,ids,rr,eligible,truth,scale in entries:
            counter["requested_state_solves"]+=len(metric.DESIGN)
            values,J,valid,diag=metric.forward(db,rr,xi[ids],metric.DESIGN,
                gradient=derivatives,allow_failures=True)
            if not np.isfinite(values[eligible]).all():
                raise RuntimeError("Nonregular training response for "+e)
            residual.extend(((values[eligible]-truth)*scale).tolist())
            physical+=int(valid[eligible].sum());cases+=int(eligible.sum())
            if derivatives:
                block=np.zeros((int(eligible.sum()),len(records)))
                block[:,ids]=J[eligible]*scale;matrices.append(block)
        return np.array(residual),(np.vstack(matrices) if derivatives else None),dict(
            training_elements=elements,response_cases=cases,
            physically_valid_responses=physical,mathematical_extension_cases=cases-physical)
    return callback,counter

def gates():
    initialize();db,r,oracle=metric.blocks()
    tr=[v for v in r if v["element"]!="Si"]
    response,counter=response_callback(db,tr,oracle)
    model=NonlinearPropertyKernel(**CONFIG).prepare(tr)
    beta=model.beta.copy()
    loss,gradient,H,meta=model.loss_and_gradient(beta,response,True)
    rng=np.random.default_rng(71)
    directions=[rng.normal(size=len(beta)),gradient.copy()]
    probes=[]
    for d in directions:
        d/=np.linalg.norm(d);h=1e-5
        ap=model.loss_and_gradient(beta+h*d,response,False)[0]
        bp=model.loss_and_gradient(beta-h*d,response,False)[0]
        fd=(ap-bp)/(2*h);an=float(gradient@d)
        probes.append(dict(analytic=an,central_difference=fd,
            scaled_error=abs(an-fd)/max(1.,abs(an),abs(fd))))
    audit=dict(initial_loss=loss,gradient_probes=probes,
        gauss_newton_min_eigenvalue=float(np.linalg.eigvalsh(H).min()),
        response_metadata=meta,response_counter=counter,
        guarantee="gradient and positive proposal metric, not optimized material prediction")
    audit["passed"]=bool(max(v["scaled_error"] for v in probes)<3e-5 and audit["gauss_newton_min_eigenvalue"]>0)
    energy.write(OUT/"gradient_gates.json",audit);print(json.dumps(audit,indent=2),flush=True)
    assert audit["passed"]

def fit_one(held):
    initialize()
    assert json.loads((OUT/"gradient_gates.json").read_text())["passed"]
    path=OUT/"pilot"/f"{held}.json"
    if path.exists():return held
    db,r,oracle=metric.blocks();tr=[v for v in r if v["element"]!=held]
    te=[v for v in r if v["element"]==held]
    output=[];predictions=[]
    for name in ("energy","local_full","nonlinear"):
        t=time.perf_counter();counter={}
        if name=="nonlinear":
            cb,counter=response_callback(db,tr,oracle)
            model=NonlinearPropertyKernel(**CONFIG).fit_nonlinear(tr,cb)
        else:
            model=PropertyMetricKernel(length=CONFIG["length"],alpha=CONFIG["alpha"],
                metric="energy" if name=="energy" else "full",
                strength=0. if name=="energy" else CONFIG["strength"]).fit(tr,oracle)
        result=metric.evaluate_model(model,te,db,oracle[held],metric.EVAL,"eval")
        q=OUT/"models"/f"{held}__{name}.pkl";q.parent.mkdir(parents=True,exist_ok=True)
        with q.open("wb") as f:pickle.dump(model,f)
        output.append(dict(element=held,model=name,energy_MAE=result["energy_MAE"],
            virtual_property_MAE=result["property_MAE"],seconds=time.perf_counter()-t,
            response_counter=counter,diagnostics=model.diagnostics,failure=result["failure"],
            held_element_excluded=held not in model.training_elements))
        for j,(T,C) in enumerate(metric.EVAL):
            predictions.append(dict(element=held,model=name,Tf_K=T,dose_cm3=C,
                reference_log_nH=result["reference"][j],predicted_log_nH=result["values"][j],
                reference_eligible=bool(result["eligible"][j]),predicted_physical_valid=bool(result["physical"][j])))
        print("nonlinear pilot",held,name,"energy",result["energy_MAE"],
              "property",result["property_MAE"],"sec",output[-1]["seconds"],flush=True)
    energy.write(path,output);energy.csvwrite(path.with_suffix(".csv"),predictions)
    return held

def run():
    with ProcessPoolExecutor(max_workers=2) as pool:
        for e in pool.map(fit_one,("Si","Sn")):print("pilot complete",e,flush=True)
    rows=[]
    for e in ("Si","Sn"):
        for r in json.loads((OUT/"pilot"/f"{e}.json").read_text()):
            rows.append(dict(element=e,model=r["model"],energy_MAE=r["energy_MAE"],
                virtual_property_MAE=r["virtual_property_MAE"],seconds=r["seconds"],
                accepted_steps=r["diagnostics"].get("accepted_steps"),
                stop_reason=r["diagnostics"].get("stop_reason"),
                response_batches=r["response_counter"].get("batch_calls",0),
                requested_state_solves=r["response_counter"].get("requested_state_solves",0)))
    pd.DataFrame(rows).to_csv(OUT/"pilot_summary.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--initialize",action="store_true")
    ap.add_argument("--gates",action="store_true");ap.add_argument("--pilot",action="store_true")
    a=ap.parse_args()
    if a.initialize:initialize()
    if a.gates:gates()
    if a.pilot:run()
