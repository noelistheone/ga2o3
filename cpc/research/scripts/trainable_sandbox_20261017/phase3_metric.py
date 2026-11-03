"""Property-sensitive shared energy learning, with whole-element exclusion.

Training property references are computed from supplied published energies.
They are not new material measurements. The frozen real-source task is separate.
"""
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import os,sys,json,csv,pickle,hashlib,time,argparse
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sandbox.energy_learning import (
    ChargeEnergyLearner,features,group_weights,macro_error,reconstruct)
from sandbox.property_metric_learning import PropertyMetricKernel,element_database
from sandbox.implicit_defects import ImplicitDefectLayer,ImplicitDomainError
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"phase3_target_loss"
DESIGN=list(product((850.,1325.,1700.),(1e17,1e18,1e19,1e20)))
EVAL=list(product((1000.,1450.,1850.),(2e17,2e19)))
ALPHAS=(1e-4,1e-2,1.)
LENGTHS=(.5,2.)
SPECS={"kernel_energy":[dict(metric="energy",strength=0.,alpha=a,length=l)
                         for a,l in product(ALPHAS,LENGTHS)]}
for metric in ("diagonal","full"):
    for strength in (.1,1.,10.):
        name=f"kernel_{metric}_l{strength:g}"
        SPECS[name]=[dict(metric=metric,strength=strength,alpha=a,length=l)
                     for a,l in product(ALPHAS,LENGTHS)]
SPECS["ridge"]=[dict(alpha=a) for a in (.01,.1,1.,10.,100.,10000.)]
SPECS["global_mean"]=[{}]
SPECS["charge_pair_mean"]=[{}]

def frozen():
    energy.frozen()
    p=json.loads((OUT/"protocol.json").read_text())
    for rel,h in p["code_hashes"].items():
        if energy.sha(ROOT/rel)!=h:raise RuntimeError("Frozen phase3 code changed: "+rel)
    for rel,h in p["input_hashes"].items():
        if energy.sha(ROOT/rel)!=h:raise RuntimeError("Frozen phase3 input changed: "+rel)
    return p

def blocks():
    db,r,p=energy.records()
    result={}
    for e in sorted({v["element"] for v in r}):
        x=np.load(OUT/"oracle"/f"{e}.npz")
        result[e]=dict(label_ids=x["label_ids"].tolist(),
            jacobian=x["design_jacobian"][x["design_metric_eligible"]],
            design_y=x["design_y"],design_eligible=x["design_metric_eligible"],
            eval_y=x["eval_y"],eval_eligible=x["eval_metric_eligible"])
    return db,r,result

def forward(db,recs,prediction,inputs,gradient=False,allow_failures=False):
    e=recs[0]["element"]
    active,original=element_database(reconstruct(db,recs,prediction),e)
    position={int(j):i for i,j in enumerate(original)}
    B=None
    if gradient:
        B=np.zeros((active.nCS,len(recs)))
        for j,r in enumerate(recs):B[position[r["index"]],j]=-r["delta_q"]
    layer=ImplicitDefectLayer(active,(e,),B)
    ys=[];js=[];valid=[];diags=[]
    for T,C in inputs:
        try:
            p=layer.predict(layer.parameters(T=T,Nd=9.1e15,doses=(C,),pO2=1e-5))
            ys.append(p.values[0])
            if gradient:js.append(p.jacobian[0,2:2+len(recs)])
            valid.append(p.diagnostics["single_electron_Hall_valid"] and
                         p.diagnostics["above_carrier_detection_floor"])
            diags.append(p.diagnostics)
        except (ImplicitDomainError,ValueError,np.linalg.LinAlgError) as exc:
            if not allow_failures:raise
            ys.append(np.nan)
            if gradient:js.append(np.full(len(recs),np.nan))
            valid.append(False);diags.append(dict(failure=repr(exc),Tf_K=T,dose_cm3=C))
    return np.asarray(ys),np.asarray(js),np.asarray(valid),diags

def initialize():
    if (OUT/"protocol.json").exists():frozen();return
    OUT.mkdir(parents=True,exist_ok=True)
    db,r,parents=energy.records()
    coverage=[]
    for e in sorted({v["element"] for v in r}):
        rr=[v for v in r if v["element"]==e]
        target=np.array([v["crossing_eV"] for v in rr])
        a,J,va,da=forward(db,rr,target,DESIGN,True,True)
        b,_,vb,dbb=forward(db,rr,target,EVAL,allow_failures=True)
        p=OUT/"oracle"/f"{e}.npz";p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p,label_ids=np.array([v["label_id"] for v in rr]),
            design_y=a,design_jacobian=J,design_metric_eligible=va,
            eval_y=b,eval_metric_eligible=vb)
        energy.write(p.with_suffix(".json"),dict(element=e,
            design_diagnostics=da,evaluation_diagnostics=dbb,
            data_grade="virtual full-published-energy forward references; not new material data"))
        active,indices=element_database(db,e)
        coverage.append(dict(element=e,relative_labels=len(rr),physical_states=active.nCS,
            physical_parents=active.nD,design_eligible=int(va.sum()),design_total=len(va),
            eval_eligible=int(vb.sum()),eval_total=len(vb),
            jacobian_rank=int(np.linalg.matrix_rank(J[va],tol=1e-8)) if va.any() else 0))
        print("oracle",e,coverage[-1],flush=True)
    energy.csvwrite(OUT/"oracle_coverage.csv",coverage)
    proto=dict(version=1,created_utc=energy.now(),
        direction="learn shared energetic relationships through a physical Sandbox for doped-carrier prediction",
        task="conditional completion of mono-single nonreference charge energies, with exact supplied parent references and catalogue",
        energy_data="same269labels/15elements from the frozen phase2 input",
        primary_comparison=["kernel_energy","kernel_diagonal_l1","kernel_full_l1"],
        auxiliary="strength.1/10 sensitivity, Ridge, global mean and charge-pair mean; all retained",
        input_features="same18 features as phase2, training-only weighted scaling",
        known_reference_energy_feature=False,
        main_objective="r^T D r + lambda r^T J^T W J r + alpha||beta||^2 in RBF RKHS features",
        weighting="D equal element/parent/charge, total1; property equal training element and eligible design input, scale.3dex",
        matrix="full blocks include energy-error correlations; diagonal ablation discards only offdiagonal metric entries",
        kernel_spectrum="retain eigenvalues >1e-12*max(1,largest); check equivalence to weighted sklearn KRR",
        exact_absence="exclude all charge states containing other impurities, compact parent indices; no held-element energies enter another training-element metric",
        virtual_references="full known energy library, explicit process scenarios; not experimental property labels",
        design_inputs=DESIGN,outer_evaluation_inputs=EVAL,
        Nd_cm3=9.1e15,pO2_atm=1e-5,observation_T_K=300.,
        metric_eligibility="reference regular root, single-electron Hall, nH>=1e13; no selection on candidate errors",
        no_reference_states="element retains energy supervision, contributes zero to property metric; not scored in property macro if zero eligible reference states",
        nested_split="outer entire element; inner entire remaining element; recompute scaling/kernel/metric membership per training set",
        primary_hyper_selection="inner eligible-element-equal full nonlinear forward lognH MAE; any numerical refusal at an eligible target -> infinite element score",
        extra_selection_control="same kernel_energy grid selected by inner energy MAE, separating training loss from hyperparameter criterion",
        tie_rule="rounded1e-10 score, then stronger alpha and smoother longer kernel",
        model_grid=SPECS,
        compute_budget="primary three methods each exactly6 candidate configurations and same inner/outer physical evaluation conditions; all per-fit and forwarding costs retained; simple controls use <=6",
        sensitivity_budget="auxiliary strengths are separately reported fixed-strength experiments, not free extra tuning folded into the primary comparison",
        cpu_workers=4,GPU="not required for <269-row kernels; GPU1 authorization remains",
        real_property_followup="same29/5 real source split and0/1/2 anchors; refit process only on training sources, keep all pO2 scenarios explicit",
        energy_cost="same masked energy states as phase2; separately report active physical catalogue and provided source library; no new DFT jobs",
        interpretation="optimizer uniqueness differs from physical identifiability; local loss does not certify nonlinear/material error",
        grade="retrospective development; the additional virtual evaluation conditions are not independent material validation",
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
            (Path(__file__),ROOT/"src/sandbox/property_metric_learning.py",
             ROOT/"src/sandbox/energy_learning.py",ROOT/"src/sandbox/implicit_defects.py",
             ROOT/"scripts/trainable_sandbox_20261017/csv_io_v2.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
            [ROOT/"results/phase0/kroger_ga2o3_092724_numeric.npz"]+
            sorted((OUT/"oracle").glob("*.npz"))})
    energy.write(OUT/"protocol.json",proto)

def numerical_gates():
    frozen()
    db,r,metric=blocks();tr=[x for x in r if x["element"]!="Si"]
    test=[x for x in r if x["element"]=="Si"]
    m=PropertyMetricKernel(length=.5,alpha=1e-3).fit(tr)
    w=group_weights(tr)
    X=(features(tr)-m.center)/m.scale
    Z=(features(test)-m.center)/m.scale
    y=np.array([x["crossing_eV"] for x in tr])-m.mean
    ref=KernelRidge(alpha=m.alpha*len(tr),kernel="rbf",gamma=m.gamma).fit(X,y,sample_weight=w)
    krr_error=float(abs(m.predict(test)-(ref.predict(Z)+m.mean)).max())
    zero=PropertyMetricKernel(length=.5,alpha=1e-3,metric="full",strength=0.).fit(tr,metric)
    zero_error=float(abs(m.predict(test)-zero.predict(test)).max())
    full=PropertyMetricKernel(length=.5,alpha=1e-3,metric="full",strength=1.).fit(tr,metric)
    Phi=rbf_kernel(full.X,full.X,gamma=full.gamma)@full.projector
    sw=np.sqrt(w/w.sum())
    A=[Phi*sw[:,None],np.sqrt(full.alpha)*np.eye(Phi.shape[1])]
    b=[y*sw,np.zeros(Phi.shape[1])]
    for e in full.training_elements:
        ids=np.flatnonzero([v["element"]==e for v in tr]);J=metric[e]["jacobian"]
        if J.size:
            block=np.zeros((len(J),len(tr)))
            block[:,ids]=J/(np.sqrt(len(J)*len(full.training_elements))*.3)
            A.append(block@Phi);b.append(block@y)
    beta=np.linalg.lstsq(np.vstack(A),np.concatenate(b),rcond=None)[0]
    lstsq_error=float(abs(beta-full.beta).max())
    isolation=[];gradient=[]
    for e in ("Si","Sn","Fe","H"):
        rr=[v for v in r if v["element"]==e];truth=np.array([v["crossing_eV"] for v in rr])
        y0,J,valid,dg=forward(db,rr,truth,[(1325.,1e19)],True)
        raw=ImplicitDefectLayer(db,(e,))
        original=raw.predict(raw.parameters(T=1325.,Nd=9.1e15,doses=(1e19,)))
        isolation.append(dict(element=e,difference=float(abs(y0[0]-original.values[0]))))
        j=int(np.argmax(abs(J[0])));h=1e-5;a=truth.copy();b=truth.copy();a[j]+=h;b[j]-=h
        ap=forward(db,rr,a,[(1325.,1e19)])[0][0]
        bp=forward(db,rr,b,[(1325.,1e19)])[0][0]
        fd=(ap-bp)/(2*h);err=abs(fd-J[0,j])/max(1.,abs(fd),abs(J[0,j]))
        gradient.append(dict(element=e,label=rr[j]["label_id"],analytic=float(J[0,j]),
            central_difference=float(fd),scaled_error=float(err)))
        active,keep=element_database(db,e)
        altered=db.dHo.copy();altered[np.setdiff1d(np.arange(db.nCS),keep)]+=37.
        check,_=element_database(replace(db,dHo=altered),e)
        assert np.array_equal(check.dHo,active.dHo)
    audit=dict(weighted_sklearn_KRR_prediction_error_eV=krr_error,
        zero_strength_control_error_eV=zero_error,
        independent_augmented_lstsq_beta_error=lstsq_error,
        full_metric_normal_residual=full.diagnostics["normal_equation_residual"],
        exact_absence_forward_comparisons=isolation,gradient_probes=gradient,
        excluded_energy_invariance=True)
    audit["passed"]=bool(krr_error<1e-6 and zero_error<1e-10 and lstsq_error<1e-7
        and max(x["difference"] for x in isolation)<2e-6
        and max(x["scaled_error"] for x in gradient)<3e-5
        and full.diagnostics["normal_equation_residual"]<1e-10)
    energy.write(OUT/"numerical_gates.json",audit)
    print(json.dumps(audit,indent=2),flush=True)
    assert audit["passed"]

def learner(name,cfg,train,metric):
    if name.startswith("kernel_"):
        return PropertyMetricKernel(**cfg).fit(train,metric)
    return ChargeEnergyLearner(name,cfg).fit(train)

def evaluate_model(model,test,db,oracle,inputs,which):
    pred=model.predict(test)
    energy_error=macro_error(test,pred)
    reference=oracle[which+"_y"];eligible=oracle[which+"_eligible"]
    try:
        values,_,physical,dg=forward(db,test,pred,inputs,allow_failures=True)
        failure=" | ".join(f"{j}:{v['failure']}" for j,v in enumerate(dg) if "failure" in v)
    except (ImplicitDomainError,ValueError,np.linalg.LinAlgError) as e:
        values=np.full(len(inputs),np.nan);physical=np.zeros(len(inputs),bool);failure=repr(e)
    prop=(float(np.mean(abs(values[eligible]-reference[eligible])))
          if eligible.any() and np.isfinite(values[eligible]).all() else
          (np.inf if eligible.any() else None))
    return dict(energy_MAE=energy_error,property_MAE=prop,values=values,
        physical=physical,reference=reference,eligible=eligible,failure=failure,predicted_crossings=pred)

def outer(held):
    frozen()
    assert json.loads((OUT/"numerical_gates.json").read_text())["passed"]
    dest=OUT/"outer"/f"{held}.json"
    if dest.exists():return held
    db,records,metric=blocks()
    elements=sorted({v["element"] for v in records});training=[e for e in elements if e!=held]
    tr=[v for v in records if v["element"]!=held]
    te=[v for v in records if v["element"]==held]
    selections=[];epreds=[];ppreds=[];started=time.perf_counter()
    for name,configs in SPECS.items():
        scored=[]
        for cfg in configs:
            candidates=[];fit_seconds=0.;forward_seconds=0.
            for inner in training:
                a=[v for v in tr if v["element"]!=inner]
                b=[v for v in tr if v["element"]==inner]
                t=time.perf_counter();m=learner(name,cfg,a,metric);fit_seconds+=time.perf_counter()-t
                assert held not in m.training_elements and inner not in m.training_elements
                t=time.perf_counter();res=evaluate_model(m,b,db,metric[inner],DESIGN,"design")
                forward_seconds+=time.perf_counter()-t
                candidates.append(dict(element=inner,energy_MAE=res["energy_MAE"],
                    property_MAE=res["property_MAE"],failure=res["failure"],
                    eligible=int(res["eligible"].sum()),training_elements=m.training_elements))
            active=[v["property_MAE"] for v in candidates if v["property_MAE"] is not None]
            scored.append(dict(config=cfg,inner_energy_MAE=float(np.mean([v["energy_MAE"] for v in candidates])),
                inner_property_MAE=float(np.mean(active)) if active else np.inf,
                inner_details=candidates,fit_seconds=fit_seconds,forward_seconds=forward_seconds))
        selected=[("property",min(scored,key=lambda s:(
            round(s["inner_property_MAE"],10),-s["config"].get("alpha",0.),-s["config"].get("length",0.))))]
        if name=="kernel_energy":
            selected.append(("energy",min(scored,key=lambda s:(
                round(s["inner_energy_MAE"],10),-s["config"]["alpha"],-s["config"]["length"]))))
        for selection,best in selected:
            method=name if selection=="property" else name+"_select_energy"
            model=learner(name,best["config"],tr,metric)
            result=evaluate_model(model,te,db,metric[held],EVAL,"eval")
            path=OUT/"models"/f"{held}__{method}.pkl";path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("wb") as f:pickle.dump(model,f)
            selections.append(dict(method=method,selector=selection,selected=best,
                all_candidates=scored,training_elements=model.training_elements,
                energy_MAE=result["energy_MAE"],property_MAE=result["property_MAE"],
                failure=result["failure"],diagnostics=getattr(model,"diagnostics",{})))
            for rec,pred in zip(te,result["predicted_crossings"]):
                epreds.append(dict(**rec,method=method,predicted_crossing_eV=float(pred),
                    training_elements="|".join(model.training_elements)))
            for j,(T,C) in enumerate(EVAL):
                ppreds.append(dict(element=held,method=method,Tf_K=T,dose_cm3=C,
                    predicted_log_nH=result["values"][j],reference_log_nH=result["reference"][j],
                    reference_eligible=bool(result["eligible"][j]),
                    predicted_physical_valid=bool(result["physical"][j]),failure=result["failure"]))
            print("metric outer",held,method,"energy",result["energy_MAE"],
                  "property",result["property_MAE"],flush=True)
    energy.csvwrite(OUT/"energy_outer"/f"{held}.csv",epreds)
    energy.csvwrite(OUT/"property_outer"/f"{held}.csv",ppreds)
    energy.write(dest,dict(held=held,selections=selections,seconds=time.perf_counter()-started))
    return held

def summarize():
    ep=pd.concat([read_result_csv(p) for p in sorted((OUT/"energy_outer").glob("*.csv"))],ignore_index=True)
    pp=pd.concat([read_result_csv(p) for p in sorted((OUT/"property_outer").glob("*.csv"))],ignore_index=True)
    ep.to_csv(OUT/"energy_predictions.csv",index=False);pp.to_csv(OUT/"virtual_property_predictions.csv",index=False)
    rows=[]
    for (m,e),g in ep.groupby(["method","element"]):
        q=pp[pp.method.eq(m)&pp.element.eq(e)&pp.reference_eligible]
        complete=bool(q.predicted_log_nH.notna().all())
        rows.append(dict(method=m,element=e,energy_MAE=macro_error(g.to_dict("records"),g.predicted_crossing_eV),
            eligible=len(q),complete=complete,
            property_MAE=float(abs(q.predicted_log_nH-q.reference_log_nH).mean()) if len(q) and complete else None,
            refusal=int(q.predicted_log_nH.isna().sum())))
    by=pd.DataFrame(rows);by.to_csv(OUT/"by_element.csv",index=False)
    s=by.groupby("method").agg(elements=("element","nunique"),energy_MAE=("energy_MAE","mean"),
        eligible_elements=("property_MAE","count"),virtual_cases=("eligible","sum"),
        refusals=("refusal","sum"),property_MAE=("property_MAE","mean"),worst_property=("property_MAE","max")).reset_index()
    s.to_csv(OUT/"summary.csv",index=False);print(s.to_string(index=False),flush=True)

def run(workers):
    frozen()
    db,r,p=energy.records();helds=sorted({v["element"] for v in r})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for held in pool.map(outer,helds):print("completed element",held,flush=True)
    summarize()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--initialize",action="store_true")
    ap.add_argument("--gates",action="store_true")
    ap.add_argument("--train",action="store_true")
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--summarize",action="store_true")
    a=ap.parse_args()
    if a.initialize:initialize()
    if a.gates:numerical_gates()
    if a.train:run(a.workers)
    if a.summarize:summarize()
