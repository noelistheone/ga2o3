"""Known-reference feature audit and same-budget nested ML->Sandbox benchmark."""
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os,sys,json,pickle,time,argparse
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.reference_features import ReferenceFeatureBank,ReferenceEnergyLearner,KINDS
from sandbox.energy_learning import macro_error
from sandbox.property_metric_learning import PropertyMetricKernel,element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase3_metric as metric

OUT=energy.BASE/"phase5_reference_features"
GRIDS={
    "rbf":[dict(alpha=a,length=l) for a,l in product((1e-4,.01,1.),(.5,2.))],
    "ridge":[dict(alpha=a,length=1.) for a in (1e-4,.01,1.,10.,100.,1e4)]}
SELECTORS=("energy","property")
METHODS=tuple(f"{learner}_{kind}_{selector}" for learner in GRIDS for kind in KINDS for selector in SELECTORS)

def data():
    db,r,oracle=metric.blocks()
    return db,r,oracle,ReferenceFeatureBank.from_database(db)

def initialize():
    metric.frozen()
    db,r,oracle,bank=data()
    path=OUT/"protocol.json"
    payload=dict(version=1,created_utc=energy.now(),
        direction="improve ML+Sandbox doped-property prediction using known reference inputs, not mechanism-only diagnosis",
        primary=["rbf_base_energy","rbf_closed_energy"],
        auxiliary="atomic-only/raw-reference ablations, Ridge controls and coverage-first property selectors all reported separately",
        feature_kinds=dict(base="same18features",atomic="append (I-P_[1,dmGa,dmO]) A_ref",
            closed="append (I-P_[1,dmGa,dmO,qref]) A_ref",
            raw="append supplied A_ref, fixed-reference-frame auxiliary, no invariance claim"),
        projection="per-element catalogue, all supplied one-reference-per-mono-single parent; SVD relative cutoff1e-12",
        interpretation="balanced linear combinations of known energies, not new local geometries or elementary reaction barriers",
        supplied_inputs="all parent reference covariates already in phase2 budget; no additional nonreference energies",
        electron_reference="all crossing targets use the same zero-K VBM; closed feature invariance does not itself align arbitrary target labels",
        catalogue_dependence="projection changes if catalogue changes; supplied catalogue held fixed in this conditional task",
        training_features="target-element reference context used only at inference, not training scaling or learner parameters",
        training_bank="ReferenceFeatureBank passed explicitly, never stored in trained model",
        grid=GRIDS,selectors=SELECTORS,methods=METHODS,
        outer="leave each of15 elements out",inner="leave each remaining14 elements out",
        primary_selection="element-equal parent-equal secant energy MAE; deterministic ties prefer stronger penalty/larger length",
        auxiliary_property_selection="lexicographically minimize element-equal unusable-reference fraction then all-numeric nonlinear MAE; numeric refusal -> infinite MAE; not old phase3 criterion",
        kernel_bandwidth="gamma=1/(2*18*length^2) for base and augmented inputs",
        weighting="same equal-element/parent/charge normalized sum1; energy scale1eV",
        property_design=metric.DESIGN,property_evaluation=metric.EVAL,
        virtual_reference="previous known-energy model outputs; not material measurements",
        property_scope="report physical validity separately from mathematical logn/rH extensions",
        real_scope="same29records/5DOI sources/0-1-2anchor policy/common18; source-excluded process fit",
        cost="six candidates per learner/representation; two selectors reuse the same inner fits; actual fit and root cost retained",
        raw_reference_boundary="given in budget, not hidden-label leakage, but dependent on declared input energy frame",
        validation_grade="retrospective all sources/elements previously exposed; not blind or preregistered",
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            Path(__file__),ROOT/"src/sandbox/reference_features.py",
            ROOT/"scripts/trainable_sandbox_20261017/csv_io_v2.py")},
        parent_protocol_sha256=energy.sha(metric.OUT/"protocol.json"),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (
            energy.BASE/"dataset_v1.csv",energy.BASE/"episodes_v1.csv",
            ROOT/"results/phase0/kroger_ga2o3_092724_numeric.npz")})
    if path.exists():
        old=json.loads(path.read_text())
        assert old["code_hashes"]==payload["code_hashes"] and old["input_hashes"]==payload["input_hashes"]
    else:energy.write(path,payload)
    return bank

def frozen():
    p=json.loads((OUT/"protocol.json").read_text())
    for key in ("code_hashes","input_hashes"):
        for name,expected in p[key].items():
            if energy.sha(ROOT/name)!=expected:raise RuntimeError("Frozen phase5 input/code changed: "+name)
    metric.frozen()
    return p

def gates():
    initialize();db,r,oracle,bank=data()
    rng=np.random.default_rng(5117)
    delta=rng.normal(0,3.,len(db.elements))
    shifted=replace(db,dHo=db.dHo+db.dm@delta)
    atomic_bank=ReferenceFeatureBank.from_database(shifted)
    electronic_bank=ReferenceFeatureBank.from_database(replace(db,dHo=db.dHo+.37*db.charge))
    tests={}
    for kind in ("atomic","closed"):
        tests[kind+"_atomic_reference_error"]=float(abs(
            bank.descriptor(r,kind)-atomic_bank.descriptor(r,kind)).max())
    tests["closed_electronic_reference_error"]=float(abs(
        bank.descriptor(r,"closed")-electronic_bank.descriptor(r,"closed")).max())
    hidden=[v["index"] for v in r];a=db.dHo.copy();a[hidden]=np.nan
    erased=ReferenceFeatureBank.from_database(replace(db,dHo=a))
    stripped=[{k:v for k,v in row.items() if k not in
               ("crossing_eV","target_energy_eV","reference_energy_eV")} for row in r]
    tests["hidden_energy_erasure_feature_error"]=max(float(abs(
        bank.descriptor(r,k)-erased.descriptor(stripped,k)).max()) for k in KINDS)
    cat=bank.catalogue.copy();rng.shuffle(cat)
    permuted=ReferenceFeatureBank(cat)
    tests["permutation_error"]=max(float(abs(
        bank.descriptor(r,k)-permuted.descriptor(r,k)).max()) for k in KINDS)
    tr=[v for v in r if v["element"]!="Si"];te=[v for v in r if v["element"]=="Si"]
    new=ReferenceEnergyLearner("rbf","base",alpha=.01,length=.5).fit(tr,bank)
    old=PropertyMetricKernel(alpha=.01,length=.5).fit(tr)
    tests["base_kernel_equivalence_eV"]=float(abs(new.predict(te,bank)-old.predict(te)).max())
    # Perturb a supplied Si reference. No descriptor of another element may change.
    si=max((v for v in cat if v["element"]=="Si"),key=lambda v:abs(bank.by_parent[v["parent"]]["closed"]))
    altered=[dict(v,reference_eV=v["reference_eV"]+(.73 if v["parent"]==si["parent"] else 0.)) for v in cat]
    altered=ReferenceFeatureBank(altered)
    tests["other_element_context_error"]=float(abs(bank.descriptor(tr,"closed")-altered.descriptor(tr,"closed")).max())
    tests["known_Si_reference_feature_response"]=float(abs(bank.descriptor(te,"closed")-altered.descriptor(te,"closed")).max())
    flow=[]
    for e in ("Si","Sn"):
        partial,_=element_database(db,e)
        cols=partial.col(e)
        shifted=replace(partial,dHo=partial.dHo+.23*partial.dm[:,cols])
        before=ImplicitDefectLayer(partial,(e,))
        after=ImplicitDefectLayer(shifted,(e,))
        v=before.predict(before.parameters(T=1325.,Nd=9.1e15,doses=(1e18,),pO2=1e-5))
        w=after.predict(after.parameters(T=1325.,Nd=9.1e15,doses=(1e18,),pO2=1e-5))
        flow.append(dict(element=e,property_max_error_dex=float(abs(v.values-w.values).max()),
            solved_dopant_mu_shift_error_eV=float(abs(w.state[1]-v.state[1]-.23))))
    tests["dopant_reference_full_chain_error"]=max(v["property_max_error_dex"] for v in flow)
    tests["dopant_reference_mu_shift_error"]=max(v["solved_dopant_mu_shift_error_eV"] for v in flow)
    finite=[v for k,v in tests.items() if k!="known_Si_reference_feature_response"]
    assert max(finite)<5e-8 and tests["known_Si_reference_feature_response"]>1e-4
    raw_response=float(abs(bank.descriptor(r,"raw")-atomic_bank.descriptor(r,"raw")).max())
    assert raw_response>.1
    energy.write(OUT/"feature_gates.json",dict(passed=True,tests=tests,
        raw_control_atomic_reference_response_eV=raw_response,
        full_chain_reference_tests=flow,bank_diagnostics=bank.diagnostics,
        scope="algebra/input/solver invariance checks, not material validation or correction-label certification"))
    energy.csvwrite(OUT/"reference_catalogue_private.csv",bank.catalogue)
    rows=[]
    groups=defaultdict(list)
    for rec,x in zip(r,bank.descriptor(r,"base")):groups[tuple(x)].append(rec)
    for recs in groups.values():
        if len(recs)<2:continue
        row=dict(element=recs[0]["element"],labels=len(recs),
            label_ids="|".join(v["label_id"] for v in recs),
            original_label_span_eV=float(np.ptp([v["crossing_eV"] for v in recs])))
        for kind in ("atomic","closed","raw"):
            values=bank.descriptor(recs,kind)[:,-1]
            row[kind+"_distinct_rounded_1e9"]=len(set(np.round(values,9)))
            row[kind+"_feature_span_eV"]=float(np.ptp(values))
        rows.append(row)
    energy.csvwrite(OUT/"feature_collision_audit_private.csv",rows)
    print(json.dumps(dict(tests=tests,known_reference_parents=len(bank.catalogue),
                          original_collision_groups=len(rows)),indent=2),flush=True)

def evaluate(model,test,bank,db,oracle,inputs,tag):
    pred=model.predict(test,bank)
    n=int(oracle[tag+"_eligible"].sum())
    result=dict(predicted_crossings=pred,energy_MAE=macro_error(test,pred),
                eligible=n,property_MAE=None,unusable_fraction=None,numeric_refusals=0)
    if not n:return result
    values,J,physical,diagnostics=metric.forward(db,test,pred,inputs,gradient=False,allow_failures=True)
    mask=oracle[tag+"_eligible"];truth=oracle[tag+"_y"]
    good=np.isfinite(values[mask]);refusals=int((~good).sum())
    result.update(values=values,physical=physical,
        property_MAE=float(abs(values[mask]-truth[mask]).mean()) if not refusals else None,
        numeric_refusals=refusals,
        unusable_fraction=float(1-(physical[mask]&good).mean()),diagnostics=diagnostics)
    return result

def candidate(task):
    held,learner,kind,k=task
    frozen();db,r,oracle,bank=data();config=GRIDS[learner][k]
    tr=[v for v in r if v["element"]!=held]
    rows=[]
    for inner in sorted({v["element"] for v in tr}):
        path=OUT/"inner"/held/learner/kind/f"config{k}"/f"{inner}.json"
        if path.exists():row=json.loads(path.read_text())
        else:
            a=[v for v in tr if v["element"]!=inner];b=[v for v in tr if v["element"]==inner]
            start=time.perf_counter()
            model=ReferenceEnergyLearner(learner,kind,**config).fit(a,bank)
            seconds=time.perf_counter()-start
            assert held not in model.training_elements and inner not in model.training_elements
            start=time.perf_counter()
            result=evaluate(model,b,bank,db,oracle[inner],metric.DESIGN,"design")
            row=dict(held=held,inner=inner,learner=learner,kind=kind,config=config,
                training_elements=model.training_elements,training_label_ids=model.training_label_ids,
                energy_MAE=result["energy_MAE"],property_MAE=result["property_MAE"],
                eligible=result["eligible"],unusable_fraction=result["unusable_fraction"],
                numeric_refusals=result["numeric_refusals"],fit_seconds=seconds,
                forward_seconds=time.perf_counter()-start)
            energy.write(path,row)
        rows.append(row)
    active=[v for v in rows if v["eligible"]>0]
    refusal=any(v["numeric_refusals"] for v in active)
    result=dict(held=held,learner=learner,kind=kind,index=k,config=config,
        inner_energy_MAE=float(np.mean([v["energy_MAE"] for v in rows])),
        inner_property_MAE=None if refusal else float(np.mean([v["property_MAE"] for v in active])),
        inner_unusable_fraction=float(np.mean([v["unusable_fraction"] for v in active])),
        any_numeric_refusal=bool(refusal),fit_seconds=float(sum(v["fit_seconds"] for v in rows)),
        forward_seconds=float(sum(v["forward_seconds"] for v in rows)),details=rows)
    energy.write(OUT/"candidates"/f"{held}__{learner}__{kind}__{k}.json",result)
    print("reference candidate",held,learner,kind,k,"energy",result["inner_energy_MAE"],flush=True)
    return held,learner,kind,k

def outer(task):
    held,learner,kind=task
    path=OUT/"outer"/f"{held}__{learner}__{kind}.json"
    if path.exists():return
    frozen();db,r,oracle,bank=data()
    tr=[v for v in r if v["element"]!=held];te=[v for v in r if v["element"]==held]
    candidates=[json.loads((OUT/"candidates"/f"{held}__{learner}__{kind}__{k}.json").read_text())
                for k in range(len(GRIDS[learner]))]
    erows=[];prows=[];selections=[]
    for selector in SELECTORS:
        def score(c):
            complexity=(-c["config"]["alpha"],-c["config"]["length"])
            if selector=="energy":return (round(c["inner_energy_MAE"],10),*complexity)
            prop=np.inf if c["any_numeric_refusal"] else c["inner_property_MAE"]
            return (round(c["inner_unusable_fraction"],12),round(prop,10),
                    round(c["inner_energy_MAE"],10),*complexity)
        selected=min(candidates,key=score)
        model=ReferenceEnergyLearner(learner,kind,**selected["config"]).fit(tr,bank)
        method=f"{learner}_{kind}_{selector}"
        modelpath=OUT/"models"/f"{held}__{method}.pkl";modelpath.parent.mkdir(parents=True,exist_ok=True)
        with modelpath.open("wb") as f:pickle.dump(model,f)
        result=evaluate(model,te,bank,db,oracle[held],metric.EVAL,"eval")
        for rec,pred in zip(te,result["predicted_crossings"]):
            erows.append(dict(**rec,method=method,predicted_crossing_eV=float(pred),
                             training_elements="|".join(model.training_elements)))
        for j,(T,C) in enumerate(metric.EVAL):
            prows.append(dict(element=held,method=method,Tf_K=T,dose_cm3=C,
                reference_log_nH=oracle[held]["eval_y"][j],
                predicted_log_nH=result.get("values",np.full(len(metric.EVAL),np.nan))[j],
                reference_eligible=bool(oracle[held]["eval_eligible"][j]),
                predicted_physical_valid=bool(result.get("physical",np.zeros(len(metric.EVAL),bool))[j])))
        selections.append(dict(method=method,selected=selected,
            training_elements=model.training_elements,energy_MAE=result["energy_MAE"],
            property_MAE=result["property_MAE"],numeric_refusals=result["numeric_refusals"],
            unusable_fraction=result["unusable_fraction"]))
    energy.csvwrite(OUT/"energy_outer"/f"{held}__{learner}__{kind}.csv",erows)
    energy.csvwrite(OUT/"property_outer"/f"{held}__{learner}__{kind}.csv",prows)
    energy.write(path,dict(held=held,learner=learner,kind=kind,selections=selections))
    print("reference outer",held,learner,kind,
        [(v["method"],v["energy_MAE"],v["property_MAE"]) for v in selections],flush=True)

def run(workers):
    frozen()
    assert json.loads((OUT/"feature_gates.json").read_text())["passed"]
    _,r,_,_=data();elements=sorted({v["element"] for v in r})
    elements=["Si","Sn"]+[e for e in elements if e not in ("Si","Sn")]
    tasks=[(e,l,k,i) for e,l,k in product(elements,GRIDS,KINDS) for i in range(len(GRIDS[l]))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(candidate,tasks):pass
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(outer,product(elements,GRIDS,KINDS)):pass
    import pandas as pd
    energy_predictions=pd.concat([read_result_csv(p) for p in sorted((OUT/"energy_outer").glob("*.csv"))],ignore_index=True)
    energy_predictions.to_csv(OUT/"energy_predictions.csv",index=False)
    virtual=pd.concat([read_result_csv(p) for p in sorted((OUT/"property_outer").glob("*.csv"))],ignore_index=True)
    virtual.to_csv(OUT/"virtual_predictions.csv",index=False)
    rows=[]
    for method,g in energy_predictions.groupby("method"):
        rr=g.to_dict("records")
        v=virtual[virtual.method.eq(method)&virtual.reference_eligible]
        by=(v.assign(error=abs(v.predicted_log_nH-v.reference_log_nH)).groupby("element").error.mean())
        rows.append(dict(method=method,labels=len(g),elements=int(g.element.nunique()),
            energy_macro_MAE_eV=macro_error(rr,g.predicted_crossing_eV.to_numpy(float)),
            reference_eligible=len(v),numeric=int(v.predicted_log_nH.notna().sum()),
            physical=int(v.predicted_physical_valid.sum()),
            mathematical_property_macro_MAE_dex=float(by.mean()),
            any_numeric_refusal=bool(v.predicted_log_nH.isna().any())))
    energy.csvwrite(OUT/"summary.csv",rows)
    print("Completed reference-feature nested comparison",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--initialize",action="store_true")
    ap.add_argument("--gates",action="store_true");ap.add_argument("--train",action="store_true")
    ap.add_argument("--workers",type=int,default=4);a=ap.parse_args()
    if a.initialize:initialize()
    if a.gates:gates()
    if a.train:run(a.workers)
