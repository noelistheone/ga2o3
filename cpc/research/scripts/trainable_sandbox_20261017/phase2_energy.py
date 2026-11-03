"""Freeze, fit and evaluate conditional charge-energy ML -> Sandbox studies."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
import sys, os, json, csv, hashlib, argparse, pickle, time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.kroger_db import load, NPZ
from sandbox.energy_learning import (
    FEATURE_NAMES, relative_charge_records, features, macro_error,
    ChargeEnergyLearner, reconstruct, stable_crossings)

BASE=ROOT/"results/trainable_sandbox_20261017"
OUT=BASE/"phase2_energy"
GRID={
    "global_mean":[{}],
    "charge_pair_mean":[{}],
    "ridge":[dict(alpha=a) for a in (.01,1.,100.,10000.)],
    "rbf":[dict(alpha=a,length=l) for a,l in product((.01,.1,1.),(.5,1.,2.,4.))],
    "extra_trees":[dict(max_depth=d,min_samples_leaf=l)
                   for d,l in product((2,5,None),(2,5))],
}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):
        return float(x) if np.isfinite(x) else None
    return x
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def records():
    db=load();sites=np.load(NPZ)["cs_num_each_site"]
    r,p=relative_charge_records(db,sites)
    return db,r,p
def frozen():
    path=OUT/"protocol.json"
    if not path.exists():raise RuntimeError("Run --initialize before fitting")
    p=json.loads(path.read_text())
    for k,v in p["input_hashes"].items():
        if sha(ROOT/k)!=v:raise RuntimeError("Frozen input changed: "+k)
    for k,v in p["code_hashes"].items():
        if sha(ROOT/k)!=v:raise RuntimeError("Frozen code changed; version before rerunning: "+k)
    return p
def initialize():
    if (OUT/"protocol.json").exists():
        frozen();return
    OUT.mkdir(parents=True,exist_ok=True)
    db,r,parents=records()
    # The audit precedes all fits. Source numeric arrays stay in a private result
    # directory and must not be bundled into a public code release.
    csvwrite(OUT/"relative_charge_labels_private.csv",r)
    csvwrite(OUT/"parent_catalogue.csv",parents)
    counts=[]
    for e in db.elements[2:]:
        rr=[x for x in r if x["element"]==e];pp=[x for x in parents if x["element"]==e]
        counts.append(dict(element=e,parents=len(pp),states=sum(x["charge_states"] for x in pp),
            labels=len(rr),parents_with_labels=sum(x["hidden_labels"]>0 for x in pp),
            isolated_labels=sum(x["isolated"] for x in rr),
            isolated_parents=sum(x["hidden_labels"]>0 and x["isolated"] for x in pp),
            neutral_references=sum(x["q_reference"]==0 for x in pp)))
    csvwrite(OUT/"coverage.csv",counts)
    collisions=[]
    for name,rr in (("all",r),("isolated",[x for x in r if x["isolated"]])):
        groups=defaultdict(list)
        for rec,xx in zip(rr,features(rr)):groups[tuple(xx)].append(rec)
        for xx,rrr in groups.items():
            if len(rrr)<2:continue
            y=np.asarray([v["crossing_eV"] for v in rrr])
            collisions.append(dict(stratum=name,labels=len(y),
                label_ids="|".join(v["label_id"] for v in rrr),
                parents="|".join(str(v["parent"]) for v in rrr),
                element=rrr[0]["element"],span_eV=float(np.ptp(y)),
                best_unweighted_L1_error_eV=float(np.abs(y-np.median(y)).mean())))
    csvwrite(OUT/"feature_collisions.csv",collisions)
    stable=[]
    for pp in parents:
        ids=np.flatnonzero(db.cs_ID==pp["parent"])
        stable.extend(dict(parent=pp["parent"],element=pp["element"],**v)
                      for v in stable_crossings(db,ids))
    csvwrite(OUT/"stable_zeroK_crossings_private.csv",stable)
    reconstructed=reconstruct(db,r,[x["crossing_eV"] for x in r])
    # Algebraic invariance tests: a common parent intercept shift cancels; a
    # q-dependent shift changes the electronic reference, hence shifts xi.
    original=np.asarray([x["crossing_eV"] for x in r])
    rng=np.random.default_rng(1729)
    shifted=db.dHo.copy()
    for p in np.unique(db.cs_ID):shifted[db.cs_ID==p]+=rng.normal()
    from dataclasses import replace
    ss,sp=relative_charge_records(replace(db,dHo=shifted),np.load(NPZ)["cs_num_each_site"])
    electronic_shift=.37
    ee,ep=relative_charge_records(replace(db,dHo=db.dHo+db.charge*electronic_shift),
                                np.load(NPZ)["cs_num_each_site"])
    audit=dict(reconstruction_max_error_eV=float(abs(reconstructed.dHo-db.dHo).max()),
        parent_intercept_shift_max_error_eV=float(abs(np.array([x["crossing_eV"] for x in ss])-original).max()),
        electronic_reference_shift_error_eV=float(abs(np.array([x["crossing_eV"] for x in ee])-original+electronic_shift).max()),
        parent22_site_change="native parent charge-dependent site occupancy; outside mono-single learner, retained unchanged",
        source="KROGER PCCP2025 DOI10.1039/D4CP04817B; numeric database loaded without new DFT",
        validated_new_DFT_labels=False,
        source_methods_read="cached main text Calculation Methods, formation energy and concentrations; online abstract verified",
        third_party_numeric_release="do not include private derived labels or original arrays in public code release")
    assert max(audit[k] for k in ("reconstruction_max_error_eV","parent_intercept_shift_max_error_eV","electronic_reference_shift_error_eV"))<1e-11
    write(OUT/"label_algebra_audit.json",audit)
    protocol=dict(version=1,created_utc=now(),
        direction="train ML inside the energetic input of a physical Sandbox to predict doped-material properties",
        validation_grade="retrospective development; not blind, independent experiment or preregistered study",
        task="known parent/charge catalogue plus one supplied energy per parent; infer missing charge energies of an element",
        target="xi_q = -(A_q-A_ref)/(q-q_ref); these are secant crossings, not all stable CTLs",
        reconstruction="A_q = A_ref - (q-q_ref)*xi_q",
        reference_selection="minimum absolute known charge, tie prefers negative; not selected by energy",
        energy_reference="same published zero-K VBM convention and finite-size-corrected input database; atom reference cancels, electronic reference does not",
        primary_energy_subset="all mono-single impurity parents, no target-based exclusions",
        secondary_stratum="isolated: abs(dm_Ga)+abs(dm_O)<=1",
        features=list(FEATURE_NAMES),
        feature_exclusions=["A_ref","true hidden A_q","parent ID","measured carrier/mobility"],
        split="outer leave entire dopant element out; inner leave one remaining element out; parent/charge groups never split across elements",
        metric="element-equal then parent-equal MAE in secant eV; raw row MAE and energy reconstruction MAE also retained",
        training_weighting="equal element, equal parent within element, equal nonreference charge within parent",
        scaling="training-only weighted mean/std",
        model_grid=GRID,random_seed=1729,
        tree_estimators=64,
        energy_cost="count unique supplied original charge-state energies; pair labels share reference energies and are not independent DFT jobs",
        property_task=dict(
            dataset="29 records, 5 sources, Si/Sn; same dataset_v1 and episodes_v1",
            masking="for each target element, replace only that element's mono-single nonreference energies; all other database energies and exact reference energies supplied",
            omitted_labels=dict(Si=29,Sn=23),
            retained_labels=dict(Si=844,Sn=850),
            comparison="oracle full energy, global crossing mean, charge-pair mean, nested Ridge/RBF/ExtraTrees",
            source_split="outer leave publication family out; shared Tf and Nd trained only on remaining sources for each reconstructed database",
            parameters="same bounded logTf 600..1950 K and logNd 14..18; two starts 850/1550 K; max_nfev=60 per start; analytic IFT",
            pressure="all 29 missing pO2; fixed declared 1e-5 atm scenario",
            adaptation="0/1/2 same lowest/middle distinct-dose nH anchors; Gaussian output offset only",
            adaptation_selection="inner source exclusion, lambda=sigma^2/tau^2 in [.1,1,10,inf]; no query labels",
            common_query="also report fixed 18 query samples across budgets",
            outcomes=["family-equal nH MAE against measurements","disagreement with oracle full-energy prediction","numeric and physical coverage","worst family"],
            no_claims=["new-element discovery","independent experimental validation","all energies computed by us","complete cost saving for an unknown dopant"]),
        sources=[
            dict(title="KROGER",url="https://arxiv.org/abs/2501.17373",doi="10.1039/D4CP04817B",
                 depth="cached main text selected Calculation Methods; current online abstract"),
            dict(title="Polak2022",url="https://arxiv.org/abs/2203.10349",doi="10.1063/5.0083877",
                 depth="cached main text selected data/features/nested CV sections; current online abstract",
                 boundary="energy/CTL ML has prior art; do not claim its first invention"),
            dict(title="Freysoldt2014",url="https://doi.org/10.1103/RevModPhys.86.253",
                 depth="publisher abstract/metadata this round; not claimed full-text review")],
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                     (NPZ,BASE/"dataset_v1.csv",BASE/"episodes_v1.csv")},
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                     (Path(__file__),ROOT/"src/sandbox/energy_learning.py",
                      ROOT/"src/sandbox/implicit_defects.py")})
    write(OUT/"protocol.json",protocol)
    print(json.dumps(dict(labels=len(r),parents=len(parents),
        elements=len({x["element"] for x in r}),audit=audit),ensure_ascii=False),flush=True)


def nested_energy():
    frozen()
    db,r,pp=records();elements=sorted({v["element"] for v in r})
    allpred=[]
    for held in elements:
        dest=OUT/"energy_outer"/f"{held}.csv"
        if dest.exists():
            with dest.open(newline="") as f:allpred.extend(list(csv.DictReader(f)))
            continue
        train=[v for v in r if v["element"]!=held]
        test=[v for v in r if v["element"]==held]
        predrows=[];selections=[]
        for method,grid in GRID.items():
            scored=[]
            for config in grid:
                errors=[]
                # Baselines with no hyperparameter need no inner refits.
                if len(grid)>1:
                    for inner in sorted({v["element"] for v in train}):
                        tr=[v for v in train if v["element"]!=inner]
                        va=[v for v in train if v["element"]==inner]
                        m=ChargeEnergyLearner(method,config).fit(tr)
                        errors.append(macro_error(va,m.predict(va)))
                scored.append(dict(config=config,inner_mae=float(np.mean(errors)) if errors else None))
            best=min(scored,key=lambda s:s["inner_mae"] if s["inner_mae"] is not None else 0.)
            model=ChargeEnergyLearner(method,best["config"]).fit(train)
            y=model.predict(test)
            modelpath=OUT/"models"/f"{held}__{method}.pkl";modelpath.parent.mkdir(parents=True,exist_ok=True)
            with modelpath.open("wb") as f:pickle.dump(model,f)
            selections.append(dict(held_element=held,method=method,selected=best,
                all_candidates=scored,training_elements=model.training_elements,
                supplied_training_pair_labels=len(train),held_pair_labels=len(test)))
            for rec,pred in zip(test,y):
                predrows.append(dict(**rec,method=method,predicted_crossing_eV=float(pred),
                    predicted_energy_eV=rec["reference_energy_eV"]-rec["delta_q"]*float(pred),
                    training_elements="|".join(model.training_elements)))
            print("energy",held,method,"inner",best["inner_mae"],
                  "outer",macro_error(test,y),flush=True)
        csvwrite(dest,predrows);allpred.extend(predrows)
        write(OUT/"energy_selections"/f"{held}.json",selections)
    csvwrite(OUT/"energy_predictions.csv",allpred)
    # Standard-library CSV ingestion avoids a known local pandas reader crash.
    from csv_io import read_result_csv
    d=read_result_csv(OUT/"energy_predictions.csv")
    rows=[]
    for (method,e),g in d.groupby(["method","element"]):
        for stratum in ("all","isolated","complex"):
            h=g if stratum=="all" else g[g.isolated.eq(stratum=="isolated")]
            if not len(h):continue
            rr=h.to_dict("records");p=h.predicted_crossing_eV.to_numpy(float)
            ae=np.abs(p-h.crossing_eV.to_numpy(float))
            energy=np.abs(h.predicted_energy_eV-h.target_energy_eV)
            rows.append(dict(method=method,element=e,stratum=stratum,labels=len(h),
                parents=int(h.parent.nunique()),parent_equal_MAE_eV=macro_error(rr,p),
                row_MAE_eV=float(ae.mean()),energy_row_MAE_eV=float(energy.mean())))
    csvwrite(OUT/"energy_by_element.csv",rows)
    import pandas as pd
    by=pd.DataFrame(rows)
    summary=by.groupby(["method","stratum"]).agg(
        elements=("element","nunique"),labels=("labels","sum"),
        element_parent_MAE_eV=("parent_equal_MAE_eV","mean"),
        worst_element_MAE_eV=("parent_equal_MAE_eV","max")).reset_index()
    summary.to_csv(OUT/"energy_summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--initialize",action="store_true")
    ap.add_argument("--energy",action="store_true")
    args=ap.parse_args()
    if args.initialize:initialize()
    if args.energy:nested_energy()
