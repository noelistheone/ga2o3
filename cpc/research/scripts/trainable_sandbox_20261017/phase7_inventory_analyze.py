"""Audit the completed frozen inventory trial; no new model selection/refitting."""
from pathlib import Path
import os,sys,json
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase7_inventory_property as study

BASE=energy.BASE;OUT=study.OUT

def main():
    study.protocol.freeze()
    data=read_result_csv(OUT/"predictions.csv");summary=read_result_csv(OUT/"summary.csv")
    assert len(data)==6*29*3 and len(list((OUT/"outer").glob("*.csv")))==30
    fits={p.stem:json.loads(p.read_text()) for p in (OUT/"fits").glob("*.json")}
    assert len(fits)==90
    protected=[]
    for file in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
                 "artifact_manifest_phase34.json","artifact_manifest_phase56.json"):
        d=json.loads((BASE/file).read_text());entries=d["files"] if isinstance(d,dict) else d
        changed=[v["path"] for v in entries if not (ROOT/v["path"]).exists()
                 or energy.sha(ROOT/v["path"])!=v["sha256"]]
        protected.append(dict(manifest=file,checked=len(entries),changed=changed))
    assert all(not v["changed"] for v in protected)
    scopes=0;selection_evaluations=0
    lookup={(f["method"],tuple(f["train_families"])):f for f in fits.values()}
    for held in sorted(study.DATA.family.unique()):
        training=sorted(set(study.DATA.family)-{held})
        for method in study.MODELS:
            f=lookup[(method,tuple(training))];assert held not in f["train_families"]
            rows=data[data.method.eq(method)&data.family.eq(held)]
            assert set(rows.fit_key)=={f["key"]}
            sels=json.loads((OUT/"selections"/f"{study.evaluator.digest(held)}__{method}.json").read_text())
            assert len(sels)==3 and not any(v["selection_failure"] for v in sels)
            for inner in training:
                expected=tuple(x for x in training if x!=inner)
                fi=lookup[(method,expected)]
                assert held not in fi["train_families"] and inner not in fi["train_families"]
                scopes+=1
            selection_evaluations+=sum(len(v["candidates"])*len(training) for v in sels)
    # eta0 reconstruction must replay the old exact-energy control, including
    # the same inner source-selected anchor shrinkage.
    reference=read_result_csv(BASE/"phase56_validation/all_primary_predictions.csv")
    reference=reference[reference.origin.eq("phase2")&reference.method.eq("oracle_full")]
    replay=data[data.method.eq("eq")].merge(reference,on=["sample_id","budget"],suffixes=("_new","_old"))
    replay_error=float(abs(replay.predicted_log_nH_new-replay.predicted_log_nH_old).max())
    assert len(replay)==87 and replay_error<2e-7
    fitrows=[]
    for f in fits.values():
        b=min([t for t in f["trials"] if t.get("loss") is not None],key=lambda t:t["loss"])
        w=np.asarray(f["parameters"]);method=f["method"]
        r=dict(fit_key=f["key"],method=method,train_sources=len(f["train_families"]),
               train_families=";".join(f["train_families"]),loss=b["loss"],seconds=f["seconds"],
               best_success=b["success"],best_nfev=b["nfev"],best_optimality=b["optimality"],
               all_starts_success=all(t["success"] for t in f["trials"]),
               root_failures=len(f["training_state_failures"]),parameters=json.dumps(f["parameters"]))
        r["eta_intercept_at_lower_bound"]=method in ("fill_const","fill_temperature") and abs(w[2]+6)<1e-5
        r["eta_slope_at_upper_bound"]=method=="fill_temperature" and abs(w[3]-4)<1e-5
        fitrows.append(r)
    fr=pd.DataFrame(fitrows);fr.to_csv(OUT/"fit_audit.csv",index=False)
    # Evaluate training Jacobian geometry, without changing any fitted value.
    geometry=[]
    for f in fits.values():
        if len(f["train_families"])!=4:continue
        tr=study.DATA[study.DATA.family.isin(f["train_families"])];J=[];ys=[]
        for row in tr.itertuples():
            p,j,_=study.point(row,np.array(f["parameters"]),f["method"])
            J.append(j[0]);ys.append(p.values[0])
        J=np.asarray(J);norm=np.linalg.norm(J,axis=0)
        singular=np.linalg.svd(J,compute_uv=False)
        normalized=np.linalg.svd(J/np.maximum(norm,1e-300),compute_uv=False)
        geometry.append(dict(fit_key=f["key"],method=f["method"],training_sources=f["train_families"],
            jacobian_singular_values=singular,normalized_column_singular_values=normalized,
            rank=int(np.linalg.matrix_rank(J)),parameters=len(f["parameters"]),
            warning="local prediction sensitivity, not global mechanism or inventory identifiability"))
    energy.write(OUT/"training_geometry.json",geometry)
    by=read_result_csv(OUT/"by_family.csv");paired=[]
    for budget in (0,1,2):
        for scope in ("all_query","common18"):
            h=by[by.budget.eq(budget)&by.scope.eq(scope)].pivot(index="family",columns="method",values="MAE")
            for challenger in study.MODELS:
                if challenger=="eq":continue
                delta=h[challenger]-h["eq"]
                paired.append(dict(challenger=challenger,reference="eq",budget=budget,scope=scope,
                    better_sources=int((delta<-1e-8).sum()),tied_sources=int((abs(delta)<=1e-8).sum()),
                    worse_sources=int((delta>1e-8).sum()),source_equal_difference=float(delta.mean())))
    pd.DataFrame(paired).to_csv(OUT/"paired_comparisons.csv",index=False)
    old=read_result_csv(BASE/"phase56_validation/canonical_comparison.csv")
    names={"eq":"Full-energy equilibrium","fill_endpoint":"Fixed filling endpoint","fill_const":"Constant stock closure",
           "fill_temperature":"Learned temperature-stock closure","temperature_relation":"Temperature relation",
           "temperature_element_relation":"Temperature and element relation"}
    new=summary.copy();new["origin"]="phase7";new["label"]=new.method.map(names)
    pd.concat([old,new],ignore_index=True).to_csv(OUT/"canonical_comparison.csv",index=False)
    audit=dict(passed=True,source_membership_contexts=scopes,nested_candidate_evaluations=selection_evaluations,
        fits=len(fits),trials=sum(len(f["trials"]) for f in fits.values()),
        nonconverged_trials=sum(not t["success"] for f in fits.values() for t in f["trials"]),
        training_state_failures=sum(len(f["training_state_failures"]) for f in fits.values()),
        summed_fit_seconds=sum(f["seconds"] for f in fits.values()),seconds_interpretation="sum per-fit process walltime, not job elapsed",
        rows=len(data),actual_carrier_samples=29,DOI_groups=5,measured_oxygen_pressure_rows=0,
        eq_old_control_replay_max_dex=replay_error,previous_protected=protected,
        scope="physical flag belongs to raw cold root, before empirical anchor offset",
        all_primary_numeric=bool(summary.full_numeric_coverage.all()),
        all_primary_declared_domain=bool(summary.physical.eq(summary.queries).all()),
        minimum_constant_eta=float(1/(1+np.exp(6))),
        boundary_limitation="finite sigmoid intercept bounds exclude exact eta=0 inside learned branches; exact eta0 was a separate declared eq control",
        no_new_DFT_MD_or_material_measurements=True,goal_status="active",
        conclusion="no stable incremental carrier prediction from this stock-closure family; preserve predictive goal, do not deploy it as an improvement")
    energy.write(OUT/"contract_audit.json",audit)
    figures(summary,by)
    print(json.dumps(audit,ensure_ascii=False,indent=2))

def figures(s,by):
    p=OUT/"figures";p.mkdir(exist_ok=True)
    methods=list(study.MODELS);colors=["#1b4f72","#b9770e","#148f77","#922b21","#7d3c98","#566573"]
    fig,ax=plt.subplots(figsize=(8.5,4.2))
    for m,c in zip(methods,colors):
        d=s[s.method.eq(m)&s.scope.eq("all_query")].sort_values("budget")
        ax.plot(d.budget,d.MAE,"o-",label=m,color=c)
    ax.set(xlabel="Same-source calibration anchors",ylabel="Source-equal Hall MAE (dex)",xticks=[0,1,2])
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(p/"prediction_comparison.png",dpi=170);fig.savefig(p/"prediction_comparison.pdf");plt.close(fig)
    g=by[by.budget.eq(0)&by.scope.eq("all_query")].pivot(index="family",columns="method",values="MAE")
    fig,ax=plt.subplots(figsize=(8.5,4.2));x=np.arange(len(g))
    for k,m in enumerate(("eq","fill_const","fill_temperature")):
        ax.bar(x+(k-1)*.25,g[m],width=.25,label=m)
    ax.set(xticks=x,xticklabels=["TEOS Si","MOVPE Sn","MOCVD Si","PAMBE Sn","MBE Si"],
           ylabel="Zero-anchor Hall MAE (dex)")
    ax.legend(fontsize=8);ax.grid(axis="y",alpha=.25);fig.tight_layout()
    fig.savefig(p/"source_errors.png",dpi=170);fig.savefig(p/"source_errors.pdf");plt.close(fig)

if __name__=="__main__":main()
