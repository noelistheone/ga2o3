"""Unify retained prediction evidence, audit boundaries, and derive full chains."""
from pathlib import Path
import sys,json,re,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import transport_chain as transport

OUT=energy.BASE/"phase56_validation"
P5=energy.BASE/"phase5_reference_features";P6=energy.BASE/"phase6_native_reference"
DIRECT=energy.BASE/"phase56_direct_controls"
CANONICAL={
    "Dose line":("direct","dose_line"),
    "Direct Ridge":("direct","direct_ridge"),
    "Direct trees":("direct","direct_trees"),
    "Mean energies":("phase2","global_mean"),
    "Full energies":("phase2","oracle_full"),
    "Old Ridge":("phase2","ridge"),
    "Tuned Ridge":("phase5","ridge_base_energy"),
    "Kernel base":("phase5","rbf_base_energy"),
    "Projected reference":("phase5","rbf_closed_energy"),
    "Projected property-selected":("phase5","rbf_closed_property"),
    "Raw reference":("phase5","rbf_raw_energy"),
    "Native reference":("phase6","rbf_native_energy"),
    "Native Ridge":("phase6","ridge_native_energy"),
    "Nonlinear loss":("phase4","kernel_nonlinear_l1"),
    "Ridge +2 energies":("phase2","Ridge_diagonal_xi_k2"),
    "Mean +2 energies":("phase2","mean_goal_k2"),
    "ET +2 energies":("phase2","ET_goal_k2"),
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=read_result_csv(DIRECT/"predictions.csv")
    direct=original.copy();direct["physical_valid"]=None
    direct["physical_validity"]="not_applicable_empirical"
    direct.to_csv(DIRECT/"predictions_applicability_v2.csv",index=False)
    assert np.array_equal(direct.predicted_log_nH.to_numpy(float),original.predicted_log_nH.to_numpy(float))
    energy.write(DIRECT/"physical_applicability_annotation_v2.json",dict(
        original_retained=True,prediction_values_changed=False,changed_rows=len(direct),
        reason="generic score adapter initially wrote True alongside not_applicable_empirical; canonical applicability is NA, not a physical validation",
        original_sha256=energy.sha(DIRECT/"predictions.csv"),
        canonical_sha256=energy.sha(DIRECT/"predictions_applicability_v2.csv")))
    paths={"phase2":energy.OUT/"all_property_predictions.csv",
           "phase3":energy.BASE/"phase3_target_loss/real_property/predictions.csv",
           "phase4":energy.BASE/"phase4_nonlinear/real_property/predictions.csv",
           "phase5":P5/"real_property/predictions.csv",
           "phase6":P6/"real_property/predictions.csv",
           "direct":DIRECT/"predictions_applicability_v2.csv"}
    frames=[]
    for origin,p in paths.items():
        d=read_result_csv(p);d["origin"]=origin;d["physical_applicable"]=origin!="direct"
        frames.append(d)
    data=pd.concat(frames,ignore_index=True)
    dataset=read_result_csv(energy.BASE/"dataset_v1.csv").set_index("sample_id")
    reference=dataset.observed_log_nH.to_dict()
    assert all(abs(r.observed_log_nH-reference[r.sample_id])<1e-10 for r in data.itertuples())
    data.to_csv(OUT/"all_primary_predictions.csv",index=False)
    roles=read_result_csv(energy.BASE/"episodes_v1.csv")
    common=set(roles[roles.budget.eq(2)&roles.role.eq("query")].sample_id)
    rows=[]
    for (o,m,b,f),g in data[data.role.eq("query")].groupby(["origin","method","budget","family"]):
        for scope in ("all_query","common18"):
            h=g if scope=="all_query" else g[g.sample_id.isin(common)]
            numeric=h.predicted_log_nH.notna()
            rows.append(dict(origin=o,method=m,budget=int(b),family=f,scope=scope,n=len(h),
                numeric=int(numeric.sum()),physical=int(h.physical_valid.eq(True).sum()) if o!="direct" else None,
                MAE=float(abs(h.predicted_log_nH-h.observed_log_nH).mean()) if numeric.all() else None))
    by=pd.DataFrame(rows);by.to_csv(OUT/"all_by_family.csv",index=False)
    summary=by.groupby(["origin","method","budget","scope"]).agg(
        families=("family","nunique"),queries=("n","sum"),numeric=("numeric","sum"),
        physical=("physical","sum"),scored_families=("MAE","count"),MAE=("MAE","mean"),worst=("MAE","max")).reset_index()
    summary.loc[summary.origin.eq("direct"),"physical"]=np.nan
    summary["full_numeric_coverage"]=summary.numeric.eq(summary.queries)&summary.scored_families.eq(5)
    summary.loc[~summary.full_numeric_coverage,"MAE"]=np.nan
    summary.to_csv(OUT/"all_summary.csv",index=False)
    canonical=[]
    for name,(origin,method) in CANONICAL.items():
        d=summary[summary.origin.eq(origin)&summary.method.eq(method)].copy()
        d["label"]=name;canonical.append(d)
    canonical=pd.concat(canonical,ignore_index=True)
    canonical.to_csv(OUT/"canonical_comparison.csv",index=False)
    # Full-chain assessment of all methods uses the same explicitly joined five specimens.
    valid=set(dataset[dataset.mobility_eligible.fillna(False).astype(bool)&dataset.mu_cm2Vs.gt(0)].index)
    assert len(valid)==5
    mu=read_result_csv(ROOT/"results/chain_diagnosis_20261015/executed/mobility_bounds_and_inputs.csv")
    heads={}
    for f in sorted(dataset.loc[list(valid)].family.unique()):
        tr=mu[mu.family.ne(f)]
        w=1/tr.family.map(tr.family.value_counts()).to_numpy(float)
        residual=np.log10(tr.mu_cm2Vs).to_numpy()-np.array([transport.logCT(x) for x in np.log10(tr.carrier_cm3)])
        heads[f]=dict(offset=transport.median(residual,w),training_families=sorted(tr.family.unique()))
    chains=[]
    for r in data[data.role.eq("query")&data.sample_id.isin(valid)].itertuples():
        pred_mu=transport.logCT(r.predicted_log_nH)+heads[r.family]["offset"] if np.isfinite(r.predicted_log_nH) else np.nan
        obs_mu=float(np.log10(dataset.loc[r.sample_id,"mu_cm2Vs"]))
        chains.append(dict(origin=r.origin,method=r.method,budget=int(r.budget),family=r.family,sample_id=r.sample_id,
            predicted_log_nH=r.predicted_log_nH,observed_log_nH=r.observed_log_nH,
            predicted_log_mu=pred_mu,observed_log_mu=obs_mu,
            predicted_log_sigma=np.log10(1.602176634e-19)+r.predicted_log_nH+pred_mu,
            observed_derived_log_sigma=np.log10(1.602176634e-19)+r.observed_log_nH+obs_mu))
    chains=pd.DataFrame(chains);chains.to_csv(OUT/"all_full_chain_predictions.csv",index=False)
    rows=[]
    for keys,g in chains.groupby(["origin","method","budget","family"]):
        q=dict(origin=keys[0],method=keys[1],budget=int(keys[2]),family=keys[3],n=len(g))
        for name in ("nH","mu","sigma"):
            obs="observed_derived_log_sigma" if name=="sigma" else "observed_log_"+name
            err=abs(g["predicted_log_"+name]-g[obs])
            q[name+"_MAE"]=float(err.mean()) if err.notna().all() else None
        rows.append(q)
    cb=pd.DataFrame(rows);cb.to_csv(OUT/"all_full_chain_by_family.csv",index=False)
    cs=cb.groupby(["origin","method","budget"]).agg(families=("family","nunique"),records=("n","sum"),
        nH_MAE=("nH_MAE","mean"),mu_MAE=("mu_MAE","mean"),sigma_MAE=("sigma_MAE","mean")).reset_index()
    cs.to_csv(OUT/"all_full_chain_summary.csv",index=False)
    audits=contract_audit(paths,heads)
    energy.write(OUT/"contract_audit.json",audits)
    figures(canonical)
    print(json.dumps(audits,ensure_ascii=False,indent=2),flush=True)

def contract_audit(paths,heads):
    protected=[]
    for file in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json","artifact_manifest_phase34.json"):
        x=json.loads((energy.BASE/file).read_text());items=x["files"] if isinstance(x,dict) else x
        changed=[v["path"] for v in items if not (ROOT/v["path"]).exists() or energy.sha(ROOT/v["path"])!=v["sha256"]]
        protected.append(dict(manifest=file,checked=len(items),changed=changed))
    assert all(not r["changed"] for r in protected)
    label_frame=read_result_csv(energy.OUT/"relative_charge_labels_private.csv")
    label_elements=dict(zip(label_frame.label_id,label_frame.element))
    leaks=[];inner_counts={};fit_cost={}
    for stage,folder in (("phase5",P5),("phase6",P6)):
        n=0;fit_seconds=0.;forward_seconds=0.
        for p in (folder/"inner").rglob("*.json"):
            v=json.loads(p.read_text());n+=1
            if v["held"] in v["training_elements"] or v["inner"] in v["training_elements"]:
                leaks.append(str(p))
            expected={label for label,e in label_elements.items() if e not in (v["held"],v["inner"])}
            if set(v["training_label_ids"])!=expected:leaks.append([str(p),"training label membership"])
            fit_seconds+=v["fit_seconds"];forward_seconds+=v["forward_seconds"]
        inner_counts[stage]=n;fit_cost[stage]=dict(fit_seconds_sum=fit_seconds,forward_seconds_sum=forward_seconds)
    assert inner_counts==dict(phase5=10080,phase6=336) and not leaks
    source_leaks=[]
    for folder,origin in ((P5,"phase5"),(P6,"phase6")):
        protocol=json.loads((folder/"real_property/protocol.json").read_text())
        for p in (folder/"real_property/outer").glob("*.csv"):
            d=read_result_csv(p);method=d.method.iloc[0]
            if method in protocol["methods"]:
                for key,g in d.groupby("fit_key"):
                    f=json.loads((folder/"real_property/fits"/f"{key}.json").read_text())
                    if any(x in f["train_families"] for x in g.family.unique()):source_leaks.append([origin,key])
    for p in (DIRECT/"fits").glob("*.json"):
        f=json.loads(p.read_text())
        assert f["train_families"]
    # Check direct outer and every nested fit membership from saved source choices.
    direct=read_result_csv(DIRECT/"predictions_applicability_v2.csv")
    for key,g in direct.groupby("fit_key"):
        f=json.loads((DIRECT/"fits"/f"{key}.json").read_text())
        if any(x in f["train_families"] for x in g.family.unique()):source_leaks.append(["direct",key])
    # Every candidate's inner fit must exclude both the outer and inner source;
    # outer-only metadata checks would not cover nested hyperparameter choice.
    selections=json.loads((DIRECT/"selections.json").read_text())
    families=sorted(direct.family.unique());checked=set()
    for selection in selections:
        held=selection["held"]
        for candidate in selection["candidates"]:
            for inner in (f for f in families if f!=held):
                training=sorted(f for f in families if f not in (held,inner))
                payload=dict(families=training,method=selection["method"],config=candidate["config"],
                             protocol=energy.sha(DIRECT/"protocol.json"))
                key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]
                scope=(held,inner,key)
                if scope in checked:continue
                checked.add(scope)
                fit=json.loads((DIRECT/"fits"/f"{key}.json").read_text())
                if fit["train_families"]!=training:source_leaks.append(["direct nested",held,inner,key])
    assert not source_leaks
    old=read_result_csv(energy.BASE/"phase3_target_loss/real_property/predictions.csv")
    new=read_result_csv(P5/"real_property/predictions.csv")
    a=old[old.method.eq("kernel_energy_select_energy")]
    b=new[new.method.eq("rbf_base_energy")]
    merged=a.merge(b,on=["sample_id","budget"],suffixes=("_old","_new"))
    base_error=float(abs(merged.predicted_log_nH_old-merged.predicted_log_nH_new).max())
    assert base_error<2e-6
    statuses={stage:json.loads((folder/"new_property_source/model_statuses.json").read_text())["refused_methods"]
              for stage,folder in (("phase5",P5),("phase6",P6))}
    return dict(created_utc=energy.now(),protected=protected,nested_inner_fits=inner_counts,
        actual_nested_cost=fit_cost,cost_scope="sum of CPU fit/forward seconds; not parallel wall time",
        element_leakage_cases=leaks,source_leakage_cases=source_leaks,
        direct_nested_source_memberships_checked=len(checked),
        original_kernel_replay_max_error_dex=base_error,related_study_refused_methods=statuses,
        transport_heads=heads,input_prediction_hashes={k:energy.sha(p) for k,p in paths.items()},
        current_data="29carrier/5sources;52conditionalmu/11sources;5full-chain pairs/2sources",
        related_study="12conditionalmu but2roundedchemical paired inputs; previously exposed and related team",
        new_DFT_or_MD_jobs=0,new_material_measurements=0,
        goal_direction="ML+Sandbox doped-property prediction; feature consistency is supporting evidence")

def figures(canonical):
    path=OUT/"figures";path.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    names=["Dose line","Direct Ridge","Full energies","Tuned Ridge","Ridge +2 energies","Mean +2 energies"]
    for name in names:
        for ax,scope in zip(axes,("all_query","common18")):
            d=canonical[canonical.label.eq(name)&canonical.scope.eq(scope)].sort_values("budget")
            ax.plot(d.budget,d.MAE,marker="o",label=name)
    for ax in axes:ax.set_xlabel("Measured Hall anchors");ax.set_xticks([0,1,2]);ax.set_ylabel("Source-equal nH MAE (dex)");ax.grid(alpha=.2)
    axes[0].set_title("29 / 23 / 18 queries");axes[1].set_title("Fixed18 common queries");axes[0].legend(fontsize=7)
    save(fig,path/"supported_options")
    e=read_result_csv(P5/"summary.csv").set_index("method")
    names=["rbf_base_energy","rbf_closed_energy","rbf_raw_energy"]
    labels=["Base18","Balanced reference","Raw reference"]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(labels,[e.loc[m,"energy_macro_MAE_eV"] for m in names])
    axes[0].set_ylabel("Secant energy MAE (eV)");axes[0].set_title("15-element nested energy test")
    q=canonical[canonical.scope.eq("all_query")&canonical.budget.eq(0)].set_index("label")
    axes[1].bar(labels,[q.loc[n,"MAE"] for n in ["Kernel base","Projected reference","Raw reference"]])
    axes[1].set_ylabel("Measured nH MAE (dex)");axes[1].set_title("29specimens /5sources")
    for ax in axes:ax.tick_params(axis="x",labelsize=8)
    save(fig,path/"representation_vs_property")
    src=read_result_csv(P5/"new_property_source/summary.csv")
    names=["identity_dose","ridge_base_energy","rbf_closed_property","P2_Ridge_goal_k2","oracle_full"]
    labels=["Dose identity","Tuned Ridge","Projected /property","Ridge+2energies","Full energies"]
    fig,axes=plt.subplots(1,2,figsize=(10.8,4))
    for ax,b in zip(axes,(0,1)):
        d=src[src.budget.eq(b)].set_index("method")
        y=np.array([d.loc[n,"nH_MAE"] for n in names]);lo=np.array([d.loc[n,"nH_scenario_min"] for n in names]);hi=np.array([d.loc[n,"nH_scenario_max"] for n in names])
        ax.bar(range(len(names)),y);ax.errorbar(range(len(names)),y,yerr=[y-lo,hi-y],fmt="none",color="k",capsize=3)
        ax.set_xticks(range(len(names)),labels,rotation=28,ha="right",fontsize=8)
        ax.set_ylabel("Measured nH MAE (dex)");ax.set_title(f"Related study:{b}anchor,{2-b}queries")
        ax.text(.02,.96,"Whiskers: rounding scenarios,\nnot confidence intervals",va="top",fontsize=7,transform=ax.transAxes)
    save(fig,path/"adverse_transfer")

def save(fig,p):
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(str(p)+"."+ext,dpi=180)
    plt.close(fig)

if __name__=="__main__":main()
