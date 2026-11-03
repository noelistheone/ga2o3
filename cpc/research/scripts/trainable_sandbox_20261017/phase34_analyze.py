"""Audit phase3/4 boundaries and export prediction-focused comparison figures."""
from pathlib import Path
import sys,json,pickle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase3_metric as metric
import phase3_property as old_property
import phase4_nested as nonlinear
import phase4_property as study

OUT=energy.BASE/"phase34_validation"
def main():
    protected=[]
    for filename in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json"):
        v=json.loads((energy.BASE/filename).read_text())
        entries=v["files"] if isinstance(v,dict) else v
        changed=[r["path"] for r in entries if not (ROOT/r["path"]).exists()
                 or energy.sha(ROOT/r["path"])!=r["sha256"]]
        protected.append(dict(manifest=filename,checked=len(entries),changed=changed))
    assert not any(v["changed"] for v in protected)
    # Check all recorded outer/inner memberships, including unselected candidates.
    leaks=[];inner_checks=0
    for p in sorted((metric.OUT/"outer").glob("*.json")):
        x=json.loads(p.read_text());held=x["held"]
        for s in x["selections"]:
            if held in s["training_elements"]:leaks.append([str(p),"outer",s["method"]])
            for candidate in s["all_candidates"]:
                for row in candidate["inner_details"]:
                    inner_checks+=1
                    if held in row["training_elements"] or row["element"] in row["training_elements"]:
                        leaks.append([str(p),s["method"],row["element"]])
    nonlinear_checks=0;accepted=0;trial_failures=0;loss_increases=[]
    stops={};states=0;fit_seconds=0.
    for p in sorted((nonlinear.OUT/"nested_inner").rglob("*.json")):
        x=json.loads(p.read_text());nonlinear_checks+=1
        if x["held_element"] in x["train_elements"] or x["inner_element"] in x["train_elements"]:
            leaks.append([str(p),"nonlinear inner"])
        states+=x["response_counter"]["requested_state_solves"];fit_seconds+=x["fit_seconds"]
        d=x["diagnostics"];reason=d["stop_reason"];stops[reason]=stops.get(reason,0)+1
        for row in d["optimization_trace"]:
            for trial in row.get("trials",[]):
                trial_failures+=int("error" in trial)
                if trial.get("accepted"):
                    accepted+=1
                    if trial["objective"]>row["objective"]+1e-8:
                        loss_increases.append([str(p),row["iteration"]])
    assert nonlinear_checks==168 and not leaks and not loss_increases
    source_leaks=[]
    predictions=read_result_csv(study.OUT/"predictions.csv")
    for row in predictions.itertuples():
        folder=(metric.OUT/"real_property") if row.method in study.CONTROLS else study.OUT
        fit=json.loads((folder/"fits"/f"{row.fit_key}.json").read_text())
        if row.family in fit["train_families"]:
            source_leaks.append([row.method,row.sample_id,row.budget])
    assert not source_leaks
    selected=[]
    for e in ("Si","Sn"):
        x=json.loads((nonlinear.OUT/"nested_outer"/f"{e}.json").read_text())
        assert e not in x["training_elements"]
        for s in x["selections"]:
            selected.append(dict(element=e,method=s["method"],
                candidate=s["selected_candidate"]["index"],config=s["selected_candidate"]["config"],
                inner_score=s["inner_nonlinear_MAE"],ordinary_inner_score=s["inner_ordinary_MAE"],
                energy_MAE_eV=s["energy_MAE"],virtual_MAE_dex=s["virtual_property_MAE"],
                accepted_steps=s["diagnostics"]["accepted_steps"],
                stop=s["diagnostics"]["stop_reason"]))
    summary=read_result_csv(study.OUT/"summary.csv")
    query=predictions[predictions.role.eq("query")]
    baseline=query[query.method.eq("kernel_energy")][["sample_id","budget","predicted_log_nH"]]
    current=query[query.method.eq("kernel_nonlinear_l1")].merge(
        baseline,on=["sample_id","budget"],suffixes=("_nonlinear","_ordinary"))
    current["error_difference"]=abs(current.predicted_log_nH_nonlinear-current.observed_log_nH)-abs(
        current.predicted_log_nH_ordinary-current.observed_log_nH)
    comparison=current.groupby(["budget","family"]).agg(
        records=("sample_id","count"),nonlinear_minus_ordinary_MAE=("error_difference","mean")).reset_index()
    audit=dict(created_utc=energy.now(),goal_direction="ML+Sandbox prediction of doped properties",
        protected=protected,phase3_inner_memberships_checked=inner_checks,
        phase4_nested_fits=nonlinear_checks,element_leakage_cases=leaks,source_leakage_cases=source_leaks,
        accepted_training_steps=accepted,accepted_loss_increases=loss_increases,
        rejected_numerical_trial_responses=trial_failures,finite_budget_stop_reasons=stops,
        nonlinear_inner_fit_seconds_sum=fit_seconds,
        nonlinear_inner_requested_state_solves=states,
        cost_scope="inner fits only; sum of CPU-fit seconds, not elapsed parallel wall time",
        outer_selections=selected,property_prediction_rows=len(predictions),
        property_query_rows=len(query),property_methods=int(predictions.method.nunique()),
        real_property_scope="same29 specimens/5DOI families;0/1/2anchors; common18 queries",
        virtual_scope="phase3 47eligible/90virtual inputs across9/15elements; not material measurements",
        full_chain_scope="5 explicitly paired specimens/2DOI families; sigma derived",
        new_study_scope="12 conditional mobility pairs but only2 rounded paired chemical inputs, related-study retrospective test",
        new_DFT_or_MD_jobs=0,new_material_measurements=0,
        numerical_checks_are_material_validation=False,
        current_model_advantage="nonlinear real-material increment not established")
    energy.write(OUT/"contract_audit.json",audit)
    comparison.to_csv(OUT/"nonlinear_vs_ordinary_by_source.csv",index=False)
    summary.to_csv(OUT/"real_property_summary.csv",index=False)
    figures(summary)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("outer_selections",)},ensure_ascii=False,indent=2))

def figures(summary):
    figdir=OUT/"figures";figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8))
    methods=["global_mean","kernel_energy","kernel_diagonal_l1","kernel_full_l1","kernel_nonlinear_l1","oracle_full"]
    labels=["Mean","Energy kernel","Local diagonal","Local full","Full nonlinear","All energies"]
    for m,label in zip(methods,labels):
        d=summary[summary.method.eq(m)&summary.scope.eq("all_query")].sort_values("budget")
        axes[0].plot(d.budget,d.family_equal_MAE,marker="o",label=label)
        d=summary[summary.method.eq(m)&summary.scope.eq("common18")].sort_values("budget")
        axes[1].plot(d.budget,d.family_equal_MAE,marker="o",label=label)
    for ax in axes:
        ax.set_xlabel("Measured Hall anchor budget");ax.set_xticks([0,1,2])
        ax.set_ylabel("Source-equal measured nH MAE (dex)");ax.grid(alpha=.2)
    axes[0].set_title("29 / 23 / 18 query records")
    axes[1].set_title("Fixed common18 query records")
    axes[0].legend(fontsize=7)
    save(fig,figdir/"real_property")
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    p=read_result_csv(nonlinear.OUT/"pilot_summary.csv")
    for i,e in enumerate(("Si","Sn")):
        d=p[p.held_element.eq(e)] if "held_element" in p else p[p.element.eq(e)]
        # Use explicit dictionaries: row order is not a model-selection rule.
        lookup={r.model:r for r in d.itertuples()}
        names=list(lookup)
        values=[lookup[n].property_MAE if hasattr(lookup[n],"property_MAE") else lookup[n].virtual_property_MAE for n in names]
        axes[i].bar(range(len(names)),values)
        axes[i].set_xticks(range(len(names)),[n.replace("kernel_","").replace("_","\n") for n in names],fontsize=8)
        axes[i].set_title(e+" fixed configuration, diagnostic only")
        axes[i].set_ylabel("Virtual response MAE (dex)")
    save(fig,figdir/"nonlinear_pilot")
    src=read_result_csv(energy.BASE/"new_property_sources/azizie2023_full_chain_summary.csv")
    new=read_result_csv(nonlinear.OUT/"new_property_source/azizie2023_full_chain_summary.csv")
    data=pd.concat([src,new],ignore_index=True)
    methods=["identity_dose","global_mean","kernel_energy","kernel_nonlinear_l1","P2_Ridge_goal_k2","oracle_full"]
    labels=["Dose identity","Mean","Energy kernel","Full nonlinear","Ridge +2 energies","All energies"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,b in zip(axes,(0,1)):
        d=data[data.budget.eq(b)].set_index("method")
        y=[d.loc[m,"nH_MAE"] for m in methods]
        low=[d.loc[m,"nH_scenario_min"] for m in methods];hi=[d.loc[m,"nH_scenario_max"] for m in methods]
        ax.bar(range(len(methods)),y,color="#4c78a8")
        ax.errorbar(range(len(methods)),y,yerr=[np.array(y)-low,np.array(hi)-y],fmt="none",color="black",capsize=3)
        ax.set_xticks(range(len(methods)),labels,rotation=35,ha="right",fontsize=8)
        ax.set_ylabel("Measured nH MAE (dex)")
        ax.set_title(f"Azizie2023: {b} anchor, {2-b} queries")
        ax.text(.02,.96,"Bars: primary rounded input\nWhiskers: fixed rounding scenarios,\nnot confidence intervals",transform=ax.transAxes,va="top",fontsize=7)
    save(fig,figdir/"related_study")

def save(fig,stem):
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(str(stem)+"."+ext,dpi=180)
    plt.close(fig)

if __name__=="__main__":main()
