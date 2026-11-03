"""Independent loss, source, feature, anchor and full-coverage audit."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase2_energy as energy
from phase9_prediction_audit import summary_rows

BASE=energy.BASE;OUT=BASE/"phase11_parent_allocation"


def asfloat(x):return np.inf if x is None else float(x)


def main():
    protocol=json.loads((OUT/"prediction_protocol.json").read_text())
    for file in ("prediction_protocol.json","numerical_protocol.json","learning_protocol.json"):
        p=json.loads((OUT/file).read_text())
        for group in ("code_hashes","input_hashes"):
            for path,sha in p.get(group,{}).items():assert energy.sha(ROOT/path)==sha,path
    obs=read_result_csv(BASE/"phase9_complete_temperature/observations_and_inputs.csv")
    data=read_result_csv(BASE/"dataset_v1.csv");roles=read_result_csv(BASE/"episodes_v1.csv")
    specs=read_result_csv(BASE/"phase8_process_temperature/extraction_v2/paired_specimens.csv")
    curves=read_result_csv(BASE/"phase8_process_temperature/extraction_v2/temperature_observations.csv")
    curves=curves[curves.endpoint.eq("log10_nH_cm3")]
    template=read_result_csv(BASE/"phase9_complete_temperature/methods/learned_screen_joint/temperature_predictions.csv")
    template=template.set_index(["obs_id","budget"])
    common=set(roles[roles.budget.eq(2)&roles.role.eq("query")].sample_id)
    reading=set(curves[curves.T_upper_K.ge(40)&curves.T_lower_K.le(125)].read_id)
    fs=set(data.family);fits={p.stem:json.loads(p.read_text()) for p in (OUT/"fits").glob("*.json")}
    lookup={(f["method"],tuple(f["train_families"]),json.dumps(f["config"],sort_keys=True)):f for f in fits.values()}
    assert len(fits)==540,len(fits)
    assert len(lookup)==len(fits)
    stats=[];trialrows=[];parameter_rows=[];origin_replays=[];loss_checks=[]
    for f in fits.values():
        d=protocol["methods"][f["method"]]
        selected=obs[obs.family.isin(f["train_families"])]
        rt=selected[selected.kind.eq("RT")].copy()
        rt["weight"]=1/(rt.groupby("family").sample_id.transform("count").to_numpy(float)*rt.family.nunique())
        curve=selected[selected.kind.eq("curve")&selected.Tobs_K.le(275)].copy()
        if len(curve):curve["weight"]=.1/(curve.groupby("sample_id").obs_id.transform("count").to_numpy(float)*
            curve.groupby("family").sample_id.transform("nunique").to_numpy(float)*curve.family.nunique())
        tr=rt if d["training"]=="RT" else pd.concat([rt,curve],ignore_index=True).sort_values(["family","sample_id","Tobs_K","kind"])
        assert tr.obs_id.tolist()==f["training_obs_ids"]
        assert sorted(tr.sample_id.unique())==f["training_specimens"]
        assert np.allclose(tr.weight,f["weight_per_observation"],atol=1e-13,rtol=1e-13)
        m=f["feature_metadata"];w=rt.weight.to_numpy(float)
        center=float(w@np.log10(rt.incorporated_cm3))
        scale=float(np.sqrt(w@((np.log10(rt.incorporated_cm3)-center)**2)))
        assert np.isclose(center,m["logC_center"],atol=1e-13,rtol=0)
        assert np.isclose(scale,m["logC_scale"],atol=1e-13,rtol=0)
        assert set(m["training_specimens"])==set(rt.sample_id)
        pred={p["obs_id"]:p for p in f["predictions"]};assert set(pred)==set(obs.obs_id)
        best=min([t for t in f["trials"] if t.get("loss") is not None],key=lambda t:t["loss"],default=None)
        if best:
            assert np.allclose(f["parameters"],best["parameters"],atol=0,rtol=0)
            vals=np.array([pred[i]["predicted"] for i in tr.obs_id],float)
            residual=vals-tr.target.to_numpy(float)
            nbase=4 if d["training"]=="joint" else 2
            penalty=f["config"].get("reg",0.)*np.sum(np.array(f["parameters"])[nbase:]**2) if d["field"] in ("parent","Tf","output") else 0.
            loss=float(np.sum(tr.weight.to_numpy(float)*residual**2)+penalty)
            error=abs(loss-best["loss"])
            assert error<2e-7*(1+abs(loss)),(f["key"],error,loss,best["loss"])
            loss_checks.append(dict(fit_key=f["key"],objective=loss,reported=best["loss"],error=error))
        for i,t in enumerate(f["trials"]):
            trialrows.append(dict(fit_key=f["key"],method=f["method"],trial=i,success=t["success"],
                selected=t is best,loss=t.get("loss"),nfev=t.get("nfev"),reused=f["origin"] is not None))
        if f["parameters"] is not None:
            parameter_rows.append(dict(fit_key=f["key"],method=f["method"],training_sources=len(f["train_families"]),
                reg=f["config"].get("reg"),parameters=json.dumps(f["parameters"]),
                selected_start_converged=best["success"] if best else None))
        if f["origin"]:
            origin=ROOT/f["origin"]["path"];assert energy.sha(origin)==f["origin"]["sha256"]
            old=json.loads(origin.read_text());original={p["obs_id"]:p for p in old["predictions"]}
            errors=[]
            for i,p in pred.items():
                assert p["numeric"]==original[i]["numeric"]
                if p["numeric"]:errors.append(abs(p["predicted"]-original[i]["predicted"]))
            assert max(errors)<2e-8
            origin_replays.append(dict(fit_key=f["key"],rows=len(errors),maximum_error=max(errors)))
        stats.append(dict(fit_key=f["key"],method=f["method"],seconds=f["seconds"],
            trials=len(f["trials"]),reused=f["origin"] is not None,
            nonconverged=sum(not t["success"] for t in f["trials"]),
            selected_nonconverged=bool(best and not best["success"]),state_failures=len(f["state_failures"]),**f["counters"]))

    def canonical(method,config):
        config=dict(config)
        if "reg" in config and config["reg"] is None:
            return ("joint_baseline" if protocol["methods"][method]["training"]=="joint" else "thermal_RT"),{}
        return method,config
    def locate(method,families,config):
        m,c=canonical(method,config)
        return lookup[(m,tuple(sorted(families)),json.dumps(c,sort_keys=True))]
    def primary_score(f,family,budget,lam):
        r=roles[roles.family.eq(family)&roles.budget.eq(budget)].set_index("sample_id").role
        d=data[data.family.eq(family)];p={x["obs_id"]:x["predicted"] for x in f["predictions"]}
        a=np.array([r.loc[s]=="anchor" for s in d.sample_id]);q=np.array([r.loc[s]=="query" for s in d.sample_id])
        vals=np.array([p["RT::"+s] for s in d.sample_id],float)
        delta=0.
        if a.any() and lam is not None:
            if np.isfinite(vals[a]).all():delta=np.sum(d.observed_log_nH.to_numpy(float)[a]-vals[a])/(a.sum()+lam)
            else:vals[:]=np.nan
        vals+=delta
        score=float(abs(vals[q]-d.observed_log_nH.to_numpy(float)[q]).mean()) if np.isfinite(vals[q]).all() else np.inf
        return score,delta,dict(zip(d.sample_id,vals))
    allp=[];allt=[];designs=[];selected_keys=set();candidate_evals=0;lineage=0
    for method,definition in protocol["methods"].items():
        folder=OUT/"methods"/method
        p=read_result_csv(folder/"primary_predictions.csv");t=read_result_csv(folder/"temperature_predictions.csv")
        assert len(p)==87 and len(t)==447
        assert not p.duplicated(["sample_id","budget"]).any() and not t.duplicated(["obs_id","budget"]).any()
        for budget in (0,1,2):
            assert set(p[p.budget.eq(budget)].sample_id)==set(data.sample_id)
        tt=t.set_index(["obs_id","budget"])
        assert set(tt.index)==set(template.index)
        for col in ("role","common40_125","sample_id","family"):
            assert tt[col].reindex(template.index).equals(template[col]),(method,col)
        choices=json.loads((folder/"selections.json").read_text());assert len(choices)==15
        for s in choices:
            training=fs-{s["held"]};b=s["budget"]
            for c in s["candidates"]:
                values=[];keys=[]
                for inner in sorted(training):
                    f=locate(method,training-{inner},c["config"])
                    assert inner not in f["train_families"] and s["held"] not in f["train_families"]
                    values.append(primary_score(f,inner,b,c["lam"])[0]);keys.append(f["key"])
                    candidate_evals+=1
                assert keys==c["inner_keys"]
                assert np.isclose(np.mean(values),asfloat(c["score"]),atol=3e-10,rtol=0)
            best=min(s["candidates"],key=lambda c:(round(asfloat(c["score"]),10),-asfloat(c["lam"]),c["index"]))
            assert best==s["selected"]
            f=locate(method,training,best["config"]);assert f["key"]==s["fit_key"]
            selected_keys.add(f["key"])
            score,delta,expected=primary_score(f,s["held"],b,best["lam"])
            assert np.isclose(score,asfloat(s["score"]),atol=3e-10,rtol=0)
            q=p[p.family.eq(s["held"])&p.budget.eq(b)]
            for row in q.itertuples():
                assert row.sample_id not in f["training_specimens"] and row.family not in f["train_families"]
                assert roles[roles.sample_id.eq(row.sample_id)&roles.budget.eq(b)].role.iloc[0]==row.role
                assert (pd.isna(row.predicted_log_nH) and np.isnan(expected[row.sample_id])) or np.isclose(row.predicted_log_nH,expected[row.sample_id],atol=3e-12,rtol=0)
                lineage+=1
            designs.append(dict(method=method,held=s["held"],budget=b,config=json.dumps(best["config"],sort_keys=True),
                shrinkage=best["lam"],fit_key=f["key"],inner_MAE=best["score"],outer_MAE=s["score"]))
        for (family,sample,budget),g in t.groupby(["family","sample_id","budget"]):
            f=fits[g.fit_key.iloc[0]]
            assert family not in f["train_families"] and sample not in f["training_specimens"]
            s=next(c for c in choices if c["held"]==family and c["budget"]==budget)
            assert f["key"]==s["fit_key"]
            a=g[g.role.eq("anchor")];assert len(a)==budget
            pred={r["obs_id"]:r["predicted"] for r in f["predictions"]}
            delta=0.;complete=all(pred[r.obs_id] is not None for r in a.itertuples())
            if budget and s["selected"]["lam"] is not None and complete:
                delta=sum(r.observed-pred[r.obs_id] for r in a.itertuples())/(budget+s["selected"]["lam"])
            for r in g.itertuples():
                val=pred[r.obs_id]
                if val is not None and complete:assert np.isclose(val+delta,r.predicted,atol=3e-12,rtol=0)
                else:assert pd.isna(r.predicted)
                lineage+=1
        allp.append(p);allt.append(t)
    p=pd.concat(allp,ignore_index=True);t=pd.concat(allt,ignore_index=True)
    mask=lambda g:pd.Series(True,index=g.index)
    pf,ps=summary_rows(p,"observed_log_nH","predicted_log_nH",dict(all_query=mask,common18=lambda g:g.sample_id.isin(common)))
    tf,ts=summary_rows(t,"observed","predicted",dict(all_query=mask,common40_125=lambda g:g.common40_125,
        reading_bound40_125=lambda g:g.obs_id.isin(reading)),temperature=True)
    for frame,name in ((p,"primary_predictions"),(t,"temperature_predictions"),(pf,"primary_by_source"),(ps,"primary_summary"),
        (tf,"temperature_by_source"),(ts,"temperature_summary"),(pd.DataFrame(stats),"fit_costs"),
        (pd.DataFrame(trialrows),"optimizer_trials"),(pd.DataFrame(parameter_rows),"parameters"),
        (pd.DataFrame(designs),"selected_designs"),(pd.DataFrame(loss_checks),"training_loss_replays")):
        frame.to_csv(OUT/(name+".csv"),index=False)
    pairs=[]
    for task,frame,scope in (("primary",pf,"all_query"),("primary_common",pf,"common18"),("temperature",tf,"common40_125")):
        z=frame[frame.scope.eq(scope)]
        for learner,baseline in (("parent_rbf_joint","joint_baseline"),("parent_rbf_joint","tf_rbf_joint"),
            ("parent_rbf_joint","output_rbf_joint"),("parent_rbf_joint","original_RT"),
            ("parent_rbf_RT","thermal_RT"),("parent_rbf_RT","output_rbf_RT")):
            for budget in (0,1,2):
                a=z[z.method.eq(learner)&z.budget.eq(budget)].set_index("family").MAE
                b=z[z.method.eq(baseline)&z.budget.eq(budget)].set_index("family").MAE;diff=a-b
                pairs.append(dict(task=task,method=learner,baseline=baseline,budget=budget,sources=len(diff),
                    better=int((diff< -1e-10).sum()),tied=int((abs(diff)<=1e-10).sum()),worse=int((diff>1e-10).sum()),
                    source_mean_difference=float(diff.mean())))
    pd.DataFrame(pairs).to_csv(OUT/"paired_source_comparisons.csv",index=False)
    # All numerical/Hall/detection/domain flags are kept without deleting rows.
    annotations=[]
    for frame,task in ((p,"primary"),(t,"temperature")):
        for r in frame.itertuples():
            f=fits[r.fit_key];obs_id="RT::"+r.sample_id if task=="primary" else r.obs_id
            raw=next(x for x in f["predictions"] if x["obs_id"]==obs_id);dg=raw.get("diagnostics",{})
            inp=obs[obs.obs_id.eq(obs_id)].iloc[0]
            annotations.append(dict(task=task,method=r.method,budget=int(r.budget),sample_id=r.sample_id,obs_id=obs_id,role=r.role,
                numeric=raw["numeric"],raw_physical_valid=raw.get("physical_valid"),
                flag_scope="numerical/oneelectronHall only;not complete material validation" if raw.get("physical_valid") is not None else "empirical output,no new physical state",
                above_reference_floor=dg.get("above_carrier_detection_floor"),chemical_dose_cm3=inp.incorporated_cm3,
                above_Si_Mott_reference=bool(inp.incorporated_cm3>2.4e18),
                Mott_scope="Si reference only,not validated Sn threshold",H_reported=bool(inp.H_known),
                carbon_known_unmodeled=bool(inp.carbon_known),process_bias=raw.get("extra",{}).get("process_bias")))
    ann=pd.DataFrame(annotations);ann.to_csv(OUT/"canonical_prediction_annotations.csv",index=False)
    protected=[]
    for file in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json","artifact_manifest_phase34.json",
                 "artifact_manifest_phase56.json","artifact_manifest_phase7.json","artifact_manifest_phase8.json","artifact_manifest_phase9.json","artifact_manifest_phase10.json"):
        x=json.loads((BASE/file).read_text());entries=x["files"] if isinstance(x,dict) else x
        changed=[r["path"] for r in entries if not (ROOT/r["path"]).is_file() or energy.sha(ROOT/r["path"])!=r["sha256"]]
        assert not changed,changed;protected.append(dict(manifest=file,checked=len(entries),changed=changed))
    cost=pd.DataFrame(stats);trials=pd.DataFrame(trialrows)
    outerbad=trials[trials.selected&~trials.success&trials.fit_key.isin(selected_keys)]
    audit=dict(passed=True,created_utc=energy.now(),goal_status="active",goal_complete=False,
        main_endpoint="unchanged actualchemicaldose toRT29specimens5DOI",methods=len(protocol["methods"]),
        cached_fits=len(fits),source_candidate_evaluations=candidate_evals,saved_prediction_lineage_checks=lineage,
        primary_rows=len(p),temperature_rows=len(t),new_material_specimens=0,
        new_optimizer_starts=int((~trials.reused).sum()),reused_optimizer_starts=int(trials.reused.sum()),
        new_nonconverged_starts=int(((~trials.success)&(~trials.reused)).sum()),
        selected_nonconverged_fits=int(cost.selected_nonconverged.sum()),selected_outer_nonconverged_fits=outerbad.fit_key.nunique(),
        all_new_optimizer_starts_converged=not bool(((~trials.success)&(~trials.reused)).any()),
        temperature_episode_roles_match_frozen_template=True,
        state_failures=int(cost.state_failures.sum()),new_fit_seconds=float(cost.seconds.sum()),
        root_counters={c:int(cost[c].sum()) for c in ("root_calls","hot_solve_calls","hot_cache_hits","root_evaluations")},
        counter_scope="actual complete predict calls,hot solves,cache hits,hot residual evaluations;not equal elapsed budget",
        inherited_control_replay_rows=sum(r["rows"] for r in origin_replays),
        inherited_control_maximum_error=max(r["maximum_error"] for r in origin_replays),
        objective_reconstruction_max_error=max(r["error"] for r in loss_checks),
        training_inputs_weights_feature_statistics_source_exclusion_verified=True,
        candidate_scores_and_anchor_offsets_reconstructed=True,
        failed_primary_queries=int(p[p.role.eq("query")].predicted_log_nH.isna().sum()),
        failed_temperature_queries=int(t[t.role.eq("query")].predicted.isna().sum()),
        raw_Hall_or_numerical_invalid_query_entries=int(ann[ann.role.eq("query")].raw_physical_valid.eq(False).sum()),
        prior_scope="effective parent process prior,not new formation energy orunique microscopic compensation mechanism",
        task_scope="newjointloss RT +.1curves;compare matched baselines;RTselected hyperparameters transferred toT",
        scope="retrospective exposed data,7sameSiTcurves/2DOI,no newDFT/MD/GPUtraining;oldproduction models unchanged",
        previous_protected=protected,sufficient_to_mark_goal_complete=False)
    energy.write(OUT/"contract_audit.json",audit);energy.write(OUT/"inherited_control_replays.json",origin_replays)
    print(json.dumps({k:v for k,v in audit.items() if k!="previous_protected"},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
