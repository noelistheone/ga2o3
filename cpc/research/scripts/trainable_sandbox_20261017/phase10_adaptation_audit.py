"""Independent saved-result/source/label audit and complete metric aggregation."""
from pathlib import Path
import sys,json
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/"phase10_internal_adaptation"


def value(v):return np.inf if v is None else float(v)


def main():
    protocol=json.loads((OUT/"protocol.json").read_text())
    for group in ("code_hashes","input_hashes"):
        for path,sha in protocol[group].items():assert energy.sha(ROOT/path)==sha,path
    for file in ("prediction_protocol.json","numerical_protocol.json"):
        p=json.loads((BASE/"phase9_complete_temperature"/file).read_text())
        for group in ("code_hashes","input_hashes","inputs"):
            for path,sha in p.get(group,{}).items():assert energy.sha(ROOT/path)==sha,path
    obs=read_result_csv(BASE/"phase9_complete_temperature/observations_and_inputs.csv")
    data=read_result_csv(BASE/"dataset_v1.csv")
    roles=read_result_csv(BASE/"episodes_v1.csv")
    specs=read_result_csv(BASE/"phase8_process_temperature/extraction_v2/paired_specimens.csv")
    curves=read_result_csv(BASE/"phase8_process_temperature/extraction_v2/temperature_observations.csv")
    curves=curves[curves.endpoint.eq("log10_nH_cm3")]
    fs=set(data.family);sample_index=obs[obs.kind.eq("RT")].set_index("sample_id")
    by_obs=obs.set_index("obs_id")
    common=set(roles[roles.budget.eq(2)&roles.role.eq("query")].sample_id)
    reading_ids=set(curves[curves.T_upper_K.ge(40)&curves.T_lower_K.le(125)].read_id)
    files=list((OUT/"calibrations").glob("*/*.json"))
    paths={p.stem:p for p in files}
    assert len(paths)==len(files)
    base_registry={}

    @lru_cache(maxsize=None)
    def cal(key):
        p=paths[key];c=json.loads(p.read_text());bp=ROOT/c["base_path"]
        if c["base_path"] not in base_registry:
            assert energy.sha(bp)==c["base_sha256"]
            f=json.loads(bp.read_text());base_registry[c["base_path"]]=f
            tr=obs[obs.family.isin(f["train_families"]) & ~obs.sample_id.isin(f.get("dropped_specimens",[]))]
            tr=tr[tr.kind.eq("RT")] if f["method"].endswith("_RT") else tr[tr.kind.eq("RT")|tr.Tobs_K.le(275)]
            assert tr.obs_id.tolist()==f["training_obs_ids"]
            assert sorted(tr.sample_id.unique())==f["training_specimens"]
            w=1/(tr.groupby("sample_id").obs_id.transform("count").to_numpy(float)*
                 tr.groupby("family").sample_id.transform("nunique").to_numpy(float));w/=w.mean()
            assert np.allclose(w,f["weight_per_observation"],atol=1e-13,rtol=1e-13)
        assert base_registry[c["base_path"]]["key"]==c["base_fit_key"]
        return c

    def expected_roles(task,context,budget):
        if task=="primary":
            r=roles[roles.family.eq(context)&roles.budget.eq(budget)]
            return {"RT::"+row.sample_id:row.role for row in r.itertuples()}
        d=obs[obs.sample_id.eq(context)]
        anchors=[]
        if budget:anchors=[d[d.kind.eq("RT")].obs_id.iloc[0]]
        if budget==2:
            q=d[d.kind.eq("curve")&d.Tobs_K.between(125,175)]
            anchors.append(q.loc[abs(q.Tobs_K-150).sort_values(kind="stable").index[0]].obs_id)
        result={}
        for r in d.itertuples():
            role="unavailable_reference" if r.kind=="RT" else "query"
            if r.obs_id in anchors:role="anchor"
            elif budget and abs(r.Tobs_K-300)<=25:role="guard"
            if budget==2 and r.obs_id not in anchors and abs(r.Tobs_K-by_obs.loc[anchors[1],"Tobs_K"])<=10:role="guard"
            result[r.obs_id]=role
        return result

    checked_keys=set();candidate_evals=0;lineage_checks=0;outer_keys=set();frames=[]
    selection_rows=[]
    def check_calibration(c):
        if c["key"] in checked_keys:return
        er=expected_roles(c["task"],c["context"],c["budget"])
        assert {r["obs_id"]:r["role"] for r in c["rows"]}==er
        a=[x for x,role in er.items() if role=="anchor"]
        assert set(a)==set(c["anchor_obs_ids"])
        assert len(a)==c["budget"]
        assert np.allclose([by_obs.loc[x,"target"] for x in c["anchor_obs_ids"]],c["anchor_observations"],atol=1e-13)
        for r in c["rows"]:
            assert np.isclose(r["observed"],by_obs.loc[r["obs_id"],"target"],atol=1e-13)
        q=[r for r in c["rows"] if r["role"]=="query" and
           (c["task"]=="primary" or r["common40_125"])]
        score=float(np.mean([abs(r["predicted"]-r["observed"]) for r in q])) if all(r["numeric"] for r in q) else np.inf
        assert np.isclose(score,value(c["score"]),atol=3e-12,rtol=0)
        checked_keys.add(c["key"])

    for method in protocol["methods"]:
        folder=OUT/"methods"/method
        frame=read_result_csv(folder/"predictions.csv")
        assert len(frame)==534
        assert not frame.duplicated(["task","obs_id","budget"]).any()
        selections=json.loads((folder/"selections.json").read_text());assert len(selections)==21
        for s in selections:
            task,held,budget=s["task"],s["held"],s["budget"]
            expected_fs=fs-{held}
            inners=sorted(expected_fs) if task=="primary" else sorted(specs[specs.family.ne(held)].sample_id)
            for candidate in s["candidates"]:
                scores=[];seen=[]
                for key in candidate["inner_calibration_keys"]:
                    c=cal(key);check_calibration(c);f=base_registry[c["base_path"]]
                    assert c["task"]==task and c["budget"]==budget and c["method"]==method
                    assert c["config"]==candidate["config"] and c["lam"]==candidate["lam"]
                    seen.append(c["context"])
                    if task=="primary":
                        assert set(f["train_families"])==expected_fs-{c["context"]}
                        assert c["context"] not in f["train_families"]
                    else:
                        assert set(f["train_families"])==expected_fs
                        assert c["context"] not in f["training_specimens"]
                        assert set(f.get("dropped_specimens",[]))=={c["context"]}
                    assert held not in f["train_families"]
                    scores.append(value(c["score"]));lineage_checks+=1;candidate_evals+=1
                assert sorted(seen)==inners
                assert np.isclose(np.mean(scores),value(candidate["score"]),atol=3e-12,rtol=0)
            best=min(s["candidates"],key=lambda c:(round(value(c["score"]),10),-value(c["lam"]),c["index"]))
            assert best==s["selected"]
            expected_queries=[held] if task=="primary" else sorted(specs[specs.family.eq(held)].sample_id)
            scores=[];seen=[]
            for key in s["outer_calibration_keys"]:
                outer_keys.add(key);c=cal(key);check_calibration(c);f=base_registry[c["base_path"]]
                assert set(f["train_families"])==expected_fs and not f.get("dropped_specimens",[])
                assert c["config"]==best["config"] and c["lam"]==best["lam"]
                assert held not in f["train_families"]
                seen.append(c["context"]);scores.append(value(c["score"]))
                q=frame[frame.calibration_key.eq(key)]
                assert len(q)==len(c["rows"])
                saved={r["obs_id"]:r for r in c["rows"]}
                for r in q.itertuples():
                    assert r.sample_id not in f["training_specimens"]
                    ref=saved[r.obs_id]
                    assert r.role==ref["role"]
                    assert (pd.isna(r.predicted) and ref["predicted"] is None) or np.isclose(r.predicted,ref["predicted"],atol=3e-12,rtol=0)
                    lineage_checks+=1
            assert sorted(seen)==expected_queries
            assert np.isclose(np.mean(scores),value(s["score"]),atol=3e-12,rtol=0)
            selection_rows.append(dict(task=task,held=held,method=method,budget=budget,
                candidate_count=len(s["candidates"]),selected_lambda=best["lam"],
                selected_config=json.dumps(best["config"],sort_keys=True),
                inner_MAE=best["score"],outer_MAE=s["score"]))
        frames.append(frame)
    predictions=pd.concat(frames,ignore_index=True)
    by=[]
    for (task,method,budget,family),g in predictions[predictions.role.eq("query")].groupby(["task","method","budget","family"]):
        scopes=("all_query","common18") if task=="primary" else ("all_query","common40_125","reading_bound40_125","below100K","at_least100K")
        for scope in scopes:
            h=g if scope=="all_query" else (g[g.sample_id.isin(common)] if scope=="common18" else
                (g[g.common40_125] if scope=="common40_125" else (g[g.obs_id.isin(reading_ids)] if scope=="reading_bound40_125" else
                 (g[g.Tobs_K.lt(100)] if scope=="below100K" else g[g.Tobs_K.ge(100)]))))
            if not len(h):continue
            error=abs(h.predicted-h.observed)
            if error.notna().all():
                mae=float(error.mean()) if task=="primary" else float(h.assign(error=error).groupby("sample_id").error.mean().mean())
            else:mae=np.inf
            by.append(dict(task=task,method=method,budget=int(budget),family=family,scope=scope,
                specimens=h.sample_id.nunique(),reads=len(h),numeric=int(error.notna().sum()),MAE=mae))
    by=pd.DataFrame(by)
    summary=by.groupby(["task","method","budget","scope"]).agg(
        source_groups=("family","nunique"),specimens=("specimens","sum"),reads=("reads","sum"),
        numeric=("numeric","sum"),MAE=("MAE","mean"),worst_source_MAE=("MAE","max")).reset_index()
    trials=[];costs=[];diagnostics=[];failures=[];objective_checks=[]
    for key in paths:
        c=cal(key)
        dg=c["diagnostics"]
        costs.append(dict(calibration_key=key,method=c["method"],task=c["task"],budget=c["budget"],
            outer=key in outer_keys,seconds=c["seconds"],status=c["status"],trials=len(c["trials"]),
            forward_calls=dg.get("forward_calls",0),state_failures=len(dg.get("state_failures",[]))))
        if c["status"].startswith("adapted") and c["trials"]:
            a=[r for r in c["rows"] if r["role"]=="anchor"]
            rawloss=sum((r["raw_prediction"]-r["observed"])**2 for r in a)
            adaptedloss=sum((r["predicted"]-r["observed"])**2 for r in a)
            total=adaptedloss+dg["standardized_prior_penalty"]
            objective_checks.append(dict(calibration_key=key,outer=key in outer_keys,
                base_anchor_loss=rawloss,adapted_penalized_loss=total,
                worsened=bool(total>rawloss+1e-8*(1+rawloss))))
        selected=min([t for t in c["trials"] if t.get("loss") is not None],key=lambda t:t["loss"],default=None)
        for i,t in enumerate(c["trials"]):
            trials.append(dict(calibration_key=key,method=c["method"],task=c["task"],budget=c["budget"],outer=key in outer_keys,
                trial=i,success=t["success"],selected=t is selected,loss=t.get("loss"),nfev=t.get("nfev")))
        if key in outer_keys:
            diagnostics.append(dict(calibration_key=key,method=c["method"],task=c["task"],context=c["context"],
                budget=c["budget"],status=c["status"],lam=c["lam"],parameters=c["parameters"],
                config=c["config"],rank=dg.get("rank"),coordinates=dg.get("coordinates"),condition=dg.get("condition"),
                effective_data_dimensions=dg.get("effective_data_dimensions"),singular_values=dg.get("singular_values"),
                resolution=dg.get("data_resolution_eigenvalues"),active_bounds=dg.get("active_bounds"),
                initial_parameters=dg.get("initial_parameters"),standardized_shift=dg.get("standardized_shift"),
                prior_penalty=dg.get("standardized_prior_penalty")))
        for p in c["predictions"]:
            if not p["numeric"]:failures.append(dict(calibration_key=key,outer=key in outer_keys,**p))
    costs=pd.DataFrame(costs);trials=pd.DataFrame(trials)
    for frame,name in ((predictions,"predictions"),(by,"by_source"),(summary,"summary"),
        (pd.DataFrame(selection_rows),"selected_designs"),(costs,"calibration_costs"),(trials,"calibration_trials")):
        frame.to_csv(OUT/(name+".csv"),index=False)
    energy.write(OUT/"outer_parameter_diagnostics.json",diagnostics)
    energy.write(OUT/"prediction_failures.json",failures)
    pd.DataFrame(objective_checks).to_csv(OUT/"anchor_objective_checks.csv",index=False)
    annotations=[]
    for r in predictions.itertuples():
        c=cal(r.calibration_key);base=base_registry[c["base_path"]]
        mode=protocol["methods"][r.method]["mode"]
        physically_updated=mode in ("Tf","E","Tf_E")
        state=next(x for x in c["predictions"] if x["obs_id"]==r.obs_id)
        dg=state.get("diagnostics",{})
        inp=by_obs.loc[r.obs_id]
        annotations.append(dict(task=r.task,method=r.method,budget=int(r.budget),sample_id=r.sample_id,
            obs_id=r.obs_id,role=r.role,internal_parameter_update=physically_updated,
            raw_physical_flag=r.physical_valid,
            flag_scope="numerical/electronHall only" if physically_updated else "empirical output adjustment, no recalculated physical state",
            above_reference_floor=dg.get("above_carrier_detection_floor"),
            chemical_dose_cm3=inp.incorporated_cm3,above_Si_Mott_reference=bool(inp.incorporated_cm3>2.4e18),
            Mott_scope="Si reference only,not validated Sn threshold",
            H_status="reported_detected_and_supplied" if inp.H_known else "unknown_conditional_zero",
            carbon_detected_unmodeled=bool(inp.carbon_known),
            calibration_status=r.calibration_status,
            prediction_above_reference_site_density=bool(pd.notna(r.predicted) and r.predicted>np.log10(1.91e22))))
    pd.DataFrame(annotations).to_csv(OUT/"canonical_prediction_annotations.csv",index=False)
    # Explicit complete paired-source comparisons; these are descriptive with
    # only five RT sources or two temperature sources, not significance tests.
    pairs=[]
    for task,scope in (("primary","all_query"),("temperature","common40_125")):
        z=by[by.task.eq(task)&by.scope.eq(scope)]
        for learner,baseline in (("shared_joint_Tf_E","shared_joint_offset"),
            ("shared_joint_Tf_E","fixed_joint_Tf_E"),("shared_joint_Tf_E","direct_joint_affine"),
            ("shared_joint_Tf_E","original_RT_offset"),("original_RT_Tf","original_RT_offset")):
            for budget in (0,1,2):
                a=z[z.method.eq(learner)&z.budget.eq(budget)].set_index("family").MAE
                b=z[z.method.eq(baseline)&z.budget.eq(budget)].set_index("family").MAE
                delta=a-b
                pairs.append(dict(task=task,budget=budget,method=learner,baseline=baseline,sources=len(delta),
                    better=int((delta< -1e-10).sum()),tied=int((abs(delta)<=1e-10).sum()),
                    worse=int((delta>1e-10).sum()),mean_difference=float(delta.mean())))
    pd.DataFrame(pairs).to_csv(OUT/"paired_source_comparisons.csv",index=False)
    protected=[]
    for name in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json", "artifact_manifest_phase34.json",
                 "artifact_manifest_phase56.json","artifact_manifest_phase7.json","artifact_manifest_phase8.json","artifact_manifest_phase9.json"):
        f=json.loads((BASE/name).read_text());entries=f["files"] if isinstance(f,dict) else f
        changed=[r["path"] for r in entries if not (ROOT/r["path"]).is_file() or energy.sha(ROOT/r["path"])!=r["sha256"]]
        assert not changed,changed
        protected.append(dict(manifest=name,checked=len(entries),changed=changed))
    newfits=[json.loads(p.read_text()) for p in (OUT/"base_fits").glob("*.json")]
    base_trials=[t for f in newfits for t in f["trials"]]
    checks=json.loads((OUT/"numerical_gates.json").read_text())
    audit=dict(passed=True,created_utc=energy.now(),goal_status="active",goal_complete=False,
        previous_goal_turn="progress",current_turn="progress",main_endpoint="actualdose-to-RT Hall29specimens5DOI",
        methods=len(protocol["methods"]),specimens=29,temperature_specimens=7,temperature_sources=2,new_specimens=0,
        numerical_checks=len(checks["checks"]),zero_update_replays=len(checks["replays"]),
        candidate_inner_evaluations=candidate_evals,source_and_sample_lineage_checks=lineage_checks,
        saved_outer_rows=len(predictions),calibration_records=len(files),outer_calibrations=len(outer_keys),
        new_base_fit_records=len(newfits),new_base_optimizer_starts=len(base_trials),
        new_base_nonconverged_starts=sum(not t["success"] for t in base_trials),
        new_base_state_failures=sum(len(f["state_failures"]) for f in newfits),
        summed_new_base_fit_seconds=sum(f["seconds"] for f in newfits),
        local_optimizer_starts=len(trials),nonconverged_local_starts=int((~trials.success).sum()) if len(trials) else 0,
        selected_nonconverged_local_starts=int(((~trials.success)&trials.selected).sum()) if len(trials) else 0,
        selected_outer_nonconverged_starts=int(((~trials.success)&trials.selected&trials.outer).sum()) if len(trials) else 0,
        calibration_status_counts=costs.status.value_counts().to_dict(),summed_calibration_seconds=float(costs.seconds.sum()),
        selected_outer_status_counts=costs[costs.outer].status.value_counts().to_dict(),
        local_state_failures=int(costs.state_failures.sum()),
        successful_anchor_vector_evaluations=int(costs.forward_calls.sum()),
        root_count_scope="callback vector evaluations counted;perhot/cold root iterations not instrumented;actual seconds separate",
        anchor_penalized_objective_worsened=sum(r["worsened"] for r in objective_checks),
        selected_outer_anchor_objective_worsened=sum(r["worsened"] for r in objective_checks if r["outer"]),
        failed_query_predictions=sum(not p.numeric for p in predictions.itertuples() if p.role=="query"),
        training_only_hyperselection_verified=True,temperature_inner_full_specimen_exclusion_verified=True,
        selection_scores_roles_anchors_and_output_records_reconstructed=True,
        score_scope="sourceequal primary,source/specimenequal temperature;completecoverage scores,missing predictions never dropped",
        prior_scope="trainselected scales/shrinkage,localGNdiagnostics not microscopic identification or posteriorCI",
        current_predictor="v4unchanged;no validated lowTmobility/completechain ornewdopant ranking",
        evidence_scope="retrospective exposed cohort;2Si temperature DOI only;nonewDFT/MD/materialexperiment",
        previous_protected=protected,sufficient_to_mark_goal_complete=False)
    energy.write(OUT/"contract_audit.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k!="previous_protected"},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
