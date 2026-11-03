"""Reconstruct source membership, objective, inner selection and predictions."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase12_prediction as study

OUT=study.OUT;FINAL=OUT/"physical_scope_v2";e=study.energy;core=study.core


def inf(x):return np.inf if x is None else float(x)


def main():
    protocol=json.loads((OUT/"prediction_protocol.json").read_text())
    for path in (OUT/"prediction_protocol.json",study.base.NEW/"prediction_protocol.json",
                 OUT/"base_numerical_protocol.json",OUT/"adaptation_numerical_protocol.json"):
        p=json.loads(path.read_text())
        for k in ("code_hashes","input_hashes"):
            for name,sha in p.get(k,{}).items():assert e.sha(ROOT/name)==sha,name
    scope=json.loads((FINAL/"scope_protocol.json").read_text())
    assert e.sha(ROOT/"scripts/trainable_sandbox_20261017/phase12_physical_scope_v2.py")==scope["code_sha256"]
    old=json.loads((e.BASE/"artifact_manifest_phase11.json").read_text())
    assert all((ROOT/r["path"]).is_file() and e.sha(ROOT/r["path"])==r["sha256"] for r in old["files"])
    newfits={p.stem:json.loads(p.read_text()) for p in (study.base.NEW/"fits").glob("*.json")}
    assert len(newfits)==150
    costs=[];loss_errors=[]
    for f in newfits.values():
        tr=core.training_data(f["train_families"],f["method"])
        assert tr.kind.eq("RT").all() and tr.obs_id.tolist()==f["training_obs_ids"]
        assert np.allclose(tr.loss_weight,f["weight_per_observation"],atol=1e-13,rtol=0)
        meta=core.feature_metadata(f["train_families"])
        for name in ("logC_center","logC_scale"):assert abs(meta[name]-f["feature_metadata"][name])<1e-12
        assert meta["training_specimens"]==f["feature_metadata"]["training_specimens"]
        best=min((t for t in f["trials"] if t.get("loss") is not None),key=lambda t:t["loss"])
        pred={r["obs_id"]:r["predicted"] for r in f["predictions"]}
        vals=np.array([pred[x] for x in tr.obs_id],float)
        loss=float(tr.loss_weight.to_numpy(float)@((vals-tr.target.to_numpy(float))**2)+
            f["config"].get("reg",0.)*np.sum(np.array(f["parameters"])[2:]**2))
        err=abs(loss-best["loss"]);assert err<2e-7*(1+loss),(f["key"],err)
        assert np.allclose(f["parameters"],best["parameters"],atol=0,rtol=0)
        loss_errors.append(err)
        costs.append(dict(fit_key=f["key"],method=f["method"],seconds=f["seconds"],starts=len(f["trials"]),
            nonconverged=sum(not t["success"] for t in f["trials"]),best_nonconverged=not best["success"],
            state_failures=len(f["state_failures"]),**f["counters"]))
    episodes={p.stem:json.loads(p.read_text()) for p in (OUT/"episodes").glob("*.json")}
    assert len(episodes)==900
    fixes={p.stem:json.loads(p.read_text()) for p in (FINAL/"episodes").glob("*.json")}
    fitmap={};profile_checks=0;offset_checks=0;lineage=0;candidate_checks=0
    for ep in episodes.values():
        ref=ep["base_reference"];path=ROOT/ref["path"];assert e.sha(path)==ref["sha256"]
        f=json.loads(path.read_text());fitmap[f["key"]]=f
        assert f["key"]==ep["base_key"] and ep["held"] not in f["train_families"]
        d=core.PRED[core.PRED.family.eq(ep["held"])&core.PRED.kind.eq("RT")]
        pred={r["obs_id"]:r["predicted"] for r in f["predictions"]}
        assert len(ep["results"])==15
        for r in ep["results"]:
            budget=r["budget"];lam=inf(r["lam"])
            role=core.ROLES[core.ROLES.family.eq(ep["held"])&core.ROLES.budget.eq(budget)].set_index("sample_id").role
            a=d[d.sample_id.map(role).eq("anchor")]
            assert r["anchors"]==a.sample_id.tolist() and len(a)==budget
            assert np.allclose(r["anchor_observed"],a.target,atol=1e-13,rtol=0)
            assert set(x["obs_id"] for x in r["records"])==set(d.obs_id)
            for row in r["records"]:
                inp=d[d.obs_id.eq(row["obs_id"])].iloc[0]
                assert row["role"]==role[row["sample_id"]] and row["family"]==ep["held"]
                assert abs(row["observed"]-inp.target)<1e-13
                lineage+=1
            if r["profile_candidates"]:
                finite=[]
                assert sum(c["kind"]=="grid" for c in r["profile_candidates"])==25
                for c in r["profile_candidates"]:
                    if not c["valid"]:continue
                    residual=np.asarray(c["prediction"])-a.target.to_numpy(float)
                    cost=float(residual@residual+lam*c["shift"]**2)
                    deriv=float(2*residual@np.asarray(c["jacobian"])+2*lam*c["shift"])
                    assert abs(cost-c["objective"])<2e-9*(1+cost)
                    assert abs(deriv-c["derivative"])<2e-9*(1+abs(deriv))
                    finite.append(c);profile_checks+=1
                if r["status"]=="profile_adapted":
                    best=min(finite,key=lambda c:(c["objective"],abs(c["shift"])))
                    assert best["shift"]==r["shift"] and abs(best["objective"]-r["diagnostics"]["objective"])<1e-13
                else:assert not finite and r["status"]=="no_feasible_anchor_state"
            if ep["kind"]=="offset" and budget and np.isfinite(lam) and r["anchor_complete"]:
                raw=np.array([pred[x] for x in a.obs_id],float)
                expected=float(np.sum(raw-a.target.to_numpy(float))/(budget+lam))
                assert abs(expected-r["shift"])<2e-12;offset_checks+=1
        if ep["key"] in fixes:
            corrected=fixes[ep["key"]]
            assert corrected["scope_correction"]["original_sha256"]==e.sha(OUT/"episodes"/(ep["key"]+".json"))
            assert corrected["kind"] in ("stock","Tf")
    def getep(key):return fixes[key] if key in fixes else episodes[key]
    def score(ep,budget,lam):
        r=next(x for x in ep["results"] if x["budget"]==budget and inf(x["lam"])==lam)
        query=[x for x in r["records"] if x["role"]=="query"]
        value=float(np.mean([abs(x["predicted"]-x["observed"]) for x in query])) if all(x["predicted"] is not None for x in query) else np.inf
        assert (np.isinf(value) and r["MAE"] is None) or abs(value-r["MAE"])<2e-12
        return value,r
    allrows=[];designs=[];changes=[];replay=[];selected_keys=set()
    for model in study.MODELS:
        frame=read_result_csv(FINAL/"models"/model/"predictions.csv")
        selections=json.loads((FINAL/"models"/model/"selections.json").read_text())
        original=json.loads((OUT/"models"/model/"selections.json").read_text())
        assert len(frame)==87 and len(selections)==15
        for s,oldchoice in zip(selections,original):
            outertrain=set(core.FAMILIES)-{s["held"]};budget=s["budget"]
            for c in s["candidates"]:
                values=[]
                for key in c["inner_episode_keys"]:
                    ep=getep(key)
                    assert ep["held"] in outertrain and set(ep["train_families"])==outertrain-{ep["held"]}
                    values.append(score(ep,budget,inf(c["lam"]))[0]);candidate_checks+=1
                assert len(values)==4
                mean=float(np.mean(values));assert (np.isinf(mean) and c["score"] is None) or abs(mean-c["score"])<3e-11
            best=min(s["candidates"],key=lambda c:(round(inf(c["score"]),10),-inf(c["lam"]),c["index"]))
            assert s["selected"]==best
            ep=getep(s["episode_key"]);assert ep["held"]==s["held"] and set(ep["train_families"])==outertrain
            value,r=score(ep,budget,inf(best["lam"]))
            assert (np.isinf(value) and s["outer_MAE"] is None) or abs(value-s["outer_MAE"])<2e-12
            selected_keys.add(s["fit_key"])
            byid={x["sample_id"]:x for x in r["records"]}
            rows=frame[frame.family.eq(s["held"])&frame.budget.eq(budget)]
            assert set(rows.sample_id)==set(byid)
            for row in rows.itertuples():
                q=byid[row.sample_id]
                assert row.role==q["role"] and row.fit_key==s["fit_key"]
                assert (pd.isna(row.predicted) and q["predicted"] is None) or abs(row.predicted-q["predicted"])<2e-12
            designs.append(dict(model=model,held=s["held"],budget=budget,base_fit_key=s["fit_key"],
                config=json.dumps(best["config"],sort_keys=True),lam=best["lam"],shift=r["shift"],
                status=r["status"],inner_MAE=best["score"],outer_MAE=s["outer_MAE"],
                data_resolution=r["diagnostics"].get("data_resolution"),at_boundary=r["diagnostics"].get("at_boundary")))
            if best["config"]!=oldchoice["selected"]["config"] or best["lam"]!=oldchoice["selected"]["lam"]:
                changes.append(dict(model=model,held=s["held"],budget=budget,old=oldchoice["selected"],new=best))
        allrows.append(frame)
    frame=pd.concat(allrows,ignore_index=True);assert len(frame)==1740
    common=set(core.ROLES[core.ROLES.budget.eq(2)&core.ROLES.role.eq("query")].sample_id)
    by=[]
    for (model,budget,family),g in frame[frame.role.eq("query")].groupby(["model","budget","family"]):
        for scope in ("all_query","common18"):
            d=g if scope=="all_query" else g[g.sample_id.isin(common)]
            err=abs(d.predicted-d.observed)
            by.append(dict(model=model,budget=int(budget),family=family,scope=scope,queries=len(d),
                finite_queries=int(np.isfinite(err).sum()),MAE=float(err.mean()) if np.isfinite(err).all() else np.inf))
    by=pd.DataFrame(by);summary=by.groupby(["model","budget","scope"]).agg(MAE=("MAE","mean"),
        worst_source=("MAE","max"),sources=("family","nunique"),queries=("queries","sum"),finite_queries=("finite_queries","sum")).reset_index()
    # Explicitly refuse any mean that pandas might otherwise drop as nonfinite.
    for i,row in summary.iterrows():
        group=by[by.model.eq(row.model)&by.budget.eq(row.budget)&by.scope.eq(row.scope)]
        if not np.isfinite(group.MAE).all():summary.loc[i,["MAE","worst_source"]]=np.inf
    pairs=[]
    for method,baseline in (("original_stock","original_offset"),("shared_RT_stock","shared_RT_offset"),
        ("shared_RT_stock","original_stock"),("shared_RT_offset","matched_output_offset"),
        ("shared_RT_offset","matched_Tf_offset"),("shared_joint_stock","shared_joint_offset"),
        ("joint_stock","joint_offset")):
        for budget in (0,1,2):
            for scope in ("all_query","common18"):
                a=by[by.model.eq(method)&by.budget.eq(budget)&by.scope.eq(scope)].set_index("family").MAE
                b=by[by.model.eq(baseline)&by.budget.eq(budget)&by.scope.eq(scope)].set_index("family").MAE
                diff=a-b
                pairs.append(dict(model=method,baseline=baseline,budget=budget,scope=scope,
                    better=int((diff< -1e-10).sum()),tied=int((abs(diff)<=1e-10).sum()),worse=int((diff>1e-10).sum()),
                    difference=float(diff.mean()) if np.isfinite(diff).all() else None))
    costs=pd.DataFrame(costs)
    for data,name in ((frame,"predictions"),(by,"by_source"),(summary,"summary"),(pd.DataFrame(designs),"selected_designs"),
                      (costs,"base_costs"),(pd.DataFrame(pairs),"paired_comparisons")):
        data.to_csv(FINAL/(name+".csv"),index=False)
    e.write(FINAL/"selection_changes.json",changes)
    selectedbad=costs[costs.best_nonconverged&costs.fit_key.isin(selected_keys)]
    audit=dict(passed=True,created_utc=e.now(),models=len(study.MODELS),new_base_fits=len(newfits),new_base_starts=int(costs.starts.sum()),
        new_base_nonconverged_starts=int(costs.nonconverged.sum()),new_base_best_nonconverged=int(costs.best_nonconverged.sum()),
        new_base_outer_selected_nonconverged=int(len(selectedbad)),new_base_state_failures=int(costs.state_failures.sum()),
        new_base_objective_reconstruction_error=max(loss_errors),episodes=len(episodes),profile_candidate_objectives_checked=profile_checks,
        closed_offset_checks=offset_checks,saved_episode_lineage_checks=lineage,inner_candidate_evaluations=candidate_checks,
        outer_rows=len(frame),failed_outer_queries=int(frame[frame.role.eq("query")].predicted.isna().sum()),
        scope_corrected_prediction_entries=json.loads((FINAL/"execution.json").read_text())["changed_prediction_entries"],
        V2_selection_changes=len(changes),old_phase11_artifacts_checked=len(old["files"]),old_artifacts_unchanged=True,
        original_primary_and_common18_preserved=True,goal_complete=False,
        scope="data/selection/objective contract;full physical-output runtime replay performed separately;not all optimization starts or material advantage proved")
    e.write(FINAL/"contract_audit.json",audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(summary[summary.scope.eq("all_query")].pivot(index="model",columns="budget",values="MAE").round(6).to_string())


if __name__=="__main__":main()
