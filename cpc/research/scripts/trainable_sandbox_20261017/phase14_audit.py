"""Independently reconstruct fit/selection/coverage contracts and metrics."""
from pathlib import Path
import sys,json,itertools
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv
import phase14_prediction as adapter
study=adapter.core
import phase2_energy as energy

OUT=adapter.OUT;DEST=OUT/"validation"


def number(v):return np.inf if v is None else float(v)


def expected_training(f):
    allrows=study.PRED[study.PRED.family.isin(f["train_families"])]
    rt=allrows[allrows.kind.eq("RT")].copy();rt["loss_weight"]=0.
    for family,g in rt.groupby("family"):rt.loc[g.index,"loss_weight"]=1/(rt.family.nunique()*len(g))
    if study.METHODS[f["method"]]["training"]=="RT":return rt
    c=allrows[allrows.kind.eq("curve")&allrows.Tobs_K.le(275)].copy();c["loss_weight"]=0.
    for family,g in c.groupby("family"):
        for sample,s in g.groupby("sample_id"):
            c.loc[s.index,"loss_weight"]=.1/(c.family.nunique()*g.sample_id.nunique()*len(s))
    return pd.concat([rt,c],ignore_index=True).sort_values(["family","sample_id","Tobs_K","kind"])


def features(d):
    c=np.log10(d.incorporated_cm3.to_numpy(float))-18;t=d.Tobs_K.to_numpy(float)
    g=(d.growth_T_C.to_numpy(float)+273.15-1100)/300
    hk=d.H_known.to_numpy(float);ck=d.carbon_known.to_numpy(float)
    h=np.log10(np.maximum(d.H_cm3.to_numpy(float),1))-17
    carbon=np.log10(np.maximum(d.carbon_cm3.to_numpy(float),1))-17
    return np.c_[c,c*c,g,d.element.eq("Sn").to_numpy(float),np.log(t/300),300/t-1,c*(300/t-1),hk,h*hk,ck,carbon*ck]


def primary(f,held,budget,lam):
    d=study.DATA[study.DATA.family.eq(held)]
    roles=study.ROLES[study.ROLES.family.eq(held)&study.ROLES.budget.eq(budget)].set_index("sample_id").role
    lookup={r["obs_id"]:r for r in f["predictions"]}
    raw=np.array([np.nan if lookup["RT::"+s]["predicted"] is None else lookup["RT::"+s]["predicted"] for s in d.sample_id])
    anchor=np.array([roles[s]=="anchor" for s in d.sample_id]);query=np.array([roles[s]=="query" for s in d.sample_id])
    y=d.observed_log_nH.to_numpy(float);delta=0.
    if anchor.any() and np.isfinite(lam):
        if not np.isfinite(raw[anchor]).all():raw[:]=np.nan
        else:delta=float(np.sum(y[anchor]-raw[anchor])/(anchor.sum()+lam))
    pred=raw+delta;score=float(abs(pred[query]-y[query]).mean()) if np.isfinite(pred[query]).all() else np.inf
    return score,dict(zip(d.sample_id,pred)),delta


def summarize(d,temperature=False):
    rows=[]
    common=set(study.ROLES[study.ROLES.budget.eq(2)&study.ROLES.role.eq("query")].sample_id)
    scopes=("all_query","common40_125") if temperature else ("all_query","common18")
    for (m,b,f),g in d[d.role.eq("query")].groupby(["method","budget","family"]):
        for scope in scopes:
            h=g
            if scope=="common18":h=g[g.sample_id.isin(common)]
            if scope=="common40_125":h=g[g.Tobs_K.between(40,125)]
            if not len(h):continue
            errors=abs(h.predicted-h.observed)
            # Curves are repeated reads: average within specimen, then source.
            mae=float(errors.groupby(h.sample_id).mean().mean()) if errors.notna().all() else np.inf
            rows.append(dict(method=m,budget=int(b),family=f,scope=scope,MAE=mae,
                queries=len(h),finite_queries=int(errors.notna().sum()),specimens=h.sample_id.nunique()))
    by=pd.DataFrame(rows);summary=by.groupby(["method","budget","scope"]).agg(
        MAE=("MAE","mean"),worst_source=("MAE","max"),sources=("family","nunique"),
        queries=("queries","sum"),finite_queries=("finite_queries","sum"),specimens=("specimens","sum")).reset_index()
    return by,summary


def main():
    assert (OUT/"training_execution_complete.json").exists() and (OUT/"selection_execution_complete.json").exists()
    adapter.freeze();files=list(adapter.composite().values());assert len(files)==450
    fits={};costs=[];objective_error=[];empirical_errors=[];allstarts=0;unconverged=0;best_failed=set()
    for f in files:
        fits[f["key"]]=f
        assert f["definition"]==study.METHODS[f["method"]]
        assert len(f["train_families"]) in (3,4) and len(set(f["train_families"]))==len(f["train_families"])
        expected=expected_training(f);assert expected.obs_id.tolist()==f["training_obs_ids"]
        assert set(expected.sample_id)==set(f["training_specimens"])
        w=expected.loss_weight.to_numpy(float);assert np.max(abs(w-np.array(f["weight_per_observation"])))<1e-14
        assert set(expected.family)==set(f["train_families"])
        lookup={r["obs_id"]:r for r in f["predictions"]};assert set(lookup)==set(study.PRED.obs_id)
        raw=np.array([np.nan if r["predicted"] is None else r["predicted"] for r in f["predictions"]])
        field=f["definition"].get("empirical");meta=f["metadata"]
        if not field:
            allstarts+=len(f["trials"]);unconverged+=sum(not t["success"] for t in f["trials"])
            finite=[t for t in f["trials"] if t["loss"] is not None]
            if finite:
                best=min(finite,key=lambda t:t["loss"]);assert np.array_equal(best["parameters"],f["parameters"])
                yhat=np.array([lookup[o]["predicted"] for o in expected.obs_id],float)
                obj=float(w@((yhat-expected.target.to_numpy(float))**2))
                objective_error.append(abs(obj-best["loss"]));assert np.isfinite(obj)
                if not best["success"]:best_failed.add(f["key"])
        elif field=="ridge":
            X=features(expected);center=np.average(X,axis=0,weights=w)
            scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1
            wr=w*len(expected)/w.sum();Z=(X-center)/scale;yy=expected.target.to_numpy(float)
            intercept=np.average(yy,weights=wr)
            coef=np.linalg.solve(Z.T@(wr[:,None]*Z)+f["config"]["alpha"]*np.eye(Z.shape[1]),Z.T@(wr*(yy-intercept)))
            for v in (abs(center-np.array(meta["center"])).max(),abs(scale-np.array(meta["scale"])).max(),
                abs(coef-np.array(meta["coefficients"])).max(),abs(intercept-meta["intercept"])):empirical_errors.append(float(v))
            replay=((features(study.PRED)-center)/scale)@coef+intercept
            empirical_errors.append(float(abs(replay-raw).max()))
        elif field=="dose":
            X=np.c_[np.ones(len(expected)),np.log10(expected.incorporated_cm3)-18]
            coef=np.linalg.lstsq(X*np.sqrt(w)[:,None],expected.target.to_numpy(float)*np.sqrt(w),rcond=None)[0]
            replay=coef[0]+coef[1]*(np.log10(study.PRED.incorporated_cm3)-18)
            empirical_errors.append(float(abs(replay-raw).max()))
        else:
            replay=np.log10(study.PRED.incorporated_cm3.to_numpy(float)+(study.PRED.H_cm3 if field=="identityH" else 0.))
            if field=="activation":
                residual=expected.target.to_numpy(float)-np.log10(expected.incorporated_cm3.to_numpy(float))
                order=np.argsort(residual,kind="stable");ii=np.searchsorted(np.cumsum(w[order]),.5*sum(w),side="left")
                replay+=residual[order[ii]]
            empirical_errors.append(float(abs(replay-raw).max()))
        costs.append(dict(key=f["key"],method=f["method"],training_sources=len(f["train_families"]),
            starts=len(f["trials"]),unconverged=sum(not t["success"] for t in f["trials"]),
            intermediate_state_failures=len(f["state_failures"]),seconds=f["seconds"],
            **f["counters"],**{"training_"+k:v for k,v in f["training_counters"].items()}))
    assert max(objective_error,default=0)<2e-7 and max(empirical_errors,default=0)<1e-7
    rows=[];temps=[];selected=[];candidate_count=0;outer_failed=0;maxreplay=0.;selected_keys=set()
    for method in study.METHODS:
        folder=OUT/"methods"/method;choices=json.loads((folder/"selections.json").read_text())
        saved=read_result_csv(folder/"primary_predictions.csv");t=read_result_csv(folder/"temperature_predictions.csv")
        assert len(choices)==15 and len(saved)==87
        for choice in choices:
            held,budget=choice["held"],choice["budget"];train=set(study.FAMILIES)-{held};candidates=[]
            for c in choice["candidates"]:
                scores=[];seen=[]
                for key in c["inner_keys"]:
                    f=fits[key];inner=list(train-set(f["train_families"]));assert len(inner)==1
                    assert held not in f["train_families"] and inner[0] not in f["train_families"]
                    seen.append(inner[0]);scores.append(primary(f,inner[0],budget,number(c["lam"]))[0]);candidate_count+=1
                assert set(seen)==train and len(seen)==4
                score=float(np.mean(scores));oldscore=number(c["score"])
                assert (np.isinf(score) and np.isinf(oldscore)) or abs(score-oldscore)<1e-11
                candidates.append(dict(**c,replayed_score=score))
            best=min(candidates,key=lambda c:(round(c["replayed_score"],10),-number(c["lam"]),c["index"]))
            assert best["config"]==choice["selected"]["config"] and best["lam"]==choice["selected"]["lam"]
            f=fits[choice["fit_key"]];assert set(f["train_families"])==train and f["method"]==method
            selected_keys.add(f["key"]);outer_failed+=f["key"] in best_failed
            score,preds,delta=primary(f,held,budget,number(best["lam"]))
            g=saved[saved.family.eq(held)&saved.budget.eq(budget)].copy()
            rolemap=study.ROLES[study.ROLES.family.eq(held)&study.ROLES.budget.eq(budget)].set_index("sample_id").role
            obsmap=study.DATA.set_index("sample_id").observed_log_nH
            for r in g.itertuples():
                assert r.role==rolemap[r.sample_id] and abs(r.observed_log_nH-obsmap[r.sample_id])<1e-12
                wanted=preds[r.sample_id];v=r.predicted_log_nH
                assert (np.isnan(v) and np.isnan(wanted)) or abs(v-wanted)<1e-11
                if np.isfinite(v):maxreplay=max(maxreplay,abs(v-wanted))
                rr=r._asdict();rr.update(predicted=v,observed=r.observed_log_nH,raw_physical_valid=r.physical_valid,
                    same_physical_state_for_output=bool(f["definition"].get("empirical") is None and not f["definition"].get("output") and delta==0),
                    observation_scope="singleitinerantchannel Hall assumption;outputoffset empirical ifnonzero")
                if delta!=0 or f["definition"].get("output"):rr["physical_valid"]=None
                rows.append(rr)
            selected.append(dict(method=method,held=held,budget=budget,fit_key=f["key"],config=best["config"],
                lam=best["lam"],offset=delta,MAE=score,inner_MAE=best["replayed_score"],parameters=f["parameters"],
                selected_best_converged=f["key"] not in best_failed))
        # Independent temperature anchors and offset reconstruction.
        for (sample,budget),g in t.groupby(["sample_id","budget"]):
            held=g.family.iloc[0];choice=next(c for c in choices if c["held"]==held and c["budget"]==budget)
            f=fits[choice["fit_key"]];lookup={r["obs_id"]:r for r in f["predictions"]}
            source=study.PRED[study.PRED.sample_id.eq(sample)]
            expected_anchors=[source[source.kind.eq("RT")].iloc[0]] if budget else []
            if budget==2:
                eligible=source[source.kind.eq("curve")&source.Tobs_K.between(125,175)]
                ii=abs(eligible.Tobs_K-150).sort_values(kind="stable").index[0]
                expected_anchors.append(source.loc[ii])
            aid={r.obs_id for r in expected_anchors};assert set(g[g.role.eq("anchor")].obs_id)==aid
            source=source.set_index("obs_id")
            a=g[g.role.eq("anchor")];lam=number(choice["selected"]["lam"]);delta=0.
            complete=all(lookup[o]["predicted"] is not None for o in a.obs_id)
            if len(a) and np.isfinite(lam) and complete:
                delta=sum(r.observed-lookup[r.obs_id]["predicted"] for r in a.itertuples())/(len(a)+lam)
            for r in g.itertuples():
                observed=source.loc[r.obs_id]
                role="unavailable_reference" if observed.kind=="RT" else "query"
                if r.obs_id in aid:role="anchor"
                elif budget and abs(r.Tobs_K-300)<=25:role="guard"
                if budget==2 and abs(r.Tobs_K-expected_anchors[1].Tobs_K)<=10 and r.obs_id not in aid:role="guard"
                assert r.role==role and abs(r.observed-observed.target)<1e-12 and abs(r.Tobs_K-observed.Tobs_K)<1e-10
                raw=lookup[r.obs_id]["predicted"]
                wanted=np.nan if raw is None or not complete else raw+delta
                assert (np.isnan(wanted) and np.isnan(r.predicted)) or abs(wanted-r.predicted)<1e-11
                rr=r._asdict();rr.update(same_physical_state_for_output=bool(not f["definition"].get("empirical") and not f["definition"].get("output") and delta==0))
                temps.append(rr)
    df=pd.DataFrame(rows);tf=pd.DataFrame(temps);assert len(df)==1740 and df.method.nunique()==20
    by,summary=summarize(df);tb,ts=summarize(tf,True)
    for frame,name in ((df,"primary_predictions"),(tf,"temperature_predictions"),(by,"primary_by_source"),
        (summary,"primary_summary"),(tb,"temperature_by_source"),(ts,"temperature_summary"),
        (pd.DataFrame(costs),"costs"),(pd.DataFrame(selected),"selected_designs")):
        (DEST).mkdir(exist_ok=True);frame.to_csv(DEST/(name+".csv"),index=False)
    pred_fail=int(df[df.role.eq("query")].predicted.isna().sum())
    energy.write(DEST/"contract_audit.json",dict(passed=True,created_utc=energy.now(),new_fit_records=45,inherited_fit_records=405,
        physical_fit_records=210,optimizer_starts=allstarts,unconverged_starts=unconverged,
        inherited_actual_optimizer_starts=333,new_optimizer_starts=90,actual_total_optimizer_starts=423,
        originalV1_results_retained=True,composite_records=450,
        best_fit_nonconverged=len(best_failed),selected_outer_nonconverged=outer_failed,
        maximum_objective_reconstruction_error=max(objective_error,default=0),
        maximum_empirical_reconstruction_error=max(empirical_errors,default=0),
        intermediate_state_failures=sum(r["intermediate_state_failures"] for r in costs),
        nested_candidate_inner_evaluations=candidate_count,outer_saved_rows=len(df),temperature_saved_rows=len(tf),
        selected_unique_basefits=len(selected_keys),missing_primary_queries=pred_fail,
        original_roles_and_common18_preserved=True,source_exclusion_verified=True,
        output_scope_annotation="V1numericpredictions preserved;validationCSVseparatesrawstate from empiricaloffset,not a newsolvedstate",
        numerical_access_guard_not_this_audit=True,goal_complete=False,
        scope="fit/selection/metric contract;not globaloptimality or newmaterial replication"))
    print("Phase14 contract passed;compositefitrecords450;outerrows",len(df),"nonconverged selected",outer_failed,"missingqueries",pred_fail)
    print(summary[summary.scope.eq("all_query")].pivot(index="method",columns="budget",values="MAE").to_string())


if __name__=="__main__":main()
