"""Independently audit all source-coordinate objectives and nested choices."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import phase15_source_adaptation as bench
import phase2_energy as energy
from csv_io_v2 import read_result_csv

OUT=bench.OUT;DEST=OUT/"validation"


def num(v):return np.inf if v is None else float(v)


def main():
    assert (OUT/"execution_complete.json").exists();bench.freeze()
    bank=bench.bank();eps={};costs=[];objective_errors=[];closed_errors=[];candidate_count=0;offset_replay=[]
    for path in (OUT/"episodes").glob("*.json"):
        ep=json.loads(path.read_text());eps[ep["key"]]=ep;f=bank[ep["fit_key"]]
        assert ep["held"] not in f["train_families"] and set(ep["train_families"])==set(f["train_families"])
        assert ep["fit_origin"]==f["composite_origin"]
        assert energy.sha(ROOT/ep["fit_origin"]["path"])==ep["fit_origin"]["sha256"]
        assert len(ep["results"])==15
        for r in ep["results"]:
            b=r["budget"];lam=num(r["lam"]);z=r["shift"]
            roles=bench.core.ROLES[bench.core.ROLES.family.eq(ep["held"])&bench.core.ROLES.budget.eq(b)].set_index("sample_id").role
            ids=[s for s in roles.index if roles[s]=="anchor"]
            assert set(ids)==set(r["anchor_samples"]) and len(ids)==b
            truth=bench.core.DATA.set_index("sample_id").observed_log_nH
            yy=np.array([truth[s] for s in r["anchor_samples"]]);assert np.max(abs(yy-np.array(r["anchor_observed"])),initial=0)<1e-12
            records=pd.DataFrame(r["records"])
            assert set(records.sample_id)==set(roles.index)
            for x in records.itertuples():
                assert x.role==roles[x.sample_id] and abs(x.observed-truth[x.sample_id])<1e-12
                if x.predicted is not None and np.isfinite(x.predicted):
                    assert abs(x.predicted_log_sigma-(np.log10(1.602176634e-19)+x.predicted+x.predicted_log_mu_H))<1e-10
                    assert x.hall_factor>=1
            if b==0 or np.isinf(lam):assert z==0
            elif ep["kind"]=="offset":
                rr=records.set_index("sample_id").loc[r["anchor_samples"]]
                expected=float(np.sum(rr.raw_prediction.to_numpy(float)-yy)/(b+lam))
                closed_errors.append(abs(expected-z));assert abs(expected-z)<1e-10
                if r["anchor_complete"]:
                    err=abs(records.predicted-records.raw_prediction+z).max()
                    closed_errors.append(float(err));assert err<1e-9
            else:
                finite=[]
                for c in r["profile_candidates"]:
                    if not c["valid"]:continue
                    pred=np.array(c["prediction"]);jac=np.array(c["jacobian"]);zz=c["shift"]
                    obj=float((pred-yy)@(pred-yy)+lam*zz*zz)
                    derivative=float(2*(pred-yy)@jac+2*lam*zz)
                    error=max(abs(obj-c["objective"]),abs(derivative-c["derivative"]));objective_errors.append(error)
                    assert error<1e-8;finite.append(c);candidate_count+=1
                if finite:
                    best=min(finite,key=lambda c:(c["objective"],abs(c["shift"])))
                    assert abs(z-best["shift"])<1e-12
                    expected_I=float(np.array(best["jacobian"])@np.array(best["jacobian"]))
                    resolution=expected_I/(expected_I+lam) if expected_I+lam>0 else 0.
                    assert abs(resolution-r["diagnostics"]["data_resolution"])<1e-10
                    got=records.set_index("sample_id").loc[r["anchor_samples"]].predicted.to_numpy(float)
                    assert np.max(abs(got-np.array(best["prediction"])))<1e-9
                else:assert not r["anchor_complete"]
            query=records[records.role.eq("query")];error=abs(query.predicted.astype(float)-query.observed)
            score=float(error.mean()) if error.notna().all() else np.inf
            assert (np.isinf(score) and np.isinf(num(r["MAE"]))) or abs(score-num(r["MAE"]))<1e-11
        costs.append(dict(episode=ep["key"],method=ep["method"],kind=ep["kind"],held=ep["held"],seconds=ep["seconds"],**ep["counters"]))
    assert len(eps)==450
    rows=[];choices=[];inner_count=0
    for variant,(method,kind) in bench.VARIANTS.items():
        source=OUT/"variants"/variant;selections=json.loads((source/"selections.json").read_text())
        csv=read_result_csv(source/"predictions.csv");assert len(csv)==87
        for s in selections:
            train=set(bench.core.FAMILIES)-{s["held"]};candidates=[]
            for c in s["candidates"]:
                values=[];seen=[]
                for k in c["inner_episode_keys"]:
                    ep=eps[k];inner=list(train-set(ep["train_families"]));assert len(inner)==1 and ep["held"]==inner[0]
                    assert s["held"] not in ep["train_families"] and ep["method"]==method and ep["kind"]==kind
                    seen.append(inner[0]);values.append(num(bench.choose(ep,s["budget"],num(c["lam"]))["MAE"]));inner_count+=1
                assert set(seen)==train
                score=float(np.mean(values));assert (np.isinf(score) and np.isinf(num(c["score"]))) or abs(score-num(c["score"]))<1e-11
                candidates.append(dict(lam=c["lam"],score=score))
            best=min(candidates,key=lambda c:(round(c["score"],10),-num(c["lam"])))
            assert best["lam"]==s["selected"]["lam"]
            ep=eps[s["outer_episode_key"]];assert set(ep["train_families"])==train and ep["held"]==s["held"]
            selected=bench.choose(ep,s["budget"],num(best["lam"]))
            assert abs(selected["shift"]-s["shift"])<1e-12
            g=csv[csv.family.eq(s["held"])&csv.budget.eq(s["budget"])]
            for r in selected["records"]:
                x=g[g.sample_id.eq(r["sample_id"])].iloc[0]
                assert (r["predicted"] is None and np.isnan(x.predicted)) or abs(x.predicted-r["predicted"])<1e-11
            rows.extend(g.to_dict("records"));choices.append(dict(variant=variant,held=s["held"],budget=s["budget"],
                source_kind=kind,base_method=method,lam=best["lam"],shift=selected["shift"],MAE=num(selected["MAE"]),
                at_boundary=selected["diagnostics"].get("at_boundary",False),resolution=selected["diagnostics"].get("data_resolution")))
        if kind=="offset":
            reference=read_result_csv(bench.ref.OUT/"validation/primary_predictions.csv")
            ref=reference[reference.method.eq(method)].set_index(["family","budget","sample_id"])
            for r in csv.itertuples():offset_replay.append(abs(r.predicted-ref.loc[(r.family,r.budget,r.sample_id),"predicted"]))
    df=pd.DataFrame(rows);assert len(df)==1566
    common=set(bench.core.ROLES[bench.core.ROLES.budget.eq(2)&bench.core.ROLES.role.eq("query")].sample_id);by=[]
    for (v,b,f),g in df[df.role.eq("query")].groupby(["variant","budget","family"]):
        for scope in ("all_query","common18"):
            h=g if scope=="all_query" else g[g.sample_id.isin(common)];errors=abs(h.predicted-h.observed)
            by.append(dict(variant=v,budget=int(b),family=f,scope=scope,MAE=float(errors.mean()) if errors.notna().all() else np.inf,
                queries=len(h),finite_queries=int(errors.notna().sum())))
    by=pd.DataFrame(by);summary=by.groupby(["variant","budget","scope"]).agg(MAE=("MAE","mean"),worst_source=("MAE","max"),
        queries=("queries","sum"),finite_queries=("finite_queries","sum")).reset_index()
    DEST.mkdir(exist_ok=True)
    for frame,name in ((df,"predictions"),(by,"by_source"),(summary,"summary"),(pd.DataFrame(costs),"costs"),(pd.DataFrame(choices),"selected_designs")):
        frame.to_csv(DEST/(name+".csv"),index=False)
    assert max(offset_replay)<1e-7
    energy.write(DEST/"contract_audit.json",dict(passed=True,created_utc=energy.now(),episodes=450,variants=18,
        profile_candidate_objectives=candidate_count,closed_checks=len(closed_errors),inner_candidate_evaluations=inner_count,
        outer_saved_rows=1566,missing_query_rows=int(df[df.role.eq("query")].predicted.isna().sum()),
        maximum_profile_objective_error=max(objective_errors,default=0),maximum_output_replay_error=max(offset_replay),
        scope="sourcefit/selection/coverage contract;notglobaloptimum or uniquephysicalprocess",goal_complete=False))
    print("Source contract passed;episodes450/variants18;querymissing",int(df[df.role.eq("query")].predicted.isna().sum()))
    print(summary[summary.scope.eq("all_query")].pivot(index="variant",columns="budget",values="MAE").to_string())


if __name__=="__main__":main()
