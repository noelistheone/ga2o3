"""Reconstruct old-source selection and every new-source prediction/offset."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase11_external_source as ext
import phase11_external_source_v2 as repaired


def main():
    study=ext.study;energy=ext.energy
    p=json.loads((ext.OUT/"evaluation_protocol.json").read_text())
    for path,sha in p["hashes"].items():assert energy.sha(ROOT/path)==sha,path
    fix=json.loads((ext.OUT/"diagnostic_render_fix_v2_protocol.json").read_text())
    assert energy.sha(Path(repaired.__file__))==fix["code_sha256"]
    assert energy.sha(Path(ext.__file__))==fix["original_code_sha256"]
    assert energy.sha(energy.BASE/"phase11_external.log")==fix["failure_log_sha256"]
    fits={p.stem:json.loads(p.read_text()) for p in (ext.PARENT/"fits").glob("*.json")}
    full={p.stem:json.loads(p.read_text()) for p in (ext.TRAIN/"fits").glob("*.json")}
    for f in full.values():
        assert set(f["train_families"])==set(study.FAMILIES)
        assert set(f["training_specimens"])==set(study.DATA.sample_id)
        assert set(f["training_obs_ids"])<=set(study.PRED.obs_id)
        assert not any("Li2024" in x for x in f["training_obs_ids"])
        tr=study.training_data(study.FAMILIES,f["method"])
        assert f["training_obs_ids"]==tr.obs_id.tolist()
        assert np.allclose(f["weight_per_observation"],tr.loss_weight,atol=1e-13,rtol=0)
    roles=study.ROLES;data=study.DATA
    def score(f,held,budget,lam):
        frame=data[data.family.eq(held)]
        role=roles[roles.family.eq(held)&roles.budget.eq(budget)].set_index("sample_id").role
        rawmap={r["obs_id"]:r["predicted"] for r in f["predictions"]}
        vals=np.array([rawmap["RT::"+s] for s in frame.sample_id],float)
        obs=frame.observed_log_nH.to_numpy(float)
        anchor=np.array([role[s]=="anchor" for s in frame.sample_id])
        query=np.array([role[s]=="query" for s in frame.sample_id])
        delta=np.sum(obs[anchor]-vals[anchor])/(anchor.sum()+lam) if anchor.any() and np.isfinite(lam) else 0.
        return float(abs(vals[query]+delta-obs[query]).mean())
    choices=json.loads((ext.OUT/"training_only_selections.json").read_text());assert len(choices)==42
    evaluations=0
    for s in choices:
        for c in s["candidates"]:
            lam=np.inf if c["lam"] is None else c["lam"]
            values=[]
            for held,key in zip(study.FAMILIES,c["source_excluded_fit_keys"]):
                f=fits[key];assert set(f["train_families"])==set(study.FAMILIES)-{held}
                values.append(score(f,held,s["budget"],lam));evaluations+=1
            assert np.allclose(values,c["per_source_scores"],atol=3e-12,rtol=0)
            assert np.isclose(np.mean(values),c["score"],atol=3e-12,rtol=0)
        best=min(s["candidates"],key=lambda c:(round(c["score"],10),-(np.inf if c["lam"] is None else c["lam"]),c["index"]))
        assert best==s["selected"]
    assert evaluations==1890
    frame=read_result_csv(ext.OUT/"predictions.csv");assert len(frame)==720
    assert not frame.duplicated(["method","budget","sample_id","growth_quantile"]).any()
    samples=read_result_csv(ext.OUT/"paired_samples.csv").sort_values("incorporated_cm3")
    samples["H_known"]=False;samples["H_cm3"]=0.;samples["carbon_known"]=False;samples["carbon_cm3"]=0.
    samples["Tobs_K"]=300.;samples["element"]="Sn"
    observed=np.log10(samples.Hall_n_cm3.to_numpy(float));replays=[]
    rt=study.PRED[study.PRED.kind.eq("RT")]
    weights=1/(rt.groupby("family").sample_id.transform("count").to_numpy(float)*rt.family.nunique())
    for (method,budget,q),g in frame.groupby(["method","budget","growth_quantile"]):
        g=g.set_index("sample_id").loc[samples.sample_id];budget=int(budget)
        growth=ext.weighted_quantile(rt.growth_T_C.to_numpy(float),weights,q)
        assert np.allclose(g.assumed_growth_T_C,growth,atol=1e-12,rtol=0)
        assert g.role.tolist()==["anchor" if i<budget else "query" for i in range(3)]
        assert g.fixed_highest_query.tolist()==[False,False,True]
        assert np.allclose(g.observed,observed,atol=1e-13,rtol=0)
        if method.startswith("identity"):
            raw=np.log10(samples.incorporated_cm3.to_numpy(float))
            delta=float(np.mean(observed[:budget]-raw[:budget])) if budget and method.endswith("offset") else 0.
        else:
            selection=next(s for s in choices if s["method"]==method and s["budget"]==budget)
            f=full[g.fit_key.iloc[0]]
            c=selection["selected"]["config"].copy()
            if "reg" in c and c["reg"] is None:c["reg"]=np.inf
            canonical,config=study.canonical(method,c)
            assert f["method"]==canonical and f["config"]==energy.clean(config)
            rows=samples.copy();rows["growth_T_C"]=growth
            raw,_=repaired.raw_predict(f,rows)
            lam=selection["selected"]["lam"]
            delta=float(np.sum(observed[:budget]-raw[:budget])/(budget+lam)) if budget and lam is not None else 0.
        assert np.allclose(g.raw_predicted,raw,atol=2e-9,rtol=0,equal_nan=True)
        assert np.allclose(g.offset,delta,atol=2e-9,rtol=0,equal_nan=True)
        assert np.allclose(g.predicted,raw+delta,atol=2e-9,rtol=0,equal_nan=True)
        replays.extend(abs(g.predicted.to_numpy(float)-(raw+delta)))
    best=[min([t for t in f["trials"] if t.get("loss") is not None],key=lambda t:t["loss"],default=None) for f in full.values()]
    audit=dict(passed=True,created_utc=energy.now(),source_conditions=3,additional_source_count=1,
        original_source_CV_evaluations=evaluations,source_selection_records=len(choices),full_original_source_fits=len(full),
        saved_prediction_rows=len(frame),replayed_prediction_rows=len(replays),maximum_replay_error=float(np.nanmax(replays)),
        nonfinite_predictions=int((~np.isfinite(frame.predicted)).sum()),
        full_fits_best_start_nonconverged=sum(t is not None and not t["success"] for t in best),
        all_growth_scenarios_retained=True,external_labels_excluded_from_shared_training=True,
        no_new_mobility_validation=True,goal_complete=False,
        scope="new-source conditional retrospective transfer,unknown growth/pO2/H;not blind/full-input validation;one source cannot establish universal improvement")
    energy.write(ext.OUT/"contract_audit.json",audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
