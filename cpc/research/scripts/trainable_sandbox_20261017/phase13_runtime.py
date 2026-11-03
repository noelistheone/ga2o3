"""Export new learned relations and replay in a property-table-free process."""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(tasks_path,output_path):
    forbidden=("dataset_v1.csv","episodes_v1.csv","paired_specimens.csv","temperature_observations.csv",
        "mobility_bounds_and_inputs.csv","/phase13_donor_mixture/fits/","/solver_budget_v2/fits/","/methods/","/validation/")
    blocked=[];opened=[]
    def guard(event,args):
        if event!="open" or not args or not isinstance(args[0],(str,bytes)):return
        p=str(args[0]);opened.append(p)
        if any(x in p for x in forbidden):blocked.append(p);raise RuntimeError("Property-table access forbidden during prediction")
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/"src"))
    import numpy as np
    from types import SimpleNamespace
    from sandbox.donor_mixture_model import DonorMixtureModel
    from sandbox.material_profile import MaterialProfile
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    tasks=json.loads(Path(tasks_path).read_text());rows=[];adaptations=[]
    for task in tasks:
        b=json.loads(Path(task["bundle"]).read_text());assert sha(NPZ)==b["energetics_sha256"]
        model=DonorMixtureModel(b["definition"],profile=MaterialProfile.from_dict(b["material_profile"]),pO2_atm=b["pO2_atm"])
        w=b["parameters"];lam=np.inf if b["lambda"] is None else b["lambda"]
        delta=0.;anchors=task["anchors"]
        if anchors and np.isfinite(lam):
            residual=[r["observed_log_nH"]-model.point(SimpleNamespace(**r["input"]),w).values[0] for r in anchors]
            delta=float(sum(residual)/(len(anchors)+lam))
        adaptations.append(dict(task_id=task["task_id"],offset=delta))
        for point in task["points"]:
            values=None;mu=None;sigma=None;error=None;same_state=None
            try:
                a=model.point(SimpleNamespace(**point["input"]),w);values=a.values.copy();values[0]+=delta
                same_state=bool(a.diagnostics["same_physical_state_for_output"] and delta==0)
                mu=float(np.log10(transport.mu_caughey_thomas(10**values[0],300))+b["transport_offset_dex"])
                sigma=float(np.log10(transport.E)+values[0]+mu);values=values.tolist()
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:error=repr(exc)
            rows.append(dict(task_id=task["task_id"],obs_id=point["obs_id"],values=values,offset=delta,
                predicted_log_mu=mu,predicted_log_sigma=sigma,same_physical_state_for_output=same_state,error=error))
    Path(output_path).write_text(json.dumps(dict(rows=rows,adaptations=adaptations,property_table_open_guard_active=True,
        blocked_paths=blocked,opened_paths_count=len(opened),property_tables_read=False),ensure_ascii=False,indent=2)+"\n")


def parent():
    sys.path.insert(0,str(Path(__file__).parent))
    import numpy as np
    import pandas as pd
    import phase13_prediction as study
    import phase13_refinement as ref
    import phase2_energy as energy
    from csv_io_v2 import read_result_csv
    from sandbox.material_profile import PUBLISHED_PROFILE
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    final=ref.OUT/"validation";out=study.OUT/"research_runtime"
    assert json.loads((final/"contract_audit.json").read_text())["passed"]
    fitbank=ref.load_composite();pred=read_result_csv(final/"primary_predictions.csv")
    hp=energy.BASE/"phase12_stock_adaptation/full_property_chain/head_parameters.json"
    heads={r["held"]:r for r in json.loads(hp.read_text()) if r["head"]=="direct_Hall_argument"}
    selected=[m for m,d in study.METHODS.items() if "empirical" not in d]
    protocol=dict(created_utc=energy.now(),code_sha256=sha(Path(__file__)),
        inference_code_sha256=sha(ROOT/"src/sandbox/donor_mixture_model.py"),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (hp,final/"primary_predictions.csv",ref.OUT/"validation/contract_audit.json")},
        scope="sourceexcluded research bundles;single itinerant Hall scenario;outputoffset separate from physicalstate",
        no_new_sigma_labels=True,no_new_physical_measurements=True,goal_complete=False)
    pp=out/"protocol.json"
    if pp.exists():assert json.loads(pp.read_text())["code_sha256"]==protocol["code_sha256"]
    else:energy.write(pp,protocol)
    def physical_input(row):
        return dict(element=row.element,incorporated_cm3=float(row.incorporated_cm3),growth_T_C=float(row.growth_T_C),
            Tobs_K=float(row.Tobs_K),H_known=bool(row.H_known),H_cm3=float(row.H_cm3))
    tasks=[];expected={};expected_shift={};index=[]
    for method in selected:
        choices=json.loads((ref.OUT/"methods"/method/"selections.json").read_text())
        for choice in choices:
            held=choice["held"];budget=choice["budget"];f=fitbank[choice["fit_key"]]
            assert held not in f["train_families"] and held not in heads[held]["training_families"]
            task_id=f"{method}::{held}::{budget}";bundlepath=out/"bundles"/(study.prior.old.digest(task_id)+".json")
            b=dict(model=method,held=held,budget=budget,parameters=f["parameters"],definition=f["definition"],
                fit_key=f["key"],fit_sha256=f["composite_origin"]["sha256"],lambda_value=choice["selected"]["lam"],
                material_profile=PUBLISHED_PROFILE.to_dict(),profile_sha256=PUBLISHED_PROFILE.fingerprint,
                pO2_atm=1e-5,energetics_sha256=sha(NPZ),transport_offset_dex=heads[held]["offset_dex"])
            b["lambda"]=b.pop("lambda_value");energy.write(bundlepath,b)
            source=study.PRED[study.PRED.family.eq(held)&study.PRED.kind.eq("RT")]
            roles=study.ROLES[study.ROLES.family.eq(held)&study.ROLES.budget.eq(budget)].set_index("sample_id").role
            anchors=[dict(input=physical_input(r),observed_log_nH=float(r.target)) for r in source.itertuples() if roles[r.sample_id]=="anchor"]
            assert len(anchors)==budget
            tasks.append(dict(task_id=task_id,bundle=str(bundlepath),anchors=anchors,
                points=[dict(obs_id=r.obs_id,input=physical_input(r)) for r in source.itertuples()]))
            index.append(dict(task_id=task_id,model=method,held=held,budget=budget,bundle=str(bundlepath.relative_to(ROOT))))
            g=pred[pred.method.eq(method)&pred.family.eq(held)&pred.budget.eq(budget)]
            expected_shift[task_id]=float(g.offset_dex.iloc[0])
            raw={r["obs_id"]:r for r in f["predictions"]}
            for r in g.itertuples():expected[task_id,"RT::"+r.sample_id]=(r.predicted,raw["RT::"+r.sample_id].get("charge_outputs"))
    energy.write(out/"tasks.json",tasks);energy.write(out/"index.json",index)
    result=out/"independent_outputs.json"
    subprocess.run([sys.executable,str(Path(__file__)),"--worker",str(out/"tasks.json"),str(result)],cwd=ROOT,check=True)
    p=json.loads(result.read_text());errors=[];qerrors=[];offset_errors=[]
    lookup={(r["task_id"],r["obs_id"]):r for r in p["rows"]}
    for key,(wanted,charges) in expected.items():
        r=lookup[key]
        if np.isnan(wanted):assert r["values"] is None
        else:
            assert r["error"] is None;errors.append(abs(r["values"][0]-wanted))
            qerrors.extend(abs(np.array(r["values"][1:])-np.array(charges)))
    for r in p["adaptations"]:offset_errors.append(abs(r["offset"]-expected_shift[r["task_id"]]))
    assert len(p["rows"])==957 and not p["blocked_paths"] and max(errors)<1e-6 and max(qerrors)<1e-6
    # All17 models get exactly the same frozen source-excluded empirical head.
    data=study.DATA.set_index("sample_id");eligible=set(data[data.mobility_eligible.fillna(False)&data.mu_cm2Vs.gt(0)].index)
    chain=[];replay=[]
    for r in pred[pred.role.eq("query")&pred.sample_id.isin(eligible)].itertuples():
        h=heads[r.family];lm=float(np.log10(transport.mu_caughey_thomas(10**r.predicted,300))+h["offset_dex"])
        sigma=float(np.log10(transport.E)+r.predicted+lm);observed_mu=np.log10(data.loc[r.sample_id,"mu_cm2Vs"])
        chain.append(dict(method=r.method,family=r.family,budget=r.budget,sample_id=r.sample_id,
            predicted_log_nH=r.predicted,observed_log_nH=r.observed,predicted_log_mu=lm,observed_log_mu=observed_mu,
            predicted_log_sigma=sigma,observed_log_sigma=np.log10(transport.E)+r.observed+observed_mu,
            target_source_mu_excluded=True,conductivity_is_derived=True))
        key=f"{r.method}::{r.family}::{int(r.budget)}","RT::"+r.sample_id
        if key in lookup:
            rr=lookup[key];replay.extend([abs(rr["predicted_log_mu"]-lm),abs(rr["predicted_log_sigma"]-sigma)])
    df=pd.DataFrame(chain);df.to_csv(out/"full_chain_predictions.csv",index=False);by=[]
    for (m,b,f),g in df.groupby(["method","budget","family"]):
        d=dict(method=m,budget=b,family=f,queries=len(g))
        for key in ("nH","mu","sigma"):d[key+"_MAE"]=float(abs(g["predicted_log_"+key]-g["observed_log_"+key]).mean())
        by.append(d)
    by=pd.DataFrame(by);by.to_csv(out/"full_chain_by_source.csv",index=False)
    summary=by.groupby(["method","budget"])[["nH_MAE","mu_MAE","sigma_MAE"]].mean().reset_index()
    summary.to_csv(out/"full_chain_summary.csv",index=False)
    assert len(df)==187 and len(replay)==242 and max(replay)<1e-6
    energy.write(out/"verification.json",dict(passed=True,created_utc=energy.now(),physical_methods=len(selected),
        saved_bundles=len(tasks),carrier_replay_rows=len(p["rows"]),maximum_carrier_error=max(errors),
        maximum_charge_budget_error=max(qerrors),maximum_offset_error=max(offset_errors),
        independent_paired_transport_replay_rows=len(replay)//2,maximum_transport_error=max(replay),
        property_table_open_guard_active=True,property_tables_read=False,query_property_labels_in_tasks=False,
        full_chain_prediction_rows=len(df),chemical_paired_mu_samples=5,chemical_paired_mu_sources=2,
        new_sigma_labels=0,goal_complete=False))
    print("Independent inference passed",len(tasks),"bundles",len(p["rows"]),"carrier rows;maxerror",max(errors))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker",nargs=2);a=ap.parse_args()
    if a.worker:worker(*a.worker)
    else:parent()


if __name__=="__main__":main()
