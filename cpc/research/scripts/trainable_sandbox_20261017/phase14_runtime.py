"""Replay shared learning and internal source fitting without property tables."""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(tasks_path,output_path):
    forbidden=("dataset_v1.csv","episodes_v1.csv","paired_specimens.csv","temperature_observations.csv",
        "mobility_bounds_and_inputs.csv","/phase14_joint_statistics/fits/","/phase13_donor_mixture/fits/",
        "/solver_budget_v2/fits/","/methods/","/validation/","/episodes/","/variants/")
    blocked=[];opened=[]
    def guard(event,args):
        if event!="open" or not args or not isinstance(args[0],(str,bytes)):return
        p=str(args[0]);opened.append(p)
        if any(x in p for x in forbidden):blocked.append(p);raise RuntimeError("Property/fit-table access forbidden during inference")
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/"src"))
    import numpy as np
    from types import SimpleNamespace
    from sandbox.donor_mixture_model import DonorMixtureModel
    from sandbox.scalar_profile_adaptation import ScalarProfile
    from sandbox.material_profile import MaterialProfile
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    tasks=json.loads(Path(tasks_path).read_text());rows=[];adaptations=[]
    for task in tasks:
        b=json.loads(Path(task["bundle"]).read_text());assert sha(NPZ)==b["energetics_sha256"]
        model=DonorMixtureModel(b["definition"],profile=MaterialProfile.from_dict(b["material_profile"]),pO2_atm=b["pO2_atm"])
        w=b["parameters"];kind=b["source_kind"];lam=np.inf if b["lambda"] is None else b["lambda"]
        z=0.;anchors=task["anchors"];status="base_no_update"
        aa=[SimpleNamespace(**r["input"]) for r in anchors];yy=np.array([r["observed_log_nH"] for r in anchors])
        if anchors and np.isfinite(lam):
            if kind=="offset":
                residual=[model.point(row,w).values[0]-y for row,y in zip(aa,yy)]
                z=float(sum(residual)/(len(anchors)+lam));status="closed_offset"
            else:
                def forward(z):
                    p=[model.point(row,w,source_kind=kind,source_shift=z) for row in aa]
                    return np.array([a.values[0] for a in p]),np.array([a.source_jacobian[0] for a in p])
                fit=ScalarProfile(forward).fit(yy,lam);z=fit.shift;status=fit.status
        adaptations.append(dict(task_id=task["task_id"],shift=z,status=status))
        for point in task["points"]:
            values=None;mu=None;sigma=None;error=None;same_state=None
            try:
                if status=="no_feasible_anchor_state":raise ValueError("No feasible anchor state")
                a=model.point(SimpleNamespace(**point["input"]),w,source_kind=kind,source_shift=z)
                values=a.values.copy();same_state=a.diagnostics["same_physical_state_for_output"]
                mu=float(np.log10(transport.mu_caughey_thomas(10**values[0],300))+b["transport_offset_dex"])
                sigma=float(np.log10(transport.E)+values[0]+mu);values=values.tolist()
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:error=repr(exc)
            rows.append(dict(task_id=task["task_id"],obs_id=point["obs_id"],values=values,shift=z,
                predicted_log_mu=mu,predicted_log_sigma=sigma,same_physical_state_for_output=same_state,error=error))
    Path(output_path).write_text(json.dumps(dict(rows=rows,adaptations=adaptations,property_table_open_guard_active=True,
        blocked_paths=blocked,opened_paths_count=len(opened),property_tables_read=False),ensure_ascii=False,indent=2)+"\n")


def parent():
    sys.path.insert(0,str(Path(__file__).parent))
    import numpy as np
    import pandas as pd
    import phase14_source_adaptation as bench
    import phase2_energy as energy
    from csv_io_v2 import read_result_csv
    from sandbox.material_profile import PUBLISHED_PROFILE
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    final=bench.ref.OUT/"validation";source=bench.OUT/"validation";out=bench.BASE/"research_runtime"
    assert json.loads((final/"contract_audit.json").read_text())["passed"]
    assert json.loads((source/"contract_audit.json").read_text())["passed"]
    fitbank=bench.bank();pred=read_result_csv(final/"primary_predictions.csv")
    extras={v:pair for v,pair in bench.VARIANTS.items() if pair[1]!="offset"}
    epred=read_result_csv(source/"predictions.csv");epred=epred[epred.variant.isin(extras)].rename(columns={"variant":"method"})
    allpred=pd.concat([pred,epred],ignore_index=True)
    hp=energy.BASE/"phase12_stock_adaptation/full_property_chain/head_parameters.json"
    heads={r["held"]:r for r in json.loads(hp.read_text()) if r["head"]=="direct_Hall_argument"}
    physical=[m for m,d in bench.core.METHODS.items() if "empirical" not in d]
    protocol=dict(created_utc=energy.now(),code_sha256=sha(Path(__file__)),inference_code_sha256=sha(ROOT/"src/sandbox/donor_mixture_model.py"),
        adaptation_code_sha256=sha(ROOT/"src/sandbox/scalar_profile_adaptation.py"),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (hp,final/"primary_predictions.csv",source/"predictions.csv",source/"contract_audit.json")},
        scope="14basemodels+7internal-coordinate variants;state solve andallowedanchorfit;no propertytable reads",
        transport="same frozen empiricalCT head;newSI/BTE reference not used",goal_complete=False)
    pp=out/"protocol.json"
    if pp.exists():assert json.loads(pp.read_text())["code_sha256"]==protocol["code_sha256"]
    else:energy.write(pp,protocol)
    def inputs(row):
        return dict(element=row.element,incorporated_cm3=float(row.incorporated_cm3),growth_T_C=float(row.growth_T_C),
            Tobs_K=float(row.Tobs_K),H_known=bool(row.H_known),H_cm3=float(row.H_cm3))
    tasks=[];expected={};expected_shift={};index=[]
    plans=[(m,m,"offset",bench.ref.OUT/"methods"/m/"selections.json") for m in physical]
    plans += [(v,m,k,bench.OUT/"variants"/v/"selections.json") for v,(m,k) in extras.items()]
    for variant,method,kind,path in plans:
        for choice in json.loads(path.read_text()):
            held=choice["held"];budget=choice["budget"];f=fitbank[choice["fit_key"]]
            assert held not in f["train_families"] and held not in heads[held]["training_families"]
            tid=f"{variant}::{held}::{budget}";bp=out/"bundles"/(bench.core.prior.old.digest(tid)+".json")
            bundle=dict(model=variant,base_method=method,held=held,budget=budget,parameters=f["parameters"],definition=f["definition"],
                fit_key=f["key"],fit_sha256=f["composite_origin"]["sha256"],source_kind=kind,material_profile=PUBLISHED_PROFILE.to_dict(),
                pO2_atm=1e-5,energetics_sha256=sha(NPZ),transport_offset_dex=heads[held]["offset_dex"])
            bundle["lambda"]=choice["selected"]["lam"];energy.write(bp,bundle)
            d=bench.core.PRED[bench.core.PRED.family.eq(held)&bench.core.PRED.kind.eq("RT")]
            roles=bench.core.ROLES[bench.core.ROLES.family.eq(held)&bench.core.ROLES.budget.eq(budget)].set_index("sample_id").role
            anchors=[dict(input=inputs(r),observed_log_nH=float(r.target)) for r in d.itertuples() if roles[r.sample_id]=="anchor"]
            assert len(anchors)==budget
            tasks.append(dict(task_id=tid,bundle=str(bp),anchors=anchors,points=[dict(obs_id=r.obs_id,input=inputs(r)) for r in d.itertuples()]))
            index.append(dict(task_id=tid,model=variant,base_method=method,source_kind=kind,held=held,budget=budget,bundle=str(bp.relative_to(ROOT))))
            frame=allpred[allpred.method.eq(variant)&allpred.family.eq(held)&allpred.budget.eq(budget)]
            expected_shift[tid]=float(choice["shift"]) if variant in extras else -float(frame.offset_dex.iloc[0])
            for r in frame.itertuples():expected[tid,"RT::"+r.sample_id]=r.predicted
    energy.write(out/"tasks.json",tasks);energy.write(out/"index.json",index)
    result=out/"independent_outputs.json"
    subprocess.run([sys.executable,str(Path(__file__)),"--worker",str(out/"tasks.json"),str(result)],cwd=ROOT,check=True)
    output=json.loads(result.read_text());lookup={(r["task_id"],r["obs_id"]):r for r in output["rows"]}
    errors=[];shift_errors=[]
    for key,wanted in expected.items():
        r=lookup[key]
        if np.isnan(wanted):assert r["values"] is None
        else:assert r["error"] is None;errors.append(abs(r["values"][0]-wanted))
    for r in output["adaptations"]:shift_errors.append(abs(r["shift"]-expected_shift[r["task_id"]]))
    assert len(output["rows"])==1827 and not output["blocked_paths"] and max(errors)<1e-5
    # Fullchain applies the same empirical head to every baseline/source variant.
    data=bench.core.DATA.set_index("sample_id");eligible=set(data[data.mobility_eligible.fillna(False)&data.mu_cm2Vs.gt(0)].index)
    chain=[];replay=[]
    for r in allpred[allpred.role.eq("query")&allpred.sample_id.isin(eligible)].itertuples():
        offset=heads[r.family]["offset_dex"];lm=float(np.log10(transport.mu_caughey_thomas(10**r.predicted,300))+offset)
        ls=float(np.log10(transport.E)+r.predicted+lm);om=np.log10(data.loc[r.sample_id,"mu_cm2Vs"])
        chain.append(dict(method=r.method,family=r.family,budget=r.budget,sample_id=r.sample_id,predicted_log_nH=r.predicted,
            observed_log_nH=r.observed,predicted_log_mu=lm,observed_log_mu=om,predicted_log_sigma=ls,
            observed_log_sigma=np.log10(transport.E)+r.observed+om,target_source_mu_excluded=True,conductivity_is_derived=True))
        key=f"{r.method}::{r.family}::{int(r.budget)}","RT::"+r.sample_id
        if key in lookup:rr=lookup[key];replay.extend([abs(rr["predicted_log_mu"]-lm),abs(rr["predicted_log_sigma"]-ls)])
    df=pd.DataFrame(chain);df.to_csv(out/"full_chain_predictions.csv",index=False);by=[]
    for (m,b,f),g in df.groupby(["method","budget","family"]):
        row=dict(method=m,budget=b,family=f,queries=len(g))
        for key in ("nH","mu","sigma"):row[key+"_MAE"]=float(abs(g["predicted_log_"+key]-g["observed_log_"+key]).mean())
        by.append(row)
    by=pd.DataFrame(by);by.to_csv(out/"full_chain_by_source.csv",index=False)
    by.groupby(["method","budget"])[["nH_MAE","mu_MAE","sigma_MAE"]].mean().reset_index().to_csv(out/"full_chain_summary.csv",index=False)
    assert len(df)==297 and len(replay)==462 and max(replay)<1e-5
    energy.write(out/"verification.json",dict(passed=True,created_utc=energy.now(),physical_base_models=14,internal_source_variants=7,
        saved_bundles=len(tasks),carrier_replay_rows=len(output["rows"]),maximum_carrier_error=max(errors),
        maximum_source_coordinate_error=max(shift_errors),paired_transport_replay_rows=len(replay)//2,maximum_transport_error=max(replay),
        property_table_open_guard_active=True,property_tables_read=False,query_property_labels_in_tasks=False,
        full_chain_rows=len(df),chemical_paired_mu_specimens=5,chemical_paired_mu_DOIs=2,new_sigma_labels=0,
        new_BTE_reference_used_in_property_predictor=False,goal_complete=False))
    print("Runtime passed",len(tasks),"bundles",len(output["rows"]),"carrier rows;maxerror",max(errors))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker",nargs=2);a=ap.parse_args()
    if a.worker:worker(*a.worker)
    else:parent()


if __name__=="__main__":main()
