"""Export and replay source-excluded physical models without property tables."""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib,math
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(tasks_path,output_path):
    forbidden=("dataset_v1.csv","episodes_v1.csv","observations_and_inputs.csv","paired_specimens.csv",
        "temperature_observations.csv","mobility_bounds_and_inputs.csv","/new_base_models/fits/","/episodes/","/physical_scope_v2/models/")
    blocked=[];read_paths=[]
    def guard(event,args):
        if event!="open" or not args or not isinstance(args[0],(str,bytes)):return
        p=str(args[0]);read_paths.append(p)
        if any(x in p for x in forbidden):
            blocked.append(p);raise RuntimeError("Property-table access forbidden in independent inference")
    sys.addaudithook(guard)
    sys.path.insert(0,str(ROOT/"src"))
    import numpy as np
    from types import SimpleNamespace
    from sandbox.source_stock_model import SourceStockModel
    from sandbox.scalar_profile_adaptation import ScalarProfile
    from sandbox.material_profile import MaterialProfile
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    tasks=json.loads(Path(tasks_path).read_text());rows=[];calibrations=[]
    for task in tasks:
        bundle=json.loads(Path(task["bundle"]).read_text())
        assert sha(NPZ)==bundle["energetics_sha256"]
        profile=MaterialProfile.from_dict(bundle["material_profile"])
        assert profile.fingerprint==bundle["material_profile_sha256"]
        m=SourceStockModel(bundle["parameters"],bundle["definition"],bundle["feature_metadata"],
            profile=profile,pO2_atm=bundle["pO2_atm"])
        kind=bundle["source_kind"];lam=np.inf if bundle["lambda"] is None else bundle["lambda"]
        anchors=[SimpleNamespace(**r["input"]) for r in task["anchors"]]
        y=np.array([r["observed_log_nH"] for r in task["anchors"]]);z=0.;status="base_no_update"
        assert len(anchors)==bundle["budget"]
        if len(anchors) and np.isfinite(lam):
            if kind in ("stock","Tf"):
                def forward(z):
                    p=[m.predict(row,source_kind=kind,source_shift=z) for row in anchors]
                    return np.array([v.values[0] for v in p]),np.array([v.source_jacobian[0] for v in p])
                r=ScalarProfile(forward).fit(y,lam);z=r.shift;status=r.status
            else:
                raw=np.array([m.predict(row).values[0] for row in anchors]);z=float(np.sum(raw-y)/(len(y)+lam));status="closed_offset"
        calibrations.append(dict(task_id=task["task_id"],source_shift=float(z),status=status))
        for point in task["points"]:
            values=None;mu=None;sigma=None;error=None
            try:
                p=m.predict(SimpleNamespace(**point["input"]),source_kind=kind,source_shift=z)
                values=p.values.tolist()
                # Explicitly empirical Hall mobility head; only300K is exported.
                assert point["input"]["Tobs_K"]==300.
                mu=float(np.log10(transport.mu_caughey_thomas(10**values[0],300.))+bundle["transport_offset_dex"])
                sigma=float(np.log10(transport.E)+values[0]+mu)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:error=repr(exc)
            rows.append(dict(task_id=task["task_id"],obs_id=point["obs_id"],values=values,
                predicted_log_mu=mu,predicted_log_sigma=sigma,error=error))
    Path(output_path).write_text(json.dumps(dict(rows=rows,calibrations=calibrations,
        blocked_paths=blocked,opened_path_count=len(read_paths),property_tables_read=False),ensure_ascii=False,indent=2)+"\n")


def parent():
    sys.path.insert(0,str(Path(__file__).parent))
    import numpy as np
    import phase12_prediction as study
    from csv_io_v2 import read_result_csv
    from sandbox.kroger_db import NPZ
    from sandbox.material_profile import PUBLISHED_PROFILE
    e=study.energy;out=study.OUT/"research_runtime";final=study.OUT/"physical_scope_v2"
    assert json.loads((final/"contract_audit.json").read_text())["passed"]
    assert json.loads((study.OUT/"full_property_chain/audit.json").read_text())["passed"]
    selected=("original_offset","original_stock","original_Tf","constant_offset","constant_stock",
              "shared_RT_offset","shared_RT_stock","joint_stock","shared_joint_stock")
    heads=json.loads((study.OUT/"full_property_chain/head_parameters.json").read_text())
    payload=dict(created_utc=e.now(),code_sha256=e.sha(Path(__file__)),
        inference_code_sha256=e.sha(ROOT/"src/sandbox/source_stock_model.py"),
        adaptation_code_sha256=e.sha(ROOT/"src/sandbox/scalar_profile_adaptation.py"),
        task="saved-sharedrelation+fullphysicalsolver+0/1/2sourceanchors;deny propertytable opens in freshprocess",
        models=selected,numerical_tolerance_dex=1e-5,
        outputs="nH,Nabs,Nq2;empirical300Kmu andderivedsigma,not microscopic scattering or newmobilityvalidation",
        scope="source-excluded research checkpoints;not fullfive-source deployment ornew-element certification")
    pp=out/"protocol.json"
    if pp.exists():assert json.loads(pp.read_text())["code_sha256"]==payload["code_sha256"]
    else:e.write(pp,payload)
    def physical_input(r):
        return dict(element=r.element,incorporated_cm3=float(r.incorporated_cm3),growth_T_C=float(r.growth_T_C),
            Tobs_K=float(r.Tobs_K),H_known=bool(r.H_known),H_cm3=float(r.H_cm3))
    tasks=[];expected={};expected_shift={};index=[]
    allpred=read_result_csv(final/"predictions.csv")
    for model in selected:
        choices=json.loads((final/"models"/model/"selections.json").read_text())
        method,kind=study.MODELS[model]
        for s in choices:
            f=study.getfit([x for x in study.core.FAMILIES if x!=s["held"]],method,s["selected"]["config"])
            assert f["key"]==s["fit_key"]
            group=allpred[allpred.model.eq(model)&allpred.family.eq(s["held"])&allpred.budget.eq(s["budget"])]
            d=study.core.PRED[study.core.PRED.family.eq(s["held"])&study.core.PRED.kind.eq("RT")]
            role=study.core.ROLES[study.core.ROLES.family.eq(s["held"])&study.core.ROLES.budget.eq(s["budget"])].set_index("sample_id").role
            key=study.core.old.digest(dict(model=model,held=s["held"],budget=s["budget"]))
            head=next(h for h in heads if h["held"]==s["held"] and h["head"]=="direct_Hall_argument")
            bundle=dict(version=1,model=model,budget=s["budget"],source_kind=kind,parameters=f["parameters"],
                definition=study.core.METHODS[f["method"]],
                feature_metadata={k:f["feature_metadata"][k] for k in ("logC_center","logC_scale")},
                material_profile=PUBLISHED_PROFILE.to_dict(),material_profile_sha256=PUBLISHED_PROFILE.fingerprint,
                pO2_atm=1e-5,energetics_sha256=e.sha(NPZ),
                lambda_value_scope="regularization not calibrated posterior variance",**{"lambda":s["selected"]["lam"]},
                transport_offset_dex=head["offset_dex"],transport_training_families=head["training_families"],
                training_sources=f["train_families"],held_source=s["held"],base_reference=study.reference(f),
                limitations="SiSn,RT,unknownH conditionalzero,Cnotmodeled,pO2scenario;mu empirical,sigma derived")
            path=out/"bundles"/(key+".json");e.write(path,bundle)
            task=dict(task_id=key,bundle=str(path),anchors=[dict(input=physical_input(r),observed_log_nH=float(r.target))
                for r in d[d.sample_id.map(role).eq("anchor")].itertuples()],
                points=[dict(obs_id=r.obs_id,input=physical_input(r)) for r in d.itertuples()])
            assert len(task["anchors"])==s["budget"]
            tasks.append(task);index.append(dict(task_id=key,model=model,held=s["held"],budget=s["budget"],bundle=str(path.relative_to(ROOT))))
            for r in group.itertuples():expected[key,"RT::"+r.sample_id]=r.predicted
            expected_shift[key]=float(group.source_shift.iloc[0])
    e.write(out/"tasks.json",tasks);e.write(out/"index.json",index)
    command=[sys.executable,str(Path(__file__).resolve()),"--worker",str(out/"tasks.json"),str(out/"independent_outputs.json")]
    run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (out/"worker.log").write_text(run.stdout+"\n"+run.stderr);run.check_returncode()
    result=json.loads((out/"independent_outputs.json").read_text());replay=[]
    for row in result["rows"]:
        value=expected[row["task_id"],row["obs_id"]]
        error=abs(row["values"][0]-value) if row["values"] is not None and np.isfinite(value) else np.inf
        replay.append(dict(task_id=row["task_id"],obs_id=row["obs_id"],error=error,error_message=row["error"]))
    maximum=max(r["error"] for r in replay)
    shifts=[abs(c["source_shift"]-expected_shift[c["task_id"]]) for c in result["calibrations"]]
    passed=maximum<=1e-5 and not result["blocked_paths"] and len(replay)==len(expected)
    e.csvwrite(out/"replay.csv",replay)
    audit=dict(passed=passed,created_utc=e.now(),models=len(selected),saved_bundles=len(index),prediction_rows=len(replay),
        maximum_log_nH_error=maximum,maximum_source_shift_error=max(shifts),
        property_table_open_guard_active=True,property_tables_read=False,blocked_paths=result["blocked_paths"],
        query_property_labels_in_runtime_tasks=False,allowed_property_inputs="only explicit anchor lognH",
        new_mu_validation=False,derived_sigma_only=True,goal_complete=False)
    e.write(out/"verification.json",audit);print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--worker",nargs=2,metavar=("TASKS","OUTPUT"));a=ap.parse_args()
    if a.worker:worker(*a.worker)
    else:parent()
