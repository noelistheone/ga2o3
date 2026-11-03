"""Fixed Azizie2023 related-study transfer with explicit per-model refusals."""
from pathlib import Path
import sys,json,shutil,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter
import phase3_new_source as source
import phase3_new_source_predict as driver
import phase5_property as study

OUT=study.training.OUT/"new_property_source"

def protocol():
    study.freeze()
    p=OUT/"protocol.json"
    inputs=(source.OUT/"azizie2023_audited_samples.csv",
            source.OUT/"azizie2023_validation_protocol.json",
            source.OUT/"azizie2023_full_chain_predictions.csv",
            study.OUT/"predictions.csv")
    payload=dict(version=1,created_utc=energy.now(),methods=study.METHODS,
        wrapper_code_sha256=energy.sha(Path(__file__)),
        driver_code_sha256=energy.sha(Path(driver.__file__)),
        input_hashes={str(x.relative_to(ROOT)):energy.sha(x) for x in inputs},
        selection="reuse old related-source-excluded fits/shrinkage; no new-study tuning",
        chemical_inputs="exact frozen e/k rounded values and9 fixed resolution scenarios",
        source_grade="already exposed, corrected-label related-study test, not independent-lab blind validation",
        refusals="catch and record each whole-method failure; never drop a failed method from the evidence inventory",
        copied_controls=["identity_dose","global_mean","oracle_full","kernel_energy_select_energy","P2_Ridge_goal_k2"],
        transport="exact same49old pairs/10sources, excludes related DOI; no query measured nH or target mu")
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("wrapper_code_sha256","driver_code_sha256","input_hashes"):assert old[k]==payload[k]
    else:energy.write(p,payload)
    return payload

def library_and_fold(method):
    data=read_result_csv(study.OUT/"outer"/f"{fitter.digest(source.RELATED)}__{method}.csv")
    key=data.fit_key.iloc[0]
    fit=json.loads((study.OUT/"fits"/f"{key}.json").read_text())
    assert source.RELATED not in fit["train_families"] and source.DOI not in fit["train_families"]
    if fit["parameters"] is None:raise RuntimeError("All old-source process fitting starts failed")
    choices=json.loads((study.OUT/"selections"/f"{fitter.digest(source.RELATED)}__{method}.json").read_text())
    lam={int(x["budget"]):(np.inf if x["selected"]["lam"] is None else x["selected"]["lam"]) for x in choices}
    db,_=element_database(study.database("Si",method),"Si")
    return ImplicitDefectLayer(db,("Si",)),fit,lam,0

def main():
    payload=protocol();statuses=[];frames=[]
    original_freeze=driver.freeze
    for method in study.METHODS:
        folder=OUT/"methods"/method;status=folder/"status.json"
        if status.exists():
            state=json.loads(status.read_text())
        else:
            folder.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source.OUT/"azizie2023_audited_samples.csv",
                            folder/"azizie2023_audited_samples.csv")
            # Per-model setup is a subset of the parent protocol, with no new choices.
            def freeze(folder=folder,method=method):
                path=folder/"input_contract.json"
                content=dict(parent_protocol_sha256=energy.sha(OUT/"protocol.json"),method=method)
                if path.exists():assert json.loads(path.read_text())==content
                else:energy.write(path,content)
            driver.OUT=folder;driver.METHODS=(method,)
            driver.freeze=freeze;driver.library_and_fold=library_and_fold
            try:
                driver.main()
                state=dict(method=method,status="predicted",refusal=None)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                state=dict(method=method,status="refused",refusal=repr(exc))
                (folder/"failure_trace.txt").write_text(traceback.format_exc())
            energy.write(status,state)
        statuses.append(state)
        if state["status"]=="predicted":
            frames.append(read_result_csv(folder/"azizie2023_full_chain_predictions.csv"))
    controls=read_result_csv(source.OUT/"azizie2023_full_chain_predictions.csv")
    controls=controls[controls.method.isin(payload["copied_controls"])]
    frames.append(controls)
    allrows=pd.concat(frames,ignore_index=True)
    allrows.to_csv(OUT/"full_chain_predictions.csv",index=False)
    rows=[]
    for (m,b),g in allrows[allrows.role.eq("query")].groupby(["method","budget"]):
        primary=g[g.primary_midpoint]
        entry=dict(method=m,budget=int(b),query_samples=len(primary),
                   primary_physical=int(primary.physical_valid.eq(True).sum()),
                   primary_numeric=int(primary.predicted_log_nH.notna().sum()))
        for name in ("nH","mu","sigma"):
            err=abs(g["predicted_log_"+name]-g["observed_log_"+name])
            scenario=err.groupby(g.scenario).mean()
            entry[name+"_MAE"]=float(abs(primary["predicted_log_"+name]-primary["observed_log_"+name]).mean())
            entry[name+"_scenario_min"]=float(scenario.min());entry[name+"_scenario_max"]=float(scenario.max())
        rows.append(entry)
    energy.csvwrite(OUT/"summary.csv",rows)
    energy.write(OUT/"model_statuses.json",dict(methods=statuses,
        refused_methods=[v["method"] for v in statuses if v["status"]!="predicted"],
        copied_controls=payload["copied_controls"],new_material_measurements=0))
    shutil.copyfile(source.OUT/"azizie2023_conditional_transport.csv",OUT/"conditional_transport_reference.csv")
    print("New-study predictions and refusals retained:",len(statuses),flush=True)

if __name__=="__main__":main()
