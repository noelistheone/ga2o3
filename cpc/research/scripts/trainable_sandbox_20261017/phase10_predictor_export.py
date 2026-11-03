"""Export source-excluded learned models and replay the standalone runtime."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.thermal_predictor import ThermalDopantPredictor,EffectiveThermalState
from sandbox.material_profile import PUBLISHED_PROFILE
from csv_io_v2 import read_result_csv
import phase10_internal_adaptation as study
import phase2_energy as energy


def main():
    folder=study.OUT/"research_predictor_v2";folder.mkdir(exist_ok=True)
    protocol=dict(API_code_sha256=energy.sha(ROOT/"src/sandbox/thermal_predictor.py"),
        export_script_sha256=energy.sha(Path(__file__)),parent_protocol_sha256=energy.sha(study.OUT/"protocol.json"),
        scope="explicit material/pressure/energetic checksums;trained source-excluded weights;no runtime table lookup")
    p=folder/"protocol.json"
    if p.exists():assert json.loads(p.read_text())==protocol
    else:energy.write(p,protocol)
    models={};index=[]
    for held in study.FAMILIES:
        path,base=study.base_fit([f for f in study.FAMILIES if f!=held],"learned_screen_joint")
        theta=study.initial_coordinates(base);w=base["parameters"]
        state=EffectiveThermalState(float(10**theta[0]),float(10**w[1]),float(10**theta[1]),float(w[3]))
        model=ThermalDopantPredictor(state,PUBLISHED_PROFILE,metadata=dict(
            train_families=base["train_families"],excluded_family=held,base_model_sha256=energy.sha(path),
            material_scope="Si/Sn effective donor,onlySi temperature curves validated",source_excluded=True))
        dest=folder/(energy.sha(path)[:16]+".json");model.save(dest);models[held]=dest
        index.append(dict(excluded_family=held,model=dest.name,train_families=base["train_families"]))
    frames=read_result_csv(study.OUT/"methods/shared_joint_Tf_E/predictions.csv")
    choices=json.loads((study.OUT/"methods/shared_joint_Tf_E/selections.json").read_text())
    runtime_open=[];capture={"active":False}
    def audit(event,args):
        if event=="open" and capture["active"] and isinstance(args[0],(str,bytes)):
            runtime_open.append(str(args[0]))
    sys.addaudithook(audit)
    errors=[];parameter_errors=[]
    for (family,sample,budget),g in frames[frames.task.eq("temperature")].groupby(["family","sample_id","budget"]):
        choice=next(s for s in choices if s["task"]=="temperature" and s["held"]==family and s["budget"]==budget)["selected"]
        r=study.PRED[study.PRED.sample_id.eq(sample)].iloc[0]
        H=float(r.H_cm3) if r.H_known else None
        anchors=[dict(element=r.element,concentration_cm3=float(r.incorporated_cm3),T_K=float(a.Tobs_K),
            nH_cm3=float(10**a.observed),H_cm3=H) for a in g[g.role.eq("anchor")].itertuples()]
        capture["active"]=True
        model=ThermalDopantPredictor.load(models[family])
        model,fit=model.adapt(anchors,regularization=np.inf if choice["lam"] is None else choice["lam"],
            energy_scale=choice["config"]["E_scale"],mode="Tf_E")
        result=model.predict(element=r.element,concentration_cm3=float(r.incorporated_cm3),temperatures_K=g.Tobs_K.to_numpy(float),H_cm3=H)
        capture["active"]=False
        for expected,actual in zip(g.itertuples(),result["predictions"]):
            errors.append(dict(sample_id=sample,budget=int(budget),obs_id=expected.obs_id,
                error_dex=float(abs(actual["log10_nH_cm3"]-expected.predicted))))
    allowed={str(p) for p in models.values()}|{str(p.resolve()) for p in models.values()}
    forbidden=[p for p in runtime_open if "trainable_sandbox_20261017" in p and p not in allowed]
    assert not forbidden,forbidden
    maximum=max(r["error_dex"] for r in errors)
    assert maximum<3e-5,maximum
    energy.write(folder/"index.json",dict(models=index,
        status="five source-excluded trained research models,not refitall-data deployment or independent newvalidation",
        files_required="saved modelJSON and separately licensed/local checksummed KROGERnumericNPZ;no experimental tables at inference"))
    energy.write(folder/"verification.json",dict(passed=True,replay_rows=len(errors),maximum_error_dex=maximum,
        runtime_training_table_reads=forbidden,runtime_open_paths=sorted(set(runtime_open)),
        shared_relation_trained=True,independent_new_material_validation=False,
        API_code_sha256=energy.sha(ROOT/"src/sandbox/thermal_predictor.py"),
        export_script_sha256=energy.sha(Path(__file__))))
    energy.csvwrite(folder/"replay_errors.csv",errors)
    print(json.dumps(dict(passed=True,replays=len(errors),maximum_error_dex=maximum,no_training_table_lookup=True)))


if __name__=="__main__":main()
