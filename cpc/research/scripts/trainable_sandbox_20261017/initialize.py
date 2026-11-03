"""Freeze the prediction contract before fitting new learned models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/trainable_sandbox_20261017"
OUT.mkdir(parents=True,exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    protocol=OUT/"protocol_v1.json"
    if protocol.exists():
        raise RuntimeError("Protocol v1 already exists; use a new version for changes")
    frames=[]
    for name in ["sample_master_v4.csv","external_samples.csv"]:
        p=ROOT/"results/next_study_20261015"/name
        frame=pd.read_csv(p)
        frame["input_file"]=str(p.relative_to(ROOT))
        frames.append(frame)
    d=pd.concat(frames,ignore_index=True)
    d=d[d.carrier_task_eligible.fillna(False).astype(bool)].copy()
    assert len(d)==29 and d.family.nunique()==5 and d.sample_id.is_unique
    assert d.pO2_atm.isna().all()
    d=d.sort_values(["family","incorporated_cm3","sample_id"]).reset_index(drop=True)
    d["log_dose"]=np.log10(d.incorporated_cm3)
    d["observed_log_nH"]=np.log10(d.carrier_cm3)
    d["scenario_pO2_atm"]=1e-5
    d["quantitative_process_inputs_complete"]=False
    d["prediction_task"]="incorporated_dopant_to_properties"
    d.to_csv(OUT/"dataset_v1.csv",index=False)
    episodes=[]
    # One reading per anchor; near-duplicate dose readings are guarded from query.
    for family,g in d.groupby("family",sort=True):
        groups=[];previous=None
        for row in g.itertuples():
            if previous is None or row.log_dose-previous>.005:
                groups.append([])
            groups[-1].append(row.sample_id);previous=row.log_dose
        for budget in (0,1,2):
            selected=[] if budget==0 else [0]
            if budget==2:
                selected.append(max(1,(len(groups)-1)//2))
            anchors=[groups[j][0] for j in selected]
            guard=[x for j in selected for x in groups[j][1:]]
            for row in g.itertuples():
                role="anchor" if row.sample_id in anchors else ("guard" if row.sample_id in guard else "query")
                episodes.append(dict(family=family,sample_id=row.sample_id,budget=budget,role=role))
    pd.DataFrame(episodes).to_csv(OUT/"episodes_v1.csv",index=False)
    protected=sorted((ROOT/"src/sandbox").glob("*.py"))
    protected=[p for p in protected if p.name!="implicit_defects.py"]
    protected += [ROOT/"results/phase0/kroger_ga2o3_092724_numeric.npz",
                  ROOT/"results/next_study_20261015/sample_master_v4.csv",
                  ROOT/"results/next_study_20261015/external_samples.csv"]
    protected += sorted((ROOT/"results/chain_diagnosis_20261015/executed").glob("*"))
    protected=[p for p in protected if p.is_file()]
    manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in protected]
    (OUT/"protected_inputs_before.json").write_text(json.dumps(manifest,indent=2)+"\n")
    contract=dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        user_goal="ML + Sandbox prediction of doped-material properties with rigorous theory",
        primary_endpoint="log10 Hall carrier density from actual incorporated concentration",
        secondary_endpoints=["Hall mobility","conductivity consistent with nH and muH"],
        future_endpoint="validated doping-candidate ranking when energetic/process inputs support it",
        dataset=dict(records=29,families=5,elements=["Si","Sn"],dataset_sha256=sha(OUT/"dataset_v1.csv"),
                     missing_oxygen_partial_pressure=29,independently_complete_process_records=0),
        pressure=dict(primary_atm=1e-5,status="explicit unmeasured common scenario",
                      sensitivity_atm=[1e-8,1e-2],selection="not optimized on test labels"),
        anchor_policy="m=0,1,2; lowest unique dose, then middle unique dose; one reading per anchor; other readings within .005 dex of anchor guarded",
        split="outer leave publication family out; inner leave training family out for model/hyperparameter selection",
        validation_grade="retrospective research and validation, not blind or preregistered independent confirmation",
        primary_metric="family-equal MAE on all numeric query predictions, coverage/refusals separately reported",
        additional_metrics=["same fixed legacy covered subset","RMSE","worst family","per-family MAE",
                            "numeric and physical coverage","paired mobility/conductivity where available"],
        models=["constant","unit_dose","line_dose","Ridge_process",
                "physics_reference","physics_global_process","physics_shared_process_input",
                "physics_shared_energy"],
        shared_process_input="log10 Tf = t0 + w*(growth_K-1100)/300, shared across samples",
        shared_energy="one parent-consistent native Ga vacancy formation-energy correction at fixed Tf=1325 K",
        confounding_control="process and energy branches are competitors; not jointly unconstrained",
        adaptation=["none","shared Gaussian output offset","one-dimensional physical process/energy latent"],
        within_noise_dex=[.1,.3,.6,1.],
        offset_prior_sd_dex=[0.,.1,.3,.6,1.],
        process_prior_sd_log10T=[0.,.02,.05,.10],
        energy_prior_sd_eV=[0.,.05,.15,.35],
        ridge_alpha=[.01,.1,1.,10.,100.],
        selection_metric="inner family-equal query MAE per anchor budget; deterministic ties prefer lower complexity",
        gradient_requirements=dict(interior_root=True,reference_error_dex_max=2e-6,
             central_difference_scaled_error_max=3e-5,float64=True,
             forbid_silent_clipping=True,first_order_only=True),
        resource=dict(authorized_physical_GPU=1,CUDA_VISIBLE_DEVICES="1",CPU_first_for_small_models=True),
        stopping_rule="Complete numerical gates before real-data learning; retain failures and version any theory or protocol changes",
        favorable_scores_required=False)
    protocol.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+"\n")
    (OUT/"progress.json").write_text(json.dumps(dict(status="active",stage="implicit_layer_verification",
        completed=["goal checked and prediction direction written","dataset and episode contract frozen"],
        next=["independent reference and gradient audit","same-budget nested source-exclusion models"],
        protocol_sha256=sha(protocol)),ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(records=len(d),families=d.family.nunique(),episodes=len(episodes),
        protected_files=len(manifest),protocol_sha256=sha(protocol)),indent=2))

if __name__=="__main__":
    main()
