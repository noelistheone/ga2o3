"""Integration checks against separately saved final-fit predictions."""
from pathlib import Path
import sys,json,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.kroger_db import load
from sandbox.trainable_predictor import DopedPropertyPredictor
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"predictor_bundle_v1"

def main():
    predictor=DopedPropertyPredictor(load(),OUT)
    dataset=read_result_csv(energy.BASE/"dataset_v1.csv")
    checks=[]
    for e in ("Si","Sn"):
        row=dataset[dataset.element.eq(e)].iloc[0]
        for method in predictor.manifest["methods"]:
            fit_key=predictor.manifest["process_models"][e][method]["fit_key"]
            fit=json.loads((OUT/"training/fits"/f"{fit_key}.json").read_text())
            expected=next(r for r in fit["predictions"] if r["sample_id"]==row.sample_id)
            result=predictor.predict(e,float(row.incorporated_cm3),method=method)
            diff=abs(result["log10_nH_cm3"]-expected["predicted_log_nH"])
            assert diff<2e-6
            assert abs(result["sigma_S_cm"]/(1.602176634e-19*result["nH_cm3"]*
                       result["muH_cm2Vs"])-1)<1e-12
            achieved=result["raw_Sandbox_diagnostics"]["achieved_concentrations_cm3"][e]
            assert abs(achieved/row.incorporated_cm3-1)<3e-8
            checks.append(dict(element=e,method=method,fit_replay_error_dex=diff,
                chemical_relative_residual=abs(achieved/row.incorporated_cm3-1)))
    # A synthetic readout perturbation checks anchor application and cannot count as material validation.
    anchor_C=1e18
    base=predictor.predict("Si",anchor_C)
    anchor=dict(incorporated_cm3=anchor_C,measured_nH_cm3=base["nH_cm3"]*10**.12)
    adjusted=predictor.predict("Si",2e18,anchors=[anchor])
    lam=adjusted["shrinkage_lambda"]
    expected=0. if lam=="infinity" else .12/(1+lam)
    assert abs(adjusted["observation_offset_dex"]-expected)<1e-9
    refusals=[]
    for label,kwargs in (
        ("unsupported rare earth",dict(dopant="Er",incorporated_cm3=1e18)),
        ("negative chemical dose",dict(dopant="Si",incorporated_cm3=-1)),
        ("three anchors",dict(dopant="Si",incorporated_cm3=1e18,anchors=[anchor]*3)),
        ("duplicate anchor doses",dict(dopant="Si",incorporated_cm3=1e18,anchors=[anchor]*2))):
        try:predictor.predict(**kwargs)
        except ValueError as exc:refusals.append(dict(case=label,reason=str(exc)))
        else:raise AssertionError("Missing input/domain rejection: "+label)
    # CLI path and JSON serialization must work from the project tree.
    example=OUT/"example_Si_1e18.json"
    cmd=[sys.executable,str(ROOT/"scripts/trainable_sandbox_20261017/predict_properties.py"),
         "--dopant","Si","--incorporated-cm3","1e18","--output",str(example)]
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
    parsed=json.loads(example.read_text())
    assert parsed["status"]=="conditional_research_prediction"
    assert abs(parsed["log10_nH_cm3"]-base["log10_nH_cm3"])<2e-6
    energy.write(OUT/"verification.json",dict(passed=True,
        predictor_code_sha256=energy.sha(ROOT/"src/sandbox/trainable_predictor.py"),
        cli_code_sha256=energy.sha(ROOT/"scripts/trainable_sandbox_20261017/predict_properties.py"),
        trained_bundle_manifest_sha256=energy.sha(OUT/"manifest.json"),
        saved_fit_replays=checks,synthetic_anchor_offset_error=abs(adjusted["observation_offset_dex"]-expected),
        required_refusals=refusals,cli_json_roundtrip=True,
        scope="integration/numerical checks, not new materials experiments"))
    print("8 saved-fit replays, dose conservation, conductivity identity, anchor and refusal/CLI checks passed")

if __name__=="__main__":main()
