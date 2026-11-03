"""Held-energy erasure and known-input invariance for conditional inference."""
from pathlib import Path
from dataclasses import replace
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.kroger_db import load
from sandbox.trainable_predictor_v2 import DopedPropertyPredictor
from sandbox.trainable_predictor import DopedPropertyPredictor as Original
import phase2_energy as energy

OUT=energy.BASE/"predictor_bundle_v2"

def main():
    db=load();old=Original(db,energy.BASE/"predictor_bundle_v1")
    manifest=json.loads((OUT/"manifest.json").read_text());rows=[];refusals=[]
    for e in ("Si","Sn"):
        hidden=[int(r["index"]) for r in manifest["target_records"][e]]
        a=db.dHo.copy();a[hidden]=np.nan
        masked=replace(db,dHo=a)
        predictor=DopedPropertyPredictor(masked,OUT)
        for method in ("nonlinear_kernel","ordinary_kernel","ridge"):
            expected=old.predict(e,1e18,method=method)
            result=predictor.predict(e,1e18,method=method)
            error=max(abs(result[k]-expected[k]) for k in
                ("log10_nH_cm3","log10_muH_cm2Vs","log10_sigma_S_cm"))
            assert error<2e-10
            rows.append(dict(element=e,method=method,erased_entries=len(hidden),
                             prediction_max_difference_dex=error))
        try:predictor.predict(e,1e18,method="full_library")
        except ValueError as exc:refusals.append(dict(element=e,mode="full_library",reason=str(exc)))
        else:raise AssertionError("Full-information reference used missing energies")
        known=next(i for i in range(db.nCS) if i not in hidden)
        a=db.dHo.copy();a[known]+=.1
        predictor=DopedPropertyPredictor(replace(db,dHo=a),OUT)
        try:predictor.predict(e,1e18)
        except ValueError as exc:refusals.append(dict(element=e,mode="changed supplied input",reason=str(exc)))
        else:raise AssertionError("Unchanged-input model accepted altered known energy")
    path=OUT/"example_Si_1e18.json"
    subprocess.run([sys.executable,str(ROOT/"scripts/trainable_sandbox_20261017/predict_properties_v2.py"),
        "--dopant","Si","--incorporated-cm3","1e18","--output",str(path)],cwd=ROOT,check=True)
    assert json.loads(path.read_text())["status"]=="conditional_research_prediction"
    energy.write(OUT/"verification.json",dict(passed=True,held_energy_erasure_replays=rows,
        required_refusals=refusals,version1_predictions_unchanged=True,
        material_validation=False,cli_roundtrip=True,
        manifest_sha256=energy.sha(OUT/"manifest.json"),
        runtime_sha256=energy.sha(ROOT/"src/sandbox/trainable_predictor_v2.py")))
    print("Si29/Sn23 hidden energies erased: all6 ML replays unchanged; required-input refusals passed")

if __name__=="__main__":main()
