"""Integration of tuned energy Ridge and exact inheritance of earlier modes."""
from pathlib import Path
from dataclasses import replace
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.kroger_db import load
from sandbox.trainable_predictor_v4 import DopedPropertyPredictor
from sandbox.trainable_predictor_v3 import DopedPropertyPredictor as Previous
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"predictor_bundle_v4"
def main():
    db=load();model=DopedPropertyPredictor(db,OUT)
    previous=Previous(db,energy.BASE/"predictor_bundle_v3")
    inherited=[];new=[];dataset=read_result_csv(energy.BASE/"dataset_v1.csv")
    for e in ("Si","Sn"):
        for method in previous.manifest["methods"]:
            a=previous.predict(e,1e18,method=method);b=model.predict(e,1e18,method=method)
            diff=max(abs(a[k]-b[k]) for k in ("log10_nH_cm3","log10_muH_cm2Vs","log10_sigma_S_cm"))
            assert diff<2e-10;inherited.append(dict(element=e,method=method,max_error_dex=diff))
        row=dataset[dataset.element.eq(e)].iloc[0]
        key=model.manifest["process_models"][e]["ridge_energy_tuned"]["fit_key"]
        fit=json.loads((OUT/"training/fits"/f"{key}.json").read_text())
        expected=next(p for p in fit["predictions"] if p["sample_id"]==row.sample_id)
        result=model.predict(e,row.incorporated_cm3)
        a=db.dHo.copy();a[[p["index"] for p in model.manifest["target_records"][e]]]=np.nan
        masked=DopedPropertyPredictor(replace(db,dHo=a),OUT)
        output=masked.predict(e,row.incorporated_cm3)
        error=abs(result["log10_nH_cm3"]-expected["predicted_log_nH"])
        erasure=max(abs(result[k]-output[k]) for k in ("log10_nH_cm3","log10_muH_cm2Vs","log10_sigma_S_cm"))
        assert error<2e-6 and erasure<2e-10
        assert result["energy_budget"]["extra_revealed_energies"]==0
        assert result["supplied_energy_observations"]==[]
        new.append(dict(element=e,saved_fit_error_dex=error,hidden_energy_erasure_error_dex=erasure))
    p=OUT/"example_Si_1e18.json"
    subprocess.run([sys.executable,str(ROOT/"scripts/trainable_sandbox_20261017/predict_properties_v4.py"),
        "--dopant","Si","--incorporated-cm3","1e18","--output",str(p)],cwd=ROOT,check=True)
    assert json.loads(p.read_text())["method"]=="ridge_energy_tuned"
    energy.write(OUT/"verification.json",dict(passed=True,inherited_modes=inherited,tuned_ridge=new,
        cli_default_verified=True,material_validation=False,
        runtime_sha256=energy.sha(ROOT/"src/sandbox/trainable_predictor_v4.py"),
        manifest_sha256=energy.sha(OUT/"manifest.json")))
    print("12 inherited-mode replays,2tuned-Ridge saved/erased replays and CLI passed")

if __name__=="__main__":main()
