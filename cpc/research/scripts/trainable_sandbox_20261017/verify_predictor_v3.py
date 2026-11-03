"""Saved-fit and hidden-input checks for the explicit two-energy predictor."""
from pathlib import Path
from dataclasses import replace
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.kroger_db import load
from sandbox.trainable_predictor_v3 import DopedPropertyPredictor
from sandbox.trainable_predictor_v2 import DopedPropertyPredictor as Previous
from csv_io_v2 import read_result_csv
import phase2_energy as energy

OUT=energy.BASE/"predictor_bundle_v3"
def main():
    db=load();model=DopedPropertyPredictor(db,OUT)
    old=Previous(db,energy.BASE/"predictor_bundle_v2")
    data=read_result_csv(energy.BASE/"dataset_v1.csv")
    inherited=[];conditional=[]
    for e in ("Si","Sn"):
        for method in old.manifest["methods"]:
            before=old.predict(e,1e18,method=method);after=model.predict(e,1e18,method=method)
            error=max(abs(before[k]-after[k]) for k in ("log10_nH_cm3","log10_muH_cm2Vs","log10_sigma_S_cm"))
            assert error<2e-10;inherited.append(dict(element=e,method=method,max_difference_dex=error))
        row=data[data.element.eq(e)].iloc[0]
        a=db.dHo.copy();a[[r["index"] for r in model.manifest["target_records"][e]]]=np.nan
        masked=DopedPropertyPredictor(replace(db,dHo=a),OUT)
        for method in ("goal_ridge_2","goal_mean_2"):
            key=model.manifest["process_models"][e][method]["fit_key"]
            fit=json.loads((OUT/"training/fits"/f"{key}.json").read_text())
            reference=next(r for r in fit["predictions"] if r["sample_id"]==row.sample_id)
            result=model.predict(e,float(row.incorporated_cm3),method=method)
            erased=masked.predict(e,float(row.incorporated_cm3),method=method)
            replay=abs(result["log10_nH_cm3"]-reference["predicted_log_nH"])
            erase=max(abs(erased[k]-result[k]) for k in ("log10_nH_cm3","log10_muH_cm2Vs","log10_sigma_S_cm"))
            assert replay<2e-6 and erase<2e-10
            assert len(result["supplied_energy_observations"])==2
            assert result["energy_budget"]["extra_revealed_energies"]==2
            assert all(not o["new_calculation"] for o in result["supplied_energy_observations"])
            assert abs(result["sigma_S_cm"]/(1.602176634e-19*result["nH_cm3"]*result["muH_cm2Vs"])-1)<1e-12
            conditional.append(dict(element=e,method=method,saved_fit_error_dex=replay,
                all_target_energies_erased_error_dex=erase,provided_observations=2))
    example=OUT/"example_Si_1e18.json"
    subprocess.run([sys.executable,str(ROOT/"scripts/trainable_sandbox_20261017/predict_properties_v3.py"),
        "--dopant","Si","--incorporated-cm3","1e18","--output",str(example)],cwd=ROOT,check=True)
    assert json.loads(example.read_text())["method"]=="goal_ridge_2"
    energy.write(OUT/"verification.json",dict(passed=True,inherited_replays=inherited,
        conditional_replays=conditional,CLI=True,material_validation=False,
        runtime_sha256=energy.sha(ROOT/"src/sandbox/trainable_predictor_v3.py"),
        manifest_sha256=energy.sha(OUT/"manifest.json"),
        inference="no unknown db target energies; two additional known entries explicitly supplied in manifest"))
    print("8 inherited replays,4 two-energy saved/erased-input replays and CLI passed")

if __name__=="__main__":main()
