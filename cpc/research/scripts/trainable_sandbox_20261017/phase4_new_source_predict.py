"""Same frozen Azizie2023 held-study test for the nonlinear energy learner."""
from pathlib import Path
import sys, json, shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.property_metric_learning import element_database
from sandbox.implicit_defects import ImplicitDefectLayer
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase2_property as fitter
import phase3_new_source as source
import phase3_new_source_predict as previous
import phase4_property as models

OUT=models.training.OUT/"new_property_source"
METHODS=models.METHODS

def freeze():
    models.freeze()
    payload=dict(version=1,created_utc=energy.now(),methods=METHODS,
        source_contract_sha256=energy.sha(source.OUT/"azizie2023_validation_protocol.json"),
        previous_prediction_sha256=energy.sha(source.OUT/"azizie2023_prediction_protocol.json"),
        nested_training_sha256=energy.sha(models.training.OUT/"nested_protocol.json"),
        real_validation_sha256=energy.sha(models.OUT/"protocol.json"),
        code_sha256=energy.sha(Path(__file__)),
        driver_code_sha256=energy.sha(Path(previous.__file__)),
        reuse="exact previous sample/chemical rounding/related-source exclusion/transport protocol",
        exclusion=source.RELATED,
        selection="old four-source fitted process and inner-selected shrinkage; no new-study tuning",
        caveat="source labels were already exposed in previous phase; retrospective confirmation only")
    p=OUT/"protocol.json"
    if p.exists():
        old=json.loads(p.read_text())
        assert old["code_sha256"]==payload["code_sha256"]
        assert old["real_validation_sha256"]==payload["real_validation_sha256"]
    else:energy.write(p,payload)
    p=OUT/"azizie2023_audited_samples.csv"
    original=source.OUT/p.name
    if p.exists():assert energy.sha(p)==energy.sha(original)
    else:shutil.copyfile(original,p)

def library_and_fold(method):
    data=read_result_csv(models.OUT/"outer"/f"{fitter.digest(source.RELATED)}__{method}.csv")
    key=data.fit_key.iloc[0]
    fit=json.loads((models.OUT/"fits"/f"{key}.json").read_text())
    assert source.RELATED not in fit["train_families"] and source.DOI not in fit["train_families"]
    choices=json.loads((models.OUT/"selections"/f"{fitter.digest(source.RELATED)}__{method}.json").read_text())
    lam={int(x["budget"]):(np.inf if x["selected"]["lam"] is None else x["selected"]["lam"])
         for x in choices}
    db,_=element_database(models.database("Si",method),"Si")
    return ImplicitDefectLayer(db,("Si",)),fit,lam,0

if __name__=="__main__":
    previous.OUT=OUT;previous.METHODS=METHODS
    previous.freeze=freeze;previous.library_and_fold=library_and_fold
    previous.main()
