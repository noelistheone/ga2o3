"""Add the validated, energy-selected zero-extra-label Ridge option."""
from pathlib import Path
import json,pickle
import numpy as np
from .trainable_predictor_v3 import DopedPropertyPredictor as _V3
from .trainable_predictor_v2 import structure_fingerprint
from .trainable_predictor import file_sha256
from .energy_learning import features,reconstruct
from .property_metric_learning import element_database
from .implicit_defects import ImplicitDefectLayer

class BaseFeatureBank:
    def descriptor(self,records,kind):
        if kind!="base":raise ValueError("This runtime path supplies base features only")
        return features(records)

class DopedPropertyPredictor(_V3):
    def __init__(self,db,bundle):
        self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/"manifest.json").read_text())
        if self.manifest.get("version")!=4:raise ValueError("Version4 input contract required")
        if structure_fingerprint(db)!=self.manifest["structure_fingerprint"]:
            raise ValueError("Supplied catalogue or prefactors differ")
        for entry in self.manifest["files"]:
            path=(self.bundle/entry["path"]).resolve()
            if self.bundle not in path.parents or file_sha256(path)!=entry["sha256"]:
                raise ValueError("Bundle file or checksum mismatch: "+entry["path"])
        self.db=db;self.layers={};self.learners={};self.checked_inputs=set()

    def _layer(self,dopant,method):
        info=self.manifest["energy_models"].get(dopant,{}).get(method,{})
        if info.get("prediction_interface")!="explicit_base_bank":
            return super()._layer(dopant,method)
        key=(dopant,method)
        if key not in self.layers:
            with (self.bundle/info["path"]).open("rb") as f:model=pickle.load(f)
            if dopant in model.training_elements or model.kind!="base":
                raise ValueError("Target-element or descriptor contract violation")
            records=self.manifest["target_records"][dopant]
            if any({"crossing_eV","target_energy_eV"}.intersection(r) for r in records):
                raise ValueError("Unknown target labels entered inference")
            values=np.asarray(model.predict(records,BaseFeatureBank()),float)
            if not np.isfinite(values).all():raise ValueError("Invalid energy prediction")
            db=reconstruct(self.db,records,values);db,_=element_database(db,dopant)
            self.layers[key]=ImplicitDefectLayer(db,(dopant,));self.learners[key]=model
        return self.layers[key]

    def predict(self,dopant,incorporated_cm3,*,method="ridge_energy_tuned",anchors=(),pO2_atm=None):
        result=super().predict(dopant,incorporated_cm3,method=method,anchors=anchors,pO2_atm=pO2_atm)
        if method=="ridge_energy_tuned":
            result["scope"]["learning"]="training-only energy-selected weighted Ridge, target element excluded, zero additional energetic observations"
        return result
