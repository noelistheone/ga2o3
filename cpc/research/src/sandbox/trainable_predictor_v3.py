"""Add explicit two-energy conditional updates to the trained prediction chain.

Additional observations are supplied published energetic inputs in the bundle,
not values inferred from missing energies, new calculations, or Hall labels.
"""
from pathlib import Path
import json,pickle
import numpy as np
from .trainable_predictor import file_sha256
from .trainable_predictor_v2 import (
    DopedPropertyPredictor as _V2,structure_fingerprint,supplied_energy_fingerprint)
from .energy_learning import reconstruct
from .property_metric_learning import element_database
from .implicit_defects import ImplicitDefectLayer

class DopedPropertyPredictor(_V2):
    def __init__(self,db,bundle):
        self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/"manifest.json").read_text())
        if self.manifest.get("version")!=3:
            raise ValueError("Version3 predictor requires the version3 input contract")
        if structure_fingerprint(db)!=self.manifest["structure_fingerprint"]:
            raise ValueError("Supplied catalogue or prefactors differ from the trained bundle")
        for entry in self.manifest["files"]:
            p=(self.bundle/entry["path"]).resolve()
            if self.bundle not in p.parents or file_sha256(p)!=entry["sha256"]:
                raise ValueError("Bundle file or checksum mismatch: "+entry["path"])
        self.db=db;self.layers={};self.learners={};self.checked_inputs=set()

    def _layer(self,dopant,method):
        info=self.manifest["energy_models"].get(dopant,{}).get(method,{})
        observations=info.get("known_energy_observations",[])
        if not observations:return super()._layer(dopant,method)
        key=(dopant,method)
        if key not in self.layers:
            with (self.bundle/info["path"]).open("rb") as f:learner=pickle.load(f)
            if dopant in learner.training_elements:
                raise ValueError("Target-element labels entered the energy learner")
            records=self.manifest["target_records"][dopant]
            if any({"crossing_eV","target_energy_eV"}.intersection(r) for r in records):
                raise ValueError("An unknown energy label entered the inference catalogue")
            indices={r["label_id"]:i for i,r in enumerate(records)}
            values=np.asarray(learner.predict(records),float)
            if len(observations)!=2 or len({o["label_id"] for o in observations})!=2:
                raise ValueError("The current conditional update requires exactly two supplied energies")
            for obs in observations:
                i=indices[obs["label_id"]];r=records[i]
                if obs["index"]!=r["index"]:raise ValueError("Observed energy index mismatch")
                # The known reference and explicitly supplied intercept define
                # this crossing; missing db.dHo[r['index']] is never read.
                values[i]=-(obs["formation_intercept_eV"]-self.db.dHo[r["reference_index"]])/r["delta_q"]
            if not np.isfinite(values).all():raise ValueError("Nonfinite conditional energy prediction")
            db=reconstruct(self.db,records,values)
            db,_=element_database(db,dopant)
            self.layers[key]=ImplicitDefectLayer(db,(dopant,));self.learners[key]=learner
        return self.layers[key]

    def predict(self,dopant,incorporated_cm3,*,method="goal_ridge_2",anchors=(),pO2_atm=None):
        result=super().predict(dopant,incorporated_cm3,method=method,anchors=anchors,pO2_atm=pO2_atm)
        info=self.manifest["energy_models"].get(dopant,{}).get(method,{})
        result["supplied_energy_observations"]=info.get("known_energy_observations",[])
        if result["supplied_energy_observations"]:
            result["scope"]["learning"]="target-element-excluded energy learner, conditioned on two explicitly supplied published energies"
            result["scope"]["energy_acquisition"]="previously frozen target-directed/variance replay, not new DFT or live experimental acquisition"
        return result
