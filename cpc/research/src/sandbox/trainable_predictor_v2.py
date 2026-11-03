"""Inference checks use known inputs only; held energy values may be NaN.

Version1 checked the complete energy database at construction. This version
keeps its forward equations but checks only supplied energetic entries for
each requested dopant/method. The full-library reference still requires all.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .trainable_predictor import DopedPropertyPredictor as _V1, file_sha256

def structure_fingerprint(db):
    h=hashlib.sha256(json.dumps(dict(elements=list(db.elements),nCS=int(db.nCS),
                                    nD=int(db.nD)),sort_keys=True).encode())
    for a in (db.charge,db.dm,db.cs_ID,db.prefactor,db.sum_dm):
        b=np.asarray(a,dtype="<f8");h.update(str(b.shape).encode());h.update(b.tobytes())
    return h.hexdigest()

def supplied_energy_fingerprint(db,hidden_indices=()):
    keep=np.ones(db.nCS,dtype=bool);keep[list(hidden_indices)]=False
    indices=np.flatnonzero(keep).astype("<i8")
    h=hashlib.sha256(indices.tobytes())
    h.update(np.asarray(db.dHo[keep],dtype="<f8").tobytes())
    return h.hexdigest()

class DopedPropertyPredictor(_V1):
    def __init__(self,db,bundle):
        self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/"manifest.json").read_text())
        if self.manifest.get("version")!=2:
            raise ValueError("Version2 predictor requires the version2 input contract")
        if structure_fingerprint(db)!=self.manifest["structure_fingerprint"]:
            raise ValueError("The supplied defect catalogue/prefactors differ from the trained bundle")
        for entry in self.manifest["files"]:
            p=(self.bundle/entry["path"]).resolve()
            if self.bundle not in p.parents or file_sha256(p)!=entry["sha256"]:
                raise ValueError("Bundle file or checksum mismatch: "+entry["path"])
        self.db=db;self.layers={};self.learners={};self.checked_inputs=set()

    def predict(self,dopant,incorporated_cm3,*,method="nonlinear_kernel",anchors=(),pO2_atm=None):
        if dopant in self.manifest["supported_dopants"] and method in self.manifest["methods"]:
            key=(dopant,method)
            if key not in self.checked_inputs:
                hidden=[] if method=="full_library" else [
                    int(r["index"]) for r in self.manifest["target_records"][dopant]]
                expected=self.manifest["supplied_energy_fingerprints"][dopant][method]
                if supplied_energy_fingerprint(self.db,hidden)!=expected:
                    raise ValueError("A required supplied energy differs or is missing")
                self.checked_inputs.add(key)
        return super().predict(dopant,incorporated_cm3,method=method,anchors=anchors,pO2_atm=pO2_atm)
