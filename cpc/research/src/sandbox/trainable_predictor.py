"""Versioned ML-energy -> defect Sandbox -> Hall-property research predictor.

Numeric third-party energies remain external. A model bundle supplies trained
learners, the known conditional catalogue and fitted process/observation rules.
This API does not certify new dopants, real kinetic growth or calibrated CIs.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, pickle
import numpy as np
from scipy.optimize import brentq
from .energy_learning import reconstruct
from .property_metric_learning import element_database
from .implicit_defects import ImplicitDefectLayer, ImplicitDomainError
from . import transport

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def database_fingerprint(db):
    h=hashlib.sha256(json.dumps(dict(elements=list(db.elements),nCS=int(db.nCS),
                                    nD=int(db.nD)),sort_keys=True).encode())
    for a in (db.dHo,db.charge,db.dm,db.cs_ID,db.prefactor,db.sum_dm):
        b=np.asarray(a,dtype="<f8")
        h.update(str(b.shape).encode());h.update(b.tobytes())
    return h.hexdigest()

def _ct_from_log_hall(log_nH):
    def residual(log_n):
        return log_n-np.log10(float(transport.hall_factor(300.,10.**log_n)))-log_nH
    free_n=10.**brentq(residual,log_nH-1.,log_nH+1.,xtol=1e-12)
    return float(np.log10(transport.mu_caughey_thomas(free_n,300.)))

class DopedPropertyPredictor:
    """Small-data conditional predictor with explicit 0/1/2 Hall anchors.

    Load bundles generated locally by build_predictor.py. They use pickle and
    should not be loaded from untrusted sources. Passing another energy library
    requires rebuilding/revalidating the bundle.
    """
    def __init__(self,db,bundle):
        self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/"manifest.json").read_text())
        if database_fingerprint(db)!=self.manifest["database_fingerprint"]:
            raise ValueError("The supplied energy database differs from the trained bundle")
        for entry in self.manifest["files"]:
            p=(self.bundle/entry["path"]).resolve()
            if self.bundle not in p.parents or file_sha256(p)!=entry["sha256"]:
                raise ValueError("Bundle file or checksum mismatch: "+entry["path"])
        self.db=db;self.layers={};self.learners={}

    def _layer(self,dopant,method):
        key=(dopant,method)
        if key not in self.layers:
            db=self.db
            if method!="full_library":
                info=self.manifest["energy_models"][dopant][method]
                with (self.bundle/info["path"]).open("rb") as f:
                    learner=pickle.load(f)
                if dopant in learner.training_elements:
                    raise ValueError("Held-element prediction would use target-element training labels")
                records=self.manifest["target_records"][dopant]
                forbidden={"crossing_eV","target_energy_eV"}
                if any(forbidden.intersection(r) for r in records):
                    raise ValueError("Hidden target energy labels entered the inference catalogue")
                prediction=np.asarray(learner.predict(records),float)
                if prediction.shape!=(len(records),) or not np.isfinite(prediction).all():
                    raise ValueError("Energy learner returned invalid predictions")
                db=reconstruct(db,records,prediction)
                self.learners[key]=learner
            db,_=element_database(db,dopant)
            self.layers[key]=ImplicitDefectLayer(db,(dopant,))
        return self.layers[key]

    def predict(self,dopant,incorporated_cm3,*,method="nonlinear_kernel",anchors=(),
                pO2_atm=None):
        """Predict room-temperature nH, empirical muH and derived sigma.

        anchors: zero to two same-condition dictionaries with incorporated_cm3
        and measured_nH_cm3. Their Hall values calibrate an observation offset;
        they do not define a new charge-conserving microscopic state.
        """
        if dopant not in self.manifest["supported_dopants"]:
            raise ValueError("Validated bundle scope is Si/Sn; no rare-earth inference is supplied")
        if method not in self.manifest["methods"]:
            raise ValueError("Unknown bundled model: "+method)
        C=float(incorporated_cm3)
        if not np.isfinite(C) or C<=0:
            raise ValueError("incorporated_cm3 must be positive and finite")
        anchors=list(anchors)
        if len(anchors)>2:
            raise ValueError("The current adaptation protocol supports 0, 1 or 2 anchors")
        doses=[];measurements=[]
        for a in anchors:
            c=float(a["incorporated_cm3"]);n=float(a["measured_nH_cm3"])
            if not (np.isfinite(c) and np.isfinite(n) and c>0 and n>0):
                raise ValueError("Each anchor requires positive finite chemical dose and Hall density")
            doses.append(c);measurements.append(np.log10(n))
        if len(set(doses))!=len(doses):
            raise ValueError("The two-anchor protocol requires distinct chemical doses")
        process=self.manifest["process_models"][dopant][method]
        pressure=self.manifest["fixed_pressure_atm"] if pO2_atm is None else float(pO2_atm)
        if not np.isfinite(pressure) or pressure<=0:
            raise ValueError("pO2_atm must be positive and finite")
        layer=self._layer(dopant,method)
        def forward(c):
            return layer.predict(layer.parameters(T=process["effective_Tf_K"],
                Nd=process["effective_Nd_cm3"],doses=(c,),pO2=pressure))
        solved_anchors=[forward(c) for c in doses]
        raw=forward(C)
        all_states=solved_anchors+[raw]
        usable=all(v.diagnostics["single_electron_Hall_valid"] and
                   v.diagnostics["above_carrier_detection_floor"] for v in all_states)
        if not usable:
            raise ImplicitDomainError("Single-electron Hall/detection-domain gate failed")
        chosen=process["anchor_shrinkage"][str(len(anchors))]
        lam=np.inf if chosen=="infinity" else float(chosen)
        delta=0. if not anchors or np.isinf(lam) else float(
            (np.asarray(measurements)-np.array([v.values[0] for v in solved_anchors])).sum()/
            (len(anchors)+lam))
        log_nH=float(raw.values[0]+delta)
        if log_nH<13.:
            raise ImplicitDomainError("Observation-calibrated Hall density is below the declared floor")
        log_mu=_ct_from_log_hall(log_nH)+self.manifest["transport"]["offset_dex"]
        log_sigma=float(np.log10(1.602176634e-19)+log_nH+log_mu)
        support=self.manifest["chemical_support_cm3"][dopant]
        return dict(dopant=dopant,method=method,incorporated_cm3=C,observation_T_K=300.,
            nH_cm3=10.**log_nH,muH_cm2Vs=10.**log_mu,sigma_S_cm=10.**log_sigma,
            log10_nH_cm3=log_nH,log10_muH_cm2Vs=float(log_mu),log10_sigma_S_cm=log_sigma,
            raw_Sandbox_log10_nH_cm3=float(raw.values[0]),
            raw_Sandbox_diagnostics=raw.diagnostics,
            observation_offset_dex=delta,anchor_count=len(anchors),shrinkage_lambda=chosen,
            process=dict(effective_Tf_K=process["effective_Tf_K"],
                         effective_Nd_cm3=process["effective_Nd_cm3"],
                         training_records=process["training_records"],
                         training_families=process["training_families"]),
            oxygen=dict(pO2_atm=pressure,status="effective-O2 scenario; unmeasured in calibration cohort",
                        matches_validation_scenario=bool(np.isclose(np.log10(pressure),-5.,atol=1e-12))),
            energy_budget=self.manifest["energy_budget"][dopant][method],
            outside_training_chemical_range=not support[0]<=C<=support[1],
            chemical_training_range_cm3=support,
            scope=dict(energy="conditional completion with known catalogue/reference/unmasked energies",
                       learning="full nonlinear training used virtual Sandbox targets, not new measured properties"
                           if method=="nonlinear_kernel" else "see bundled training protocol",
                       process="shared effective freeze closure, not actual growth-temperature/oxygen kinetics",
                       observation_offset="Hall readout correction; raw conserved state reported separately",
                       mobility="empirical CT+learned scalar head; complete-chain accuracy remains limited",
                       conductivity="derived q*nH*muH, not independent validation",
                       uncertainty="no calibrated confidence interval supplied"),
            evidence=self.manifest["validation_evidence"])
