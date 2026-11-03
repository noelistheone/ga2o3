"""Standalone trained effective-relation + defect Sandbox research predictor.

Runtime reads model parameters and the explicitly checksummed energetic input;
it never reads an experimental table. Supports the audited Si/Sn state extension,
with variable-T carrier/charge outputs, not validated variable-T mobility.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import expit
from .kroger_db import load,NPZ
from .material_profile import MaterialProfile
from .temperature_defects import TemperatureDefectLayer
from .parameter_adaptation import regularized_adapt


def serializable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


@dataclass(frozen=True)
class EffectiveThermalState:
    Tf_K: float
    background_cm3: float
    binding_E0_eV: float
    concentration_s: float

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():raise ValueError("Finite state required")
        if not (600<=self.Tf_K<=1950 and self.background_cm3>0 and
                .003<=self.binding_E0_eV<=.2 and 0<=self.concentration_s<=5):
            raise ValueError("Outside declared effective-state scope")


class ThermalDopantPredictor:
    def __init__(self,state,profile,*,energetic_path=NPZ,energetic_sha256=None,pO2_atm=1e-5,metadata=None):
        if not isinstance(state,EffectiveThermalState) or not isinstance(profile,MaterialProfile):
            raise TypeError("Explicit immutable state/profile required")
        self._state=state;self._profile=profile;self._energetic_path=Path(energetic_path)
        digest=hashlib.sha256(self._energetic_path.read_bytes()).hexdigest()
        if energetic_sha256 is not None and energetic_sha256!=digest:raise ValueError("Energetic input checksum mismatch")
        self._energetic_sha256=digest;self._db=load(self._energetic_path);self._layers={}
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError("Explicit positive oxygen-pressure scenario required")
        self.pO2_atm=float(pO2_atm)
        self.metadata=dict(metadata or {})

    @property
    def state(self):return self._state
    @property
    def profile(self):return self._profile

    def _point(self,element,concentration,T_K,H_cm3,theta):
        if element not in ("Si","Sn"):raise ValueError("Current audited extension supports Si/Sn only")
        if not np.isfinite(concentration) or concentration<=0:raise ValueError("Positive actual chemical concentration required")
        if H_cm3 is not None and (not np.isfinite(H_cm3) or H_cm3<0):raise ValueError("Nonnegative H or explicit unknown(None) required")
        has_H=H_cm3 is not None and H_cm3>0
        elements=(element,"H") if has_H else (element,)
        if elements not in self._layers:
            self._layers[elements]=TemperatureDefectLayer(self._db,elements,profile=self.profile,
                donor_extension=True,donor_element=element,hall="rH1")
        layer=self._layers[elements]
        doses=(concentration,H_cm3) if has_H else (concentration,)
        p=layer.parameters(T=10**theta[0],Nd=self.state.background_cm3,doses=doses,Tobs=float(T_K),pO2=self.pO2_atm,
            binding_eV=10**theta[1],screen_s=self.state.concentration_s)
        out=layer.predict(p);J=out.jacobian[:,[0,layer.ie]]
        return out,J

    def predict(self,*,element,concentration_cm3,temperatures_K,H_cm3=None):
        theta=np.log10([self.state.Tf_K,self.state.binding_E0_eV]);rows=[]
        for T in np.atleast_1d(temperatures_K).astype(float):
            a,J=self._point(element,float(concentration_cm3),T,H_cm3,theta)
            rows.append(dict(T_K=float(T),log10_nH_cm3=float(a.values[0]),nH_cm3=float(10**a.values[0]),
                absolute_charge_cm3=float(10**a.values[1]),q2_charge_cm3=float(10**a.values[2]),
                coordinate_jacobian=J,diagnostics=a.diagnostics))
        return dict(predictions=rows,state=asdict(self.state),pO2_atm=self.pO2_atm,profile_sha256=self.profile.fingerprint,
            energetic_sha256=self._energetic_sha256,
            H_status="unknown_conditional_zero" if H_cm3 is None else "user_supplied_not_independently_verified",
            scope="effective localized donor statistics;single-electron unitHall;not validated lowTmobility/Mott transition")

    def adapt(self,anchors,*,regularization,energy_scale=.3,mode="Tf_E",max_nfev=30):
        """Anchors: element,concentration_cm3,T_K,nH_cm3[,H_cm3]."""
        if len(anchors)>2:raise ValueError("This interface declares only 0/1/2-anchor scope")
        if mode not in ("Tf","E","Tf_E"):raise ValueError("Unknown physical adaptation mode")
        y=[]
        for a in anchors:
            if not np.isfinite(a["nH_cm3"]) or a["nH_cm3"]<=0:raise ValueError("Positive measured Hall density required")
            y.append(np.log10(a["nH_cm3"]))
        theta0=np.log10([self.state.Tf_K,self.state.binding_E0_eV])
        D=np.array([.05 if mode in ("Tf","Tf_E") else 0.,energy_scale if mode in ("E","Tf_E") else 0.])
        L0,L1=np.log10(600),np.log10(1950)
        lower=[L0+(L1-L0)*expit(-6),np.log10(.003)]
        upper=[L0+(L1-L0)*expit(6),np.log10(.2)]
        def forward(theta):
            v=[];j=[]
            for a in anchors:
                out,J=self._point(a["element"],a["concentration_cm3"],a["T_K"],a.get("H_cm3"),theta)
                v.append(out.values[0]);j.append(J[0])
            return np.asarray(v),np.asarray(j).reshape(len(anchors),2)
        fitted=regularized_adapt(forward,y,theta0,D,lower,upper,regularization,max_nfev=max_nfev)
        # An allowed log boundary can exponentiate one ULP outside its decimal
        # bound (e.g.10**log10(.2)). Canonicalize only after the bounded solve.
        energy=float(np.clip(10**fitted.parameters[1],.003,.2))
        state=replace(self.state,Tf_K=float(10**fitted.parameters[0]),binding_E0_eV=energy)
        meta=dict(self.metadata)
        meta["adaptation"]=dict(anchor_count=len(anchors),mode=mode,energy_scale=float(energy_scale),
            regularization=float(regularization) if np.isfinite(regularization) else None,
            max_nfev=int(max_nfev),status=fitted.status,
            anchor_sha256=hashlib.sha256(json.dumps(anchors,sort_keys=True,default=serializable).encode()).hexdigest(),
            parent_state=asdict(self.state))
        result=ThermalDopantPredictor(state,self.profile,energetic_path=self._energetic_path,
            energetic_sha256=self._energetic_sha256,pO2_atm=self.pO2_atm,metadata=meta)
        return result,fitted

    def save(self,path):
        payload=dict(schema_version=2,state=asdict(self.state),pO2_atm=self.pO2_atm,profile=self.profile.to_dict(),
            profile_sha256=self.profile.fingerprint,energetic_sha256=self._energetic_sha256,
            shared_relation="E(C)=E0 exp[-s(C/1e18cm^-3)^(1/3)]",metadata=self.metadata,
            scope="research prediction of Si/Sn carrier/charge budgets;no experimental-label lookup or variableTmobility claim")
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=serializable)+"\n")

    @classmethod
    def load(cls,path,*,energetic_path=NPZ):
        payload=json.loads(Path(path).read_text())
        if payload["schema_version"]!=2:raise ValueError("Unsupported model schema;use the explicit-pressure v2 export")
        profile=MaterialProfile.from_dict(payload["profile"])
        if profile.fingerprint!=payload["profile_sha256"]:raise ValueError("Material profile checksum mismatch")
        return cls(EffectiveThermalState(**payload["state"]),profile,energetic_path=energetic_path,pO2_atm=payload["pO2_atm"],
            energetic_sha256=payload["energetic_sha256"],metadata=payload.get("metadata"))
