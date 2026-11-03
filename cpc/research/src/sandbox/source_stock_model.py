"""Saved shared relation -> conservative Sandbox -> source-adapted properties.

This inference layer reads no training/validation tables or measurement labels.
Its source coordinate is an effective calibration relative to frozen parameters,
not a uniquely identified physical process variable.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from .parent_allocation import ParentAllocationLayer
from .material_profile import MaterialProfile
from .kroger_db import load

LN10=float(np.log(10.));L0=float(np.log10(600.));L1=float(np.log10(1950.))


@dataclass
class SourcePrediction:
    values: np.ndarray
    source_jacobian: np.ndarray
    physical_values: np.ndarray
    diagnostics: dict


class SourceStockModel:
    def __init__(self,parameters,definition,feature_metadata,*,profile:MaterialProfile,pO2_atm,db=None):
        if not isinstance(profile,MaterialProfile):raise TypeError("Explicit immutable material profile required")
        if not (np.isfinite(pO2_atm) and pO2_atm>0):raise ValueError("Explicit positive pressure scenario required")
        self.parameters=np.asarray(parameters,float).copy();self.parameters.setflags(write=False)
        self.definition=dict(definition);self.metadata=dict(feature_metadata);self.profile=profile
        self.pO2_atm=float(pO2_atm);self.db=load() if db is None else db;self.layers={};self.calls=0
        self.nbase=4 if definition["training"]=="joint" else 2
        field=definition["field"]
        if field not in ("none","parent","Tf","output"):raise ValueError("Physical/shared-field model required")
        k=0 if field=="none" else {"constant":1,"linear":4,"rbf":7}[definition["features"]]
        if self.parameters.shape!=(self.nbase+k,) or not np.isfinite(self.parameters).all():
            raise ValueError("Parameter/definition dimensions differ")

    def features(self,row):
        kind=self.definition.get("features")
        if kind is None:return np.empty(0)
        x=(np.log10(row.incorporated_cm3)-self.metadata["logC_center"])/self.metadata["logC_scale"]
        g=(row.growth_T_C-800.)/200.;sn=float(row.element=="Sn")
        if kind=="constant":return np.ones(1)
        if kind=="linear":return np.array([1.,x,sn,g])
        return np.array([1.,x,np.exp(-.5*(x+1)**2),np.exp(-.5*x*x),np.exp(-.5*(x-1)**2),sn,g])

    def predict(self,row,*,source_shift=0.,source_kind="none"):
        z=float(source_shift)
        if not np.isfinite(z):raise ValueError("Finite source coordinate required")
        if source_kind not in ("none","offset","stock","Tf"):raise ValueError("Unknown source coordinate")
        if source_kind=="none" and z!=0:raise ValueError("Nonzero shift needs an explicit coordinate")
        if row.element not in ("Si","Sn"):raise ValueError("Audited Si/Sn domain only")
        values=[row.incorporated_cm3,row.growth_T_C,row.Tobs_K,row.H_cm3]
        if not np.isfinite(values).all() or row.incorporated_cm3<=0 or row.H_cm3<0:
            raise ValueError("Finite positive chemical dose and explicit temperature/process/H inputs required")
        if row.H_known and row.H_cm3<=0:raise ValueError("Known H requires a positive measured supply")
        if not row.H_known and row.H_cm3!=0:raise ValueError("Unknown H must explicitly use the zero-supply scenario")
        if source_kind in ("stock","Tf") and abs(z)>6+1e-12:raise ValueError("Outside declared internal-coordinate range")
        p=self.parameters;field=self.definition["field"];phi=self.features(row)
        v=float(phi@p[self.nbase:]) if len(phi) else 0.
        b=24*np.tanh(v/24);db=0.
        if source_kind=="stock":
            if field not in ("none","parent"):raise ValueError("Stock adaptation requires no field or a saved stock field")
            v+=LN10*z;b=24*np.tanh(v/24);db=LN10*(1-np.tanh(v/24)**2)
        coordinate=p[0]+(b/4 if field=="Tf" else 0.)
        if source_kind=="Tf":
            if field!="none":raise ValueError("Matched Tf source control uses a no-field base")
            bz=24*np.tanh(LN10*z/24);db=LN10*(1-np.tanh(LN10*z/24)**2)
            coordinate+=bz/4
        st=expit(coordinate);Tf=10**(L0+(L1-L0)*st)
        E=10**p[2] if self.nbase==4 else .0366;s=p[3] if self.nbase==4 else 0.
        elements=(row.element,"H") if row.H_known else (row.element,)
        if elements not in self.layers:
            self.layers[elements]=ParentAllocationLayer(self.db,elements,profile=self.profile,extension=self.definition["thermal"])
        layer=self.layers[elements]
        dose=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,)
        bias=b if field=="parent" or source_kind=="stock" else 0.
        self.calls+=1
        pred,J,_=layer.point(Tf_K=Tf,background_cm3=10**p[1],doses=dose,Tobs_K=row.Tobs_K,
            E0_eV=E,s=s,bias=bias,pO2_atm=self.pO2_atm)
        if not pred.diagnostics["single_electron_Hall_valid"]:raise ValueError("Outside single-electron Hall observation domain")
        physical=pred.values.copy();out=physical.copy();jac=np.zeros(3)
        if field=="output":out[0]-=b/LN10
        if source_kind=="stock":jac=J[:,4]*db
        elif source_kind=="Tf":jac=J[:,0]*(L1-L0)*st*(1-st)*db/4
        elif source_kind=="offset":out[0]-=z;jac[0]=-1.
        dg={**pred.diagnostics,"source_kind":source_kind,"source_shift":z,"process_bias":bias,
            "learned_field":field,"same_physical_state_for_output":field!="output" and not(source_kind=="offset" and z!=0),
            "interpretation":"saved shared conditional relation and effective source adjustment,not microscopic parameter truth"}
        return SourcePrediction(out,jac,physical,dg)

    def counters(self):
        keys=("root_calls","hot_solve_calls","hot_cache_hits","root_evaluations")
        return {"inference_calls":self.calls,**{k:sum(getattr(l.core,k) for l in self.layers.values()) for k in keys}}
