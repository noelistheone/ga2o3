"""Reusable parameter-to-Sandbox map for shared donor-fraction learning.

No property tables or observed labels are read by this module. The caller
supplies physical inputs and parameters learned under its validation protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from .donor_mixture import DonorMixtureLayer
from .parent_allocation import isolated_parent_contrast
from .material_profile import MaterialProfile
from .kroger_db import load
from .thermo import KB_EV

L0,L1=np.log10(600.),np.log10(1950.);LN10=float(np.log(10.))


@dataclass
class MixtureModelPrediction:
    values:np.ndarray
    jacobian:np.ndarray
    source_jacobian:np.ndarray
    physical_values:np.ndarray
    diagnostics:dict


class DonorMixtureModel:
    def __init__(self,definition,*,profile:MaterialProfile,pO2_atm,db=None):
        if not isinstance(profile,MaterialProfile):raise TypeError("Explicit immutable material profile required")
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError("Positive explicit pressure scenario required")
        self.definition=dict(definition);self.profile=profile;self.pO2_atm=float(pO2_atm)
        self.db=load() if db is None else db;self.layers={}
        names=["logit_Tf","log10_background"]
        self.iE=None;self.iscreen=None;self.ia=None;self.ib=None;self.iout=None
        binding=self.definition.get("binding","fixed")
        if binding in ("learned","screen"):
            self.iE=len(names);names.append("log10_binding_E0")
        if binding=="screen":self.iscreen=len(names);names.append("binding_screen_s")
        fraction=self.definition.get("fraction")
        if fraction in ("constant","dose"):
            self.ia=len(names);names.append("localized_logit")
        if fraction=="dose":self.ib=len(names);names.append("localized_slope")
        if self.definition.get("output")=="linear":
            self.iout=len(names);names.extend(("output_intercept","output_logdose_slope"))
        self.parameter_names=tuple(names)

    def point(self,row,parameters,*,source_kind="none",source_shift=0.):
        w=np.asarray(parameters,float);z=float(source_shift)
        if w.shape!=(len(self.parameter_names),) or not np.isfinite(w).all():raise ValueError("Finite model parameters required")
        if source_kind not in ("none","offset","stock","localized","Tf"):raise ValueError("Unknown source coordinate")
        if not np.isfinite(z) or (source_kind=="none" and z!=0):raise ValueError("Explicit finite source coordinate required")
        if source_kind in ("stock","localized","Tf") and abs(z)>6+1e-12:raise ValueError("Outside source-coordinate interval")
        if row.element not in ("Si","Sn"):raise ValueError("Only audited Si/Sn donor domain supported")
        if not (np.isfinite([row.incorporated_cm3,row.growth_T_C,row.Tobs_K,row.H_cm3]).all()
            and row.incorporated_cm3>0 and row.H_cm3>=0):raise ValueError("Explicit finite physical inputs required")
        if bool(row.H_known)!=(row.H_cm3>0):raise ValueError("Reported H supply/unknown-zero scenario must be explicit")
        frac=self.definition.get("fraction");fixed=0. if frac=="zero" else (1. if frac=="one" else None)
        elements=(row.element,"H") if row.H_known else (row.element,)
        if elements not in self.layers:
            contrast=isolated_parent_contrast(self.db,elements)
            self.layers[elements]=DonorMixtureLayer(self.db,elements,profile=self.profile,hall="rH1",
                fixed_fraction=fixed,basis=contrast.charge_basis[:,None])
        layer=self.layers[elements]
        bias=0.;dbias=0.;coordinate=w[0]
        if source_kind in ("stock","Tf"):
            bias=24*np.tanh(LN10*z/24);dbias=LN10*(1-np.tanh(LN10*z/24)**2)
            if source_kind=="Tf":coordinate+=bias/4
        st=expit(coordinate);logT=L0+(L1-L0)*st;T=10**logT;dlogT=(L1-L0)*st*(1-st)
        energy_weight=KB_EV*T*bias if source_kind=="stock" else 0.
        a=w[self.ia] if self.ia is not None else 0.;b=w[self.ib] if self.ib is not None else 0.
        if source_kind=="localized":
            if self.ia is None:raise ValueError("Localized source update requires a learned fraction relation")
            a+=LN10*z
        p=layer.parameters(T=T,Nd=10**w[1],weights=[energy_weight],pO2=self.pO2_atm,
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),Tobs=row.Tobs_K,
            binding_eV=10**w[self.iE] if self.iE is not None else .0366,
            screen_s=w[self.iscreen] if self.iscreen is not None else 0.,localized_logit=a,localized_slope=b)
        result=layer.predict(p)
        if not result.diagnostics["single_electron_Hall_valid"]:raise ValueError("Hole-dominated:outside declared Hall operator")
        chain=np.zeros((layer.np,len(w)));chain[0,0]=dlogT;chain[1,1]=1.
        if source_kind=="stock":chain[2,0]=energy_weight*LN10*dlogT
        if self.iE is not None:chain[layer.ie,self.iE]=1.
        if self.iscreen is not None:chain[layer.ie+1,self.iscreen]=1.
        if self.ia is not None:chain[layer.iloc,self.ia]=1.
        if self.ib is not None:chain[layer.iloc+1,self.ib]=1.
        jac=result.jacobian@chain;source=np.zeros(3);physical=result.values.copy();values=physical.copy()
        if self.iout is not None:
            x=np.log10(row.incorporated_cm3)-18.;values[0]-=w[self.iout]+w[self.iout+1]*x
            jac[0,self.iout:self.iout+2]=[-1.,-x]
        if source_kind=="stock":source=result.jacobian[:,2]*(KB_EV*T*dbias)
        elif source_kind=="localized":source=result.jacobian[:,layer.iloc]*LN10
        elif source_kind=="Tf":source=result.jacobian[:,0]*dlogT*dbias/4
        elif source_kind=="offset":values[0]-=z;source[0]=-1.
        dg={**result.diagnostics,"source_kind":source_kind,"source_shift":z,"process_bias":bias if source_kind=="stock" else 0.,
            "same_physical_state_for_output":self.iout is None and not(source_kind=="offset" and z!=0),
            "learning_definition":self.definition,"effective_source_interpretation":"relative to saved shared relation,not measured process truth"}
        return MixtureModelPrediction(values,jac,source,physical,dg)

    def counters(self):
        return {key:sum(getattr(layer,key) for layer in self.layers.values()) for key in
            ("root_calls","hot_solve_calls","hot_cache_hits","root_evaluations")}
