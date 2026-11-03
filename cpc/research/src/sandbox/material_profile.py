"""Immutable band/material input contracts; no global material-setting mutation."""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import hashlib,json
import numpy as np
from . import thermo
from .implicit_defects import _fd_log,LN10

@dataclass(frozen=True)
class MaterialProfile:
    name: str
    Nc_ref: float
    Nv_ref: float
    Eg0: float
    Ec_fraction: float=.4
    T_ref: float=300.
    varshni_a: float=.00105248259206626
    varshni_b: float=676.975385507958
    N_site: float=1.91e22
    STH_energies: tuple=(.52,.53)
    vib_T0: float=870.
    energy_reference: str="KROGER supplied formation inputs;zero0KVBM;no new realignment"
    provenance: str="explicit conditional material scenario"

    def __post_init__(self):
        if not np.isfinite([self.Nc_ref,self.Nv_ref,self.Eg0,self.Ec_fraction,self.T_ref,
                            self.varshni_a,self.varshni_b,self.N_site,self.vib_T0,*self.STH_energies]).all():
            raise ValueError("Finite material constants required")
        if not (self.Nc_ref>0 and self.Nv_ref>0 and self.Eg0>0 and
                0<=self.Ec_fraction<=1 and self.T_ref>0 and self.varshni_b>0 and self.vib_T0>0
                and self.N_site>0 and self.varshni_a>=0):
            raise ValueError("Invalid material profile")
        if not isinstance(self.STH_energies,tuple) or len(self.STH_energies)!=2:
            raise ValueError("Two immutable STH relaxation energies required")

    def to_dict(self):return asdict(self)
    @classmethod
    def from_dict(cls,payload):
        values=dict(payload)
        if "STH_energies" in values:values["STH_energies"]=tuple(values["STH_energies"])
        return cls(**values)
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def shifted(self,deltas):
        """Four differentiable coordinates, not automatically trainable physics."""
        dc,dv,dg,df=map(float,deltas)
        return replace(self,Nc_ref=self.Nc_ref*10**dc,Nv_ref=self.Nv_ref*10**dv,
                       Eg0=self.Eg0+dg,Ec_fraction=self.Ec_fraction+df)

    def bands(self,T):
        gap=self.varshni_a*T*T/(T+self.varshni_b)
        dg=self.varshni_a*T*(T+2*self.varshni_b)/(T+self.varshni_b)**2
        Ec=self.Eg0-self.Ec_fraction*gap;Ev=(1-self.Ec_fraction)*gap
        dEc=-self.Ec_fraction*dg;dEv=(1-self.Ec_fraction)*dg
        # Columns delta_logNc,delta_logNv,deltaEg,deltaf.
        mEc=np.array([0.,0.,1.,-gap]);mEv=np.array([0.,0.,0.,-gap])
        return Ev,Ec,dEv,dEc,mEv,mEc

    def carriers(self,EF,T):
        """n,p,STH1,STH2; EF,T and four explicit material partials."""
        Ev,Ec,dEv,dEc,mEv,mEc=self.bands(T);beta=1/(thermo.KB_EV*T)
        etas=np.array([(EF-Ec)*beta,(Ev-EF)*beta,
            (Ev+self.STH_energies[0]-EF)*beta,(Ev+self.STH_energies[1]-EF)*beta])
        logs,g=_fd_log(etas)
        pref=np.array([self.Nc_ref*(T/self.T_ref)**1.5,self.Nv_ref*(T/self.T_ref)**1.5,
                       self.N_site,self.N_site])
        values=pref*np.exp(logs)
        ef=values*g*beta*np.array([1.,-1.,-1.,-1.])
        dteta=np.array([-dEc,dEv,dEv,dEv])*beta-etas/T
        dt=values*(g*dteta+np.array([1.5/T,1.5/T,0.,0.]))
        dm=values[:,None]*g[:,None]*beta*np.vstack([-mEc,mEv,mEv,mEv])
        dm[0,0]+=values[0]*LN10;dm[1,1]+=values[1]*LN10
        return values,ef,dt,dm

REPOSITORY_PROFILE=MaterialProfile("repository_default",1e19,5e20,5.,
    provenance="cached KROGER materialconditions defaults;not finalpublication parameter reproduction")
PUBLISHED_PROFILE=MaterialProfile("published_DOS_gap",3.33e18,1.71e20,5.1,
    provenance="PCCP2025 DOI10.1039/D4CP04817B Temperaturedependenceofbandgap;conditional same energy input")
