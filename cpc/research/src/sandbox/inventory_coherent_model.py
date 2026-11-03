"""Shared nonequilibrium inventory allocation followed by a cold Hall solve."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from .coherent_fermi import CoherentMaterialProfile
from .coherent_hall_model import CoherentPrediction
from .coherent_transport import fermi_transport
from .donor_mixture import DonorMixtureLayer
from .fixed_inventory_charge import FixedInventoryCharge
from .parent_allocation import isolated_parent_contrast
from .kroger_db import load

LN10=float(np.log(10.));FIXED_TEMPERATURE_FRACTION=float(expit(6.))


@dataclass
class InventoryPrediction(CoherentPrediction):
    input_names:tuple
    input_jacobian:np.ndarray
    transport_input_jacobian:np.ndarray


class InventoryCoherentModel:
    def __init__(self,definition,*,profile:CoherentMaterialProfile,pO2_atm,db=None):
        if not isinstance(profile,CoherentMaterialProfile):raise TypeError('Explicit coherent profile required')
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError('Explicit positive pressure scenario required')
        self.definition=dict(definition);self.profile=profile;self.pO2_atm=float(pO2_atm)
        self.db=load() if db is None else db;self.layers={};self.solvers={}
        self.kind=definition['inventory_kind'];self.relation=definition.get('relation','constant')
        if self.kind not in ('none','native','complex','output'):raise ValueError('Explicit inventory/output definition required')
        if self.relation not in ('constant','dose_plus','dose_minus'):raise ValueError('Unknown shared dose basis')
        self.readout=definition.get('readout','ionized_extra')
        if self.readout not in ('ionized_extra','rH1'):raise ValueError('Explicit fixed-extra Hall or unit-Hall approximation required')
        names=['log10_background'];self.iE=None;self.iscreen=None
        if definition.get('binding')=='screen':
            self.iE=len(names);names.append('log10_binding_E0');self.iscreen=len(names);names.append('binding_screen_s')
        self.ic0=None;self.ic1=None
        if self.kind!='none':
            self.ic0=len(names);names.append('output_c0' if self.kind=='output' else 'inventory_s0')
            if self.relation!='constant':self.ic1=len(names);names.append('output_c1' if self.kind=='output' else 'inventory_s1')
        self.parameter_names=tuple(names)

    def layer_for(self,row):
        elements=(row.element,'H') if row.H_known else (row.element,)
        if elements not in self.layers:
            contrast=isolated_parent_contrast(self.db,elements)
            layer=DonorMixtureLayer(self.db,elements,profile=self.profile,hall='rH1',
                fixed_fraction=1. if self.definition.get('fraction')=='one' else 0.,basis=contrast.charge_basis[:,None])
            self.layers[elements]=layer;self.solvers[elements]=FixedInventoryCharge(layer)
        return self.layers[elements],self.solvers[elements]

    def source_interval(self,parameters):
        if self.kind not in ('native','complex'):raise ValueError('No internal inventory source coordinate in this model')
        return max(-6.,-float(parameters[self.ic0])/LN10),6.

    def point(self,row,parameters,*,source_kind='none',source_shift=0.):
        w=np.asarray(parameters,float);z=float(source_shift);nt=len(self.parameter_names)
        if w.shape!=(nt,) or not np.isfinite(w).all():raise ValueError('Finite shared parameters required')
        if self.kind in ('native','complex') and any(w[j]<0 for j in (self.ic0,self.ic1) if j is not None):
            raise ValueError('Nonnegative shared inventory coefficients required')
        if source_kind not in ('none','offset','native','complex') or not np.isfinite(z):raise ValueError('Explicit supported source coordinate required')
        if source_kind=='none' and z!=0:raise ValueError('Nonzero source coordinate requires a kind')
        if source_kind in ('native','complex'):
            if source_kind!=self.kind:raise ValueError('Internal source coordinate must match trained inventory type')
            lo,hi=self.source_interval(w)
            if not lo-1e-12<=z<=hi+1e-12:raise ValueError('Outside declared native/complex source interval')
        if row.element not in ('Si','Sn'):raise ValueError('Only audited Si/Sn domain supported')
        if not np.isfinite([row.incorporated_cm3,row.growth_T_C,row.Tobs_K,row.H_cm3,row.thermal_cap_K]).all():raise ValueError('Finite explicit physical inputs required')
        if row.incorporated_cm3<=0 or row.H_cm3<0 or bool(row.H_known)!=(row.H_cm3>0):raise ValueError('Explicit chemical/H inputs required')
        cap=float(row.thermal_cap_K)
        if not 600<cap<=1950:raise ValueError('Conditional cap must be in(600,1950]K')
        l,solver=self.layer_for(row)
        input_names=('log10_dose_cm3','log10_Tcap_K','log10_Tobs_K')+(('log10_H_cm3',) if row.H_known else ())
        ni=len(input_names);nv=nt+ni+1;chain=np.zeros((l.np,nv))
        logTf=np.log10(600.)+(np.log10(cap)-np.log10(600.))*FIXED_TEMPERATURE_FRACTION
        p=l.parameters(T=10**logTf,Nd=10**w[0],weights=[0.],pO2=self.pO2_atm,
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),Tobs=row.Tobs_K,
            binding_eV=10**w[self.iE] if self.iE is not None else .0366,
            screen_s=w[self.iscreen] if self.iscreen is not None else 0.)
        chain[1,0]=1.;chain[0,nt+1]=FIXED_TEMPERATURE_FRACTION
        chain[l.idose,nt]=1.;chain[l.iobs,nt+2]=1.
        if row.H_known:chain[2+l.b+1,nt+3]=1.
        if self.iE is not None:chain[l.ie,self.iE]=1.;chain[l.ie+1,self.iscreen]=1.
        feature=0.;feature_derivative=0.
        if self.relation!='constant':
            sign=1. if self.relation=='dose_plus' else -1.;x=sign*(np.log10(row.incorporated_cm3)-18.)
            feature=float(np.logaddexp(0.,x));feature_derivative=float(sign*expit(x))
        coordinate=0.;coordinate_jac=np.zeros(nv)
        if self.ic0 is not None:
            coordinate=float(w[self.ic0]);coordinate_jac[self.ic0]=1.
        if self.ic1 is not None:
            coordinate+=w[self.ic1]*feature;coordinate_jac[self.ic1]=feature;coordinate_jac[nt]=w[self.ic1]*feature_derivative
        if source_kind in ('native','complex'):coordinate+=LN10*z;coordinate_jac[-1]=LN10
        if self.kind in ('native','complex') and coordinate<0:raise ValueError('Nonnegative inventory suppression coordinate required')
        inventory=solver.allocate(p,chain,native_s=coordinate if self.kind=='native' else 0.,
            complex_s=coordinate if self.kind=='complex' else 0.,
            native_s_jacobian=coordinate_jac if self.kind=='native' else None,
            complex_s_jacobian=coordinate_jac if self.kind=='complex' else None)
        a=solver.solve(p,inventory,chain)
        if not a.diagnostics['single_electron_Hall_valid']:raise ValueError('Hole-dominated:outside declared single-electron readout')
        r=fermi_transport(a.values[0],a.values[2],float(row.Tobs_K),Nc_ref_cm3=self.profile.Nc_ref,T_ref_K=self.profile.T_ref,
            mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=205.,hall=self.readout!='rH1')
        readout_jac=r.jacobian[:,:2]@a.jacobian[[0,2]]
        readout_jac[:,nt+2]+=r.jacobian[:,2]
        values=a.values.copy();values[0]=r.values[0];J=a.jacobian.copy();J[0]=readout_jac[0]
        physical=values.copy();transport_values=r.values[1:].copy();TJ=readout_jac[1:].copy()
        if self.kind=='output':
            values[0]+=coordinate;J[0]+=coordinate_jac;transport_values[1]+=coordinate;TJ[1]+=coordinate_jac
        if source_kind=='offset':
            values[0]-=z;J[0,-1]=-1.;transport_values[1]-=z;TJ[1,-1]=-1.
        same=(self.kind!='output' or coordinate==0.) and (source_kind!='offset' or z==0.)
        dg={**a.diagnostics,'learning_definition':self.definition,'source_kind':source_kind,'source_shift':z,
            'same_physical_state_for_output':same,'hall_factor':r.diagnostics['readout_hall_factor'],
            'coherent_transport':r.diagnostics,'physical_transport_values':r.values[1:].tolist(),
            'above_carrier_detection_floor':bool(values[0]>=13),'thermal_cap_K':cap,
            'thermal_cap_interpretation':'conditional recorded upper,not actual film Tmax','temperature_coordinate_fixed_at':6.,
            'predicted_log_mu_H':float(transport_values[0]),'predicted_log_sigma':float(transport_values[1]),
            'statistics_scope':'same phase15 scalar FD bands/free-onlyscreening/205 empiricalextra;no new microscopic rates'}
        return InventoryPrediction(values,J[:,:nt],J[:,-1],physical,dg,transport_values,TJ[:,:nt],TJ[:,-1],
            input_names,J[:,nt:nt+ni],TJ[:,nt:nt+ni])

    def counters(self):
        return dict(root_calls=sum(s.cold_calls for s in self.solvers.values()),
            hot_solve_calls=sum(l.hot_solve_calls for l in self.layers.values()),
            hot_cache_hits=sum(l.hot_cache_hits for l in self.layers.values()),
            root_evaluations=sum(l.root_evaluations for l in self.layers.values())+sum(s.cold_evaluations for s in self.solvers.values()))
