"""Shared inventory odds -> fixed-stock charge -> same-state Hall properties."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from .coherent_fermi import CoherentMaterialProfile
from .coherent_hall_model import CoherentPrediction
from .coherent_transport import fermi_transport
from .donor_mixture import DonorMixtureLayer
from .fixed_inventory_charge import FixedInventoryCharge
from .relative_inventory import allocate_relative_odds
from .parent_allocation import isolated_parent_contrast
from .kroger_db import load

LN10=float(np.log(10.));L0=float(np.log10(600.));L1=float(np.log10(1950.))


@dataclass
class RelativeInventoryPrediction(CoherentPrediction):
    input_jacobian:np.ndarray
    transport_input_jacobian:np.ndarray
    input_names:tuple


class RelativeInventoryModel:
    def __init__(self,definition,*,profile:CoherentMaterialProfile,pO2_atm,db=None):
        if not isinstance(profile,CoherentMaterialProfile):raise TypeError('Explicit coherent profile required')
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError('Explicit positive oxygen scenario required')
        self.definition=dict(definition);self.kind=definition['field']
        if self.kind not in ('inventory','output'):raise ValueError('Declared inventory/output field required')
        self.profile=profile;self.pO2_atm=float(pO2_atm);self.db=load() if db is None else db
        self.layers={};self.solvers={};self.parameter_names=('logit_Tf','log10_background','shared_log10_field')

    def layer_for(self,row):
        elements=(row.element,'H') if row.H_known else (row.element,)
        if elements not in self.layers:
            contrast=isolated_parent_contrast(self.db,elements)
            l=DonorMixtureLayer(self.db,elements,profile=self.profile,hall='rH1',fixed_fraction=0.,basis=contrast.charge_basis[:,None])
            self.layers[elements]=l;self.solvers[elements]=FixedInventoryCharge(l)
        return self.layers[elements],self.solvers[elements]

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift)
        if w.shape!=(3,) or not np.isfinite(w).all() or not np.isfinite(z):raise ValueError('Finite three shared parameters/source coordinate required')
        if row.element not in ('Si','Sn'):raise ValueError('Current trained domain is Si/Sn')
        if not np.isfinite([row.incorporated_cm3,row.Tobs_K,row.H_cm3]).all() or row.incorporated_cm3<=0 or row.H_cm3<0:
            raise ValueError('Explicit finite chemical/measurement inputs required')
        if bool(row.H_known)!=(row.H_cm3>0):raise ValueError('Explicit reported-H or unknown-zero scenario required')
        if source_kind not in ('none','field','offset') or (source_kind=='none' and z!=0) or abs(z)>6+1e-12:
            raise ValueError('Supported source coordinate in [-6,6] required')
        if source_kind=='field' and self.kind!='inventory':raise ValueError('Internal source update requires inventory closure')
        if abs(w[2])>6+1e-12:raise ValueError('Shared field outside [-6,6]')
        l,solver=self.layer_for(row);s=expit(w[0]);T=10**(L0+(L1-L0)*s)
        p=l.parameters(T=T,Nd=10**w[1],weights=[0.],doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=self.pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        # Columns:3shared,source,logC,logTobs,(logH).
        ni=3 if row.H_known else 2;nv=4+ni;chain=np.zeros((l.np,nv))
        chain[0,0]=(L1-L0)*s*(1-s);chain[1,1]=1.;chain[l.idose,4]=1.;chain[l.iobs,5]=1.
        if row.H_known:chain[2+l.b+1,6]=1.
        h=w[2] if self.kind=='inventory' else 0.;dh=np.zeros(nv)
        if self.kind=='inventory':dh[2]=1.
        if source_kind=='field':h+=z;dh[3]=1.
        inventory=allocate_relative_odds(solver,p,chain,h,dh);a=solver.solve(p,inventory,chain)
        # A numerical continuation is available to the training optimizer and
        # anchor root bracketing only. It is never accepted as a material/Hall
        # prediction when its observation-domain flag is false.
        observation_valid=bool(a.diagnostics['single_electron_Hall_valid'])
        transport=fermi_transport(a.values[0],a.values[2],float(row.Tobs_K),Nc_ref_cm3=self.profile.Nc_ref,
            T_ref_K=self.profile.T_ref,mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=205.,hall=True)
        readout=transport.jacobian[:,:2]@a.jacobian[[0,2]];readout[:,5]+=transport.jacobian[:,2]
        values=a.values.copy();values[0]=transport.values[0];J=a.jacobian.copy();J[0]=readout[0]
        physical=values.copy();tv=transport.values[1:].copy();TJ=readout[1:].copy();offset=0.
        if self.kind=='output':offset+=w[2];J[0,2]+=1.;TJ[1,2]+=1.
        if source_kind=='offset':offset+=z;J[0,3]+=1.;TJ[1,3]+=1.
        values[0]+=offset;tv[1]+=offset
        observation_valid=observation_valid and bool(values[0]>=13.)
        if require_observation and not observation_valid:
            raise ValueError('Outside single-electron Hall/detection observation domain')
        dg={**a.diagnostics,'source_kind':source_kind,'source_shift':z,'shared_field':float(w[2]),
            'learning_definition':self.definition,'Tf_K':T,'pO2_atm':self.pO2_atm,
            'same_physical_state_for_output':offset==0.,'hall_factor':transport.diagnostics['readout_hall_factor'],
            'coherent_transport':transport.diagnostics,'physical_transport_values':transport.values[1:].tolist(),
            'above_carrier_detection_floor':bool(values[0]>=13.),'observation_offset_dex':float(offset),
            'observation_domain_valid':observation_valid,'numerical_observation_continuation':not observation_valid,
            'process_interpretation':'effective-temperature endpoint closure;not suppliedgrowth history or microscopic rate'}
        return RelativeInventoryPrediction(values,J[:,:3],J[:,3],physical,dg,tv,TJ[:,:3],TJ[:,3],
            J[:,4:],TJ[:,4:],('log10_dose_cm3','log10_Tobs_K')+(('log10_H_cm3',) if row.H_known else ()))

    def counters(self):
        return dict(hot_solve_calls=sum(l.hot_solve_calls for l in self.layers.values()),
            hot_cache_hits=sum(l.hot_cache_hits for l in self.layers.values()),
            cold_solve_calls=sum(s.cold_calls for s in self.solvers.values()))
