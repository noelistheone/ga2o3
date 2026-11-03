"""Shared defect-state learning with coherent Fermi/Hall property readout."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .coherent_fermi import CoherentMaterialProfile
from .coherent_transport import fermi_transport
from .donor_mixture_model import DonorMixtureModel


@dataclass
class CoherentPrediction:
    values:np.ndarray
    jacobian:np.ndarray
    source_jacobian:np.ndarray
    physical_values:np.ndarray
    diagnostics:dict
    transport_values:np.ndarray
    transport_jacobian:np.ndarray
    transport_source_jacobian:np.ndarray


class CoherentHallModel:
    def __init__(self,definition,*,profile:CoherentMaterialProfile,pO2_atm,db=None):
        if not isinstance(profile,CoherentMaterialProfile):raise TypeError('Explicit coherent immutable profile required')
        self.definition=dict(definition);self.profile=profile
        self.readout=definition.get('readout')
        if self.readout not in ('rH1','ionized','ionized_extra'):raise ValueError('Explicit coherent readout required')
        if 'output' in definition:raise ValueError('This version does not implement shared output-feature correction')
        self.base=DonorMixtureModel(definition,profile=profile,pO2_atm=pO2_atm,db=db)
        for key in ('parameter_names','iE','iscreen','ia','ib','iout','layers'):setattr(self,key,getattr(self.base,key))

    def point(self,row,parameters,*,source_kind='none',source_shift=0.):
        internal=source_kind!='offset'
        a=self.base.point(row,parameters,source_kind=source_kind if internal else 'none',source_shift=source_shift if internal else 0.)
        z=float(source_shift)
        if not np.isfinite(z):raise ValueError('Finite source coordinate required')
        hall=self.readout!='rH1';extra=np.inf if self.readout=='ionized' else 205.
        r=fermi_transport(a.physical_values[0],a.physical_values[2],float(row.Tobs_K),Nc_ref_cm3=self.profile.Nc_ref,
            T_ref_K=self.profile.T_ref,mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=extra,hall=hall)
        state_columns=[0,2];readout_columns=[0,1];raw=a.values.copy();raw[0]=r.values[0]
        jac=a.jacobian.copy();jac[0]=r.jacobian[0,readout_columns]@a.jacobian[state_columns]
        sj=a.source_jacobian.copy();sj[0]=float(r.jacobian[0,readout_columns]@a.source_jacobian[state_columns])
        tj=r.jacobian[1:,readout_columns]@a.jacobian[state_columns]
        ts=r.jacobian[1:,readout_columns]@a.source_jacobian[state_columns]
        physical=raw.copy();values=raw.copy();transport_values=r.values[1:].copy()
        if not internal:
            values[0]-=z;sj[0]=-1.;transport_values[1]-=z;ts[1]=-1.
        dg={**a.diagnostics,'hall_factor':r.diagnostics['readout_hall_factor'],'hall_convention':'coherent_'+self.readout,
            'source_kind':source_kind,'source_shift':z,'same_physical_state_for_output':internal or z==0,
            'above_carrier_detection_floor':bool(values[0]>=13),
            'physical_log_Hall_density':float(physical[0]),'coherent_transport':r.diagnostics,
            'physical_transport_values':r.values[1:].tolist(),'predicted_log_mu_H':float(transport_values[0]),
            'predicted_log_sigma':float(transport_values[1]),'learning_definition':self.definition,
            'statistics_scope':'C2FD freebands;legacySTH preserved;one scalar electron DOS/transport mass'}
        return CoherentPrediction(values,jac,sj,physical,dg,transport_values,tj,ts)

    def counters(self):return self.base.counters()
