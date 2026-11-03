"""Versioned thermal-input closure for the coherent defect/Hall predictor.

The recorded cap is a supplied scenario input, never a recovered true history.
The immutable phase15 model is reused through an exact coordinate transform.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logit
from .coherent_hall_model import CoherentHallModel,CoherentPrediction
from .donor_mixture_model import L0,L1,LN10


@dataclass
class ThermalPrediction(CoherentPrediction):
    cap_jacobian:np.ndarray
    transport_cap_jacobian:np.ndarray


class ThermalCoherentHallModel(CoherentHallModel):
    def __init__(self,definition,**kwargs):
        super().__init__(definition,**kwargs)
        self.thermal_mode=definition.get('thermal_mode')
        if self.thermal_mode not in ('effective','record_cap'):
            raise ValueError('Explicit effective or conditional record_cap law required')

    def point(self,row,parameters,*,source_kind='none',source_shift=0.):
        w=np.asarray(parameters,float)
        if self.thermal_mode=='effective':
            a=super().point(row,w,source_kind=source_kind,source_shift=source_shift)
            return ThermalPrediction(**vars(a),cap_jacobian=np.zeros(3),transport_cap_jacobian=np.zeros(2))
        if w.shape!=(len(self.parameter_names),) or not np.isfinite(w).all():
            raise ValueError('Finite model parameters required')
        z=float(source_shift)
        if not np.isfinite(z) or source_kind not in ('none','offset','stock','Tf'):
            raise ValueError('Explicit supported finite source coordinate required')
        if source_kind=='none' and z!=0:raise ValueError('Nonzero source coordinate requires a kind')
        if source_kind in ('stock','Tf') and abs(z)>6+1e-12:raise ValueError('Outside source-coordinate domain')
        cap=float(row.thermal_cap_K)
        if not np.isfinite(cap) or not 600.<cap<=1950.:
            raise ValueError('Explicit conditional cap in (600,1950] K required')
        bias=24*np.tanh(LN10*z/24)
        dbias=LN10*(1-np.tanh(LN10*z/24)**2)
        coordinate=w[0]+(bias/4 if source_kind=='Tf' else 0.)
        s=expit(coordinate);span=np.log10(cap)-L0
        logT=L0+span*s;q=(logT-L0)/(L1-L0)
        if not 0.<q<1.:raise ValueError('Temperature transform lost an interior coordinate')
        transformed=w.copy();transformed[0]=logit(q)
        old_slope=(L1-L0)*q*(1-q)
        parameter_ratio=span*s*(1-s)/old_slope
        cap_ratio=s/old_slope
        # A Tf source update belongs before the capped sigmoid; never after it.
        a=super().point(row,transformed,source_kind='none' if source_kind=='Tf' else source_kind,
            source_shift=0. if source_kind=='Tf' else z)
        cap_jac=a.jacobian[:,0].copy()*cap_ratio
        transport_cap_jac=a.transport_jacobian[:,0].copy()*cap_ratio
        a.jacobian[:,0]*=parameter_ratio;a.transport_jacobian[:,0]*=parameter_ratio
        if source_kind=='Tf':
            a.source_jacobian=a.jacobian[:,0]*dbias/4
            a.transport_source_jacobian=a.transport_jacobian[:,0]*dbias/4
        a.diagnostics.update(thermal_mode='record_cap',thermal_cap_K=cap,
            thermal_cap_interpretation='conditional reported bound;not proven actual film Tmax',
            cap_log10_derivative_order='values=(lognH,logNabs,logNq2);transport=(logmuH,logsigma)',
            shared_temperature_coordinate=float(w[0]),effective_temperature_fraction=float(s),
            source_kind=source_kind,source_shift=z)
        if source_kind=='Tf':
            a.diagnostics['same_physical_state_for_output']=True
        return ThermalPrediction(**vars(a),cap_jacobian=cap_jac,transport_cap_jacobian=transport_cap_jac)
