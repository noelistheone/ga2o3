"""Label-free evaluation of a saved absolute-share model or strong control."""
import numpy as np
from .pool_share_model import PoolShareModel
from .pool_share_controls import PoolReferenceModel,DirectActivationModel
from .coherent_fermi import published_coherent_profile
from .context_saved_model import SavedContextModel,ContextPropertyPoint
from . import transport


class SavedPoolModel(SavedContextModel):
    def __init__(self,fit,*,profile=None):
        self.fit=dict(fit);self.method=fit['method'];actual=fit.get('evaluation_method',self.method)
        if not fit.get('training_valid') or not fit.get('selected_success',False):
            raise ValueError('Saved absolute-share/control fit is invalid or did not stop')
        profile=published_coherent_profile() if profile is None else profile
        if actual in ('ParentReference','GrowthReference'):
            self.operator=PoolReferenceModel(temperature_mode='effective' if actual=='ParentReference' else 'growth',profile=profile);self.physical=True
        elif actual.startswith('Activation'):
            basis='none' if actual=='ActivationReference' else actual.replace('Activation','').lower()
            self.operator=DirectActivationModel(basis=basis);self.physical=False
        else:
            prefix='ShareEffective' if actual.startswith('ShareEffective') else ('ShareGrowth' if actual.startswith('ShareGrowth') else None)
            if prefix is None:raise ValueError('Unknown saved pool-share evaluation method')
            self.operator=PoolShareModel(basis=actual.replace(prefix,'').lower(),temperature_mode='effective' if prefix=='ShareEffective' else 'growth',profile=profile);self.physical=True
        self.engine=self.operator if self.physical else None

    def point(self,row,*,source_kind='none',shift=0.,require_observation=True):
        if row.element not in ('Si','Sn') or row.Tobs_K!=300.:
            raise ValueError('Saved predictor requires declared Si/Sn300Kcomparison inputs')
        kind='share' if source_kind=='field' else source_kind
        p=self.operator.point(row,self.fit['parameters'],source_kind=kind,source_shift=shift,require_observation=require_observation)
        g=float(p.values[0]);ct=float(np.log10(transport.mu_caughey_thomas(10**g,300.)))
        mu=float(p.transport_values[0]) if self.physical else ct
        sigma=float(p.transport_values[1]) if self.physical else float(np.log10(transport.E)+g+mu)
        diag=dict(p.diagnostics,physical_state_available=self.physical,actual_evaluation_method=self.fit.get('evaluation_method',self.method),
            explicit_reference_fallback=bool(self.fit.get('reference_fallback',False)))
        return ContextPropertyPoint(g,mu,sigma,ct,float(np.log10(transport.E)+g+ct),float(p.source_jacobian[0]),
            bool(p.diagnostics['observation_domain_valid']),diag)
