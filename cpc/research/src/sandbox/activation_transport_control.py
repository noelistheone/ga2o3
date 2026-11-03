"""Matched-capacity empirical activation/CT pair; no inferred defect state."""
from dataclasses import replace
import numpy as np
from .pool_share_controls import DirectActivationModel
from .relative_inventory_model import RelativeInventoryPrediction
from . import transport


def ct_log_readout(log_nH,T_K):
    n=10.**float(log_nH);T=float(T_K)
    mu=float(transport.mu_caughey_thomas(n,T))
    power=(n/transport.CT_N_REF)**transport.CT_ALPHA
    ceiling=transport.CT_MU_MAX*transport.mu_pop_hall(T)/transport.mu_pop_hall(300.)
    slope=-(ceiling-transport.CT_MU_MIN)*transport.CT_ALPHA*power/((1+power)**2*mu)
    dpop=-56.*508.*np.exp(508./T)/T**2
    logT=T*transport.CT_MU_MAX*dpop/transport.mu_pop_hall(300.)/((1+power)*mu)
    return np.log10(mu),float(slope),float(logT)


class ActivationTransportControl:
    def __init__(self,*,basis='process'):
        self.base=DirectActivationModel(basis=basis);self.basis=basis
        self.parameter_names=self.base.parameter_names+('empirical_log_mu_offset',)

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float)
        if w.shape!=(len(self.parameter_names),) or not np.isfinite(w).all():
            raise ValueError('Matching finite empirical activation/transport parameters required')
        p=self.base.point(row,w[:-1],source_kind=source_kind,source_shift=source_shift,
            require_observation=require_observation)
        g=float(p.values[0]);mu,slope,logT=ct_log_readout(g,row.Tobs_K);mu+=w[-1]
        J=np.c_[p.jacobian,np.zeros((1,1))]
        mj=slope*J[0];mj[-1]+=1.
        ij=p.input_jacobian.copy();mi=slope*ij[0]
        mi[p.input_names.index('log10_Tobs_K')]+=logT
        sj=float(p.source_jacobian[0]);sm=slope*sj
        tv=np.array([mu,np.log10(transport.E)+g+mu])
        return RelativeInventoryPrediction(np.array([g]),J,np.array([sj]),np.full(1,np.nan),
            dict(p.diagnostics,physical_state_available=False,same_physical_state_for_output=False,
                empirical_transport_offset=float(w[-1]),learning_scope='direct Hall activation and CT plus shared output scale'),
            tv,np.stack([mj,mj+J[0]]),np.array([sm,sm+sj]),
            ij,np.stack([mi,mi+ij[0]]),p.input_names)

    def counters(self):return self.base.counters()
