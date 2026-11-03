"""Label-free serialized nonlinear controls; no physical inventory is inferred."""
from dataclasses import replace
import numpy as np
from sandbox.joint_saved_model import SavedJointModel,JointPropertyPoint
from sandbox.context_saved_model import feature_vector
from sandbox.activation_transport_control import ct_log_readout
from sandbox.ionized_transport import E_CHARGE

METHODS=('DirectMultiExtraTrees','DoseResidualExtraTrees')


def forest_value(forest,x):
    values=[]
    for tree in forest:
        i=0
        while tree['left'][i]>=0:
            # sklearn.tree internally converts prediction features to float32.
            i=tree['left'][i] if float(np.float32(x[tree['feature'][i]]))<=tree['threshold'][i] else tree['right'][i]
        values.append(tree['value'][i])
    return float(np.mean(values))


def raw_values(fit,row):
    x=(feature_vector(row)-np.asarray(fit['center']))/np.asarray(fit['scale'])
    g=forest_value(fit['nH_forest'],x)
    dose=float(np.log10(row.incorporated_cm3))
    if fit['method']=='DoseResidualExtraTrees':g+=dose
    if fit['rho_forest'] is None:
        mu=ct_log_readout(g,row.Tobs_K)[0]
        sigma=np.log10(E_CHARGE)+g+mu
    else:
        rho=forest_value(fit['rho_forest'],x)
        if fit['method']=='DoseResidualExtraTrees':
            rho-=np.log10(E_CHARGE)+dose+ct_log_readout(dose,row.Tobs_K)[0]
        sigma=-rho;mu=sigma-g-np.log10(E_CHARGE)
    return np.array([g,mu,sigma],float)


class StrongTreeSavedModel:
    def __init__(self,fit):
        self.fit=fit;self.method=fit['method'];self.delegate=None
        if self.method not in METHODS:
            self.delegate=SavedJointModel(fit);self.physical=self.delegate.physical;return
        if not fit.get('training_valid') or not fit.get('selected_success'):
            raise ValueError('Unqualified tree fit')
        self.physical=False

    def point(self,row,*,source_kind='none',shift=0.,require_observation=True):
        if self.delegate is not None:
            return self.delegate.point(row,source_kind=source_kind,shift=shift,require_observation=require_observation)
        z=float(shift)
        if source_kind not in ('none','offset') or not np.isfinite(z) or abs(z)>6 or (source_kind=='none' and z!=0):
            raise ValueError('Declared finite offset in[-6,6] required')
        if row.element not in ('Si','Sn') or row.Tobs_K!=300. or row.incorporated_cm3<=0:
            raise ValueError('Same positive-dose Si/Sn 300K domain required')
        if bool(row.H_known)!=(row.H_cm3>0) or bool(row.carbon_known)!=(row.carbon_cm3>0):
            raise ValueError('Same explicit composition scenarios required')
        g,mu,sigma=raw_values(self.fit,row)
        base=JointPropertyPoint(float(g),float(mu),float(sigma),ct_log_readout(g,row.Tobs_K)[0],
            1.,np.zeros(2),bool(np.isfinite([g,mu,sigma]).all() and g>=13.),
            dict(physical_state_available=False,same_physical_state_for_output=False,
                 transport_scope='direct primitive nH/rho forest heads;CT fallback if no paired training labels',
                 source_kind='none',source_shift=0.,observation_offset_dex=0.))
        return self.offset_point(row,base,z,require_observation=require_observation)

    def offset_point(self,row,base,shift,*,require_observation=True):
        if self.delegate is not None:
            return self.delegate.offset_point(row,base,shift,require_observation=require_observation)
        z=float(shift)
        if not np.isfinite(z) or abs(z)>6:raise ValueError('Finite offset in[-6,6] required')
        g=float(base.log_nH+z);ct,slope,_=ct_log_readout(g,row.Tobs_K)
        if self.fit['rho_forest'] is not None:
            sigma=base.log_sigma;mu=base.log_muH-z;derivative=np.array([-1.,0.])
        else:
            mu=ct;sigma=np.log10(E_CHARGE)+g+mu;derivative=np.array([slope,1+slope])
        valid=bool(np.isfinite([g,mu,sigma]).all() and g>=13.)
        if require_observation and not valid:raise ValueError('Tree prediction outside declared detection domain')
        dg=dict(base.diagnostics,source_kind='offset',source_shift=z,observation_offset_dex=z,
                observation_domain_valid=valid,same_physical_state_for_output=False)
        return replace(base,log_nH=g,log_muH=float(mu),log_sigma=float(sigma),log_mu_CT=float(ct),
                       source_derivative=1.,source_transport_derivative=derivative,valid=valid,diagnostics=dg)
