"""Label-free evaluator of the phase29 shared models and matched controls."""
from dataclasses import dataclass
import numpy as np
from .coherent_fermi import published_coherent_profile
from .pool_share_controls import PoolReferenceModel
from .charge_feasible_model_v2 import ChargeFeasibleModel
from .joint_inventory_scattering import JointInventoryScatteringModel
from .reference_scattering import ReferenceScatteringModel
from .activation_transport_control import ActivationTransportControl,ct_log_readout
from .context_saved_model import feature_vector
from .ionized_transport import E_CHARGE


@dataclass
class JointPropertyPoint:
    log_nH:float
    log_muH:float
    log_sigma:float
    log_mu_CT:float
    source_derivative:float
    source_transport_derivative:np.ndarray
    valid:bool
    diagnostics:dict


class SavedJointModel:
    """Does not load condition tables, labels, split files or other fit files."""
    def __init__(self,fit,*,profile=None,db=None):
        self.fit=dict(fit);self.method=fit['method'];self.engine=None
        if not fit.get('training_valid') or not fit.get('selected_success'):
            raise ValueError('Saved fit is invalid or its selected optimizer did not stop')
        p=published_coherent_profile() if profile is None else profile
        if self.method=='Reference205':self.engine=PoolReferenceModel(temperature_mode='effective',profile=p)
        elif self.method=='ReferenceTau':self.engine=ReferenceScatteringModel(profile=p,db=db)
        elif self.method.startswith('Share') and self.method.endswith('205'):
            self.engine=ChargeFeasibleModel(basis=self.method[5:-3].lower(),profile=p,db=db)
        elif self.method.startswith('Joint'):
            self.engine=JointInventoryScatteringModel(basis=self.method[5:].lower(),profile=p,db=db)
        elif self.method=='ActivationProcessPair':self.engine=ActivationTransportControl(basis='process')
        elif self.method not in ('DirectMultiRidge','UnitDose','MedianActivation','DoseLine'):
            raise ValueError('Unknown phase29 saved model')
        self.physical=self.engine is not None and self.method!='ActivationProcessPair'
        self.head=fit.get('transport_head','native')
        self.head_offset=float(fit.get('transport_head_offset_dex',0.))
        if self.head not in ('native','native_offset','CT_offset'):
            raise ValueError('Unknown declared transport output control')

    def point(self,row,*,source_kind='none',shift=0.,require_observation=True):
        z=float(shift)
        if source_kind=='field':source_kind='share'
        if source_kind not in ('none','offset','share') or not np.isfinite(z) or abs(z)>6. or (source_kind=='none' and z!=0):
            raise ValueError('Explicit source coordinate in[-6,6] required')
        if row.element not in ('Si','Sn') or not np.isfinite([
                row.incorporated_cm3,row.growth_T_C,row.scenario_pO2_atm,row.Tobs_K,row.H_cm3,row.carbon_cm3]).all():
            raise ValueError('Finite explicit inputs in the Si/Sn research domain required')
        if row.incorporated_cm3<=0 or row.scenario_pO2_atm<=0 or row.Tobs_K!=300.:
            raise ValueError('This saved version serves the declared positive-dose300K comparison')
        if bool(row.H_known)!=(row.H_cm3>0) or bool(row.carbon_known)!=(row.carbon_cm3>0):
            raise ValueError('Known positive or explicit unknown-zero composition scenario required')
        if self.engine is not None:
            p=self.engine.point(row,self.fit['parameters'],source_kind=source_kind,source_shift=z,require_observation=False)
            g=float(p.values[0]);mu=float(p.transport_values[0]);sigma=float(p.transport_values[1])
            derivative=float(p.source_jacobian[0]);dj=np.array(p.transport_source_jacobian,float)
            valid=bool(p.diagnostics['observation_domain_valid'])
            dg=dict(p.diagnostics,physical_state_available=self.physical)
        else:
            if source_kind=='share':raise ValueError('Direct controls have no internal inventory coordinate')
            if self.method=='DirectMultiRidge':
                x=np.r_[1.,(feature_vector(row)-np.asarray(self.fit['center']))/np.asarray(self.fit['scale'])]
                g0=float(x@np.asarray(self.fit['nH_coefficients']));g=g0+z
                if self.fit['rho_coefficients'] is None:
                    mu,slope,_=ct_log_readout(g,row.Tobs_K);sigma=np.log10(E_CHARGE)+g+mu
                    dj=np.array([slope,1+slope])
                    dg=dict(transport_scope='no paired training direction available;CT fallback',physical_state_available=False)
                else:
                    sigma=-float(x@np.asarray(self.fit['rho_coefficients']))
                    mu=sigma-g-np.log10(E_CHARGE);dj=np.array([-1.,0.])
                    dg=dict(transport_scope='two direct primitive-observable heads;source offset updates Hall head while rho head stays fixed',
                        physical_state_available=False)
            else:
                x=float(np.log10(row.incorporated_cm3))
                g=(float(self.fit['coefficients'][0]+self.fit['coefficients'][1]*(x-18.))
                    if self.method=='DoseLine' else x+float(self.fit['offset_dex']))+z
                mu,slope,_=ct_log_readout(g,row.Tobs_K);sigma=float(np.log10(E_CHARGE)+g+mu)
                dj=np.array([slope,1+slope])
                dg=dict(transport_scope='CT of predicted Hall density',physical_state_available=False)
            derivative=1. if source_kind=='offset' else 0.
            if source_kind=='none':dj=np.zeros(2)
            valid=bool(np.isfinite([g,mu,sigma]).all() and g>=13.)
            dg.update(same_physical_state_for_output=False,observation_domain_valid=valid)
        ct,slope,_=ct_log_readout(g,row.Tobs_K)
        if self.head=='CT_offset':
            mu=ct+self.head_offset;sigma=float(np.log10(E_CHARGE)+g+mu)
            dj=np.array([slope*derivative,(1+slope)*derivative])
        elif self.head=='native_offset':
            mu+=self.head_offset;sigma+=self.head_offset
        dg.update(actual_evaluation_method=self.method,transport_head=self.head,
            transport_head_offset_dex=self.head_offset,
            output_control_changes_latent_state=False,
            no_calibrated_predictive_intervals=True,material_Born_validity_established=False)
        if self.head!='native':dg['same_physical_state_for_output']=False
        if require_observation and not valid:raise ValueError('Outside the implemented observation domain')
        return JointPropertyPoint(g,mu,sigma,float(ct),derivative,dj,valid,dg)

    def offset_point(self,row,base,shift,*,require_observation=True):
        """Output algebra, preserving each control's declared rho/mobility rule."""
        z=float(shift)
        if not np.isfinite(z) or abs(z)>6.:raise ValueError('Explicit offset in[-6,6] required')
        g=base.log_nH+z;ct,slope,_=ct_log_readout(g,row.Tobs_K)
        if self.head=='CT_offset':
            mu=ct+self.head_offset;sigma=np.log10(E_CHARGE)+g+mu;dj=np.array([slope,1+slope])
        elif self.physical:
            mu=base.log_muH;sigma=base.log_sigma+z;dj=np.array([0.,1.])
        elif self.method=='DirectMultiRidge' and self.fit['rho_coefficients'] is not None:
            sigma=base.log_sigma;mu=base.log_muH-z;dj=np.array([-1.,0.])
        else:
            preserved=base.log_muH-ct_log_readout(base.log_nH,row.Tobs_K)[0]
            mu=ct+preserved;sigma=np.log10(E_CHARGE)+g+mu;dj=np.array([slope,1+slope])
        underlying=bool(base.diagnostics.get('single_electron_Hall_valid',True))
        valid=bool(underlying and np.isfinite([g,mu,sigma]).all() and g>=13.)
        dg=dict(base.diagnostics,source_kind='offset',source_shift=z,
            same_physical_state_for_output=False,observation_domain_valid=valid,
            observation_offset_dex=float(base.diagnostics.get('observation_offset_dex',0.)+z))
        if require_observation and not valid:raise ValueError('Output offset outside implemented observation domain')
        return JointPropertyPoint(g,float(mu),float(sigma),float(ct),1.,dj,valid,dg)
