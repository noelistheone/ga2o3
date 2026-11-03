"""Label-free evaluation of a saved phase25 relation and its matched controls."""
from dataclasses import dataclass
import numpy as np
from .context_inventory_model import ContextInventoryModel, bounded_field
from .coherent_fermi import published_coherent_profile
from . import transport


INPUT_NAMES = ('element','incorporated_cm3','Tobs_K','H_known','H_cm3','carbon_known','carbon_cm3',
    'growth_T_C','scenario_pO2_atm','oxygen_input_kind')


def feature_vector(row):
    c=float(np.log10(row.incorporated_cm3)-18.)
    hk=float(row.H_known);ck=float(row.carbon_known)
    h=float(np.log10(max(row.H_cm3,1.))-17.)
    carbon=float(np.log10(max(row.carbon_cm3,1.))-17.)
    p=float(np.log10(row.scenario_pO2_atm)+5.)
    # This is measurement availability, not a DOI/source one-hot feature.
    # On the fixed training cohort Leedy is the only reported molecular-O2
    # case; its identity is not needed in a prediction request.
    reported=float(str(row.oxygen_input_kind).startswith('reported molecular'))
    return np.array([c,c*c,(row.growth_T_C-700.)/300.,float(row.element=='Sn'),hk,h*hk,ck,carbon*ck,p,reported])


@dataclass
class ContextPropertyPoint:
    log_nH:float
    log_muH:float
    log_sigma:float
    log_mu_CT:float
    log_sigma_CT:float
    source_derivative:float
    valid:bool
    diagnostics:dict


class SavedContextModel:
    def __init__(self, fit, *, profile=None, db=None):
        self.fit=dict(fit)
        self.method=fit['method']
        self.engine=None
        if not fit.get('training_valid') or not fit.get('selected_success',True):
            raise ValueError('Saved fit is invalid or did not stop')
        if self.method=='FixedPhysical' or self.method.startswith(('Inventory','Output')):
            basis='none' if self.method=='FixedPhysical' else self.method.replace('Inventory','').replace('Output','').lower()
            kind='output' if self.method.startswith('Output') else 'inventory'
            self.engine=ContextInventoryModel(basis=basis,field=kind,profile=published_coherent_profile() if profile is None else profile,db=db)

    def point(self,row,*,source_kind='none',shift=0.,require_observation=True):
        z=float(shift)
        if not np.isfinite(z) or abs(z)>6.+1e-12:
            raise ValueError('Declared scalar source domain is [-6,6]')
        if row.element not in ('Si','Sn') or not np.isfinite([row.incorporated_cm3,row.Tobs_K,row.growth_T_C,row.scenario_pO2_atm]).all():
            raise ValueError('Finite explicit inputs in the Si/Sn research domain required')
        if row.incorporated_cm3<=0 or row.Tobs_K!=300. or row.scenario_pO2_atm<=0:
            raise ValueError('This saved predictor is restricted to the declared 300K comparison task')
        if self.engine is not None:
            p=self.engine.point(row,self.fit['parameters'],source_kind=source_kind,source_shift=z,require_observation=require_observation)
            g=float(p.values[0]);mu=float(p.transport_values[0]);sigma=float(p.transport_values[1])
            derivative=float(p.source_jacobian[0]);valid=bool(p.diagnostics['observation_domain_valid'])
            diagnostics=dict(p.diagnostics,physical_state_available=True)
        else:
            if source_kind not in ('none','offset') or (source_kind=='none' and z!=0):
                raise ValueError('Direct/empirical controls support only an explicit output offset')
            if self.method.startswith('Direct') and self.method!='DirectRidge':
                basis=self.method.replace('Direct','').lower();w=np.asarray(self.fit['parameters'])
                h,_,_=bounded_field(row,w[2:],basis)
                g=float(w[0]+w[1]*(np.log10(row.incorporated_cm3)-18.)+h+z)
            elif self.method=='DirectRidge':
                x=(feature_vector(row)-np.asarray(self.fit['center']))/np.asarray(self.fit['scale'])
                g=float(self.fit['intercept']+x@np.asarray(self.fit['coefficients'])+z)
            elif self.method in ('UnitDose','MedianActivation','DoseLine'):
                x=float(np.log10(row.incorporated_cm3))
                g=float(self.fit['coefficients'][0]+self.fit['coefficients'][1]*(x-18.)) if self.method=='DoseLine' else x+self.fit['offset_dex']
                g+=z
            else:
                raise ValueError('Unknown saved method')
            derivative=1.;valid=bool(np.isfinite(g) and g>=13.)
            mu=float(np.log10(transport.mu_caughey_thomas(10**g,300.)))
            sigma=float(np.log10(transport.E)+g+mu)
            diagnostics=dict(physical_state_available=False,same_physical_state_for_output=False,
                transport_scope='CT(predicted nH) empirical readout,not inferred ionized defect density')
            if require_observation and not valid:
                raise ValueError('Empirical output outside the declared Hall detection domain')
        ct=float(np.log10(transport.mu_caughey_thomas(10**g,300.)))
        return ContextPropertyPoint(g,mu,sigma,ct,float(np.log10(transport.E)+g+ct),derivative,valid,diagnostics)

    def offset_point(self, row, base, shift, *, require_observation=True):
        """Exact output-offset algebra; the physical state is not solved again."""
        z=float(shift)
        if not np.isfinite(z) or abs(z)>6.+1e-12:
            raise ValueError('Declared scalar source domain is [-6,6]')
        g=base.log_nH+z
        physical=bool(base.diagnostics.get('physical_state_available',False))
        underlying=bool(base.diagnostics.get('single_electron_Hall_valid',True))
        valid=bool(underlying and np.isfinite(g) and g>=13.)
        ct=float(np.log10(transport.mu_caughey_thomas(10**g,300.)))
        # Direct controls use CT(new predicted density); physical/output
        # controls retain the old state mobility, exactly as point(offset).
        mu=base.log_muH if physical else ct
        sigma=float(np.log10(transport.E)+g+mu)
        diag=dict(base.diagnostics,source_kind='offset',source_shift=z,
            observation_offset_dex=float(base.diagnostics.get('observation_offset_dex',0.)+z))
        diag['same_physical_state_for_output']=physical and diag['observation_offset_dex']==0.
        diag['observation_domain_valid']=valid
        if require_observation and not valid:
            raise ValueError('Output-offset candidate outside the observation domain')
        return ContextPropertyPoint(g,mu,sigma,ct,float(np.log10(transport.E)+g+ct),1.,valid,diag)
