"""Shared inventory allocation coupled to a positive charge-disorder spectrum."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from .pool_share_model import PoolShareModel
from .relative_inventory_model import RelativeInventoryPrediction,L0,L1
from .context_inventory_model import field_features
from .charge_feasible_share import allocate_feasible_share
from .coherent_transport import fermi_transport
from .correlated_disorder_transport import correlated_fermi_transport

LN10=float(np.log(10.))


@dataclass
class CorrelatedInventoryPrediction(RelativeInventoryPrediction):
    uncorrelated_values:np.ndarray
    uncorrelated_jacobian:np.ndarray


class CorrelatedInventoryModel(PoolShareModel):
    def __init__(self,*,feature,profile,basis='process',db=None):
        if feature not in ('charge_ratio','constant','dose','complex'):
            raise ValueError('Declared disorder-amplitude feature required')
        super().__init__(basis=basis,temperature_mode='effective',profile=profile,db=db)
        self.feature=feature;self.inventory_parameter_names=tuple(self.parameter_names)
        self.parameter_names=self.inventory_parameter_names+(
            'log10_disorder_amplitude_scale','log10_disorder_length_nm')

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift)
        n=len(self.parameter_names);nb=len(self.inventory_parameter_names)
        if w.shape!=(n,) or not np.isfinite(w).all():
            raise ValueError('Finite matching correlated-inventory parameters required')
        if row.element not in ('Si','Sn') or not np.isfinite([
                row.incorporated_cm3,row.Tobs_K,row.H_cm3,row.growth_T_C]).all():
            raise ValueError('Finite explicit Si/Sn chemical and temperature inputs required')
        if row.incorporated_cm3<=0 or row.Tobs_K<=0 or row.H_cm3<0 or bool(row.H_known)!=(row.H_cm3>0):
            raise ValueError('Positive dose/temperature and explicit Hknown/unknown-zero required')
        if source_kind not in ('none','share','offset') or not np.isfinite(z) or abs(z)>6 or (source_kind=='none' and z!=0):
            raise ValueError('Explicit supported source coordinate required')
        layer,solver=self.engine_for(row).layer_for(row)
        ni=3 if row.H_known else 2;nv=n+1+ni+1
        source_col=n;input_col=n+1;growth_col=nv-1
        chain=np.zeros((layer.np,nv))
        logistic=float(expit(w[0]));T=10**(L0+(L1-L0)*logistic)
        chain[0,0]=(L1-L0)*logistic*(1-logistic);chain[1,1]=1.
        chain[layer.idose,input_col]=1.;chain[layer.iobs,input_col+1]=1.
        if row.H_known:chain[2+layer.b+1,input_col+2]=1.
        phi=field_features(row,self.basis)
        psi=float(phi@w[2:nb]);dpsi=np.zeros(nv);dpsi[2:nb]=phi
        if self.basis!='constant':dpsi[input_col]=w[3]/2.
        if self.basis=='process':dpsi[growth_col]=w[4]/300.
        if source_kind=='share':psi+=LN10*z;dpsi[source_col]=LN10
        pars=layer.parameters(T=T,Nd=10**w[1],weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        inventory=allocate_feasible_share(solver,pars,chain,psi,dpsi)
        cold=solver.solve(pars,inventory,chain)
        dfeature=np.zeros(nv)
        if self.feature=='charge_ratio':
            logfeature=float(cold.values[1]-cold.values[0])
            dfeature=cold.jacobian[1]-cold.jacobian[0]
        elif self.feature=='dose':
            logfeature=float(np.log10(row.incorporated_cm3)-19.)
            dfeature[input_col]=1.
        elif self.feature=='complex':
            amount=float(inventory.totals[solver.complex].sum())
            if amount<=0:raise ValueError('Positive complex-center population required')
            logfeature=float(np.log10(amount)-19.)
            dfeature=inventory.jacobian[solver.complex].sum(axis=0)/(amount*LN10)
        else:logfeature=0.
        dA=dfeature.copy();dA[nb]+=1.
        dlength=np.zeros(nv);dlength[nb+1]=1.
        kwargs=dict(Nc_ref_cm3=self.profile.Nc_ref,T_ref_K=self.profile.T_ref,
            mass_ratio=self.profile.electron_mass_ratio,hall=True)
        tr=correlated_fermi_transport(cold.values[0],cold.values[2],float(row.Tobs_K),
            log10_A=w[nb]+logfeature,log10_length_nm=w[nb+1],**kwargs)
        readout=tr.jacobian[:,:2]@cold.jacobian[[0,2]]
        readout[:,input_col+1]+=tr.jacobian[:,2]
        readout+=tr.jacobian[:,3,None]*dA[None,:]
        readout+=tr.jacobian[:,4,None]*dlength[None,:]
        base=fermi_transport(cold.values[0],cold.values[2],float(row.Tobs_K),
            extra_mobility_cm2Vs=205.,**kwargs)
        baseJ=base.jacobian[:,:2]@cold.jacobian[[0,2]]
        baseJ[:,input_col+1]+=base.jacobian[:,2]
        values=cold.values.copy();J=cold.jacobian.copy()
        values[0]=tr.values[0];J[0]=readout[0]
        physical=values.copy();tv=tr.values[1:].copy();TJ=readout[1:].copy()
        if source_kind=='offset':
            values[0]+=z;tv[1]+=z;J[0,source_col]+=1.;TJ[1,source_col]+=1.
        valid=bool(cold.diagnostics['single_electron_Hall_valid'] and values[0]>=13.)
        if require_observation and not valid:raise ValueError('Outside single-electron Hall/detection domain')
        dg=dict(cold.diagnostics,source_kind=source_kind,source_shift=z,
            share_basis=self.basis,share_coefficients=w[2:nb].tolist(),Tf_K=T,
            temperature_mode='effective',temperature_scope='effective endpoint;not measured thermal history',
            coherent_transport=tr.diagnostics,hall_factor=tr.diagnostics['readout_hall_factor'],
            disorder_feature=self.feature,disorder_log10_feature=logfeature,
            shared_log10_amplitude_scale=float(w[nb]),shared_log10_length_nm=float(w[nb+1]),
            same_physical_state_for_output=bool(source_kind!='offset' or z==0),
            observation_domain_valid=valid,numerical_observation_continuation=not valid,
            observation_offset_dex=z if source_kind=='offset' else 0.,
            learning_scope='shared atomic allocation and continuum disorder-spectrum closure;no measured correlation/neutral-impurity identification',
            charged_inventory_and_mean_neutrality_unchanged_by_correlation=True,
            scattering_changes_free_charge_state=False,all_properties_coupled_via_same_Hall_moments=True)
        names=('log10_dose_cm3','log10_Tobs_K')+(('log10_H_cm3',) if row.H_known else ())+('growth_T_C',)
        return CorrelatedInventoryPrediction(values,J[:,:n],J[:,source_col],physical,dg,
            tv,TJ[:,:n],TJ[:,source_col],J[:,input_col:],TJ[:,input_col:],names,
            base.values.copy(),baseJ[:,:n].copy())
