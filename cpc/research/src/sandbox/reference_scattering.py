"""Small-capacity reference-inventory plus one shared positive scattering scale.

The zero shared/source shifts reproduce the existing effective ParentReference.
The optional source field uses its original relative inventory odds coordinate;
it is distinct from the excess-share source coordinate of the joint learner.
"""
import numpy as np
from scipy.special import expit
from .pool_share_model import PoolShareModel
from .relative_inventory_model import RelativeInventoryPrediction,L0,L1
from .relative_inventory import allocate_relative_odds
from .learnable_constant_tau import learnable_fermi_transport
from .joint_inventory_scattering import BASE_LOG_MU0


class ReferenceScatteringModel(PoolShareModel):
    def __init__(self,*,profile,db=None):
        super().__init__(basis='constant',temperature_mode='effective',profile=profile,db=db)
        self.parameter_names=('logit_Tf','log10_background','shared_log10_mu0_shift')

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift);n=3
        if w.shape!=(n,) or not np.isfinite(w).all():raise ValueError('Three finite reference/scattering parameters required')
        if row.element not in ('Si','Sn') or not np.isfinite([
                row.incorporated_cm3,row.Tobs_K,row.H_cm3,row.growth_T_C]).all():
            raise ValueError('Explicit finite Si/Sn input scenario required')
        if row.incorporated_cm3<=0 or row.Tobs_K<=0 or row.H_cm3<0 or bool(row.H_known)!=(row.H_cm3>0):
            raise ValueError('Positive dose/temperature and explicit Hknown/unknown-zero required')
        if source_kind not in ('none','share','offset') or not np.isfinite(z) or abs(z)>6 or (source_kind=='none' and z!=0):
            raise ValueError('Explicit supported source shift in[-6,6] required')
        layer,solver=self.engine_for(row).layer_for(row)
        ni=3 if row.H_known else 2;nv=n+1+ni+1;source_col=n;input_col=n+1
        chain=np.zeros((layer.np,nv))
        s=float(expit(w[0]));T=10**(L0+(L1-L0)*s)
        chain[0,0]=(L1-L0)*s*(1-s);chain[1,1]=1.
        chain[layer.idose,input_col]=1.;chain[layer.iobs,input_col+1]=1.
        if row.H_known:chain[2+layer.b+1,input_col+2]=1.
        pars=layer.parameters(T=T,Nd=10**w[1],weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        dh=np.zeros(nv)
        if source_kind=='share':dh[source_col]=1.
        inventory=allocate_relative_odds(solver,pars,chain,z if source_kind=='share' else 0.,dh)
        cold=solver.solve(pars,inventory,chain)
        tr=learnable_fermi_transport(cold.values[0],cold.values[2],float(row.Tobs_K),
            log10_extra_mobility=BASE_LOG_MU0+w[-1],Nc_ref_cm3=self.profile.Nc_ref,
            T_ref_K=self.profile.T_ref,mass_ratio=self.profile.electron_mass_ratio,hall=True)
        readout=tr.jacobian[:,:2]@cold.jacobian[[0,2]]
        readout[:,input_col+1]+=tr.jacobian[:,2]
        readout[:,2]+=tr.jacobian[:,3]
        values=cold.values.copy();J=cold.jacobian.copy()
        values[0]=tr.values[0];J[0]=readout[0]
        physical=values.copy();tv=tr.values[1:].copy();TJ=readout[1:].copy()
        if source_kind=='offset':
            values[0]+=z;tv[1]+=z;J[0,source_col]+=1.;TJ[1,source_col]+=1.
        valid=bool(cold.diagnostics['single_electron_Hall_valid'] and values[0]>=13.)
        if require_observation and not valid:raise ValueError('Outside implemented single-electron Hall/detection domain')
        dg=dict(cold.diagnostics,source_kind=source_kind,source_shift=z,Tf_K=T,
            temperature_mode='effective',reference_only=True,
            temperature_scope='effective reference endpoint;not measured freeze history',
            source_coordinate='reference-relative log10 inventory odds' if source_kind=='share' else source_kind,
            coherent_transport=tr.diagnostics,hall_factor=tr.diagnostics['readout_hall_factor'],
            shared_log10_mu0_shift=float(w[-1]),extra_mobility_cm2Vs=10**(BASE_LOG_MU0+w[-1]),
            same_physical_state_for_output=bool(source_kind!='offset' or z==0),
            observation_domain_valid=valid,observation_offset_dex=z if source_kind=='offset' else 0.,
            numerical_observation_continuation=not valid,
            learning_scope='one global positive constant-tau scale;reference inventory unchanged before permitted source update',
            scattering_changes_free_charge_state=False,material_Born_validity_established=False)
        names=('log10_dose_cm3','log10_Tobs_K')+(('log10_H_cm3',) if row.H_known else ())+('growth_T_C',)
        return RelativeInventoryPrediction(values,J[:,:n],J[:,source_col],physical,dg,
            tv,TJ[:,:n],TJ[:,source_col],J[:,input_col:],TJ[:,input_col:],names)
