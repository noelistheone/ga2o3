"""A shared absolute inventory logit followed by charge and coherent Hall solves."""
import numpy as np
from scipy.special import expit
from .relative_inventory_model import RelativeInventoryModel,RelativeInventoryPrediction,L0,L1
from .context_inventory_model import field_features
from .absolute_pool_share import allocate_absolute_share
from .coherent_transport import fermi_transport

LN10=float(np.log(10.))


class PoolShareModel:
    def __init__(self,*,basis,temperature_mode,profile,db=None):
        if basis not in ('constant','dose','process') or temperature_mode not in ('effective','growth'):
            raise ValueError('Explicit share basis and effective/growth reference scenario required')
        self.basis=basis;self.temperature_mode=temperature_mode;self.profile=profile;self.db=db;self.engines={}
        self.nbase=2 if temperature_mode=='effective' else 1
        n=dict(constant=1,dose=2,process=4)[basis]
        self.parameter_names=(('logit_Tf','log10_background') if self.nbase==2 else ('log10_background',))+tuple(f'share_logit_coefficient_{i}' for i in range(n))

    def engine_for(self,row):
        pressure=float(row.scenario_pO2_atm)
        if not np.isfinite(pressure) or pressure<=0:raise ValueError('Explicit positive oxygen scenario required')
        if pressure not in self.engines:
            self.engines[pressure]=RelativeInventoryModel(dict(field='inventory'),profile=self.profile,pO2_atm=pressure,db=self.db)
        return self.engines[pressure]

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift);n=len(self.parameter_names)
        if w.shape!=(n,) or not np.isfinite(w).all():raise ValueError('Finite matching share-model parameters required')
        if row.element not in ('Si','Sn') or not np.isfinite([row.incorporated_cm3,row.Tobs_K,row.H_cm3,row.growth_T_C]).all():
            raise ValueError('Explicit finite Si/Sn chemical and temperature inputs required')
        if row.incorporated_cm3<=0 or row.Tobs_K<=0 or row.H_cm3<0 or bool(row.H_known)!=(row.H_cm3>0):
            raise ValueError('Positive dose/temperature and explicit Hknown or unknownzero scenario required')
        if source_kind not in ('none','share','offset') or not np.isfinite(z) or abs(z)>6 or (source_kind=='none' and z!=0):
            raise ValueError('Supported source kind and explicit coordinate in [-6,6] required')
        layer,solver=self.engine_for(row).layer_for(row)
        ni=3 if row.H_known else 2;nv=n+1+ni+1;source_col=n;input_col=n+1;growth_col=nv-1
        chain=np.zeros((layer.np,nv))
        if self.temperature_mode=='effective':
            s=float(expit(w[0]));T=10**(L0+(L1-L0)*s);chain[0,0]=(L1-L0)*s*(1-s);ibg=1
        else:
            T=float(row.growth_T_C+273.15)
            if not 300<=T<=1950:raise ValueError('Nominal-growth reference outside declared temperature domain')
            chain[0,growth_col]=1/(LN10*T);ibg=0
        chain[1,ibg]=1.;chain[layer.idose,input_col]=1.;chain[layer.iobs,input_col+1]=1.
        if row.H_known:chain[2+layer.b+1,input_col+2]=1.
        phi=field_features(row,self.basis);psi=float(phi@w[self.nbase:]);dpsi=np.zeros(nv)
        dpsi[self.nbase:n]=phi
        if self.basis!='constant':dpsi[input_col]=w[self.nbase+1]/2.
        if self.basis=='process':dpsi[growth_col]=w[self.nbase+2]/300.
        if source_kind=='share':psi+=LN10*z;dpsi[source_col]=LN10
        pars=layer.parameters(T=T,Nd=10**w[ibg],weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=float(row.scenario_pO2_atm),Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        inventory=allocate_absolute_share(solver,pars,chain,psi,dpsi);cold=solver.solve(pars,inventory,chain)
        tr=fermi_transport(cold.values[0],cold.values[2],float(row.Tobs_K),Nc_ref_cm3=self.profile.Nc_ref,
            T_ref_K=self.profile.T_ref,mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=205.,hall=True)
        readout=tr.jacobian[:,:2]@cold.jacobian[[0,2]];readout[:,input_col+1]+=tr.jacobian[:,2]
        values=cold.values.copy();values[0]=tr.values[0];J=cold.jacobian.copy();J[0]=readout[0]
        physical=values.copy();tv=tr.values[1:].copy();TJ=readout[1:].copy()
        if source_kind=='offset':values[0]+=z;tv[1]+=z;J[0,source_col]+=1.;TJ[1,source_col]+=1.
        valid=bool(cold.diagnostics['single_electron_Hall_valid'] and values[0]>=13.)
        if require_observation and not valid:raise ValueError('Outside implemented single-electron Hall/detection domain')
        dg=dict(cold.diagnostics,source_kind=source_kind,source_shift=z,share_basis=self.basis,
            share_coefficients=w[self.nbase:].tolist(),temperature_mode=self.temperature_mode,Tf_K=T,
            temperature_scope='effective endpoint' if self.temperature_mode=='effective' else 'nominalgrowth reference scenario;not complete measured freezehistory',
            pO2_atm=float(row.scenario_pO2_atm),coherent_transport=tr.diagnostics,hall_factor=tr.diagnostics['readout_hall_factor'],
            same_physical_state_for_output=bool(source_kind!='offset' or z==0),observation_domain_valid=valid,
            observation_offset_dex=z if source_kind=='offset' else 0.,numerical_observation_continuation=not valid,
            learning_scope='shared absolute parent-atom fraction;not microscopic donorbinding or kinetics')
        names=('log10_dose_cm3','log10_Tobs_K')+(('log10_H_cm3',) if row.H_known else ())+('growth_T_C',)
        return RelativeInventoryPrediction(values,J[:,:n],J[:,source_col],physical,dg,tv,TJ[:,:n],TJ[:,source_col],
            J[:,input_col:],TJ[:,input_col:],names)

    def counters(self):
        values=[m.counters() for m in self.engines.values()]
        return {k:sum(v[k] for v in values) for k in ('hot_solve_calls','hot_cache_hits','cold_solve_calls')}
