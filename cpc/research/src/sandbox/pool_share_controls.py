"""Matched reference scenarios and a strong empirical activation control."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from scipy.special import expit,logsumexp,logit
from .relative_inventory_model import RelativeInventoryModel,L0,L1
from .context_inventory_model import field_features

LN10=float(np.log(10.))


class PoolReferenceModel:
    def __init__(self,*,temperature_mode,profile):
        if temperature_mode not in ('effective','growth'):raise ValueError('Declared reference scenario required')
        self.temperature_mode=temperature_mode;self.profile=profile;self.engines={}
        self.parameter_names=('logit_Tf','log10_background') if temperature_mode=='effective' else ('log10_background',)

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float)
        if w.shape!=(len(self.parameter_names),):raise ValueError('Matching reference parameters required')
        if row.scenario_pO2_atm not in self.engines:
            self.engines[row.scenario_pO2_atm]=RelativeInventoryModel(dict(field='inventory'),profile=self.profile,pO2_atm=row.scenario_pO2_atm)
        if self.temperature_mode=='effective':v=np.r_[w,0.];growth_factor=0.
        else:
            T=float(row.growth_T_C+273.15);frac=(np.log10(T)-L0)/(L1-L0)
            if not 0<frac<1:raise ValueError('Nominalgrowth reference must be interior to600-1950K')
            v=np.r_[logit(frac),w[0],0.];growth_factor=1/(T*LN10*(L1-L0)*frac*(1-frac))
        kind='field' if source_kind=='share' else source_kind
        p=self.engines[row.scenario_pO2_atm].point(row,v,source_kind=kind,source_shift=source_shift,require_observation=require_observation)
        keep=[0,1] if self.temperature_mode=='effective' else [1]
        return replace(p,jacobian=p.jacobian[:,keep],transport_jacobian=p.transport_jacobian[:,keep],
            input_jacobian=np.c_[p.input_jacobian,p.jacobian[:,0]*growth_factor],
            transport_input_jacobian=np.c_[p.transport_input_jacobian,p.transport_jacobian[:,0]*growth_factor],
            input_names=p.input_names+('growth_T_C',),diagnostics=dict(p.diagnostics,temperature_mode=self.temperature_mode,
                reference_only=True,reference_fallback_scope='discrete model;not infinite penalty limit of the absolute linear-logit family'))

    def counters(self):
        a=[m.counters() for m in self.engines.values()]
        return {k:sum(v[k] for v in a) for k in ('hot_solve_calls','hot_cache_hits','cold_solve_calls')}


class DirectActivationModel:
    """nH = empirical background + C*sigmoid(psi) + H*sigmoid(eta).

    This predicts the Hall label directly. Its background/H coefficient do not
    identify true ionized donors, and there is no inferred defect state.
    """
    def __init__(self,*,basis):
        if basis not in ('none','constant','dose','process'):raise ValueError('Declared activation basis required')
        self.basis=basis;n=dict(none=0,constant=1,dose=2,process=4)[basis]
        self.parameter_names=('log10_empirical_background','H_activation_logit')+tuple(f'activation_coefficient_{i}' for i in range(n))

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift)
        if w.shape!=(len(self.parameter_names),) or not np.isfinite(w).all():raise ValueError('Matching finite activation parameters required')
        if source_kind not in ('none','offset') or not np.isfinite(z) or abs(z)>6 or (source_kind=='none' and z!=0):
            raise ValueError('Empirical activation supports only declared output offsets')
        if row.incorporated_cm3<=0 or bool(row.H_known)!=(row.H_cm3>0):raise ValueError('Explicit positive chemistry/H scenario required')
        phi=np.zeros(0) if self.basis=='none' else field_features(row,self.basis);psi=float(phi@w[2:]);eta=float(w[1])
        terms=np.array([LN10*w[0],np.log(row.incorporated_cm3)-np.logaddexp(0.,-psi),
            np.log(row.H_cm3)-np.logaddexp(0.,-eta) if row.H_known else -np.inf])
        total=float(logsumexp(terms));weights=np.exp(terms-total);f=float(expit(psi));a=float(expit(eta))
        value=total/LN10+(z if source_kind=='offset' else 0.)
        J=np.r_[weights[0],weights[2]*(1-a)/LN10,weights[1]*(1-f)*phi/LN10]
        names=('log10_dose_cm3','log10_Tobs_K')+(('log10_H_cm3',) if row.H_known else ())+('growth_T_C',)
        X=np.zeros(len(names));X[0]=weights[1]
        if self.basis in ('dose','process'):X[0]+=weights[1]*(1-f)*w[3]/(2*LN10)
        if row.H_known:X[2]=weights[2]
        if self.basis=='process':X[-1]=weights[1]*(1-f)*w[4]/(300*LN10)
        valid=bool(np.isfinite(value) and value>=13.)
        if require_observation and not valid:raise ValueError('Empirical activation below declared Hall detection floor')
        return SimpleNamespace(values=np.array([value]),jacobian=J[None,:],source_jacobian=np.array([float(source_kind=='offset')]),
            input_names=names,input_jacobian=X[None,:],diagnostics=dict(observation_domain_valid=valid,
                physical_state_available=False,same_physical_state_for_output=False,
                activation_scope='direct Hall regression;no inferred defect concentrations or microscopic H activation truth'))

    def counters(self):return dict(hot_solve_calls=0,hot_cache_hits=0,cold_solve_calls=0)
