"""Independent free-statistics, full-state and coherent-readout gates."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import FrozenInstanceError
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import mpmath as mp
from sandbox.coherent_fermi import reference,published_coherent_profile,CoherentMaterialProfile,TABLE
from sandbox.coherent_transport import fermi_transport
from sandbox.ionized_transport import ionized_fermi_transport,normalized_moments,ionized_rate_s,K_B_J,E_CHARGE,M_E
from sandbox.material_profile import PUBLISHED_PROFILE
from sandbox.coherent_hall_model import CoherentHallModel
from sandbox.donor_mixture import DonorMixtureLayer
from sandbox.kroger_db import load
import phase15_prediction as study
import phase2_energy as energy


def main():
    checks=[];derivative_errors=[];clock=time.perf_counter();ref=reference();profile=published_coherent_profile()
    def add(name,good,**kw):checks.append(dict(name=name,passed=bool(good),**kw))
    def err(a,b):return float(np.max(abs(np.asarray(a)-b)/(1+abs(np.asarray(b)))))
    def derivative(name,actual,wanted,tolerance=3e-5,**kw):
        e=err(actual,wanted);derivative_errors.append(e);add(name,e<tolerance,error=e,**kw)
    # Polylogarithms independently probe quadrature generation, spline and tails.
    mp.mp.dps=45
    etas=np.r_[[-1000,-80,-21,-20.000001,-20,-19.999999,-12,-3],np.linspace(-2,12,13),[35,49.999999,50,50.000001,59.999999,60,60.000001,100,500,1e4]]
    for eta in etas:
        a=-mp.polylog(mp.mpf('1.5'),-mp.exp(float(eta)))
        b=-mp.polylog(mp.mpf('.5'),-mp.exp(float(eta)))
        c=-mp.polylog(mp.mpf('-.5'),-mp.exp(float(eta)))
        wanted=np.array([float(mp.re(mp.log(a))),float(mp.re(b/a)),float(mp.re(c/a-(b/a)**2))])
        actual=np.array(ref.log(eta));e=err(actual,wanted)
        add('FD_against_highprecision_polylog',e<2e-7,eta=float(eta),error=e)
        inverse=ref.eta_from_logF(actual[0]);add('FD_inverse',abs(inverse-eta)<2e-10*(1+abs(eta)),eta=float(eta),error=abs(inverse-eta))
    grid=np.r_[np.linspace(-1e4,-20,500),np.linspace(-20,60,32001),np.geomspace(60,1e4,500)]
    _,g,_=ref.log(grid);add('positive_free_compressibility_full_declared_root_range',np.isfinite(g).all() and min(g)>0,min_slope=float(min(g)))
    add('profile_roundtrip_type_hash',CoherentMaterialProfile.from_dict(profile.to_dict()).fingerprint==profile.fingerprint)
    add('profile_statistics_tag_distinct',profile.fingerprint!=PUBLISHED_PROFILE.fingerprint)
    try:profile.Nc_ref=1.;immutable=False
    except FrozenInstanceError:immutable=True
    add('profile_immutable',immutable)
    for T in (10.,40.,300.,1600.):
        for offset in (-5.,-.1,.2,7.):
            EF=profile.bands(T)[1]+offset;a=profile.carriers(EF,T);h=1e-5
            derivative('profile_EF_chain',(profile.carriers(EF+h,T)[0]-profile.carriers(EF-h,T)[0])/(2*h),a[1],tolerance=3e-4,T=T,offset=offset)
            dT=1e-4*T
            derivative('profile_T_chain',(profile.carriers(EF,T+dT)[0]-profile.carriers(EF,T-dT)[0])/(2*dT),a[2],tolerance=3e-4,T=T,offset=offset)
            add('legacySTH_preserved',np.array_equal(a[0][2:],PUBLISHED_PROFILE.carriers(EF,T)[0][2:]),T=T,offset=offset)
    for eta,T in ((-12.,40.),(0.,300.),(3.,300.),(30.,150.),(200.,40.)):
        raw=ionized_fermi_transport(eta,T,2e18,mass_ratio=profile.electron_mass_ratio)
        wanted=np.log10([raw.Hall_density_cm3,raw.Hall_mobility_cm2Vs,raw.conductivity_Scm])
        result=fermi_transport(np.log10(raw.n_cm3),np.log10(raw.Nq2_cm3),T,Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,mass_ratio=profile.electron_mass_ratio)
        add('pureII_against_independent_adaptive_moments',max(abs(result.values-wanted))<2e-7,eta=eta,T=T,error=float(max(abs(result.values-wanted))))
    # Different source integration, including the finite constant-tau channel.
    for eta in (-12.,0.,8.,40.):
        T=300.;NI=2e18;m=profile.electron_mass_ratio
        raw=ionized_fermi_transport(eta,T,NI,mass_ratio=m);ns=raw.n_screen_cm3
        tau0=205e-4*m*M_E/E_CHARGE
        def tau(x):return 1/(ionized_rate_s(x*K_B_J*T,T,NI,ns,mass_ratio=m)+1/tau0)
        norm=tau(max(1.5,eta));mom,_=normalized_moments(eta,lambda x:tau(x)/norm)
        rh=mom[2]*mom[0]/mom[1]**2;mud=E_CHARGE/(m*M_E)*norm*mom[1]/mom[0]*1e4
        wanted=np.log10([raw.n_cm3/rh,mud*rh,E_CHARGE*raw.n_cm3*mud])
        result=fermi_transport(np.log10(raw.n_cm3),np.log10(NI),T,Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,mass_ratio=m,extra_mobility_cm2Vs=205.)
        add('extra_channel_adaptive_moments',max(abs(result.values-wanted))<2e-7,eta=eta,error=float(max(abs(result.values-wanted))))
    for logn,logNI,T in ((14.,12.,300.),(18.,18.,300.),(20.,22.,40.),(18.,25.,650.),(21.,14.,40.)):
        for extra in (np.inf,205.):
            def forward(v,order=256):
                return fermi_transport(v[0],v[1],10**v[2],Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
                    mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=extra,order=order)
            v=np.array([logn,logNI,np.log10(T)]);result=forward(v)
            for j in range(3):
                h=2e-5;x=v.copy();y=v.copy();x[j]+=h;y[j]-=h
                derivative('coherent_readout_fullinput_chain',(forward(x).values-forward(y).values)/(2*h),result.jacobian[:,j],logn=logn,logNI=logNI,T=T,extra=None if np.isinf(extra) else extra,j=j)
            fine=forward(v,512)
            add('moment_quadrature_refinement',max(abs(result.values-fine.values))<2e-7,error=float(max(abs(result.values-fine.values))),logn=logn,logNI=logNI,T=T,extra=None if np.isinf(extra) else extra)
            add('Hall_Cauchy_and_conductivity',result.log_hall_factor>=0 and abs(result.diagnostics['Hall_conductivity_identity_error_dex'])<1e-12)
    db=load()
    # Complete original raw IFT, including observation T and material directions.
    for elements,doses,To in ((('Si',),(1e18,),40.),(('Sn',),(3e18,),300.),(('Si','H'),(5.4e17,1.8e17),150.)):
        layer=DonorMixtureLayer(db,elements,profile=profile,hall='rH1',fixed_fraction=1.)
        pars=layer.parameters(T=1400.,Nd=1e16,doses=doses,Tobs=To,binding_eV=.03,screen_s=.2)
        result=layer.predict(pars)
        for j,name in enumerate(layer.parameter_names):
            h=1e-5;x=pars.copy();y=pars.copy();x[j]+=h;y[j]-=h
            # At the fixed lower slope boundary only supported one-sided tests.
            if j==layer.iloc+1:
                fd=(layer.predict(x).values-result.values)/h
            else:fd=(layer.predict(x).values-layer.predict(y).values)/(2*h)
            derivative('complete_state_IFT_with_coherent_profile',fd,result.jacobian[:,j],elements=elements,T=To,parameter=name)
        same=layer.parameters(T=1400.,Nd=1e16,doses=doses,Tobs=1400.,binding_eV=.03,screen_s=.2)
        layer.predict(same);s=layer.last_state
        add('hot_cold_sameT_identity',max(abs(s['hot_charge']-s['cold_charge'])/(1+abs(s['hot_charge'])))<2e-8,elements=elements)
    for method in study.NEW_METHODS:
        model=study.mapper(method);_,_,starts=study.core.specification(method)
        for T in (40.,300.):
            for element,C,H in (('Si',1e18,0.),('Sn',3e18,0.),('Si',5.4e17,1.8e17)):
                row=SimpleNamespace(element=element,incorporated_cm3=C,growth_T_C=830.,Tobs_K=T,H_known=H>0,H_cm3=H)
                w=np.array(starts[0]);result=model.point(row,w)
                for j,name in enumerate(model.parameter_names):
                    h=1e-5;x=w.copy();y=w.copy();x[j]+=h;y[j]-=h
                    plus=model.point(row,x);minus=model.point(row,y)
                    derivative('actual_sharedmodel_carrier_gradient',(plus.values-minus.values)/(2*h),result.jacobian[:,j],method=method,T=T,element=element,parameter=name)
                    derivative('actual_sharedmodel_mu_sigma_gradient',(plus.transport_values-minus.transport_values)/(2*h),result.transport_jacobian[:,j],method=method,T=T,element=element,parameter=name)
                for kind in ('stock','Tf','offset'):
                    z=.2;h=1e-5;r=model.point(row,w,source_kind=kind,source_shift=z)
                    p=model.point(row,w,source_kind=kind,source_shift=z+h);m=model.point(row,w,source_kind=kind,source_shift=z-h)
                    derivative('actual_source_carrier_gradient',(p.values-m.values)/(2*h),r.source_jacobian,method=method,kind=kind,T=T,element=element)
                    derivative('actual_source_transport_gradient',(p.transport_values-m.transport_values)/(2*h),r.transport_source_jacobian,method=method,kind=kind,T=T,element=element)
                    zero=model.point(row,w,source_kind=kind,source_shift=0.)
                    current=model.point(row,w)
                    add('source_exact_zero_update_same_solver_cache',np.array_equal(current.values,zero.values) and np.array_equal(current.transport_values,zero.transport_values),method=method,kind=kind)
                    warm_error=max(abs(result.values-zero.values).max(),abs(result.transport_values-zero.transport_values).max())
                    add('source_zero_warmstart_roundoff',warm_error<1e-10,error=float(warm_error),method=method,kind=kind)
    paths=(Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase15_prediction.py',ROOT/'src/sandbox/coherent_fermi.py',
        ROOT/'src/sandbox/coherent_transport.py',ROOT/'src/sandbox/coherent_hall_model.py',ROOT/'docs/ML_SANDBOX_COHERENT_HALL_THEORY_20261018.md',TABLE)
    payload=dict(passed=all(c['passed'] for c in checks),created_utc=energy.now(),checks=checks,
        maximum_derivative_error=max(derivative_errors),seconds=time.perf_counter()-clock,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        scope='independent mathematics/stateIFT/readoutchain;notmaterialaccuracy',goal_complete=False)
    energy.write(study.OUT/'numerical_gates.json',payload)
    print(json.dumps(dict(passed=payload['passed'],checks=len(checks),maximum_derivative_error=payload['maximum_derivative_error'],
        seconds=payload['seconds'],failed=[c for c in checks if not c['passed']]),indent=2))
    if not payload['passed']:raise RuntimeError('Coherent numerical gate failed;training prohibited')


if __name__=='__main__':main()
