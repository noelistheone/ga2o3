"""Independent quadrature, limits, positivity and full-chain derivative gates."""
from pathlib import Path
from types import SimpleNamespace
import sys,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.correlated_disorder_transport import correlation_kernel,correlated_fermi_transport
from sandbox.correlated_inventory_model import CorrelatedInventoryModel
from sandbox.coherent_fermi import published_coherent_profile,reference
from sandbox.coherent_transport import fermi_transport,quadrature
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log

OUT=c.BASE/'phase31_rate_redesign/gates_v1'


def independent_C(b,zeta):
    upper=float(np.log1p(b))
    points=[float(np.log1p(v/zeta)) for v in (1.,10.,50.) if zeta>0 and v/zeta<b]
    def fn(y):
        zt=zeta*np.expm1(y)
        return -np.expm1(-y)*np.exp(-zt)
    value=quad(fn,0.,upper,points=points,epsabs=1e-25,epsrel=3e-11,limit=200)[0]
    derivative=quad(lambda y:-zeta*np.expm1(y)*fn(y),0.,upper,
        points=points,epsabs=1e-25,epsrel=3e-11,limit=200)[0]
    return value,derivative


def independent_readout(v,profile):
    nlog,Mlog,Tlog,Alog,elllog=v;T=10.**Tlog;A=10.**Alog
    n=10.**nlog;NI=10.**Mlog
    mass=profile.electron_mass_ratio*M_E;eps=10.2*EPSILON0;kT=K_B_J*T
    Nc=profile.Nc_ref*(T/profile.T_ref)**1.5
    eta=reference().eta_from_logF(np.log(n/Nc));g=reference().log(eta)[1]
    ns=n*g;qs2=E_CHARGE**2*ns*1e6/(eps*kT)
    zeta=qs2*(10.**elllog*1e-9)**2
    nodes,weights=quadrature(512)
    lo=np.sqrt(max(0.,eta-50.));hi=np.sqrt(max(60.,eta+50.))
    t=(nodes+1)*(hi-lo)/2+lo;x=t*t
    f=expit(eta-x)
    w=weights*(hi-lo)/2*2*t**4*f*expit(x-eta)/float(expit(eta));w/=w.sum()
    b=8*mass*kT*x/(HBAR**2*qs2)
    # Independent adaptive scalar integration, not the production Gauss kernel.
    C=np.array([independent_C(bb,zeta)[0] for bb in b])
    R=NI*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)*x**(-1.5)*(coulomb_log(b)+A*C)
    R0=E_CHARGE/(mass*205.*1e-4);tau=1/(R+R0)
    t1=float(w@tau);t2=float(w@(tau*tau))
    rH=t2/t1**2;mu=E_CHARGE/mass*t2/t1*1e4
    return np.array([nlog-np.log10(rH),np.log10(mu),
        np.log10(E_CHARGE)+nlog+np.log10(E_CHARGE/mass*t1*1e4)])


def main():
    assert not (OUT/'protocol_before.json').exists(),'Preserve previous gate attempt'
    files=[Path(__file__),c.ROOT/'src/sandbox/correlated_disorder_transport.py',
        c.ROOT/'src/sandbox/correlated_inventory_model.py',
        c.ROOT/'docs/ML_SANDBOX_CORRELATED_DISORDER_THEORY_20261020.md',
        c.OLD/'data/scenarios49.csv']
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        thresholds=dict(kernel_scaled=3e-8,transport_log_difference=3e-7,
            transport_scaled_jacobian=3e-6,full_model_scaled_jacobian=3e-6),
        evidence_sha256=c.file_hashes(files),no_training_or_material_accuracy_claim=True,
        goal_complete=False))
    start=time.perf_counter();checks=[];details=[]
    def check(name,error,tolerance):
        ok=bool(np.isfinite(error) and error<=tolerance);checks.append(ok)
        details.append(dict(name=name,error=float(error),tolerance=float(tolerance),passed=ok))
    for b in (1e-6,1e-3,1.,100.,1e6):
        for zeta in (0.,1e-6,.1,1.,1e4):
            got,dgot=correlation_kernel(np.array([b]),zeta)
            expected,dexpected=independent_C(b,zeta)
            check(f'kernel:{b}:{zeta}',abs(got[0]-expected)/max(expected,1e-300),3e-8)
            check(f'kernel_zeta:{b}:{zeta}',abs(dgot[0]-dexpected)/(1+abs(dexpected)),3e-8)
            check(f'kernel_bounds:{b}:{zeta}',max(0.,-got[0],got[0]-float(coulomb_log(np.array([b]))[0])),1e-12)
    p=published_coherent_profile()
    kwargs=dict(Nc_ref_cm3=p.Nc_ref,T_ref_K=p.T_ref,mass_ratio=p.electron_mass_ratio)
    states=[np.array([15.,18.,np.log10(300.),-2.,-.3]),
        np.array([18.,20.,np.log10(300.),0.,.5]),
        np.array([20.5,21.,np.log10(330.),3.,1.])]
    for i,v in enumerate(states):
        def transport(w):
            return correlated_fermi_transport(w[0],w[1],10.**w[2],
                log10_A=w[3],log10_length_nm=w[4],**kwargs)
        a=transport(v)
        independent=independent_readout(v,p)
        check(f'independent_512_moments:{i}',np.max(abs(a.values-independent)),3e-7)
        check(f'Hall_identity:{i}',abs(a.values[2]-np.log10(E_CHARGE)-a.values[0]-a.values[1]),1e-12)
        check(f'Hall_factor_lower:{i}',max(0.,1.-10.**a.log_hall_factor),1e-12)
        zero=correlated_fermi_transport(v[0],v[1],10.**v[2],log10_A=None,log10_length_nm=v[4],**kwargs)
        base=fermi_transport(v[0],v[1],10.**v[2],extra_mobility_cm2Vs=205.,**kwargs)
        check(f'zero_amplitude_values:{i}',np.max(abs(zero.values-base.values)),1e-12)
        check(f'zero_amplitude_jacobian:{i}',np.max(abs(zero.jacobian[:,:3]-base.jacobian)),1e-12)
        for j in range(5):
            h=2e-5;u=np.eye(5)[j]*h
            fd=(transport(v+u).values-transport(v-u).values)/(2*h)
            error=np.max(abs(fd-a.jacobian[:,j])/(1+abs(fd)+abs(a.jacobian[:,j])))
            check(f'transport_jacobian:{i}:{j}',error,3e-6)
    cases=c.cases()
    rows=[SimpleNamespace(**c.design.inputs(cases[cases.element.eq(x)].iloc[0])) for x in ('Si','Sn')]
    rowH=SimpleNamespace(**vars(rows[0]));rowH.H_known=True;rowH.H_cm3=1e17;rows.append(rowH)
    w=np.array([.5,16.,.5,.1,.15,-.1,-1.,.2])
    def outputs(prediction):
        return np.r_[prediction.values[0],prediction.transport_values]
    for feature in ('charge_ratio','constant','dose','complex'):
        model=CorrelatedInventoryModel(feature=feature,profile=p)
        for i,row in enumerate(rows):
            z=.17;a=model.point(row,w,source_kind='share',source_shift=z,require_observation=False)
            J=np.r_[a.jacobian[[0]],a.transport_jacobian]
            check(f'root:{feature}:{i}',a.diagnostics['root_residual_log'],2e-8)
            check(f'atom_budget:{feature}:{i}',a.diagnostics['foreign_atom_relative_error'],3e-8)
            check(f'atom_derivative:{feature}:{i}',a.diagnostics['foreign_atom_derivative_error'],3e-7)
            for j in range(len(w)):
                h=2e-5;u=np.eye(len(w))[j]*h
                fd=(outputs(model.point(row,w+u,source_kind='share',source_shift=z,require_observation=False))
                    -outputs(model.point(row,w-u,source_kind='share',source_shift=z,require_observation=False)))/(2*h)
                check(f'model_parameter:{feature}:{i}:{j}',
                    np.max(abs(fd-J[:,j])/(1+abs(fd)+abs(J[:,j]))),3e-6)
            h=2e-5
            fd=(outputs(model.point(row,w,source_kind='share',source_shift=z+h,require_observation=False))
                -outputs(model.point(row,w,source_kind='share',source_shift=z-h,require_observation=False)))/(2*h)
            expected=np.r_[a.source_jacobian[0],a.transport_source_jacobian]
            check(f'model_source:{feature}:{i}',np.max(abs(fd-expected)/(1+abs(fd)+abs(expected))),3e-6)
            X=np.r_[a.input_jacobian[[0]],a.transport_input_jacobian]
            for j,name in enumerate(a.input_names):
                h=.01 if name=='growth_T_C' else 2e-5
                up=SimpleNamespace(**vars(row));down=SimpleNamespace(**vars(row))
                key={'log10_dose_cm3':'incorporated_cm3','log10_Tobs_K':'Tobs_K',
                     'log10_H_cm3':'H_cm3','growth_T_C':'growth_T_C'}[name]
                if name=='growth_T_C':
                    setattr(up,key,getattr(row,key)+h);setattr(down,key,getattr(row,key)-h)
                else:
                    setattr(up,key,getattr(row,key)*10**h);setattr(down,key,getattr(row,key)*10**(-h))
                fd=(outputs(model.point(up,w,source_kind='share',source_shift=z,require_observation=False))
                    -outputs(model.point(down,w,source_kind='share',source_shift=z,require_observation=False)))/(2*h)
                check(f'model_input:{feature}:{i}:{name}',
                    np.max(abs(fd-X[:,j])/(1+abs(fd)+abs(X[:,j]))),3e-6)
    c.write(OUT/'gates.json',dict(passed=all(checks),created_utc=c.e.now(),
        checks=len(checks),details=details,seconds=time.perf_counter()-start,
        maximum_scaled_or_absolute_error=max(r['error'] for r in details),
        numerical_states_only=True,new_model_fits=0,material_prediction_gain_established=False,
        code_and_input_hashes=c.file_hashes(files),goal_complete=False))
    print('Correlated-disorder gates',len(checks),'failed',sum(not z for z in checks),flush=True)
    for r in details:
        if not r['passed']:print(r,flush=True)
    assert all(checks),'Preserve failed gates and diagnose before versioned repair'


if __name__=='__main__':main()
