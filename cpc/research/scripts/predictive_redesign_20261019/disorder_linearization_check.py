"""Conditional continuum-potential diagnostic from actual saved cold states."""
from pathlib import Path
import argparse,sys
import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.coherent_fermi import published_coherent_profile,reference
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J


def H(x):
    x=float(x)
    if x==0:return 1.
    if x<=1:
        value=quad(lambda t:t*t/(1+t*t)**2*np.exp(-(x*t)**2),0,np.inf,epsabs=1e-12,epsrel=1e-10)[0]*4/np.pi
    else:
        value=quad(lambda v:v*v/(1+(v/x)**2)**2*np.exp(-v*v),0,12,epsabs=1e-12,epsrel=1e-10)[0]*4/(np.pi*x**3)
    return float(value)


def diagnostic(n,q,T,ns,g,A=0.,length_nm=0.):
    eps=10.2*EPSILON0;qs=np.sqrt(E_CHARGE**2*ns*1e6/(eps*K_B_J*T))
    x=qs*length_nm*1e-9;factor=1+A*H(x)
    rms_J=np.sqrt(E_CHARGE**4*q*1e6/(8*np.pi*eps**2*qs)*factor)
    energy=K_B_J*T/g
    return dict(qs_m_inv=float(qs),correlation_qs_length=float(x),variance_factor=float(factor),
        potential_rms_eV=float(rms_J/E_CHARGE),compressibility_energy_eV=float(energy/E_CHARGE),
        linearization_chi=float(rms_J/energy),chi_ge_one=bool(rms_J>=energy),
        diagnosis_conditional_not_material_phase_or_Born_certificate=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['old','31','32'],default='old')
    args=parser.parse_args();checks=[];details=[];rows=[]
    if args.phase=='old':
        out=c.BASE/'disorder_linearization_old_v1'
        source=c.BASE/'phase31_rate_redesign/probe_v2/saved_state_features.csv'
    else:
        folder='phase31_rate_redesign' if args.phase=='31' else 'phase32_rate_controls'
        root=c.BASE/folder;out=root/'linearization_v1'
        source=root/'prediction/prediction_diagnostics.json'
    assert not (out/'protocol_before.json').exists()
    c.write(out/'protocol_before.json',dict(created_utc=c.e.now(),code_sha256=c.e.sha(__file__),
        source_sha256=c.e.sha(source),theory_sha256=c.e.sha(c.ROOT/'docs/ML_SANDBOX_DISORDER_LINEARIZATION_CHECK_20261020.md'),
        diagnosis_not_training_or_selection=True,conditional_continuum_assumptions=True))
    for x in [0.,.001,.03,.3,1.,3.,10.,30.]:
        actual=H(x);closed=(1+2*x*x)*erfcx(x)-2*x/np.sqrt(np.pi)
        rel=abs(actual-closed)/actual;checks.extend([0<actual<=1+1e-12,rel<3e-7])
        details.append(dict(x=x,H=actual,closed=closed,relative_error=rel))
    if args.phase=='old':
        out=c.BASE/'disorder_linearization_old_v1'
        source=c.BASE/'phase31_rate_redesign/probe_v2/saved_state_features.csv'
        frame=c.read(source);profile=published_coherent_profile()
        for r in frame.itertuples():
            T=300.;Nc=profile.Nc_ref*(T/profile.T_ref)**1.5
            eta=reference().eta_from_logF(np.log(r.free_n_cm3/Nc));_,g,_=reference().log(eta)
            rows.append(dict(variant=r.variant,case_id=r.case_id,sample_id=r.sample_id,
                element=r.element,A=0.,length_nm=0.,free_n_cm3=r.free_n_cm3,Nq2_cm3=r.Nq2_cm3,
                **diagnostic(r.free_n_cm3,r.Nq2_cm3,T,r.free_n_cm3*g,g)))
    else:
        folder='phase31_rate_redesign' if args.phase=='31' else 'phase32_rate_controls'
        root=c.BASE/folder;out=root/'linearization_v1'
        source=root/'prediction/prediction_diagnostics.json'
        for item in c.document(source):
            dg=item.get('diagnostics') or {};tr=dg.get('coherent_transport')
            if not tr:continue
            g=tr['n_screen_cm3']/tr['free_n_cm3']
            rows.append(dict(row_id=item['row_id'],A=tr.get('correlation_amplitude',0.),
                length_nm=tr.get('correlation_length_nm',0.),
                free_n_cm3=tr['free_n_cm3'],Nq2_cm3=tr['Nq2_cm3'],
                **diagnostic(tr['free_n_cm3'],tr['Nq2_cm3'],tr['temperature_K'],tr['n_screen_cm3'],g,
                             tr.get('correlation_amplitude',0.),tr.get('correlation_length_nm',0.))))
    c.e.csvwrite(out/'states.csv',rows)
    c.write(out/'audit.json',dict(passed=all(checks),checks=len(checks),kernel_checks=details,
        actual_scenario_model_states=len(rows),chi_ge_one=sum(r['chi_ge_one'] for r in rows),
        chi_min=min(r['linearization_chi'] for r in rows),chi_max=max(r['linearization_chi'] for r in rows),
        new_material_samples=0,not_a_real_material_phase_certificate=True,goal_complete=False))
    assert all(checks),'Preserve initial diagnostic failure before versioned repair'
    print('Actual-state conditional linearization diagnostic:',len(rows),'states;',sum(r['chi_ge_one'] for r in rows),'chi>=1',flush=True)


if __name__=='__main__':main()
