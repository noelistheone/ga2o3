"""Conditional fixed-state neutral-scattering envelope; no optimization."""
from pathlib import Path
import sys,time,numpy as np
from scipy.special import expit,eval_legendre
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.coherent_transport import fermi_transport
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log

OUT=c.BASE/'neutral_unitarity_reachability_v1'
SOURCE=c.BASE/'native_neutral_state_probe_v1/native_and_foreign_neutral_states.csv'


def angular_checks():
    nodes,weights=np.polynomial.legendre.leggauss(128)
    rng=np.random.default_rng(20261021);checks=[];errors=[]
    for L in [0,1,2,3]:
        for i in range(10):
            d=np.r_[rng.uniform(-np.pi,np.pi,L+1),0.]
            partial=sum((l+1)*np.sin(d[l+1]-d[l])**2 for l in range(L+1))
            amp=sum((2*l+1)*np.exp(1j*d[l])*np.sin(d[l])*eval_legendre(l,nodes) for l in range(L+1))
            direct=2*np.pi*float(weights@((1-nodes)*abs(amp)**2))
            expected=4*np.pi*partial
            error=abs(direct-expected)/(1+expected);errors.append(error)
            checks.extend([error<1e-11,partial<=(L+1)*(L+2)/2+1e-12])
    return checks,errors


def main():
    assert not (OUT/'protocol_before.json').exists()
    files=[SOURCE,c.OLD/'prediction/plan.json',
           c.ROOT/'docs/ML_SANDBOX_NEUTRAL_SCATTERING_REACHABILITY_20261020.md']
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_sha256=c.e.sha(__file__),
        input_hashes=c.file_hashes(files),phase_truncations=[0,1,2,3],expected_fixed_states=50,
        no_new_fit=True,labels_only_for_conditional_compatibility_flags=True,
        scope='independent central neutral scattering with finite declared partialwavecutoff;not extendeddefects or arbitraryL',
        no_new_material_or_predictive_accuracy_claim=True,goal_complete=False))
    checks,errors=angular_checks();p=published_coherent_profile();mass=p.electron_mass_ratio*M_E
    eps=10.2*EPSILON0;T=300.;kT=K_B_J*T
    plans=c.document(c.OLD/'prediction/plan.json')['plans']
    plans={(v['variant'],v['outer_held']):v for v in plans if v['outer_held'] is not None}
    cases=c.read(SOURCE);nodes,weights=np.polynomial.legendre.leggauss(512)
    rows=[];reconstruction=[];clock=time.perf_counter()
    for r in cases.itertuples():
        fit=plans[(r.variant,r.related_cluster)]['minimal_fit']
        mu0=205. if fit['method']=='Reference205' else 205.*10.**float(fit['parameters'][-1])
        tr=fermi_transport(np.log10(r.free_n_cm3),np.log10(r.Nq2_cm3),T,
            Nc_ref_cm3=p.Nc_ref,T_ref_K=p.T_ref,mass_ratio=p.electron_mass_ratio,
            extra_mobility_cm2Vs=mu0,hall=True)
        eta=tr.diagnostics['eta'];lo=np.sqrt(max(0.,eta-55.));hi=np.sqrt(max(65.,eta+55.))
        t=(nodes+1)*(hi-lo)/2+lo;x=t*t
        W=weights*2*t**4*expit(eta-x)*expit(x-eta);W/=W.sum()
        qs2=E_CHARGE**2*tr.diagnostics['n_screen_cm3']*1e6/(eps*kT)
        b=8*mass*kT*x/(HBAR**2*qs2)
        ion=r.Nq2_cm3*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)*x**(-1.5)*coulomb_log(b)
        base_rate=ion+E_CHARGE/(mass*mu0*1e-4);tau0=1/base_rate
        t10=float(W@tau0);t20=float(W@(tau0*tau0));cm=E_CHARGE/mass*1e4
        nh0=r.free_n_cm3*t10*t10/t20;mh0=cm*t20/t10
        err=max(abs(np.log10(nh0)-r.predicted_log_nH),abs(np.log10(mh0)-r.predicted_log_mu))
        reconstruction.append(err);checks.append(err<1e-9)
        k=np.sqrt(2*mass*kT*x)/HBAR;v=HBAR*k/mass
        s_wave=r.neutral_all_cm3*1e6*v*4*np.pi/(k*k)
        sigma_example=4*np.pi*(.5e-9)**2/(1+(k*.5e-9)**2)
        rate_example=r.neutral_all_cm3*1e6*v*sigma_example
        tau_example=1/(base_rate+rate_example)
        e1=float(W@tau_example);e2=float(W@(tau_example*tau_example))
        example=(r.free_n_cm3*e1*e1/e2,cm*e2/e1,E_CHARGE*r.free_n_cm3*cm*e1)
        for L in [0,1,2,3]:
            C=(L+1)*(L+2)/2;minus=1/(base_rate+C*s_wave)
            t1m=float(W@minus);t2m=float(W@(minus*minus))
            lower_mu=max(cm*t1m,cm*t2m/t10);upper_mu=cm*t20/t1m
            lower_n=r.free_n_cm3*t1m*t1m/t20
            upper_n=min(r.free_n_cm3,r.free_n_cm3*t10*t10/t2m)
            lower_sigma=E_CHARGE*r.free_n_cm3*cm*t1m
            upper_sigma=E_CHARGE*r.free_n_cm3*cm*t10
            intervals=[(lower_n,upper_n),(lower_mu,upper_mu),(lower_sigma,upper_sigma)]
            for value,(loi,hii) in zip(example,intervals):
                checks.append(loi*(1-5e-12)<=value<=hii*(1+5e-12))
            checks.extend([lower_mu<=mh0*(1+5e-12),upper_mu>=mh0*(1-5e-12),
                           lower_n<=nh0*(1+5e-12),upper_n>=nh0*(1-5e-12)])
            rows.append(dict(variant=r.variant,case_id=r.case_id,sample_id=r.sample_id,
                family=r.family,element=r.element,partialwave_L=L,channel_bound=C,
                neutral_all_cm3=r.neutral_all_cm3,native_neutral_cm3=r.neutral_native_cm3,
                base_extra_mu_cm2Vs=mu0,base_nH_cm3=nh0,base_muH_cm2Vs=mh0,
                nH_lower=lower_n,nH_upper=upper_n,muH_lower=lower_mu,muH_upper=upper_mu,
                sigma_lower=lower_sigma,sigma_upper=upper_sigma,
                maximum_conductivity_fractional_reduction=1-lower_sigma/upper_sigma,
                Hall_mu_outer_fractional_span=upper_mu/lower_mu-1,
                observed_nH_cm3=10.**r.observed_log_nH,observed_mu_cm2Vs=r.observed_mu_cm2Vs,
                asreported_nH_in_envelope=bool(lower_n<=10.**r.observed_log_nH<=upper_n),
                asreported_mu_in_envelope=bool(lower_mu<=r.observed_mu_cm2Vs<=upper_mu),
                fixed_model_state_not_measured_microstate=True,
                partialwave_truncation_not_experimentally_established=True))
    assert len(rows)==200
    c.e.csvwrite(OUT/'fixed_state_envelopes200.csv',rows)
    summary=[]
    for variant in sorted(cases.variant.unique()):
        for L in [0,1,2,3]:
            q=[r for r in rows if r['variant']==variant and r['partialwave_L']==L]
            summary.append(dict(variant=variant,partialwave_L=L,scenario_states=len(q),
                max_conductivity_fractional_reduction=max(r['maximum_conductivity_fractional_reduction'] for r in q),
                max_Hall_mu_outer_fractional_span=max(r['Hall_mu_outer_fractional_span'] for r in q),
                nH_scenario_compatibilities=sum(r['asreported_nH_in_envelope'] for r in q),
                mu_scenario_compatibilities=sum(r['asreported_mu_in_envelope'] for r in q)))
    c.e.csvwrite(OUT/'summary.csv',summary)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(passed=all(checks),created_utc=c.e.now(),checks=len(checks),
        actual_frozen_states=50,envelopes=200,max_angular_formula_error=max(errors),
        max_old_property_reconstruction_error=max(reconstruction),seconds=time.perf_counter()-clock,
        no_parameters_reoptimized=True,conditional_necessary_bounds_not_material_falsification=True,
        feasible_box_not_guaranteed_jointly_attainable=True,goal_complete=False))
    print('Neutral-unitarity conditional envelope',len(checks),'checks;failed',sum(not v for v in checks),flush=True)
    assert all(checks),'Preserve first envelope failure before versioned repair'
    for r in summary:print(r,flush=True)


if __name__=='__main__':main()
