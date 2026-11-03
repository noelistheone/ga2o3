"""Rebuild saved transport moments and inspect conditional k(E)ell(E) weights."""
from pathlib import Path
import sys,time,numpy as np,pandas as pd
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log
from sandbox.coherent_fermi import published_coherent_profile

SOURCE=c.BASE/'phase32_rate_controls/prediction'
OUT=c.BASE/'phase32_transport_scope_v1'


def moments(d):
    mass=d['mass_ratio']*M_E;eps=d['epsilon_r']*EPSILON0;T=d['temperature_K'];kT=K_B_J*T
    eta=d['eta'];lo,hi=np.sqrt(d['x_domain'])
    nodes,weights=np.polynomial.legendre.leggauss(d['quadrature_order'])
    t=(nodes+1)*(hi-lo)/2+lo;x=t*t
    P=weights*(hi-lo)/2*2*t**4*expit(eta-x)*expit(x-eta)/float(expit(eta));P/=sum(P)
    qs2=E_CHARGE**2*d['n_screen_cm3']*1e6/(eps*kT)
    b=8*mass*kT*x/(HBAR**2*qs2)
    rate=d['Nq2_cm3']*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)*x**(-1.5)*coulomb_log(b)
    if d['extra_mobility_cm2Vs'] is not None:rate+=E_CHARGE/(mass*d['extra_mobility_cm2Vs']*1e-4)
    tau=1/rate;unit=float(max(tau));z=tau/unit
    m1=float(P@z);m2=float(P@(z*z));rH=m2/m1**2
    drift=E_CHARGE/mass*unit*m1*1e4
    values=np.log10([d['free_n_cm3']/rH,drift*rH,E_CHARGE*d['free_n_cm3']*drift])
    kl=2*kT*x*tau/HBAR
    w1=P*z/m1;w2=P*z*z/m2
    order=np.argsort(kl);cdf=np.cumsum(w1[order])
    def quantile(q):return float(kl[order[min(int(np.searchsorted(cdf,q)),len(order)-1)]])
    return values,dict(conductivity_fraction_kl_lt1=float(sum(w1[kl<1.])),
        Hall_second_moment_fraction_kl_lt1=float(sum(w2[kl<1.])),
        conductivity_weighted_kl=float(w1@kl),kl_q05=quantile(.05),kl_q50=quantile(.5),kl_q95=quantile(.95),
        mean_tau_s=float(unit*m1),minimum_tau_s=float(min(tau)),maximum_tau_s=float(max(tau)),
        reconstructed_Hall_factor=rH,eta=eta,extra_mu_cm2Vs=d['extra_mobility_cm2Vs'],
        free_n_cm3=d['free_n_cm3'],Nq2_cm3=d['Nq2_cm3'])


def main():
    assert not (OUT/'protocol_before.json').exists()
    assert c.document(SOURCE/'independent_result_audit.json')['passed']
    files=[SOURCE/'predictions.csv',SOURCE/'prediction_diagnostics.json',SOURCE/'independent_result_audit.json',
        c.OLD/'data/scenarios49.csv',c.ROOT/'docs/ML_SANDBOX_TRANSPORT_SCOPE_20261020.md']
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_sha256=c.e.sha(__file__),
        input_hashes=c.file_hashes(files),criterion='k(E)ell(E)=2E tau(E)/hbar;transport-moment-weighted fractions',
        unity_threshold_is_order_of_magnitude_not_universal_material_limit=True,
        no_refit_or_data_exclusion=True,no_material_phase_or_microscopic_causal_claim=True,goal_complete=False))
    diagnostics={r['row_id']:r for r in c.document(SOURCE/'prediction_diagnostics.json')}
    table=c.read(SOURCE/'predictions.csv');rows=[];cache={};errors=[];start=time.monotonic()
    for row in table.itertuples():
        d=diagnostics[row.row_id]['diagnostics']['coherent_transport']
        key=c.digest(d)
        if key not in cache:cache[key]=moments(d)
        values,stats=cache[key]
        err=float(max(abs(values-np.array([row.predicted,row.predicted_log_mu,row.predicted_log_sigma]))));errors.append(err)
        rows.append(dict(row_id=row.row_id,variant=row.variant,budget=int(row.budget),case_id=row.case_id,
            sample_id=row.sample_id,family=row.family,related_cluster=row.related_cluster,role=row.role,
            primary_common20=bool(row.primary_common20),mobility_eligible=bool(row.mobility_eligible),
            observed_log_mu=float(row.observed_log_mu) if row.mobility_eligible else None,
            predicted_log_mu=float(row.predicted_log_mu),property_reconstruction_error=err,**stats))
    c.e.csvwrite(OUT/'conditional_scope_rows.csv',rows)
    frame=pd.DataFrame(rows);summary=[]
    q=frame[frame.role.eq('query')&frame.primary_common20.eq(True)]
    for (variant,budget,family),part in q.groupby(['variant','budget','family']):
        cond=part.groupby('sample_id').agg(f1=('conductivity_fraction_kl_lt1','mean'),
            f2=('Hall_second_moment_fraction_kl_lt1','mean'),median=('kl_q50','mean'))
        summary.append(dict(variant=variant,budget=int(budget),family=family,conditions=len(cond),
            mean_moment1_fraction_lt1=float(cond.f1.mean()),mean_moment2_fraction_lt1=float(cond.f2.mean()),
            condition_average_median_kl=float(cond['median'].mean()),
            conditions_with_more_than_half_m1_lt1=int(sum(cond.f1>.5))))
    c.e.csvwrite(OUT/'common20_per_family.csv',summary)
    measured=[];profile=published_coherent_profile();mass=profile.electron_mass_ratio*M_E
    for row in c.cases().drop_duplicates('sample_id').query('mobility_eligible == True').itertuples():
        nh=10.**row.target;kf=(3*np.pi**2*nh*1e6)**(1/3);mu=row.mu_cm2Vs*1e-4
        measured.append(dict(sample_id=row.sample_id,family=row.family,element=row.element,nH_cm3=nh,muH_cm2Vs=row.mu_cm2Vs,
            uniform_rH1_kFell=HBAR*kf*kf*mu/E_CHARGE,
            uniform_rH1_kTell=2*K_B_J*row.Tobs_K*mass*mu/(E_CHARGE*HBAR),
            uniform_homogeneous_interpretation_only_not_material_truth=True))
    c.e.csvwrite(OUT/'reported_Hall_conditional_scales.csv',measured)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'audit.json',dict(created_utc=c.e.now(),passed=max(errors)<1e-9,
        prediction_rows=len(rows),unique_saved_transport_states=len(cache),
        maximum_property_reconstruction_error=max(errors),observed_conditions=len(measured),seconds=time.monotonic()-start,
        original_predictions_unchanged=True,no_material_falsification_or_prediction_gain_claim=True,goal_complete=False))
    print('State-rate conditional transport scope',len(rows),'rows',len(cache),'states;max reconstruction error',max(errors),flush=True)
    assert max(errors)<1e-9,'Preserve first scope reconstruction failure'


if __name__=='__main__':main()
