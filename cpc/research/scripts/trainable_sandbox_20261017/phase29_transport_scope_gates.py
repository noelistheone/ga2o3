"""Check independent transport reconstruction before auditing learned states."""
from pathlib import Path
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.transport_scope_diagnostic import reconstruct_transport
from sandbox.ionized_transport import coulomb_log
import phase29_data as data
import phase2_energy as e

OUT=data.OUT/'transport_scope_gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists(),'preserve prior gate before versioned repair'
    previous=data.OUT/'numerical_gates_v2/numerical_states.json'
    fields=[Path(__file__),ROOT/'src/sandbox/transport_scope_diagnostic.py',
        ROOT/'docs/ML_SANDBOX_TRANSPORT_SCOPE_PHASE29_20261019.md']
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),reconstruction_tolerance=1e-6,
        split_fraction_refinement_tolerance=1e-6,code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in fields},
        input_hashes={str(previous.relative_to(ROOT)):e.sha(previous)},
        no_new_training=True,no_material_accuracy_or_Born_claim=True,goal_complete=False))
    profile=published_coherent_profile();states=json.loads(previous.read_text());checks=[];outputs=[];maximum=0.
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    for state in states:
        if state['method']!='ReferenceTau':continue
        dg=state['diagnostics'];tr=dg['coherent_transport']
        kwargs=dict(Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
            mass_ratio=tr['mass_ratio'],epsilon_r=tr['epsilon_r'])
        a=reconstruct_transport(tr['free_n_cm3'],tr['Nq2_cm3'],tr['temperature_K'],
            tr['extra_mobility_cm2Vs'],order=512,**kwargs)
        b=reconstruct_transport(tr['free_n_cm3'],tr['Nq2_cm3'],tr['temperature_K'],
            tr['extra_mobility_cm2Vs'],order=1024,**kwargs)
        expected=np.array([state['values'][0],*state['values'][-2:]],float)
        reconstructed=np.asarray(a['values'])
        if state['source']=='offset':reconstructed+=np.array([dg['source_shift'],0.,dg['source_shift']])
        error=float(np.max(abs(reconstructed-expected)));maximum=max(maximum,error)
        add('independent_positive_moments_reproduce_native_readout',error<=1e-6,error=error,case=state['case_id'],source=state['source'])
        for field in ('fraction_sigma_kl_lt1','fraction_Hall_kl_lt1'):
            add('fraction_in_unit_interval',-1e-12<=a[field]<=1+1e-12,field=field,case=state['case_id'])
            add('threshold_split_quadrature_refines',abs(a[field]-b[field])<=1e-6,
                error=abs(a[field]-b[field]),field=field,case=state['case_id'])
        add('nonnegative_variance_Hall_bound',a['hall_factor']>=1-1e-12,case=state['case_id'])
        outputs.append(dict(case=state['case_id'],source=state['source'],order512=a,order1024=b))
    for b in np.logspace(-10,10,41):
        D=(b/(1+b))**2/coulomb_log(b)
        add('logarithmic_rate_bound_matches_proof',0<D<=2+1e-10,b=float(b),D=float(D))
    add('real_numerical_states_present',len(outputs)==9)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),
        rows=checks,numerical_states=len(outputs),maximum_absolute_reconstruction_error=maximum,
        no_new_training_or_predictive_validation=True,goal_complete=False)
    e.write(OUT/'reconstruction_states.json',outputs);e.write(OUT/'gates.json',result)
    print('Scope29 gates',result['passed'],len(checks),'max',maximum,flush=True)
    if not result['passed']:raise RuntimeError('Preserve scope gate before versioned repair')


if __name__=='__main__':main()
