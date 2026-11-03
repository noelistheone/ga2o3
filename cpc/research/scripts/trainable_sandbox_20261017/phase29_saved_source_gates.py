"""Saved readout, source algebra and isolated artificial-anchor smoke checks."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,subprocess,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.joint_saved_model import SavedJointModel
from sandbox.context_saved_model import INPUT_NAMES
from sandbox.kroger_db import NPZ
from sandbox.ionized_transport import E_CHARGE
import phase29_learning as l
import phase29_data as data
import phase29_gates as g
import phase2_energy as e

OUT=data.OUT/'saved_source_gates_v1'


def vector(p):return np.array([p.log_nH,p.log_muH,p.log_sigma])


def main():
    assert not (OUT/'protocol_before.json').exists()
    files=[Path(__file__),Path(__file__).with_name('phase29_source_runtime.py'),
        ROOT/'src/sandbox/joint_saved_model.py',ROOT/'src/sandbox/context_source.py',
        ROOT/'src/sandbox/functional_source.py']
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files},
        artificial_anchor_labels={'A':17.5,'B':17.2},
        lambdas=[None,0.,.01,.1,1.,10.],finite_difference_step=.001,replay_tolerance=2e-10,
        no_actual_material_label_training=True,no_outer_validation=True,
        data_sha=e.sha(data.OUT/'data/scenarios49.csv'),goal_complete=False))
    checks=[];clock=time.perf_counter()
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    def close(name,a,b,tol=2e-10,**kw):
        error=float(np.max(abs(np.asarray(a)-np.asarray(b))/(1+abs(np.asarray(b)))))
        add(name,error<=tol,error=error,tolerance=tol,**kw)
    cases=data.load_cases()
    synthetic=cases.copy();synthetic['target']=np.log10(synthetic.incorporated_cm3)-.37;synthetic['mu_cm2Vs']=23.
    fits={}
    for method in l.METHODS:
        f=dict(key='numerical_only_'+method,method=method,training_valid=True,selected_success=True,
            parameters=g.parameters(method).tolist() if method in l.NONLINEAR else None,
            artificial_numerical_parameters_not_actual_trained_model=True)
        if method=='DirectMultiRidge':f.update(l.fit_direct(synthetic,.1))
        elif method in l.EMPIRICAL:f.update(l.fit_empirical(method,synthetic))
        fits[method]=f
    selected=[cases[cases.H_known.eq(True)].iloc[0],
        cases[cases.family.eq(data.GOGOVA)&cases.sample_label.eq('A')].iloc[2]]
    for method,f in fits.items():
        mapper=SavedJointModel(f)
        for row in selected:
            r=SimpleNamespace(**{k:getattr(row,k) for k in INPUT_NAMES})
            p=mapper.point(r,require_observation=False)
            if method in l.NONLINEAR:expected=l.point_tuple(l.model(method).point(r,f['parameters'],require_observation=False))[0]
            elif method=='DirectMultiRidge':expected=l.direct_values(f,cases.loc[[row.name]])[0]
            else:expected=l.empirical_values(method,f,cases.loc[[row.name]])[0]
            close('saved_zero_matches_actual_map',vector(p),expected,method=method)
            for head in ('native','native_offset','CT_offset'):
                use=SavedJointModel(dict(f,transport_head=head,transport_head_offset_dex=.2 if head!='native' else 0.))
                base=use.point(r,require_observation=False)
                fast=use.offset_point(r,base,.1,require_observation=False)
                slow=use.point(r,source_kind='offset',shift=.1,require_observation=False)
                close('fast_offset_preserves_declared_readout',vector(fast),vector(slow),method=method,head=head)
                plus=use.point(r,source_kind='offset',shift=.101,require_observation=False)
                minus=use.point(r,source_kind='offset',shift=.099,require_observation=False)
                close('source_transport_derivative',(vector(plus)-vector(minus))/.002,
                    np.r_[slow.source_derivative,slow.source_transport_derivative],3e-6,method=method,head=head)
                close('derived_sigma_only',fast.log_sigma,np.log10(E_CHARGE)+fast.log_nH+fast.log_muH,1e-12,method=method)
            if method=='DirectMultiRidge':
                q=mapper.point(r,source_kind='offset',shift=.1,require_observation=False)
                close('direct_source_update_leaves_rho_head_fixed',q.log_sigma,p.log_sigma,1e-13)
    for field in ('training_valid','selected_success'):
        try:SavedJointModel(dict(fits['JointProcess'],**{field:False}));refused=False
        except ValueError:refused=True
        add('unqualified_fit_refused',refused,field=field)
    new=cases[cases.family.eq(data.GOGOVA)].copy()
    nodes=[{k:e.clean(getattr(row,k)) for k in INPUT_NAMES} for row in new.itertuples()]
    counts=new.groupby('sample_id').size()
    add('source_scenarios_equal_count_per_condition',set(counts)=={8})
    episodes=[]
    for b in (0,1,2):
        names=[] if b==0 else [data.GOGOVA+'::B'] if b==1 else [data.GOGOVA+'::B',data.GOGOVA+'::A']
        ai=[i for i,r in enumerate(new.itertuples()) if r.sample_id in names]
        labels=[17.5 if new.iloc[i].sample_label=='A' else 17.2 for i in ai]
        episodes.append(dict(budget=b,anchor_condition_ids=names,anchor_indices=ai,anchor_labels=labels))
        add('anchor_budget_counts_conditions_not_scenarios',len(names)==b and len(ai)==8*b,budget=b)
    tasks=[]
    for method,kind in (('JointProcess','internal'),('ReferenceTau','internal'),
        ('Reference205','offset'),('DirectMultiRidge','offset'),('Reference205','affine'),('UnitDose','affine')):
        tasks.append(dict(id=method+'_'+kind,fit=fits[method],kind=kind,nodes=nodes,episodes=episodes))
    packet=dict(tasks=tasks,lambdas=[None,0.,.01,.1,1.,10.],
        physical_reference_sha256=e.sha(NPZ),artificial_anchor_test_only=True)
    taskpath=OUT/'packet.json';destination=OUT/'runtime_result.json'
    e.write(taskpath,packet)
    with (OUT/'runtime.log').open('w') as log:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('phase29_source_runtime.py')),
            str(taskpath),str(destination)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    add('isolated_runtime_exit',result.returncode==0,returncode=result.returncode)
    if destination.exists():
        payload=json.loads(destination.read_text())
        add('no_runtime_table_or_query_label_reads',not payload['denied_reads'] and not payload['runtime_query_labels'])
        add('all_declared_source_tasks_returned',len(payload['records'])==6)
        for record in payload['records']:
            add('source_episode_count',len(record['episodes'])==3,task=record['id'])
            for ep in record['episodes']:
                add('runtime_physical_anchor_count',len(ep['anchor_condition_ids'])==ep['budget'],task=record['id'],budget=ep['budget'])
                add('all_source_candidates_retained',len(ep['solutions'])==(1 if ep['budget']==0 else 6),task=record['id'],budget=ep['budget'])
                zero=[s for s in ep['solutions'] if s['lam'] is None]
                add('no_update_candidate_valid',len(zero)==1 and zero[0]['valid'],task=record['id'],budget=ep['budget'])
                for sol in ep['solutions']:
                    add('all_scenario_records_retained',len(sol['records'])==16,task=record['id'],budget=ep['budget'],lam=sol['lam'])
    e.write(OUT/'gates.json',dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),
        checks=len(checks),rows=checks,seconds=time.perf_counter()-clock,
        no_actual_material_label_training=True,no_actual_predictive_accuracy_validation=True,goal_complete=False))
    print('Saved/source29 gates',all(c['passed'] for c in checks),len(checks),flush=True)
    if not all(c['passed'] for c in checks):raise RuntimeError('Preserve saved/source gate failure before versioned repair')


if __name__=='__main__':main()
