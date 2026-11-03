"""API scope/anchor checks and replay of the frozen artificial source examples."""
from pathlib import Path
import sys,json,subprocess,copy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.kroger_db import NPZ
import phase29_data as data
import phase29_delivery as delivery
import phase2_energy as e

OUT=data.OUT/'api_gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists(),'preserve prior gate before versioned repair'
    parent=data.OUT/'saved_source_gates_v1'
    assert json.loads((parent/'gates.json').read_text())['passed']
    source=json.loads((parent/'packet.json').read_text())
    known={r['id']:r for r in json.loads((parent/'runtime_result.json').read_text())['records']}
    code=[Path(__file__),Path(__file__).with_name('phase29_api_runtime.py'),*delivery.core_files()]
    e.write(OUT/'protocol_before.json',dict(created_utc=e.now(),source_policy_positive_budget=.1,
        log_readout_replay_tolerance=2e-7,contract_refusals_expected=True,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (parent/'packet.json',parent/'runtime_result.json')},
        no_material_label_training=True,no_predictive_accuracy_claim=True,goal_complete=False))
    bundle=OUT/'bundle';presets={};queries=[];expect={}
    for index,task in enumerate(source['tasks']):
        model=bundle/'models'/(str(index)+'.json');e.write(model,task['fit'])
        preset=task['id']
        presets[preset]=dict(model_file='models/'+model.name,model_sha256=e.sha(model),
            variant=preset,context='numerical_test',source_kind=task['kind'],
            source_lambdas={'0':None,'1':.1,'2':.1},serving_qualified=True)
        conditions=[dict(node,condition_id=data.GOGOVA+'::'+('A' if i<8 else 'B'),case_id='numerical::'+str(i))
            for i,node in enumerate(task['nodes'])]
        for b in (0,1,2):
            names=[] if b==0 else ['B'] if b==1 else ['B','A']
            anchors=[dict(condition_id=data.GOGOVA+'::'+name,observed_nH_cm3=10**(17.5 if name=='A' else 17.2)) for name in names]
            qid=preset+'::budget'+str(b)
            queries.append(dict(id=qid,input=dict(preset=preset,conditions=conditions,anchors=anchors)))
            ep=[v for v in known[preset]['episodes'] if v['budget']==b][0]
            sol=[v for v in ep['solutions'] if v['lam']==(None if b==0 else .1)][0]
            expect[qid]=dict(refuse=False,records=sol['records'],valid=sol['valid'],budget=b)
    example=copy.deepcopy(queries[0]['input'])
    tests={}
    bad=copy.deepcopy(example);bad['conditions'][0]['target']=17.;tests['query_response_label']=bad
    bad=copy.deepcopy(example);bad['conditions'][0]['H_known']='False';tests['string_composition_flag']=bad
    bad=copy.deepcopy(example);bad['conditions'][0]['element']='Er';tests['unsupported_element']=bad
    bad=copy.deepcopy(example);bad['conditions'][0]['growth_T_C']=None;tests['missing_process_scenario']=bad
    bad=copy.deepcopy(example);bad['conditions'][1]['case_id']=bad['conditions'][0]['case_id'];tests['duplicate_case']=bad
    bad=copy.deepcopy(example);bad['conditions'][0]['incorporated_cm3']*=2;tests['condition_changes_chemistry']=bad
    one=dict(condition_id=data.GOGOVA+'::B',observed_nH_cm3=10**17.2)
    bad=copy.deepcopy(example);bad['anchors']=[one,one];tests['duplicated_physical_anchor']=bad
    bad=copy.deepcopy(example);bad['anchors']=[dict(condition_id='absent',observed_nH_cm3=1e17)];tests['unknown_anchor']=bad
    bad=copy.deepcopy(example);bad['anchors']=[one,one,one];tests['three_anchors']=bad
    bad=copy.deepcopy(example);bad['anchors']=[one];bad['conditions']=bad['conditions'][:-1];tests['unequal_scenario_multiplicity']=bad
    bad=copy.deepcopy(example);bad['anchors']=[dict(one,observed_nH_cm3=None)];tests['missing_anchor_label']=bad
    bad=copy.deepcopy(example);bad['anchors']=[dict(one,mu_cm2Vs=20.)];tests['mobility_anchor_label']=bad
    presets['unqualified']=dict(presets[queries[0]['input']['preset']],serving_qualified=False)
    bad=copy.deepcopy(example);bad['preset']='unqualified';tests['unqualified_saved_fit']=bad
    for name,value in tests.items():
        qid='refuse::'+name;queries.append(dict(id=qid,input=value));expect[qid]=dict(refuse=True)
    e.write(bundle/'manifest.json',dict(version='phase29_joint_scattering_v1',presets=presets,
        physical_reference_sha256=e.sha(NPZ),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in delivery.core_files()}))
    packet=OUT/'packet.json';target=OUT/'runtime_result.json'
    e.write(packet,dict(bundle=str(bundle),queries=queries))
    with (OUT/'runtime.log').open('w') as log:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('phase29_api_runtime.py')),
            str(packet),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    checks=[];maximum=0.
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    add('isolated_API_exit',result.returncode==0,returncode=result.returncode)
    if target.exists():
        returned=json.loads(target.read_text())
        add('no_query_or_training_table_reads',not returned['blocked_paths'] and
            not returned['query_tables_read'] and not returned['training_tables_read'])
        add('all_requests_retained',len(returned['rows'])==len(queries))
        for row in returned['rows']:
            expected=expect[row['id']]
            if expected['refuse']:
                add('declared_contract_refusal',row['result'] is None and row['error'] is not None,id=row['id']);continue
            add('numerical_source_request_returned',row['result'] is not None and row['error'] is None,id=row['id'])
            if row['result'] is None:continue
            actual=row['result']
            add('physical_anchor_budget_preserved',actual['budget']==expected['budget']
                and len(actual['physical_anchor_conditions'])==expected['budget']
                and actual['anchor_scenario_rows']==8*expected['budget'],id=row['id'])
            add('no_global_refit_or_query_mu',not actual['global_model_retrained'] and not actual['query_mu_used'],id=row['id'])
            for a,b in zip(actual['rows'],expected['records']):
                add('numeric_and_domain_flags_replayed',a['numeric']==b['numeric']
                    and a['valid']==bool(b['valid'] and expected['valid']),id=row['id'],case=a['case_id'])
                if b['numeric']:
                    x=np.array([a['log10_nH_cm3'],a['log10_muH_cm2Vs'],a['log10_sigma_Scm']])
                    y=np.array([b['predicted'],b['predicted_log_mu'],b['predicted_log_sigma']])
                    error=float(np.max(abs(x-y)));maximum=max(maximum,error)
                    add('saved_API_log_readout_replay',error<=2e-7,error=error,id=row['id'],case=a['case_id'])
    value=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        requests=len(queries),expected_contract_refusals=len(tests),maximum_absolute_log_readout_difference=maximum,
        actual_material_label_training=0,actual_material_predictive_validation=0,goal_complete=False)
    e.write(OUT/'gates.json',value)
    print('API29 gates',value['passed'],len(checks),'max',maximum,flush=True)
    if not value['passed']:raise RuntimeError('Preserve API gate failure before versioned repair')


if __name__=='__main__':main()
