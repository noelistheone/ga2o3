"""Replay all actual held-source predictions through fresh saved API workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,subprocess,argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from sandbox.context_saved_model import INPUT_NAMES
from sandbox.transport import mu_caughey_thomas
from sandbox.ionized_transport import E_CHARGE
import phase29_data as data
import phase29_delivery as delivery
import phase2_energy as e

OUT=data.OUT/'api_replay'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    assert not (OUT/'protocol_before_replay.json').exists(),'inspect existing replay processes before versioned resume'
    assert json.loads((data.OUT/'prediction/audit.json').read_text())['passed']
    assert json.loads((data.OUT/'delivery/audit.json').read_text())['passed']
    bundle=data.OUT/'delivery/predictor_bundle';manifest=json.loads((bundle/'manifest.json').read_text())
    files=[Path(__file__),Path(__file__).with_name('phase29_api_runtime.py'),*delivery.core_files()]
    e.write(OUT/'protocol_before_replay.json',dict(created_utc=e.now(),log_tolerance=2e-7,
        all_48_variants_and_5_exclusions_and_full=True,no_retraining=True,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (
            bundle/'manifest.json',data.OUT/'prediction/predictions.csv',
            data.OUT/'data/scenarios49.csv',data.OUT/'data/episodes35.csv')},goal_complete=False))
    cases=data.load_cases();roles=data.load_roles();raw=read(data.OUT/'prediction/predictions.csv')
    queries=[];expected={};checks=[];maximum=0.
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    for name,spec in manifest['presets'].items():
        table=cases if spec['context']=='full' else cases[cases.related_cluster.eq(spec['context'].removeprefix('without:'))]
        if spec['context']=='full':
            splits=[('full_training_inputs',table,0)]
        else:
            splits=[(family,part,b) for family,part in table.groupby('family',sort=True) for b in (0,1,2)]
        for family,part,budget in splits:
            conditions=[dict({k:e.clean(getattr(row,k)) for k in INPUT_NAMES},
                condition_id=row.sample_id,case_id=row.case_id) for row in part.itertuples()]
            selected=roles[roles.budget.eq(budget)&roles.role.eq('anchor')&
                roles.sample_id.isin(part.sample_id)] if spec['context']!='full' else roles.iloc[:0]
            names=sorted(selected.sample_id.unique())
            anchors=[dict(condition_id=s,observed_nH_cm3=float(10**part[part.sample_id.eq(s)].iloc[0].target)) for s in names]
            assert len(anchors)==budget
            qid=name+'::'+family+'::'+str(budget)
            queries.append(dict(id=qid,input=dict(preset=name,conditions=conditions,anchors=anchors)))
            if spec['context']!='full':
                truth=raw[raw.variant.eq(spec['variant'])&raw.budget.eq(budget)&raw.family.eq(family)].set_index('case_id')
                values=[dict(case_id=r.case_id,numeric=bool(truth.loc[r.case_id,'numeric']),
                    valid=bool(truth.loc[r.case_id,'valid']),
                    log_values=[truth.loc[r.case_id,k] for k in ('predicted','predicted_log_mu','predicted_log_sigma')])
                    for r in part.itertuples()]
            else:
                fit=json.loads((ROOT/spec['base_fit_path']).read_text())
                model=json.loads((bundle/spec['model_file']).read_text())
                good=bool(fit['training_valid'] and fit['selected_success'])
                base=np.asarray(fit.get('selected_training_values',[]),float)
                values=[]
                for index,row in enumerate(part.itertuples()):
                    if not good:values.append(dict(case_id=row.case_id,numeric=False,valid=False,log_values=None));continue
                    v=base[index].copy()
                    if model.get('transport_head')=='native_offset':v[1:]+=model['transport_head_offset_dex']
                    elif model.get('transport_head')=='CT_offset':
                        v[1]=np.log10(mu_caughey_thomas(10**v[0],300.))+model['transport_head_offset_dex']
                        v[2]=np.log10(E_CHARGE)+v[0]+v[1]
                    values.append(dict(case_id=row.case_id,numeric=True,valid=True,log_values=v.tolist()))
            expected[qid]=dict(rows=values,qualified=spec['serving_qualified'],budget=budget,
                context=spec['context'],variant=spec['variant'])
    assert len(queries)==1056
    batches=[[] for _ in range(args.workers)];loads=np.zeros(args.workers)
    for q in queries:
        j=int(np.argmin(loads));batches[j].append(q)
        kind=manifest['presets'][q['input']['preset']]['source_kind']
        loads[j]+=len(q['input']['conditions'])*(100 if kind=='internal' and q['input']['anchors'] else 2)
    def worker(index):
        packet=OUT/('packet'+str(index)+'.json');target=OUT/('worker'+str(index)+'.json')
        e.write(packet,dict(bundle=str(bundle),queries=batches[index]))
        with (OUT/('worker'+str(index)+'.log')).open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('phase29_api_runtime.py')),
                str(packet),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            e.write(OUT/('worker'+str(index)+'_started.json'),dict(created_utc=e.now(),pid=proc.pid,requests=len(batches[index])))
            code=proc.wait()
        if code:raise RuntimeError('Preserve failed API worker '+str(index))
        return json.loads(target.read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:returned=list(pool.map(worker,range(args.workers)))
    rows=[r for item in returned for r in item['rows']]
    add('all_requests_returned',len(rows)==1056 and len({r['id'] for r in rows})==1056)
    for item in returned:
        add('fresh_workers_never_read_property_tables',not item['blocked_paths'] and not item['query_tables_read'] and not item['training_tables_read'])
    output_points=0;refusals=0
    for row in rows:
        ex=expected[row['id']];got=row['result']
        if got is None:
            refusals+=1
            add('API_refusal_preserves_original_qualification',not any(x['numeric'] for x in ex['rows'])
                and row['error'] is not None,id=row['id'])
            continue
        add('qualified_fit_is_required_for_serving',ex['qualified'],id=row['id'])
        add('same_physical_anchor_budget',got['budget']==ex['budget'] and len(got['physical_anchor_conditions'])==ex['budget'],id=row['id'])
        add('no_refit_or_query_labels',not got['global_model_retrained'] and not got['query_mu_used'] and not got['query_labels_read'],id=row['id'])
        add('every_input_case_retained',len(got['rows'])==len(ex['rows']),id=row['id'])
        for actual,reference in zip(got['rows'],ex['rows']):
            output_points+=1
            add('same_case_and_numeric_domain_flags',actual['case_id']==reference['case_id']
                and actual['numeric']==reference['numeric'] and actual['valid']==reference['valid'],id=row['id'],case=reference['case_id'])
            if actual['numeric'] and reference['numeric']:
                a=np.array([actual[k] for k in ('log10_nH_cm3','log10_muH_cm2Vs','log10_sigma_Scm')])
                b=np.asarray(reference['log_values'],float);error=float(np.max(abs(a-b)));maximum=max(maximum,error)
                add('actual_trained_model_and_source_policy_replay',error<=2e-7,error=error,id=row['id'],case=reference['case_id'])
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        requests=len(rows),presets=288,heldsource_expected_points=7056,full_training_input_expected_points=2352,
        actual_returned_points=output_points,request_refusals=refusals,maximum_absolute_log_difference=maximum,
        no_model_or_score_changes=True,not_independent_material_validation=True,goal_complete=False)
    e.write(OUT/'audit.json',result)
    print('API29 actual replay',result['passed'],len(rows),'max',maximum,'refusals',refusals,flush=True)
    if not result['passed']:raise RuntimeError('Preserve replay failure before versioned repair')


if __name__=='__main__':main()
