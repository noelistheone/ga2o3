"""Conditional zero-anchor transfer to two same-row Sn transport observations.

No new training: this checks the current reference/learned predictor before
declaring any additional joint-learning experiment. Source temperatures and
oxygen inputs are sampled scenarios, not identified preparation conditions.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import argparse
import json
import subprocess
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

ROOT=e.ROOT
BASE=e.BASE/'phase28_paired_transport'
OUT=BASE/'frozen_transport_v1'
BUNDLE=e.BASE/'phase27_charge_feasible/delivery/predictor_bundle'
CONTEXTS=('cluster::without:10.1007/s10853-015-9693-6','cluster::full')
TEMPERATURES=(775.,800.,825.,850.)
OXYGEN={'old_effective':1e-5,'higher_effective_activity':1e-2}
SOURCE_DOI='10.1039/C5CE01106J'
RELATED_OLD_DOI='10.1007/s10853-015-9693-6'
E_CHARGE=1.602176634e-19


def declared():
    protocol=json.loads((OUT/'protocol_before_transfer.json').read_text())
    for kind in ('input_hashes','code_hashes'):
        for path,digest in protocol[kind].items():
            assert e.sha(ROOT/path)==digest,path
    return protocol


def declare():
    qual=json.loads((BASE/'source_qualification_v1/summary.json').read_text())
    assert qual['conditional_frozen_transfer_candidates']==['Gogova2015::A','Gogova2015::B']
    assert json.loads((BASE/'source_qualification_v1/audit.json').read_text())['passed']
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    for context in CONTEXTS:
        specs=[s for s in manifest['presets'].values() if s['context']==context]
        assert len(specs)==48
        assert all(SOURCE_DOI.lower() not in [x.lower() for x in s['train_families']] for s in specs)
        if context==CONTEXTS[0]:assert all(RELATED_OLD_DOI not in s['train_families'] for s in specs)
    inputs=[BASE/'source_observations_v1.json',BASE/'source_qualification_v1/protocol_before_calculation.json',
        BASE/'source_qualification_v1/condition_audit.csv',BASE/'source_qualification_v1/summary.json',
        BASE/'source_qualification_v1/audit.json',BUNDLE/'manifest.json',
        e.BASE/'phase25_shared_process/data/conditions33.csv',e.BASE/'artifact_manifest_phase27.json']
    code=[Path(__file__),Path(__file__).with_name('phase27_api_runtime_v2.py'),
        ROOT/'src/sandbox/feasible_property_predictor.py',ROOT/'src/sandbox/feasible_property_predictor_v2.py',
        ROOT/'src/sandbox/pool_property_predictor.py',ROOT/'src/sandbox/context_saved_model.py',
        ROOT/'src/sandbox/charge_feasible_saved_model.py']
    value=dict(created_utc=e.now(),stage='conditional frozen full-property transfer',
        primary_goal='ML+Sandbox dopant-property prediction;nH remains primary',
        primary_context=CONTEXTS[0],comparison_context=CONTEXTS[1],
        contexts=CONTEXTS,variants=48,requests=96,point_rows=1536,
        source_conditions=['Gogova2015::A','Gogova2015::B'],
        distinct_observed_conditions=2,DOIs=1,new_independent_laboratory_sources=0,
        related_source_exclusion='IKZ oldBaldini cohort excluded in primary context;not proof of no sample reuse',
        growth_T_scenarios_C=TEMPERATURES,oxygen_scenarios=OXYGEN,
        oxygen_interpretation='two declared effective oxygen-activity settings of the existing model;neither measured film pO2 nor a proven bracket;water chemistry cannot be converted from total pressure alone',
        process_interpretation='temperature values are the published series settings,individual A/B mapping absent;do not select a temperature from heldout labels',
        H_C_interpretation='API unknown-zero convention;quantitative H/C missing,qualitative C-H present;not carbon-free or complete-composition state',
        observation_temperature='source says RT;300K comparison convention,not an exactly reported temperature',
        chemical_input='central printed Table1 concentration;individual direct-versus-calibrated SIMS pathway and measurement uncertainty not established',
        Hall_anchor_budget=0,new_shared_fits=0,new_source_fits=0,
        primary_endpoint='absolute log10 nH error at each declared scenario;all scenarios retained',
        other_endpoints='native muH and uniform CT(predictednH) readout;conductivity derived from nH/mu,not independent',
        scenario_aggregation='equal mean over4 listedT settings separately for each oxygen setting;diagnostic average,not an uncertainty distribution or robust interval',
        coverage='no score/range for an incomplete group;all48 methods,refusals and invalid continuations retained',
        observed_labels_passed_to_runtime=False,query_and_training_table_reads_forbidden=True,
        no_new_DFT_MD=True,no_new_material_measurements=True,GPU_used=False,not_blind=True,goal_complete=False,
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs},
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code})
    path=OUT/'protocol_before_transfer.json'
    assert not path.exists(),path
    e.write(path,value)
    return value


def worker(i):
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('phase27_api_runtime_v2.py')),
        str(OUT/f'tasks_{i}.json'),str(OUT/f'outputs_{i}.json')],cwd=ROOT,capture_output=True,text=True)
    (OUT/f'worker_{i}.log').write_text(result.stdout+'\n'+result.stderr)
    result.check_returncode()
    return json.loads((OUT/f'outputs_{i}.json').read_text())


def run():
    protocol=declared()
    assert not (OUT/'execution_started.json').exists(),'inspect existing job/output,never blindly restart'
    data=json.loads((BASE/'source_observations_v1.json').read_text())
    observed={r['condition_id']:r for r in data['conditions'] if r['condition_id'] in protocol['source_conditions']}
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    nodes=[];cases=[]
    for oxygen,T,cid in product(OXYGEN,TEMPERATURES,protocol['source_conditions']):
        raw=observed[cid]['reported']
        nodes.append(dict(element='Sn',incorporated_cm3=float(raw['Sn_cm3']),Tobs_K=300.,growth_T_C=T,
            H_known=False,H_cm3=0.,carbon_known=False,carbon_cm3=0.,scenario_pO2_atm=OXYGEN[oxygen],
            oxygen_input_kind=oxygen+';declared effective oxygen activity,not measured H2O redox or pO2'))
        cases.append(dict(condition_id=cid,oxygen_scenario=oxygen,growth_T_scenario_C=T,
            C_cm3=float(raw['Sn_cm3']),qualitative_C_H_reported=True,full_composition_modeled=False,
            RT_300K_is_comparison_convention=True))
    tasks=[];refs={}
    for context in CONTEXTS:
        presets=sorted(n for n,s in manifest['presets'].items() if s['context']==context)
        for preset in presets:
            spec=manifest['presets'][preset];key='q'+str(len(tasks))
            fit=json.loads((BUNDLE/('models/'+spec['model_key']+'.json')).read_text()) if spec['serving_qualified'] else {}
            tasks.append(dict(id=key,input=dict(preset=preset,inputs=nodes,anchors=[])))
            refs[key]=dict(context=context,variant=spec['variant'],preset=preset,
                serving_qualified=spec['serving_qualified'],model_key=spec['model_key'],
                saved_evaluation_method=fit.get('evaluation_method',fit.get('method',spec['method'])))
    assert len(tasks)==96 and len(nodes)==16
    e.write(OUT/'execution_started.json',dict(created_utc=e.now(),requests=96,point_rows=1536,
        bundle_sha256=e.sha(BUNDLE/'manifest.json'),new_fits=0,goal_complete=False))
    e.write(OUT/'request_reference.json',refs)
    for i in range(4):e.write(OUT/f'tasks_{i}.json',dict(bundle=str(BUNDLE.resolve()),queries=tasks[i::4]))
    with ThreadPoolExecutor(max_workers=4) as pool:packets=list(pool.map(worker,range(4)))
    rows=[]
    for packet in packets:
        assert not packet['blocked_paths']
        for response in packet['rows']:
            ref=refs[response['id']];result=response['result']
            if result is not None:assert len(result['predictions'])==16
            for j,case in enumerate(cases):
                values=observed[case['condition_id']]['reported']
                targets=dict(nH=np.log10(float(values['nH_cm3'])),muH=np.log10(float(values['muH_cm2Vs'])))
                targets['sigma']=np.log10(E_CHARGE)+targets['nH']+targets['muH']
                point=result['predictions'][j] if result is not None else dict(numeric=False,valid=False)
                row=dict(request_id=response['id'],input_index=j,**ref,**case,
                    numeric=bool(point['numeric']),valid=bool(point['valid']),error=response['error'],
                    physical_state_available=point.get('physical_state_available'),
                    same_physical_state_for_output=point.get('same_physical_state_for_output'),
                    warnings=point.get('warnings',[]),continuation_log_nH=point.get('continuation_log10_nH'),
                    native_transport=result.get('transport') if result else None,
                    actual_evaluation_method=result.get('actual_evaluation_method',ref['saved_evaluation_method']) if result else None)
                for name in ('nH','muH','sigma'):
                    row['observed_log_'+name]=targets[name]
                    row['predicted_log_'+name]=None
                    row[name+'_abs_error_dex']=None
                row.update(predicted_log_mu_CT=None,predicted_log_sigma_CT=None,mu_CT_abs_error_dex=None,sigma_CT_abs_error_dex=None)
                if row['valid']:
                    for name,column in [('nH','log10_nH_cm3'),('muH','log10_muH_cm2Vs'),('sigma','log10_sigma_S_cm')]:
                        row['predicted_log_'+name]=point[column]
                        row[name+'_abs_error_dex']=abs(point[column]-targets[name])
                    row.update(predicted_log_mu_CT=point['log10_mu_CT'],predicted_log_sigma_CT=point['log10_sigma_CT'],
                        mu_CT_abs_error_dex=abs(point['log10_mu_CT']-targets['muH']),
                        sigma_CT_abs_error_dex=abs(point['log10_sigma_CT']-targets['sigma']))
                rows.append(row)
    assert len(rows)==1536
    e.csvwrite(OUT/'predictions.csv',rows)
    summaries=[]
    for context,oxygen in product(CONTEXTS,OXYGEN):
        for variant in sorted({r['variant'] for r in rows}):
            group=[r for r in rows if r['context']==context and r['variant']==variant and r['oxygen_scenario']==oxygen]
            for T in TEMPERATURES+(None,):
                g=[r for r in group if T is None or r['growth_T_scenario_C']==T]
                expected=8 if T is None else 2
                assert len(g)==expected
                complete=all(r['numeric'] and r['valid'] for r in g)
                row=dict(context=context,variant=variant,oxygen_scenario=oxygen,
                    scope='four_T_diagnostic_average' if T is None else 'one_T_scenario',growth_T_scenario_C=T,
                    scenario_point_rows=len(g),distinct_observed_conditions=2,coverage_complete=complete,
                    numeric=sum(r['numeric'] for r in g),valid=sum(r['valid'] for r in g),
                    actual_evaluation_methods='|'.join(sorted({str(r['actual_evaluation_method']) for r in g})),
                    independent_sigma_labels=0)
                for name in ('nH','muH','sigma','mu_CT','sigma_CT'):
                    row[name+'_MAE_dex']=float(np.mean([r[name+'_abs_error_dex'] for r in g])) if complete else None
                    row[name+'_worst_error_dex']=max(r[name+'_abs_error_dex'] for r in g) if complete else None
                summaries.append(row)
    assert len(summaries)==960
    e.csvwrite(OUT/'summary.csv',summaries)
    result=dict(created_utc=e.now(),requests=96,point_rows=len(rows),summary_rows=len(summaries),
        numeric=sum(r['numeric'] for r in rows),valid=sum(r['valid'] for r in rows),
        observed_conditions=2,DOIs=1,independent_laboratory_validation=False,
        new_shared_fits=0,new_source_fits=0,anchors=0,independent_sigma_labels=0,
        all48retained=True,goal_complete=False)
    e.write(OUT/'execution_complete.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def audit():
    protocol=declared()
    f=read(OUT/'predictions.csv');s=read(OUT/'summary.csv')
    observations=json.loads((BASE/'source_observations_v1.json').read_text())
    obs={x['condition_id']:x for x in observations['conditions']}
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    for item in manifest['files']:assert e.sha(BUNDLE/item['path'])==item['sha256']
    checks=[]
    def check(name,error,tolerance=0.):
        checks.append(dict(name=name,error=float(error),tolerance=tolerance,passed=bool(error<=tolerance)))
    check('all1536points',abs(len(f)-1536));check('all960summaries',abs(len(s)-960))
    check('all48methods',abs(f.variant.nunique()-48));check('only2observedconditions',abs(f.condition_id.nunique()-2))
    input_fields={'element','incorporated_cm3','Tobs_K','growth_T_C','H_known','H_cm3','carbon_known','carbon_cm3','scenario_pO2_atm','oxygen_input_kind'}
    for i in range(4):
        tasks=json.loads((OUT/f'tasks_{i}.json').read_text())['queries']
        packet=json.loads((OUT/f'outputs_{i}.json').read_text())
        check(f'worker{i}/forbidden_reads',len(packet['blocked_paths']))
        check(f'worker{i}/24requests',abs(len(tasks)-24))
        actual={x['id']:x for x in packet['rows']};check(f'worker{i}/24responses',abs(len(actual)-24))
        for task in tasks:
            inp=task['input'];spec=manifest['presets'][inp['preset']]
            assert set(inp)=={'preset','inputs','anchors'} and not inp['anchors']
            assert len(inp['inputs'])==16
            check(task['id']+'/no_new_source_in_training',int(SOURCE_DOI.lower() in [x.lower() for x in spec['train_families']]))
            if spec['context']==CONTEXTS[0]:check(task['id']+'/no_IKZ_old_family',int(RELATED_OLD_DOI in spec['train_families']))
            response=actual[task['id']]['result']
            for j,node in enumerate(inp['inputs']):
                assert set(node)==input_fields and node['element']=='Sn'
                assert node['growth_T_C'] in TEMPERATURES and node['scenario_pO2_atm'] in OXYGEN.values()
                assert node['Tobs_K']==300. and not node['H_known'] and not node['carbon_known']
                row=f[f.request_id.eq(task['id']) & f.input_index.eq(j)]
                assert len(row)==1;row=row.iloc[0]
                values=obs[row.condition_id]['reported']
                check(task['id']+f'/{j}/sample_chemical_input',abs(node['incorporated_cm3']/float(values['Sn_cm3'])-1.),1e-14)
                check(task['id']+f'/{j}/same_nH_label',abs(row.observed_log_nH-np.log10(float(values['nH_cm3']))),1e-13)
                check(task['id']+f'/{j}/same_mu_label',abs(row.observed_log_muH-np.log10(float(values['muH_cm2Vs']))),1e-13)
                check(task['id']+f'/{j}/derived_target_sigma',abs(row.observed_log_sigma-(np.log10(E_CHARGE)+row.observed_log_nH+row.observed_log_muH)),1e-13)
                point=response['predictions'][j] if response is not None else dict(valid=False,numeric=False)
                check(task['id']+f'/{j}/numeric_match',int(bool(point['numeric'])!=bool(row.numeric)))
                check(task['id']+f'/{j}/valid_match',int(bool(point['valid'])!=bool(row.valid)))
                check(task['id']+f'/{j}/qualified_flag',int(bool(spec['serving_qualified'])!=bool(row.serving_qualified)))
                if row.valid:
                    for name,column in [('nH','log10_nH_cm3'),('muH','log10_muH_cm2Vs'),('sigma','log10_sigma_S_cm'),('mu_CT','log10_mu_CT'),('sigma_CT','log10_sigma_CT')]:
                        target='muH' if name=='mu_CT' else 'sigma' if name=='sigma_CT' else name
                        predicted=getattr(row,'predicted_log_'+name)
                        check(task['id']+f'/{j}/{name}/replay',abs(predicted-point[column]),1e-13)
                        check(task['id']+f'/{j}/{name}/score',abs(getattr(row,name+'_abs_error_dex')-abs(predicted-getattr(row,'observed_log_'+target))),1e-13)
                    check(task['id']+f'/{j}/native_sigma',abs(row.predicted_log_sigma-(np.log10(E_CHARGE)+row.predicted_log_nH+row.predicted_log_muH)),1e-10)
                    check(task['id']+f'/{j}/CT_sigma',abs(row.predicted_log_sigma_CT-(np.log10(E_CHARGE)+row.predicted_log_nH+row.predicted_log_mu_CT)),1e-10)
                else:
                    check(task['id']+f'/{j}/no_invalid_score',sum(np.isfinite(getattr(row,n+'_abs_error_dex')) for n in ('nH','muH','sigma','mu_CT','sigma_CT')))
    for row in s.itertuples():
        g=f[f.context.eq(row.context)&f.variant.eq(row.variant)&f.oxygen_scenario.eq(row.oxygen_scenario)]
        if row.scope=='one_T_scenario':g=g[g.growth_T_scenario_C.eq(row.growth_T_scenario_C)]
        complete=bool(g.numeric.all() and g.valid.all())
        key=row.context+'/'+row.variant+'/'+row.oxygen_scenario+'/'+row.scope+'/'+str(row.growth_T_scenario_C)
        check(key+'/coverage',int(complete!=bool(row.coverage_complete)))
        check(key+'/point_count',abs(len(g)-row.scenario_point_rows))
        for name in ('nH','muH','sigma','mu_CT','sigma_CT'):
            if complete:
                check(key+'/'+name+'/mean',abs(getattr(row,name+'_MAE_dex')-g[name+'_abs_error_dex'].mean()),1e-13)
                check(key+'/'+name+'/worst',abs(getattr(row,name+'_worst_error_dex')-g[name+'_abs_error_dex'].max()),1e-13)
            else:check(key+'/'+name+'/no_partial_score',int(np.isfinite(getattr(row,name+'_MAE_dex')) or np.isfinite(getattr(row,name+'_worst_error_dex'))))
    result=dict(passed=all(x['passed'] for x in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        point_rows=len(f),summary_rows=len(s),numeric=int(f.numeric.sum()),valid=int(f.valid.sum()),
        observed_conditions=2,source_labels_not_in_runtime_inputs=True,no_forbidden_table_reads=True,
        old_related_source_excluded_in_primary_context=True,all48methods_retained=True,
        current_state_scenario_sampling_not_calibrated_uncertainty=True,goal_complete=False,
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [OUT/'protocol_before_transfer.json',OUT/'predictions.csv',OUT/'summary.csv']})
    e.write(OUT/'audit.json',result)
    print('Frozen Sn full-property audit',result['passed'],len(checks),flush=True)
    if not result['passed']:raise RuntimeError('preserve failure before any versioned audit repair')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['declare','run','audit'])
    action=parser.parse_args().action
    {'declare':declare,'run':run,'audit':audit}[action]()
