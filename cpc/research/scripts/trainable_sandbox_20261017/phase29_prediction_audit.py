"""Independent split, label-budget, selection, readout and score reconciliation."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from sandbox.context_saved_model import INPUT_NAMES
from sandbox.ionized_transport import E_CHARGE
from sandbox.transport import mu_caughey_thomas
import phase29_data as data
import phase2_energy as e

OUT=data.OUT/'prediction'


def macro(rows,pred='predicted',obs='observed'):
    if not rows or not all(r['numeric'] and r['valid'] for r in rows):return np.inf
    values=[]
    for family in sorted({r['family'] for r in rows}):
        sub=[r for r in rows if r['family']==family]
        condition=[]
        for sid in sorted({r['sample_id'] for r in sub}):
            condition.append(float(np.mean([abs(r[pred]-r[obs]) for r in sub if r['sample_id']==sid])))
        values.append(float(np.mean(condition)))
    return float(np.mean(values))


def main():
    assert not (OUT/'audit.json').exists(),'preserve prior audit before any versioned repair'
    paths=[OUT/n for n in ('protocol_before_prediction.json','execution_complete.json','plan.json',
        'predictions.csv','summary.csv','per_family.csv','diagnostics.json','source_choices.json',
        'full_deployment_presets.json')]
    paths+=list((OUT/'runtime').glob('worker[0-9].json'))
    e.write(OUT/'audit_protocol_before.json',dict(created_utc=e.now(),code_sha256=e.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths},
        value_tolerance=2e-10,missingness_masks_checked=True,no_new_training=True,no_prediction_changes=True))
    checks=[];maximum=0.
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    def close(name,a,b,tol=2e-10,**kw):
        nonlocal maximum
        x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
        if x.shape!=y.shape or not np.array_equal(np.isnan(x),np.isnan(y)) or not np.array_equal(np.isinf(x),np.isinf(y)):
            add(name,False,shape_or_missingness_mismatch=True,**kw);return
        finite=np.isfinite(x)&np.isfinite(y)
        error=float(np.max(abs(x[finite]-y[finite])/(1+abs(y[finite])))) if finite.any() else 0.
        maximum=max(maximum,error);add(name,error<=tol,error=error,tolerance=tol,**kw)
    for filename in ('training_protocol.json','prediction_design_protocol.json','pipeline_protocol_before_wait.json'):
        p=json.loads((data.OUT/filename).read_text())
        maps=('bound_files',) if filename.startswith('pipeline_') else ('code_hashes','input_hashes')
        for kind in maps:
            for name,digest in p[kind].items():add('bound_dependency_unchanged',e.sha(ROOT/name)==digest,path=name)
    cases=data.load_cases();roles=data.load_roles();cb={r.case_id:r for r in cases.itertuples()}
    rb={(r.sample_id,int(r.budget)):r.role for r in roles.itertuples()}
    frame=read(OUT/'predictions.csv');summaries=read(OUT/'summary.csv')
    plan=json.loads((OUT/'plan.json').read_text());tasks={t['id']:t for t in plan['tasks']}
    worker_results=[]
    for path in sorted((OUT/'runtime').glob('worker[0-9].json')):
        p=json.loads(path.read_text());worker_results+=p['records']
        add('runtime_did_not_read_result_or_query_tables',not p['denied_reads'] and not p['runtime_query_labels'],worker=path.name)
    results={r['id']:r for r in worker_results}
    add('runtime_task_coverage',len(results)==len(worker_results)==len(tasks) and set(results)==set(tasks))
    add('fixed_outer_scenario_grid',len(frame)==7056 and len(plan['plans'])==288
        and frame.variant.nunique()==48 and not frame.duplicated(['variant','budget','case_id']).any())
    add('35conditions_not49independent_samples',frame.sample_id.nunique()==35)
    add('original_common20_unchanged',frame[frame.primary_common20.eq(True)].sample_id.nunique()==20)
    add('no_newSn_twoanchor_query',frame[frame.new_Sn.eq(True)&frame.budget.eq(2)&frame.role.eq('query')].empty)
    choices=json.loads((OUT/'source_choices.json').read_text())
    choice_map={(x['plan_id'],int(x['budget'])):x for x in choices}
    add('all_source_choices_retained',len(choices)==288*3 and len(choice_map)==len(choices))
    row_map={(r.variant,int(r.budget),r.case_id):r for r in frame.itertuples()}
    for filename in sorted((OUT/'shared_choices').glob('*.json')):
        choice=json.loads(filename.read_text());train=set(choice['train_units']);scored=[]
        for candidate in choice['candidates']:
            scores=[]
            for fit_info in candidate['fits']:
                fit=json.loads((ROOT/fit_info['path']).read_text())
                add('shared_selection_fit_is_training_inner_only',set(fit['train_units'])<=train
                    and fit_info['held'] not in fit['train_units'] and fit_info['held'] in train,
                    key=fit['key'])
                cache_key=e.sha(OUT/'protocol_before_prediction.json')
                payload=dict(protocol=cache_key,fit_sha=e.sha(ROOT/fit_info['path']),held=fit_info['held'])
                import hashlib
                key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
                cached=json.loads((OUT/'zero_cache'/(key+'.json')).read_text())
                for row in cached['rows']:
                    truth=cb[row['case_id']]
                    add('zero_score_uses_correct_held_labels',truth.related_cluster==fit_info['held']
                        and row['observed']==float(truth.target),key=fit['key'],case=row['case_id'])
                scores.append(macro(cached['rows']))
            score=float(np.mean(scores));stored=np.inf if candidate['score'] is None else candidate['score']
            close('shared_hyperparameter_scores_reconciled',[stored],[score],file=filename.name)
            cfg=candidate['config']
            scored.append((score,cfg))
        best=min(scored,key=lambda q:(round(q[0],10),
            -float(q[1].get('alpha',q[1].get('ridge_alpha',0.))),float(q[1].get('gamma',0.))))
        add('shared_hyperparameter_choice_training_only',choice['selected']['config']==best[1],file=filename.name)
    def selected(request,budget,lam):
        eps=[v for v in results[request['task_id']]['episodes'] if v['budget']==budget]
        assert len(eps)==1
        matches=[s for s in eps[0]['solutions'] if s['lam']==lam];assert len(matches)==1
        return matches[0]
    for p in plan['plans']:
        held=p['outer_held'];train=set(p['train_units'])
        add('whole_outer_group_excluded',len(train)==(5 if held is None else 4) and held not in train,plan=p['id'])
        for where in ('inner','outer'):
            for request in p[where]:
                f=json.loads((ROOT/request['fit_path']).read_text())
                add('actual_saved_fit_excludes_request_group',request['held_unit'] not in f['train_units']
                    and set(f['train_units'])<=train and set(request['fit_train_units'])==set(f['train_units'])
                    and e.sha(ROOT/request['fit_path'])==request['fit_sha256'],plan=p['id'],where=where)
                task=tasks[request['task_id']]
                add('packet_minimal_fit_matches_actual_parameters',task['fit']['key']==f['key']
                    and task['fit']['parameters']==f['parameters']
                    and task['fit']['training_valid']==f['training_valid']
                    and task['fit']['selected_success']==f['selected_success'],task=task['id'])
                for field in ('center','scale','nH_coefficients','rho_coefficients','coefficients','offset_dex'):
                    if field in task['fit']:
                        add('runtime_coefficients_are_the_declared_fit',task['fit'][field]==f[field],task=task['id'],field=field)
                add('no_training_tables_in_runtime_fit',not any(k in task['fit'] for k in
                    ('training_condition_ids','selected_training_values','target','mu_cm2Vs')),task=task['id'])
                for node,case_id in zip(task['nodes'],request['case_ids']):
                    case=cb[case_id]
                    add('runtime_nodes_only_allowed_inputs',set(node)==set(INPUT_NAMES)
                        and all(node[k]==e.clean(getattr(case,k)) for k in INPUT_NAMES),task=task['id'],case=case_id)
                for ep in task['episodes']:
                    names=sorted(roles.loc[roles.family.eq(request['family'])&roles.budget.eq(ep['budget'])&
                        roles.role.eq('anchor'),'sample_id'])
                    indices=[i for i,s in enumerate(request['case_ids']) if cb[s].sample_id in names]
                    labels=[float(cb[request['case_ids'][i]].target) for i in indices]
                    add('only_allowed_physical_Hall_anchors',names==sorted(ep['anchor_condition_ids'])
                        and len(names)==ep['budget'] and indices==ep['anchor_indices']
                        and labels==ep['anchor_labels'],task=task['id'],budget=ep['budget'])
                h=request['head_audit']
                if 'training_condition_ids' in h:
                    expected=sorted(cases.loc[cases.related_cluster.isin(f['train_units']),'sample_id'].unique())
                    add('transport_calibration_excludes_all_held_labels',h['training_condition_ids']==expected
                        and h['no_query_mu'],task=task['id'])
                    own=cases.set_index('case_id').loc[f['training_case_ids']]
                    mask=own.mobility_eligible.eq(True).to_numpy(bool);paired=own.loc[mask]
                    ng=paired.related_cluster.nunique()
                    nf=paired.groupby('related_cluster').family.transform('nunique').to_numpy(float)
                    ni=paired.groupby('family').sample_id.transform('nunique').to_numpy(float)
                    weight=paired.scenario_weight.to_numpy(float)/(ng*nf*ni)
                    values=np.asarray(f['selected_training_values'],float)
                    if h['head']=='CT_mu_mean':
                        mu=np.log10([mu_caughey_thomas(10**v,300.) for v in values[mask,0]])
                        error=np.log10(paired.mu_cm2Vs.to_numpy(float))-mu
                        expected_offset=float(weight@error)
                    elif h['head']=='native_rho_mean':
                        obs_sigma=np.log10(E_CHARGE)+paired.target.to_numpy(float)+np.log10(paired.mu_cm2Vs.to_numpy(float))
                        expected_offset=float(weight@(obs_sigma-values[mask,2]))
                    else:
                        error=np.log10(paired.mu_cm2Vs.to_numpy(float))-values[mask,1]
                        if h['head']=='native_mu_median':
                            order=np.argsort(error,kind='stable')
                            expected_offset=float(error[order][np.searchsorted(np.cumsum(weight[order]),weight.sum()/2,side='left')])
                        else:expected_offset=float(weight@error)
                    close('postfit_transport_offset_from_training_pairs_only',
                        [task['fit']['transport_head_offset_dex']],[expected_offset],task=task['id'])
        for b in (0,1,2):
            choice=choice_map[(p['id'],b)];scored=[]
            for candidate in choice['candidates']:
                rows=[]
                for request in p['inner']:
                    sol=selected(request,b,candidate['lam'])
                    for case_id,v in zip(request['case_ids'],sol['records']):
                        r=cb[case_id]
                        if rb[(r.sample_id,b)]!='query':continue
                        rows.append(dict(family=r.family,sample_id=r.sample_id,observed=float(r.target),
                            predicted=v.get('predicted'),numeric=v['numeric'],valid=v['valid'] and sol['valid']))
                score=macro(rows);stored=np.inf if candidate['score'] is None else candidate['score']
                close('source_selection_score_uses_inner_queries_only',[stored],[score],plan=p['id'],budget=b,lam=candidate['lam'])
                scored.append((score,candidate['lam']))
            chosen=min(scored,key=lambda q:(round(q[0],10),-np.inf if q[1] is None else -q[1]))
            add('source_policy_matches_declared_training_only_rule',choice['selected']['lam']==chosen[1],plan=p['id'],budget=b)
            for request in p['outer']:
                sol=selected(request,b,chosen[1])
                for case_id,v in zip(request['case_ids'],sol['records']):
                    case=cb[case_id];actual=row_map[(p['variant'],b,case_id)]
                    add('outer_scope_and_role',actual.family==case.family and actual.outer_held==case.related_cluster
                        and actual.role==rb[(case.sample_id,b)] and actual.sample_id==case.sample_id,case=case_id)
                    close('outer_target_is_the_original_label',[actual.observed],[float(case.target)],case=case_id)
                    if bool(case.mobility_eligible):
                        close('paired_observations_and_derived_sigma',[actual.observed_log_mu,actual.observed_log_sigma],
                            [np.log10(case.mu_cm2Vs),np.log10(E_CHARGE)+case.target+np.log10(case.mu_cm2Vs)],case=case_id)
                    add('outer_missingness_and_qualification_preserved',bool(actual.numeric)==v['numeric']
                        and bool(actual.valid)==bool(v['valid'] and sol['valid']),case=case_id)
                    if v['numeric']:
                        close('stored_raw_source_readout',[actual.predicted,actual.predicted_log_mu,actual.predicted_log_sigma],
                            [v['predicted'],v['predicted_log_mu'],v['predicted_log_sigma']],case=case_id)
    numeric=frame[frame.numeric.eq(True)]
    close('single_state_sigma_identity',numeric.predicted_log_sigma.to_numpy(float),
        np.log10(E_CHARGE)+numeric.predicted.to_numpy(float)+numeric.predicted_log_mu.to_numpy(float),1e-11)
    for row in summaries.itertuples():
        part=frame[frame.variant.eq(row.variant)&frame.budget.eq(row.budget)]
        if row.scope=='all_query':part=part[part.role.eq('query')]
        elif row.scope=='common20':part=part[part.role.eq('query')&part.primary_common20.eq(True)]
        elif row.scope=='newSn_query':part=part[part.role.eq('query')&part.new_Sn.eq(True)]
        elif row.scope=='Hall_anchor_conditional':part=part[part.role.eq('anchor')]
        else:raise RuntimeError('Undeclared summary scope')
        add('summary_condition_and_scenario_counts',row.scenario_rows==len(part) and row.conditions==part.sample_id.nunique()
            and row.DOIs==part.family.nunique(),variant=row.variant,budget=row.budget,scope=row.scope)
        for field,pred,obs,paired in (
            ('nH_MAE','predicted','observed',False),('paired_nH_MAE','predicted','observed',True),
            ('mu_MAE','predicted_log_mu','observed_log_mu',True),
            ('sigma_MAE','predicted_log_sigma','observed_log_sigma',True),
            ('mu_CT_MAE','predicted_log_mu_CT','observed_log_mu',True),
            ('sigma_CT_MAE','predicted_log_sigma_CT','observed_log_sigma',True)):
            sub=part[part.mobility_eligible.eq(True)] if paired else part
            score=macro(sub.to_dict('records'),pred,obs)
            expected=np.nan if not np.isfinite(score) else score
            close('independent_DOI_condition_scenario_MAE',[getattr(row,field)],[expected],
                variant=row.variant,budget=row.budget,scope=row.scope,metric=field)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        variants=48,outer_scenario_rows=len(frame),reported_conditions=35,
        actual_new_training=0,maximum_scaled_difference=maximum,
        all_matching_NaN_masks_explicit=True,no_queries_or_scores_changed=True,
        not_independent_material_validation=True,goal_complete=False)
    e.write(OUT/'audit.json',result)
    print('Prediction29 audit',result['passed'],result['checks'],'max',maximum,flush=True)
    if not result['passed']:raise RuntimeError('Preserve failed audit;do not change predictions or widen tolerance')


if __name__=='__main__':main()
