"""Training-inner chain-aware selection and matched 0/1/2-anchor forecasts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,sys,time,argparse
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase29_prediction as old
from sandbox.kroger_db import NPZ


def declare():
    target=c.OUT/'selection_protocol_before.json'
    assert not target.exists(),'Preserve any previous selection attempt'
    assert c.document(c.OUT/'zero_replay_audit.json')['passed']
    c.verify_hashes(c.document(c.OUT/'protocol_before_zero.json'))
    files=[c.OUT/'candidate_zero_predictions.csv',c.OUT/'zero_replay_audit.json',
        c.OUT/'zero_complete.json',c.OUT/'protocol_before_zero.json',
        c.OLD/'prediction/plan.json',c.OLD/'prediction/predictions.csv',
        c.OLD/'prediction/summary.csv',c.OLD/'prediction/audit.json',
        c.BASE/'phase30_theory_and_protocol.md',NPZ]
    files += sorted((c.OLD/'prediction/runtime').glob('worker[0-9].json'))
    code=[Path(__file__),Path(c.__file__),Path(old.__file__),
        c.OLD_SCRIPTS/'phase29_source_runtime.py',c.OLD_SCRIPTS/'phase29_prediction_design.py']
    value=dict(created_utc=c.e.now(),primary=c.PRIMARY,methods=c.METHODS,
        policies=c.POLICIES,planned_variants=39,
        global_gate='minimum nH;rho-safe or rho+mu-safe compared with same-context nH-selected Reference205',
        source_gate='minimum inner query nH;constraints relative to same-model no adaptation,lambda=None',
        no_outer_labels_for_selection=True,no_query_mu_in_source_packets=True,
        retrospective_development=True,independent_blind_validation=False,
        gate_constraints_are_not_population_safety_certificates=True,
        same_scenario_roles_common20_and_old_lambda_grid=True,
        new_model_training=0,all_old_48_comparisons_retained=True,
        code_hashes=c.file_hashes(code),input_hashes=c.file_hashes(files),
        goal_complete=False)
    c.write(target,value);c.verify_hashes(value)
    return value


def candidates(train,method,rows_by_pair):
    values=[]
    for config in c.learning.configs(method):
        rows=[];refs=[]
        for held in train:
            inside=[u for u in train if u!=held]
            fit,path=c.fit(inside,method,config)
            rows.extend(rows_by_pair[(fit['key'],held)])
            refs.append(dict(fit_key=fit['key'],train_units=inside,held=held,
                path=str(path.relative_to(c.ROOT)),sha256=c.e.sha(path)))
        actual,path=c.fit(train,method,config)
        vals=c.metrics(rows)
        values.append(dict(method=method,config=config,metrics=vals,
            global_fit_key=actual['key'],global_fit_path=str(path.relative_to(c.ROOT)),
            global_fit_sha256=c.e.sha(path),validation_fit_refs=refs,
            paired_conditions=len({r['sample_id'] for r in rows if r['mobility_eligible']}),
            paired_groups=len({r['related_cluster'] for r in rows if r['mobility_eligible']}),
            fallback_reference=False))
    return values


def select_global():
    zero=c.read(c.OUT/'candidate_zero_predictions.csv')
    rows_by_pair={key:part.to_dict('records') for key,part in zero.groupby(['fit_key','held'],sort=False)}
    selections=[];scopes=[(u,tuple(v for v in c.units() if v!=u)) for u in c.units()]+[(None,c.units())]
    for held,train in scopes:
        all_candidates={m:candidates(train,m,rows_by_pair) for m in c.METHODS}
        refs=all_candidates['Reference205']
        good=[r for r in refs if c.finite(r['metrics']['nH'])]
        reference=min(good,key=lambda r:(round(r['metrics']['nH'],10),*c.design.config_tie(r['config']))) if good else refs[0]
        for method in c.METHODS:
            for policy in c.POLICIES:
                if policy!='nH' and not c.finite(reference['metrics']['nH']):
                    chosen=dict(reference,fallback_reference=True,reason='unqualified reference;no safety claim')
                else:chosen=c.select(all_candidates[method],reference,policy)
                selections.append(dict(variant=method+'__'+policy,requested_method=method,policy=policy,
                    outer_held=held,train_units=list(train),selected=chosen,
                    reference=reference,candidates=all_candidates[method],
                    no_outer_labels_for_selection=True))
    assert len(selections)==234
    c.write(c.OUT/'shared_selections.json',selections)
    return selections


def make_plan(selections):
    table=c.cases();roles=c.roles();plans=[];tasks={}
    for selection in selections:
        chosen=selection['selected'];method=chosen['method'];config=chosen['config']
        train=selection['train_units'];held=selection['outer_held']
        fitted,path=c.fit(train,method,config)
        amended,_=c.design.with_transport_head(fitted,table[table.related_cluster.isin(train)],'native')
        plan=dict(variant=selection['variant'],spec=dict(method=method,
            kind='internal' if method in c.design.PHYSICAL else 'offset'),
            policy=selection['policy'],requested_method=selection['requested_method'],
            fallback_reference=bool(chosen.get('fallback_reference')),
            outer_held=held,train_units=train,config=config,
            fit_key=fitted['key'],fit_path=str(path.relative_to(c.ROOT)),
            fit_sha256=c.e.sha(path),minimal_fit=c.design.minimal_fit(amended),inner=[],outer=[])
        plan['id']=c.digest(dict(variant=plan['variant'],outer_held=held))
        contexts=[('inner',u,[v for v in train if v!=u]) for u in train]
        if held is not None:contexts.append(('outer',held,train))
        for where,target,fit_train in contexts:
            actual,actual_path=c.fit(fit_train,method,config)
            amended,_=c.design.with_transport_head(actual,table[table.related_cluster.isin(fit_train)],'native')
            minimal=c.design.minimal_fit(amended)
            for family in sorted(table.loc[table.related_cluster.eq(target),'family'].unique()):
                nodes,caseids,episodes=c.design.source_design(family,table,roles)
                ident=c.digest(dict(fit=minimal,kind=plan['spec']['kind'],family=family,nodes=nodes,episodes=episodes))
                task=dict(id=ident,fit=minimal,kind=plan['spec']['kind'],nodes=nodes,episodes=episodes)
                if ident in tasks:assert tasks[ident]==task
                tasks[ident]=task
                plan[where].append(dict(task_id=ident,family=family,held_unit=target,
                    case_ids=caseids,fit_key=actual['key'],fit_path=str(actual_path.relative_to(c.ROOT)),
                    fit_sha256=c.e.sha(actual_path),fit_train_units=fit_train))
                assert target not in actual['train_units']
        plans.append(plan)
    value=dict(created_utc=c.e.now(),plans=plans,tasks=list(tasks.values()),
        new_training=0,all_source_packets_label_isolated=True)
    c.write(c.OUT/'source_plan.json',value)
    return value


def sources(plan,workers):
    original=c.document(c.OLD/'prediction/plan.json')
    old_tasks={t['id']:t for t in original['tasks']};records={}
    for path in sorted((c.OLD/'prediction/runtime').glob('worker[0-9].json')):
        value=c.document(path);assert value['denied_reads']==[]
        records.update({r['id']:r for r in value['records']})
    reused={};new=[]
    for task in plan['tasks']:
        if task['id'] in records:
            assert task==old_tasks[task['id']]
            reused[task['id']]=records[task['id']]
        else:new.append(task)
    c.write(c.OUT/'source_cache_reuse.json',dict(created_utc=c.e.now(),
        total_tasks=len(plan['tasks']),reused_frozen_tasks=len(reused),new_tasks=len(new),
        reused_ids=sorted(reused),reused_fits_and_input_packets_identical=True,
        old_computation_not_counted_as_new=True,goal_complete=False))
    print('Phase30 source tasks',len(plan['tasks']),'reused',len(reused),'new',len(new),flush=True)
    packets=[[] for _ in range(workers)];loads=np.zeros(workers)
    for task in sorted(new,key=lambda t:(t['kind']!='internal',t['id'])):
        i=int(np.argmin(loads));packets[i].append(task)
        loads[i]+=len(task['nodes'])*(100 if task['kind']=='internal' else 2)
    runtime=c.OUT/'source_runtime';runtime.mkdir()
    def launch(index):
        packet=runtime/f'packet{index}.json';target=runtime/f'worker{index}.json'
        c.write(packet,dict(tasks=packets[index],lambdas=c.design.LAM,physical_reference_sha256=c.e.sha(NPZ)))
        with (runtime/f'worker{index}.log').open('w') as log:
            job=__import__('subprocess').Popen([sys.executable,str(c.OLD_SCRIPTS/'phase29_source_runtime.py'),
                str(packet),str(target)],cwd=c.ROOT,stdout=log,stderr=__import__('subprocess').STDOUT)
            c.write(runtime/f'worker{index}_started.json',dict(created_utc=c.e.now(),
                pid=job.pid,packet_tasks=len(packets[index])))
            code=job.wait()
        assert code==0,(index,code)
        value=c.document(target);assert value['denied_reads']==[]
        return value['records']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(launch,range(workers)):
            reused.update({r['id']:r for r in batch})
    assert len(reused)==len(plan['tasks'])
    return reused,new


def source_select(plan,records):
    cb={r.case_id:r for r in c.cases().itertuples()}
    rb={(r.sample_id,int(r.budget)):r.role for r in c.roles().itertuples()}
    predictions=[];choices=[];presets=[]
    for p in plan['plans']:
        policies={}
        for budget in (0,1,2):
            candidates=[]
            for lam in ([None] if budget==0 else c.design.LAM):
                rows=[]
                for request in p['inner']:
                    rows.extend(r for r in old.prediction_rows(records,request,budget,lam,cb,rb) if r['role']=='query')
                metrics=c.metrics(rows,group_equal=False) if rows else dict(nH=None,mu=None,rho=None)
                candidates.append(dict(lam=lam,metrics=metrics,
                    query_conditions=len({r['sample_id'] for r in rows}),
                    paired_conditions=len({r['sample_id'] for r in rows if r['mobility_eligible']}),
                    paired_groups=len({r['related_cluster'] for r in rows if r['mobility_eligible']})))
            baseline=next(v for v in candidates if v['lam'] is None)
            allowed=[v for v in candidates if c.safe(v['metrics'],baseline['metrics'],p['policy'])]
            if not allowed:chosen=dict(baseline,reason='no empirical support or qualified candidate')
            else:chosen=min(allowed,key=lambda v:(round(v['metrics']['nH'],10),c.design.lambda_tie(v['lam'])))
            policies[str(budget)]=chosen['lam']
            choices.append(dict(plan_id=p['id'],variant=p['variant'],budget=budget,
                outer_held=p['outer_held'],train_units=p['train_units'],
                selected=chosen,candidates=candidates,no_outer_label_selection=True,
                direct_mu_metric_used_in_inner_selection=p['policy']=='chain_safe',
                paired_training_labels_used_in_inner_selection=p['policy']!='nH'))
            for request in p['outer']:
                for row in old.prediction_rows(records,request,budget,chosen['lam'],cb,rb):
                    row.update(variant=p['variant'],source_kind=p['spec']['kind'],
                        outer_held=p['outer_held'],actual_method=p['spec']['method'],
                        requested_method=p['requested_method'],fallback_reference=p['fallback_reference'],
                        global_fit_key=p['fit_key'],plan_id=p['id'])
                    predictions.append(row)
        if p['outer_held'] is None:
            presets.append(dict(variant=p['variant'],fit=p['minimal_fit'],
                source_kind=p['spec']['kind'],source_lambdas=policies,
                actual_method=p['spec']['method'],fallback_reference=p['fallback_reference'],
                fit_path=p['fit_path'],fit_sha256=p['fit_sha256'],
                all_training_input_presets_not_validation=True))
    assert len(predictions)==5733 and len(presets)==39
    diagnostics=[]
    for row in predictions:
        row['row_id']=c.digest(dict(variant=row['variant'],budget=row['budget'],case_id=row['case_id']))
        diagnostics.append(dict(row_id=row['row_id'],diagnostics=row.pop('diagnostics',None),
            source_transport_derivative=row.pop('source_transport_derivative',None)))
    c.e.csvwrite(c.OUT/'predictions.csv',predictions)
    c.write(c.OUT/'prediction_diagnostics.json',diagnostics)
    c.write(c.OUT/'source_choices.json',choices)
    c.write(c.OUT/'full_training_presets.json',presets)
    return pd.DataFrame(predictions)


def summarize(table):
    def metrics(part):
        result=dict(scenario_rows=len(part),conditions=part.sample_id.nunique(),
            DOIs=part.family.nunique(),related_groups=part.related_cluster.nunique(),
            numeric=int(part.numeric.eq(True).sum()),valid=int(part.valid.eq(True).sum()),
            nH_MAE=None,mu_MAE=None,sigma_MAE=None,paired_conditions=0,paired_groups=0)
        if part.empty:return result
        vals=c.metrics(part.to_dict('records'),group_equal=False)
        paired=part[part.mobility_eligible.eq(True)]
        result.update(nH_MAE=vals['nH'],mu_MAE=vals['mu'],sigma_MAE=vals['rho'],
            paired_conditions=paired.sample_id.nunique(),paired_groups=paired.related_cluster.nunique())
        return result
    rows=[];families=[]
    for (variant,budget),part in table.groupby(['variant','budget'],sort=True):
        scopes={'all_query':part[part.role.eq('query')],
            'common20':part[part.role.eq('query') & part.primary_common20.eq(True)],
            'newSn_query':part[part.role.eq('query') & part.new_Sn.eq(True)],
            'Hall_anchor_conditional':part[part.role.eq('anchor')]}
        for scope,q in scopes.items():
            rows.append(dict(variant=variant,budget=int(budget),scope=scope,**metrics(q)))
            for family,v in q.groupby('family'):
                families.append(dict(variant=variant,budget=int(budget),scope=scope,family=family,**metrics(v)))
    c.e.csvwrite(c.OUT/'summary.csv',rows)
    c.e.csvwrite(c.OUT/'per_family.csv',families)


def replay_nH_controls(table):
    previous=c.read(c.OLD/'prediction/predictions.csv')
    previous=previous.set_index(['variant','budget','case_id'])
    checks=[];diff=[]
    for row in table[table.variant.str.endswith('__nH')].itertuples():
        method=row.requested_method
        variant=method+('_internal' if method in c.design.PHYSICAL else '_offset')
        oldrow=previous.loc[(variant,row.budget,row.case_id)]
        checks.extend([row.numeric==oldrow.numeric,row.valid==oldrow.valid])
        if row.numeric and oldrow.numeric:
            a=np.array([row.predicted,row.predicted_log_mu,row.predicted_log_sigma])
            b=oldrow[['predicted','predicted_log_mu','predicted_log_sigma']].to_numpy(float)
            err=float(np.max(abs(a-b)));checks.append(err<1e-9);diff.append(err)
    c.write(c.OUT/'nH_policy_replay_audit.json',dict(passed=all(checks),
        created_utc=c.e.now(),checks=len(checks),maximum_log_output_difference=max(diff),
        original_fixed_policies_replayed=True,new_training=0,goal_complete=False))
    assert all(checks),'Preserve selection replay discrepancies before versioned repair'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();assert args.workers==4
    assert not (c.OUT/'selection_started.json').exists(),'Inspect original handles before resuming'
    protocol=declare()
    c.write(c.OUT/'selection_started.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=args.workers))
    start=time.perf_counter();selection=select_global();plan=make_plan(selection)
    records,new=sources(plan,args.workers)
    table=source_select(plan,records);summarize(table);replay_nH_controls(table)
    c.verify_hashes(protocol)
    c.write(c.OUT/'selection_complete.json',dict(passed=True,created_utc=c.e.now(),
        variants=39,plans=len(plan['plans']),outer_scenario_rows=len(table),
        new_source_tasks=len(new),reused_source_tasks=len(plan['tasks'])-len(new),
        new_physical_point_calls=sum(records[t['id']]['actual_physical_point_calls'] for t in new),
        seconds=time.perf_counter()-start,new_optimizer_model_fits=0,
        source_adaptation_uses_only_allowed_Hall_anchors=True,goal_complete=False))
    print('Phase30 complete:39policies/5733rows;goal remains active',flush=True)


if __name__=='__main__':main()
