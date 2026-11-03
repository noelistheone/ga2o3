"""Frozen nested evaluation of the actual trained charge-disorder closures.

The inherited physical solver sees only input nodes and permitted Hall anchors.
Reported labels are attached after subprocess evaluation, for scoring/selection.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, os, sys, time, subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase30_core as c
import strong_tree_learning as l
import phase29_prediction as old
from sandbox.kroger_db import NPZ

OUT = l.OUT / 'prediction'
RUNTIME = Path(__file__).with_name('strong_tree_source_runtime.py')
VARIANTS = {m+'__nH':dict(method=m,policy='nH',gamma_zero=False) for m in l.METHODS}
PRIMARY = 'DirectMultiExtraTrees__nH'


def fitted(train, method, config):
    if method not in l.METHODS:
        return c.fit(train, method, config)
    key, path = l.fit_path(train, method, config)
    value = c.document(path)
    assert (value['key'] == key and value['train_units'] == sorted(train)
            and value['method'] == method and value['config'] == config)
    return value, path


def declare():
    target=OUT/'design_before_prediction.json';assert not target.exists()
    protocol=l.OUT/'protocol_before_training.json'
    c.verify_hashes(c.document(protocol))
    value=dict(created_utc=c.e.now(),primary=PRIMARY,variants=VARIANTS,
        comparison_control_only=True,planned_variants=2,planned_outer_rows=294,
        primary_scope='same current35/common20 and fullchain;no new physical method claim',
        candidate_and_source_selection=c.document(protocol)['model_selection'],
        code_hashes=c.file_hashes([Path(__file__),Path(l.__file__),RUNTIME,
            Path(__file__).with_name('strong_tree_saved.py'),Path(c.__file__),Path(old.__file__)]),
        input_hashes=c.file_hashes([protocol,c.OLD/'data/conditions35.csv',c.OLD/'data/scenarios49.csv',
            c.OLD/'data/episodes35.csv',c.OUT/'candidate_zero_predictions.csv']),goal_complete=False)
    c.write(target,value);return value


def audit_training():
    from strong_tree_saved import raw_values
    protocol=c.document(l.OUT/'protocol_before_training.json');c.verify_hashes(protocol)
    complete=c.document(l.OUT/'training_complete.json');assert complete['fit_records']==288
    checks=[];errors=[];forests=0
    for train in c.learning.contexts():
        rows=c.cases();rows=rows[rows.related_cluster.isin(train)]
        for method in l.METHODS:
            for config in l.configs():
                f,path=fitted(train,method,config)
                checks.extend([f['training_condition_ids']==sorted(rows.sample_id.unique()),
                    f['training_case_ids']==rows.case_id.tolist(),
                    f['training_nH_weights']==c.learning.weights(rows).tolist(),
                    f['training_rho_weights']==c.learning.weights(rows,paired_only=True).tolist(),
                    f['n_estimators']==64,f['random_seed']==1729,
                    f['training_valid'],f['selected_success'],len(f['nH_forest'])==64])
                values=np.array([raw_values(f,r) for r in rows.itertuples()])
                error=float(np.max(abs(values-np.array(f['selected_training_values']))))
                errors.append(error);checks.append(error<1e-10)
                forests+=f['actual_forest_fits']
    checks.append(forests==complete['forest_fits'])
    c.write(OUT/'training_audit.json',dict(passed=all(checks),checks=len(checks),fit_records=288,
        forest_fits=forests,maximum_property_replay_error=max(errors),created_utc=c.e.now(),goal_complete=False))
    assert all(checks),'Preserve actual tree training-audit failure'


def runtime(tasks, folder, lambdas, workers):
    """Run each packet exactly once, with actual PIDs recorded before waiting."""
    folder.mkdir(parents=True)
    packets=[[] for _ in range(workers)];loads=np.zeros(workers)
    for task in sorted(tasks,key=lambda t:(-len(t['nodes']),t['id'])):
        i=int(np.argmin(loads));packets[i].append(task);loads[i]+=len(task['nodes'])
    def launch(i):
        packet=folder/f'packet{i}.json';target=folder/f'worker{i}.json'
        assert not packet.exists() and not target.exists()
        c.write(packet,dict(tasks=packets[i],lambdas=lambdas,
                            physical_reference_sha256=c.e.sha(NPZ)))
        with (folder/f'worker{i}.log').open('w') as stream:
            process=subprocess.Popen([sys.executable,str(RUNTIME),str(packet),str(target)],
                                     cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
            c.write(folder/f'worker{i}_started.json',dict(pid=process.pid,
                    created_utc=c.e.now(),tasks=len(packets[i]),input_nodes=int(loads[i])))
            status=process.wait()
        assert status==0,(i,status)
        result=c.document(target);assert result['denied_reads']==[]
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(launch,range(workers)))
    records={r['id']:r for result in results for r in result['records']}
    assert len(records)==len(tasks)
    return records,results


def zero_predictions(workers):
    tasks=[];requests=[];table=c.cases()
    for train in c.learning.contexts():
        if len(train)==5:continue
        for method in l.METHODS:
            for config in l.configs():
                f,path=fitted(train,method,config)
                for held in c.units():
                    if held in train:continue
                    test=table[table.related_cluster.eq(held)]
                    task=dict(fit=l.minimal_fit(f),kind='offset',
                              nodes=[c.design.inputs(row) for row in test.itertuples()],
                              episodes=[dict(budget=0,anchor_condition_ids=[],anchor_indices=[],anchor_labels=[])])
                    task['id']=c.digest(task);tasks.append(task)
                    requests.append(dict(task_id=task['id'],fit_key=f['key'],held=held,
                        method=method,config=config,train_units=list(train),
                        case_ids=test.case_id.tolist(),fit_path=str(path.relative_to(c.ROOT)),
                        fit_sha256=c.e.sha(path)))
    assert len(tasks)==450
    c.write(OUT/'zero_plan.json',dict(tasks=tasks,requests=requests))
    records,workers_data=runtime(tasks,OUT/'zero_runtime',[None],workers)
    cb={r.case_id:r for r in table.itertuples()};rows=[]
    for request in requests:
        solution=records[request['task_id']]['episodes'][0]['solutions'][0]
        assert len(solution['records'])==len(request['case_ids'])
        for case_id,p in zip(request['case_ids'],solution['records']):
            row=cb[case_id]
            item=dict(task_id=request['task_id'],fit_key=request['fit_key'],held=request['held'],
                method=request['method'],case_id=case_id,sample_id=row.sample_id,
                family=row.family,related_cluster=row.related_cluster,
                mobility_eligible=bool(row.mobility_eligible),observed=float(row.target),
                observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
                observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+row.target+np.log10(row.mu_cm2Vs))
                    if row.mobility_eligible else None,**p)
            item.pop('diagnostics',None);item.pop('source_transport_derivative',None)
            rows.append(item)
    assert len(rows)==4410
    c.e.csvwrite(OUT/'candidate_zero_predictions.csv',rows)
    c.write(OUT/'zero_complete.json',dict(tasks=len(tasks),rows=len(rows),
            physical_calls=sum(r['actual_physical_point_calls'] for r in records.values()),
            worker_seconds=sum(v['seconds'] for v in records.values()),
            created_utc=c.e.now(),new_fits=0,goal_complete=False))
    return rows


def candidate(train,method,config,index):
    rows=[];refs=[]
    for held in train:
        inner=[u for u in train if u!=held]
        f,path=fitted(inner,method,config)
        rows.extend(index[(f['key'],held)])
        refs.append(dict(fit_key=f['key'],train_units=inner,held=held,
                         path=str(path.relative_to(c.ROOT)),sha256=c.e.sha(path)))
    f,path=fitted(train,method,config)
    return dict(method=method,config=config,metrics=c.metrics(rows),
                global_fit_key=f['key'],global_fit_path=str(path.relative_to(c.ROOT)),
                global_fit_sha256=c.e.sha(path),validation_fit_refs=refs,
                fallback_reference=False)


def select_shared(zero_rows):
    reference=c.read(c.OUT/'candidate_zero_predictions.csv')
    reference=reference[reference.method.eq('Reference205')]
    zero=pd.concat([pd.DataFrame(zero_rows),reference],ignore_index=True)
    index={key:part.to_dict('records') for key,part in zero.groupby(['fit_key','held'],sort=False)}
    scopes=[(u,tuple(v for v in c.units() if v!=u)) for u in c.units()]+[(None,c.units())]
    selections=[]
    for held,train in scopes:
        refs=[candidate(train,'Reference205',q,index) for q in c.learning.configs('Reference205')]
        good=[v for v in refs if c.finite(v['metrics']['nH'])]
        ref=min(good,key=lambda v:(round(v['metrics']['nH'],10),*c.design.config_tie(v['config']))) if good else refs[0]
        all_values={m:[candidate(train,m,q,index) for q in l.configs()] for m in l.METHODS}
        for variant,spec in VARIANTS.items():
            values=all_values[spec['method']]
            if spec['gamma_zero']:values=[v for v in values if v['config']['gamma']==0]
            selected=c.select(values,ref,spec['policy'])
            selections.append(dict(variant=variant,requested_method=spec['method'],
                policy=spec['policy'],gamma_zero=spec['gamma_zero'],outer_held=held,
                train_units=list(train),selected=selected,reference=ref,candidates=values,
                no_outer_labels_for_selection=True))
    assert len(selections)==12
    c.write(OUT/'shared_selections.json',selections)
    return selections


def source_plan(selections):
    cases=c.cases();roles=c.roles();plans=[];tasks={}
    for selection in selections:
        chosen=selection['selected'];method=chosen['method'];config=chosen['config']
        train=selection['train_units'];held=selection['outer_held']
        f,path=fitted(train,method,config)
        minimal=l.minimal_fit(f)
        plan=dict(id=c.digest(dict(variant=selection['variant'],outer_held=held)),
            variant=selection['variant'],requested_method=selection['requested_method'],
            policy=selection['policy'],outer_held=held,train_units=train,config=config,
            spec=dict(method=method,kind='offset' if method in l.METHODS else 'internal'),
            fallback_reference=bool(chosen.get('fallback_reference')),
            fit_key=f['key'],fit_path=str(path.relative_to(c.ROOT)),fit_sha256=c.e.sha(path),
            minimal_fit=minimal,inner=[],outer=[])
        contexts=[('inner',u,[v for v in train if v!=u]) for u in train]
        if held is not None:contexts.append(('outer',held,train))
        for where,target,actual_train in contexts:
            actual,actual_path=fitted(actual_train,method,config)
            minimal=l.minimal_fit(actual)
            for family in sorted(cases.loc[cases.related_cluster.eq(target),'family'].unique()):
                nodes,caseids,episodes=c.design.source_design(family,cases,roles)
                ident=c.digest(dict(fit=minimal,kind=plan['spec']['kind'],family=family,nodes=nodes,episodes=episodes))
                task=dict(id=ident,fit=minimal,kind=plan['spec']['kind'],nodes=nodes,episodes=episodes)
                if ident in tasks:assert tasks[ident]==task
                tasks[ident]=task
                plan[where].append(dict(task_id=ident,family=family,held_unit=target,
                    case_ids=caseids,fit_key=actual['key'],fit_train_units=actual_train,
                    fit_path=str(actual_path.relative_to(c.ROOT)),fit_sha256=c.e.sha(actual_path)))
                assert target not in actual['train_units']
        plans.append(plan)
    value=dict(plans=plans,tasks=list(tasks.values()),created_utc=c.e.now())
    c.write(OUT/'source_plan.json',value)
    return value


def source_evaluate(plan,workers):
    # Reuse only identical original packets; new learned fits have new keys.
    oldplan=c.document(c.OLD/'prediction/plan.json')
    oldtasks={t['id']:t for t in oldplan['tasks']};oldrecords={}
    for path in sorted((c.OLD/'prediction/runtime').glob('worker[0-9].json')):
        value=c.document(path);assert value['denied_reads']==[]
        oldrecords.update({r['id']:r for r in value['records']})
    reused={};fresh=[]
    for task in plan['tasks']:
        if task['id'] in oldrecords:
            assert task==oldtasks[task['id']]
            reused[task['id']]=oldrecords[task['id']]
        else:fresh.append(task)
    records,workers_data=runtime(fresh,OUT/'source_runtime',c.design.LAM,workers)
    records.update(reused)
    c.write(OUT/'source_compute.json',dict(tasks=len(plan['tasks']),new_tasks=len(fresh),
        reused_tasks=len(reused),reused_ids=sorted(reused),
        actual_new_physical_calls=sum(records[t['id']]['actual_physical_point_calls'] for t in fresh),
        actual_worker_seconds=sum(v['seconds'] for v in workers_data),
        prior_compute_not_counted_as_new=True,created_utc=c.e.now()))
    return records


def choose_sources(plan,records):
    cb={r.case_id:r for r in c.cases().itertuples()}
    rb={(r.sample_id,int(r.budget)):r.role for r in c.roles().itertuples()}
    predictions=[];choices=[];presets=[];diagnostics=[]
    for p in plan['plans']:
        policies={}
        for budget in (0,1,2):
            values=[]
            for lam in ([None] if budget==0 else c.design.LAM):
                rows=[]
                for request in p['inner']:
                    rows.extend(r for r in old.prediction_rows(records,request,budget,lam,cb,rb) if r['role']=='query')
                score=c.metrics(rows,group_equal=False) if rows else dict(nH=None,mu=None,rho=None)
                values.append(dict(lam=lam,metrics=score))
            base=next(v for v in values if v['lam'] is None)
            allowed=[v for v in values if c.safe(v['metrics'],base['metrics'],p['policy'])]
            chosen=min(allowed,key=lambda v:(round(v['metrics']['nH'],10),c.design.lambda_tie(v['lam']))) if allowed else dict(base,reason='no qualified empirical support')
            policies[str(budget)]=chosen['lam']
            choices.append(dict(plan_id=p['id'],variant=p['variant'],budget=budget,
                outer_held=p['outer_held'],train_units=p['train_units'],
                selected=chosen,candidates=values,no_outer_labels_for_selection=True))
            for request in p['outer']:
                for row in old.prediction_rows(records,request,budget,chosen['lam'],cb,rb):
                    row.update(variant=p['variant'],actual_method=p['spec']['method'],
                        requested_method=p['requested_method'],source_kind=p['spec']['kind'],
                        outer_held=p['outer_held'],fallback_reference=p['fallback_reference'],
                        global_fit_key=p['fit_key'],plan_id=p['id'])
                    row['row_id']=c.digest(dict(variant=row['variant'],budget=budget,case_id=row['case_id']))
                    diagnostics.append(dict(row_id=row['row_id'],diagnostics=row.pop('diagnostics',None),
                         source_transport_derivative=row.pop('source_transport_derivative',None)))
                    predictions.append(row)
        if p['outer_held'] is None:
            presets.append(dict(variant=p['variant'],fit=p['minimal_fit'],source_kind=p['spec']['kind'],
                source_lambdas=policies,fit_path=p['fit_path'],fit_sha256=p['fit_sha256'],
                actual_method=p['spec']['method'],fallback_reference=p['fallback_reference'],
                all_training_preset_is_not_validation=True))
    assert len(predictions)==294 and len(presets)==2
    c.e.csvwrite(OUT/'predictions.csv',predictions)
    c.write(OUT/'prediction_diagnostics.json',diagnostics)
    c.write(OUT/'source_choices.json',choices)
    c.write(OUT/'full_training_presets.json',presets)
    return pd.DataFrame(predictions)


def summary(table,study):
    rows=[];families=[]
    def stats(q):
        v=c.metrics(q.to_dict('records'),group_equal=False) if len(q) else dict(nH=None,mu=None,rho=None)
        paired=q[q.mobility_eligible.eq(True)]
        return dict(scenario_rows=len(q),conditions=q.sample_id.nunique(),DOIs=q.family.nunique(),
            related_groups=q.related_cluster.nunique(),paired_conditions=paired.sample_id.nunique(),
            paired_groups=paired.related_cluster.nunique(),numeric=int(q.numeric.eq(True).sum()),
            valid=int(q.valid.eq(True).sum()),nH_MAE=v['nH'],mu_MAE=v['mu'],sigma_MAE=v['rho'])
    for (variant,budget),part in table.groupby(['variant','budget'],sort=True):
        scopes={'all_query':part[part.role.eq('query')],
                'common20':part[part.role.eq('query') & part.primary_common20.eq(True)],
                'newSn_query':part[part.role.eq('query') & part.new_Sn.eq(True)],
                'Hall_anchor_conditional':part[part.role.eq('anchor')]}
        for scope,q in scopes.items():
            rows.append(dict(study=study,variant=variant,budget=int(budget),scope=scope,**stats(q)))
            for family,v in q.groupby('family',sort=True):
                families.append(dict(study=study,variant=variant,budget=int(budget),scope=scope,family=family,**stats(v)))
    return rows,families


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['declare','run'])
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    assert args.workers==2
    if args.action=='declare':
        print('Locked',declare()['primary'],'and both controls before outer predictions',flush=True);return
    design=c.document(OUT/'design_before_prediction.json');c.verify_hashes(design)
    assert not (OUT/'prediction_started.json').exists(),'Inspect actual original PID;never duplicate on timeout'
    audit_training()
    c.write(OUT/'prediction_started.json',dict(pid=os.getpid(),workers=args.workers,created_utc=c.e.now()))
    before=time.perf_counter();zero=zero_predictions(args.workers)
    selections=select_shared(zero);plan=source_plan(selections)
    records=source_evaluate(plan,args.workers);table=choose_sources(plan,records)
    rows,families=summary(table,'strong_trees')
    c.e.csvwrite(OUT/'summary.csv',rows);c.e.csvwrite(OUT/'per_family.csv',families)
    for study,path in [('phase29',c.OLD/'prediction/predictions.csv'),('phase30',c.OUT/'predictions.csv')]:
        a,b=summary(c.read(path),study);rows.extend(a);families.extend(b)
    c.e.csvwrite(OUT/'all_comparisons.csv',rows)
    c.e.csvwrite(OUT/'all_per_family.csv',families)
    c.verify_hashes(design);c.verify_hashes(c.document(l.OUT/'protocol_before_training.json'))
    c.write(OUT/'prediction_complete.json',dict(created_utc=c.e.now(),variants=len(VARIANTS),
        rows=len(table),seconds=time.perf_counter()-before,goal_complete=False,
        independent_result_audit_still_required=True))
    print('Actual2-variant nonlinear controls complete; independent audit pending',flush=True)


if __name__=='__main__':main()
