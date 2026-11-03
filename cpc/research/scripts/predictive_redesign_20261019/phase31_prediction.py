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
import phase31_learning as l
import phase29_prediction as old
from sandbox.kroger_db import NPZ

OUT = l.OUT / 'prediction'
RUNTIME = Path(__file__).with_name('phase31_source_runtime.py')
VARIANTS = {m+'__'+p: dict(method=m, policy=p, gamma_zero=False)
            for m in l.METHODS for p in c.POLICIES}
VARIANTS.update({m+'__nHonly': dict(method=m, policy='nH', gamma_zero=True)
                 for m in l.METHODS})
PRIMARY = 'ChargeRatioSpectrum__chain_safe'


def fitted(train, method, config):
    if method not in l.METHODS:
        return c.fit(train, method, config)
    key, path = l.fit_path(train, method, config)
    value = c.document(path)
    assert (value['key'] == key and value['train_units'] == sorted(train)
            and value['method'] == method and value['config'] == config)
    return value, path


def declare():
    target = OUT/'design_before_prediction.json'
    assert not target.exists(), 'Preserve prior declarations; repair by version'
    protocol = c.document(l.OUT/'training_protocol.json')
    c.verify_hashes(protocol)
    code = [Path(__file__), Path(l.__file__), Path(c.__file__), RUNTIME,
            Path(old.__file__), c.OLD_SCRIPTS/'phase29_prediction_design.py',
            c.OLD_SCRIPTS/'phase29_source_runtime.py']
    code += list((c.ROOT/'src/sandbox').glob('*.py'))
    inputs = [l.OUT/'training_protocol.json',
              c.OUT/'candidate_zero_predictions.csv',
              c.OUT/'independent_result_audit.json',
              c.OLD/'data/conditions35.csv', c.OLD/'data/scenarios49.csv',
              c.OLD/'data/episodes35.csv', c.OLD/'prediction/predictions.csv',
              c.OUT/'predictions.csv', NPZ]
    value = dict(created_utc=c.e.now(), primary=PRIMARY, variants=VARIANTS,
        primary_endpoint='zero-anchor original-common20 cross-related-group nH; full-chain compatibility required',
        planned_variants=16, planned_outer_rows=2352,
        primary_scope_locked_before_outer_predictions=True,
        shared_selection='same group-macro inner-CV nH; empirical rho/mu gate relative to same-context nH-selected Reference205',
        source_selection='same training-inner query DOI-macro nH; optional empirical gate relative to same-model no update',
        gamma_zero_ablation='same fitted stock and spectrum family; config candidates gamma=0 only; nH selection',
        reference_zeros_reused_from_identical_frozen_phase30=True,
        source_anchor_budget='0/1/2 physical Hall conditions; scenario duplicates are not additional labels',
        no_query_mu_in_source_packets=True, no_outer_labels_for_selection=True,
        all_old48_and_phase30_39_controls_retained=True,
        retrospective=True, independent_blind_test=False,
        failure_policy='invalid/nonstopped fits not served; incomplete coverage has no finite MAE',
        no_material_precision_from_math_gates=True,
        training_may_be_running=True, code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes(inputs), goal_complete=False)
    c.write(target, value)
    return value


def audit_training():
    protocol = c.document(l.OUT/'training_protocol.json')
    c.verify_hashes(protocol)
    complete = c.document(l.OUT/'training_complete.json')
    assert complete['fit_records'] == 576
    checks = []; refs = []; starts = 0; invalid = []; nonstopped = []
    for train in c.learning.contexts():
        rows = l.subset(train)
        for method in l.METHODS:
            for config in l.configs():
                f, path = fitted(train, method, config)
                checks.extend([f['training_condition_ids'] == sorted(rows.sample_id.unique()),
                               f['training_case_ids'] == rows.case_id.tolist(),
                               f['training_nH_weights'] == c.learning.weights(rows).tolist()])
                finite = [t for t in f['trials'] if t['loss'] is not None]
                best = min(finite, key=lambda t:t['loss']) if finite else None
                checks.extend([f['new_optimizer_starts'] == len(f['trials']),
                               f['selected_success'] == bool(best and best['success'])])
                if best:
                    checks.extend([f['parameters'] == best['parameters'],
                                   f['selected_loss'] == best['loss'],
                                   f['training_valid'] == all(f['selected_training_valid']),
                                   len(f['parameters']) == 8])
                if not f['training_valid']: invalid.append(f['key'])
                if not f['selected_success']: nonstopped.append(f['key'])
                warm = f['initializer_reference']
                checks.append(c.e.sha(c.ROOT/warm['path']) == warm['sha256'])
                parent = c.document(c.ROOT/warm['path'])
                checks.extend([parent['train_units'] == sorted(train),
                               parent['config'] == config,
                               parent['method'] == 'ShareProcess205'])
                starts += f['new_optimizer_starts']
                refs.append(dict(path=str(path.relative_to(c.ROOT)),sha256=c.e.sha(path)))
    checks.extend([len(refs)==576, starts==complete['actual_optimizer_starts'],
                   len(invalid)==complete['invalid_training_fits'],
                   len(nonstopped)==complete['selected_nonstopped']])
    result = dict(passed=all(checks),checks=len(checks),fit_records=len(refs),
                  actual_optimizer_starts=starts,invalid_keys=invalid,
                  nonstopped_keys=nonstopped,fit_hashes=refs,
                  inherited_warm_start_compute_linked_not_new=True,
                  created_utc=c.e.now(),goal_complete=False)
    c.write(OUT/'training_audit.json',result)
    assert result['passed'], 'Keep failed audit; do not overwrite'


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
                    task=dict(fit=c.design.minimal_fit(f),kind='internal',
                              nodes=[c.design.inputs(row) for row in test.itertuples()],
                              episodes=[dict(budget=0,anchor_condition_ids=[],anchor_indices=[],anchor_labels=[])])
                    task['id']=c.digest(task);tasks.append(task)
                    requests.append(dict(task_id=task['id'],fit_key=f['key'],held=held,
                        method=method,config=config,train_units=list(train),
                        case_ids=test.case_id.tolist(),fit_path=str(path.relative_to(c.ROOT)),
                        fit_sha256=c.e.sha(path)))
    assert len(tasks)==900
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
    assert len(rows)==8820
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
    assert len(selections)==96
    c.write(OUT/'shared_selections.json',selections)
    return selections


def source_plan(selections):
    cases=c.cases();roles=c.roles();plans=[];tasks={}
    for selection in selections:
        chosen=selection['selected'];method=chosen['method'];config=chosen['config']
        train=selection['train_units'];held=selection['outer_held']
        f,path=fitted(train,method,config)
        minimal=c.design.minimal_fit(f)
        plan=dict(id=c.digest(dict(variant=selection['variant'],outer_held=held)),
            variant=selection['variant'],requested_method=selection['requested_method'],
            policy=selection['policy'],outer_held=held,train_units=train,config=config,
            spec=dict(method=method,kind='internal'),
            fallback_reference=bool(chosen.get('fallback_reference')),
            fit_key=f['key'],fit_path=str(path.relative_to(c.ROOT)),fit_sha256=c.e.sha(path),
            minimal_fit=minimal,inner=[],outer=[])
        contexts=[('inner',u,[v for v in train if v!=u]) for u in train]
        if held is not None:contexts.append(('outer',held,train))
        for where,target,actual_train in contexts:
            actual,actual_path=fitted(actual_train,method,config)
            minimal=c.design.minimal_fit(actual)
            for family in sorted(cases.loc[cases.related_cluster.eq(target),'family'].unique()):
                nodes,caseids,episodes=c.design.source_design(family,cases,roles)
                ident=c.digest(dict(fit=minimal,kind='internal',family=family,nodes=nodes,episodes=episodes))
                task=dict(id=ident,fit=minimal,kind='internal',nodes=nodes,episodes=episodes)
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
                        requested_method=p['requested_method'],source_kind='internal',
                        outer_held=p['outer_held'],fallback_reference=p['fallback_reference'],
                        global_fit_key=p['fit_key'],plan_id=p['id'])
                    row['row_id']=c.digest(dict(variant=row['variant'],budget=budget,case_id=row['case_id']))
                    diagnostics.append(dict(row_id=row['row_id'],diagnostics=row.pop('diagnostics',None),
                         source_transport_derivative=row.pop('source_transport_derivative',None)))
                    predictions.append(row)
        if p['outer_held'] is None:
            presets.append(dict(variant=p['variant'],fit=p['minimal_fit'],source_kind='internal',
                source_lambdas=policies,fit_path=p['fit_path'],fit_sha256=p['fit_sha256'],
                actual_method=p['spec']['method'],fallback_reference=p['fallback_reference'],
                all_training_preset_is_not_validation=True))
    assert len(predictions)==2352 and len(presets)==16
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
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    assert args.workers==4
    if args.action=='declare':
        print('Locked',declare()['primary'],'and all16variants before outer predictions',flush=True);return
    design=c.document(OUT/'design_before_prediction.json');c.verify_hashes(design)
    assert not (OUT/'prediction_started.json').exists(),'Inspect actual original PID;never duplicate on timeout'
    audit_training()
    c.write(OUT/'prediction_started.json',dict(pid=os.getpid(),workers=args.workers,created_utc=c.e.now()))
    before=time.perf_counter();zero=zero_predictions(args.workers)
    selections=select_shared(zero);plan=source_plan(selections)
    records=source_evaluate(plan,args.workers);table=choose_sources(plan,records)
    rows,families=summary(table,'phase31')
    c.e.csvwrite(OUT/'summary.csv',rows);c.e.csvwrite(OUT/'per_family.csv',families)
    for study,path in [('phase29',c.OLD/'prediction/predictions.csv'),('phase30',c.OUT/'predictions.csv')]:
        a,b=summary(c.read(path),study);rows.extend(a);families.extend(b)
    c.e.csvwrite(OUT/'all_comparisons.csv',rows)
    c.e.csvwrite(OUT/'all_per_family.csv',families)
    c.verify_hashes(design);c.verify_hashes(c.document(l.OUT/'training_protocol.json'))
    c.write(OUT/'prediction_complete.json',dict(created_utc=c.e.now(),variants=len(VARIANTS),
        rows=len(table),seconds=time.perf_counter()-before,goal_complete=False,
        independent_result_audit_still_required=True))
    print('Actual16-variant held-out predictions complete; independent audit pending',flush=True)


if __name__=='__main__':main()
