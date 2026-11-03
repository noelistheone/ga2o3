"""Actual nested selection and source-excluded phase29 property prediction."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import sys,json,time,argparse,subprocess,os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.joint_saved_model import SavedJointModel
from sandbox.kroger_db import NPZ
from sandbox.ionized_transport import E_CHARGE
import phase29_learning as l
import phase29_data as data
import phase29_prediction_design as d
import phase2_energy as e

OUT=data.OUT/'prediction'


def number(value):return np.inf if value is None else float(value)


def verify_hashes(protocol):
    for kind in ('code_hashes','input_hashes'):
        for name,digest in protocol[kind].items():assert e.sha(ROOT/name)==digest,(kind,name)


def freeze():
    design=d.declare();verify_hashes(design)
    assert json.loads((data.OUT/'training_audit.json').read_text())['passed']
    assert json.loads((data.OUT/'training_complete.json').read_text())['fit_records']==1392
    code=[Path(__file__),Path(__file__).with_name('phase29_prediction_design.py'),
        Path(__file__).with_name('phase29_source_runtime.py'),ROOT/'src/sandbox/joint_saved_model.py']
    files=[data.OUT/'training_audit.json',data.OUT/'training_complete.json',
        data.OUT/'prediction_design_protocol.json',data.OUT/'data/scenarios49.csv',
        data.OUT/'data/conditions35.csv',data.OUT/'data/episodes35.csv',NPZ]
    value=dict(created_utc=e.now(),expected_variants=48,expected_outer_scenario_rows=7056,
        primary=d.PRIMARY,actual_new_training_in_prediction=0,
        isolated_source_runtime=True,full_deployment_not_validation=True,
        separate_inner_and_outer_contexts=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files})
    path=OUT/'protocol_before_prediction.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==value[k],k
        return old
    e.write(path,value);return value


def zero_score(fit,path,held,cases):
    key=l.digest(dict(protocol=e.sha(OUT/'protocol_before_prediction.json'),fit_sha=e.sha(path),held=held))
    dest=OUT/'zero_cache'/(key+'.json')
    if dest.exists():
        value=json.loads(dest.read_text());return number(value['score']),value['rows']
    assert held not in fit['train_units']
    table=cases[cases.related_cluster.eq(held)];rows=[]
    try:mapper=SavedJointModel(d.minimal_fit(fit));failure=None
    except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:mapper=None;failure=repr(exc)
    for row in table.itertuples():
        item=dict(case_id=row.case_id,sample_id=row.sample_id,family=row.family,held=held,
            observed=float(row.target),numeric=False,valid=False)
        if mapper is None:item['error']=failure
        else:
            try:
                p=mapper.point(SimpleNamespace(**d.inputs(row)),require_observation=False)
                item.update(predicted=p.log_nH,numeric=bool(np.isfinite(p.log_nH)),valid=p.valid)
            except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:item['error']=repr(exc)
        rows.append(item)
    score=d.condition_score(rows)
    e.write(dest,dict(key=key,fit_key=fit['key'],fit_sha256=e.sha(path),held=held,
        train_units=fit['train_units'],score=score,rows=rows,no_held_labels_given_to_mapper=True))
    return score,rows


def shared_choice(train,method,selection,cases):
    candidates=[]
    for config in d.configs(method,selection):
        scores=[];fits=[]
        for held in train:
            inner=[u for u in train if u!=held]
            fit,path=d.load_fit(inner,method,config)
            score,_=zero_score(fit,path,held,cases);scores.append(score)
            fits.append(dict(key=fit['key'],path=str(path.relative_to(ROOT)),sha256=e.sha(path),
                held=held,train_units=inner))
        candidates.append(dict(config=config,score=float(np.mean(scores)),held_units=list(train),scores=scores,fits=fits))
    best=min(candidates,key=lambda c:(round(c['score'],10),*d.config_tie(c['config'])))
    return best,candidates


def prepare(cases,roles):
    dest=OUT/'plan.json'
    if dest.exists():return json.loads(dest.read_text())
    units=l.units();plans=[];tasks={}
    scopes=[(u,tuple(v for v in units if v!=u)) for u in units]+[(None,units)]
    needed=sorted(set((v['method'],v['selection']) for v in d.VARIANTS.values()))
    for held,train in scopes:
        choices={}
        for method,selection in needed:
            best,candidates=shared_choice(train,method,selection,cases)
            choices[(method,selection)]=best['config']
            e.write(OUT/'shared_choices'/(l.digest(dict(train=train,method=method,selection=selection))+'.json'),
                dict(train_units=train,outer_held=held,method=method,selection=selection,
                    selected=best,candidates=candidates,only_training_inner_labels=True))
        for variant,spec in d.VARIANTS.items():
            method=spec['method'];config=choices[(method,spec['selection'])]
            fit,path=d.load_fit(train,method,config)
            own=cases[cases.related_cluster.isin(train)]
            amended,headmeta=d.with_transport_head(fit,own,spec['head'])
            plan=dict(id=l.digest(dict(held=held,variant=variant)),variant=variant,spec=spec,
                outer_held=held,train_units=train,config=config,fit_key=fit['key'],fit_path=str(path.relative_to(ROOT)),
                fit_sha256=e.sha(path),minimal_fit=d.minimal_fit(amended),transport_head_audit=headmeta,
                inner=[],outer=[])
            contexts=[('inner',u,[v for v in train if v!=u]) for u in train]
            if held is not None:contexts.append(('outer',held,list(train)))
            for where,target,fit_train in contexts:
                actual,actual_path=d.load_fit(fit_train,method,config)
                fitted,head=d.with_transport_head(actual,cases[cases.related_cluster.isin(fit_train)],spec['head'])
                for family in sorted(cases.loc[cases.related_cluster.eq(target),'family'].unique()):
                    nodes,caseids,episodes=d.source_design(family,cases,roles)
                    packetfit=d.minimal_fit(fitted)
                    taskid=l.digest(dict(fit=packetfit,kind=spec['kind'],family=family,nodes=nodes,episodes=episodes))
                    task=dict(id=taskid,fit=packetfit,kind=spec['kind'],nodes=nodes,episodes=episodes)
                    if taskid in tasks:assert tasks[taskid]==task
                    else:tasks[taskid]=task
                    plan[where].append(dict(task_id=taskid,family=family,held_unit=target,
                        case_ids=caseids,fit_key=actual['key'],fit_path=str(actual_path.relative_to(ROOT)),
                        fit_sha256=e.sha(actual_path),fit_train_units=fit_train,head_audit=head))
                    assert target not in actual['train_units']
            plans.append(plan)
            print('plan29',held,variant,flush=True)
    assert len(plans)==48*6
    value=dict(created_utc=e.now(),plans=plans,tasks=list(tasks.values()),no_new_model_fits=True,
        no_query_mu_in_source_packets=True,goal_complete=False)
    e.write(dest,value);return e.clean(value)


def run_sources(tasks,workers):
    packets=[[] for _ in range(workers)];loads=np.zeros(workers)
    for task in sorted(tasks,key=lambda q:(q['kind']!='internal',q['id'])):
        index=int(np.argmin(loads));packets[index].append(task)
        loads[index]+=len(task['nodes'])*(100 if task['kind']=='internal' else 2)
    def worker(index):
        target=OUT/'runtime'/('worker'+str(index)+'.json')
        taskpath=OUT/'runtime'/('packet'+str(index)+'.json')
        packet=dict(tasks=packets[index],lambdas=d.LAM,physical_reference_sha256=e.sha(NPZ))
        if taskpath.exists():assert json.loads(taskpath.read_text())==e.clean(packet)
        else:e.write(taskpath,packet)
        if target.exists():
            saved=json.loads(target.read_text())
            assert {r['id'] for r in saved['records']}=={r['id'] for r in packets[index]}
            assert not saved['denied_reads'] and not saved['runtime_query_labels']
            return saved['records']
        logpath=target.with_suffix('.log')
        assert not logpath.exists(),'inspect previous child handle/output before a versioned resume'
        with logpath.open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('phase29_source_runtime.py')),
                str(taskpath),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            e.write(OUT/'runtime'/('worker'+str(index)+'_started.json'),dict(created_utc=e.now(),pid=process.pid,packet_tasks=len(packets[index])))
            code=process.wait()
        if code!=0:raise RuntimeError('Source child failed,retain original log: '+str(logpath))
        saved=json.loads(target.read_text())
        assert not saved['denied_reads'] and not saved['runtime_query_labels']
        return saved['records']
    with ThreadPoolExecutor(max_workers=workers) as pool:values=list(pool.map(worker,range(workers)))
    records=[r for packet in values for r in packet]
    assert len(records)==len(tasks) and len({r['id'] for r in records})==len(tasks)
    return {r['id']:r for r in records}


def solution(records,request,budget,lam):
    record=records[request['task_id']]
    ep=[v for v in record['episodes'] if v['budget']==budget]
    assert len(ep)==1
    selected=[s for s in ep[0]['solutions'] if s['lam']==lam]
    assert len(selected)==1
    return selected[0]


def prediction_rows(records,request,budget,lam,cases_by_id,roles_by_id):
    sol=solution(records,request,budget,lam);result=[]
    assert len(sol['records'])==len(request['case_ids'])
    for case_id,p in zip(request['case_ids'],sol['records']):
        row=cases_by_id[case_id]
        item=dict(case_id=case_id,sample_id=row.sample_id,family=row.family,
            related_cluster=row.related_cluster,observed=float(row.target),
            role=roles_by_id[(row.sample_id,budget)],scenario_index=int(row.scenario_index),
            scenario_count=int(row.scenario_count),scenario_weight=float(row.scenario_weight),
            primary_common20=bool(row.primary_common20),new_Sn=row.family==data.GOGOVA,
            mobility_eligible=bool(row.mobility_eligible),
            observed_log_mu=float(np.log10(row.mu_cm2Vs)) if row.mobility_eligible else None,
            incorporated_cm3=float(row.incorporated_cm3),growth_T_C=float(row.growth_T_C),
            scenario_pO2_atm=float(row.scenario_pO2_atm),budget=budget,
            source_lambda=lam,source_raw_shift=sol.get('raw_shift'),
            task_id=request['task_id'],source_solution_valid=sol['valid'],**p)
        item['valid']=bool(item['valid'] and sol['valid'])
        if item['mobility_eligible']:
            item['observed_log_sigma']=float(np.log10(E_CHARGE)+item['observed']+item['observed_log_mu'])
        else:item['observed_log_sigma']=None
        if item['numeric']:
            item['predicted_log_sigma_CT']=float(np.log10(E_CHARGE)+item['predicted']+item['predicted_log_mu_CT'])
        result.append(item)
    return result


def summarize(plan,records,cases,roles):
    cb={r.case_id:r for r in cases.itertuples()};rb={(r.sample_id,int(r.budget)):r.role for r in roles.itertuples()}
    predictions=[];choices=[];presets=[]
    for p in plan['plans']:
        policies={}
        for b in (0,1,2):
            candidates=[]
            for lam in ([None] if b==0 else d.LAM):
                rows=[]
                for request in p['inner']:
                    rows.extend(r for r in prediction_rows(records,request,b,lam,cb,rb) if r['role']=='query')
                candidates.append(dict(lam=lam,score=d.condition_score(rows),
                    scenario_query_rows=len(rows),condition_query_count=len({r['sample_id'] for r in rows})))
            chosen=min(candidates,key=lambda q:(round(q['score'],10),d.lambda_tie(q['lam'])))
            policies[str(b)]=chosen['lam']
            choices.append(dict(plan_id=p['id'],variant=p['variant'],budget=b,outer_held=p['outer_held'],
                train_units=p['train_units'],selected=chosen,candidates=candidates,
                query_mu_used=False,outer_labels_used_for_selection=False))
            for request in p['outer']:
                for row in prediction_rows(records,request,b,chosen['lam'],cb,rb):
                    row.update(variant=p['variant'],source_kind=p['spec']['kind'],outer_held=p['outer_held'],
                        global_method=p['spec']['method'],global_selection=p['spec']['selection'],
                        global_config=p['config'],global_fit_key=p['fit_key'],plan_id=p['id'])
                    predictions.append(row)
        if p['outer_held'] is None:
            presets.append(dict(variant=p['variant'],fit=p['minimal_fit'],source_kind=p['spec']['kind'],
                source_lambdas=policies,fit_path=p['fit_path'],fit_sha256=p['fit_sha256'],
                training_units=p['train_units'],training_config=p['config'],domain='conditional Si/Sn300K;explicit process/H/C scenarios',
                no_material_Born_certificate=True,no_new_element_ranking_claim=True))
    assert len(predictions)==7056 and len(presets)==48
    diagnostics=[]
    for row in predictions:
        rid=l.digest(dict(variant=row['variant'],budget=row['budget'],case_id=row['case_id']))
        row['row_id']=rid
        diagnostics.append(dict(row_id=rid,diagnostics=row.pop('diagnostics',None),
            source_transport_derivative=row.pop('source_transport_derivative',None),
            global_config=row.pop('global_config')))
    e.csvwrite(OUT/'predictions.csv',predictions);e.write(OUT/'diagnostics.json',diagnostics)
    e.write(OUT/'source_choices.json',choices);e.write(OUT/'full_deployment_presets.json',presets)
    table=pd.DataFrame(predictions);summaries=[];families=[]
    def metrics(part):
        ok=bool(len(part) and part.numeric.eq(True).all() and part.valid.eq(True).all())
        result=dict(scenario_rows=len(part),conditions=part.sample_id.nunique(),DOIs=part.family.nunique(),
            related_groups=part.related_cluster.nunique(),numeric=int(part.numeric.eq(True).sum()),valid=int(part.valid.eq(True).sum()),
            nH_MAE=None,paired_nH_MAE=None,mu_MAE=None,sigma_MAE=None,mu_CT_MAE=None,sigma_CT_MAE=None)
        def macro(sub,column,obs):
            if sub.empty or not (sub.numeric.eq(True).all() and sub.valid.eq(True).all()):return None
            tmp=sub.assign(error=abs(sub[column]-sub[obs]))
            percondition=tmp.groupby(['family','sample_id']).error.mean()
            return float(percondition.groupby(level='family').mean().mean())
        if ok:result['nH_MAE']=macro(part,'predicted','observed')
        paired=part[part.mobility_eligible.eq(True)]
        result.update(paired_conditions=paired.sample_id.nunique(),paired_scenario_rows=len(paired),
            paired_nH_MAE=macro(paired,'predicted','observed'),
            mu_MAE=macro(paired,'predicted_log_mu','observed_log_mu'),
            sigma_MAE=macro(paired,'predicted_log_sigma','observed_log_sigma'),
            mu_CT_MAE=macro(paired,'predicted_log_mu_CT','observed_log_mu'),
            sigma_CT_MAE=macro(paired,'predicted_log_sigma_CT','observed_log_sigma'))
        if ok:
            tmp=part.assign(error=abs(part.predicted-part.observed))
            result['nH_group_macro_MAE']=float(tmp.groupby(['related_cluster','family','sample_id']).error.mean()
                .groupby(level=['related_cluster','family']).mean().groupby(level='related_cluster').mean().mean())
            result['worst_DOI_nH_MAE']=float(tmp.groupby(['family','sample_id']).error.mean().groupby(level='family').mean().max())
        else:result.update(nH_group_macro_MAE=None,worst_DOI_nH_MAE=None)
        return result
    for (variant,b),part in table.groupby(['variant','budget'],sort=True):
        scopes={'all_query':part[part.role.eq('query')],
            'common20':part[part.role.eq('query')&part.primary_common20.eq(True)],
            'newSn_query':part[part.role.eq('query')&part.new_Sn.eq(True)],
            'Hall_anchor_conditional':part[part.role.eq('anchor')]}
        for scope,sub in scopes.items():
            summaries.append(dict(variant=variant,budget=int(b),scope=scope,**metrics(sub)))
            for family,values in sub.groupby('family'):
                families.append(dict(variant=variant,budget=int(b),scope=scope,family=family,**metrics(values)))
    e.csvwrite(OUT/'summary.csv',summaries);e.csvwrite(OUT/'per_family.csv',families)
    result=dict(created_utc=e.now(),variants=48,outer_scenario_rows=len(table),
        outer_reported_conditions=table.sample_id.nunique(),full_deployment_presets=48,
        summary_rows=len(summaries),source_choices=len(choices),unique_source_tasks=len(records),
        actual_physical_point_calls=sum(r['actual_physical_point_calls'] for r in records.values()),
        source_worker_task_seconds=sum(r['seconds'] for r in records.values()),
        outer_label_selection=False,no_query_mu_in_source_runtime=True,
        sigma_is_derived=True,shared_increment_not_automatically_established=True,goal_complete=False)
    e.write(OUT/'execution_complete.json',result)
    print(table[table.variant.eq(d.PRIMARY)].groupby('budget')[['numeric','valid']].sum().to_string(),flush=True)
    print('Prediction29 complete',result,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();protocol=freeze();verify_hashes(protocol)
    assert not (OUT/'execution_started.json').exists(),'inspect actual parent/children before a versioned resume'
    e.write(OUT/'execution_started.json',dict(created_utc=e.now(),pid=os.getpid(),workers=args.workers))
    cases=data.load_cases();roles=data.load_roles()
    plan=prepare(cases,roles);records=run_sources(plan['tasks'],args.workers)
    summarize(plan,records,cases,roles)


if __name__=='__main__':main()
