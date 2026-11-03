"""Independent data/selection/primitive-output checks for the new rate studies."""
from pathlib import Path
import argparse,sys,importlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c


def score(rows,endpoint,group_equal):
    frame=pd.DataFrame(rows)
    if endpoint!='nH':frame=frame[frame.mobility_eligible.eq(True)]
    if frame.empty:return None
    pred,obs={'nH':('predicted','observed'),'mu':('predicted_log_mu','observed_log_mu'),
              'rho':('predicted_log_sigma','observed_log_sigma')}[endpoint]
    if not (frame.numeric.eq(True).all() and frame.valid.eq(True).all()):return None
    if not np.isfinite(frame[[pred,obs]].to_numpy(float)).all():return None
    condition=frame.assign(loss=abs(frame[pred]-frame[obs])).groupby(
        ['related_cluster','family','sample_id']).loss.mean()
    doi=condition.groupby(level=['related_cluster','family']).mean()
    return float(doi.groupby(level='related_cluster').mean().mean() if group_equal else doi.mean())


def metrics(rows,group_equal):return {k:score(rows,k,group_equal) for k in ['nH','mu','rho']}
def finite(x):return x is not None and np.isfinite(x)
def accepts(v,ref,policy):
    keys=[] if policy=='nH' else ['rho'] if policy=='rho_safe' else ['rho','mu']
    return finite(v['nH']) and all(finite(v[k]) and finite(ref[k]) and v[k]<=ref[k]+1e-10 for k in keys)
def same(a,b):return (not finite(a) and not finite(b)) or (finite(a) and finite(b) and abs(a-b)<2e-11)
def tie(v):return (round(v['metrics']['nH'],10),-float(v['config'].get('alpha',0.)),float(v['config'].get('gamma',0.)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',type=int,choices=[31,32],required=True)
    args=parser.parse_args();module=importlib.import_module(f'phase{args.phase}_prediction')
    out=module.OUT;target=out/'independent_result_audit.json';assert not target.exists()
    assert (out/'prediction_complete.json').exists()
    c.verify_hashes(c.document(out/'design_before_prediction.json'))
    c.verify_hashes(c.document(module.l.OUT/'training_protocol.json'))
    checks=[];errors=[]
    def check(value,kind):
        value=bool(value);checks.append(value)
        if not value:errors.append(kind)
    audit=c.document(out/'training_audit.json');check(audit['passed'],'training audit')
    for item in audit['fit_hashes']:
        check(c.e.sha(c.ROOT/item['path'])==item['sha256'],'fit changed')
    cases={r.case_id:r for r in c.cases().itertuples()}
    roles={(r.sample_id,int(r.budget)):r.role for r in c.roles().itertuples()}
    table=c.read(out/'predictions.csv');summary=c.read(out/'summary.csv')
    zero=c.read(out/'candidate_zero_predictions.csv')
    ref=c.read(c.OUT/'candidate_zero_predictions.csv');ref=ref[ref.method.eq('Reference205')]
    z=pd.concat([zero,ref],ignore_index=True)
    index={key:part.to_dict('records') for key,part in z.groupby(['fit_key','held'])}
    choices=c.document(out/'shared_selections.json')
    for choice in choices:
        train=set(choice['train_units']);outer=choice['outer_held']
        check(outer is None or outer not in train,'outer group leakage')
        for candidate in [*choice['candidates'],choice['reference']]:
            rows=[]
            for item in candidate['validation_fit_refs']:
                f=c.document(c.ROOT/item['path'])
                check(c.e.sha(c.ROOT/item['path'])==item['sha256'],'inner fit changed')
                check(item['held'] in train and item['held'] not in item['train_units'],'inner group leakage')
                check(set(item['train_units'])==train-{item['held']},'inner training context')
                check(set(f['train_units'])==set(item['train_units']),'actual fit context')
                rows.extend(index[(item['fit_key'],item['held'])])
            actual=metrics(rows,True)
            for endpoint in actual:check(same(actual[endpoint],candidate['metrics'][endpoint]),'candidate score')
        permitted=[v for v in choice['candidates'] if accepts(v['metrics'],choice['reference']['metrics'],choice['policy'])]
        if choice['policy']!='nH':permitted.append(dict(choice['reference'],fallback_reference=True))
        expected=min(permitted,key=lambda v:(round(v['metrics']['nH'],10),bool(v.get('fallback_reference',False)),*tie(v)[1:])) if permitted else choice['reference']
        actual=choice['selected']
        check(actual['global_fit_key']==expected['global_fit_key'],'global selected fit')
        check(actual['config']==expected['config'] and actual['method']==expected['method'],'global selected config')
        if choice['gamma_zero']:
            check(all(v['config']['gamma']==0 for v in choice['candidates']),'gamma-zero ablation')
    plan=c.document(out/'source_plan.json');tasks={t['id']:t for t in plan['tasks']}
    records={}
    for path in sorted((out/'source_runtime').glob('worker[0-9].json')):
        value=c.document(path);check(value['denied_reads']==[],'forbidden runtime read')
        records.update({r['id']:r for r in value['records']})
    reused=c.document(out/'source_compute.json')['reused_ids']
    if reused:
        originals=c.document(c.OLD/'prediction/plan.json');oldtasks={t['id']:t for t in originals['tasks']}
        for path in sorted((c.OLD/'prediction/runtime').glob('worker[0-9].json')):
            value=c.document(path);check(value['denied_reads']==[],'old forbidden read')
            for r in value['records']:
                if r['id'] in reused:
                    check(tasks[r['id']]==oldtasks[r['id']],'nonidentical cache reuse');records[r['id']]=r
    check(set(tasks)==set(records),'runtime coverage')
    pmap={p['id']:p for p in plan['plans']}
    requests={}
    for p in plan['plans']:
        for where in ['inner','outer']:
            for request in p[where]:
                task=tasks[request['task_id']]
                check(request['held_unit'] not in request['fit_train_units'],'source fit leakage')
                check(c.e.sha(c.ROOT/request['fit_path'])==request['fit_sha256'],'source fit changed')
                check(task['fit']['key']==request['fit_key'],'packet fit key')
                for case_id,node in zip(request['case_ids'],task['nodes']):
                    check(node==c.design.inputs(cases[case_id]),'source input mismatch')
                    check(not set(node)&{'target','observed','mu_cm2Vs','observed_log_mu'},'label in input node')
                for episode in task['episodes']:
                    b=episode['budget'];ids=episode['anchor_condition_ids']
                    expected_ids=sorted({cases[i].sample_id for i in request['case_ids'] if roles[(cases[i].sample_id,b)]=='anchor'})
                    check(ids==expected_ids and len(ids)==b,'physical anchor budget')
                    expected_indices=[i for i,name in enumerate(request['case_ids']) if cases[name].sample_id in ids]
                    check(episode['anchor_indices']==expected_indices,'anchor node indices')
                    check(episode['anchor_labels']==[float(cases[request['case_ids'][i]].target) for i in expected_indices],'anchor labels')
                if where=='outer':requests[(p['id'],request['family'])]=request
    def read_rows(request,budget,lam):
        episode=next(x for x in records[request['task_id']]['episodes'] if x['budget']==budget)
        sol=next(x for x in episode['solutions'] if x['lam']==lam)
        result=[]
        for cid,r in zip(request['case_ids'],sol['records']):
            a=cases[cid]
            if roles[(a.sample_id,budget)]!='query':continue
            result.append(dict(family=a.family,sample_id=a.sample_id,related_cluster=a.related_cluster,
                observed=float(a.target),observed_log_mu=float(np.log10(a.mu_cm2Vs)) if a.mobility_eligible else None,
                observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+a.target+np.log10(a.mu_cm2Vs)) if a.mobility_eligible else None,
                **r,mobility_eligible=bool(a.mobility_eligible)))
            result[-1]['valid']=bool(r['valid'] and sol['valid'])
        return result
    for choice in c.document(out/'source_choices.json'):
        p=pmap[choice['plan_id']]
        for candidate in choice['candidates']:
            rows=[]
            for request in p['inner']:rows.extend(read_rows(request,choice['budget'],candidate['lam']))
            actual=metrics(rows,False) if rows else dict(nH=None,mu=None,rho=None)
            for k in actual:check(same(actual[k],candidate['metrics'][k]),'source inner metric')
        base=next(v for v in choice['candidates'] if v['lam'] is None)
        permitted=[v for v in choice['candidates'] if accepts(v['metrics'],base['metrics'],p['policy'])]
        expected=min(permitted,key=lambda v:(round(v['metrics']['nH'],10),-np.inf if v['lam'] is None else -float(v['lam']))) if permitted else base
        check(expected['lam']==choice['selected']['lam'],'source lambda selection')
    identity=[]
    for row in table.itertuples():
        a=cases[row.case_id]
        check(row.sample_id==a.sample_id and row.family==a.family and row.related_cluster==a.related_cluster,'row provenance')
        check(row.role==roles[(row.sample_id,int(row.budget))] and abs(row.observed-a.target)<1e-12,'roles or Hall label')
        if row.mobility_eligible:check(abs(row.observed_log_mu-np.log10(a.mu_cm2Vs))<1e-12,'paired mu label')
        request=requests[(row.plan_id,row.family)]
        episode=next(v for v in records[request['task_id']]['episodes'] if v['budget']==row.budget)
        lam=None if not np.isfinite(row.source_lambda) else float(row.source_lambda)
        sol=next(v for v in episode['solutions'] if v['lam']==lam)
        raw=sol['records'][request['case_ids'].index(row.case_id)]
        check(row.numeric==raw['numeric'] and row.valid==bool(raw['valid'] and sol['valid']),'raw qualification')
        if row.numeric:
            for key in ['predicted','predicted_log_mu','predicted_log_sigma']:check(abs(getattr(row,key)-raw[key])<2e-11,'raw output')
            delta=abs(row.predicted_log_sigma-(np.log10(c.learning.E_CHARGE)+row.predicted+row.predicted_log_mu))
            check(delta<1e-10,'Hall identity');identity.append(delta)
        if row.budget==0:check(row.source_raw_shift==0,'zero budget source shift')
    for row in summary.itertuples():
        q=table[table.variant.eq(row.variant)&table.budget.eq(row.budget)]
        if row.scope=='common20':q=q[q.role.eq('query')&q.primary_common20.eq(True)]
        elif row.scope=='all_query':q=q[q.role.eq('query')]
        elif row.scope=='newSn_query':q=q[q.role.eq('query')&q.new_Sn.eq(True)]
        else:q=q[q.role.eq('anchor')]
        if row.scope=='common20':check(q.sample_id.nunique()==20,'fixed common20')
        actual=metrics(q.to_dict('records'),False) if len(q) else dict(nH=None,mu=None,rho=None)
        for endpoint,key in [('nH','nH_MAE'),('mu','mu_MAE'),('rho','sigma_MAE')]:
            check(same(actual[endpoint],getattr(row,key)),'reported summary')
    check(len(table)==2352 and table.variant.nunique()==16 and len(choices)==96,'declared coverage')
    main=[r for r in choices if r['variant']==module.PRIMARY and r['outer_held'] is not None]
    value=dict(passed=all(checks),created_utc=c.e.now(),checks=len(checks),failures=errors,
        primary=module.PRIMARY,primary_outer_groups=len(main),
        primary_reference_fallback_groups=sum(r['selected']['fallback_reference'] for r in main),
        max_Hall_identity_error=max(identity),original_labels_roles_common20_verified=True,
        independent_nested_and_source_selection_recomputed=True,
        audit_does_not_establish_predictive_gain=True,goal_complete=False,
        code_sha256=c.e.sha(__file__),evidence_sha256=c.file_hashes([
            out/'predictions.csv',out/'summary.csv',out/'shared_selections.json',out/'source_choices.json']))
    c.write(target,value)
    print('Independent phase',args.phase,'audit',len(checks),'checks;failures',len(errors),flush=True)
    assert value['passed'],errors[:10]


if __name__=='__main__':main()
