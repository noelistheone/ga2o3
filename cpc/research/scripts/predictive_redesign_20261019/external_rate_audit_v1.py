"""Independent external-cohort raw-worker, label/split, preset and score audit."""
from pathlib import Path
import sys,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

DATA=c.BASE/'external_rate_cohort_v2';OUT=DATA/'prediction'
AUDIT=c.BASE/'external_rate_independent_audit_v1'


def metrics(frame):
    if frame.empty:return dict(nH=None,mu=None,sigma=None)
    if not frame.numeric.eq(True).all() or not frame.valid.eq(True).all():return dict(nH=None,mu=None,sigma=None)
    result={}
    for name,pred,obs in [('nH','predicted','observed'),('mu','predicted_log_mu','observed_log_mu'),('sigma','predicted_log_sigma','observed_log_sigma')]:
        if not np.isfinite(frame[[pred,obs]].to_numpy(float)).all():result[name]=None;continue
        errors=frame.assign(error=abs(frame[pred]-frame[obs])).groupby(['family','sample_id']).error.mean()
        result[name]=float(errors.groupby(level='family').mean().mean())
    return result


def main():
    assert not (AUDIT/'protocol_before.json').exists()
    files=[p for p in OUT.rglob('*') if p.is_file()]
    files += [DATA/'scenarios24.csv',DATA/'conditions3.csv',DATA/'episodes.csv',DATA/'qualification.json']
    c.write(AUDIT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes([Path(__file__)]),
        input_hashes=c.file_hashes(files),new_fits=0,new_model_evaluations=0,goal_complete=False))
    checks=[];failures=[]
    def check(ok,why):
        checks.append(bool(ok))
        if not ok:failures.append(why)
    c.verify_hashes(c.document(OUT/'design_before_prediction.json'))
    presets=c.document(OUT/'actual_presets_before_prediction.json');c.verify_hashes(dict(input_hashes=presets['preset_hashes']))
    presets={p['id']:p['preset'] for p in presets['presets']}
    table=c.read(DATA/'scenarios24.csv');cases={r.case_id:r for r in table.itertuples()}
    roles={(r.sample_id,int(r.budget)):r.role for r in c.read(DATA/'episodes.csv').itertuples()}
    plan=c.document(OUT/'plan_before_runtime.json');tasks={}
    for label in ['zero','adapt']:
        tasks.update({t['id']:t for t in plan[label+'_tasks']})
    recordmap={};actualcalls=0
    for folder in ['zero_runtime','source_runtime']:
        packet_ids=[];worker_ids=[]
        for i in [0,1]:
            packet=c.document(OUT/folder/f'packet{i}.json');worker=c.document(OUT/folder/f'worker{i}.json')
            check(worker['denied_reads']==[],'forbidden runtime read');check(worker['runtime_query_labels'] is False,'runtime query labels')
            for t in packet['tasks']:
                packet_ids.append(t['id']);check(tasks[t['id']]==t,'packet != frozen task')
                for node in t['nodes']:
                    check(not set(node)&{'target','observed','mu_cm2Vs','observed_log_mu','predicted'},'labels in physical input')
                if folder=='zero_runtime':
                    check(packet['lambdas']==[None],'zero lambda grid')
                    check(len(t['episodes'])==1 and t['episodes'][0]['budget']==0,'zero episode')
                    check(t['episodes'][0]['anchor_labels']==[] and t['episodes'][0]['anchor_indices']==[],'zero anchors')
            for r in worker['records']:
                worker_ids.append(r['id']);recordmap[r['id']]=r;actualcalls+=r['actual_physical_point_calls']
                check(r['fit_key']==tasks[r['id']]['fit']['key'],'actual fit key')
            check({v['id'] for v in packet['tasks']}=={v['id'] for v in worker['records']},'worker task coverage')
        check(len(set(packet_ids))==len(packet_ids)==len(worker_ids),'duplicate packet task')
    check(set(tasks)==set(recordmap),'overall task coverage')
    frame=c.read(OUT/'predictions.csv');index={key:part for key,part in frame.groupby(['variant_id','family','budget'])}
    for p in plan['plans']:
        preset=presets[p['id']]
        check(p['source_kind']==preset['source_kind'] and p['source_lambdas']==preset['source_lambdas'],'source selection differs from original training')
        for request in p['requests']:
            t0=tasks[request['zero_id']];t1=tasks[request['adapt_id']]
            check(t0['fit']==preset['fit'] and t1['fit']==preset['fit'],'external altered shared model')
            expectednodes=[c.design.inputs(cases[x]) for x in request['case_ids']]
            check(t0['nodes']==expectednodes and t1['nodes']==expectednodes,'actual physical inputs differ')
            for t in [t0,t1]:
                raw=recordmap[t['id']]
                for ep,actual in zip(t['episodes'],raw['episodes']):
                    b=ep['budget'];check(actual['budget']==b,'budget differs')
                    ids=sorted({cases[x].sample_id for x in request['case_ids'] if roles[(cases[x].sample_id,b)]=='anchor'})
                    ai=[i for i,x in enumerate(request['case_ids']) if cases[x].sample_id in ids]
                    check(ep['anchor_condition_ids']==ids and len(ids)==b,'condition anchor budget')
                    check(ep['anchor_indices']==ai,'anchor indices');check(ep['anchor_labels']==[float(cases[request['case_ids'][i]].target) for i in ai],'anchor labels')
                    lam=None if b==0 else preset['source_lambdas'][str(b)]
                    sol=next(q for q in actual['solutions'] if q['lam']==lam)
                    if b==0:check(sol['raw_shift']==0.,'nonzero zero-anchor shift')
                    part=index[(p['id'],request['family'],b)]
                    observed={r.case_id:r for r in part.itertuples()}
                    check(set(observed)==set(request['case_ids']),'output coverage')
                    for cid,r in zip(request['case_ids'],sol['records']):
                        s=cases[cid];out=observed[cid]
                        check(out.sample_id==s.sample_id and out.family==s.family and out.related_cluster==s.related_cluster,'provenance')
                        check(out.role==roles[(s.sample_id,b)],'role')
                        check(abs(out.observed-s.target)<1e-12 and abs(out.observed_log_mu-np.log10(s.mu_cm2Vs))<1e-12,'labels')
                        check(abs(out.observed_log_sigma-np.log10(c.learning.E_CHARGE)-s.target-np.log10(s.mu_cm2Vs))<1e-12,'compatible sigma label')
                        check(out.numeric==r['numeric'] and out.valid==bool(r['valid'] and sol['valid']),'coverage flags')
                        check(out.H_measured_exact is False or out.H_measured_exact==False,'H scenario not exact measurement')
                        if out.numeric:
                            for name in ['predicted','predicted_log_mu','predicted_log_sigma']:
                                check(abs(getattr(out,name)-r[name])<1e-11,'raw worker/output mismatch')
                            check(abs(out.predicted_log_sigma-np.log10(c.learning.E_CHARGE)-out.predicted-out.predicted_log_mu)<1e-10,'Hall identity')
    summary=c.read(OUT/'summary.csv')
    for r in summary.itertuples():
        q=frame[frame.study.eq(r.study)&frame.variant.eq(r.variant)&frame.budget.eq(r.budget)]
        if r.scope=='query':q=q[q.role.eq('query')]
        elif r.scope=='anchor_conditional':q=q[q.role.eq('anchor')]
        elif r.scope=='explicit_caption_query':q=q[q.role.eq('query')&q.family.eq('10.1063/1.5109678')]
        elif r.scope=='context_linked_query':q=q[q.role.eq('query')&q.family.eq('10.1063/1.5017616')]
        else:raise ValueError(r.scope)
        m=metrics(q)
        for k in m:
            v=getattr(r,k+'_MAE');check(pd.isna(v) if m[k] is None else abs(v-m[k])<1e-11,'summary metric')
        check(r.scenario_rows==len(q) and r.conditions==q.sample_id.nunique() and r.DOIs==q.family.nunique(),'summary counts')
    check(len(frame)==6426 and frame.variant_id.nunique()==119,'planned output size')
    check(actualcalls==c.document(OUT/'prediction_complete.json')['actual_new_physical_calls'],'physical call accounting')
    c.verify_hashes(c.document(AUDIT/'protocol_before.json'))
    c.write(AUDIT/'audit.json',dict(created_utc=c.e.now(),passed=all(checks),checks=len(checks),failures=failures,
        variants=119,rows=len(frame),actual_physical_calls=actualcalls,physical_conditions=3,DOIs=2,related_groups=1,
        new_fits=0,new_model_evaluations=0,no_blind_or_predictive_gain_claim=True,goal_complete=False))
    print('External independent audit',len(checks),'failures',len(failures),flush=True)
    assert all(checks),failures[:8]


if __name__=='__main__':main()
