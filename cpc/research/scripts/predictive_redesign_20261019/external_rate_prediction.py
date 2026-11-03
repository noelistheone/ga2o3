"""One frozen conditional external cohort, with no fitting on its query labels."""
from pathlib import Path
import argparse,sys,time,os
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase32_prediction as runtime

DATA=c.BASE/'external_rate_cohort_v1'
OUT=DATA/'prediction'
PRESETS={
    'phase29':c.OLD/'prediction/full_deployment_presets.json',
    'phase30':c.OUT/'full_training_presets.json',
    'phase31':c.BASE/'phase31_rate_redesign/prediction/full_training_presets.json',
    'phase32':c.BASE/'phase32_rate_controls/prediction/full_training_presets.json'}


def declare():
    target=OUT/'design_before_prediction.json';assert not target.exists()
    qualification=c.document(DATA/'qualification.json');c.verify_hashes(qualification)
    files=[DATA/'protocol_before_transcription.json',DATA/'qualification.json',
           DATA/'conditions3.csv',DATA/'scenarios24.csv',DATA/'episodes.csv',
           PRESETS['phase29'],PRESETS['phase30'],
           c.BASE/'phase31_rate_redesign/prediction/design_before_prediction.json',
           c.BASE/'phase32_rate_controls/prediction/design_before_prediction.json']
    code=[Path(__file__),Path(runtime.__file__),runtime.RUNTIME,
          c.OLD_SCRIPTS/'phase29_source_runtime.py',Path(c.__file__)]
    value=dict(created_utc=c.e.now(),primary='phase31:ChargeRatioSpectrum__chain_safe',
        variants_by_study=dict(phase29=48,phase30=39,phase31=16,phase32=16),planned_variants=119,
        physical_conditions=3,DOIs=2,conservative_related_groups=1,input_scenarios=24,
        primary_endpoint='zeroanchor conditional nH;paired predictedmu andcompatible sigma also reported',
        split_pair_quality=True,zero_packets_have_no_labels=True,
        source_lambdas_from_original_training_only=True,no_external_query_mu_for_source_updates=True,
        permitted_budgets='Rafique0/1/2,Feng0/1;Feng1andRafique2have noquery andareanchorconditionalonly',
        metric='equal scenarios percondition thenDOI;also eachcondition andworstscenario;no calibrated input probabilities',
        no_new_training_or_external_model_selection=True,
        not_blind=True,prior_corpus_exposure_and_missing_input_scope_disclosed=True,
        workers=2,code_hashes=c.file_hashes(code),input_hashes=c.file_hashes(files),goal_complete=False)
    c.write(target,value);return value


def load_presets():
    values=[]
    for study,path in PRESETS.items():
        for p in c.document(path):
            values.append(dict(study=study,variant=p['variant'],id=study+':'+p['variant'],preset=p))
    assert len(values)==119 and len({v['id'] for v in values})==119
    return values


def make_plan(presets,cases,roles):
    plans=[];zero={};adapt={}
    for item in presets:
        p=item['preset'];requests=[]
        for family,part in cases.groupby('family',sort=True):
            nodes=[c.design.inputs(r) for r in part.itertuples()]
            episodes=[]
            for b in sorted(roles.loc[roles.family.eq(family),'budget'].unique()):
                b=int(b);role=roles[roles.family.eq(family)&roles.budget.eq(b)]
                ids=sorted(role.loc[role.role.eq('anchor'),'sample_id'])
                ai=[i for i,r in enumerate(part.itertuples()) if r.sample_id in ids]
                episodes.append(dict(budget=b,anchor_condition_ids=ids,anchor_indices=ai,
                    anchor_labels=[float(part.iloc[i].target) for i in ai]))
            t0=dict(fit=p['fit'],kind=p['source_kind'],nodes=nodes,episodes=[episodes[0]])
            assert episodes[0]['budget']==0 and episodes[0]['anchor_labels']==[]
            t0['id']=c.digest(t0);zero[t0['id']]=t0
            t1=dict(fit=p['fit'],kind=p['source_kind'],nodes=nodes,episodes=episodes[1:])
            t1['id']=c.digest(t1);adapt[t1['id']]=t1
            requests.append(dict(family=family,case_ids=part.case_id.tolist(),zero_id=t0['id'],adapt_id=t1['id']))
        plans.append(dict(id=item['id'],study=item['study'],variant=item['variant'],
            source_kind=p['source_kind'],source_lambdas=p['source_lambdas'],
            actual_method=p['fit']['method'],fit_key=p['fit']['key'],requests=requests))
    value=dict(plans=plans,zero_tasks=list(zero.values()),adapt_tasks=list(adapt.values()))
    c.write(OUT/'plan_before_runtime.json',value);return value


def attach(plan,records0,records1,cases,roles):
    cb={r.case_id:r for r in cases.itertuples()}
    rb={(r.sample_id,int(r.budget)):r.role for r in roles.itertuples()}
    result=[];checks=[];failures=[]
    for p in plan['plans']:
        for request in p['requests']:
            episodes=records0[request['zero_id']]['episodes']+records1[request['adapt_id']]['episodes']
            for episode in episodes:
                b=episode['budget'];lam=None if b==0 else p['source_lambdas'][str(b)]
                sol=next(q for q in episode['solutions'] if q['lam']==lam)
                checks.append(len(sol['records'])==len(request['case_ids']))
                for cid,raw in zip(request['case_ids'],sol['records']):
                    a=cb[cid];q=dict(raw);q.pop('diagnostics',None);q.pop('source_transport_derivative',None)
                    q.update(study=p['study'],variant=p['variant'],variant_id=p['id'],
                        actual_method=p['actual_method'],fit_key=p['fit_key'],case_id=cid,
                        sample_id=a.sample_id,family=a.family,related_cluster=a.related_cluster,
                        pairing_grade=a.pairing_grade,budget=b,role=rb[(a.sample_id,b)],
                        source_lambda=lam,source_raw_shift=sol.get('raw_shift'),
                        source_solution_valid=sol['valid'],
                        valid=bool(raw['valid'] and sol['valid']),observed=float(a.target),
                        observed_log_mu=float(np.log10(a.mu_cm2Vs)),
                        observed_log_sigma=float(np.log10(c.learning.E_CHARGE)+a.target+np.log10(a.mu_cm2Vs)),
                        mobility_eligible=True,scenario_index=a.scenario_index,
                        scenario_weight=a.scenario_weight,H_measured_exact=False,
                        conditional_external_scope=True,not_blind=True)
                    if b==0:checks.append(sol.get('raw_shift')==0.)
                    if q['numeric']:
                        err=abs(q['predicted_log_sigma']-np.log10(c.learning.E_CHARGE)-q['predicted']-q['predicted_log_mu'])
                        checks.append(err<1e-10)
                    result.append(q)
    assert len(result)==119*(6*3+18*2)
    c.e.csvwrite(OUT/'predictions.csv',result)
    c.write(OUT/'readout_audit.json',dict(passed=all(checks),checks=len(checks),
        rows=len(result),variants=119,no_external_query_label_training=True,
        zero_source_shift_fixed=True,goal_complete=False))
    assert all(checks)
    return pd.DataFrame(result)


def summarize(frame):
    rows=[];percondition=[]
    for (study,variant,budget),part in frame.groupby(['study','variant','budget']):
        scopes={'query':part[part.role.eq('query')],
                'explicit_caption_query':part[part.role.eq('query')&part.family.eq('10.1063/1.5109678')],
                'context_linked_query':part[part.role.eq('query')&part.family.eq('10.1063/1.5017616')],
                'anchor_conditional':part[part.role.eq('anchor')]}
        for scope,q in scopes.items():
            metrics=c.metrics(q.to_dict('records'),group_equal=False) if len(q) else dict(nH=None,mu=None,rho=None)
            rows.append(dict(study=study,variant=variant,budget=int(budget),scope=scope,
                scenario_rows=len(q),conditions=q.sample_id.nunique(),DOIs=q.family.nunique(),
                related_groups=q.related_cluster.nunique(),numeric=int(q.numeric.eq(True).sum()),
                valid=int(q.valid.eq(True).sum()),nH_MAE=metrics['nH'],mu_MAE=metrics['mu'],sigma_MAE=metrics['rho']))
        for sample,q in part.groupby('sample_id'):
            m=c.metrics(q.to_dict('records'),group_equal=False)
            valid=bool(q.numeric.eq(True).all() and q.valid.eq(True).all())
            percondition.append(dict(study=study,variant=variant,budget=int(budget),sample_id=sample,
                role=q.role.iloc[0],scenario_rows=len(q),numeric=int(q.numeric.eq(True).sum()),
                valid=int(q.valid.eq(True).sum()),nH_MAE=m['nH'],mu_MAE=m['mu'],sigma_MAE=m['rho'],
                worst_nH_error=float(abs(q.predicted-q.observed).max()) if valid else None,
                worst_mu_error=float(abs(q.predicted_log_mu-q.observed_log_mu).max()) if valid else None))
    c.e.csvwrite(OUT/'summary.csv',rows);c.e.csvwrite(OUT/'per_condition.csv',percondition)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['declare','run'])
    args=parser.parse_args()
    if args.action=='declare':print('Frozen external design',declare()['planned_variants'],'variants',flush=True);return
    design=c.document(OUT/'design_before_prediction.json');c.verify_hashes(design)
    assert not (OUT/'prediction_started.json').exists(),'Inspect original watcher/PID;never duplicate on timeout'
    for name in ['phase31_rate_redesign','phase32_rate_controls']:
        assert c.document(c.BASE/name/'prediction/independent_result_audit.json')['passed']
    presets=load_presets()
    c.write(OUT/'actual_presets_before_prediction.json',dict(created_utc=c.e.now(),
        preset_hashes=c.file_hashes(PRESETS.values()),presets=presets,
        no_external_label_selection=True,goal_complete=False))
    cases=c.read(DATA/'scenarios24.csv');roles=c.read(DATA/'episodes.csv')
    plan=make_plan(presets,cases,roles);before=time.perf_counter()
    c.write(OUT/'prediction_started.json',dict(created_utc=c.e.now(),pid=os.getpid(),workers=2))
    # Physically separate packets mean zero-anchor workers cannot read any labels.
    records0,workers0=runtime.runtime(plan['zero_tasks'],OUT/'zero_runtime',[None],2)
    records1,workers1=runtime.runtime(plan['adapt_tasks'],OUT/'source_runtime',c.design.LAM,2)
    frame=attach(plan,records0,records1,cases,roles);summarize(frame)
    c.verify_hashes(design)
    c.write(OUT/'prediction_complete.json',dict(created_utc=c.e.now(),variants=119,rows=len(frame),
        actual_new_physical_calls=sum(r['actual_physical_point_calls'] for r in [*records0.values(),*records1.values()]),
        seconds=time.perf_counter()-before,new_shared_model_fits=0,new_material_samples=0,
        distinct_public_conditions=3,conditional_missing_inputs=True,not_blind=True,goal_complete=False))
    print('External119variant conditional predictions complete;interpretation still required',flush=True)


if __name__=='__main__':main()
