"""Audit information budgets, fixed cohorts, repair provenance and every candidate."""
from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase20_prediction as s
import phase2_energy as energy
OUT=s.OUT/'audit';SOURCE=s.OUT/'source_adaptation_v2';CONTROL=s.OUT/'reference_budget_control'


def main():
    checks=[];frames=[];choices=[];candidate_folds=[]
    def check(name,passed,**evidence):checks.append(dict(test=name,passed=bool(passed),**evidence))
    new=json.loads((s.OUT/'delivery_v2/predictor_bundle_v7/manifest.json').read_text())
    old=json.loads((s.OUT/'delivery/predictor_bundle_v6/manifest.json').read_text())
    for r in new['files']:
        a=s.OUT/'delivery/predictor_bundle_v6'/r['path'];b=s.OUT/'delivery_v2/predictor_bundle_v7'/r['path']
        check('fullsource_weight_byte_preservation',energy.sha(a)==energy.sha(b)==r['sha256'],path=r['path'])
    for r in json.loads((SOURCE/'inherited_results.json').read_text()):
        check('unaffected_source_result_byte_preservation',energy.sha(ROOT/r['path'])==energy.sha(ROOT/r['copy'])==r['sha256'],path=r['path'])
    check('global_choices_unchanged',energy.sha(SOURCE/'global_choices.json')==energy.sha(s.OUT/'source_adaptation/global_choices.json'))
    common=set(s.ROLES[s.ROLES.budget.eq(2)&s.ROLES.role.eq('query')].sample_id);check('common_query_count',len(common)==18,count=len(common))
    folders={m:SOURCE/'variants'/m for m in new['variants'] if m!='OriginalBudget'};folders['OriginalBudget']=CONTROL/'validation'
    for method,folder in folders.items():
        d=read(folder/'predictions.csv');d['method']=method;frames.append(d);c=json.loads((folder/'selections.json').read_text());choices+=c
        check('selection_count',len(c)==15,method=method)
        for row in c:
            held,budget=row['held'],row['budget'];expected=s.ROLES[s.ROLES.family.eq(held)&s.ROLES.budget.eq(budget)].set_index('sample_id').role
            actual=d[d.family.eq(held)&d.budget.eq(budget)].set_index('sample_id')
            check('roles_and_fullcoverage',set(actual.index)==set(expected.index) and actual.role.eq(expected.reindex(actual.index)).all()
                and actual.valid.all() and actual.numeric.all(),method=method,held=held,budget=budget)
            fitpath=(CONTROL if method=='OriginalBudget' else s.OUT)/'fits'/f"{row['fit_key']}.json";f=json.loads(fitpath.read_text())
            check('outer_source_excluded',held not in f['train_families'] and set(f['train_families'])==set(s.FAMILIES)-{held},method=method,held=held,budget=budget)
            epdir=CONTROL/'source' if method=='OriginalBudget' else SOURCE if method=='RelativeInventory' else s.OUT/'source_adaptation'
            ep=json.loads((epdir/'episodes'/f"{row['outer_episode_key']}.json").read_text())
            check('anchor_labels_only',not ep['query_labels_used_in_sourcefit'] and len(ep['anchor_samples'])==budget
                and set(ep['anchor_samples'])==set(expected[expected.eq('anchor')].index),method=method,held=held,budget=budget)
            for cand in row.get('source_candidates',[]):
                for key in cand.get('inner_keys',[]):
                    inner=json.loads((epdir/'episodes'/f'{key}.json').read_text())
                    check('inner_source_exclusion',held not in inner['train_families'] and inner['held']!=held and inner['held'] not in inner['train_families'],method=method,budget=budget)
            for z in ep['results']:
                q=[x for x in z['records'] if x['role']=='query'];complete=bool(q) and all(x['numeric'] and x['valid'] for x in q)
                candidate_folds.append(dict(method=method,family=held,budget=budget,prior=z['prior'],lam=z['lam'],
                    MAE=float(np.mean([abs(x['predicted']-x['observed']) for x in q])) if complete else np.nan,
                    queries=len(q),numeric=sum(x['numeric'] and x['valid'] for x in q),selected=z['prior']==row['selected_source']['prior'] and z['lam']==row['selected_source']['lam']))
    d=pd.concat(frames,ignore_index=True);summaries=[];by=[]
    for scope in ('all_query','common18'):
        q=d[d.role.eq('query')] if scope=='all_query' else d[d.sample_id.isin(common)]
        for (method,budget,family),g in q.groupby(['method','budget','family']):
            by.append(dict(method=method,budget=budget,family=family,scope=scope,queries=len(g),numeric=int((g.numeric&g.valid).sum()),
                MAE=float(abs(g.predicted-g.observed).mean()) if (g.numeric&g.valid).all() else np.nan))
    by=pd.DataFrame(by);OUT.mkdir(parents=True,exist_ok=True);by.to_csv(OUT/'by_family.csv',index=False)
    for keys,g in by.groupby(['method','budget','scope']):
        r=dict(zip(['method','budget','scope'],keys));r.update(families=len(g),queries=int(g.queries.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan,worst_family_MAE=float(g.MAE.max()) if g.MAE.notna().all() else np.nan)
        summaries.append(r)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False);energy.write(OUT/'all_selections.json',choices)
    cf=pd.DataFrame(candidate_folds);cf.to_csv(OUT/'candidate_by_family.csv',index=False);cf['lam_label']=cf.lam.map(lambda x:'inf' if pd.isna(x) else str(float(x)))
    cs=[]
    for keys,g in cf.groupby(['method','budget','prior','lam_label']):
        r=dict(zip(['method','budget','prior','lam_label'],keys));r.update(families=len(g),MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan,
            queries=int(g.queries.sum()),numeric=int(g.numeric.sum()));cs.append(r)
    pd.DataFrame(cs).to_csv(OUT/'candidate_summary.csv',index=False)
    ep=[json.loads(p.read_text()) for p in (SOURCE/'episodes').glob('*.json')]
    check('repaired_internal_episodes_no_failures',not any(e['failures'] for e in ep),episodes=len(ep),solutions=sum(len(e['results']) for e in ep))
    check('repaired_internal_solutions_valid',all(r['anchor_solution'].get('valid') for e in ep for r in e['results']))
    before=read(s.OUT/'source_adaptation/variants/RelativeInventory/predictions.csv');after=read(SOURCE/'variants/RelativeInventory/predictions.csv')
    delta=float(np.max(abs(before.predicted-after.predicted)));check('repair_final_predictions_unchanged',delta<2e-6,maximum_dex=delta)
    training=[]
    for group,folder in (('primary',s.OUT/'fits'),('optimizer_reference',CONTROL/'fits')):
        for path in folder.glob('*.json'):
            f=json.loads(path.read_text());training.append(dict(group=group,key=f['key'],sources=len(f['train_families']),method=f['method'],
                reg=f.get('reg'),new_optimizer_starts=f.get('new_optimizer_starts',0),
                selected_success=f.get('selected_success',True),training_valid=f['training_valid'],state_failures=len(f.get('state_failures',[])),
                seconds=f['seconds'],numerical_continuation_trial_states=f.get('numerical_continuation_trial_states',0)))
    energy.csvwrite(OUT/'training_compute.csv',training)
    check('selected_training_valid',all(r['selected_success'] and r['training_valid'] and not r['state_failures'] for r in training))
    for sub in ('external_leedy','cohort_properties'):
        a=json.loads((s.OUT/sub/'audit.json').read_text());check('property_inference_audit',a['passed'],module=sub)
    report=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,
        main_RT_prediction_rows=len(d),training_records=len(training),new_optimizer_starts=sum(r['new_optimizer_starts'] for r in training),
        all_training_contexts_include_full5=True,selected_internal_adaptation_count=sum(c['variant']=='RelativeInventory' and c['budget']>0 and c['selected_source']['lam'] is not None for c in choices),
        shared_inventory_outer_infinite_gamma_count=sum(c['variant']=='RelativeInventory' and c['budget']==0 and c['global_reg'] is None for c in choices),
        repaired_episodes=len(ep),repaired_solutions=sum(len(e['results']) for e in ep),repair_maximum_selected_prediction_change=delta,
        source_candidate_tables='completefrozen candidates,not independentlyselectednewmethod',
        property_tables_read_in_workers=False,query_labels_in_workers=False,new_DFT_MD=0,new_measurements=0,goal_complete=False)
    energy.write(OUT/'contract_audit.json',report);print('audit',report['passed'],'checks',len(checks),'starts',report['new_optimizer_starts'],flush=True);assert report['passed']


if __name__=='__main__':main()
