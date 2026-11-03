"""Audit nested choices, allowed information, coverage and full property identities."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase27_registry as t
import phase2_energy as e

OUT=t.OUT/'prediction'


def number(v):return np.inf if v is None else float(v)


def independent_nested_audit(check,error,tasks,refs,plans,choices,shared):
    """Reconstruct CV scores from actual predictions and immutable labels."""
    zero=read(OUT/'inner_zero_predictions.csv');labels=t.TABLE.set_index('sample_id');raw={}
    for i in range(4):
        for r in json.loads((OUT/f'outputs_{i}.json').read_text())['records']:raw[r['id']]=r
    for c in shared:
        for candidate in c['candidates']:
            scores=[]
            for held,saved_score in zip(candidate['inner_units'],candidate['inner_scores']):
                train=[u for u in c['train_units'] if u!=held]
                key,path=t.fit_path(c['scheme'],train,c['method'],candidate['reg'])
                fit=json.loads(path.read_text());qualified=bool(fit['training_valid'] and fit['selected_success'])
                g=zero[zero.fit_key.eq(key)&zero.inner_held_unit.eq(held)]
                name='independent_shared/'+key+'/'+held
                if not qualified:
                    check(name+'/unqualified_zero_rows',len(g));score=np.inf
                else:
                    expected=t.TABLE[t.TABLE[t.unit_column(c['scheme'])].eq(held)]
                    check(name+'/all_conditions',float(set(g.sample_id)!=set(expected.sample_id)))
                    check(name+'/no_duplicate_conditions',float(g.sample_id.duplicated().any()))
                    for r in g.itertuples():check(name+'/'+r.sample_id+'/frozen_target',abs(r.observed-float(labels.loc[r.sample_id].target)),1e-12)
                    score=float(abs(g.predicted-g.observed).groupby(g.family).mean().mean()) if len(g) and g.numeric.all() and g.valid.all() else np.inf
                check(name+'/inner_score',error(score,saved_score),1e-10);scores.append(score)
            check('independent_candidate/'+t.digest([c['scheme'],c['held'],c['method'],candidate['reg']]),
                error(float(np.mean(scores)),candidate['score']),1e-10)
    role=t.ROLES.set_index(['family','budget','sample_id']).role;cache={}
    def query_rows(task_id,budget,lam):
        key=(task_id,budget,lam)
        if key not in cache:
            ep=next(r for r in raw[task_id]['episodes'] if r['budget']==budget)
            solution=next(r for r in ep['solutions'] if r['lam']==lam);ref=refs[task_id];rows=[]
            for r in t.TABLE[t.TABLE.family.eq(ref['family'])].itertuples():
                if role[r.family,budget,r.sample_id]!='query':continue
                point=solution['records'][ref['sample_node'][r.sample_id]]
                rows.append((r.family,bool(point['numeric'] and point['valid']),point.get('predicted'),r.target))
            cache[key]=rows
        return cache[key]
    for choice in choices:
        b=choice['budget']
        if not b:continue
        for trial in choice['source_candidates']:
            by_unit={}
            for request in choice['inner']:
                by_unit.setdefault(request['unit'],[]).extend(query_rows(request['task_id'],b,trial['lam']))
            scores=[]
            for unit,rows in by_unit.items():
                if not rows or not all(r[1] for r in rows):score=np.inf
                else:
                    families={}
                    for family,valid,pred,target in rows:families.setdefault(family,[]).append(abs(pred-target))
                    score=float(np.mean([np.mean(v) for v in families.values()]))
                name='independent_source/'+t.digest([choice['scheme'],choice['held_unit'],choice['variant'],b,trial['lam'],unit])
                check(name,error(score,trial['inner_unit_scores'][unit]),1e-10);scores.append(score)
            check('independent_source_candidate/'+t.digest([choice['scheme'],choice['held_unit'],choice['variant'],b,trial['lam']]),
                error(float(np.mean(scores)),trial['score']),1e-10)


def main():
    completed=json.loads((OUT/'execution_complete.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    for k in ('code_hashes','input_hashes'):
        for name,h in protocol[k].items():assert e.sha(ROOT/name)==h,name
    refs=json.loads((OUT/'parent_reference.json').read_text());plans=json.loads((OUT/'plans.json').read_text())
    choices=json.loads((OUT/'selections.json').read_text());shared=json.loads((OUT/'shared_choices.json').read_text())
    tasks=json.loads((OUT/'tasks.json').read_text())['tasks'];df=read(OUT/'predictions.csv');summary=read(OUT/'summary.csv')
    checks=[]
    def check(name,value,tol=0.):checks.append(dict(name=name,error=float(value),tolerance=tol,passed=bool(value<=tol)))
    def error(a,b):
        a,b=number(a),number(b)
        if np.isnan(a) or np.isnan(b):
            return 0. if np.isnan(a) and np.isnan(b) else np.inf
        return 0. if a==b else abs(a-b)
    revised=json.loads((OUT.parent/'prediction_audit_resume_protocol_v2.json').read_text())
    for group in ('code_hashes','input_hashes'):
        for path,digest in revised[group].items():assert e.sha(ROOT/path)==digest,path
    for label,a,b,same in [('missing_both',np.nan,np.nan,True),('missing_left',np.nan,0.,False),
        ('missing_right',0.,np.nan,False),('missing_vs_infinity',np.nan,np.inf,False),
        ('same_finite',1.25,1.25,True),('different_finite',1.25,1.3,False),
        ('same_nonfinite_source_score',None,np.inf,True),('opposite_infinities',np.inf,-np.inf,False)]:
        check('comparison_semantics/'+label,float((error(a,b)==0.)!=same))
    check('planned_rows',abs(len(df)-2376));check('summary_rows',abs(len(summary)-144))
    check('selections',abs(len(choices)-468));check('plans',abs(len(plans)-156));check('shared_choices',abs(len(shared)-78))
    for task in tasks:
        name=task['id'];ref=refs[name]
        check(name+'/no_labels_in_nodes',sum('target' in n or 'observed' in n or 'family' in n for n in task['nodes']))
        for ep in task['episodes']:
            check(name+f"/{ep['budget']}/anchor_budget",abs(len(ep['anchor_labels'])-ep['budget']))
            roles=t.ROLES[t.ROLES.family.eq(ref['family'])&t.ROLES.budget.eq(ep['budget'])].set_index('sample_id').role
            allowed=t.TABLE[t.TABLE.family.eq(ref['family'])&t.TABLE.sample_id.isin(roles[roles.eq('anchor')].index)].sort_values(['incorporated_cm3','sample_id'])
            expected=[float(r.target) for r in allowed.itertuples()]
            check(name+f"/{ep['budget']}/only_allowed_labels",max([0.]+[abs(a-b) for a,b in zip(ep['anchor_labels'],expected)]))
            check(name+f"/{ep['budget']}/anchor_indices",float(ep['anchor_indices']!=[ref['sample_node'][r.sample_id] for r in allowed.itertuples()]))
        f=json.loads((ROOT/ref['fit_path']).read_text())
        check(name+'/fit_checksum',float(e.sha(ROOT/ref['fit_path'])!=ref['fit_sha256']))
        check(name+'/held_family_not_trained',float(ref['family'] in f['train_families']))
        if f['scheme']=='cluster':
            check(name+'/related_group_not_trained',float(t.data.cluster(ref['family']) in f['train_units']))
    for c in shared:
        trial=min(c['candidates'],key=lambda r:(round(number(r['score']),10),-np.inf if r['reg'] is None else -r['reg']))
        check('shared/'+t.digest([c['scheme'],c['held'],c['method']]),float(trial['reg']!=c['reg']))
        for candidate in c['candidates']:
            check('shared_score/'+t.digest([c['scheme'],c['held'],c['method'],candidate['reg']]),
                error(candidate['score'],np.mean([number(x) for x in candidate['inner_scores']])),1e-10)
    role_index=t.ROLES.set_index(['family','budget','sample_id']).role
    for row in df.itertuples():
        label=f'{row.scheme}/{row.variant}/{row.budget}/{row.sample_id}'
        check(label+'/same_role',float(role_index[row.family,row.budget,row.sample_id]!=row.role))
        common=role_index[row.family,2,row.sample_id]=='query'
        check(label+'/same_common_queries',float(common!=row.common_query))
        if row.numeric and row.valid:
            check(label+'/finite',float(not np.isfinite([row.predicted,row.predicted_log_mu,row.predicted_log_sigma]).all()))
            check(label+'/derived_sigma',abs(row.predicted_log_sigma-(np.log10(1.602176634e-19)+row.predicted+row.predicted_log_mu)),2e-9)
            check(label+'/derived_sigma_CT',abs(row.predicted_log_sigma_CT-(np.log10(1.602176634e-19)+row.predicted+row.predicted_log_mu_CT)),2e-9)
    for choice in choices:
        candidates=choice['source_candidates'];best=min(candidates,key=lambda c:(round(number(c['score']),10),-np.inf if c['lam'] is None else -c['lam']))
        check('source_choice/'+t.digest([choice['scheme'],choice['held_unit'],choice['variant'],choice['budget']]),
            float(best['lam']!=choice['selected_source']['lam']))
        if choice['held_unit'] is not None:
            for entry in choice['inner']:check('source_inner_exclusion/'+t.digest([choice['fit_key'],entry['task_id']]),float(entry['unit']==choice['held_unit']))
    family_rows=[];chain=[];budget_rows=[]
    for (scheme,variant,b),g in df.groupby(['scheme','variant','budget']):
        for scope in ('all_query','common20'):
            q=g[g.role.eq('query')]
            if scope=='common20':q=q[q.common_query.eq(True)]
            expected=20 if scope=='common20' else {0:33,1:26,2:20}[int(b)]
            key=f'{scheme}/{variant}/{b}/{scope}';check(key+'/coverage_count',abs(len(q)-expected))
            saved=summary[summary.scheme.eq(scheme)&summary.variant.eq(variant)&summary.budget.eq(b)&summary.scope.eq(scope)]
            check(key+'/one_summary',abs(len(saved)-1))
            if len(saved)!=1:continue
            s=saved.iloc[0];valid=bool(q.numeric.all() and q.valid.all())
            if valid:
                fam=np.abs(q.predicted-q.observed).groupby(q.family).mean()
                check(key+'/macro_MAE',error(s.nH_MAE,float(fam.mean())),1e-12)
            else:check(key+'/no_partial_MAE',float(pd.notna(s.nH_MAE)))
            for family,fg in q.groupby('family'):
                complete=bool(fg.numeric.all() and fg.valid.all())
                family_rows.append(dict(scheme=scheme,variant=variant,budget=int(b),scope=scope,
                    family=family,related_cluster=t.data.cluster(family),queries=len(fg),numeric=int(fg.numeric.sum()),valid=int(fg.valid.sum()),
                    coverage_complete=complete,nH_MAE=float(abs(fg.predicted-fg.observed).mean()) if complete else None))
            mu=q[q.mobility_eligible.eq(True)].copy();entry=dict(scheme=scheme,variant=variant,budget=int(b),scope=scope,
                paired_conditions=len(mu),paired_DOIs=mu.family.nunique(),independent_sigma_labels=0)
            for name in ('physical_or_native','common_CT'):
                pc='predicted_log_mu' if name=='physical_or_native' else 'predicted_log_mu_CT'
                ps='predicted_log_sigma' if name=='physical_or_native' else 'predicted_log_sigma_CT'
                usable=bool(len(mu) and mu.numeric.all() and mu.valid.all())
                entry[name+'_mu_MAE']=float(np.abs(mu[pc]-mu.observed_log_mu).groupby(mu.family).mean().mean()) if usable else None
                observed_sigma=np.log10(1.602176634e-19)+mu.observed+mu.observed_log_mu
                entry[name+'_derived_sigma_MAE']=float(np.abs(mu[ps]-observed_sigma).groupby(mu.family).mean().mean()) if usable else None
            chain.append(entry)
    fdf=pd.DataFrame(family_rows)
    for (scheme,variant,family),g in fdf[fdf.scope.eq('common20')].groupby(['scheme','variant','family']):
        by=g.set_index('budget').nH_MAE
        if set(by.index)=={0,1,2}:
            budget_rows.append(dict(scheme=scheme,variant=variant,family=family,related_cluster=t.data.cluster(family),
                MAE0=float(by[0]),MAE1=float(by[1]),MAE2=float(by[2]),delta_1_minus_0=float(by[1]-by[0]),delta_2_minus_1=float(by[2]-by[1])))
    for i in range(4):
        packet=json.loads((OUT/f'outputs_{i}.json').read_text())
        check(f'worker{i}/label_files_denied_reads',len(packet['denied_reads']))
        check(f'worker{i}/no_query_labels',float(packet['runtime_query_labels']))
    # Verify the inherited controls as complete rows, not a newly selected
    # favorable subset. The model-specific native property columns are kept.
    parent=e.BASE/'phase26_pool_share/prediction'
    old=read(parent/'combined_predictions.csv');combined=read(OUT/'combined_predictions.csv')
    oldsummary=read(parent/'combined_summary.csv');combined_summary=read(OUT/'combined_summary.csv')
    check('combined_rows',abs(len(combined)-9504));check('combined_summaries',abs(len(combined_summary)-576))
    check('combined_variants',abs(combined.variant.nunique()-48))
    keys=['scheme','variant','budget','family','sample_id']
    inherited=combined[combined.variant.isin(old.variant.unique())].set_index(keys).sort_index()
    original=old.set_index(keys).sort_index()
    check('inherited_row_identity',float(not inherited.index.equals(original.index)))
    for col in ('observed','predicted','predicted_log_mu','predicted_log_sigma','predicted_log_mu_CT','predicted_log_sigma_CT'):
        x=inherited[col].to_numpy(float);y=original[col].to_numpy(float)
        check('inherited_nonfinite_identity/'+col,float(not(
            np.array_equal(np.isnan(x),np.isnan(y)) and np.array_equal(np.isposinf(x),np.isposinf(y))
            and np.array_equal(np.isneginf(x),np.isneginf(y)))))
        check('inherited_values/'+col,float(np.nanmax(abs(x-y))),2e-12)
    for col in ('role','numeric','valid','common_query'):
        check('inherited_flags/'+col,float(not inherited[col].equals(original[col])))
    for row in oldsummary.itertuples():
        g=combined_summary[combined_summary.scheme.eq(row.scheme)&combined_summary.variant.eq(row.variant)&
            combined_summary.budget.eq(row.budget)&combined_summary.scope.eq(row.scope)]
        check('inherited_summary/'+str(row.Index),abs(len(g)-1))
        if len(g)==1:check('inherited_MAE/'+str(row.Index),error(g.nH_MAE.iloc[0],row.nH_MAE),2e-12)
    independent_nested_audit(check,error,tasks,refs,plans,choices,shared)
    e.csvwrite(OUT/'per_family.csv',family_rows);e.csvwrite(OUT/'paired_chain_summary.csv',chain);e.csvwrite(OUT/'common_budget_changes.csv',budget_rows)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        prediction_rows=len(df),numeric=int(df.numeric.sum()),valid=int(df.valid.sum()),summary_rows=len(summary),
        variants=12,shared_model_retraining_during_prediction=False,query_labels_in_worker_packets=False,
        independent_sigma_labels=0,new_experiments=0,goal_complete=False,audit_version=2,
        original_failure_preserved=True,no_predictions_models_labels_or_scores_changed=True,
        original_failure_sha256=e.sha(OUT.parent/'prediction_audit_failure_v1/audit.json'),
        resume_protocol_sha256=e.sha(OUT.parent/'prediction_audit_resume_protocol_v2.json'),
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(OUT/'audit.json',result)
    print('phase27 prediction audit V2',result['passed'],len(checks),len(df),'rows',flush=True)
    if not result['passed']:raise RuntimeError('Preserve failed prediction audit before repair')


if __name__=='__main__':main()
