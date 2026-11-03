"""Audit nested choices, allowed information, coverage and full property identities."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase25_training as t
import phase2_energy as e

OUT=t.OUT/'prediction'


def number(v):return np.inf if v is None else float(v)


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
        return 0. if a==b else abs(a-b)
    check('planned_rows',abs(len(df)-3762));check('summary_rows',abs(len(summary)-228))
    check('selections',abs(len(choices)-741));check('plans',abs(len(plans)-247));check('shared_choices',abs(len(shared)-169))
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
                for family,mae in fam.items():family_rows.append(dict(scheme=scheme,variant=variant,budget=int(b),scope=scope,
                    family=family,related_cluster=t.data.cluster(family),nH_MAE=float(mae),queries=int(q.family.eq(family).sum())))
            else:check(key+'/no_partial_MAE',float(pd.notna(s.nH_MAE)))
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
    e.csvwrite(OUT/'per_family.csv',family_rows);e.csvwrite(OUT/'paired_chain_summary.csv',chain);e.csvwrite(OUT/'common_budget_changes.csv',budget_rows)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,
        prediction_rows=len(df),numeric=int(df.numeric.sum()),valid=int(df.valid.sum()),summary_rows=len(summary),
        variants=19,shared_model_retraining_during_prediction=False,query_labels_in_worker_packets=False,
        independent_sigma_labels=0,new_experiments=0,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(OUT/'audit.json',result)
    print('phase25 prediction audit',result['passed'],len(checks),len(df),'rows',flush=True)
    if not result['passed']:raise RuntimeError('Preserve failed prediction audit before repair')


if __name__=='__main__':main()
