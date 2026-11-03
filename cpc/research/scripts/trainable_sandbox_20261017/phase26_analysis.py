"""All36-method coverage and source-balanced comparisons, without new selection."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase26_registry as t
import phase2_energy as e

BASE=t.OUT;OUT=BASE/'analysis';PRIMARY='ShareEffectiveProcess_internal'


def main():
    assert json.loads((BASE/'prediction/audit.json').read_text())['passed']
    df=read(BASE/'prediction/combined_predictions.csv')
    frozen=read(BASE/'prediction/combined_summary.csv')
    checks=[];families=[];metrics=[]
    def check(name,error,tol=0.):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    for (scheme,variant,budget),g in df.groupby(['scheme','variant','budget']):
        for scope in ('all_query','common20'):
            q=g[g.role.eq('query')]
            if scope=='common20':q=q[q.common_query.eq(True)]
            tag=f'{scheme}/{variant}/{budget}/{scope}'
            expected=20 if scope=='common20' else {0:33,1:26,2:20}[int(budget)]
            check(tag+'/condition_count',abs(len(q)-expected))
            check(tag+'/unique_conditions',float(q.sample_id.duplicated().any()))
            complete=bool(len(q)==expected and q.numeric.all() and q.valid.all())
            row=dict(scheme=scheme,variant=variant,budget=int(budget),scope=scope,
                conditions=len(q),numeric=int(q.numeric.sum()),valid=int(q.valid.sum()),
                coverage_complete=complete,DOIs=q.family.nunique(),related_groups=5,
                DOI_macro_nH_MAE=None,group_macro_nH_MAE=None,worst_DOI_nH_MAE=None,
                primary_method=variant==PRIMARY,independent_sigma_labels=0)
            fr=[]
            for family,f in q.groupby('family'):
                ok=bool(f.numeric.all() and f.valid.all())
                r=dict(scheme=scheme,variant=variant,budget=int(budget),scope=scope,family=family,
                    related_cluster=t.data.cluster(family),conditions=len(f),numeric=int(f.numeric.sum()),valid=int(f.valid.sum()),
                    coverage_complete=ok,nH_MAE=float(abs(f.predicted-f.observed).mean()) if ok else None)
                fr.append(r);families.append(r)
            if complete:
                fd=pd.DataFrame(fr);row.update(DOI_macro_nH_MAE=float(fd.nH_MAE.mean()),
                    group_macro_nH_MAE=float(fd.groupby('related_cluster').nH_MAE.mean().mean()),worst_DOI_nH_MAE=float(fd.nH_MAE.max()))
            saved=frozen[frozen.scheme.eq(scheme)&frozen.variant.eq(variant)&frozen.budget.eq(budget)&frozen.scope.eq(scope)]
            check(tag+'/unique_frozen_summary',abs(len(saved)-1))
            if len(saved)==1:
                if complete:check(tag+'/frozen_MAE',abs(row['DOI_macro_nH_MAE']-saved.nH_MAE.iloc[0]),1e-12)
                else:check(tag+'/no_partial_MAE',float(pd.notna(saved.nH_MAE.iloc[0])))
            mu=q[q.mobility_eligible.eq(True)]
            row.update(mu_pairs=len(mu),mu_DOIs=mu.family.nunique())
            mu_ok=bool(len(mu) and mu.numeric.all() and mu.valid.all())
            for label,pc,sc in [('native','predicted_log_mu','predicted_log_sigma'),('CT','predicted_log_mu_CT','predicted_log_sigma_CT')]:
                row[label+'_mu_MAE']=float(abs(mu[pc]-mu.observed_log_mu).groupby(mu.family).mean().mean()) if mu_ok else None
                obs_sigma=np.log10(1.602176634e-19)+mu.observed+mu.observed_log_mu
                row[label+'_derived_sigma_MAE']=float(abs(mu[sc]-obs_sigma).groupby(mu.family).mean().mean()) if mu_ok else None
            metrics.append(row)
    check('all36_metrics',abs(len(metrics)-432))
    e.csvwrite(OUT/'all_method_metrics.csv',metrics);e.csvwrite(OUT/'all_method_per_family.csv',families)
    fdf=pd.DataFrame(families);comparisons=[]
    for scheme in ('DOI','cluster'):
        for budget in (0,1,2):
            d=fdf[fdf.scheme.eq(scheme)&fdf.budget.eq(budget)&fdf.scope.eq('common20')]
            a=d[d.variant.eq(PRIMARY)].set_index('family')
            for control in sorted(set(d.variant)-{PRIMARY}):
                b=d[d.variant.eq(control)].set_index('family')
                j=a[['nH_MAE','related_cluster','coverage_complete']].join(b[['nH_MAE','coverage_complete']],rsuffix='_control',how='outer')
                ok=bool(len(j)==6 and j.coverage_complete.all() and j.coverage_complete_control.all() and j.nH_MAE.notna().all() and j.nH_MAE_control.notna().all())
                delta=j.nH_MAE-j.nH_MAE_control;groups=delta.groupby(j.related_cluster).mean()
                comparisons.append(dict(scheme=scheme,budget=budget,primary=PRIMARY,control=control,
                    full_coverage=ok,DOIs=len(j),related_groups=len(groups),
                    delta_DOI_macro=float(delta.mean()) if ok else None,
                    delta_group_macro=float(groups.mean()) if ok else None,
                    better_DOIs=int((delta < -1e-8).sum()) if ok else None,
                    tied_DOIs=int((abs(delta)<=1e-8).sum()) if ok else None,
                    worse_DOIs=int((delta > 1e-8).sum()) if ok else None,
                    maximum_DOI_deterioration=float(delta.max()) if ok else None,
                    leave_one_group_delta_min=float(min(groups.drop(g).mean() for g in groups.index)) if ok else None,
                    leave_one_group_delta_max=float(max(groups.drop(g).mean() for g in groups.index)) if ok else None,
                    grade='descriptive small-source paired comparison;not confidenceinterval or multiplicity-adjusted test'))
    e.csvwrite(OUT/'primary_matched_comparisons.csv',comparisons)
    choices=json.loads((BASE/'prediction/selections.json').read_text());fitrows=[];policies=[]
    for c in choices:
        policies.append(dict(scheme=c['scheme'],held_unit=c['held_unit'],variant=c['variant'],budget=c['budget'],
            shared_reg=c['reg'],source_lambda=c['selected_source']['lam'],no_source_update=c['selected_source']['lam'] is None,
            final_context=c['held_unit'] is None))
        if c['budget']!=0:continue
        f=json.loads((ROOT/c['fit_path']).read_text());sv=np.asarray(f.get('weighted_prediction_jacobian_singular_values',[]));n=len(f['parameters']) if f['parameters'] is not None else 0
        rank=f.get('prediction_jacobian_numeric_rank');full=bool(rank==n and len(sv) and sv[-1]>0)
        fitrows.append(dict(scheme=c['scheme'],held_unit=c['held_unit'],variant=c['variant'],method=f['method'],
            evaluation_method=f['evaluation_method'],reference_fallback=bool(f.get('reference_fallback')),shared_reg=c['reg'],
            parameters=f['parameters'],parameter_count=n,jacobian_rank=rank,rank_deficient=not full,
            finite_full_rank_condition_number=float(sv[0]/sv[-1]) if full else None,
            training_valid=f['training_valid'],selected_success=f['selected_success'],training_nH_MAE=f.get('training_nH_MAE'),
            fit_key=f['key'],fit_sha256=e.sha(ROOT/c['fit_path']),numeric_rank_is_not_structural_identifiability=True))
    e.csvwrite(OUT/'selected_fit_diagnostics.csv',fitrows);e.csvwrite(OUT/'selected_policies.csv',policies)
    costs=[];failures=[]
    for i in range(4):
        packet=json.loads((BASE/f'prediction/outputs_{i}.json').read_text());check(f'worker{i}/no_denied_reads',len(packet['denied_reads']))
        for r in packet['records']:
            sols=[s for ep in r['episodes'] for s in ep['solutions']]
            costs.append(dict(task_id=r['id'],kind=r['kind'],actual_point_calls=r['actual_physical_point_calls'],
                actual_source_vectors=r['actual_source_vectors'],seconds=r['seconds'],point_failures=len(r['point_failures']),
                candidate_solutions=len(sols),invalid_candidate_solutions=sum(not s['valid'] for s in sols)))
            failures.extend(dict(task_id=r['id'],kind=r['kind'],**v) for v in r['point_failures'])
    e.csvwrite(OUT/'source_costs.csv',costs);e.write(OUT/'source_trial_failures.json',failures)
    train=json.loads((BASE/'training_audit.json').read_text())
    result=dict(passed=all(x['passed'] for x in checks),checks=len(checks),rows=checks,
        primary=PRIMARY,primary_fixed_before_outer_results=True,outer_method_reselected=False,
        metric_rows=len(metrics),comparison_rows=len(comparisons),selected_fit_rows=len(fitrows),source_tasks=len(costs),
        new_point_rows=3366,combined_point_rows=len(df),all_original_rows_retained=True,
        combined_numeric=int(df.numeric.sum()),combined_valid=int(df.valid.sum()),
        incomplete_metric_rows=sum(not r['coverage_complete'] for r in metrics),
        actual_source_point_evaluations=sum(r['actual_point_calls'] for r in costs),
        actual_source_vectors=sum(r['actual_source_vectors'] for r in costs),point_failure_records=len(failures),
        invalid_candidate_solutions=sum(r['invalid_candidate_solutions'] for r in costs),
        training_fit_records=train['fit_records'],training_optimizer_starts=train['actual_optimizer_starts'],
        training_invalid=train['training_invalid'],selected_not_stopped=train['selected_not_stopped'],
        no_new_fits_or_measurements=True,cost_scope='new17 source calls include empirical calls;old19 costs separately inherited',
        goal_complete=False,code_sha256=e.sha(Path(__file__)))
    e.write(OUT/'audit.json',result)
    print('phase26 all36 predictive analysis',result['passed'],len(metrics),'metrics',len(comparisons),'comparisons',flush=True)
    if not result['passed']:raise RuntimeError('Retain failed result-analysis audit before repair')


if __name__=='__main__':main()
