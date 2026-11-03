"""Independently reconcile derived counts/comparisons and displayed report spans."""
from pathlib import Path
import sys,json,datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from phase29_report_claims import audit
import phase29_data as data
import phase2_energy as e

OUT=data.OUT;REPORT=OUT/'report_v1';TITLE='ML与Sandbox_共享库存与正散射联合预测研究'


def main():
    assert not (REPORT/'independent_audit.json').exists(),'preserve prior report audit before versioned repair'
    assert not (REPORT/'independent_audit_protocol_before.json').exists(),'inspect previous audit attempt before a versioned repair'
    source=OUT/(TITLE+'.md');claims=json.loads((REPORT/'numeric_claims.json').read_text())
    inputs=[source,REPORT/'numeric_claims.json',REPORT/'study_statistics.json',
        REPORT/'descriptive_comparisons.csv',REPORT/'descriptive_interpretation.json',
        REPORT/'selected_models.csv',REPORT/'figure_evidence.json',
        OUT/'prediction/summary.csv',OUT/'prediction/per_family.csv',OUT/'prediction/plan.json']
    e.write(REPORT/'independent_audit_protocol_before.json',dict(created_utc=e.now(),
        code_sha256=e.sha(Path(__file__)),input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs},
        no_model_or_result_changes=True,goal_complete=False))
    checked=audit(source,claims);checks=[];maximum=0.
    def add(name,ok,**kw):checks.append(dict(name=name,passed=bool(ok),**kw))
    def close(name,a,b,tol=1e-10,**kw):
        nonlocal maximum
        x=np.asarray(a,float);y=np.asarray(b,float)
        if x.shape!=y.shape or not np.array_equal(np.isnan(x),np.isnan(y)):
            add(name,False,missingness_or_shape_mismatch=True,**kw);return
        finite=np.isfinite(x)&np.isfinite(y)
        error=float(np.max(abs(x[finite]-y[finite])/(1+abs(y[finite])))) if finite.any() else 0.
        maximum=max(maximum,error);add(name,error<=tol,error=error,**kw)
    protocol=json.loads((REPORT/'protocol_before_report.json').read_text())
    for kind in ('code_hashes','input_hashes'):
        for path,digest in protocol[kind].items():add('report_dependency_unchanged',e.sha(ROOT/path)==digest,path=path)
    add('dynamic_result_values_and_exact_display_spans',checked['passed'] and e.sha(source)==claims['report_sha256'])
    for path,digest in claims['theory_hashes'].items():add('theory_reference_unchanged',e.sha(ROOT/path)==digest,path=path)
    conditions=data.load_conditions();cases=data.load_cases();roles=data.load_roles()
    statistics=json.loads((REPORT/'study_statistics.json').read_text())
    expected=dict(reported_conditions=len(conditions),scenario_rows=len(cases),
        paired_reported_conditions=int(conditions.mobility_eligible.eq(True).sum()),
        paired_scenario_rows=int(cases.mobility_eligible.eq(True).sum()),
        Si_conditions=int(conditions.element.eq('Si').sum()),Sn_conditions=int(conditions.element.eq('Sn').sum()),
        paired_Si=int((conditions.element.eq('Si')&conditions.mobility_eligible.eq(True)).sum()),
        paired_Sn=int((conditions.element.eq('Sn')&conditions.mobility_eligible.eq(True)).sum()),
        common_carrier_conditions=cases.loc[cases.primary_common20.eq(True),'sample_id'].nunique(),
        common_paired_conditions=cases.loc[cases.primary_common20.eq(True)&cases.mobility_eligible.eq(True),'sample_id'].nunique())
    for k,v in expected.items():add('reported_data_scope_matches_qualified_version',statistics[k]==v,key=k)
    for b in (0,1,2):
        r=roles[roles.budget.eq(b)]
        actual=statistics['role_counts'][b]
        add('roles_not_altered_for_report',actual['anchors']==int(r.role.eq('anchor').sum())
            and actual['queries']==int(r.role.eq('query').sum()) and actual['guards']==int(r.role.eq('guard').sum()),budget=b)
    start=json.loads((OUT/'training_started.json').read_text());end=json.loads((OUT/'training_complete.json').read_text())
    elapsed=(datetime.datetime.fromisoformat(end['created_utc'])-datetime.datetime.fromisoformat(start['created_utc'])).total_seconds()
    close('actual_wall_time',[statistics['actual_training_wall_seconds']],[elapsed])
    summary=read(OUT/'prediction/summary.csv');family=read(OUT/'prediction/per_family.csv')
    comparisons=read(REPORT/'descriptive_comparisons.csv')
    interpretation=json.loads((REPORT/'descriptive_interpretation.json').read_text())
    for row in comparisons.itertuples():
        def score(variant):
            selected=summary[summary.variant.eq(variant)&summary.budget.eq(row.budget)&summary.scope.eq('common20')]
            assert len(selected)==1;return float(selected.iloc[0].nH_MAE)
        a=score('JointProcess_internal');b=score(row.control);good=np.isfinite(a) and np.isfinite(b)
        add('comparison_preserves_full_coverage',bool(row.full_coverage)==bool(good),control=row.control,budget=row.budget)
        close('comparison_delta_recomputed',[row.delta_nH_MAE],[a-b if good else np.nan],control=row.control,budget=row.budget)
        if good:
            p=family[family.budget.eq(row.budget)&family.scope.eq('common20')]
            x=p[p.variant.eq('JointProcess_internal')].set_index('family').nH_MAE
            y=p[p.variant.eq(row.control)].set_index('family').nH_MAE
            differences=(x-y).to_numpy(float)
            add('source_direction_counts_recomputed',row.improved_DOIs==int((differences<-1e-10).sum())
                and row.worse_DOIs==int((differences>1e-10).sum())
                and row.numerically_equal_DOIs==int((abs(differences)<=1e-10).sum()),control=row.control,budget=row.budget)
    for item in interpretation['controls']:
        values=comparisons[comparisons.control.eq(item['control'])].sort_values('budget').delta_nH_MAE.to_numpy(float)
        close('interpretation_uses_all_three_budgets',values,[np.nan if v is None else v for v in item['deltas']],control=item['control'])
    models=read(REPORT/'selected_models.csv');plans=json.loads((OUT/'prediction/plan.json').read_text())['plans']
    add('all_selected_plan_records_preserved',len(models)==len(plans)==288)
    for p in plans:
        context='full' if p['outer_held'] is None else p['outer_held']
        sub=models[models.variant.eq(p['variant'])&models.context.eq(context)]
        add('selected_plan_identity_preserved',len(sub)==1 and sub.iloc[0].fit_key==p['fit_key'],plan=p['id'])
    evidence=json.loads((REPORT/'figure_evidence.json').read_text())
    for kind in ('input_hashes','output_hashes'):
        for path,digest in evidence[kind].items():add('scientific_figure_inputs_and_outputs_unchanged',e.sha(ROOT/path)==digest,path=path)
    add('all48_methods_present_in_full_result_tables',len(summary.variant.unique())==48 and len(comparisons)==18)
    result=dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),
        rows=checks,displayed_claims_checked=checked['claims_checked'],maximum_scaled_difference=maximum,
        report_sha256=e.sha(source),no_models_queries_or_scores_changed=True,goal_complete=False)
    e.write(REPORT/'independent_audit.json',result)
    print('Report29 independent audit',result['passed'],len(checks),'claims',checked['claims_checked'],flush=True)
    if not result['passed']:raise RuntimeError('Preserve report provenance/comparison failure')


if __name__=='__main__':main()
