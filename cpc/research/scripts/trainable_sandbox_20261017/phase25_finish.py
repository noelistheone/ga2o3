"""Verify the full prediction deliverable; do not mark the research goal complete."""
from pathlib import Path
import sys,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT=e.BASE/'phase25_shared_process'
AUDITS=('data/qualification.json','numerical_gates.json','learning_gates.json','source_gates.json','affine_gates.json',
    'source_smoke_v3/audit.json','interface_smoke/audit.json','training_audit.json','prediction/audit.json',
    'delivery/cohort_replay_v2/audit.json','external_Azizie_v2/audit.json','method_policy/audit.json',
    'physical_scope/audit.json','analysis/audit.json','delivery/final_checks/audit.json',
    'Leedy_label_sensitivity_v2/audit.json','analysis/candidate_diagnosis/audit.json','parent_verification.json')


def main():
    report=OUT/'ML与Sandbox_共享库存学习与跨来源性质预测.md';pdf=report.with_suffix('.pdf')
    claims=json.loads((OUT/'report_claims.json').read_text());assert claims['report_sha256']==e.sha(report)
    cache={}
    for c in claims['claims']:
        p=ROOT/c['path'];assert e.sha(p)==c['sha256']
        if p not in cache:cache[p]=read(p)
        d=cache[p]
        for k,v in c['selector'].items():d=d[d[k].eq(v)]
        assert len(d)==1
        actual=float(d[c['column']].iloc[0])
        assert (not np.isfinite(actual)) if c['value'] is None else actual==c['value']
    for name,h in claims['json_audit_sources'].items():assert e.sha(ROOT/name)==h,name
    assert not any(ord(c)<32 and c!='\n' for c in report.read_text())
    builds=list(OUT.glob('_pdf_build_*/export.json'));assert len(builds)==1
    render=json.loads(builds[0].read_text());assert not render['warnings']
    assert render['source_sha256']==e.sha(report) and render['pdf_sha256']==e.sha(pdf)
    info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout
    pages=int(info.split('Pages:')[1].splitlines()[0]);visual=json.loads((OUT/'visual_review.json').read_text())
    assert visual['passed'] and visual['pdf_sha256']==e.sha(pdf) and visual['pages_inspected']==list(range(1,pages+1))
    for p in AUDITS:
        d=json.loads((OUT/p).read_text());assert d['passed'],p
        for key in ('code_hashes','input_hashes'):
            for name,h in d.get(key,{}).items():assert e.sha(ROOT/name)==h,(p,name)
    e.write(OUT/'rendering_check.json',dict(passed=True,pages=pages,numeric_claims_verified=len(claims['claims']),
        report_sha256=e.sha(report),pdf_sha256=e.sha(pdf),no_render_warnings=True,math_control_characters_absent=True))
    pred=json.loads((OUT/'prediction/execution_complete.json').read_text());replay=json.loads((OUT/'delivery/cohort_replay_v2/audit.json').read_text())
    ext=json.loads((OUT/'external_Azizie_v2/execution_complete.json').read_text());training=json.loads((OUT/'training_audit.json').read_text())
    goal=json.loads((OUT/'system_goal_rechecked_before_results.json').read_text());assert goal['goal']['status']=='active'
    conclusions=json.loads((OUT/'analysis/conclusion_receipt.json').read_text())
    assert conclusions['notes_sha256']==e.sha(OUT/'analysis/predictive_conclusions.md')
    for c in conclusions['facts']:
        p=ROOT/c['path'];assert e.sha(p)==c['sha256']
        if p not in cache:cache[p]=read(p)
        d=cache[p]
        for k,v in c['selector'].items():d=d[d[k].eq(v)]
        assert len(d)==1 and float(d[c['column']].iloc[0])==c['value']
    direction=dict(created_utc=e.now(),system_goal_status='active',goal_complete=False,
        direction='actual trained ML+Sandbox dopant-property prediction;theory serves model design and validation',
        primary='incorporated chemicaldose+traceableconditions->nH,with mu/derivedsigma and evidence-supported candidates in scope',
        original29_5_roles_common18_unchanged=True,new_task_conditions=33,DOIs=6,related_groups=5,
        training_fit_records=training['fit_records'],nonlinear_fits=training['actual_nonlinear_fits'],
        optimizer_starts=training['actual_optimizer_starts'],selected_not_stopped=training['selected_not_stopped'],
        primary_prediction_rows=pred['prediction_rows'],primary_numeric=pred['numeric'],primary_valid=pred['valid'],
        API_requests=replay['requests'],API_points=replay['point_rows'],API_max_log_error=replay['maximum_log_property_difference'],
        external_requests=ext['group_requests'],external_rows=ext['rows'],external_valid=ext['valid'],
        label_sensitivity_rows=456,new_sourcehyperselection_on_external=False,
        actual_learning_object='dose/growth/element conditional KL frozen-inventory odds;not microscopic kinetics or donorbinding',
        same_Hall_and_anchor_budget=True,physical_energy_information_not_free=True,
        no_posterior_calibration_or_blind_laboratory_claim=True,new_material_measurements=0,new_DFT_MD=0,
        GPU_used=False,GPU_authorized='GPU1 only',prediction_conclusions=conclusions,
        next=conclusions['next_predictive_work'],next_theory=conclusions['next_theory'])
    e.write(OUT/'goal_direction_check.json',direction)
    for name in ('progress.json','goal_requirements_audit.json'):
        p=e.BASE/name;d=json.loads(p.read_text())
        d.update(current_phase=25,goal_complete=False,active_jobs=[],latest_utc=e.now(),checked_utc=e.now(),
            protected_files_checked=47785,changed_protected_files=0,current_execution='phase25 prediction checkpoint complete;goal active',
            current_turn='progress:shared property training,matched full prediction/API transfer,theory and source label sensitivity completed',
            next_concrete_work=[direction['next']])
        d['phase25']=dict(direction,status='complete_bounded_prediction_checkpoint',frozen=False)
        d['phase25_running']=dict(status='finished',goal_complete=False)
        if name=='progress.json':
            d.update(status='active',stage='phase25_context_shared_prediction_checkpoint',current_report=str(pdf.relative_to(ROOT)),
                latest_report_directory=str(OUT.relative_to(ROOT)),continuation=str((e.BASE/'CONTINUE_PHASE25_HERE.md').relative_to(ROOT)),
                next_theory=direction['next_theory'],next=[direction['next']])
        else:d.update(goal_status='active',objective=goal['goal']['objective'],sufficient_to_mark_goal_complete=False)
        e.write(p,d)
    print('phase25 full deliverable verified',pages,'pages',len(claims['claims']),'claims;goal active',flush=True)


if __name__=='__main__':main()
