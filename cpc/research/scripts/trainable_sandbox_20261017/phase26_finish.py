"""Verify the completed phase26 deliverable against actual files, not intent."""
from pathlib import Path
import sys,json,subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
from phase26_report import AUDITS
import phase2_energy as e

OUT=e.BASE/'phase26_pool_share'


def verify_claim(c,cache):
    path=ROOT/c['path'];assert e.sha(path)==c['sha256'],path
    if path not in cache:cache[path]=read(path)
    d=cache[path]
    for k,v in c['selector'].items():d=d[d[k].eq(v)]
    assert len(d)==1,(path,c['selector'])
    x=float(d[c['column']].iloc[0]);expected=c['value']
    assert (not np.isfinite(x)) if expected is None else x==expected,(path,c['column'],x,expected)


def main():
    for name in (*AUDITS,'delivery/documentation_audit.json','theory/visual_review.json','theory_scope_readback.json','information_audit/audit.json','fallback_equivalence/audit.json','inner_domain_diagnosis/audit.json'):
        d=json.loads((OUT/name).read_text());assert d['passed'],name
        for kind in ('code_hashes','input_hashes'):
            for path,h in d.get(kind,{}).items():assert e.sha(ROOT/path)==h,(name,path)
    for name in ('training_protocol.json','pre_evaluation_protocol.json','post_prediction_predeclared.json',
        'prediction/protocol.json','delivery/cohort_replay/protocol.json','external_Azizie/protocol_before_transfer.json',
        'analysis_predeclared.json','fallback_equivalence/protocol_before_comparison.json','inner_domain_diagnosis/protocol_before_replay.json'):
        d=json.loads((OUT/name).read_text())
        for kind in ('code_hashes','input_hashes','hashes','inherited_parent_fit_sha256','files'):
            for path,h in d.get(kind,{}).items():assert e.sha(ROOT/path)==h,(name,path)
    report=OUT/'ML与Sandbox_绝对份额学习与性质预测.md';pdf=report.with_suffix('.pdf')
    claims=json.loads((OUT/'report_claims.json').read_text());assert claims['report_sha256']==e.sha(report)
    assert e.sha(ROOT/claims['theory_path'])==claims['theory_sha256']
    cache={}
    for c in claims['claims']:verify_claim(c,cache)
    for path,h in claims['json_audit_sources'].items():assert e.sha(ROOT/path)==h,path
    conclusions=json.loads((OUT/'analysis/conclusion_receipt.json').read_text())
    assert conclusions['notes_sha256']==e.sha(OUT/'analysis/predictive_conclusions.md')
    assert conclusions['facts'],'Actual evidence-specific numeric conclusions are required'
    for c in conclusions['facts']:verify_claim(c,cache)
    for path,h in conclusions.get('supplemental_json_sources',{}).items():assert e.sha(ROOT/path)==h,path
    builds=list(OUT.glob('_pdf_build_*/export.json'));assert len(builds)==1
    render=json.loads(builds[0].read_text());assert render['source_sha256']==e.sha(report) and render['pdf_sha256']==e.sha(pdf)
    assert not render['warnings'];assert not any(ord(c)<32 and c!='\n' for c in report.read_text())
    info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout
    pages=int(info.split('Pages:')[1].splitlines()[0]);visual=json.loads((OUT/'visual_review.json').read_text())
    assert visual['passed'] and visual['pdf_sha256']==e.sha(pdf) and visual['pages_inspected']==list(range(1,pages+1))
    e.write(OUT/'rendering_check.json',dict(passed=True,pages=pages,numeric_claims_verified=len(claims['claims']),
        conclusion_facts_verified=len(conclusions['facts']),report_sha256=e.sha(report),pdf_sha256=e.sha(pdf),
        no_render_warnings=True,all_pages_actually_inspected=True,goal_complete=False))
    goal=json.loads((OUT/'goal_direction_reconfirmed.json').read_text());assert goal['system_goal_status']=='active'
    manifest=json.loads((OUT/'delivery/predictor_bundle/manifest.json').read_text())
    for f in manifest['files']:assert e.sha(OUT/'delivery/predictor_bundle'/f['path'])==f['sha256'],f['path']
    for name in ('learning_gates.json','learning_gate_failure_v1/learning_gates.json'):
        if (OUT/name).exists():assert not json.loads((OUT/name).read_text())['passed']
    assert (OUT/'learning_gate_failure_v1').is_dir()
    train=json.loads((OUT/'training_audit.json').read_text());pred=json.loads((OUT/'prediction/audit.json').read_text())
    replay=json.loads((OUT/'delivery/cohort_replay/audit.json').read_text());ext=json.loads((OUT/'external_Azizie/execution_complete.json').read_text())
    direction=dict(created_utc=e.now(),system_goal_status='active',goal_complete=False,
        direction='actual trained ML+Sandbox dopant-property prediction;theory serves prediction and validation',
        original29_roles_common18_preserved=True,conditions=33,DOIs=6,related_groups=5,
        primary='ShareEffectiveProcess_internal',main_variants=36,training_fit_records=train['fit_records'],
        actual_nonlinear_fits=train['actual_nonlinear_fits'],optimizer_starts=train['actual_optimizer_starts'],
        training_invalid=train['training_invalid'],selected_not_stopped=train['selected_not_stopped'],
        new_property_rows=pred['prediction_rows'],combined_property_rows=7128,
        API_requests=replay['requests'],API_point_rows=replay['point_rows'],API_max_log_error=replay['maximum_log_property_difference'],
        external_rows=ext['rows'],external_numeric=ext['numeric'],external_valid=ext['valid'],
        actual_learning_object='shared absolute logit atomshare;not measured kinetics/donorbinding',
        shared_policy_and_frozen_source_budget_reported=True,all_failures_and_unqualified_models_retained=True,
        physical_information_cost_separate=True,new_mu_supervision=False,independent_sigma_labels=0,
        no_new_DFT_MD=True,no_new_material_measurements=True,GPU_used=False,GPU_authorized='GPU1only',
        prediction_conclusions=conclusions,next=conclusions['next_predictive_work'])
    e.write(OUT/'goal_direction_check.json',direction)
    for name in ('progress.json','goal_requirements_audit.json'):
        path=e.BASE/name;d=json.loads(path.read_text());d['phase26']=dict(direction,status='complete_bounded_prediction_checkpoint',frozen=False)
        d.update(current_phase=26,goal_complete=False,latest_utc=e.now(),current_execution='phase26 actual prediction deliverable verified;overall goal active',
            current_turn='progress:actual sharedtraining,full36 prediction/API replay,freeze-transfer,theory/results report and adverse evidence checked',
            next_concrete_work=[direction['next']])
        if name=='progress.json':d.update(status='active',current_report=str(pdf.relative_to(ROOT)),continuation=str((e.BASE/'CONTINUE_PHASE26_HERE.md').relative_to(ROOT)))
        else:d.update(goal_status='active',objective=goal['system_goal_objective'],sufficient_to_mark_goal_complete=False)
        e.write(path,d)
    print('phase26 actual full deliverable verified',pages,'pages',len(claims['claims']),'values;goal active',flush=True)


if __name__=='__main__':main()
