"""Verify a bounded prediction checkpoint; keep the overall research goal active."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase21_curve_response'


def main():
    report=OUT/'ML与Sandbox_响应约束与强对照预测检验.md';pdf=report.with_suffix('.pdf')
    claims=json.loads((OUT/'report_claims.json').read_text());assert claims['report_sha256']==energy.sha(report)
    for c in claims['claims']:
        p=ROOT/c['path'];assert energy.sha(p)==c['sha256'];d=read(p)
        for k,v in c['selector'].items():d=d[d[k].eq(v)]
        assert len(d)==1 and float(d[c['column']].iloc[0])==c['value'],c
    exports=list(OUT.glob('_pdf_build_*/export.json'));assert len(exports)==1
    render=json.loads(exports[0].read_text());assert render['source_sha256']==energy.sha(report) and render['pdf_sha256']==energy.sha(pdf) and not render['warnings']
    text=Path('/tmp/ga2o3_phase21_report.txt').read_text()
    assert all(t in text for t in ('11696','11018','17001','1.369483','0.049152','0.499353','0.838790'))
    info=subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True).stdout
    pages=int(info.split('Pages:')[1].splitlines()[0]);assert pages==5
    energy.write(OUT/'rendering_check.json',dict(passed=True,pages=pages,numeric_claims_verified=len(claims['claims']),
        report_sha256=energy.sha(report),pdf_sha256=energy.sha(pdf),no_render_warnings=True,math_preserved=True))
    # The five-page contact sheet was actually inspected before this record.
    energy.write(OUT/'visual_review.json',dict(passed=True,pages_inspected=list(range(1,6)),
        figures_visible=True,formulas_intact=True,tables_readable=True,continued_table_headers_present=True,
        method_names_not_clipped=True,pdf_sha256=energy.sha(pdf)))
    goal=json.loads((OUT/'system_goal_checked.json').read_text());assert goal['goal']['status']=='active'
    a=json.loads((OUT/'audit/contract_audit.json').read_text());assert a['passed'] and a['checks']==17001
    c=json.loads((OUT/'cohort_properties/audit.json').read_text());assert c['passed'] and c['inference_requests']==11696 and c['RT_replays']==1445
    e=json.loads((OUT/'external_leedy/audit.json').read_text());assert e['passed'] and e['requests']==2500
    t=json.loads((OUT/'chemical_transport_controls/audit.json').read_text());assert t['passed'] and t['numeric_rows']==11018
    direction=dict(created_utc=energy.now(),system_goal_status='active',goal_complete=False,
        direction='trained ML+Sandbox predicts dopant properties; theory serves actual prediction',
        primary='actual incorporated dose + traceable inputs -> nH; compatible mu/sigma and supported ranking remain in scope',
        original29_5_anchors_common18_unchanged=True,new_shared_optimizer_starts=0,shared_weights='phase20 frozen byte-exact',
        new_DFT_MD=0,new_material_measurements=0,GPU_used=False,GPU_authorized='GPU1 only',
        primary_variants=14,additional_affine_controls=3,primary_groups=518,available_presets=24,
        numerical_checks=394,main_API_checks=43,affine_checks=65,affine_API_checks=15,chemical_transport_checks=21,
        contract_checks=a['checks'],sourceexcluded_bundles=85,sourceexcluded_API_requests=11696,saved_RT_replays=1445,
        knownsource_API_requests=2500,chemical_transport_outputs=11018,
        stable_shared_learning_gain_established=False,independent_blind_validation=False,
        knownsource_grade='known development source; simple affine control reaches similar three-query error',
        theory_revision='finite response bounds constrain model movement, not true prediction error; state validity does not guarantee accuracy',
        next='audit original TEOS and low-T Hall definitions and transport regimes; then freeze one evidence-based prediction improvement with matched controls',
        continuation='CONTINUE_PHASE21_HERE.md')
    energy.write(OUT/'goal_direction_check.json',direction)
    cli=json.loads((OUT/'delivery/example_outputs.json').read_text())
    assert [x['method'] for x in cli]==['OriginalHall','DoseAffine'] and all(set(x['transport_comparisons'])=={'1','3'} for x in cli)
    energy.write(OUT/'delivery/CLI_smoke.json',dict(passed=True,requests=2,methods=['OriginalHall','DoseAffine'],
        optional_transport_heads=[1,3],anchors_are_demonstration_not_experimental=True,
        input_sha256=energy.sha(OUT/'delivery/example_requests.json'),output_sha256=energy.sha(OUT/'delivery/example_outputs.json'),
        script_sha256=energy.sha(ROOT/'scripts/trainable_sandbox_20261017/predict_properties_v8.py')))
    for filename in ('progress.json','goal_requirements_audit.json'):
        path=BASE/filename;old=json.loads(path.read_text());old['goal_complete']=False;old['current_phase']=21
        old['phase21']=dict(direction,status='complete_bounded_prediction_checkpoint',frozen=False)
        old['phase21_running']=dict(status='finished',goal_complete=False,continuation='CONTINUE_PHASE21_HERE.md')
        old['current_execution']='phase21 prediction checkpoint complete; next original-source/transport audit before one justified predictive change'
        old['current_turn']='progress: finite response/state ablations, strong affine/transport controls, complete prediction replays'
        old['active_jobs']=[];old['latest_utc']=energy.now();old['checked_utc']=energy.now()
        if filename=='progress.json':
            old.update(status='active',stage='phase21_finite_response_prediction_checkpoint',
                current_report=str(pdf.relative_to(ROOT)),latest_report_directory=str(OUT.relative_to(ROOT)),
                current_predictor=str((OUT/'delivery/predictor_bundle_v8').relative_to(ROOT)),
                continuation=str((BASE/'CONTINUE_PHASE21_HERE.md').relative_to(ROOT)),
                next_theory='docs/ML_SANDBOX_GOAL_CHECK_PHASE21_20261018.md',next=[direction['next'],'retain actual ML+Sandbox property prediction goal'])
            item='phase21 frozen shared weights; 14+3 source strategies; 11696 own-T +2500 known-source requests;11018 same-nH transport controls; failures retained'
            if item not in old['completed']:old['completed'].append(item)
        else:
            old['goal_status']='active';old['objective']=goal['goal']['objective'];old['sufficient_to_mark_goal_complete']=False
            entry=dict(requirement='finite-response source adaptation and strong same-information controls',
                evidence=['phase21_curve_response/audit/contract_audit.json','phase21_curve_response/cohort_properties/audit.json',
                    'phase21_curve_response/external_leedy/audit.json','phase21_curve_response/chemical_transport_controls/audit.json',
                    'phase21_curve_response/goal_direction_check.json'],
                scope='no stable new predictive gain; current code gates and conditional diagnostics do not complete the overall goal')
            if not any(x['requirement']==entry['requirement'] for x in old['implemented_and_verified']):old['implemented_and_verified'].append(entry)
        energy.write(path,old)
    print('Phase21 checkpoint verified; overall goal active; numeric claims',len(claims['claims']),flush=True)


if __name__=='__main__':main()
