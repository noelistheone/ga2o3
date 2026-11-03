"""Verify report/replays and update the active research goal checkpoint."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase22_prediction as p
s=p.s;BASE=s.BASE;OUT=p.OUT


def main():
    report=OUT/'ML与Sandbox_有限状态学习与预测检验.md';pdf=report.with_suffix('.pdf');source=report.read_text()
    assert not any(ord(x)<32 and x!='\n' for x in source),'unexpected control character in mathematical report'
    assert all(token in source for token in (r'\frac',r'\tau',r'\rho',r'\sigma',r'\sum'))
    claims=json.loads((OUT/'report_claims.json').read_text());assert claims['report_sha256']==s.energy.sha(report) and len(claims['claims'])==261
    for c in claims['claims']:
        f=ROOT/c['path'];assert s.energy.sha(f)==c['sha256'];d=read(f)
        for k,v in c['selector'].items():d=d[d[k].eq(v)]
        assert len(d)==1 and float(d[c['column']].iloc[0])==c['value'],c
    exports=list(OUT.glob('_pdf_build_*/export.json'));assert len(exports)==1;render=json.loads(exports[0].read_text())
    assert not render['warnings'] and render['source_sha256']==s.energy.sha(report) and render['pdf_sha256']==s.energy.sha(pdf)
    text=Path('/tmp/ga2o3_phase22_report.txt').read_text();assert all(x in text for x in ('6880','341469','0.384350','0.373643','0.056484'))
    pages=int(subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout.split('Pages:')[1].splitlines()[0])
    visual=json.loads((OUT/'visual_review.json').read_text());assert visual['passed'] and visual['pdf_sha256']==s.energy.sha(pdf) and visual['pages_inspected']==list(range(1,pages+1))
    s.energy.write(OUT/'rendering_check.json',dict(passed=True,pages=pages,numeric_claims_verified=261,
        report_sha256=s.energy.sha(report),pdf_sha256=s.energy.sha(pdf),math_control_characters_absent=True,math_source_tokens_present=True,no_render_warnings=True))
    a=json.loads((OUT/'audit/contract_audit.json').read_text());c=json.loads((OUT/'cohort_properties/audit.json').read_text());e=json.loads((OUT/'external_leedy/audit.json').read_text())
    assert a['passed'] and c['passed'] and e['passed'] and c['numeric_requests']==6880 and e['numeric_requests']==900
    goal=json.loads((OUT/'system_goal_checked.json').read_text());assert goal['goal']['status']=='active'
    direction=dict(created_utc=s.energy.now(),system_goal_status='active',goal_complete=False,
        direction='trained ML+Sandbox dopant-property prediction; theory informs predictive improvements',
        original29_5_anchors_common18_unchanged=True,primary_variants=10,groups=406,new_shared_optimizer_starts=0,
        numerical_checks=85,API_checks=50,contract_checks=a['checks'],actual_forward_points=a['actual_forward_points'],
        nominal_forward_points=a['nominal_forward_points'],sourceexcluded_bundles=50,sourceexcluded_requests=6880,RT_replays=850,
        knownsource_requests=900,stable_new_internal_gain_established=False,continuum_posterior_convergence_established=False,
        selected_65_129_sensitive_cases=18,new_training_labels=0,new_material_measurements=0,new_DFT_MD=0,GPU_used=False,GPU_authorized='GPU1only',
        literature_scope='primary methods and public process-log acquisition; new logs not decoded or paired,Ge fulltext unavailable',
        next='audit growth logs and same-specimen chemical/Hall labels,calibration independence and source overlap; then freeze a process-informed shared-learning prediction experiment',
        next_theory='docs/ML_SANDBOX_PROCESS_INPUT_NEXT_20261018.md',continuation='CONTINUE_PHASE22_HERE.md')
    s.energy.write(OUT/'goal_direction_check.json',direction)
    cli=json.loads((OUT/'delivery/example_outputs.json').read_text());assert len(cli)==2 and all(x['error'] is None for x in cli)
    s.energy.write(OUT/'delivery/CLI_smoke.json',dict(passed=True,requests=2,example_anchors_are_not_experimental=True,
        input_sha256=s.energy.sha(OUT/'delivery/example_requests.json'),output_sha256=s.energy.sha(OUT/'delivery/example_outputs.json'),
        script_sha256=s.energy.sha(ROOT/'scripts/trainable_sandbox_20261017/predict_properties_v9.py')))
    for filename in ('progress.json','goal_requirements_audit.json'):
        f=BASE/filename;v=json.loads(f.read_text());v.update(current_phase=22,goal_complete=False,active_jobs=[],latest_utc=s.energy.now(),checked_utc=s.energy.now())
        v['phase22']=dict(direction,status='complete_bounded_prediction_checkpoint',frozen=False);v['phase22_running']=dict(status='finished',goal_complete=False)
        v['current_execution']='phase22complete;no new internal prediction gain;next public growth-log and chemically independent paired-data audit'
        v['current_turn']='progress: normalized finite-source prediction,full own-T replays,new public-process input acquisition'
        if filename=='progress.json':
            v.update(status='active',stage='phase22_finite_state_prediction_checkpoint',current_report=str(pdf.relative_to(ROOT)),
                latest_report_directory=str(OUT.relative_to(ROOT)),current_predictor=str((OUT/'delivery/predictor_bundle_v9').relative_to(ROOT)),
                continuation=str((BASE/'CONTINUE_PHASE22_HERE.md').relative_to(ROOT)),next_theory=direction['next_theory'],next=[direction['next'],'retain full predictive scope and strong baselines'])
            item='phase22normalized finite-state source inference;406groups10variants;6880ownT+900source requests;341469checks;negative/sensitive results retained'
            if item not in v['completed']:v['completed'].append(item)
        else:
            v['goal_status']='active';v['objective']=goal['goal']['objective'];v['sufficient_to_mark_goal_complete']=False
            entry=dict(requirement='same-information finite-state inference and full predictive-chain validation',
                evidence=['phase22_predictive_inference/audit/contract_audit.json','phase22_predictive_inference/cohort_properties/audit.json',
                    'phase22_predictive_inference/external_leedy/audit.json','phase22_predictive_inference/goal_direction_check.json'],
                scope='finite model exactness,not stable predictive increment or continuum posterior; next actual process/pairing audit')
            if not any(x['requirement']==entry['requirement'] for x in v['implemented_and_verified']):v['implemented_and_verified'].append(entry)
        s.energy.write(f,v)
    print('phase22 bounded checkpoint verified;goal active;261 numeric claims',flush=True)


if __name__=='__main__':main()
