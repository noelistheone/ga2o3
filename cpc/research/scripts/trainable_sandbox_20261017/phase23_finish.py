"""Finalize a bounded source-prediction checkpoint while retaining the active ML+Sandbox goal."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase23_process_inputs'


def main():
    report=OUT/'ML与Sandbox_过程输入与新来源预测.md';pdf=report.with_suffix('.pdf');source=report.read_text()
    assert not any(ord(c)<32 and c!='\n' for c in source)
    assert all(t in source for t in (r'\exp',r'\ge',r'\dot',r'\kappa',r'\eta',r'\mathrm{cm}^{-3}'))
    claims=json.loads((OUT/'report_claims.json').read_text());assert claims['report_sha256']==e.sha(report) and len(claims['claims'])==407
    cache={}
    for c in claims['claims']:
        p=ROOT/c['path'];assert e.sha(p)==c['sha256']
        if p not in cache:cache[p]=read(p)
        d=cache[p]
        for k,v in c['selector'].items():d=d[d[k].eq(v)]
        assert len(d)==1 and float(d[c['column']].iloc[0])==c['value']
    builds=list(OUT.glob('_pdf_build_*/export.json'));assert len(builds)==1;render=json.loads(builds[0].read_text())
    assert not render['warnings'] and render['source_sha256']==e.sha(report) and render['pdf_sha256']==e.sha(pdf)
    info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout;pages=int(info.split('Pages:')[1].splitlines()[0])
    visual=json.loads((OUT/'visual_review.json').read_text());assert visual['passed'] and visual['pdf_sha256']==e.sha(pdf) and visual['pages_inspected']==list(range(1,pages+1))
    text=Path('/tmp/ga2o3_phase23_report.txt').read_text();assert all(t in text for t in ('7320','83985','0.361363','0.609933','1830'))
    e.write(OUT/'rendering_check.json',dict(passed=True,pages=pages,numeric_claims_verified=407,report_sha256=e.sha(report),pdf_sha256=e.sha(pdf),
        math_control_characters_absent=True,math_source_tokens_present=True,no_render_warnings=True))
    raw=json.loads((OUT/'process_audit/audit.json').read_text());gates=json.loads((OUT/'process_checks/gates.json').read_text());audit=json.loads((OUT/'source_prediction/audit/contract_audit.json').read_text());parent=json.loads((OUT/'parent_verification.json').read_text())
    assert raw['passed'] and gates['passed'] and audit['passed'] and parent['passed']
    assert raw['verified_log_to_Hall_identities']==0 and audit['numeric_requests']==7320 and audit['summary_rows']==1830
    assert len(gates['checks'])==160 and audit['checks']==83985 and parent['entries']==43295
    goal=json.loads((OUT/'system_goal_checked.json').read_text());assert goal['goal']['status']=='active'
    direction=dict(created_utc=e.now(),system_goal_status='active',goal_complete=False,direction='trainable ML+Sandbox predicts doped-material properties;theory serves predictive development',
        primary='actual incorporated chemistry+traceable inputs->nH;compatible mu/derivedsigma and supported ranking in scope',
        original29_5_anchors_common18_unchanged=True,published_Hall_conditions=12,precise_new_chemical_pairs=0,rounded_same_specimen_conditions=2,
        process_proxy_conditions=12,process_logs=23,process_archive_entries=13584,temperature_identity_candidates=13,verified_log_to_Hall_identities=0,
        fullsource_methods=34,relatedsource_excluded_methods=27,requests=7320,numeric_requests=7320,retained_refusals=0,summary_rows=1830,
        process_checks=160,contract_checks=83985,new_shared_training=0,new_shared_training_labels=0,new_source_hyperselection=0,
        independent_sigma_labels=0,new_temperature_curves=0,stable_new_internal_gain_established=False,
        conditional_positive='FieldAnchorSafe full5 B primary2 .361363;not retained when related source excluded .609933',
        theory_revision='source-input identity and chemical calibration precede process learning;steady chemistry cannot identify a kinetic clock',
        independent_blind_validation=False,independent_new_lab=False,actual_chemical_validation_of12=False,new_DFT_MD=0,new_material_measurements=0,GPU_used=False,GPU_authorized='GPU1 only',
        next='audit SIMS stack layer/control correspondence and independent chemistry;freeze one supported process-to-chemical/shared-state prediction experiment with strong sameinput baselines',
        next_theory='docs/ML_SANDBOX_CHEMICAL_PROCESS_NEXT_20261018.md',continuation='CONTINUE_PHASE23_HERE.md')
    e.write(OUT/'goal_direction_check.json',direction)
    for name in ('progress.json','goal_requirements_audit.json'):
        p=BASE/name;v=json.loads(p.read_text());v.update(current_phase=23,goal_complete=False,active_jobs=[],latest_utc=e.now(),checked_utc=e.now(),protected_files_checked=43295,changed_protected_files=0)
        v['phase23']=dict(direction,status='complete_bounded_prediction_checkpoint',frozen=False);v['phase23_running']=dict(status='finished',goal_complete=False)
        v['current_execution']='phase23complete;7320 frozen property forecasts and process input audit;no new stable internal predictive gain'
        v['current_turn']='progress: public process logs audit and actual newsource frozen property forecasts;all failures retained'
        v['next_concrete_work']=[direction['next'],'retain primary dopantpropertyprediction scope and all strong old controls']
        if name=='progress.json':
            v.update(status='active',stage='phase23_process_inputs_prediction_checkpoint',current_report=str(pdf.relative_to(ROOT)),latest_report_directory=str(OUT.relative_to(ROOT)),
                continuation=str((BASE/'CONTINUE_PHASE23_HERE.md').relative_to(ROOT)),next_theory=direction['next_theory'],next=[direction['next'],'theory and input inference serve real property predictions'])
            item='phase23 23publiclogs/13584entries;7320frozennewsourcepropertyforecasts;83985checks;coarsechemistry/processproxyseparated;conditionalgainandfailuresretained'
            if item not in v['completed']:v['completed'].append(item)
        else:
            v['goal_status']='active';v['objective']=goal['goal']['objective'];v['sufficient_to_mark_goal_complete']=False
            item=dict(requirement='traceable process inputs and actual newsource dopantpropertyforecasting',
                evidence=['phase23_process_inputs/process_audit/source_contract.json','phase23_process_inputs/source_prediction/audit/contract_audit.json','phase23_process_inputs/goal_direction_check.json'],
                scope='two rounded chemistry conditions and12 process proxies;not precise12pair material validation or stable learning gain')
            if not any(x['requirement']==item['requirement'] for x in v['implemented_and_verified']):v['implemented_and_verified'].append(item)
        e.write(p,v)
    print('phase23 bounded prediction checkpoint verified;goal active;407claims;7pages',flush=True)


if __name__=='__main__':main()
