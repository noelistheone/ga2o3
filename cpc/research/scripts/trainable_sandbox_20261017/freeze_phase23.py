"""Freeze a completed predictive checkpoint; never rewrite an existing scientific ledger."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from freeze_phase21 import verify
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase23_process_inputs';DEST=BASE/'artifact_manifest_phase23.json'


def main():
    if DEST.exists():print('Verified phase23',verify(DEST));return
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{i}.json' for i in (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)]
    checked={name:verify(BASE/name) for name in names};assert sum(checked.values())==43295
    assert e.sha(BASE/'artifact_manifest_phase22.json')=='af18f7c8f98ff111b6724a2592b7273e96eaa4bfa26dbb7e8b5d1c7bd5223cb1'
    for rel in ('process_audit/audit.json','process_checks/gates.json','source_prediction/audit/contract_audit.json','parent_verification.json','rendering_check.json','visual_review.json'):
        assert json.loads((OUT/rel).read_text())['passed'],rel
    for rel in ('process_audit/protocol.json','source_prediction/protocol.json','process_checks/gates.json','source_prediction/audit/contract_audit.json'):
        value=json.loads((OUT/rel).read_text())
        for key in ('code_hashes','input_hashes'):
            for name,h in value.get(key,{}).items():assert e.sha(ROOT/name)==h,(rel,name)
    goal=json.loads((OUT/'goal_direction_check.json').read_text());assert goal['system_goal_status']=='active' and not goal['goal_complete']
    assert goal['requests']==7320 and goal['numeric_requests']==7320 and goal['new_shared_training']==0 and goal['verified_log_to_Hall_identities']==0
    render=json.loads((OUT/'rendering_check.json').read_text());report=OUT/'ML与Sandbox_过程输入与新来源预测.md'
    assert render['report_sha256']==e.sha(report) and render['pdf_sha256']==e.sha(report.with_suffix('.pdf')) and render['numeric_claims_verified']==407
    for rel in ('report_v1_syntax_failure/invalidated.json','report_v2_font_failure/invalidated.json'):assert (OUT/rel).exists()
    e.write(OUT/'parent_verification.json',dict(passed=True,created_utc=e.now(),entries=sum(checked.values()),checked=checked,parent_sha256={n:e.sha(BASE/n) for n in names}))
    for name in ('progress.json','goal_requirements_audit.json'):
        p=BASE/name;value=json.loads(p.read_text());value['phase23']['frozen']=True;value['protected_files_checked']=sum(checked.values());value['changed_protected_files']=0;e.write(p,value)
    p=BASE/'CONTINUE_PHASE23_HERE.md';p.write_text(p.read_text()+'\n## 最终冻结\n\nphase23_finish.py和freeze_phase23.py已完成，artifact_manifest_phase23.json为权威清单；父阶段43295项SHA逐项通过。整体goal仍active，无后台科学作业；后续用新版本目录继续独立化学约束的预测研发，不重写本阶段。\n')
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md','docs/ML_SANDBOX_GOAL_CHECK_PHASE23_20261018.md',
        'docs/ML_SANDBOX_NEW_SOURCE_INPUT_PROTOCOL_20261018.md','docs/ML_SANDBOX_PROCESS_BALANCE_20261018.md','docs/ML_SANDBOX_CHEMICAL_PROCESS_NEXT_20261018.md')]
    files += [BASE/p for p in ('progress.json','goal_requirements_audit.json','CONTINUE_PHASE23_HERE.md','CONTINUE_PHASE23_WORKING.md')]
    snapshot=BASE/'code_snapshot_phase23'
    for p in files:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(exist_ok=True,parents=True)
        if target.exists():assert e.sha(target)==e.sha(p),target
        else:shutil.copyfile(p,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'locks' not in p.parts and p.suffix!='.pyc')
    payload=dict(created_utc=e.now(),stage='process_inputs_and_frozen_newsource_property_prediction_checkpoint',goal_status='active',goal_complete=False,
        process_logs=23,archive_entries=13584,verified_log_to_Hall_identities=0,rounded_same_specimen_chemistry_conditions=2,process_proxy_conditions=12,
        newsource_requests=7320,numeric_requests=7320,fullsource_methods=34,relatedsource_excluded_methods=27,contract_checks=83985,process_checks=160,
        traceable_table_values=407,new_shared_training=0,new_shared_training_labels=0,new_source_hyperselection=0,
        stable_new_internal_gain_established=False,independent_blind_validation=False,independent_new_lab=False,
        new_DFT_MD=0,new_material_measurements=0,parent_entries_checked=checked,parent_sha256={n:e.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=e.sha(p)) for p in frozen])
    e.write(DEST,payload);digest=e.sha(DEST)
    for name in ('progress.json','goal_requirements_audit.json'):
        p=BASE/name;v=json.loads(p.read_text());v['phase23_artifact_count']=len(frozen);v['phase23_manifest_sha256']=digest
        v['phase23'].update(artifact_count=len(frozen),manifest_sha256=digest);e.write(p,v)
    print('Frozen phase23',len(frozen),'parententries',sum(checked.values()),'manifestSHA256',digest,flush=True)


if __name__=='__main__':main()
