"""Freeze a complete predictive experiment; leave the overall research goal active."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
from freeze_phase21 import verify
BASE=energy.BASE;OUT=BASE/'phase22_predictive_inference';DEST=BASE/'artifact_manifest_phase22.json'


def main():
    if DEST.exists():print('Verified phase22',verify(DEST));return
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{i}.json' for i in (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)]
    checked={n:verify(BASE/n) for n in names}
    assert energy.sha(BASE/'artifact_manifest_phase21.json')=='eb1b453405861429211bf0176d97282fd031d8ec70446f63bd0ddc7a5a64c856'
    for rel in ('regime_audit/audit.json','ensemble_gates.json','api_gates/verification.json','audit/contract_audit.json',
                'cohort_properties/audit.json','external_leedy/audit.json','delivery/export_audit.json',
                'delivery/CLI_smoke.json','rendering_check.json','visual_review.json'):
        assert json.loads((OUT/rel).read_text())['passed'],rel
    for rel in ('prediction_protocol.json','regime_audit/protocol.json','cohort_properties/protocol.json','external_leedy/protocol.json'):
        protocol=json.loads((OUT/rel).read_text())
        for key in ('code_hashes','input_hashes'):
            for name,h in protocol.get(key,{}).items():assert energy.sha(ROOT/name)==h,(rel,name)
    for rel in ('ensemble_gates.json','api_gates/verification.json'):
        for name,h in json.loads((OUT/rel).read_text())['code_hashes'].items():assert energy.sha(ROOT/name)==h,(rel,name)
    g=json.loads((OUT/'goal_direction_check.json').read_text());assert g['system_goal_status']=='active' and not g['goal_complete']
    assert g['sourceexcluded_requests']==6880 and g['knownsource_requests']==900 and g['RT_replays']==850
    c=json.loads((OUT/'cohort_properties/audit.json').read_text());assert c['all_requests_numeric'] and c['maximum_RT_replay_error']<2e-6
    b=OUT/'delivery/predictor_bundle_v9';m=json.loads((b/'manifest.json').read_text());assert m['version']==9 and len(m['variants'])==10 and len(m['files'])==6
    for f in m['files']:assert energy.sha(b/f['path'])==f['sha256']
    report=OUT/'ML与Sandbox_有限状态学习与预测检验.md';r=json.loads((OUT/'rendering_check.json').read_text())
    assert r['report_sha256']==energy.sha(report) and r['pdf_sha256']==energy.sha(report.with_suffix('.pdf'))
    assert (OUT/'report_v1_math_escape/invalidated.json').exists() and (OUT/'regime_audit_v1/failed_execution.log').exists()
    energy.write(OUT/'parent_verification.json',dict(passed=True,created_utc=energy.now(),entries=sum(checked.values()),checked=checked,
        parent_sha256={n:energy.sha(BASE/n) for n in names}))
    for filename in ('progress.json','goal_requirements_audit.json'):
        f=BASE/filename;v=json.loads(f.read_text());v['phase22']['frozen']=True;v['protected_files_checked']=sum(checked.values());v['changed_protected_files']=0;energy.write(f,v)
    cont=BASE/'CONTINUE_PHASE22_HERE.md';cont.write_text(cont.read_text()+'\n## 最终冻结\n\nphase22_finish.py及freeze_phase22.py已完成。artifact_manifest_phase22.json为权威清单；全部父SHA逐项通过。系统goal仍active、无后台任务。下一步审计已取得的日志与化学/Hall对应，不重启本阶段已完成作业。\n')
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT/f for f in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md','docs/ML_SANDBOX_GOAL_CHECK_PHASE22_20261018.md',
        'docs/ML_SANDBOX_FINITE_ENSEMBLE_20261018.md','docs/ML_SANDBOX_PROCESS_INPUT_NEXT_20261018.md')]
    files += [BASE/f for f in ('CONTINUE_PHASE22_HERE.md','CONTINUE_PHASE22_WORKING.md','progress.json','goal_requirements_audit.json')]
    snapshot=BASE/'code_snapshot_phase22'
    for f in files:
        dest=snapshot/f.relative_to(ROOT);dest.parent.mkdir(exist_ok=True,parents=True)
        if dest.exists():assert energy.sha(dest)==energy.sha(f)
        else:shutil.copyfile(f,dest)
    frozen=sorted(f for folder in (OUT,snapshot) for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts and 'locks' not in f.parts and f.suffix!='.pyc')
    payload=dict(created_utc=energy.now(),stage='finite_state_source_and_property_prediction_checkpoint',goal_status='active',goal_complete=False,
        variants=10,groups=406,new_shared_optimizer_starts=0,sourceexcluded_requests=6880,knownsource_requests=900,RT_replays=850,
        sourceexcluded_bundles=50,numerical_checks=85,API_checks=50,contract_checks=341469,traceable_table_values=261,
        stable_new_internal_gain_established=False,continuum_posterior_convergence_established=False,independent_blind_validation=False,
        new_training_labels=0,new_DFT_MD=0,new_material_measurements=0,parent_entries_checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names},
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=energy.sha(f)) for f in frozen])
    energy.write(DEST,payload);digest=energy.sha(DEST)
    for filename in ('progress.json','goal_requirements_audit.json'):
        f=BASE/filename;v=json.loads(f.read_text());v['phase22_artifact_count']=len(frozen);v['phase22_manifest_sha256']=digest
        v['phase22'].update(artifact_count=len(frozen),manifest_sha256=digest);energy.write(f,v)
    print('Frozen phase22',len(frozen),'parententries',sum(checked.values()),'manifestSHA256',digest,flush=True)


if __name__=='__main__':main()
