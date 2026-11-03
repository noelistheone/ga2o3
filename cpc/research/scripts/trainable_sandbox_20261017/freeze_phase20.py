"""Freeze the prediction checkpoint and preserve every earlier result ledger."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase20_source_inventory';DEST=BASE/'artifact_manifest_phase20.json'


def verify(path):
    value=json.loads(path.read_text());rows=value['files'] if isinstance(value,dict) else value
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad;return len(rows)


def main():
    if DEST.exists():print('Verified phase20',verify(DEST));return
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{p}.json' for p in (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19)]
    checked={n:verify(BASE/n) for n in names};assert energy.sha(BASE/'artifact_manifest_phase19.json')=='3eb8550504afef15f6fba980abced2fb8b29007bb21b29a8895419c263691faf'
    for rel in ('numerical_gates.json','learning_gates.json','inverse_repair/regression_v2.json',
            'inverse_repair/learning_gates_v2.json','audit/contract_audit.json','cohort_properties/audit.json',
            'external_leedy/audit.json','rendering_check.json','visual_review.json'):
        assert json.loads((OUT/rel).read_text())['passed'],rel
    protocols=('probe_protocol.json','prediction_protocol.json','source_adaptation/source_protocol.json',
        'source_adaptation_v2/source_protocol.json','reference_budget_control/protocol.json',
        'delivery/export_protocol.json','delivery_v2/export_protocol.json','cohort_properties/protocol.json','external_leedy/protocol.json')
    for rel in protocols:
        p=json.loads((OUT/rel).read_text())
        for kind in ('code_hashes','input_hashes'):
            for name,h in p.get(kind,{}).items():assert energy.sha(ROOT/name)==h,(rel,name)
    report=OUT/'ML与Sandbox_相对库存学习与预测适应检验.md';pdf=report.with_suffix('.pdf')
    rendering=json.loads((OUT/'rendering_check.json').read_text());assert rendering['pdf_sha256']==energy.sha(pdf) and rendering['report_sha256']==energy.sha(report)
    manifest=json.loads((OUT/'delivery_v2/predictor_bundle_v7/manifest.json').read_text());assert manifest['version']==7
    assert len(manifest['variants'])+len(manifest['energy_methods'])==17 and len(manifest['files'])==8
    for f in manifest['files']:assert energy.sha(OUT/'delivery_v2/predictor_bundle_v7'/f['path'])==f['sha256']
    g=json.loads((OUT/'goal_direction_check.json').read_text());assert g['system_goal_status']=='active' and not g['goal_complete']
    a=json.loads((OUT/'audit/contract_audit.json').read_text());assert a['training_records']==353 and a['new_optimizer_starts']==462
    c=json.loads((OUT/'cohort_properties/audit.json').read_text());assert c['inference_requests']==c['numeric_requests']==6900 and c['RT_replays']==870
    e=json.loads((OUT/'external_leedy/audit.json').read_text());assert e['rows']==e['numeric_rows']==1730
    energy.write(OUT/'parent_verification.json',dict(passed=True,created_utc=energy.now(),entries=sum(checked.values()),
        checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names}))
    for filename in ('progress.json','goal_requirements_audit.json'):
        path=BASE/filename;value=json.loads(path.read_text());value['phase20']['frozen']=True
        value['protected_files_checked']=sum(checked.values());value['changed_protected_files']=0;energy.write(path,value)
    continuation=BASE/'CONTINUE_PHASE20_HERE.md'
    continuation.write_text(continuation.read_text()+'\n## 最终冻结状态\n\nphase20_finish.py和freeze_phase20.py已完成验收。artifact_manifest_phase20.json为本轮权威文件清单；父清单逐项校验，无变化。系统goal仍active；没有后台任务。下一步按上述输入范围响应约束方案执行，不重启已完成的phase20训练。\n')
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_CHECK_PHASE20_20261018.md','docs/ML_SANDBOX_RELATIVE_INVENTORY_20261018.md',
        'docs/ML_SANDBOX_CURVE_RESPONSE_NEXT_20261018.md')]
    files += [BASE/p for p in ('CONTINUE_PHASE20_HERE.md','progress.json','goal_requirements_audit.json')]
    snapshot=BASE/'code_snapshot_phase20'
    for p in files:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p),target
        else:shutil.copyfile(p,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'locks' not in p.parts and p.suffix!='.pyc')
    payload=dict(created_utc=energy.now(),stage='relative_inventory_prediction_and_sourceadaptation_checkpoint',
        goal_status='active',goal_complete=False,original29_5_preserved=True,fit_records=353,new_optimizer_starts=462,
        core_checks=262,learning_checks=59,contract_checks=a['checks'],repaired_episodes=a['repaired_episodes'],repaired_solutions=a['repaired_solutions'],
        v7_presets=17,sourceexcluded_bundles=50,sourceexcluded_requests=6900,RT_replays=870,knownsource_requests=1730,
        stable_shared_gain_established=False,independent_blind_validation=False,new_DFT_MD=0,new_material_measurements=0,
        parent_entries_checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in frozen])
    energy.write(DEST,payload);digest=energy.sha(DEST)
    for filename in ('progress.json','goal_requirements_audit.json'):
        path=BASE/filename;value=json.loads(path.read_text());value['phase20_artifact_count']=len(frozen)
        value['phase20_manifest_sha256']=digest;value['phase20']['artifact_count']=len(frozen);value['phase20']['manifest_sha256']=digest
        energy.write(path,value)
    print('Frozen phase20',len(frozen),'parententries',sum(checked.values()),'manifestSHA256',digest,flush=True)


if __name__=='__main__':main()
