"""Freeze the completed finite-response checkpoint, never rewrite a frozen ledger."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase21_curve_response';DEST=BASE/'artifact_manifest_phase21.json'


def verify(path):
    value=json.loads(path.read_text());rows=value['files'] if isinstance(value,dict) else value
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad;return len(rows)


def main():
    if DEST.exists():print('Verified phase21',verify(DEST));return
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{p}.json' for p in (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20)]
    checked={n:verify(BASE/n) for n in names}
    assert energy.sha(BASE/'artifact_manifest_phase20.json')=='aeed6f1406922d871e85178e68e7a8f391cd2b2bac01ec3d701472e42d306bc4'
    for rel in ('numerical_gates.json','api_gates/verification.json','affine_controls/numerical_gates_v2.json',
                'affine_controls/api_gates.json','chemical_transport_controls/gates.json','audit/contract_audit.json',
                'cohort_properties/audit.json','external_leedy/audit.json','chemical_transport_controls/audit.json',
                'delivery/CLI_smoke.json','rendering_check.json','visual_review.json'):
        assert json.loads((OUT/rel).read_text())['passed'],rel
    protocols=('prediction_protocol.json','affine_controls/protocol.json','chemical_transport_controls/protocol.json',
               'cohort_properties/protocol.json','external_leedy/protocol.json','delivery/export_protocol.json')
    for rel in protocols:
        p=json.loads((OUT/rel).read_text())
        for kind in ('code_hashes','input_hashes'):
            for name,h in p.get(kind,{}).items():assert energy.sha(ROOT/name)==h,(rel,name)
    rendering=json.loads((OUT/'rendering_check.json').read_text());report=OUT/'ML与Sandbox_响应约束与强对照预测检验.md'
    assert rendering['pdf_sha256']==energy.sha(report.with_suffix('.pdf')) and rendering['report_sha256']==energy.sha(report)
    manifest=json.loads((OUT/'delivery/predictor_bundle_v8/manifest.json').read_text())
    assert manifest['version']==8 and len(manifest['variants'])+len(manifest['energy_methods'])==21 and len(manifest['files'])==8
    for folder in ('delivery/predictor_bundle_v8','affine_controls/predictor_bundle'):
        b=OUT/folder;m=json.loads((b/'manifest.json').read_text())
        for f in m['files']:assert energy.sha(b/f['path'])==f['sha256']
    g=json.loads((OUT/'goal_direction_check.json').read_text());assert g['system_goal_status']=='active' and not g['goal_complete']
    assert g['sourceexcluded_API_requests']==11696 and g['knownsource_API_requests']==2500 and g['chemical_transport_outputs']==11018
    assert (OUT/'affine_controls/numerical_gates.json').exists() and (OUT/'audit_v1/contract_audit.json').exists()
    energy.write(OUT/'parent_verification.json',dict(passed=True,created_utc=energy.now(),entries=sum(checked.values()),
        checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names}))
    for filename in ('progress.json','goal_requirements_audit.json'):
        path=BASE/filename;value=json.loads(path.read_text());value['phase21']['frozen']=True
        value['protected_files_checked']=sum(checked.values());value['changed_protected_files']=0;energy.write(path,value)
    continuation=BASE/'CONTINUE_PHASE21_HERE.md'
    continuation.write_text(continuation.read_text()+'\n## 最终冻结\n\nphase21_finish.py与freeze_phase21.py已完成。artifact_manifest_phase21.json为权威清单；父阶段全部SHA校验通过。系统goal仍active；没有后台任务。下一项为原始来源/输运核查，不重启已完成模型或把无稳定增益改写为成功。\n')
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_CHECK_PHASE21_20261018.md','docs/ML_SANDBOX_FINITE_RESPONSE_20261018.md')]
    files += [BASE/p for p in ('CONTINUE_PHASE21_HERE.md','CONTINUE_PHASE21_WORKING.md','progress.json','goal_requirements_audit.json')]
    snapshot=BASE/'code_snapshot_phase21'
    for p in files:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p),target
        else:shutil.copyfile(p,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'locks' not in p.parts and p.suffix!='.pyc')
    payload=dict(created_utc=energy.now(),stage='finite_response_and_strong_controls_prediction_checkpoint',
        goal_status='active',goal_complete=False,primary_variants=14,additional_affine_controls=3,primary_groups=518,
        new_shared_optimizer_starts=0,shared_weights='phase20 frozen',available_presets=24,
        sourceexcluded_bundles=85,sourceexcluded_requests=11696,RT_replays=1445,knownsource_requests=2500,chemical_transport_outputs=11018,
        stable_shared_gain_established=False,independent_blind_validation=False,new_DFT_MD=0,new_material_measurements=0,
        parent_entries_checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in frozen])
    energy.write(DEST,payload);digest=energy.sha(DEST)
    for filename in ('progress.json','goal_requirements_audit.json'):
        path=BASE/filename;value=json.loads(path.read_text());value['phase21_artifact_count']=len(frozen)
        value['phase21_manifest_sha256']=digest;value['phase21']['artifact_count']=len(frozen);value['phase21']['manifest_sha256']=digest
        energy.write(path,value)
    print('Frozen phase21',len(frozen),'parententries',sum(checked.values()),'manifestSHA256',digest,flush=True)


if __name__=='__main__':main()
