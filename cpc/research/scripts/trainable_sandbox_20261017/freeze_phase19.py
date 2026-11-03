"""Freeze delivery/source-test checkpoint; keep adverse evidence and all parents."""
from pathlib import Path
import json,sys,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase19_delivery';DEST=BASE/'artifact_manifest_phase19.json'


def verify(path):
    value=json.loads(path.read_text());rows=value['files'] if isinstance(value,dict) else value
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad;return len(rows)


def main():
    if DEST.exists():print('Verified phase19',verify(DEST));return
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{p}.json' for p in (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18)]
    checked={n:verify(BASE/n) for n in names}
    assert energy.sha(BASE/'artifact_manifest_phase18.json')=='123fbc7faf7b8b9fd3e9cad5c507dbb0481d971a634f1f02e1cbf93f02bcd42c'
    for rel in ('numerical_gates.json','runtime_verification.json','training_api_replay/verification.json',
            'oxygen_sensitivity/audit.json','rendering_check.json','visual_review.json'):
        assert json.loads((OUT/rel).read_text())['passed'],rel
    gate=json.loads((OUT/'numerical_gates.json').read_text());assert gate['checks']==92
    protocol=json.loads((OUT/'delivery_protocol.json').read_text())
    for key in ('code_hashes','input_hashes'):
        for p,h in protocol[key].items():assert energy.sha(ROOT/p)==h,p
    for rel in ('leedy2018/protocol.json','leedy2018_v2/protocol.json','oxygen_sensitivity/protocol.json'):
        value=json.loads((OUT/rel).read_text())
        script='phase19_pressure.py' if 'oxygen' in rel else 'phase19_leedy.py'
        assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017'/script)==value['code_sha256'],rel
        for p,h in value.get('input_hashes',{}).items():assert energy.sha(ROOT/p)==h,p
    amendment=json.loads((OUT/'leedy2018_v2/reader_amendment.json').read_text())
    assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017/phase19_leedy_v2.py')==amendment['wrapper_sha256']
    metadata=json.loads((OUT/'leedy2018_v2/source_metadata_amendment.json').read_text())
    assert not metadata['changes_numeric_results'] and metadata['measurement_T_K']==300.
    r=json.loads((OUT/'training_api_replay/verification.json').read_text())
    assert r['saved_training_prediction_replays']==497 and r['distinct_dose_anchor_tasks']==54
    source=json.loads((OUT/'leedy2018_v2/audit.json').read_text());assert source['rows']==source['numeric_rows']==1200
    oxygen=json.loads((OUT/'oxygen_sensitivity/audit.json').read_text());assert oxygen['prediction_rows']==140
    fits=[json.loads(p.read_text()) for p in (OUT/'fits').glob('*.json')]
    assert len(fits)==6 and sum(len(f['trials']) for f in fits)==12
    assert all(f['selected_success'] and not f['state_failures'] for f in fits)
    rendering=json.loads((OUT/'rendering_check.json').read_text());assert rendering['numeric_claims_verified']==59
    assert rendering['pdf_sha256']==energy.sha(OUT/'ML与Sandbox_目标核对与跨来源预测检验.pdf')
    assert rendering['report_sha256']==energy.sha(OUT/'ML与Sandbox_目标核对与跨来源预测检验.md')
    direction=json.loads((OUT/'goal_direction_check.json').read_text())
    assert direction['system_goal_status']=='active' and not direction['goal_complete']
    requirements=json.loads((BASE/'goal_requirements_audit.json').read_text())
    assert requirements['goal_status']=='active' and not requirements['sufficient_to_mark_goal_complete']
    # Preserve the exact earlier error and its source; no numeric repair hidden.
    assert 'IndexError' in (OUT/'leedy.log').read_text()
    assert 'float' in amendment['retained_v1'] or 'conversion' in amendment['retained_v1']
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_CHECK_PHASE19_20261018.md','docs/ML_SANDBOX_GENERALIZATION_NEXT_20261018.md')]
    files += [BASE/p for p in ('CONTINUE_PHASE19_HERE.md','progress.json','goal_requirements_audit.json')]
    snapshot=BASE/'code_snapshot_phase19'
    for p in files:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(p),target
        else:shutil.copyfile(p,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    manifest=dict(created_utc=energy.now(),stage='fullsource_property_predictor_and_frozen_external_prediction',
        goal_status='active',goal_complete=False,primary='old29/5unchanged;Leedy5conditionsnotpooled',
        new_fullsource_fits=6,new_optimizer_starts=12,unified_presets=16,numerical_checks=92,
        saved_training_prediction_replays=497,valid_distinctdose_anchor_tasks=54,
        interface_requests=2318,external_specimens=5,external_predictions=1200,oxygen_scenarios=140,
        stable_gain_established=False,independent_lab_claim=False,new_DFT_MD=0,new_material_measurements=0,
        parent_entries_checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in frozen])
    energy.write(DEST,manifest);print('Frozen phase19',len(frozen),'parententries',sum(checked.values()),flush=True)
    print('Manifest SHA256',energy.sha(DEST),flush=True)


if __name__=='__main__':main()
