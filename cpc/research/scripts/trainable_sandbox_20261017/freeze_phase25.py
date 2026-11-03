"""Freeze a completed prediction checkpoint without replacing earlier ledgers."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from freeze_phase21 import verify
from phase25_finish import AUDITS
import phase2_energy as e

BASE=e.BASE;OUT=BASE/'phase25_shared_process';DEST=BASE/'artifact_manifest_phase25.json'


def main():
    if DEST.exists():print('Verified phase25',verify(DEST));return
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{i}.json' for i in
        (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)]
    counts={n:verify(BASE/n) for n in names};assert sum(counts.values())==47785
    assert e.sha(BASE/'artifact_manifest_phase24.json')=='7f33dcf124b336c902694f3b24280852dd2fc021b2bad955815bd4203fe4a5be'
    for rel in (*AUDITS,'rendering_check.json','visual_review.json'):
        d=json.loads((OUT/rel).read_text());assert d['passed'],rel
        for key in ('code_hashes','input_hashes'):
            for name,h in d.get(key,{}).items():assert e.sha(ROOT/name)==h,(rel,name)
    for rel in ('training_protocol.json','prediction/protocol.json','delivery/cohort_replay_v2/protocol.json',
                'external_Azizie_v2/protocol_before_transfer.json','Leedy_label_sensitivity_v2/protocol_before_sensitivity.json'):
        d=json.loads((OUT/rel).read_text())
        for key in ('code_hashes','input_hashes'):
            for name,h in d.get(key,{}).items():assert e.sha(ROOT/name)==h,(rel,name)
    for rel in ('pre_evaluation_protocol.json','post_prediction_predeclared.json'):
        for name,h in json.loads((OUT/rel).read_text())['hashes'].items():assert e.sha(ROOT/name)==h,(rel,name)
    repair=json.loads((OUT/'delivery/serialization_v2/protocol_before_v2_replay.json').read_text())
    for name,h in repair['code_hashes'].items():assert e.sha(ROOT/name)==h,name
    assert (OUT/'delivery/API_serialization_v1_failure/diagnosis.json').exists()
    theory=json.loads((OUT/'theory_scope_readback.json').read_text())
    assert theory['theory_sha256']==e.sha(ROOT/'docs/ML_SANDBOX_CONTEXT_THEORY_20261018.md')
    g=json.loads((OUT/'goal_direction_check.json').read_text());assert g['system_goal_status']=='active' and not g['goal_complete']
    assert g['primary_prediction_rows']==3762 and g['label_sensitivity_rows']==456
    assert (OUT/'startup_name_failure').is_dir() and (OUT/'source_smoke_read_guard_failure').is_dir()
    for folder in ('delivery/predictor_bundle',):
        m=json.loads((OUT/folder/'manifest.json').read_text());assert len(m['presets'])==247
        for f in m['files']:assert e.sha(OUT/folder/f['path'])==f['sha256']
    e.write(OUT/'parent_verification_at_freeze.json',dict(passed=True,entries=sum(counts.values()),checked=counts,
        parent_sha256={n:e.sha(BASE/n) for n in names},created_utc=e.now(),changed_parent_files=0))
    for name in ('progress.json','goal_requirements_audit.json'):
        p=BASE/name;d=json.loads(p.read_text());d['phase25']['frozen']=True;e.write(p,d)
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_CHECK_PHASE25_20261018.md','docs/ML_SANDBOX_CONTEXT_INVENTORY_PROTOCOL_20261018.md',
        'docs/ML_SANDBOX_CONTEXT_STRONG_CONTROLS_ADDENDUM_20261018.md','docs/ML_SANDBOX_CONTEXT_THEORY_20261018.md',
        'docs/ML_SANDBOX_CONTEXT_PRIOR_ART_20261018.md',g['next_theory'])]
    files += [BASE/p for p in ('progress.json','goal_requirements_audit.json','CONTINUE_PHASE25_HERE.md','CONTINUE_PHASE25_WORKING.md')]
    snapshot=BASE/'code_snapshot_phase25'
    for p in files:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert e.sha(target)==e.sha(p),target
        else:shutil.copyfile(p,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'locks' not in p.parts and p.suffix!='.pyc')
    payload=dict(created_utc=e.now(),stage='shared_context_inventory_learning_and_dopant_property_prediction',
        goal_status='active',goal_complete=False,conditions=33,DOIs=6,related_groups=5,
        training_fits=1596,actual_optimizer_starts=1933,primary_variant_count=19,
        primary_property_rows=3762,API_replay_points=3762,external_property_rows=2280,prose_sensitivity_rows=456,
        original29_5_and_roles_preserved=True,new_DFT_MD=0,new_material_measurements=0,
        parent_entries_checked=counts,parent_sha256={n:e.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=e.sha(p)) for p in frozen])
    e.write(DEST,payload);digest=e.sha(DEST)
    for name in ('progress.json','goal_requirements_audit.json'):
        p=BASE/name;d=json.loads(p.read_text());d['phase25_artifact_count']=len(frozen);d['phase25_manifest_sha256']=digest
        d['phase25'].update(artifact_count=len(frozen),manifest_sha256=digest);e.write(p,d)
    print('Frozen phase25',len(frozen),'parent entries',sum(counts.values()),'SHA256',digest,flush=True)


if __name__=='__main__':main()
