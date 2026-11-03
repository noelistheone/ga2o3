"""Freeze phase28 source, transfer and untrained joint-interface evidence.

An existing manifest triggers read-only verification. It never reruns the
writers in the frozen source/transfer/core/report directories.
"""
from pathlib import Path
import os,json,shutil,sys
import numpy,scipy,pandas,matplotlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from freeze_phase21 import verify
import phase2_energy as e

BASE=e.BASE;OUT=BASE/'phase28_paired_transport'
DEST=BASE/'artifact_manifest_phase28.json'
PARENTS=['protected_inputs_before.json']+[f'artifact_manifest_phase{i}.json' for i in
    (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)]


def verify_maps(value):
    for kind in ('source_hashes','input_hashes','code_hashes'):
        for name,digest in value.get(kind,{}).items():assert e.sha(ROOT/name)==digest,(kind,name)


def jobs():
    found=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if int(p.name)==os.getpid():continue
            argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if len(argv)<2 or not Path(argv[0]).name.startswith('python'):continue
            name=Path(argv[1]).name
            if name.startswith('phase28_') or (name in ('export_research_pdf_phase19.py','phase27_api_runtime_v2.py')
                    and any('phase28_paired_transport' in x for x in argv[2:])):
                found.append(dict(pid=int(p.name),script=name))
        except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeDecodeError):pass
    return found


def main():
    if DEST.exists():
        value=json.loads(DEST.read_text())
        for name,digest in value['parent_sha256'].items():assert e.sha(BASE/name)==digest,name
        counts={name:verify(BASE/name) for name in PARENTS}
        current=verify(DEST)
        assert sum(counts.values())==54930
        print('Read-only verified phase28',current,'new entries and',sum(counts.values()),'parent entries',flush=True)
        return
    assert not jobs(),('live phase28 jobs;inspect rather than restart',jobs())
    for name in ['source_qualification_v1/protocol_before_calculation.json',
        'frozen_transport_v1/protocol_before_transfer.json','joint_scattering_core_v1/protocol_before_gates.json']:
        verify_maps(json.loads((OUT/name).read_text()))
    for name in ['source_qualification_v1/audit.json','frozen_transport_v1/audit.json',
        'joint_scattering_core_v1/gates.json','report_v2/numeric_claims_audit.json',
        'rendering_check_final.json','visual_review.json','goal_direction_verification_v1.json']:
        assert json.loads((OUT/name).read_text())['passed'],name
    goal=json.loads((OUT/'system_goal_progress_receipt_v2.json').read_text())['system_goal']['goal']
    assert goal['status']=='active'
    status=json.loads((OUT/'checkpoint_status.json').read_text())
    assert not status['goal_complete'] and status['new_training']==0
    assert status['new_predictions']==1536 and status['numeric']==status['valid']==1344
    assert status['core_checks']==318 and status['core_numerical_states']==15
    assert status['strict_goal_requirement_R6'].startswith('incomplete')
    claims=json.loads((OUT/'report_v2/numeric_claims.json').read_text())
    report=OUT/'ML与Sandbox_Sn配对观测与联合学习接口研究.md'
    assert e.sha(report)==claims['report_sha256']
    assert e.sha(ROOT/claims['theory_path'])==claims['theory_sha256']
    assert len(claims['claims'])==404
    visual=json.loads((OUT/'visual_review.json').read_text())
    render=json.loads((OUT/'rendering_check_final.json').read_text())
    assert visual['all_pages_actually_reviewed'] and render['manual_review_complete']
    assert visual['pages']==render['pages']==8 and not render['warnings']
    assert visual['pdf_sha256']==render['pdf_sha256']==e.sha(report.with_suffix('.pdf'))
    assert visual['source_sha256']==render['source_sha256']==e.sha(report)
    for name,digest in visual['page_image_sha256'].items():assert e.sha(ROOT/name)==digest,name
    assert (OUT/'rendering_failure_v1/reason.json').exists()
    assert (OUT/'documentation_draft_before_escape_fix_v1.md').exists()
    # Full prior-stage integrity, retaining the prior SHAs rather than refreshing
    # any older manifest after a change.
    assert e.sha(BASE/'artifact_manifest_phase27.json')=='4bfb026f0a75df380241fc3260fc0055737f34e06fd4a6b0dbac8456de243e9f'
    counts={name:verify(BASE/name) for name in PARENTS}
    assert sum(counts.values())==54930
    e.write(OUT/'parent_verification_at_freeze.json',dict(passed=True,created_utc=e.now(),
        entries=sum(counts.values()),checked=counts,changed_parent_files=0,
        parent_sha256={name:e.sha(BASE/name) for name in PARENTS}))
    e.write(OUT/'environment_at_freeze.json',dict(created_utc=e.now(),python=sys.version,
        executable=sys.executable,numpy=numpy.__version__,scipy=scipy.__version__,
        pandas=pandas.__version__,matplotlib=matplotlib.__version__,
        GPU_used=False,GPU_authorized='GPU1only',new_DFT_MD=0,new_material_measurements=0,
        raw_thirdparty_physical_database_copied=False,goal_complete=False))
    snapshot=BASE/'code_snapshot_phase28'
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files+=[ROOT/name for name in ['README.md','AGENTS.md',
        'docs/ML_SANDBOX_GOAL_CHECK_PHASE28_20261018.md',
        'docs/ML_SANDBOX_JOINT_SCATTERING_THEORY_PHASE28_20261019.md',
        'docs/ML_SANDBOX_JOINT_SCATTERING_TRAINING_NEXT_20261019.md']]
    files+=[BASE/name for name in ['progress.json','goal_requirements_audit.json',
        'CONTINUE_PHASE28_HERE.md','CONTINUE_PHASE28_WORKING.md']]
    for source in files:
        target=snapshot/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert e.sha(source)==e.sha(target),target
        else:shutil.copyfile(source,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'locks' not in p.parts)
    manifest=dict(created_utc=e.now(),stage='paired_Sn_source_qualification_frozen_property_transfer_and_trainable_joint_interface',
        goal_status='active',goal_complete=False,original_main_conditions=33,
        original_roles_and_common20_unchanged=True,new_main_cohort_admissions=0,
        source_conditions=10,reported_condition_triples=7,intentionally_Sn_triples=6,
        conditional_new_Sn_test_conditions=2,frozen_transfer_requests=96,
        frozen_transfer_rows=1536,numeric_valid_rows=1344,retained_refusal_rows=192,
        actual_new_training=0,actual_new_joint_training=0,core_checks=318,
        core_numerical_states=15,report_pages=8,report_data_claims=404,
        parent_entries_checked=counts,parent_sha256={name:e.sha(BASE/name) for name in PARENTS},
        missing_protocol_inputs_and_related_sources_explicit=True,
        all_adverse_and_missing_results_retained=True,no_new_predictive_gain_claim=True,
        new_DFT_MD=0,new_material_measurements=0,GPU_used=False,
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=e.sha(p)) for p in frozen])
    e.write(DEST,manifest)
    actual=verify(DEST);assert actual==len(frozen)
    receipt=dict(created_utc=e.now(),passed=True,new_entries_verified=actual,
        parent_entries_verified=54930,total_parent_entries_for_next_stage=54930+actual,
        manifest_sha256=e.sha(DEST),overall_goal_complete=False,
        outstanding=['actual joint inventory/scattering training and stable primary prediction increment',
            'full transport accuracy and material/input applicability beyond conditional stress'],
        next=status['next'])
    e.write(BASE/'phase28_freeze_receipt.json',receipt)
    for name in ('progress.json','goal_requirements_audit.json'):
        path=BASE/name;value=json.loads(path.read_text())
        value['phase28'].update(frozen=True,artifact_count=actual,manifest_sha256=e.sha(DEST))
        value.update(phase28_artifact_count=actual,phase28_manifest_sha256=e.sha(DEST))
        e.write(path,value)
    print('Frozen phase28',actual,'new entries;54930parent entries verified;SHA',e.sha(DEST),flush=True)


if __name__=='__main__':main()
