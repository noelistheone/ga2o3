"""Freeze actual joint-learning evidence only after the full manual review.

With an existing manifest, perform read-only integrity verification. A passing
freeze certifies reproducibility, not predictive superiority or goal completion.
"""
from pathlib import Path
import os
import json
import shutil
import sys
import numpy
import scipy
import pandas
import matplotlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from freeze_phase21 import verify
from freeze_phase28 import PARENTS as EARLIER_PARENTS
from phase29_objective_audit import AUDITS, verify_hashes
import phase2_energy as e

BASE=e.BASE
OUT=BASE/'phase29_joint_scattering'
DEST=BASE/'artifact_manifest_phase29.json'
PARENTS=[*EARLIER_PARENTS,'artifact_manifest_phase28.json']
TITLE='ML与Sandbox_共享库存与正散射联合预测研究'
MUTABLE={OUT/'execution_status_current.json'}


def document(path):
    return json.loads(Path(path).read_text())


def live_jobs():
    found=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if int(proc.name)==os.getpid():continue
            status=(proc/'stat').read_text().split()[2]
            if status in ('Z','X'):continue
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if len(argv)<2 or not Path(argv[0]).name.startswith('python'):continue
            script=Path(argv[1]).name
            if script.startswith('phase29_') or (script=='export_research_pdf_phase29.py'):
                found.append(dict(pid=int(proc.name),script=script,state=status))
        except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeDecodeError):pass
    return found


def parent_integrity():
    assert e.sha(BASE/'artifact_manifest_phase28.json')==(
        '8be1a180910f7fc42f020bc599af23beddfe76eedc1483acd607e5b3dca625b3')
    counts={name:verify(BASE/name) for name in PARENTS}
    assert sum(counts.values())==55554
    return counts


def main():
    if DEST.exists():
        manifest=document(DEST)
        for name,digest in manifest['parent_sha256'].items():
            assert e.sha(BASE/name)==digest,name
        counts=parent_integrity()
        current=verify(DEST)
        print('Read-only phase29 verified',current,'entries;',sum(counts.values()),'parententries',flush=True)
        return
    assert not live_jobs(),('Live phase29 jobs remain;inspect original handles',live_jobs())
    # No partial log, intent, preflight PDF or toy numerical gate can replace
    # these completed actual-result and manual evidence requirements.
    for name in [*AUDITS,'additional_source_audit_v1/qualification.json']:
        value=document(OUT/name)
        assert value['passed'],name
        verify_hashes(value)
    goal_audit=document(OUT/'objective_audit_v1.json')
    assert goal_audit['passed'] and not goal_audit['goal_complete']
    assert goal_audit['system_goal_status_not_changed']
    verify_hashes(goal_audit)
    science=document(OUT/'scientific_interpretation_v1.json')
    assert science['fixed_primary']=='JointProcess_internal'
    assert not science['outer_method_reselected']
    assert science['all_adverse_results_retained']
    assert science['next_testable_predictive_work']
    for name in ('training_protocol.json','prediction_design_protocol.json',
            'pipeline_protocol_before_wait.json','post_prediction_protocol_before_wait.json',
            'report_protocol_before_wait.json'):
        verify_hashes(document(OUT/name))
    assert not document(OUT/'numerical_gates_v1/gates.json')['passed']
    assert (OUT/'nonstopped_candidate_observation_v1.json').exists()
    train=document(OUT/'training_complete.json')
    predict=document(OUT/'prediction/execution_complete.json')
    api=document(OUT/'api_replay/audit.json')
    statistics=document(OUT/'report_v1/study_statistics.json')
    visual=document(OUT/'visual_review_v1.json')
    rendered=document(OUT/'rendering_check_v1.json')
    report=OUT/(TITLE+'.md')
    assert visual['all_pages_actually_reviewed']
    assert visual['pages']==rendered['pages'] and not rendered['warnings']
    assert visual['pdf_sha256']==rendered['pdf_sha256']==e.sha(report.with_suffix('.pdf'))
    assert visual['source_sha256']==rendered['source_sha256']==e.sha(report)
    for name,digest in visual['page_image_sha256'].items():assert e.sha(ROOT/name)==digest,name
    layout=OUT/'report_layout_v2'
    verify_hashes(document(layout/'protocol_before.json'))
    lv=document(layout/'visual_review_v2.json')
    preferred=OUT/(TITLE+'_排版修订V2.md')
    assert e.sha(preferred)==e.sha(report)
    assert lv['all_pages_actually_reviewed'] and lv['pages']==17
    assert lv['pdf_sha256']==e.sha(preferred.with_suffix('.pdf'))
    for name,digest in lv['page_image_sha256'].items():assert e.sha(ROOT/name)==digest,name
    assert train['fit_records']==1392 and train['nonlinear_fit_records']==1200
    assert predict['outer_scenario_rows']==7056 and predict['variants']==48
    assert api['passed'] and api['requests']==1056
    assert statistics['paired_Si']==9 and statistics['paired_Sn']==2
    assert statistics['common_carrier_conditions']==20
    counts=parent_integrity()
    e.write(OUT/'parent_verification_at_freeze.json',dict(passed=True,created_utc=e.now(),
        entries=sum(counts.values()),checked=counts,changed_parent_files=0,
        parent_sha256={name:e.sha(BASE/name) for name in PARENTS}))
    e.write(OUT/'environment_at_freeze.json',dict(created_utc=e.now(),python=sys.version,
        executable=sys.executable,numpy=numpy.__version__,scipy=scipy.__version__,
        pandas=pandas.__version__,matplotlib=matplotlib.__version__,
        GPU_used=False,GPU_authorized='GPU1only',new_DFT_MD=0,new_material_measurements=0,
        raw_thirdparty_physical_database_copied=False,goal_complete=False))
    # The frequently refreshed pointer is explicitly mutable; freeze its last
    # value in a separate file rather than invalidating scientific SHA records.
    if (OUT/'execution_status_current.json').exists():
        shutil.copyfile(OUT/'execution_status_current.json',OUT/'execution_status_at_freeze.json')
    snapshot=BASE/'code_snapshot_phase29'
    files=list((ROOT/'src/sandbox').glob('*.py'))
    files+=list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    files+=[ROOT/name for name in ('README.md','AGENTS.md',
        'docs/ML_SANDBOX_GOAL_20261017.md','docs/ML_SANDBOX_GOAL_CHECK_PHASE29_20261019.md',
        'docs/ML_SANDBOX_METHOD_PHASE29_V2_20261019.md',
        'docs/ML_SANDBOX_JOINT_TRAINING_MATH_PHASE29_20261019.md',
        'docs/ML_SANDBOX_SCENARIO_INFORMATION_PHASE29_20261019.md',
        'docs/ML_SANDBOX_TRANSPORT_SCOPE_PHASE29_20261019.md',
        'docs/ML_SANDBOX_SCATTER_BUDGET_CORRECTION_PHASE29_20261019.md')]
    files+=[ROOT/'docs/ML_SANDBOX_GOAL_FINAL_REVIEW_20261019.md']
    files+=[BASE/name for name in ('progress.json','goal_requirements_audit.json',
        'CONTINUE_PHASE29_WORKING.md','CONTINUE_PHASE29_HERE.md')]
    for source in files:
        assert source.is_file(),str(source)
        target=snapshot/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert e.sha(source)==e.sha(target),target
        else:shutil.copyfile(source,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*')
        if p.is_file() and p not in MUTABLE and '__pycache__' not in p.parts
        and p.suffix!='.pyc' and 'locks' not in p.parts)
    manifest=dict(created_utc=e.now(),stage='trainable_joint_inventory_positive_scattering_prediction',
        goal_status='active',goal_complete=False,reported_conditions=35,DOIs=7,related_groups=5,
        paired_conditions=11,paired_Si=9,paired_Sn=2,scenario_rows=49,
        original33_roles_common20_preserved=True,fit_records=1392,nonlinear_fit_records=1200,
        actual_optimizer_starts=train['actual_optimizer_starts'],
        variants=48,outer_scenario_rows=7056,API_requests=api['requests'],
        primary='JointProcess_internal',report_pages=rendered['pages'],
        preferred_report=str(preferred.with_suffix('.pdf').relative_to(ROOT)),
        preferred_report_sha256=e.sha(preferred.with_suffix('.pdf')),
        layout_v2_scientific_source_unchanged=True,
        report_data_claims=len(document(OUT/'report_v1/numeric_claims.json')['claims']),
        posthoc_fixed_state_transport_evaluations=98,
        posthoc_diagnostic_report_pages=3,
        posthoc_diagnostic_data_claims=len(document(OUT/'transport_attribution_v1/diagnostic_claims.json')['claims']),
        all_adverse_results_and_historical_failures_retained=True,
        missing_inputs_and_retrospective_scope_explicit=True,
        original_scatter_budget_notation_correction_preserved=True,
        no_automatic_prediction_gain_or_material_Born_claim=True,
        new_DFT_MD=0,new_material_measurements=0,GPU_used=False,
        parent_entries_checked=counts,parent_sha256={name:e.sha(BASE/name) for name in PARENTS},
        excluded_mutable_pointers=[str(p.relative_to(ROOT)) for p in MUTABLE],
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=e.sha(p)) for p in frozen])
    e.write(DEST,manifest)
    actual=verify(DEST)
    assert actual==len(frozen)
    receipt=dict(created_utc=e.now(),passed=True,new_entries_verified=actual,
        parent_entries_verified=55554,total_parent_entries_for_next_stage=55554+actual,
        manifest_sha256=e.sha(DEST),goal_complete=False,
        full_goal_audit=goal_audit['all_requirements_evidenced'],
        scientific_conclusions=goal_audit['primary_prediction_conclusion'],
        next_testable_predictive_work=science['next_testable_predictive_work'],
        no_system_goal_status_change=True)
    e.write(BASE/'phase29_freeze_receipt.json',receipt)
    for name in ('progress.json','goal_requirements_audit.json'):
        path=BASE/name;value=document(path)
        value.setdefault('phase29_joint_scattering',{}).update(
            frozen=True,artifact_count=actual,manifest_sha256=e.sha(DEST))
        e.write(path,value)
    print('Frozen phase29',actual,'entries;55554parents verified;SHA',e.sha(DEST),flush=True)


if __name__=='__main__':main()
