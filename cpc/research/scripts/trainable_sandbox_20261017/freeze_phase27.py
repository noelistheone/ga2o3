"""Freeze completed phase27 evidence after result, PDF and parent verification."""
from pathlib import Path
import json
import shutil
import sys
import numpy
import scipy
import pandas
import matplotlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from freeze_phase21 import verify
from phase27_report import AUDITS
from phase27_finish import PROTOCOLS, verify_hash_maps
import phase2_energy as e

BASE = e.BASE; OUT = BASE / 'phase27_charge_feasible'; DEST = BASE / 'artifact_manifest_phase27.json'


def live_stage_jobs():
    names = {'phase27_training.py', 'phase27_after_training.py', 'phase27_prediction.py',
        'phase27_source_runtime.py', 'phase27_cohort_replay.py', 'phase27_api_runtime.py',
        'phase27_api_runtime_v2.py', 'phase27_external.py', 'phase27_analysis.py',
        'phase27_physical_scope.py', 'phase27_delivery_final_check.py', 'phase27_capacity_limit_check.py',
        'phase27_prediction_audit_v2.py', 'phase27_resume_after_audit_v2.py',
        'phase27_lowdose_transfer.py', 'phase27_lowdose_audit.py', 'phase27_lowdose_audit_v2.py'}
    jobs = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if len(argv) > 1 and Path(argv[0].decode()).name.startswith('python') and Path(argv[1].decode()).name in names:
                jobs.append(dict(pid=int(proc.name), script=argv[1].decode()))
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError): pass
    return jobs


def main():
    if DEST.exists(): print('Verified frozen phase27', verify(DEST)); return
    live = live_stage_jobs(); assert not live, ('Actual jobs still live;do not freeze changing evidence', live)
    names = ['protected_inputs_before.json'] + [f'artifact_manifest_phase{i}.json' for i in
        (1, 2, 34, 56, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)]
    counts = {name: verify(BASE / name) for name in names}; assert sum(counts.values()) == 52718
    assert e.sha(BASE / 'artifact_manifest_phase26.json') == '4489891819c035c45b3bae9c128f6a9e79b6f63cd79742c755d3a42eaacc6bcc'
    for name in (*AUDITS, 'capacity_limit/audit.json', 'independent_input_audit/lowdose_transfer/audit.json',
        'rendering_check.json', 'visual_review.json', 'objective_audit.json'):
        value = json.loads((OUT / name).read_text()); assert value['passed'], name
        verify_hash_maps(value, name)
    for name in PROTOCOLS: verify_hash_maps(json.loads((OUT / name).read_text()), name)
    direction = json.loads((OUT / 'goal_direction_check.json').read_text())
    objective = json.loads((OUT / 'objective_audit.json').read_text())
    for name, digest in objective['evidence_sha256'].items(): assert e.sha(ROOT / name) == digest, name
    assert len(objective['requirements']) == 11
    assert direction['system_goal_status'] == 'active' and not direction['goal_complete']
    assert direction['primary'] == 'FeasibleProcess_internal' and direction['main_variants'] == 48
    assert direction['new_property_rows'] == 2376 and direction['combined_property_rows'] == 9504
    assert direction['API_point_rows'] == 9504 and direction['external_rows'] == 5760
    assert (OUT / 'gate_failure_v1').is_dir() and (OUT / 'predeclare_failure_v1').is_dir()
    assert (OUT / 'prediction_audit_failure_v1').is_dir()
    claims = json.loads((OUT / 'report_claims.json').read_text())
    report = OUT / 'ML与Sandbox_电荷可行学习与性质预测.md'
    assert e.sha(report) == claims['report_sha256'] and e.sha(ROOT / claims['theory_path']) == claims['theory_sha256']
    render = json.loads((OUT / 'rendering_check.json').read_text())
    assert e.sha(report.with_suffix('.pdf')) == render['pdf_sha256']
    bundle = OUT / 'delivery/predictor_bundle'; manifest = json.loads((bundle / 'manifest.json').read_text())
    assert len(manifest['presets']) == 624 and len(manifest['files']) == 390
    for item in manifest['files']: assert e.sha(bundle / item['path']) == item['sha256'], item['path']
    e.write(OUT / 'parent_verification_at_freeze.json', dict(passed=True, entries=sum(counts.values()),
        checked=counts, parent_sha256={name: e.sha(BASE / name) for name in names},
        changed_parent_files=0, created_utc=e.now()))
    logs = OUT / 'logs'; logs.mkdir(exist_ok=True)
    for source in sorted(Path('/tmp').glob('ga2o3_phase27_*.log')):
        target = logs / source.name
        if target.exists(): assert e.sha(source) == e.sha(target), target
        else: shutil.copyfile(source, target)
    e.write(OUT / 'environment_at_freeze.json', dict(python=sys.version, python_executable=sys.executable,
        numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__,
        matplotlib=matplotlib.__version__, GPU_used=False,
        external_physical_reference_in_bundle=False, goal_complete=False))
    for name in ('progress.json', 'goal_requirements_audit.json'):
        path = BASE / name; value = json.loads(path.read_text()); value['phase27']['frozen'] = True; e.write(path, value)
    files = list((ROOT / 'src/sandbox').glob('*.py')) + list((ROOT / 'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT / name for name in ('README.md', 'AGENTS.md', 'docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_CHECK_PHASE27_20261018.md', 'docs/ML_SANDBOX_CHARGE_FEASIBLE_NEXT_20261018.md',
        'docs/ML_SANDBOX_CHARGE_FEASIBLE_PROTOCOL_20261018.md', 'docs/ML_SANDBOX_CHARGE_FEASIBLE_COORDINATES_20261018.md',
        'docs/ML_SANDBOX_CHARGE_CAPACITY_LIMIT_20261018.md', 'docs/ML_SANDBOX_REPORTED_CARBON_SCOPE_CORRECTION_20261018.md')]
    nextdoc = direction['prediction_conclusions'].get('next_theory')
    if nextdoc: files.append(ROOT / nextdoc)
    files += [BASE / name for name in ('progress.json', 'goal_requirements_audit.json',
        'CONTINUE_PHASE27_HERE.md', 'CONTINUE_PHASE27_WORKING.md')]
    snapshot = BASE / 'code_snapshot_phase27'
    for source in files:
        target = snapshot / source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): assert e.sha(source) == e.sha(target), target
        else: shutil.copyfile(source, target)
    frozen = sorted(p for folder in (OUT, snapshot) for p in folder.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'locks' not in p.parts and p.suffix != '.pyc')
    payload = dict(created_utc=e.now(), stage='charge_feasible_shared_learning_and_dopant_property_prediction',
        goal_status='active', goal_complete=False, conditions=33, DOIs=6, related_groups=5,
        primary='FeasibleProcess_internal', training_fit_records=direction['training_fit_records'],
        actual_optimizer_starts=direction['optimizer_starts'], new_variants=12, combined_variants=48,
        new_property_rows=2376, combined_property_rows=9504, API_point_rows=9504, external_property_rows=5760,
        adverse_and_unqualified_results_retained=True, reported_C_scope_disclosed=True,
        original29_and_roles_unchanged=True, new_DFT_MD=0, new_material_measurements=0,
        parent_entries_checked=counts, parent_sha256={name: e.sha(BASE / name) for name in names},
        files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=e.sha(p)) for p in frozen])
    e.write(DEST, payload); digest = e.sha(DEST)
    for name in ('progress.json', 'goal_requirements_audit.json'):
        path = BASE / name; value = json.loads(path.read_text())
        value.update(phase27_artifact_count=len(frozen), phase27_manifest_sha256=digest)
        value['phase27'].update(artifact_count=len(frozen), manifest_sha256=digest); e.write(path, value)
    print('Frozen actual phase27', len(frozen), 'files;parent entries', sum(counts.values()), 'SHA256', digest, flush=True)


if __name__ == '__main__': main()
