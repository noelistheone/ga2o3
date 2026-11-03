"""Freeze chemical/process prediction evidence; an existing ledger is verification only."""
from pathlib import Path
import sys, json, shutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from freeze_phase21 import verify
import phase2_energy as e

BASE = e.BASE
OUT = BASE / 'phase24_chemical_process'
DEST = BASE / 'artifact_manifest_phase24.json'


def main():
    if DEST.exists():
        print('Verified phase24', verify(DEST))
        return
    names = ['protected_inputs_before.json'] + [f'artifact_manifest_phase{i}.json' for i in (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)]
    checked = {n: verify(BASE / n) for n in names}
    assert sum(checked.values()) == 43717
    assert e.sha(BASE / 'artifact_manifest_phase23.json') == '6c665bb588915cbd8a3349e32aab27bce1e456f43a9282108c121028f13e5c81'
    for rel in ('extraction/extraction_audit.json', 'chemical_learning/audit.json', 'paired_validation/audit.json',
        'property_prediction/audit/contract_audit.json', 'paired_property_prediction/audit.json',
        'process_gradients/gates.json', 'process_gradients/tied_parameter_gates.json', 'inventory_scope/audit.json',
        'delivery/CLI_smoke.json', 'delivery/paired_CLI_smoke.json', 'rendering_check.json', 'visual_review.json'):
        data = json.loads((OUT / rel).read_text())
        assert data['passed'], rel
        for key in ('code_hashes', 'input_hashes'):
            for name, h in data.get(key, {}).items():
                assert e.sha(ROOT / name) == h, (rel, name)
    for p in OUT.rglob('protocol.json'):
        data = json.loads(p.read_text())
        for key in ('code_hashes', 'input_hashes'):
            for name, h in data.get(key, {}).items():
                assert e.sha(ROOT / name) == h, (p, name)
    goal = json.loads((OUT / 'goal_direction_check.json').read_text())
    assert goal['system_goal_status'] == 'active' and not goal['goal_complete']
    assert goal['requests'] == 32940 and goal['new_shared_property_training'] == 0
    render = json.loads((OUT / 'rendering_check.json').read_text())
    report = OUT / 'ML与Sandbox_化学过程学习与完整性质预测.md'
    assert render['report_sha256'] == e.sha(report) and render['pdf_sha256'] == e.sha(report.with_suffix('.pdf'))
    for p in ('chemical_audit_v1_syntax/invalidated.json', 'report_before_pair_design_review/scope_review.json'):
        assert (OUT / p).exists()
    e.write(OUT / 'parent_verification.json', dict(passed=True, created_utc=e.now(), entries=sum(checked.values()),
        checked=checked, parent_sha256={n:e.sha(BASE / n) for n in names}))
    for name in ('progress.json', 'goal_requirements_audit.json'):
        p = BASE / name
        v = json.loads(p.read_text())
        v['phase24']['frozen'] = True
        e.write(p, v)
    files = list((ROOT / 'src/sandbox').glob('*.py')) + list((ROOT / 'scripts/trainable_sandbox_20261017').glob('*.py'))
    files += [ROOT / p for p in ('README.md', 'AGENTS.md', 'docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_CHECK_PHASE24_20261018.md', 'docs/ML_SANDBOX_HIGH_DOSE_SHARED_LEARNING_NEXT_20261018.md',
        'docs/ML_SANDBOX_PAIRED_CHEMICAL_PROTOCOL_20261018.md', 'docs/ML_SANDBOX_CHEMICAL_PROCESS_NEXT_20261018.md')]
    files += [BASE / p for p in ('progress.json', 'goal_requirements_audit.json', 'CONTINUE_PHASE24_HERE.md', 'CONTINUE_PHASE24_WORKING.md')]
    snapshot = BASE / 'code_snapshot_phase24'
    for p in files:
        target = snapshot / p.relative_to(ROOT)
        target.parent.mkdir(exist_ok=True, parents=True)
        if target.exists():
            assert e.sha(target) == e.sha(p), target
        else:
            shutil.copyfile(p, target)
    frozen = sorted(p for folder in (OUT, snapshot) for p in folder.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'locks' not in p.parts and p.suffix != '.pyc')
    payload = dict(created_utc=e.now(), stage='learned_process_chemistry_and_coupled_property_predictions', goal_status='active',
        goal_complete=False, chemical_layers=12, stacks=2, DOIs=1, chemical_fits=3534, actual_property_requests=32940,
        numeric_requests=32940, refusals=0, summary_rows=5490, traceable_table_values=571,
        stable_new_property_gain_established=False, new_Hall_training_labels=0, new_shared_property_training=0,
        new_material_measurements=0, new_DFT_MD=0, parent_entries_checked=checked,
        parent_sha256={n:e.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=e.sha(p)) for p in frozen])
    e.write(DEST, payload)
    digest = e.sha(DEST)
    for name in ('progress.json', 'goal_requirements_audit.json'):
        p = BASE / name
        v = json.loads(p.read_text())
        v.update(phase24_artifact_count=len(frozen), phase24_manifest_sha256=digest)
        v['phase24'].update(artifact_count=len(frozen), manifest_sha256=digest)
        e.write(p, v)
    print('Frozen phase24', len(frozen), 'parent entries', sum(checked.values()), 'SHA256', digest, flush=True)


if __name__ == '__main__':
    main()
