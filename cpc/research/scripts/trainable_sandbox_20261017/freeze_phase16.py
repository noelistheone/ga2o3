"""Freeze the thermal/joint prediction checkpoint and preserve all parents."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase16_thermal_joint';DEST=BASE/'artifact_manifest_phase16.json'
TABLE=ROOT/'data/physics_reference/fermi_half_v1.npz'


def verify(path):
    value=json.loads(path.read_text());rows=value['files'] if isinstance(value,dict) else value
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad
    return len(rows)


def main():
    if DEST.exists():print('Verified phase16',verify(DEST));return
    names=('protected_inputs_before.json','artifact_manifest_phase1.json','artifact_manifest_phase2.json',
        'artifact_manifest_phase34.json','artifact_manifest_phase56.json','artifact_manifest_phase7.json',
        'artifact_manifest_phase8.json','artifact_manifest_phase9.json','artifact_manifest_phase10.json',
        'artifact_manifest_phase11.json','artifact_manifest_phase12.json','artifact_manifest_phase13.json',
        'artifact_manifest_phase14.json','artifact_manifest_phase15.json')
    checked={name:verify(BASE/name) for name in names}
    gates=('numerical_gates.json','validation/contract_audit.json','research_runtime/verification.json',
        'mu_controls/contract_audit.json','property_evaluation/contract_audit.json','transport_scope/contract_audit.json',
        'parameter_scope/contract_audit.json','shared_state_head_control/contract_audit.json','rendering_check.json')
    for rel in gates:assert json.loads((OUT/rel).read_text())['passed'],rel
    protocols=('numerical_gates.json','prediction_protocol.json','report_claims.json','research_runtime/protocol.json',
        'mu_controls/protocol.json','property_evaluation/protocol.json','transport_scope/protocol.json',
        'parameter_scope/protocol.json','shared_state_head_control/protocol.json')
    for rel in protocols:
        payload=json.loads((OUT/rel).read_text())
        for group in ('code_hashes','input_hashes','evidence_hashes'):
            for filename,digest in payload.get(group,{}).items():assert energy.sha(ROOT/filename)==digest,(rel,filename)
        if 'code_sha256' in payload:
            code={'report_claims.json':'phase16_report.py','research_runtime/protocol.json':'phase16_runtime.py',
                'mu_controls/protocol.json':'phase16_mu_controls.py','property_evaluation/protocol.json':'phase16_property_evaluation.py',
                'transport_scope/protocol.json':'phase16_transport_scope.py','parameter_scope/protocol.json':'phase16_parameter_scope.py',
                'shared_state_head_control/protocol.json':'phase16_shared_state_head_control.py'}[rel]
            assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017'/code)==payload['code_sha256'],rel
    for source in json.loads((OUT/'thermal_audit/audit.json').read_text())['sources']:
        assert energy.sha(source['source_file'])==source['sha256']
    a=json.loads((OUT/'validation/contract_audit.json').read_text())
    assert a['new_fit_records']==150 and a['actual_new_optimizer_starts']==300 and a['V1_failed_start_fits']==0
    assert not a['selected_outer_nonconverged'] and a['inner_scores']==3600 and a['missing_primary_queries']==0
    runtime=json.loads((OUT/'research_runtime/verification.json').read_text())
    assert runtime['saved_bundles']==180 and runtime['independent_rows']==8280
    assert not json.loads((OUT/'research_runtime/independent_outputs.json').read_text())['blocked_paths']
    init=json.loads((OUT/'development_runs/initial_gate_failure'/OUT.relative_to(ROOT)/'numerical_gates.json').read_text())
    assert not init['passed'] and sum(not c['passed'] for c in init['checks'])==76
    assert json.loads((OUT/'goal_direction_check.json').read_text())['system_goal_status']=='active'
    sources=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    sources += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_AUDIT_20261017.md','docs/ML_SANDBOX_THERMAL_JOINT_20261018.md',
        'docs/ML_SANDBOX_INVENTORY_CLOSURE_NEXT_20261018.md','docs/ML_SANDBOX_JOINT_N_MU_NEXT_20261018.md')]
    sources += [TABLE]+[BASE/p for p in ('CONTINUE_PHASE16_HERE.md','progress.json','goal_requirements_audit.json')]
    snapshot=BASE/'code_snapshot_phase16'
    for source in sources:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(source)==energy.sha(target),target
        else:shutil.copyfile(source,target)
    files=sorted([TABLE]+[p for folder in (OUT,snapshot) for p in folder.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'])
    payload=dict(created_utc=energy.now(),stage='thermal_closure_and_joint_doped_property_prediction',goal_status='active',goal_complete=False,
        primary='unchanged29Si/Sn5DOI/0-1-2nHspecimenanchors/common18',structures=12,new_fits=150,new_optimizer_starts=300,
        inherited_fits=30,numerical_checks=945,retained_initial_failed_checks=76,inner_scores=3600,
        independent_bundles=180,independent_rows=8280,RTmu_pairs=5,RTmu_DOIs=2,muT_reads=54,muT_specimens=3,muT_DOIs=1,
        shared_state_mu_output_offsets=60,mu_transport_scope_states=708,majority_sigma_kl_lt1_states=338,
        independent_sigma_labels=0,complete_actual_thermal_histories=0,universal_gain_established=False,
        next='implement/gate shared non-equilibrium inventory closure;notimplemented;positiveextra_mu0 downstream',
        new_DFT_MD=0,new_material_measurements=0,parents_checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in files])
    energy.write(DEST,payload);print('Frozen phase16',len(files),'parent entries',sum(checked.values()))
    print('Manifest SHA256',energy.sha(DEST))


if __name__=='__main__':main()
