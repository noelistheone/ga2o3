"""Freeze joint-statistics prediction, source adaptation and unit references."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase14_joint_statistics';DEST=BASE/'artifact_manifest_phase14.json'


def verify(path):
    payload=json.loads(path.read_text());rows=payload['files'] if isinstance(payload,dict) else payload
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad
    return len(rows)


def main():
    if DEST.exists():print('Verified phase14',verify(DEST));return
    names=('protected_inputs_before.json','artifact_manifest_phase1.json','artifact_manifest_phase2.json',
        'artifact_manifest_phase34.json','artifact_manifest_phase56.json','artifact_manifest_phase7.json',
        'artifact_manifest_phase8.json','artifact_manifest_phase9.json','artifact_manifest_phase10.json',
        'artifact_manifest_phase11.json','artifact_manifest_phase12.json','artifact_manifest_phase13.json')
    checked={name:verify(BASE/name) for name in names}
    for rel in ('model_gates.json','validation/contract_audit.json','solver_budget_v2/refinement_gate.json',
        'solver_budget_v2/validation/contract_audit.json','source_adaptation/validation/contract_audit.json',
        'transport_audit/gates.json','research_runtime/verification.json','research_runtime/charge_budget_audit.json',
        'selection_sensitivity.json','rendering_check.json'):
        assert json.loads((OUT/rel).read_text())['passed'],rel
    for rel in ('model_gates.json','prediction_protocol.json','solver_budget_v2/protocol.json',
        'source_adaptation/source_protocol.json','transport_audit/protocol.json','research_runtime/protocol.json',
        'research_runtime/charge_budget_audit.json','report_claims.json'):
        payload=json.loads((OUT/rel).read_text())
        for group in ('code_hashes','input_hashes','evidence_hashes'):
            for filename,digest in payload.get(group,{}).items():assert energy.sha(ROOT/filename)==digest,(rel,filename)
    runtime=json.loads((OUT/'research_runtime/protocol.json').read_text())
    for key,filename in (
        ('code_sha256','scripts/trainable_sandbox_20261017/phase14_runtime.py'),
        ('inference_code_sha256','src/sandbox/donor_mixture_model.py'),
        ('adaptation_code_sha256','src/sandbox/scalar_profile_adaptation.py')):
        assert energy.sha(ROOT/filename)==runtime[key],filename
    charge=json.loads((OUT/'research_runtime/charge_budget_audit.json').read_text())
    assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017/phase14_charge_audit.py')==charge['code_sha256']
    a=json.loads((OUT/'solver_budget_v2/validation/contract_audit.json').read_text())
    assert a['selected_outer_nonconverged']==a['missing_primary_queries']==0
    assert a['composite_records']==450 and a['new_optimizer_starts']==93 and a['actual_total_optimizer_starts']==426
    assert a['nested_candidate_inner_evaluations']==8520
    a=json.loads((OUT/'source_adaptation/validation/contract_audit.json').read_text())
    assert a['episodes']==250 and a['missing_query_rows']==0
    assert json.loads((OUT/'goal_direction_check.json').read_text())['system_goal_status']=='active'
    sources=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    sources += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_AUDIT_20261017.md','docs/ML_SANDBOX_JOINT_STATISTICS_20261017.md',
        'docs/ML_SANDBOX_JOINT_SOURCE_ADAPTATION_20261017.md','docs/ML_SANDBOX_DONOR_MIXTURE_THEORY_20261017.md',
        'docs/ML_SANDBOX_TRANSPORT_UNIT_DERIVATION_20261017.md','docs/ML_SANDBOX_COHERENT_HALL_NEXT_20261017.md')]
    sources += [BASE/p for p in ('CONTINUE_PHASE14_HERE.md','progress.json','goal_requirements_audit.json')]
    snap=BASE/'code_snapshot_phase14'
    for source in sources:
        dst=snap/source.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert energy.sha(dst)==energy.sha(source),dst
        else:shutil.copyfile(source,dst)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for source in BASE.glob('phase14*.log'):
        dst=logs/source.name
        if dst.exists():assert energy.sha(dst)==energy.sha(source),dst
        else:shutil.copyfile(source,dst)
    files=sorted(p for folder in (OUT,snap) for p in folder.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    payload=dict(created_utc=energy.now(),stage='joint_internal_statistics_and_source_adaptation_for_doping_prediction',
        goal_status='active',goal_complete=False,primary='unchanged29Si/Sn5DOI,0/1/2sampleanchors,common18',
        supplementary='7oldSi/2DOI,142nreads,no newspecimens',methods=20,new_fit_records=45,
        composite_fit_records=450,physical_fits=210,new_optimizer_starts=93,actual_total_starts=426,composite_starts=422,
        map_numerical_checks=381,transport_reference_checks=63,inner_base_scores=8520,source_variants=10,
        source_episodes=250,source_inner_scores=3000,independent_bundles=315,independent_carrier_rows=1827,
        independent_paired_transport_rows=231,full_chain_rows=297,paired_mu_specimens=5,paired_mu_DOIs=2,
        independent_sigma_labels=0,new_transport_reference_used_for_predictions=False,
        stable_primary_increment_established=False,new_physical_measurements=0,new_DFT_MD=0,
        authorized_GPU=1,GPU_training=False,retained_failures='V1oneunconvergedstart;transportinitialattributeerror',
        next_prediction_work='same-state consistentFD/Hall and transport closure;notyetimplemented',
        parents_checked=checked,parent_sha256={name:energy.sha(BASE/name) for name in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in files])
    energy.write(DEST,payload)
    print('Frozen phase14',len(files));print('Manifest SHA256',energy.sha(DEST))


if __name__=='__main__':main()
