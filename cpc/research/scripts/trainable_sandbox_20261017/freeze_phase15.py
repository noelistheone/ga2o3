"""Freeze coherent-state prediction, all negative results and provenance."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase15_coherent_hall';DEST=BASE/'artifact_manifest_phase15.json'
TABLE=ROOT/'data/physics_reference/fermi_half_v1.npz'


def verify(path):
    payload=json.loads(path.read_text());rows=payload['files'] if isinstance(payload,dict) else payload
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad
    return len(rows)


def main():
    if DEST.exists():print('Verified phase15',verify(DEST));return
    names=('protected_inputs_before.json','artifact_manifest_phase1.json','artifact_manifest_phase2.json',
        'artifact_manifest_phase34.json','artifact_manifest_phase56.json','artifact_manifest_phase7.json',
        'artifact_manifest_phase8.json','artifact_manifest_phase9.json','artifact_manifest_phase10.json',
        'artifact_manifest_phase11.json','artifact_manifest_phase12.json','artifact_manifest_phase13.json','artifact_manifest_phase14.json')
    checked={name:verify(BASE/name) for name in names}
    gates=('numerical_gates.json','solver_budget_decision.json','validation/contract_audit.json',
        'source_adaptation/validation/contract_audit.json','research_runtime/verification.json',
        'temperature_transport/contract_audit.json','scattering_ablation/contract_audit.json','state_scope/contract_audit.json','rendering_check.json')
    for rel in gates:assert json.loads((OUT/rel).read_text())['passed'],rel
    for rel in ('numerical_gates.json','prediction_protocol.json','source_adaptation/source_protocol.json',
        'research_runtime/protocol.json','temperature_transport/protocol.json','scattering_ablation/protocol.json','state_scope/protocol.json','report_claims.json'):
        payload=json.loads((OUT/rel).read_text())
        for group in ('code_hashes','input_hashes','evidence_hashes'):
            for filename,digest in payload.get(group,{}).items():assert energy.sha(ROOT/filename)==digest,(rel,filename)
    runtime=json.loads((OUT/'research_runtime/protocol.json').read_text())
    assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017/phase15_runtime.py')==runtime['code_sha256']
    scope=json.loads((OUT/'state_scope/protocol.json').read_text())
    assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017/phase15_state_scope.py')==scope['code_sha256']
    fd=json.loads((OUT/'fd_reference_build.json').read_text())
    assert energy.sha(TABLE)==fd['sha256'] and fd['nodes']==12801
    assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017/phase15_build_fermi.py')==fd['code_sha256']
    a=json.loads((OUT/'validation/contract_audit.json').read_text())
    assert a['missing_primary_queries']==a['selected_outer_nonconverged']==0 and a['composite_records']==540
    assert a['new_optimizer_starts']==180 and a['actual_total_optimizer_starts']==606
    s=json.loads((OUT/'source_adaptation/validation/contract_audit.json').read_text())
    assert s['episodes']==450 and s['missing_query_rows']==0
    t=json.loads((OUT/'temperature_transport/contract_audit.json').read_text())
    assert t['existing_specimens']==3 and t['DOIs']==1 and t['independent_model_rows']==2916
    assert t['missing_model_outputs']==0 and t['new_sigma_labels']==0 and not t['same_T_measured_n_claimed']
    initial=json.loads((OUT/'development_runs/initial_gate_failure'/OUT.relative_to(ROOT)/'numerical_gates.json').read_text())
    assert not initial['passed']
    assert json.loads((OUT/'goal_direction_check.json').read_text())['system_goal_status']=='active'
    sources=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    sources += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_AUDIT_20261017.md','docs/ML_SANDBOX_COHERENT_HALL_THEORY_20261018.md',
        'docs/ML_SANDBOX_COHERENT_SOURCE_20261018.md','docs/ML_SANDBOX_COHERENT_PRIOR_SCOPE_20261018.md',
        'docs/ML_SANDBOX_TEMPERATURE_MOBILITY_20261018.md','docs/ML_SANDBOX_JOINT_N_MU_NEXT_20261018.md')]
    sources += [TABLE]+[BASE/p for p in ('CONTINUE_PHASE15_HERE.md','progress.json','goal_requirements_audit.json')]
    snap=BASE/'code_snapshot_phase15'
    for source in sources:
        dst=snap/source.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert energy.sha(dst)==energy.sha(source),dst
        else:shutil.copyfile(source,dst)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for source in BASE.glob('phase15*.log'):
        dst=logs/source.name
        if dst.exists():assert energy.sha(dst)==energy.sha(source),dst
        else:shutil.copyfile(source,dst)
    files=sorted([TABLE]+[p for folder in (OUT,snap) for p in folder.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'])
    payload=dict(created_utc=energy.now(),stage='coherent_Fermi_Hall_doping_property_prediction',goal_status='active',goal_complete=False,
        primary='unchanged29Si/Sn5DOI/0-1-2sampleanchors/common18',new_models=6,all_base_models=26,
        new_fit_records=90,new_optimizer_starts=180,actual_total_starts=606,composite_starts=602,
        numerical_checks=864,base_inner_scores=10320,source_variants=18,source_episodes=450,source_inner_scores=5400,
        independent_bundles=270,independent_carrier_rows=1566,paired_transport_replays=198,
        RT_mu_pairs=5,RT_mu_DOIs=2,old_temperature_mu_reads=54,temperature_mu_specimens=3,temperature_mu_DOIs=1,
        temperature_independent_rows=2916,scattering_ablation_rows=1270,new_sigma_labels=0,
        new_mu_supervision=False,fixed_empirical_mu_prior_disclosed=True,
        retained_failures='initialwrongstate/readoutgradientcolumn,coarseFDsecondderivative androundoffexpectation;notrainingbeforepass',
        stable_primary_increment_established=False,new_DFT_MD=0,new_material_measurements=0,
        next='thermalinput audit,record-boundedvs effectiveTf jointnH/mu;positiveextra_rate/outputhead tierBnotimplemented',
        parents_checked=checked,parent_sha256={name:energy.sha(BASE/name) for name in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in files])
    energy.write(DEST,payload)
    print('Frozen phase15',len(files));print('Manifest SHA256',energy.sha(DEST))


if __name__=='__main__':main()
