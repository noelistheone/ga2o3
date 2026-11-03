"""Freeze conserved-inventory property predictions and verify every prior stage."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase17_inventory_learning';DEST=BASE/'artifact_manifest_phase17.json'
TABLE=ROOT/'data/physics_reference/fermi_half_v1.npz'


def verify(path):
    value=json.loads(path.read_text());rows=value['files'] if isinstance(value,dict) else value
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad
    return len(rows)


def main():
    if DEST.exists():print('Verified phase17',verify(DEST));return
    names=('protected_inputs_before.json','artifact_manifest_phase1.json','artifact_manifest_phase2.json',
        'artifact_manifest_phase34.json','artifact_manifest_phase56.json','artifact_manifest_phase7.json',
        'artifact_manifest_phase8.json','artifact_manifest_phase9.json','artifact_manifest_phase10.json',
        'artifact_manifest_phase11.json','artifact_manifest_phase12.json','artifact_manifest_phase13.json',
        'artifact_manifest_phase14.json','artifact_manifest_phase15.json','artifact_manifest_phase16.json')
    checked={name:verify(BASE/name) for name in names}
    assert energy.sha(BASE/'artifact_manifest_phase16.json')=='29b5b25ab75ac16798452e25a4b350123cf9001cf5927192871f0b4b71749465'
    gates=('inventory_audit/contract_audit.json','numerical_gates.json','validation/contract_audit.json',
        'research_runtime_v2/verification.json','property_evaluation/contract_audit.json','rendering_check.json')
    for rel in gates:assert json.loads((OUT/rel).read_text())['passed'],rel
    codes={'inventory_audit/protocol.json':'phase17_inventory_audit.py','matched_controls/protocol.json':'phase17_controls.py',
        'refinement_protocol.json':'phase17_refinement.py','family_selection_protocol.json':'phase17_family_selection.py',
        'research_runtime/protocol.json':'phase17_runtime.py','research_runtime_v2/protocol.json':'phase17_runtime_v2.py',
        'property_evaluation/protocol.json':'phase17_properties.py','report_claims.json':'phase17_report.py',
        'rendering_check.json':'phase17_rendering.py'}
    for rel in ('numerical_gates.json','prediction_protocol.json',*codes):
        p=json.loads((OUT/rel).read_text())
        for group in ('code_hashes','input_hashes','evidence_hashes'):
            for filename,digest in p.get(group,{}).items():assert energy.sha(ROOT/filename)==digest,(rel,filename)
        if rel in codes:assert energy.sha(ROOT/'scripts/trainable_sandbox_20261017'/codes[rel])==p['code_sha256'],rel
    refined=json.loads((OUT/'refinement_protocol.json').read_text())
    assert refined['prediction_protocol_sha256']==energy.sha(OUT/'prediction_protocol.json')
    assert refined['controls_protocol_sha256']==energy.sha(OUT/'matched_controls/protocol.json')
    a=json.loads((OUT/'validation/contract_audit.json').read_text())
    assert a['fit_records']==525 and a['initial_nonlinear_optimizer_starts']==630 and a['failed_start_V1_fits']==0
    assert a['inner_scores']==15900 and a['missing_primary_queries']==0 and a['total_methods']==31
    assert not a['final_any_unconverged'] and not a['selected_outer_best_unconverged'] and a['state_failures']==0
    r=json.loads((OUT/'research_runtime_v2/verification.json').read_text())
    assert r['saved_bundles']==465 and r['independent_rows']==21390
    assert not json.loads((OUT/'research_runtime_v2/independent_outputs.json').read_text())['blocked_paths']
    q=json.loads((OUT/'property_evaluation/contract_audit.json').read_text())
    assert q['ownT_mu_predictions']==5487 and q['transport_scope_states']==944 and q['removed_rows']==0
    initial=OUT/'development/initial_gate_failure'/OUT.relative_to(ROOT)/'numerical_gates.json'
    init=json.loads(initial.read_text());assert not init['passed'] and sum(not c['passed'] for c in init['checks'])==3
    failure=json.loads((OUT/'runtime_serialization_failure.json').read_text())
    assert energy.sha(ROOT/failure['original_code'])==failure['original_code_sha256'] and not failure['numerical_model_changed']
    direction=json.loads((OUT/'goal_direction_check.json').read_text())
    assert direction['system_goal_status']=='active' and not direction['goal_complete']
    library_sources=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    library_sources += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md',
        'docs/ML_SANDBOX_GOAL_AUDIT_20261017.md','docs/ML_SANDBOX_INVENTORY_LEARNING_20261018.md',
        'docs/ML_SANDBOX_INVENTORY_COMPARISON_20261018.md','docs/ML_SANDBOX_LOG_ODDS_ALLOCATION_NEXT_20261018.md')]
    library_sources += [TABLE]+[BASE/p for p in ('CONTINUE_PHASE17_HERE.md','progress.json','goal_requirements_audit.json',
        'phase17_prediction.log','phase17_controls.log','phase17_runtime.log','phase17_runtime_v2.log',
        'phase17_properties.log','phase17_report.log','phase17_pdf.log') if (BASE/p).exists()]
    snapshot=BASE/'code_snapshot_phase17'
    for source in library_sources:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(source)==energy.sha(target),target
        else:shutil.copyfile(source,target)
    files=sorted([TABLE]+[p for folder in (OUT,snapshot) for p in folder.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'])
    energy.write(DEST,dict(created_utc=energy.now(),stage='trainable_conserved_inventory_doped_property_prediction',
        goal_status='active',goal_complete=False,primary='unchanged29Si/Sn5DOI/0-1-2nHspecimenanchors/common18',
        methods=31,new_nonlinear_fits=315,new_optimizer_starts=630,new_convex_fits=180,references=30,records=525,
        numerical_checks=1469,retained_initial_failed_checks=3,inner_scores=15900,
        independent_bundles=465,independent_rows=21390,retained_JSON_serialization_failure=True,
        ownT_mu_rows=5487,RTmu_pairs=5,RTmu_DOIs=2,muT_reads=54,muT_specimens=3,muT_DOIs=1,
        transport_scope_states=944,majority_sigma_kl_lt1_states=458,independent_sigma_labels=0,
        universal_prediction_gain_established=False,material_Born_validity_established=False,
        next='newversion log-odds inventory allocation plusmatched direct/output/oldstrongcontrols;notimplemented',
        new_DFT_MD=0,new_material_measurements=0,parents_checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in files]))
    print('Frozen phase17',len(files),'parent entries',sum(checked.values()));print('Manifest SHA256',energy.sha(DEST))


if __name__=='__main__':main()
