"""Freeze log-odds prediction evidence, including failed runs and every parent."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
BASE=energy.BASE;OUT=BASE/'phase18_logodds_prediction';DEST=BASE/'artifact_manifest_phase18.json';TABLE=ROOT/'data/physics_reference/fermi_half_v1.npz'


def verify(path):
    value=json.loads(path.read_text());rows=value['files'] if isinstance(value,dict) else value
    bad=[r['path'] for r in rows if not (ROOT/r['path']).is_file() or energy.sha(ROOT/r['path'])!=r['sha256']]
    assert not bad,bad
    return len(rows)


def main():
    if DEST.exists():print('Verified phase18',verify(DEST));return
    names=('protected_inputs_before.json','artifact_manifest_phase1.json','artifact_manifest_phase2.json',
        'artifact_manifest_phase34.json','artifact_manifest_phase56.json','artifact_manifest_phase7.json',
        'artifact_manifest_phase8.json','artifact_manifest_phase9.json','artifact_manifest_phase10.json',
        'artifact_manifest_phase11.json','artifact_manifest_phase12.json','artifact_manifest_phase13.json',
        'artifact_manifest_phase14.json','artifact_manifest_phase15.json','artifact_manifest_phase16.json','artifact_manifest_phase17.json')
    checked={n:verify(BASE/n) for n in names}
    assert energy.sha(BASE/'artifact_manifest_phase17.json')=='b4a4ed52c2266357cdb1ae6b2e8a6cc7f23f07b452f9c83d6c8694d24df7256f'
    gates=('numerical_gates.json','validation/contract_audit.json','source_adaptation_v2/validation/contract_audit.json',
        'research_runtime_v2/verification.json','property_evaluation/contract_audit.json','predictor_scope_gates.json',
        'relation_diagnostic.json','source_selected_update_audit.json','temperature_anchor_protocol_audit.json','rendering_check.json','visual_review.json')
    for rel in gates:assert json.loads((OUT/rel).read_text())['passed'],rel
    codes={'refinement_protocol.json':'phase18_refinement.py','source_adaptation/source_protocol.json':'phase18_source_adaptation.py',
        'source_adaptation_v2/source_protocol.json':'phase18_source_adaptation_v2.py','research_runtime/protocol.json':'phase18_runtime.py',
        'research_runtime_v2/protocol.json':'phase18_runtime_v2.py','property_evaluation/protocol.json':'phase18_properties.py',
        'predictor_scope_gates.json':'phase18_predictor_scope.py','relation_diagnostic.json':'phase18_relation_diagnostic.py',
        'source_selected_update_audit.json':'phase18_selected_update_audit.py','report_claims.json':'phase18_report.py','rendering_check.json':'phase18_rendering.py',
        'validation/contract_audit.json':'phase18_audit.py','source_adaptation_v2/validation/contract_audit.json':'phase18_audit.py'}
    for rel in ('numerical_gates.json','prediction_protocol.json',*codes):
        value=json.loads((OUT/rel).read_text())
        for group in ('code_hashes','input_hashes','evidence_hashes'):
            for path,h in value.get(group,{}).items():assert energy.sha(ROOT/path)==h,(rel,path)
        if rel in codes:
            path='scripts/trainable_sandbox_20261017/'+codes[rel]
            expected=value['code_sha256'] if 'code_sha256' in value else value['code_hashes'][path]
            assert energy.sha(ROOT/path)==expected,rel
    t=json.loads((OUT/'temperature_anchor_protocol_audit.json').read_text());assert energy.sha(ROOT/t['authoritative_source'])==t['source_sha256']
    v=json.loads((OUT/'visual_review.json').read_text());assert energy.sha(OUT/'ML与Sandbox_分配学习与来源泛化检验.pdf')==v['pdf_sha256']
    a=json.loads((OUT/'validation/contract_audit.json').read_text());s=json.loads((OUT/'source_adaptation_v2/validation/contract_audit.json').read_text())
    assert a['records']==120 and a['initial_new_starts']==210 and a['initial_failed_start_fits']==0
    assert a['inner_scores']==2400 and not a['selected_outer_nonconverged'] and a['missing_primary_queries']==0
    assert s['episodes']==300 and s['inner_scores']==1800 and s['source_prediction_callbacks']==26171 and not s['failed_source_probes']
    r=json.loads((OUT/'research_runtime_v2/verification.json').read_text());assert r['saved_bundles']==210 and r['independent_rows']==9660
    assert not json.loads((OUT/'research_runtime_v2/independent_outputs.json').read_text())['blocked_paths']
    initial=OUT/'development/initial_gate_failure'/OUT.relative_to(ROOT)/'numerical_gates.json'
    j=json.loads(initial.read_text());assert not j['passed'] and len(j['checks'])==662 and sum(not c['passed'] for c in j['checks'])==2
    failure=json.loads((OUT/'runtime_print_failure.json').read_text());assert energy.sha(ROOT/failure['original_code'])==failure['original_sha256']
    direction=json.loads((OUT/'goal_direction_check.json').read_text());assert direction['system_goal_status']=='active' and not direction['goal_complete']
    source_paths=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    source_paths += [ROOT/p for p in ('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md','docs/ML_SANDBOX_GOAL_AUDIT_20261017.md',
        'docs/ML_SANDBOX_LOG_ODDS_ALLOCATION_NEXT_20261018.md','docs/ML_SANDBOX_LOGODDS_PREDICTION_20261018.md',
        'docs/ML_SANDBOX_TEMPERATURE_ANCHOR_PROTOCOLS_20261018.md','docs/ML_SANDBOX_GENERALIZATION_NEXT_20261018.md')]
    source_paths += [TABLE]+[BASE/p for p in ('CONTINUE_PHASE18_HERE.md','progress.json','goal_requirements_audit.json')]+[p for p in BASE.glob('phase18*.log') if p.name!='phase18_freeze.log']
    snapshot=BASE/'code_snapshot_phase18'
    for source in source_paths:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert energy.sha(target)==energy.sha(source),target
        else:shutil.copyfile(source,target)
    files=sorted([TABLE]+[p for folder in (OUT,snapshot) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'])
    energy.write(DEST,dict(created_utc=energy.now(),stage='shared_logodds_inventory_and_internal_source_property_prediction',goal_status='active',goal_complete=False,
        primary='unchanged29Si/Sn5DOI/0-1-2RTspecimenanchors/common18',base_methods=8,source_variants=6,new_fits=105,new_starts=210,references=15,
        source_episodes=300,source_penalty_solutions=2100,source_prediction_callbacks=26171,base_inner_scores=2400,source_inner_scores=1800,
        numerical_checks=680,retained_initial_failed_checks=2,predictor_domain_checks=24,independent_bundles=210,independent_rows=9660,
        one_anchor_infinite_lambda_cases=27,temperature_protocols_explicitly_separated=True,RTmu_pairs=5,RTmu_DOIs=2,muT_reads=54,muT_specimens=3,muT_DOIs=1,
        independent_sigma_labels=0,ownT_mu_rows=2478,RTspecimenbudget_nT_rows=5964,transport_scope_states=472,majority_sigma_kl_lt1_states=119,
        universal_prediction_gain_established=False,full5source_deployment_delivered=False,new_DFT_MD=0,new_material_measurements=0,
        next='fullgoal audit;train/export bestsupported full5source predictor/domain;independentactualdoseHall sourceaudit;no automaticcapacitygrowth',
        parents_checked=checked,parent_sha256={n:energy.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=energy.sha(p)) for p in files]))
    print('Frozen phase18',len(files),'parent entries',sum(checked.values()));print('Manifest SHA256',energy.sha(DEST))


if __name__=='__main__':main()
