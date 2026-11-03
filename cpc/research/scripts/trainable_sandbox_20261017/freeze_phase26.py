"""Freeze actual phase26 prediction evidence after completion and visual review."""
from pathlib import Path
import sys,json,shutil
import numpy,scipy,pandas,matplotlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from freeze_phase21 import verify
from phase26_report import AUDITS
import phase2_energy as e

BASE=e.BASE;OUT=BASE/'phase26_pool_share';DEST=BASE/'artifact_manifest_phase26.json'


def live_stage_jobs():
    names={'phase26_training.py','phase26_after_training.py','phase26_prediction.py','phase26_after_prediction.py',
        'phase26_source_runtime.py','phase26_cohort_replay.py','phase26_api_runtime.py','phase26_external.py',
        'phase26_after_replay.py','phase26_after_transfer.py','phase26_analysis.py','phase26_physical_scope.py',
        'phase26_delivery_final_check.py','phase26_inner_domain_diagnosis.py'}
    jobs=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\x00')
            if len(argv)>1 and Path(argv[0].decode()).name.startswith('python') and Path(argv[1].decode()).name in names:
                jobs.append(dict(pid=int(proc.name),script=argv[1].decode()))
        except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeDecodeError):pass
    return jobs


def main():
    if DEST.exists():print('Verified frozen phase26',verify(DEST));return
    live=live_stage_jobs();assert not live,('Actual stage jobs still live;wait rather than freezing changing files',live)
    names=['protected_inputs_before.json']+[f'artifact_manifest_phase{i}.json' for i in
        (1,2,34,56,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25)]
    counts={n:verify(BASE/n) for n in names};assert sum(counts.values())==50233
    assert e.sha(BASE/'artifact_manifest_phase25.json')=='15003caddff0bdf6c2048a98929c2d185d05e2e56dc7d03f29ce13fc58a5d500'
    required=(*AUDITS,'delivery/documentation_audit.json','information_audit/audit.json','theory/visual_review.json',
        'theory_scope_readback.json','rendering_check.json','visual_review.json','fallback_equivalence/audit.json','inner_domain_diagnosis/audit.json')
    for rel in required:
        d=json.loads((OUT/rel).read_text());assert d['passed'],rel
        for kind in ('code_hashes','input_hashes'):
            for name,h in d.get(kind,{}).items():assert e.sha(ROOT/name)==h,(rel,name)
    for rel in ('training_protocol.json','pre_evaluation_protocol.json','post_prediction_predeclared.json',
        'prediction/protocol.json','delivery/cohort_replay/protocol.json','external_Azizie/protocol_before_transfer.json','analysis_predeclared.json','fallback_equivalence/protocol_before_comparison.json','inner_domain_diagnosis/protocol_before_replay.json'):
        d=json.loads((OUT/rel).read_text())
        for kind in ('code_hashes','input_hashes','hashes','inherited_parent_fit_sha256','files'):
            for name,h in d.get(kind,{}).items():assert e.sha(ROOT/name)==h,(rel,name)
    g=json.loads((OUT/'goal_direction_check.json').read_text());assert g['system_goal_status']=='active' and not g['goal_complete']
    assert g['new_property_rows']==3366 and g['combined_property_rows']==7128 and g['main_variants']==36
    assert g['API_point_rows']==7128 and g['external_rows']==4320
    assert (OUT/'learning_gate_failure_v1').is_dir() and not json.loads((OUT/'learning_gates.json').read_text())['passed']
    theory=json.loads((OUT/'theory_scope_readback.json').read_text());assert e.sha(ROOT/theory['theory_path'])==theory['theory_sha256']
    claims=json.loads((OUT/'report_claims.json').read_text());report=OUT/'ML与Sandbox_绝对份额学习与性质预测.md'
    assert e.sha(report)==claims['report_sha256'] and e.sha(ROOT/claims['theory_path'])==claims['theory_sha256']
    render=json.loads((OUT/'rendering_check.json').read_text());assert e.sha(report.with_suffix('.pdf'))==render['pdf_sha256']
    bundle=OUT/'delivery/predictor_bundle';m=json.loads((bundle/'manifest.json').read_text());assert len(m['presets'])==468
    for f in m['files']:assert e.sha(bundle/f['path'])==f['sha256'],f['path']
    e.write(OUT/'parent_verification_at_freeze.json',dict(passed=True,entries=sum(counts.values()),checked=counts,
        parent_sha256={n:e.sha(BASE/n) for n in names},created_utc=e.now(),changed_parent_files=0))
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for name in ('ga2o3_phase26_training_v1.log','ga2o3_phase26_after_training.log','ga2o3_phase26_after_prediction.log',
        'ga2o3_phase26_after_replay.log','ga2o3_phase26_after_transfer.log','ga2o3_phase26_theory_pdf.log','ga2o3_phase26_report_pdf.log','ga2o3_phase26_inner_domain.log'):
        source=Path('/tmp')/name
        if not source.exists():continue
        target=logs/name
        if target.exists():assert e.sha(target)==e.sha(source),target
        else:shutil.copyfile(source,target)
    e.write(OUT/'environment_at_freeze.json',dict(python=sys.version,python_executable=sys.executable,
        numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,matplotlib=matplotlib.__version__,
        GPU_used=False,external_physical_reference_in_bundle=False,goal_complete=False))
    for name in ('progress.json','goal_requirements_audit.json'):
        path=BASE/name;d=json.loads(path.read_text());d['phase26']['frozen']=True;e.write(path,d)
    files=list((ROOT/'src/sandbox').glob('*.py'))+list((ROOT/'scripts/trainable_sandbox_20261017').glob('*.py'))
    docs=('README.md','AGENTS.md','docs/ML_SANDBOX_GOAL_20261017.md','docs/ML_SANDBOX_GOAL_CHECK_PHASE26_20261018.md',
        'docs/ML_SANDBOX_GOAL_DIRECTION_RECONFIRM_20261018.md','docs/ML_SANDBOX_POOL_SHARE_PROTOCOL_20261018.md',
        'docs/ML_SANDBOX_POOL_SHARE_ATTRIBUTION_ADDENDUM_20261018.md','docs/ML_SANDBOX_POOL_SHARE_THEORY_20261018.md',
        'docs/ML_SANDBOX_CONTEXT_PRIOR_ART_20261018.md')
    files+=[ROOT/p for p in docs]
    nextdoc=g['prediction_conclusions'].get('next_theory')
    if nextdoc:files.append(ROOT/nextdoc)
    files+=[BASE/p for p in ('progress.json','goal_requirements_audit.json','CONTINUE_PHASE26_HERE.md','CONTINUE_PHASE26_WORKING.md')]
    snapshot=BASE/'code_snapshot_phase26'
    for source in files:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert e.sha(target)==e.sha(source),target
        else:shutil.copyfile(source,target)
    frozen=sorted(p for folder in (OUT,snapshot) for p in folder.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'locks' not in p.parts and p.suffix!='.pyc')
    payload=dict(created_utc=e.now(),stage='absolute_atomshare_learning_and_full_dopant_property_prediction',
        goal_status='active',goal_complete=False,conditions=33,DOIs=6,related_groups=5,
        training_fit_records=g['training_fit_records'],actual_optimizer_starts=g['optimizer_starts'],
        new_variants=17,combined_variants=36,new_property_rows=3366,combined_property_rows=7128,
        API_point_rows=7128,external_property_rows=4320,unqualified_models_and_adverse_results_retained=True,
        original29_and_roles_unchanged=True,new_DFT_MD=0,new_material_measurements=0,
        parent_entries_checked=counts,parent_sha256={n:e.sha(BASE/n) for n in names},
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=e.sha(p)) for p in frozen])
    e.write(DEST,payload);digest=e.sha(DEST)
    for name in ('progress.json','goal_requirements_audit.json'):
        path=BASE/name;d=json.loads(path.read_text());d['phase26_artifact_count']=len(frozen);d['phase26_manifest_sha256']=digest
        d['phase26'].update(artifact_count=len(frozen),manifest_sha256=digest);e.write(path,d)
    print('Frozen actual phase26',len(frozen),'files;parent entries',sum(counts.values()),'SHA256',digest,flush=True)


if __name__=='__main__':main()
