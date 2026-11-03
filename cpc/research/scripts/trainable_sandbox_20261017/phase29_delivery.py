"""Export full and source-excluded learned weights without property tables."""
from pathlib import Path
import sys,json,ast

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.kroger_db import NPZ
import phase2_energy as e
import phase29_data as data

OUT=data.OUT/'delivery'


def core_files():
    # Resolve actual local imports instead of an incomplete handwritten list.
    directory=ROOT/'src/sandbox';pending=['joint_property_predictor'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=directory/(name+'.py')
        if not path.exists():continue
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                if node.level==1:
                    if node.module:pending.append(node.module.split('.')[0])
                    else:pending.extend(a.name.split('.')[0] for a in node.names)
                elif node.module and node.module.startswith('sandbox.'):
                    pending.append(node.module.split('.')[1])
            elif isinstance(node,ast.Import):
                pending.extend(a.name.split('.')[1] for a in node.names if a.name.startswith('sandbox.'))
    paths=[directory/(name+'.py') for name in sorted(seen)]
    assert len(paths)>=20 and all(p.exists() for p in paths)
    paths.append(ROOT/'data/physics_reference/fermi_half_v1.npz')
    return paths


def main():
    assert not (OUT/'audit.json').exists(),'preserve previous export before versioned repair'
    prediction=data.OUT/'prediction'
    assert json.loads((prediction/'audit.json').read_text())['passed']
    assert json.loads((data.OUT/'api_gates_v1/gates.json').read_text())['passed']
    plans=json.loads((prediction/'plan.json').read_text())['plans']
    choices=json.loads((prediction/'source_choices.json').read_text())
    policy={(x['plan_id'],int(x['budget'])):x['selected']['lam'] for x in choices}
    code=[Path(__file__),Path(__file__).with_name('phase29_api_runtime.py'),*core_files()]
    e.write(OUT/'protocol_before_export.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (
            prediction/'plan.json',prediction/'source_choices.json',prediction/'audit.json',
            data.OUT/'api_gates_v1/gates.json')},
        no_new_training=True,no_property_tables_in_bundle=True,goal_complete=False))
    bundle=OUT/'predictor_bundle';presets={};models=set()
    for p in plans:
        model=p['minimal_fit'];key=e.sha(ROOT/p['fit_path'])
        identity=e.clean(dict(base_fit_sha256=key,fit=model))
        import hashlib
        name=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
        modelpath=bundle/'models'/(name+'.json')
        if modelpath.exists():assert json.loads(modelpath.read_text())==model
        else:e.write(modelpath,model)
        models.add(name)
        context='full' if p['outer_held'] is None else 'without:'+p['outer_held']
        preset='cluster::'+context+'::'+p['variant']
        presets[preset]=dict(variant=p['variant'],context=context,model_file='models/'+name+'.json',
            model_sha256=e.sha(modelpath),method=p['spec']['method'],source_kind=p['spec']['kind'],
            source_lambdas={str(b):policy[(p['id'],b)] for b in (0,1,2)},
            training_units=p['train_units'],global_config=p['config'],base_fit_path=p['fit_path'],
            base_fit_sha256=p['fit_sha256'],serving_qualified=bool(model['training_valid'] and model['selected_success']),
            transport_calibration=p['transport_head_audit']['head'],
            measured_process_inputs_complete=False,material_Born_validity_established=False)
    assert len(presets)==288
    manifest=dict(version='phase29_joint_scattering_v1',created_utc=e.now(),presets=presets,
        physical_reference_sha256=e.sha(NPZ),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in core_files()},
        conditional_domain='Si/Sn300K;explicit dose/process/H/C scenarios;0/1/2physicalHallanchors',
        source_scope='one source adaptation group per request;equal scenario counts percondition for positivebudget',
        training_data_summary=dict(reported_conditions=35,paired_conditions=11,DOIs=7,related_groups=5,
            original_common20_preserved=True,independent_sigma_labels=0),
        no_property_tables_copied=True,raw_thirdparty_catalogue_copied=False,
        no_blind_or_new_element_validation=True,goal_complete=False)
    e.write(bundle/'manifest.json',manifest)
    receipt=dict(passed=True,created_utc=e.now(),presets=len(presets),model_files=len(models),
        unqualified_presets=sum(not s['serving_qualified'] for s in presets.values()),
        bundle_manifest_sha256=e.sha(bundle/'manifest.json'),actual_new_training=0,
        experimental_or_training_tables_copied=False,API_replay_pending=True,goal_complete=False)
    e.write(OUT/'audit.json',receipt)
    print('Delivery29',receipt,flush=True)


if __name__=='__main__':main()
