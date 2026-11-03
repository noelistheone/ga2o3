"""Export every selected new model and the unchanged19 legacy controls."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.kroger_db import NPZ
import phase26_registry as t
import phase2_energy as e


def main():
    out=t.OUT/'delivery/predictor_bundle';pred=t.OUT/'prediction';old=e.BASE/'phase25_shared_process/delivery/predictor_bundle'
    assert json.loads((pred/'audit.json').read_text())['passed']
    done=t.OUT/'delivery/export_complete.json'
    if done.exists():
        m=json.loads((out/'manifest.json').read_text())
        for f in m['files']:assert e.sha(out/f['path'])==f['sha256']
        assert len(m['presets'])==468;print('Verified existing pool/context bundle',flush=True);return
    parent=json.loads((old/'manifest.json').read_text());presets={};files={}
    for f in parent['files']:
        source=old/f['path'];assert e.sha(source)==f['sha256'];path=out/f['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert e.sha(path)==f['sha256']
        else:shutil.copyfile(source,path)
        files[f['path']]=dict(path=f['path'],sha256=e.sha(path),model_type='legacy')
    for name,spec in parent['presets'].items():
        f=json.loads((out/'models'/(spec['model_key']+'.json')).read_text())
        presets[name]=dict(spec,model_type='legacy',serving_qualified=bool(f['training_valid'] and f.get('selected_success',True)))
    choices=json.loads((pred/'selections.json').read_text())
    minimal=('key','method','evaluation_method','parameters','reference_fallback','training_valid','selected_success')
    for choice in choices:
        scheme=choice['scheme'];held=choice['held_unit'];context=scheme+('::full' if held is None else '::without:'+held)
        name=context+'::'+choice['variant'];path=ROOT/choice['fit_path'];assert e.sha(path)==choice['fit_sha256']
        f=json.loads(path.read_text());target=out/'models'/(f['key']+'.json')
        if str(target.relative_to(out)) in files:assert files[str(target.relative_to(out))]['model_type']=='pool_share'
        else:
            e.write(target,{k:f[k] for k in minimal if k in f})
            files[str(target.relative_to(out))]=dict(path=str(target.relative_to(out)),sha256=e.sha(target),model_type='pool_share')
        if name not in presets:
            tr=t.subset(scheme,choice['train_units'])
            support={element:dict(dose_cm3=[float(g.incorporated_cm3.min()),float(g.incorporated_cm3.max())],
                growth_T_C=[float(g.growth_T_C.min()),float(g.growth_T_C.max())],reported_H_inputs=int(g.H_known.sum())) for element,g in tr.groupby('element')}
            presets[name]=dict(context=context,variant=choice['variant'],method=choice['method'],model_type='pool_share',
                source_kind=choice['kind'],model_key=f['key'],shared_reg=choice['reg'],source_lambdas={},training_support=support,
                train_families=f['train_families'],held_unit=held,validation_scheme=scheme,
                actual_evaluation_method=f['evaluation_method'],explicit_reference_fallback=bool(f.get('reference_fallback')),
                serving_qualified=bool(f['training_valid'] and f['selected_success']),
                scope='retrospective33condition study;invalid/unconverged fits retained but refused by serving interface')
        presets[name]['source_lambdas'][str(choice['budget'])]=choice['selected_source']['lam']
    assert len(presets)==468 and sum(s['model_type']=='pool_share' for s in presets.values())==221
    assert all(set(p['source_lambdas'])=={'0','1','2'} for p in presets.values())
    profile=published_coherent_profile()
    value=dict(version=1,created_utc=e.now(),framework='mixed_pool_share_context_v1',presets=presets,files=list(files.values()),
        profile=profile.to_dict(),profile_sha256=profile.fingerprint,physical_reference_sha256=e.sha(NPZ),
        parent_bundle_sha256=e.sha(old/'manifest.json'),third_party_reference_copied=False,conditions=33,DOIs=6,related_groups=5,
        primary='ShareEffectiveProcess_internal',retrospective=True,independent_blind_or_lab_validation=False,
        new_mu_supervision=False,independent_sigma_labels=0,no_calibrated_predictive_intervals=True,goal_complete=False)
    e.write(out/'manifest.json',value)
    e.write(done,dict(created_utc=e.now(),presets=len(presets),models=len(files),new_presets=221,inherited_presets=247,
        unqualified_presets=sum(not s['serving_qualified'] for s in presets.values()),parameters_refitted_in_export=False,goal_complete=False))
    print('Exported mixed property predictor',len(presets),len(files),'models;unqualified',sum(not s['serving_qualified'] for s in presets.values()),flush=True)


if __name__=='__main__':main()
