"""Export frozen full-source weights with newly trained source hyperpolicies."""
from pathlib import Path
import sys,json,copy,shutil,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import phase21_prediction as s
energy=s.energy;OUT=s.OUT/'delivery';BUNDLE=OUT/'predictor_bundle_v8';PARENT=s.PARENT/'delivery_v2/predictor_bundle_v7'


def build(bundle,selection,*,grade,energy_methods=True):
    old=json.loads((PARENT/'manifest.json').read_text());variants={};files={};bundle.mkdir(parents=True,exist_ok=True)
    for variant,choices in selection.items():
        c=choices[0];key=c['fit_key'];src=PARENT/'models'/f'{key}.json';dest=bundle/'models'/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert energy.sha(src)==energy.sha(dest)
        else:shutil.copyfile(src,dest)
        files[key]=dict(path=str(dest.relative_to(bundle)),sha256=energy.sha(dest))
        kind='field' if variant in s.FIELD_MODES else 'offset';pen,safe=s.FIELD_MODES[variant] if kind=='field' else ('curve',True)
        variants[variant]=dict(model_key=key,method=json.loads(src.read_text())['method'],global_reg=c['reg'],source_kind=kind,
            mode=variant if kind=='field' else 'OutputSafe',penalty=pen,safe=safe,
            source_selection={str(v['budget']):dict(prior=v['selected_source']['prior'],lam=v['selected_source']['lam']) for v in choices})
    m={k:copy.deepcopy(old[k]) for k in ('profile','structure_fingerprint','full_energy_fingerprint','chemical_support_cm3','training_labels')}
    m.update(version=8,created_utc=energy.now(),files=list(files.values()),variants=variants,
        energy_methods=old['energy_methods'] if energy_methods else [],parent_bundle_relative=os.path.relpath(PARENT,bundle),
        parent_manifest_sha256=energy.sha(PARENT/'manifest.json'),
        source_scope='explicitunlabeledfiniteRTdesign;input-setconditional;0/1/2indexedRTanchors;notcontinuousvalidity',
        validation_grade=grade,new_shared_optimizer_starts=0,goal_complete=False)
    energy.write(bundle/'manifest.json',m);return m


def main():
    complete=json.loads((s.OUT/'execution_complete.json').read_text());assert complete['variants']==14
    p=dict(created_utc=energy.now(),code_hashes={str(x.relative_to(ROOT)):energy.sha(x) for x in
        (Path(__file__),ROOT/'src/sandbox/functional_predictor.py',ROOT/'src/sandbox/functional_source.py')},
        input_hashes={str(x.relative_to(ROOT)):energy.sha(x) for x in (s.OUT/'prediction_protocol.json',s.OUT/'execution_complete.json',PARENT/'manifest.json')},
        action='copy all8fullsourceweightrecords byteexact;newsourcehyperparameters only',new_training_labels=0,new_shared_optimizer_starts=0,goal_complete=False)
    path=OUT/'export_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==p[k]
    else:energy.write(path,p)
    selection={v:json.loads((s.OUT/'deployment_selection'/f'{v}.json').read_text()) for v in s.VARIANTS}
    m=build(BUNDLE,selection,grade='allold29/5fit;sourcepolicychosenoldLOFO;developmentonly')
    energy.write(OUT/'export_complete.json',dict(created_utc=energy.now(),model_records=len(m['files']),presets=len(m['variants'])+len(m['energy_methods']),
        all_weights_unchanged=True,new_shared_optimizer_starts=0,goal_complete=False))
    print('exported',len(m['variants'])+len(m['energy_methods']),'presets',flush=True)


if __name__=='__main__':main()
