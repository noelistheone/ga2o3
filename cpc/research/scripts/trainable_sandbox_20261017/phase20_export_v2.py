"""Reuse all V1 full-source weights; reselect only repaired internal adaptation."""
from pathlib import Path
import sys,json,copy,os,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase20_source_v2 as repaired
base=repaired.base;s=repaired.s;OUT=s.OUT/'delivery_v2';BUNDLE=OUT/'predictor_bundle_v7';PARENT=s.OUT/'delivery'


def freeze():
    assert json.loads((repaired.OUT/'execution_complete.json').read_text())['recomputed_variants']==1
    paths=(Path(__file__),ROOT/'src/sandbox/relative_predictor_v2.py',ROOT/'src/sandbox/observation_adaptation_v2.py',Path(repaired.__file__))
    inputs=(PARENT/'predictor_bundle_v6/manifest.json',PARENT/'source_selection.json',PARENT/'global_selection.json',
        repaired.OUT/'source_protocol.json',repaired.OUT/'execution_complete.json')
    payload=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in inputs},
        action='reuse all8full5 modelrecords byteexact;reselect onlyRelativeInventory sourcepriors using repaired sameoperator',
        new_optimizer_starts=0,new_training_labels=0,version=7,goal_complete=False)
    p=OUT/'export_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==payload[k]
    else:s.energy.write(p,payload)


def main():
    freeze();parent=PARENT/'predictor_bundle_v6';manifest=json.loads((parent/'manifest.json').read_text());manifest=copy.deepcopy(manifest)
    BUNDLE.mkdir(parents=True,exist_ok=True)
    for entry in manifest['files']:
        src=parent/entry['path'];dest=BUNDLE/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert s.energy.sha(dest)==entry['sha256']
        else:shutil.copyfile(src,dest)
    variant=manifest['variants']['RelativeInventory'];reg=variant['global_reg'];selection=[]
    for budget in (0,1,2):
        if budget==0:best=dict(prior='quadratic',lam=None);candidates=[]
        else:
            episodes=[base.episode(s.fit([f for f in s.FAMILIES if f!=held],'inventory',reg),held,'field',budget) for held in s.FAMILIES]
            candidates=[]
            for prior in base.PRIORS:
                for lam in base.LAM:
                    scores=[base.value(base.pick(e,prior,lam)['MAE']) for e in episodes]
                    candidates.append(dict(prior=prior,lam=lam,score=float(np.mean(scores)),inner_scores=scores,episode_keys=[e['key'] for e in episodes]))
            chosen=min(candidates,key=lambda r:(round(base.value(r['score']),10),-base.value(r['lam']),0 if r['prior']=='quadratic' else 1))
            best=dict(prior=chosen['prior'],lam=chosen['lam'])
        variant['source_selection'][str(budget)]=best
        selection.append(dict(variant='RelativeInventory',budget=budget,global_reg=reg,selected=best,candidates=candidates))
    inherited=[r for r in json.loads((PARENT/'source_selection.json').read_text()) if r['variant']!='RelativeInventory']
    s.energy.write(OUT/'source_selection.json',inherited+selection)
    manifest.update(version=7,created_utc=s.energy.now(),export_protocol_sha256=s.energy.sha(OUT/'export_protocol.json'),
        parent_model_manifest_sha256=s.energy.sha(parent/'manifest.json'),
        energy_bundle_relative=os.path.relpath(s.BASE/'predictor_bundle_v4',BUNDLE),
        inverse_solver_revision='exactcachedroot beforestrictbracketing;unchangedequations/tolerances',
        new_training_labels=0,new_optimizer_starts=0)
    s.energy.write(BUNDLE/'manifest.json',manifest)
    s.energy.write(OUT/'export_complete.json',dict(created_utc=s.energy.now(),models=len(manifest['files']),
        total_presets=len(manifest['variants'])+len(manifest['energy_methods']),
        all_weights_unchanged=True,new_optimizer_starts=0,goal_complete=False))
    print('version7 delivered',len(manifest['variants'])+len(manifest['energy_methods']),'presets',variant['source_selection'],flush=True)


if __name__=='__main__':main()
