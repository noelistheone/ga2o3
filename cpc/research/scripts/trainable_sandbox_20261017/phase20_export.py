"""Train selected full-source models and export table-free property inference."""
from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.kroger_db import load
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.trainable_predictor_v2 import structure_fingerprint,supplied_energy_fingerprint
import phase20_source as source
import phase20_reference_control as budget_control
s=source.s;OUT=s.OUT/'delivery';BUNDLE=OUT/'predictor_bundle_v6'


def freeze():
    assert json.loads((source.OUT/'execution_complete.json').read_text())['variants']==9
    assert json.loads((budget_control.OUT/'execution_complete.json').read_text())['newfits']==15
    paths=(Path(__file__),ROOT/'src/sandbox/relative_predictor.py',ROOT/'src/sandbox/relative_inventory_model.py',ROOT/'src/sandbox/observation_adaptation.py')
    inputs=(s.OUT/'prediction_protocol.json',source.OUT/'source_protocol.json',source.OUT/'execution_complete.json',
        s.BASE/'predictor_bundle_v4/manifest.json',s.BASE/'dataset_v1.csv',budget_control.OUT/'protocol.json',budget_control.OUT/'execution_complete.json')
    data=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in inputs},
        training='all29oldspecimens5sources;gamma and sourcehyperparams chosen usingonlyoldLOFO predictions;new-source labels not read',
        score_scope='full-source fits are deployment training,not a new heldout accuracy test',
        temporal='sourceanchorsRTonly;queryTmaydiffer,priors transferredwithoutnewTlabels',
        inherited='all7 earlier explicit-budget energycompletion modes retained',goal_complete=False)
    p=OUT/'export_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==data[k],k
    else:s.energy.write(p,data)


def source_choice(method,reg,kind,budget,*,reference_control=False):
    if budget==0:return dict(prior='quadratic',lam=None),[]
    episodes=[budget_control.episode(budget_control.fit([f for f in s.FAMILIES if f!=held]),held,budget)
        if reference_control else source.episode(s.fit([f for f in s.FAMILIES if f!=held],method,reg),held,kind,budget) for held in s.FAMILIES]
    rows=[]
    for prior in source.PRIORS:
        for lam in source.LAM:
            scores=[source.value(source.pick(ep,prior,lam)['MAE']) for ep in episodes]
            rows.append(dict(prior=prior,lam=lam,score=float(np.mean(scores)),inner_scores=scores,episode_keys=[e['key'] for e in episodes]))
    best=min(rows,key=lambda r:(round(source.value(r['score']),10),-source.value(r['lam']),0 if r['prior']=='quadratic' else 1))
    return dict(prior=best['prior'],lam=best['lam']),rows


def main():
    freeze();clock=time.perf_counter();db=load();variants={};records={};global_choices={};selection=[]
    for method in ('inventory','output','ridge','median','line','unit','unitH'):
        reg,candidates=s.select_global(list(s.FAMILIES),method);global_choices[method]=dict(reg=reg,candidates=candidates)
    allvariants={**source.VARIANTS,'OriginalBudget':('inventory','offset')}
    for variant,(method,kind) in allvariants.items():
        reg=None if variant in ('OriginalHall','OriginalBudget') else global_choices[method]['reg']
        f=budget_control.fit(list(s.FAMILIES)) if variant=='OriginalBudget' else s.fit(list(s.FAMILIES),method,reg)
        if not f['training_valid']:raise ValueError('Selected full-source training state invalid')
        minimal={k:f[k] for k in ('key','method','parameters','coefficients','intercept','center','scale','offset_dex','reg','train_families','exact_reference_fallback','reference_search_control','training_valid') if k in f}
        path=BUNDLE/'models'/f"{f['key']}.json";s.energy.write(path,minimal);records[f['key']]=path
        chosen={}
        for budget in (0,1,2):
            best,candidates=source_choice(method,reg,kind,budget,reference_control=variant=='OriginalBudget');chosen[str(budget)]=best
            selection.append(dict(variant=variant,budget=budget,global_reg=reg,selected=best,candidates=candidates))
        variants[variant]=dict(model_key=f['key'],method=method,global_reg=reg,source_kind=kind,source_selection=chosen)
    parent=s.BASE/'predictor_bundle_v4';old=json.loads((parent/'manifest.json').read_text())
    manifest=dict(version=6,created_utc=s.energy.now(),profile=published_coherent_profile().to_dict(),
        structure_fingerprint=structure_fingerprint(db),full_energy_fingerprint=supplied_energy_fingerprint(db),
        files=[dict(path=str(p.relative_to(BUNDLE)),sha256=s.energy.sha(p)) for p in records.values()],
        variants=variants,energy_methods=old['methods'],energy_bundle_relative=os.path.relpath(parent,BUNDLE),
        energy_manifest_sha256=s.energy.sha(parent/'manifest.json'),
        chemical_support_cm3={e:[float(g.incorporated_cm3.min()),float(g.incorporated_cm3.max())] for e,g in s.RT.groupby('element')},
        validation_grade='retrospective old29/5development;selectedfullsource fit isnot newvalidation',
        training_labels=dict(nH_RT_specimens=29,sources=5,nH_temperature=0,mobility=0),
        physical_scope='all873energies;nativefixed,foreign/Oconserved,Gaopen;effectiveTf;coherentFD/205empiricalextra/BornRTA',
        energy_completion_scope='inherited suppliesSi844orSn850labels plus declared0or2;not unknown-elementdiscovery',
        uncertainty='no calibratedpredictioninterval;sourcepenaltydesignconditional,not a microscopicposterior',
        export_protocol_sha256=s.energy.sha(OUT/'export_protocol.json'),goal_complete=False)
    s.energy.write(BUNDLE/'manifest.json',manifest);s.energy.write(OUT/'global_selection.json',global_choices)
    s.energy.write(OUT/'source_selection.json',selection)
    s.energy.write(OUT/'export_complete.json',dict(created_utc=s.energy.now(),model_records=len(records),relative_variants=len(variants),
        inherited_energy_methods=len(old['methods']),seconds=time.perf_counter()-clock,goal_complete=False))
    print('exported',len(variants)+len(old['methods']),'presets',len(records),'newmodelrecords',flush=True)


if __name__=='__main__':main()
