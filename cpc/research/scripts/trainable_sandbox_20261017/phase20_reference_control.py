"""Generous optimizer-budget physical reference, added before outer scores.

The inventory method uses10new starts per training context over5 finite gamma
values, with up to2160 state-table evaluations (its response penalty needs a
second state for4gamma values). This fixed-h reference gets10new starts with
max_nfev216 plus the identical inherited old optimum. No new physical field.
"""
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.optimize import least_squares
import phase20_source as source
s=source.s;OUT=s.OUT/'reference_budget_control'


def freeze():
    paths=(Path(__file__),ROOT/'src/sandbox/relative_inventory_model.py')
    p=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in paths},
        parent_prediction_protocol_sha256=s.energy.sha(s.OUT/'prediction_protocol.json'),
        amendment='additional optimizer-budget control frozen before inspecting primaryouter scores',
        old_reference='same15sourceexcluded fits plus full5 deployment reference',
        starts='T850/1100/1400/1750/1900K crossedwithlogB15.5/16.5;10newstarts216nfev eachplus oldoptimum',
        maximum_new_state_table_evaluations_per_context=2160,
        objective='same sourceequal nH squared loss;hfixed0;no newphysicaldegreesoffreedom',
        source='same observableoffset priors/lambdas and0-1-2anchorroles',retrospective=True,goal_complete=False)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('code_hashes','parent_prediction_protocol_sha256'):assert old[k]==p[k]
    else:s.energy.write(path,p)
    source_protocol=dict(created_utc=s.energy.now(),control_protocol_sha256=s.energy.sha(path),
        source_code_sha256=s.energy.sha(Path(source.__file__)),algorithm='same phase20_source episode/observationcoordinate logic,outputoffset only',
        prior_families=source.PRIORS,lambdas=source.LAM)
    q=OUT/'source/source_protocol.json'
    if q.exists():assert json.loads(q.read_text())['source_code_sha256']==source_protocol['source_code_sha256']
    else:s.energy.write(q,source_protocol)


def fit(families):
    families=tuple(sorted(families));key=s.digest(dict(protocol=s.energy.sha(OUT/'protocol.json'),families=families))
    path=OUT/'fits'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    table=s.RT[s.RT.family.isin(families)];weights=s.weights(table);sw=np.sqrt(weights);y=table.target.to_numpy(float)
    ref,origin=s.reference(families);initial=np.array(ref['parameters']);cache={};calls=0;failures=[];clock=time.perf_counter()
    def calc(w):
        nonlocal calls
        key=tuple(w)
        if key not in cache:
            calls+=1
            try:values=s.forward('inventory',np.r_[w,0.],table)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(dict(parameters=w,error=repr(exc)));raise
            cache.clear();cache[key]=(values[0],values[1][:,:2])
        return cache[key]
    p,J=calc(initial);oldloss=float(weights@((p-y)**2))
    trials=[dict(stage='inherited_recomputed',parameters=initial,loss=oldloss,nfev=0,success=True)]
    for T in (850.,1100.,1400.,1750.,1900.):
        for B in (15.5,16.5):
            w=[logit((np.log10(T)-np.log10(600.))/(np.log10(1950.)-np.log10(600.))),B]
            try:
                op=least_squares(lambda w:(calc(w)[0]-y)*sw,w,jac=lambda w:calc(w)[1]*sw[:,None],bounds=([-6.,14.],[6.,18.]),
                    x_scale='jac',max_nfev=216,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(stage='extra_start',initial=w,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                    nfev=int(op.nfev),optimality=float(op.optimality),message=op.message))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:trials.append(dict(stage='extra_start',initial=w,loss=None,success=False,error=repr(exc)))
    best=min([r for r in trials if r['loss'] is not None],key=lambda r:r['loss']);pars=np.r_[best['parameters'],0.]
    valid=bool(s.forward('inventory',pars,table)[4].all())
    result=dict(key=key,method='inventory',reg=None,parameters=pars,train_families=families,training_valid=valid,
        exact_reference_fallback=True,reference_search_control=True,trials=trials,selected_success=best['success'],
        state_failures=failures,new_optimizer_starts=10,state_table_calls=calls+1,
        old_parameters=initial,old_training_loss=oldloss,best_training_loss=best['loss'],
        source_reference_path=str(origin.relative_to(ROOT)),source_reference_sha256=s.energy.sha(origin),seconds=time.perf_counter()-clock)
    s.energy.write(path,result);print('budgetreference',len(families),'sec',round(result['seconds'],2),'calls',result['state_table_calls'],flush=True)
    return s.energy.clean(result)


@contextmanager
def source_context():
    before=source.OUT;source.OUT=OUT/'source'
    try:yield
    finally:source.OUT=before


def episode(f,held,budget):
    with source_context():return source.episode(f,held,'offset',budget)


def select(held):
    train=[f for f in s.FAMILIES if f!=held];outer=fit(train);rows=[];choices=[]
    for budget in (0,1,2):
        candidates=[]
        if budget:
            inner=[episode(fit([f for f in train if f!=i]),i,budget) for i in train]
            for prior in source.PRIORS:
                for lam in source.LAM:
                    scores=[source.value(source.pick(e,prior,lam)['MAE']) for e in inner]
                    candidates.append(dict(prior=prior,lam=lam,score=float(np.mean(scores)),inner_keys=[e['key'] for e in inner]))
            best=min(candidates,key=lambda r:(round(source.value(r['score']),10),-source.value(r['lam']),0 if r['prior']=='quadratic' else 1))
        else:best=dict(prior='quadratic',lam=None)
        ep=episode(outer,held,budget);p=source.pick(ep,best['prior'],best['lam'])
        rows += [dict(**r,variant='OriginalBudget',budget=budget,global_reg=None,prior=best['prior'],lam=best['lam'],
            effect=p['effect'],raw_shift=p['raw_shift'],fit_key=outer['key'],episode_key=ep['key']) for r in p['records']]
        choices.append(dict(variant='OriginalBudget',held=held,budget=budget,global_reg=None,source_candidates=candidates,
            selected_source=best,fit_key=outer['key'],outer_episode_key=ep['key'],outer_MAE=p['MAE'],effect=p['effect'],raw_shift=p['raw_shift']))
    folder=OUT/'validation/folds'/s.digest(held);s.energy.csvwrite(folder/'predictions.csv',rows);s.energy.write(folder/'selections.json',choices)
    return held


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();freeze()
    jobs=[list(f) for n in (3,4) for f in combinations(s.FAMILIES,n)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(fit,jobs):pass
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(select,s.FAMILIES):pass
    folder=OUT/'validation'
    frames=[s.read(folder/'folds'/s.digest(h)/'predictions.csv') for h in s.FAMILIES]
    pd.concat(frames,ignore_index=True).to_csv(folder/'predictions.csv',index=False)
    choices=[r for h in s.FAMILIES for r in json.loads((folder/'folds'/s.digest(h)/'selections.json').read_text())]
    s.energy.write(folder/'selections.json',choices)
    s.energy.write(OUT/'execution_complete.json',dict(created_utc=s.energy.now(),newfits=15,newstarts=150,goal_complete=False))


if __name__=='__main__':main()
