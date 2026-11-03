"""Nested source adaptation in the same first-anchor observable coordinate."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.observation_adaptation import ObservationCoordinate,solve_effect
import phase20_prediction as s

OUT=s.OUT/'source_adaptation';LAM=(0.,.001,.01,.1,1.,10.,None);PRIORS=('quadratic','log_tail')
VARIANTS={
 'RelativeInventory':('inventory','field'),
 'RelativeInventoryReadout':('inventory','offset'),
 'SharedReadout':('output','offset'),
 'OriginalHall':('inventory','offset'),
 'DirectRidge':('ridge','offset'),
 'MedianActivation':('median','offset'),
 'DoseLine':('line','offset'),
 'UnitDose':('unit','offset'),
 'UnitDoseH':('unitH','offset'),
}


def value(x):return np.inf if x is None else float(x)


def freeze():
    assert json.loads((s.OUT/'training_complete.json').read_text())['jobs']==330
    paths=(Path(__file__),ROOT/'src/sandbox/observation_adaptation.py',ROOT/'src/sandbox/relative_inventory_model.py')
    payload=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in paths},
        parent_protocol_sha256=s.energy.sha(s.OUT/'prediction_protocol.json'),variants=VARIANTS,
        priors=PRIORS,lambdas=LAM,source_coordinate='first lowest-dose anchorresponse dex;same prior forphysical andoutput',
        selection='globalgamma viazeroanchorLOFOintrainingfamilies;thenfixedgamma for0/1/2;sourceprior/lambda selected only innerfamilies',
        first_anchor='lowestincorporateddose,then sampleID amongfixedroleanchors;guardexcluded',
        limits='rawfield/offset[-6,6];observableeffect[-6,6] intersectfirstanchorfeasibleinterval;4096anchorvectors33nodes80iterations',
        score='full query coverage required;all invalidnumericcontinuations retained butnot scoredasvalidpredictions',
        source_units='oneanchor samequadratic/logtailproblem byconstruction;querydose/state/mobility differences testphysics',
        retrospective=True,goal_complete=False)
    p=OUT/'source_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for k in ('code_hashes','parent_protocol_sha256','variants'):assert old[k]==s.energy.clean(payload[k]),k
    else:s.energy.write(p,payload)


def table(held,budget):
    d=s.RT[s.RT.family.eq(held)].copy()
    roles=s.ROLES[s.ROLES.family.eq(held)&s.ROLES.budget.eq(budget)].set_index('sample_id').role
    d['role']=d.sample_id.map(roles);return d


def predicted(f,d,kind,z):
    rows=[]
    if not f['training_valid'] or (f['method'] in s.PHYSICS and f.get('parameters') is None):
        return [dict(sample_id=r.sample_id,role=r.role,numeric=False,valid=False,error='no valid trainedmodel') for r in d.itertuples()],np.inf
    for row in d.itertuples():
        entry=dict(sample_id=row.sample_id,family=row.family,element=row.element,role=row.role,observed=float(row.target))
        try:
            p,_,v=s.predict(f,pd.DataFrame([row._asdict()]),source_kind=kind,shift=z,require_observation=False)
            entry.update(predicted=float(p[0]),numeric=True,valid=bool(v[0]),
                error=None if v[0] else 'numericcontinuation outside observation domain')
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:entry.update(predicted=None,numeric=False,valid=False,error=repr(exc))
        rows.append(entry)
    q=[r for r in rows if r['role']=='query'];complete=all(r['numeric'] and r['valid'] for r in q)
    mae=float(np.mean([abs(r['predicted']-r['observed']) for r in q])) if complete else np.inf
    return rows,mae


def _episode(f,held,kind,budget):
    assert held not in f['train_families']
    key=s.digest(dict(protocol=s.energy.sha(OUT/'source_protocol.json'),fit_key=f['key'],held=held,kind=kind,budget=budget))
    path=OUT/'episodes'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    clock=time.perf_counter();d=table(held,budget);aa=d[d.role.eq('anchor')].sort_values(['incorporated_cm3','sample_id'])
    y=aa.target.to_numpy(float);coordinate=None;failures=[];results=[]
    if budget:
        try:
            coordinate=ObservationCoordinate(lambda z:s.predict(f,aa,source_kind=kind,shift=z,require_observation=False),linear_shift=kind=='offset')
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(dict(stage='coordinate',error=repr(exc)))
    for prior in (PRIORS if budget else ('quadratic',)):
        solutions=[]
        if budget and coordinate is not None:
            try:solutions=solve_effect(coordinate,y,[value(l) for l in LAM if l is not None],kind=prior)['solutions']
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(dict(stage=prior,error=repr(exc)))
        bylambda={x['lam']:x for x in solutions}
        for lam in (LAM if budget else (None,)):
            if lam is None:op=dict(valid=True,effect=0.,raw_shift=0.,status='no_update')
            else:op=bylambda.get(lam,dict(valid=False,error='sourceinferencefailed'))
            rows=[];score=np.inf
            if op.get('valid'):
                rows,score=predicted(f,d,kind,float(op['raw_shift']))
            else:
                rows=[dict(sample_id=r.sample_id,family=held,element=r.element,role=r.role,observed=float(r.target),
                    predicted=None,numeric=False,valid=False,error=op.get('error','no feasible anchorresponse')) for r in d.itertuples()]
            results.append(dict(prior=prior,lam=lam,effect=op.get('effect'),raw_shift=op.get('raw_shift'),
                MAE=score,anchor_solution=op,records=rows))
    output=dict(key=key,fit_key=f['key'],held=held,source_kind=kind,budget=budget,train_families=f['train_families'],
        first_anchor=None if not len(aa) else aa.sample_id.iloc[0],anchor_samples=aa.sample_id.tolist(),
        anchor_labels=y,query_labels_used_in_sourcefit=False,results=results,failures=failures,
        anchor_vector_calls=0 if coordinate is None else coordinate.calls,
        failed_raw_proposals=[] if coordinate is None else coordinate.failures,seconds=time.perf_counter()-clock)
    s.energy.write(path,output)
    print('responseepisode',f['method'],f['reg'],held,budget,kind,'sec',round(output['seconds'],2),'calls',output['anchor_vector_calls'],flush=True)
    return s.energy.clean(output)


def episode(f,held,kind,budget):
    key=s.digest(dict(protocol=s.energy.sha(OUT/'source_protocol.json'),fit_key=f['key'],held=held,kind=kind,budget=budget))
    lock=OUT/'locks'/f'{key}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        return _episode(f,held,kind,budget)


def pick(ep,prior,lam):return next(x for x in ep['results'] if x['prior']==prior and x['lam']==lam)


def select(job):
    variant,outer_held=job
    method,kind=VARIANTS[variant];rows=[];choices=[]
    for held in (outer_held,):
        train=[f for f in s.FAMILIES if f!=held]
        if variant=='OriginalHall':reg=None;global_candidates=[]
        else:
            bank=json.loads((OUT/'global_choices.json').read_text())
            entry=next(r for r in bank if r['held']==held and r['method']==method)
            reg,global_candidates=entry['reg'],entry['candidates']
        f=s.fit(train,method,reg)
        for budget in (0,1,2):
            if budget:
                inner=[episode(s.fit([f for f in train if f!=i],method,reg),i,kind,budget) for i in train]
                candidates=[]
                for prior in PRIORS:
                    for lam in LAM:
                        scores=[value(pick(ep,prior,lam)['MAE']) for ep in inner]
                        candidates.append(dict(prior=prior,lam=lam,score=float(np.mean(scores)),inner_scores=scores,inner_keys=[ep['key'] for ep in inner]))
                best=min(candidates,key=lambda r:(round(value(r['score']),10),-value(r['lam']),0 if r['prior']=='quadratic' else 1))
            else:candidates=[];best=dict(prior='quadratic',lam=None,score=None)
            outer=episode(f,held,kind,budget);selected=pick(outer,best['prior'],best['lam'])
            rows += [dict(**r,variant=variant,budget=budget,global_reg=reg,prior=best['prior'],lam=best['lam'],
                effect=selected['effect'],raw_shift=selected['raw_shift'],fit_key=f['key'],episode_key=outer['key']) for r in selected['records']]
            choices.append(dict(variant=variant,held=held,budget=budget,global_reg=reg,global_candidates=global_candidates,
                source_candidates=candidates,selected_source=best,fit_key=f['key'],outer_episode_key=outer['key'],
                outer_MAE=selected['MAE'],effect=selected['effect'],raw_shift=selected['raw_shift']))
    folder=OUT/'variants'/variant/'folds'/s.digest(outer_held)
    s.energy.csvwrite(folder/'predictions.csv',rows);s.energy.write(folder/'selections.json',choices)
    print('foldcomplete',variant,outer_held,flush=True);return variant,outer_held


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();freeze()
    choices=[]
    for method in ('inventory','output','ridge','median','line','unit','unitH'):
        for held in s.FAMILIES:
            train=[f for f in s.FAMILIES if f!=held];reg,candidates=s.select_global(train,method)
            choices.append(dict(method=method,held=held,reg=reg,candidates=candidates))
    s.energy.write(OUT/'global_choices.json',choices)
    s.energy.write(OUT/'execution_started.json',dict(created_utc=s.energy.now(),variants=list(VARIANTS),workers=a.workers))
    jobs=[(variant,held) for variant in VARIANTS for held in s.FAMILIES]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(select,jobs):pass
    for variant in VARIANTS:
        folder=OUT/'variants'/variant
        frames=[s.read(folder/'folds'/s.digest(held)/'predictions.csv') for held in s.FAMILIES]
        pd.concat(frames,ignore_index=True).to_csv(folder/'predictions.csv',index=False)
        chosen=[r for held in s.FAMILIES for r in json.loads((folder/'folds'/s.digest(held)/'selections.json').read_text())]
        s.energy.write(folder/'selections.json',chosen)
    s.energy.write(OUT/'execution_complete.json',dict(created_utc=s.energy.now(),variants=len(VARIANTS),goal_complete=False))


if __name__=='__main__':main()
