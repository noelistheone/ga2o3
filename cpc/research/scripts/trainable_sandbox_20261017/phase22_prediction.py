"""Frozen-weight finite-state source inference with matched empirical controls."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.source_ensemble import SourceEnsemble,source_grid
from csv_io_v2 import read_result_csv as read
import phase21_prediction as s
OUT=s.BASE/'phase22_predictive_inference';SIZE=129
TAUS=(.05,.15,.5,1.5);SIGMAS=(.05,.15,.5);RHOS=(0.,.5,.9,1.)
VARIANTS={'InventoryEnsemble':('FieldCurveSafe',False),'InventorySpike':('FieldCurveSafe',True),
    'InventoryReadoutEnsemble':('RelativeInventoryReadout',False),'InventoryReadoutSpike':('RelativeInventoryReadout',True),
    'OriginalReadoutSpike':('OriginalHall',True),'RidgeSpike':('DirectRidge',True),
    'MedianSpike':('MedianActivation',True),'DoseLineSpike':('DoseLine',True),
    'UnitSpike':('UnitDose',True),'UnitHSpike':('UnitDoseH',True)}
DEFAULT=dict(tau=.15,sigma=.15,rho=1.)


def configs(budget):
    if not budget:return [dict(DEFAULT)]
    return [dict(tau=t,sigma=s,rho=r) for t in TAUS for s in SIGMAS for r in RHOS if r!=1. or t==TAUS[0]]


def key(config):return s.digest(config)


def plans_and_jobs():
    old=json.loads((s.OUT/'request_plans.json').read_text());plans=[];jobs={}
    for variant,(parent,spike) in VARIANTS.items():
        for p in [x for x in old if x['variant']==parent]:
            q=dict(p);q.update(variant=variant,parent_variant=parent,allow_spike=spike,inner={},outer={})
            for budget in (0,1,2):
                if p['held'] is not None:
                    ek=p['outer'][str(budget)];ep=json.loads((s.OUT/'episodes'/f'{ek}.json').read_text())
                    jk=s.digest([ep['fit_key'],ep['held'],ep['kind'],budget,SIZE]);jobs[jk]=(jk,ep['fit_path'],ep['held'],ep['kind'],budget)
                    q['outer'][str(budget)]=jk
                if budget:
                    q['inner'][str(budget)]=[]
                    for ek in p['inner'][str(budget)]:
                        ep=json.loads((s.OUT/'episodes'/f'{ek}.json').read_text());jk=s.digest([ep['fit_key'],ep['held'],ep['kind'],budget,SIZE])
                        jobs[jk]=(jk,ep['fit_path'],ep['held'],ep['kind'],budget);q['inner'][str(budget)].append(jk)
            plans.append(q)
    return plans,list(jobs.values())


def freeze():
    paths=(Path(__file__),ROOT/'src/sandbox/source_ensemble.py',ROOT/'docs/ML_SANDBOX_FINITE_ENSEMBLE_20261018.md')
    inputs=(s.BASE/'artifact_manifest_phase21.json',s.OUT/'request_plans.json',s.BASE/'dataset_v1.csv',s.BASE/'episodes_v1.csv')
    p=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in inputs},
        purpose='actual dopant-property source prediction; no new shared optimization; normalized finite probability model',
        variants=VARIANTS,taus=TAUS,sigmas=SIGMAS,rhos=RHOS,grid_size=SIZE,numerical_grid_sensitivity=[33,65,129],
        base_measure='raw-u Voronoi cell masses on cubic-spaced declared grid; finite model,not continuum posterior',
        state_domain='all known nonguard actual RT input nodes valid,mean response Q<=36,u in[-6,6]',
        selection='training-families-only fullquery MAE; NLL tie; no external query labels',
        point_action='latent lognH weighted median; return mu/sigma from selected state,not claim noisy-observation Bayes median',
        maximum_point_budget=32768,shared_weights='phase20 byte-frozen; previous global-gamma selection unchanged',
        fairness='same inputs,anchor labels,coverage,budget ceiling; physical873energy inputs,empirical0;old affine/MAP controls retained',
        zero_budget='exact reference state; no prior-mean shift',goal_complete=False)
    path=OUT/'prediction_protocol.json'
    if path.exists():
        previous=json.loads(path.read_text());assert previous['code_hashes']==p['code_hashes'] and previous['input_hashes']==p['input_hashes']
    else:s.energy.write(path,p)


def run_group(job):
    jk,path,held,kind,budget=job;dest=OUT/'episodes'/f'{jk}.json';lock=OUT/'locks'/f'{jk}.lock';lock.parent.mkdir(exist_ok=True,parents=True)
    with lock.open('a') as h:
        fcntl.flock(h.fileno(),fcntl.LOCK_EX)
        if dest.exists():return
        clock=time.perf_counter();f=json.loads((ROOT/path).read_text());model=s.model(f);d=s.table(held,budget)
        nodes,ai,y,anchorids=s.design_inputs(d);index={s.identity(r):j for j,r in enumerate(nodes)}
        u,base=source_grid(SIZE) if budget else (np.array([0.]),np.array([1.]));zero=int(np.flatnonzero(u==0.)[0]);values=np.full((len(u),len(nodes),3),np.nan);valid=np.zeros((len(u),len(nodes)),bool)
        failures=[];actual=0
        reference=[]
        for j,r in enumerate(nodes):
            actual+=1;v=model.point(r,source_kind=kind,shift=0.,require_observation=False);reference.append(v)
            values[zero,j]=[v.values[0],*v.transport_values];valid[zero,j]=v.diagnostics['observation_domain_valid']
        for i,shift in enumerate(u):
            if i==zero:continue
            if kind=='offset':
                values[i]=values[zero]+np.array([shift,0.,shift])
                valid[i]=valid[zero]&(values[i,:,0]>=13.)
                if not model.physical:
                    for j,r in enumerate(nodes):
                        actual+=1;v=model.point(r,source_kind=kind,shift=shift,require_observation=False);values[i,j]=[v.values[0],*v.transport_values];valid[i,j]=v.diagnostics['observation_domain_valid']
            else:
                for j,r in enumerate(nodes):
                    actual+=1
                    try:
                        v=model.point(r,source_kind=kind,shift=float(shift),require_observation=False);values[i,j]=[v.values[0],*v.transport_values];valid[i,j]=v.diagnostics['observation_domain_valid']
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(dict(coordinate=float(shift),node=j,error=repr(exc)))
        nominal=len(u)*len(nodes);assert nominal<=32768
        grids={}
        for n in ((33,65,129) if budget else (1,)):
            select=np.arange(SIZE)[::(SIZE-1)//(n-1)] if budget else np.array([0]);uu,bb=source_grid(n) if budget else (u,base)
            grids[n]=(select,SourceEnsemble(uu,bb,values[select,:,0],np.all(valid[select],axis=1)))
        rows=[]
        for config in configs(budget):
            bygrid={}
            for n,(select,E) in grids.items():
                posterior=E.posterior(ai,y,**config);summary=E.summarize(posterior);med=select[summary['median_indices']]
                predicted=values[med,np.arange(len(nodes))]
                records=[];queryidx=[];queryobs=[]
                for r in d.itertuples():
                    rec=dict(sample_id=r.sample_id,family=held,element=r.element,role=r.role,observed=float(r.target))
                    if r.role=='guard':rec.update(numeric=False,valid=False,error='guard not in declared source design',predicted=None)
                    else:
                        j=index[s.identity(r)];p=predicted[j];rec.update(numeric=True,valid=True,predicted=float(p[0]),predicted_log_mu=float(p[1]),predicted_log_sigma=float(p[2]),
                            raw_shift=float(u[med[j]]),error=None,latent_variance=float(summary['variance'][j]))
                        if r.role=='query':queryidx.append(j);queryobs.append(r.target)
                    records.append(rec)
                score=float(np.mean([abs(r['predicted']-r['observed']) for r in records if r['role']=='query']))
                logpdf=E.predictive_logpdf(posterior,queryidx,queryobs)
                bygrid[str(n)]=dict(MAE=score,NLL=float(-np.mean(logpdf)),records=records,weights=posterior.weights,
                    posterior_spike=summary['posterior_spike'],posterior_zero_state=summary['posterior_zero_state'],
                    log_evidence=summary['log_evidence'],effective_states=summary['effective_states'])
            rows.append(dict(config=config,key=key(config),grids=bygrid))
        value=dict(key=jk,fit_path=path,fit_key=f['key'],fit_sha256=s.energy.sha(ROOT/path),train_families=f['train_families'],
            held=held,kind=kind,budget=budget,known_input_nodes=[vars(r) for r in nodes],anchor_indices=ai,anchor_ids=anchorids,anchor_labels=y,
            query_labels_in_inference=False,coordinates=u,base_weights=base,state_values=values,state_valid=valid,
            candidates=rows,nominal_forward_points=nominal,actual_forward_points=actual,point_failures=failures,seconds=time.perf_counter()-clock)
        s.energy.write(dest,value);print('ensemblegroup',held,budget,kind,'points',actual,'sec',round(value['seconds'],1),flush=True)


def candidate(ep,config,grid=None):
    r=next(z for z in ep['candidates'] if z['key']==key(config));n=grid if grid is not None else SIZE if ep['budget'] else 1
    return r['grids'][str(n)]


def select_plans(plans):
    allrows=[];choices=[];sensitivity=[]
    for plan in plans:
        for budget in (0,1,2):
            cs=[]
            if budget:
                eps=[json.loads((OUT/'episodes'/f'{k}.json').read_text()) for k in plan['inner'][str(budget)]]
                for c in configs(budget):
                    if not plan['allow_spike'] and c['rho']!=0.:continue
                    stats=[candidate(e,c) for e in eps]
                    cs.append(dict(config=c,score=float(np.mean([z['MAE'] for z in stats])),NLL=float(np.mean([z['NLL'] for z in stats])),
                        inner_keys=plan['inner'][str(budget)],inner_scores=[z['MAE'] for z in stats]))
                best=min(cs,key=lambda a:(round(a['score'],10),round(a['NLL'],10),-a['config']['rho'],a['config']['tau'],a['config']['sigma']))
            else:best=dict(config=DEFAULT,score=None,NLL=None)
            c=dict(variant=plan['variant'],held=plan['held'],budget=budget,reg=plan['reg'],fit_key=plan['fit_key'],fit_path=plan['fit_path'],kind=plan['kind'],
                train_families=plan['train_families'],selected_source=best,source_candidates=cs)
            if plan['held'] is not None:
                k=plan['outer'][str(budget)];ep=json.loads((OUT/'episodes'/f'{k}.json').read_text());r=candidate(ep,best['config'])
                c.update(outer_episode_key=k,outer_MAE=r['MAE'],outer_NLL=r['NLL'])
                allrows.extend(dict(z,variant=plan['variant'],budget=budget,fit_key=plan['fit_key'],episode_key=k,**best['config']) for z in r['records'])
                if budget:
                    fine={z['sample_id']:z for z in r['records'] if z['role']=='query'}
                    for n in (33,65):
                        coarse=candidate(ep,best['config'],n);changes=[abs(z['predicted']-fine[z['sample_id']]['predicted']) for z in coarse['records'] if z['role']=='query']
                        sensitivity.append(dict(variant=plan['variant'],held=plan['held'],budget=budget,coarse_size=n,fine_size=129,
                            maximum_query_change=max(changes),mean_query_change=float(np.mean(changes)),
                            posterior_spike_change=abs(coarse['posterior_spike']-r['posterior_spike']),stable_at_02dex=max(changes)<=.02))
            choices.append(c)
    s.energy.csvwrite(OUT/'predictions.csv',allrows);s.energy.write(OUT/'all_selections.json',choices)
    s.energy.csvwrite(OUT/'grid_sensitivity.csv',sensitivity)
    for v in VARIANTS:
        s.energy.write(OUT/'deployment_selection'/f'{v}.json',[c for c in choices if c['variant']==v and c['held'] is None])
    d=pd.DataFrame(allrows);summary=[];families=[]
    for scope in ('all_query','common18'):
        common=set(s.ROLES[s.ROLES.budget.eq(2)&s.ROLES.role.eq('query')].sample_id)
        dd=d[d.role.eq('query')] if scope=='all_query' else d[d.sample_id.isin(common)]
        for keys,g in dd.groupby(['variant','budget','family']):
            good=bool(g.numeric.all());families.append(dict(zip(['variant','budget','family'],keys),scope=scope,queries=len(g),numeric=int(g.numeric.sum()),
                MAE=float(abs(g.predicted-g.observed).mean()) if good else None))
    by=pd.DataFrame(families);by.to_csv(OUT/'by_family.csv',index=False)
    for keys,g in by.groupby(['variant','budget','scope']):
        summary.append(dict(zip(['variant','budget','scope'],keys),queries=int(g.queries.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else None,worst_family_MAE=float(g.MAE.max()) if g.MAE.notna().all() else None))
    s.energy.csvwrite(OUT/'summary.csv',summary)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    assert json.loads((OUT/'ensemble_gates.json').read_text())['passed'];freeze();plans,jobs=plans_and_jobs()
    s.energy.write(OUT/'request_plans.json',plans);s.energy.write(OUT/'execution_started.json',dict(created_utc=s.energy.now(),jobs=len(jobs),variants=len(VARIANTS),workers=a.workers,new_shared_optimizer_starts=0))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(run_group,jobs):pass
    select_plans(plans);s.energy.write(OUT/'execution_complete.json',dict(created_utc=s.energy.now(),jobs=len(jobs),variants=len(VARIANTS),new_shared_optimizer_starts=0,goal_complete=False))
    print('completed finite-state prediction',len(jobs),len(VARIANTS),flush=True)


if __name__=='__main__':main()
