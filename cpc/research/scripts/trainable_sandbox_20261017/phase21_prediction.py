"""Frozen-weight source inference with explicit finite-input response controls."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse,copy,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.functional_source import ResponseDesign,solve_response
from sandbox.relative_predictor_v2 import SavedModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.kroger_db import load
import phase20_prediction as parent
import phase2_energy as energy
from csv_io_v2 import read_result_csv as read

BASE=energy.BASE;OUT=BASE/'phase21_curve_response';PARENT=parent.OUT
LAM=(None,1e10,1e8,1e6,1e4,1e2,10.,1.,.1,.01,.001,0.)
PRIORS=('quadratic','log_tail');FAMILIES=parent.FAMILIES;RT=parent.RT;ROLES=parent.ROLES
FIELD_MODES={'FieldFirst':('first',False),'FieldFirstSafe':('first',True),
    'FieldAnchorSafe':('anchors',True),'FieldCurve':('curve',False),'FieldCurveSafe':('curve',True)}
CONTROLS={'RelativeInventoryReadout':'inventory','SharedReadout':'output','OriginalHall':'inventory',
    'OriginalBudget':'inventory','DirectRidge':'ridge','MedianActivation':'median','DoseLine':'line','UnitDose':'unit','UnitDoseH':'unitH'}
VARIANTS=list(FIELD_MODES)+list(CONTROLS);BANK={};MODELS={};DB=None
INPUT_FIELDS=('element','incorporated_cm3','Tobs_K','H_known','H_cm3','carbon_known','carbon_cm3','growth_T_C')


def inputs(row):return {k:bool(getattr(row,k)) if k in ('H_known','carbon_known') else str(getattr(row,k)) if k=='element' else float(getattr(row,k)) for k in INPUT_FIELDS}
def identity(row):return tuple(getattr(row,k) for k in INPUT_FIELDS)
def number(x):return np.inf if x is None else float(x)
def digest(x):return parent.digest(x)


def frozen_fits():
    global BANK
    if not BANK:
        for group,folder in (('primary',PARENT/'fits'),('reference',PARENT/'reference_budget_control/fits')):
            for p in folder.glob('*.json'):
                f=json.loads(p.read_text());key=(group,f['method'],f.get('reg'),tuple(sorted(f['train_families'])))
                assert key not in BANK;BANK[key]=(f,p)
    return BANK


def getfit(variant,families,reg):
    method='inventory' if variant in FIELD_MODES else CONTROLS[variant];group='reference' if variant=='OriginalBudget' else 'primary'
    key=(group,method,reg,tuple(sorted(families)))
    if key not in frozen_fits():raise ValueError('Required frozen model absent;new sharedtraining is not permitted')
    return frozen_fits()[key]


def model(f):
    global DB
    if DB is None:DB=load()
    if f['key'] not in MODELS:MODELS[f['key']]=SavedModel(f,profile=published_coherent_profile(),db=DB)
    return MODELS[f['key']]


def table(held,budget):
    d=RT[RT.family.eq(held)].copy();roles=ROLES[ROLES.family.eq(held)&ROLES.budget.eq(budget)].set_index('sample_id').role
    d['role']=d.sample_id.map(roles);return d


def design_inputs(d):
    # Only actual, explicitly known non-guard inputs are used. No interpolation
    # of missing H/carbon histories or a continuous-depth guarantee is invented.
    source=d[~d.role.eq('guard')].sort_values(['incorporated_cm3','sample_id']);rows=[];seen={};anchors=[];labels=[];anchor_ids=[]
    for r in source.itertuples():
        row=SimpleNamespace(**inputs(r));key=identity(row)
        if key not in seen:seen[key]=len(rows);rows.append(row)
        if r.role=='anchor':anchors.append(seen[key]);labels.append(float(r.target));anchor_ids.append(r.sample_id)
    return rows,anchors,np.array(labels),anchor_ids


def predict_records(f,d,kind,u):
    rows=[];m=model(f)
    for r in d.itertuples():
        item=dict(sample_id=r.sample_id,family=r.family,element=r.element,role=r.role,observed=float(r.target))
        try:
            p=m.point(SimpleNamespace(**inputs(r)),source_kind=kind,shift=float(u),require_observation=False)
            item.update(predicted=float(p.values[0]),numeric=True,valid=bool(p.diagnostics['observation_domain_valid']),
                predicted_log_mu=float(p.transport_values[0]),predicted_log_sigma=float(p.transport_values[1]),
                error=None if p.diagnostics['observation_domain_valid'] else 'numericcontinuation outside observation domain')
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:item.update(predicted=None,numeric=False,valid=False,error=repr(exc))
        rows.append(item)
    q=[r for r in rows if r['role']=='query'];score=float(np.mean([abs(r['predicted']-r['observed']) for r in q])) if all(r['numeric'] and r['valid'] for r in q) else np.inf
    return rows,score


def freeze():
    gate=json.loads((OUT/'numerical_gates.json').read_text());assert gate['passed']
    assert gate['code_hashes']['src/sandbox/functional_source.py']==energy.sha(ROOT/'src/sandbox/functional_source.py')
    paths=(Path(__file__),ROOT/'src/sandbox/functional_source.py',ROOT/'src/sandbox/relative_predictor_v2.py',ROOT/'src/sandbox/relative_inventory_model.py')
    inputs=(BASE/'artifact_manifest_phase20.json',PARENT/'source_adaptation_v2/global_choices.json',PARENT/'delivery/global_selection.json',
        PARENT/'delivery_v2/predictor_bundle_v7/manifest.json',BASE/'episodes_v1.csv',OUT/'numerical_gates.json')
    p=dict(created_utc=energy.now(),code_hashes={str(x.relative_to(ROOT)):energy.sha(x) for x in paths},
        input_hashes={str(x.relative_to(ROOT)):energy.sha(x) for x in inputs},
        variants=VARIANTS,field_modes=FIELD_MODES,controls=CONTROLS,lambdas=LAM,priors=PRIORS,
        shared_weights='all phase20globalweights/gamma choicesfrozen;no retraining,includinggenerousreference',
        design='deduplicatedactualnon-guard knownRTinputs perheldsource;noquerylabels,syntheticgridorcontinuousintervalclaim',
        conditioning='knownquery-input-setconditional sourceadaptation;not inductiveunseeninput validation',
        objective='sum anchor logresidual^2 +lambda*Q orlambda*log1p(Q);Q=mean squared change onfirst/allanchor/allknowninput nodes',
        validity='FieldFirst/FieldCurve:anchorvalid;otherfield/offsetcontrols:alldeclarednodesvalid;allreportedqueryscores requirefullcoverage',
        inference='scalaractualIFTgradient roots on33zeroclusterednodes,80iterations;bounded rescue,notglobaloptimumproof',
        limits='raw[-6,6],Q<=36;32768nominalpointbudget/variant/episode;sharedexactforwardmemoizationonly saves actualCPU',
        selection='sameoutertrain-onlyglobalgamma;sourceprior/lambda byinnerfamilyqueryMAE,thenfreeze forouterheld;round10digits,tielargerlambda thenquadratic',
        labels='old29/5RT nHonly;0/1/2roles,guard,common18 unchanged;Leedydevelopmentonlynotblind',
        secondary='sameRTsourceanchorsforold142nT/59mu ownT;sigmaderived;no newT/mu supervision',
        new_shared_optimizer_starts=0,new_DFT_MD=0,new_material_measurements=0,goal_complete=False)
    q=OUT/'prediction_protocol.json'
    if q.exists():
        old=json.loads(q.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==p[k],k
    else:energy.write(q,p)


def episode_key(f,held,kind,budget):return digest(dict(protocol=energy.sha(OUT/'prediction_protocol.json'),fit=f['key'],held=held,kind=kind,budget=budget))


def group_episode(job):
    f,path,held,kind,budget=job;assert held not in f['train_families'];key=episode_key(f,held,kind,budget);dest=OUT/'episodes'/f'{key}.json'
    if dest.exists():return json.loads(dest.read_text())
    lock=OUT/'locks'/f'{key}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        if dest.exists():return json.loads(dest.read_text())
        return execute_episode(f,path,held,kind,budget,key,dest)


def execute_episode(f,path,held,kind,budget,key,dest):
    clock=time.perf_counter();d=table(held,budget);nodes,ai,y,anchor_ids=design_inputs(d);m=model(f);raw_cache={};physical_calls=0;point_failures=[]
    def forward(u):
        nonlocal physical_calls
        u=float(u)
        if u not in raw_cache:
            p=[]
            for i,row in enumerate(nodes):
                physical_calls+=1
                try:p.append(m.point(row,source_kind=kind,shift=u,require_observation=False))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:point_failures.append(dict(raw=u,node=i,error=repr(exc)));p.append(None)
            raw_cache[u]=(np.array([x.values[0] if x is not None else np.nan for x in p]),
                np.array([x.source_jacobian[0] if x is not None else np.nan for x in p]),
                np.array([x is not None and x.diagnostics['observation_domain_valid'] for x in p]))
        return raw_cache[u]
    modes=FIELD_MODES if kind=='field' else {'OutputSafe':('curve',True)};results={};meta={};nodeids=list(range(len(nodes)));record_cache={}
    for name,(pen,safe) in modes.items():
        rows=[];failures=[];design=None
        if budget:
            pi=[ai[0]] if pen=='first' else ai if pen=='anchors' else nodeids
            try:design=ResponseDesign(forward,ai,pi,nodeids if safe else ai,linear=kind=='offset',point_budget=32768)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(dict(stage='design',error=repr(exc)))
        for prior in (PRIORS if budget else ('quadratic',)):
            sols=[]
            if design is not None:
                try:sols=solve_response(design,y,[number(l) for l in LAM],kind=prior)['solutions']
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(dict(stage=prior,error=repr(exc)))
            for lam in (LAM if budget else (None,)):
                sol=next((r for r in sols if r['lam']==lam),dict(valid=False,error='sourceinferencefailed'))
                if lam is None:sol=dict(valid=True,raw_shift=0.,penalty=0.,penalty_effect=0.,status='no_update')
                if sol.get('valid'):
                    u=float(sol['raw_shift'])
                    if u not in record_cache:record_cache[u]=predict_records(f,d,kind,u)
                    rec,score=record_cache[u]
                else:
                    rec=[dict(sample_id=r.sample_id,family=held,element=r.element,role=r.role,observed=float(r.target),predicted=None,numeric=False,valid=False,error=sol.get('error','sourceinferenceincomplete')) for r in d.itertuples()];score=np.inf
                rows.append(dict(prior=prior,lam=lam,solution=sol,raw_shift=sol.get('raw_shift'),MAE=score,records=rec))
        results[name]=rows;meta[name]=dict(failures=failures,nominal_vector_calls=0 if design is None else design.calls,
            nominal_point_calls=0 if design is None else design.points,raw_bounds=None if design is None else design.bounds)
    value=dict(key=key,fit_key=f['key'],fit_path=str(Path(path).relative_to(ROOT)),fit_sha256=energy.sha(path),train_families=f['train_families'],
        held=held,kind=kind,budget=budget,anchor_ids=anchor_ids,anchor_labels=y,known_input_nodes=[vars(r) for r in nodes],query_labels_in_inference=False,
        results=results,mode_metadata=meta,actual_forward_points=physical_calls,actual_source_vectors=len(raw_cache),point_failures=point_failures,seconds=time.perf_counter()-clock)
    energy.write(dest,value);print('sourcegroup',held,budget,kind,'sec',round(value['seconds'],1),'actualpoints',physical_calls,flush=True)
    return energy.clean(value)


def request_plans():
    bank=json.loads((PARENT/'source_adaptation_v2/global_choices.json').read_text());full=json.loads((PARENT/'delivery/global_selection.json').read_text())
    plans=[];jobs={}
    for variant in VARIANTS:
        method='inventory' if variant in FIELD_MODES else CONTROLS[variant];kind='field' if variant in FIELD_MODES else 'offset'
        for outer in (*FAMILIES,None):
            train=list(FAMILIES) if outer is None else [f for f in FAMILIES if f!=outer]
            if variant in ('OriginalHall','OriginalBudget'):reg=None
            elif outer is None:reg=full[method]['reg']
            else:reg=next(x['reg'] for x in bank if x['method']==method and x['held']==outer)
            f,path=getfit(variant,train,reg);plan=dict(variant=variant,held=outer,train_families=train,reg=reg,fit_key=f['key'],fit_path=str(path.relative_to(ROOT)),kind=kind,
                mode=variant if kind=='field' else 'OutputSafe',inner={},outer={})
            for budget in (0,1,2):
                if outer is not None:
                    key=episode_key(f,outer,kind,budget);jobs[key]=(f,path,outer,kind,budget);plan['outer'][str(budget)]=key
                if budget:
                    plan['inner'][str(budget)]=[]
                    for held in train:
                        fi,pi=getfit(variant,[x for x in train if x!=held],reg);key=episode_key(fi,held,kind,budget)
                        jobs[key]=(fi,pi,held,kind,budget);plan['inner'][str(budget)].append(key)
            plans.append(plan)
    return plans,list(jobs.values())


def candidate(ep,mode,prior,lam):return next(r for r in ep['results'][mode] if r['prior']==prior and r['lam']==lam)


def select_plans(plans):
    allpred=[];allchoices=[]
    for plan in plans:
        variant,held,mode=plan['variant'],plan['held'],plan['mode'];choices=[];rows=[]
        for budget in (0,1,2):
            candidates=[]
            if budget:
                inner=[json.loads((OUT/'episodes'/f'{key}.json').read_text()) for key in plan['inner'][str(budget)]]
                for prior in PRIORS:
                    for lam in LAM:
                        scores=[number(candidate(e,mode,prior,lam)['MAE']) for e in inner]
                        candidates.append(dict(prior=prior,lam=lam,score=float(np.mean(scores)),inner_scores=scores,inner_keys=plan['inner'][str(budget)]))
                best=min(candidates,key=lambda r:(round(number(r['score']),10),-number(r['lam']),0 if r['prior']=='quadratic' else 1))
            else:best=dict(prior='quadratic',lam=None,score=None)
            c=dict(variant=variant,held=held,budget=budget,reg=plan['reg'],fit_key=plan['fit_key'],kind=plan['kind'],mode=mode,
                train_families=plan['train_families'],selected_source=best,source_candidates=candidates)
            if held is not None:
                key=plan['outer'][str(budget)];ep=json.loads((OUT/'episodes'/f'{key}.json').read_text());z=candidate(ep,mode,best['prior'],best['lam'])
                c.update(outer_episode_key=key,outer_MAE=z['MAE'],raw_shift=z['raw_shift'])
                rows += [dict(**r,variant=variant,budget=budget,prior=best['prior'],lam=best['lam'],fit_key=plan['fit_key'],episode_key=key,raw_shift=z['raw_shift']) for r in z['records']]
            choices.append(c)
        if held is not None:
            allpred+=rows;allchoices+=choices
            energy.csvwrite(OUT/'variants'/variant/'folds'/digest(held)/'predictions.csv',rows)
            energy.write(OUT/'variants'/variant/'folds'/digest(held)/'selections.json',choices)
        else:energy.write(OUT/'deployment_selection'/f'{variant}.json',choices)
    for variant in VARIANTS:
        energy.csvwrite(OUT/'variants'/variant/'predictions.csv',[r for r in allpred if r['variant']==variant])
        energy.write(OUT/'variants'/variant/'selections.json',[r for r in allchoices if r['variant']==variant])
    energy.write(OUT/'all_selections.json',allchoices)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();freeze();plans,jobs=request_plans()
    energy.write(OUT/'request_plans.json',plans);energy.write(OUT/'execution_started.json',dict(created_utc=energy.now(),groups=len(jobs),workers=a.workers,variants=len(VARIANTS),new_shared_training=False))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(group_episode,jobs):pass
    select_plans(plans)
    energy.write(OUT/'execution_complete.json',dict(created_utc=energy.now(),groups=len(jobs),variants=len(VARIANTS),new_shared_optimizer_starts=0,goal_complete=False))
    print('completed',len(jobs),'groups',len(VARIANTS),'variants',flush=True)


if __name__=='__main__':main()
