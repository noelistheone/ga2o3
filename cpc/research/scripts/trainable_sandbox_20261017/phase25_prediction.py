"""Nested, held-source/cluster property prediction with isolated source workers."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,argparse,subprocess,copy
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.context_saved_model import SavedContextModel,INPUT_NAMES
from sandbox.kroger_db import NPZ
import phase25_training as t
import phase2_energy as e

OUT=t.OUT/'prediction';LAM=(None,0.,.01,.1,1.,10.)
VARIANTS={m:dict(method=m,kind='offset') for m in t.METHODS}
for m in ('InventoryConstant','InventoryDose','InventoryProcess'):
    VARIANTS[m+'_internal']=dict(method=m,kind='field')
for variant,method in (('DoseLineAffine','DoseLine'),('RidgeAffine','DirectRidge'),('PhysicalAffine','FixedPhysical')):
    VARIANTS[variant]=dict(method=method,kind='affine')
MINIMAL=('key','method','parameters','coefficients','intercept','center','scale','offset_dex','training_valid','selected_success')


def saved(scheme,train,method,reg):
    key,path=t.fit_path(scheme,train,method,reg)
    if not path.exists():raise RuntimeError('Training prerequisite absent; prediction must not launch unplanned fits: '+str(path))
    f=json.loads(path.read_text());assert tuple(f['train_units'])==tuple(sorted(train))
    return f,path


def inputs(row):
    return {k:e.clean(getattr(row,k)) for k in INPUT_NAMES}


def score_rows(records):
    if not records or not all(r['numeric'] and r['valid'] for r in records):return np.inf
    d=pd.DataFrame(records)
    return float(d.groupby('family').apply(lambda g:np.abs(g.predicted-g.observed).mean(),include_groups=False).mean())


def zero_score(f,held,scheme):
    d=t.TABLE[t.TABLE[t.unit_column(scheme)].eq(held)]
    rows=[]
    try:model=SavedContextModel(f)
    except (ValueError,RuntimeError,np.linalg.LinAlgError):return np.inf,[]
    for r in d.itertuples():
        item=dict(sample_id=r.sample_id,family=r.family,observed=float(r.target),numeric=False,valid=False)
        try:
            p=model.point(SimpleNamespace(**inputs(r)),require_observation=False)
            item.update(predicted=p.log_nH,numeric=True,valid=p.valid)
        except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError) as exc:item['error']=repr(exc)
        rows.append(item)
    return score_rows(rows),rows


def select_shared(scheme,train,method):
    candidates=[];pred=[]
    for reg in t.configs(method):
        scores=[]
        for held in train:
            f,path=saved(scheme,[u for u in train if u!=held],method,reg)
            score,rows=zero_score(f,held,scheme);scores.append(score)
            pred.extend(dict(r,scheme=scheme,training_context=t.digest(train),method=method,reg=reg,
                inner_held_unit=held,fit_key=f['key']) for r in rows)
        candidates.append(dict(reg=reg,score=float(np.mean(scores)),inner_units=list(train),inner_scores=scores))
    best=min(candidates,key=lambda r:(round(r['score'],10),-np.inf if r['reg'] is None else -r['reg']))
    return best['reg'],candidates,pred


def design(held):
    d=t.TABLE[t.TABLE.family.eq(held)].sort_values(['incorporated_cm3','sample_id'])
    nodes=[];lookup={};sample_node={}
    for r in d.itertuples():
        value=inputs(r);key=json.dumps(value,sort_keys=True)
        if key not in lookup:lookup[key]=len(nodes);nodes.append(value)
        sample_node[r.sample_id]=lookup[key]
    episodes=[]
    for b in (0,1,2):
        roles=t.ROLES[t.ROLES.family.eq(held)&t.ROLES.budget.eq(b)].set_index('sample_id').role
        a=d[d.sample_id.isin(roles[roles.eq('anchor')].index)]
        assert len(a)==b
        episodes.append(dict(budget=b,anchor_indices=[sample_node[r.sample_id] for r in a.itertuples()],
            anchor_labels=[float(r.target) for r in a.itertuples()]))
    return nodes,sample_node,episodes


def freeze():
    done=json.loads((t.OUT/'training_complete.json').read_text())
    audit=json.loads((t.OUT/'training_audit.json').read_text());assert audit['passed'] and audit['fit_records']==1596
    prereg=json.loads((t.OUT/'pre_evaluation_protocol.json').read_text())
    for name,h in prereg['hashes'].items():assert e.sha(ROOT/name)==h,name
    gates=json.loads((t.OUT/'source_gates.json').read_text());assert gates['passed']
    for name,h in gates['code_hashes'].items():assert e.sha(ROOT/name)==h,name
    affine=json.loads((t.OUT/'affine_gates.json').read_text());assert affine['passed']
    for name,h in affine['code_hashes'].items():assert e.sha(ROOT/name)==h,name
    code=[Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase25_source_runtime.py',
        ROOT/'docs/ML_SANDBOX_CONTEXT_STRONG_CONTROLS_ADDENDUM_20261018.md']+[ROOT/p for p in gates['code_hashes']]+[ROOT/p for p in affine['code_hashes']]
    files=[t.OUT/'training_protocol.json',t.OUT/'training_complete.json',t.OUT/'training_audit.json',t.OUT/'pre_evaluation_protocol.json',t.OUT/'source_gates.json',t.OUT/'affine_gates.json',t.OUT/'data/conditions33.csv',t.OUT/'data/episodes33.csv',NPZ]
    p=dict(created_utc=e.now(),variants=VARIANTS,lambdas=LAM,schemes=['DOI','cluster'],
        shared_selection='nested zeroanchor MAE within outer training units;shared coefficients fixed across0/1/2',
        source_selection='inner held-unit query MAE for same roles;then fixed outer lambda;mean anchors +lambda finite-design mean response²',
        source_design='unique explicit input tuples;all source conditions including label-free guard input;no query labels in worker packets',
        source_search='scalar:33 cubic nodes,atmost3localcandidates,80iterations,32768point ceiling;additional affine:augmented convexLS,accept only finite |delta|<=6,not clipped',
        additional_strong_controls='DoseLineAffine/RidgeAffine/PhysicalAffine have2sourceparameters,oneanchor lambda0 has explicitzero-slope rule;no newsharedfit',
        source_budget_unit='0/1/2 perDOI;two-DOI held relatedgroup has0/2/4labels,totalbudget sameacrossmethods;not0/1/2 perwholecluster',
        source_fit_shared_across_related_DOIs=False,cluster_holdout='exclude whole related group;separate allowed source anchors per DOI',
        metric='full coverage required;all-query plus same common20;equalDOI and conservative group macro both reported',
        runtime='fresh interpreters;read guard permits own task and SHA-verified frozen KROGER reference only;model weights and allowed anchors embedded;experimental/query files denied',
        retrospective=True,new_shared_training_in_prediction=0,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in files})
    dest=OUT/'protocol.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        for key in ('code_hashes','input_hashes','variants'):assert old[key]==e.clean(p[key]),key
    else:e.write(dest,p)


def prepare():
    plans=[];jobs={};shared_choices=[];inner_predictions=[];cache={};references={}
    for scheme in ('DOI','cluster'):
        units=t.units(scheme)
        for held in (*units,None):
            train=tuple(u for u in units if u!=held)
            for method in t.METHODS:
                reg,candidates,pred=select_shared(scheme,train,method)
                inner_predictions.extend(pred)
                shared_choices.append(dict(scheme=scheme,held=held,method=method,train_units=train,reg=reg,candidates=candidates))
                cache[scheme,held,method]=(reg,train)
    for scheme in ('DOI','cluster'):
        col=t.unit_column(scheme);units=t.units(scheme)
        for held in (*units,None):
            for variant,spec in VARIANTS.items():
                method=spec['method'];reg,train=cache[scheme,held,method]
                f,path=saved(scheme,train,method,reg)
                plan=dict(scheme=scheme,held_unit=held,variant=variant,method=method,kind=spec['kind'],reg=reg,
                    train_units=train,fit_key=f['key'],fit_path=str(path.relative_to(ROOT)),fit_sha256=e.sha(path),inner=[],outer=[])
                needed=[]
                for ih in train:
                    fi,pi=saved(scheme,[u for u in train if u!=ih],method,reg)
                    for family in sorted(t.TABLE.loc[t.TABLE[col].eq(ih),'family'].unique()):
                        needed.append(('inner',ih,family,fi,pi))
                if held is not None:
                    for family in sorted(t.TABLE.loc[t.TABLE[col].eq(held),'family'].unique()):needed.append(('outer',held,family,f,path))
                for role,unit,family,ff,pp in needed:
                    assert family not in ff['train_families']
                    key=t.digest(dict(protocol=e.sha(OUT/'protocol.json'),fit=ff['key'],family=family,kind=spec['kind']))
                    if key not in jobs:
                        nodes,mapping,episodes=design(family)
                        jobs[key]=dict(id=key,fit={k:ff[k] for k in MINIMAL if k in ff},kind=spec['kind'],nodes=nodes,episodes=episodes)
                        references[key]=dict(family=family,scheme=scheme,sample_node=mapping,fit_path=str(pp.relative_to(ROOT)),fit_sha256=e.sha(pp),
                            train_families=ff['train_families'],train_units=ff['train_units'])
                    plan[role].append(dict(unit=unit,family=family,task_id=key))
                plans.append(plan)
    e.write(OUT/'plans.json',plans);e.write(OUT/'shared_choices.json',shared_choices);e.write(OUT/'parent_reference.json',references)
    e.csvwrite(OUT/'inner_zero_predictions.csv',inner_predictions)
    tasks=list(jobs.values());e.write(OUT/'tasks.json',dict(tasks=tasks,lambdas=LAM,physical_reference_sha256=e.sha(NPZ)))
    return tasks


def run(tasks,workers):
    processes=[];clock=time.perf_counter()
    for i in range(workers):
        packet=OUT/f'tasks_{i}.json';dest=OUT/f'outputs_{i}.json'
        e.write(packet,dict(tasks=tasks[i::workers],lambdas=LAM,physical_reference_sha256=e.sha(NPZ)))
        handle=(OUT/f'worker_{i}.log').open('w')
        proc=subprocess.Popen([sys.executable,str(Path(__file__).parent/'phase25_source_runtime.py'),str(packet),str(dest)],
            stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT)
        processes.append((proc,handle,i))
    e.write(OUT/'execution_started.json',dict(created_utc=e.now(),tasks=len(tasks),workers=workers,pids=[p.pid for p,h,i in processes]))
    codes=[]
    for p,handle,i in processes:codes.append(p.wait());handle.close()
    e.write(OUT/'execution_workers_terminal.json',dict(created_utc=e.now(),exit_codes=codes,seconds=time.perf_counter()-clock))
    if any(codes):raise RuntimeError('Source worker failed; retain all partial outputs/logs')


def number(x):return np.inf if x is None else float(x)


def summarize(workers):
    records={}
    for i in range(workers):
        packet=json.loads((OUT/f'outputs_{i}.json').read_text());assert not packet['denied_reads']
        for r in packet['records']:assert r['id'] not in records;records[r['id']]=r
    refs=json.loads((OUT/'parent_reference.json').read_text());plans=json.loads((OUT/'plans.json').read_text())
    def candidate(task,b,lam):
        ep=next(ep for ep in records[task]['episodes'] if ep['budget']==b)
        return next(s for s in ep['solutions'] if s['lam']==lam)
    def rows_for(task,b,lam):
        c=candidate(task,b,lam);ref=refs[task];family=ref['family']
        roles=t.ROLES[t.ROLES.family.eq(family)&t.ROLES.budget.eq(b)].set_index('sample_id').role
        common=set(t.ROLES[t.ROLES.family.eq(family)&t.ROLES.budget.eq(2)&t.ROLES.role.eq('query')].sample_id)
        rows=[]
        for r in t.TABLE[t.TABLE.family.eq(family)].itertuples():
            p=c['records'][ref['sample_node'][r.sample_id]]
            rows.append(dict(p,sample_id=r.sample_id,family=family,related_cluster=r.related_cluster,role=roles[r.sample_id],
                common_query=r.sample_id in common,observed=float(r.target),
                observed_log_mu=float(np.log10(r.mu_cm2Vs)) if bool(r.mobility_eligible) else None,
                mobility_eligible=bool(r.mobility_eligible),raw_shift=c.get('raw_shift'),source_parameters=c.get('source_parameters'),task_id=task))
        return rows
    choices=[];predictions=[]
    for plan in plans:
        for b in (0,1,2):
            trials=[]
            for lam in ((None,) if b==0 else LAM):
                unit_rows={}
                if b:
                    for request in plan['inner']:
                        rows=[r for r in rows_for(request['task_id'],b,lam) if r['role']=='query']
                        unit_rows.setdefault(request['unit'],[]).extend(rows)
                    scores={u:score_rows(rows) for u,rows in unit_rows.items()}
                    score=float(np.mean(list(scores.values())))
                else:scores={};score=0.
                trials.append(dict(lam=lam,score=score,inner_unit_scores=scores))
            best=min(trials,key=lambda c:(round(c['score'],10),-np.inf if c['lam'] is None else -c['lam']))
            selection=dict(plan,budget=b,selected_source=best,source_candidates=trials)
            if plan['held_unit'] is not None:
                allrows=[]
                for request in plan['outer']:allrows.extend(rows_for(request['task_id'],b,best['lam']))
                selection['outer_all_query_MAE']=score_rows([r for r in allrows if r['role']=='query'])
                for r in allrows:predictions.append(dict(r,scheme=plan['scheme'],variant=plan['variant'],method=plan['method'],
                    held_unit=plan['held_unit'],budget=b,selected_reg=plan['reg'],selected_lambda=best['lam'],fit_key=plan['fit_key']))
            choices.append(selection)
    e.write(OUT/'selections.json',choices)
    e.csvwrite(OUT/'predictions.csv',predictions)
    df=pd.DataFrame(predictions);summary=[]
    for (scheme,variant,b),g in df.groupby(['scheme','variant','budget']):
        for scope in ('all_query','common20'):
            q=g[g.role.eq('query')]
            if scope=='common20':q=q[q.common_query.eq(True)]
            full=bool(q.numeric.all() and q.valid.all())
            entry=dict(scheme=scheme,variant=variant,budget=int(b),scope=scope,queries=len(q),numeric=int(q.numeric.sum()),
                valid=int(q.valid.sum()),nH_MAE=None,nH_cluster_MAE=None)
            if full:
                q=q.copy();q['ae']=np.abs(q.predicted-q.observed)
                family=q.groupby('family').ae.mean();entry['nH_MAE']=float(family.mean())
                cl=q[['family','related_cluster']].drop_duplicates().set_index('family').related_cluster
                entry['nH_cluster_MAE']=float(family.groupby(cl).mean().mean())
                entry['worst_family_MAE']=float(family.max())
            for col in ('predicted_log_mu','predicted_log_mu_CT'):
                mu=q[q.mobility_eligible.eq(True)]
                entry[col+'_pairs']=len(mu);entry[col+'_DOIs']=mu.family.nunique()
                entry[col+'_MAE']=float(np.abs(mu[col]-mu.observed_log_mu).groupby(mu.family).mean().mean()) if len(mu) and mu.numeric.all() and mu.valid.all() else None
            summary.append(entry)
    e.csvwrite(OUT/'summary.csv',summary)
    e.write(OUT/'execution_complete.json',dict(created_utc=e.now(),tasks=len(records),selections=len(choices),prediction_rows=len(df),
        numeric=int(df.numeric.sum()),valid=int(df.valid.sum()),summary_rows=len(summary),new_shared_fits_during_prediction=0,
        denied_runtime_reads=0,query_labels_in_worker_packets=False,goal_complete=False))
    print(pd.DataFrame(summary)[lambda d:d.scope.eq('common20')][['scheme','variant','budget','queries','valid','nH_MAE']].to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--summarize-only',action='store_true')
    a=ap.parse_args();freeze()
    if a.summarize_only:summarize(a.workers)
    else:
        tasks=prepare();run(tasks,a.workers);summarize(a.workers)


if __name__=='__main__':main()
