"""Same-anchor internal log-odds adaptation on frozen shared predictors."""
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.bounded_source_learning import learn_bounded_source
import phase18_refinement as final
import phase2_energy as energy

study=final.study;OUT=study.OUT/'source_adaptation';BANK={}
VARIANTS={'SourceLogit_'+m:m for m,d in study.STRUCTURES.items() if d.get('closure')=='inventory' or 'empirical' in d}


def bank():
    global BANK
    if not BANK:BANK=final.composite()
    return BANK


def getfit(families,method):return next(f for f in bank().values() if f['method']==method and set(f['train_families'])==set(families))


def scalar_checks():
    checks=[];base=np.array([17.,18.]);q=np.array([.8,-1.3]);y=np.array([17.7,17.5])
    result=learn_bounded_source(lambda z:base+q*z,y,study.core.LAM)
    for s,lam in zip(result['solutions'],study.core.LAM):
        wanted=0. if np.isinf(lam) else float(np.clip(q@(y-base)/(q@q+lam),-6.,6.))
        checks.append(dict(name='linear_quadratic_oracle',lam=s['lam'],error=abs(s['z']-wanted),passed=abs(s['z']-wanted)<3e-5))
    r=learn_bounded_source(lambda z:np.array([np.sin(z)]),[.25],[0.])['solutions'][0]
    checks.append(dict(name='nonconvex_multibasin_nearest_zero',error=abs(r['z']-np.arcsin(.25)),passed=abs(r['z']-np.arcsin(.25))<3e-5))
    r=learn_bounded_source(lambda z:np.array([.2]),[.4],[0.])['solutions'][0]
    checks.append(dict(name='flat_support_tie_zero',passed=r['z']==0.))
    assert all(c['passed'] for c in checks),checks
    return checks


def freeze():
    study.freeze()
    files=(Path(__file__),ROOT/'src/sandbox/bounded_source_learning.py',ROOT/'src/sandbox/logodds_inventory_model.py',
        ROOT/'docs/ML_SANDBOX_LOGODDS_PREDICTION_20261018.md')
    p=dict(created_utc=energy.now(),variants=VARIANTS,code_hashes={str(f.relative_to(ROOT)):energy.sha(f) for f in files},
        input_hashes={str(f.relative_to(ROOT)):energy.sha(f) for f in (study.OUT/'prediction_protocol.json',study.OUT/'numerical_gates.json',
            energy.BASE/'dataset_v1.csv',energy.BASE/'episodes_v1.csv')},lambda_grid=study.core.LAM,
        scalar_checks=scalar_checks(),source_rule='g+=ln10*z;z[-6,6];sumanchorlogerror^2+lambda*z^2;noquerylabels',
        search='49gridnodes inclzero/endpoints;refineallstrictsampledlocalbasins,xatol1e-5,maxiter100;tiesroundobjective1e-12/minabsz',
        selection='innerwholeDOI nHMAE,round1e-10/tiespreferlargerlambda;all6variants reported',
        expected_episodes=300,expected_inner_scores=1800,primary='same29/5/0-1-2/common18',
        timing='frozen before newprimaryouterselection;historicalcohortretrospective,notblind',
        source_units='log10odds,not samefunctionalshrinkageasoutputdex despite samelambda grid',
        exact_global_continuous_optimum_claimed=False,goal_complete=False)
    path=OUT/'source_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('variants','code_hashes','input_hashes','scalar_checks','lambda_grid'):assert old[k]==energy.clean(p[k]),k
    else:energy.write(path,p)


def predict_n(f,table,z):
    if 'empirical' in f['definition']:return study.activation(f['parameters'],table,f['definition']['empirical'],source_shift=z)[0]
    model=study.mapper(f['method'])
    return np.array([model.point(r,f['parameters'],source_kind='logodds',source_shift=z).values[0] for r in table.itertuples()])


def episode(f,held,budget):
    key=study.core.prior.old.digest(dict(protocol=energy.sha(OUT/'source_protocol.json'),fit=f['key'],fit_sha=f['composite_origin']['sha256'],held=held,budget=budget))
    path=OUT/'episodes'/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    clock=time.perf_counter();d=study.PRED[study.PRED.family.eq(held)&study.PRED.kind.eq('RT')]
    roles=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
    anchors=d[d.sample_id.map(roles).eq('anchor')];assert len(anchors)==budget and held not in f['train_families']
    fit=learn_bounded_source(lambda z:predict_n(f,anchors,z),anchors.target.to_numpy(float),study.core.LAM)
    query=d.sample_id.map(roles).eq('query').to_numpy();results=[];failures=[]
    for solution in fit['solutions']:
        z=solution['z']
        try:values=predict_n(f,d,z)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            values=np.full(len(d),np.nan);failures.append(dict(lam=solution['lam'],z=z,error=repr(exc)))
        records=[dict(sample_id=r.sample_id,obs_id=r.obs_id,family=held,budget=budget,role=roles[r.sample_id],observed_log_nH=float(r.target),
            predicted_log_nH=float(v),source_z=z) for r,v in zip(d.itertuples(),values)]
        results.append(dict(**solution,records=records,MAE=float(abs(values[query]-d.target.to_numpy(float)[query]).mean()) if np.isfinite(values).all() else None))
    result=dict(key=key,fit_key=f['key'],fit_origin=f['composite_origin'],method=f['method'],held=held,budget=budget,
        train_families=f['train_families'],anchor_obs_ids=anchors.obs_id.tolist(),anchor_observed=anchors.target.to_numpy(float),
        results=results,grid=fit['grid'],prediction_callbacks=fit['prediction_callbacks'],failed_probes=fit['failed_probes'],
        query_state_failures=failures,seconds=time.perf_counter()-clock,query_labels_used_in_source_fit=False,
        source_fit='usesanchoronly;queryonlyscoring',goal_complete=False)
    energy.write(path,result);print('logitsource',f['method'],held,budget,'callbacks',fit['prediction_callbacks'],'seconds',round(result['seconds'],2),flush=True)
    return energy.clean(result)


def choose(ep,lam):return next(r for r in ep['results'] if (r['lam'] is None and np.isinf(lam)) or r['lam']==lam)


def score(f,held,budget,lam):
    if budget:return np.inf if choose(episode(f,held,budget),lam)['MAE'] is None else choose(episode(f,held,budget),lam)['MAE']
    d=study.PRED[study.PRED.family.eq(held)&study.PRED.kind.eq('RT')];look={p['obs_id']:p for p in f['predictions']}
    return float(np.mean([abs(look[r.obs_id]['predicted']-r.target) for r in d.itertuples()]))


def full_predictions(f,held,z):
    d=study.TARGETS[study.TARGETS.family.eq(held)];n=predict_n(f,d,z);result=[];physical='empirical' not in f['definition']
    for row,logn in zip(d.itertuples(),n):
        if physical:
            p=study.mapper(f['method']).point(row,f['parameters'],source_kind='logodds',source_shift=z)
            tv=p.transport_values;charges=p.values[1:];dg={k:p.diagnostics[k] for k in ('same_physical_state_for_output','foreign_atom_relative_error',
                'foreign_atom_derivative_error','free_n_cm3','hall_factor','complex_reallocation_fraction','thermal_cap_K')}
        else:
            logmu=float(np.log10(study.transport.mu_caughey_thomas(10**logn,float(row.Tobs_K))))
            tv=[logmu,float(np.log10(study.transport.E)+logn+logmu)];charges=None;dg=dict(same_physical_state_for_output=False)
        result.append(dict(obs_id=row.obs_id,sample_id=row.sample_id,kind=row.kind,endpoint=row.endpoint,T_K=float(row.Tobs_K),
            predicted_log_nH=float(logn),transport_values=tv,charge_outputs=charges,diagnostics=dg))
    return result


def select(variant,method):
    rows=[];temps=[];choices=[];properties=[];count=0
    for held in study.FAMILIES:
        train=[s for s in study.FAMILIES if s!=held]
        for budget in (0,1,2):
            candidates=[]
            for lam in ([np.inf] if budget==0 else study.core.LAM):
                fits=[getfit([s for s in train if s!=inner],method) for inner in train]
                scores=[score(f,inner,budget,lam) for f,inner in zip(fits,train)];count+=len(scores)
                candidates.append(dict(lam=lam,score=float(np.mean(scores)),inner_keys=[f['key'] for f in fits]))
            best=min(candidates,key=lambda c:(round(c['score'],10),-c['lam']));f=getfit(train,method)
            ep=None if budget==0 else episode(f,held,budget);z=0. if ep is None else choose(ep,best['lam'])['z']
            pred=full_predictions(f,held,z);lookup={p['obs_id']:p for p in pred}
            roles=study.core.ROLES[study.core.ROLES.family.eq(held)&study.core.ROLES.budget.eq(budget)].set_index('sample_id').role
            for r in study.PRED[study.PRED.family.eq(held)].itertuples():
                p=lookup[r.obs_id];meta=dict(method=variant,base_method=method,fit_key=f['key'],family=held,budget=budget,
                    sample_id=r.sample_id,role=roles[r.sample_id],source_z=z)
                if r.kind=='RT':rows.append(dict(**meta,observed_log_nH=float(r.target),predicted_log_nH=p['predicted_log_nH']))
                else:temps.append(dict(**meta,T_K=float(r.Tobs_K),observed=float(r.target),predicted=p['predicted_log_nH'],
                    common40_125=bool(40<=r.Tobs_K<=125)))
            properties += [dict(method=variant,base_method=method,fit_key=f['key'],family=held,budget=budget,source_z=z,
                role=roles[p['sample_id']],**p) for p in pred]
            choices.append(dict(method=variant,base_method=method,held=held,budget=budget,fit_key=f['key'],candidates=candidates,
                selected=best,source_z=z,episode_key=None if ep is None else ep['key'],outer_MAE=score(f,held,budget,best['lam'])))
    folder=OUT/'methods'/variant;energy.csvwrite(folder/'primary_predictions.csv',rows);energy.csvwrite(folder/'temperature_predictions.csv',temps)
    energy.write(folder/'selections.json',choices);energy.write(folder/'property_predictions.json',properties)
    print('selectedlogitsource',variant,flush=True);return count


def worker(job):
    families,method,held,budget=job
    return episode(getfit(families,method),held,budget)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();freeze()
    if args.freeze_only:return
    assert len(bank())==120
    energy.write(OUT/'fit_inventory.json',[dict(key=f['key'],**f['composite_origin']) for f in bank().values()])
    jobs=[(f,m,h,b) for n in (3,4) for f in combinations(study.FAMILIES,n) for h in study.FAMILIES if h not in f
        for m in VARIANTS.values() for b in (1,2)];assert len(jobs)==300
    energy.write(OUT/'started.json',dict(created_utc=energy.now(),episodes=300,workers=args.workers,goal_complete=False))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    count=sum(select(v,m) for v,m in VARIANTS.items());assert count==1800
    energy.write(OUT/'complete.json',dict(created_utc=energy.now(),variants=6,episodes=300,inner_scores=count,goal_complete=False))


if __name__=='__main__':main()
